# -*- coding: utf-8 -*-
"""
run_interpretability.py —— Interpretability analysis script aligned with the main program

How to use:
1) Modify only the 4 fields in RunParams below:
   - EXP_DIR: Your final training directory (containing best_model_final.pth; final_test_predictions.csv will be read first if present)
   - NUM_MUTATION_SAMPLES: Number of in-silico mutation sampling sequences
   - MOTIFS: List of motifs to detect/mutate (comma-separated or list)
   - REPLACEMENTS: Corresponding replacement sequence candidates (comma-separated or list; length must match motif)

2) Run: python run_interpretability.py
   Outputs to: Project root directory / result / interpretability_result / timestamp / ...
"""

import os
import re
import json
import time
import math
import platform
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ======================
# 0. Modify only here
# ======================
class RunParams:
    # Your final training directory (containing best_model_final.pth; if it contains final_test_predictions.csv, it will be used directly)
    EXP_DIR: str = "/ABSOLUTE/OR/RELATIVE/PATH/TO/your_final_run_dir"

    # === New: Change in-silico architecture ===
    # "full": Full enumeration (evaluating every sequence containing the motif, every occurrence position, and every alternative sequence individually)
    # "per-motif": Independently sample sequences per motif (each motif must yield at least NUM_MUTATION_SAMPLES sequences containing that motif).
    MUTATION_MODE: str = "full"   # Can be set to "per-motif"

    # Quota-based in silico mutation sampling
    # - When MUTATION_MODE="full", this parameter is ignored (full enumeration).
    # - When MUTATION_MODE="per-motif", this parameter serves as the "minimum sample quota per motif".
    NUM_MUTATION_SAMPLES: int = 2125

    # Motifs requiring statistical analysis/mutation (RNA alphabet, examples from the list you provided)
    MOTIFS =       "GAGGU,GCACU,CACCA,ACCAC,CCUAA,UCACC,CGAAU,AGAAG,UCUUG,GUGUA,CACUU,AAAGU,UUGUAU,AUAAUU,AUGCA,UUUAUG,GGUGU,AAAUGA,AUUUA,UUAUUU,CCCCC,GCGCGC"

    # Alternative sequences (same length as the MOTIFS above; can be a 1:many "candidate set," where each motif is tried individually)
    REPLACEMENTS = "GAAGU,GCGCU,CAACA,ACAAC,CCGAA,UCGCC,CGGAU,AGGAG,UCGUG,GUAUA,CAAUU,AAGGU,UUAUAU,AUGAUU,AUACA,UUGAUG,GGGGU,AAGUGA,AUAUA,UUGUGU,CUAUA,GUAUAC"


    # (Optional) Limit the number of enumeration positions per sequence and per motif to prevent an explosion of extremely long sequence combinations; None indicates no restriction.
    MAX_POS_PER_SEQ_PER_MOTIF: int | None = None


# ======================
# 1. Configuration consistent with the main program
# ======================
class Config:
    DATA_PATH = 'data/mRNA_half_life_dataset_RNA.csv'
    PRETRAINED_MODEL_NAME = "model/rna-fm"

    MODEL_MAX_LENGTH = 448
    EMBEDDING_DIM = 640
    DROPOUT = 0.1

    BATCH_SIZE = 16
    RANDOM_SEED = 42

# ======================
# 2. Utility Functions
# ======================
def set_seed(seed: int = 42):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_output_dir():
    # Output to: The directory above where the script is located / result / interpretability_result / <timestamp>
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "result", "interpretability_result")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tensorboard-log", "interpretability")
    os.makedirs(tb_dir, exist_ok=True)
    return out_dir, tb_dir, project_root


def _parse_items_list(x):
    """Convert 'A,B,C' or list to uppercase list[str], removing empty items."""
    if isinstance(x, (list, tuple)):
        return [str(s).strip().upper() for s in x if str(s).strip()]
    return [s.strip().upper() for s in str(x).split(",") if s.strip()]

def build_paired_mapping(params) -> dict[str, list[str]]:
    """
   Generate one-to-one mapping: { motif -> [replacement, ...] }
- MOTIFS and REPLACEMENTS must be of equal length (positionally aligned);
- Invalid pairs with “unequal lengths” are automatically skipped;
- If the same motif occurs multiple times, it can correspond to multiple replacements (aggregated to a list).
    """
    motifs = _parse_items_list(params.MOTIFS)
    repls  = _parse_items_list(params.REPLACEMENTS)
    if len(motifs) != len(repls):
        raise ValueError(f"MOTIFS does not match the number of REPLACEMENTS:{len(motifs)} vs {len(repls)}")
    mapping: dict[str, list[str]] = {}
    for m, r in zip(motifs, repls):
        if len(m) != len(r):
            print(f"[warn] Skip pairs of varying lengths: {m} vs {r}")
            continue
        mapping.setdefault(m, [])
        if r not in mapping[m]:
            mapping[m].append(r)
    if not mapping:
        raise ValueError("No valid motif→replacement pairs (please check for length consistency)")
    return mapping

# ======================
# 3. dataset with collate (no chunk, returns sequence)# ======================
class MRNADataset(Dataset):
    def __init__(self, sequences, targets=None):
        self.sequences = sequences
        self.targets = targets  # Can be None during assessment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {"sequence": str(self.sequences[idx])}
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

def collate_fn_no_chunk(batch, tokenizer, config: Config):
    sequences = [item['sequence'] for item in batch]
    targets = None
    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])

    tokenized = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=config.MODEL_MAX_LENGTH,
        return_tensors="pt"
    )

    out = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "sequence": sequences
    }
    if targets is not None:
        out["targets"] = targets
    return out

# ======================
# 4. Model (RNA-FM + TokenTransformerHead)
# ======================
class _SinPosEnc(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)

class _ResidualMLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.act(self.fc1(self.pre(x))))
        h = self.drop2(self.fc2(h))
        x = x + h
        return self.out(x).squeeze(-1)

class TokenTransformerHead(nn.Module):
    def __init__(self, dim: int, nhead: int = 10, num_layers: int = 3,
                 ff_mult: int = 4, dropout: float = 0.1, mlp_hidden: int = 512):
        super().__init__()
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ff_mult * dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = _SinPosEnc(dim)
        self.regressor = _ResidualMLPHead(dim, mlp_hidden, dropout)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = token_embeds.shape
        x = torch.cat([self.cls.expand(B, 1, D), token_embeds], dim=1)  # [B, L+1, D]
        if attention_mask is not None:
            kpm = torch.ones(B, L + 1, dtype=torch.bool, device=token_embeds.device)
            kpm[:, 0] = False
            kpm[:, 1:] = (attention_mask == 0)
        else:
            kpm = None
        x = self.posenc(x)
        x = self.encoder(x, src_key_padding_mask=kpm)
        cls = x[:, 0, :]
        return self.regressor(cls)

class ChunkingMRNATransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        from multimolecule import RnaFmModel
        self.backbone = RnaFmModel.from_pretrained(config.PRETRAINED_MODEL_NAME, trust_remote_code=True)
        self.token_head = TokenTransformerHead(
            dim=config.EMBEDDING_DIM,
            nhead=10, num_layers=3, ff_mult=4,
            dropout=config.DROPOUT, mlp_hidden=512
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        return self.token_head(token_embeddings, attention_mask)

# ======================
# 5. Inference and Evaluation (log space → linear space)
# ======================
@torch.no_grad()
def predict_dataset(model, loader, device, loss_fn=None, has_target=True):
    model.eval()
    total_loss = 0.0
    y_true_log, y_pred_log, seqs = [], [], []
    for batch in tqdm(loader, desc="Prediction", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        preds = model(input_ids, attention_mask)  # [B]
        if has_target and loss_fn is not None:
            targets = batch["targets"].to(device)
            loss = loss_fn(preds, targets)
            total_loss += loss.item()
            y_true_log.extend(targets.cpu().numpy())

        y_pred_log.extend(preds.cpu().numpy())
        seqs.extend(batch["sequence"])

    y_pred = np.expm1(y_pred_log)
    if has_target:
        y_true = np.expm1(y_true_log)
        avg_loss = total_loss / len(loader) if loss_fn is not None else None
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        pearson, _ = pearsonr(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)
        return {
            "loss": avg_loss, "r2": r2, "mse": mse,
            "pearson": pearson, "spearman": spearman,
            "true": y_true, "pred": y_pred, "sequence": seqs
        }
    else:
        return {"pred": y_pred, "sequence": seqs}

def _read_state_dict_flex(ckpt_path, device):
    """Extract genuine state_dict objects from various common formats wherever possible, and strip off common prefixes."""
    obj = torch.load(ckpt_path, map_location=device)

    # 1) Extracting nested structures from common enclosures
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "ema_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}")

    state = obj

    # 2) Common prefixes for consecutive stripping
    def strip_prefix(d, prefix):
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

    for pref in ["_orig_mod.", "module.", "model.", "net."]:
        state = strip_prefix(state, pref)

    return state


def load_trained_model(exp_dir, config: Config, device):
    """More robust weight loading: Load weights where possible, retaining pre-trained weights where not."""
    weight_path = os.path.join(exp_dir, "best_model_final.pth")
    if not os.path.exists(weight_path):
        cand = [f for f in os.listdir(exp_dir) if f.endswith(".pth")]
        if not cand:
            raise FileNotFoundError(f"The weight file {weight_path} was not found, and there is no .pth file in the directory.")
        weight_path = os.path.join(exp_dir, cand[0])

    # Build the model (first load the pre-trained RNA-FM)
    model = ChunkingMRNATransformer(config).to(device)

    # Extract the state_dict from the checkpoint and perform key name sanitisation.
    raw_state = _read_state_dict_flex(weight_path, device)

    # === New: Map ckpt's bert.* / pure HF keys to wrapper's backbone.*  ===
    def remap_backbone_prefix(state: dict) -> dict:
        """
        Map ckpt's bert.* / pure HF keys (embeddings./encoder./pooler.) to wrapper's backbone.*
        Leave token_head.* and other keys untouched;
        and clean up any potential backbone.backbone.* duplicate prefixes.
        """
        keys = list(state.keys())
        has_backbone = any(k.startswith("backbone.") for k in keys)
        has_bert = any(k.startswith("bert.") for k in keys)

        # Scenario 1: ckpt utilises bert.* — mapped to backbone.*
        if has_bert and not has_backbone:
            state = {("backbone." + k[5:] if k.startswith("bert.") else k): v for k, v in state.items()}
            keys = list(state.keys())
            has_backbone = any(k.startswith("backbone.") for k in keys)

        # Scenario 2: ckpt is a pure HF key (no prefix): embeddings./encoder./pooler.
        has_hf_root = any(k.startswith(("embeddings.", "encoder.", "pooler.")) for k in keys)
        if has_hf_root and not has_backbone:
            state = {("backbone." + k if k.startswith(("embeddings.", "encoder.", "pooler.")) else k): v
                     for k, v in state.items()}

        # Remove potential duplicate prefixes
        state = {(k.replace("backbone.backbone.", "backbone.") if k.startswith("backbone.backbone.") else k): v
                 for k, v in state.items()}
        return state

    raw_state = remap_backbone_prefix(raw_state)

    model_state = model.state_dict()

    # Retain only entries where the key exists and the shape is consistent.
    filtered = {}
    mismatched_shapes = {}
    for k, v in raw_state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        elif k in model_state:
            mismatched_shapes[k] = {"ckpt": tuple(v.shape), "model": tuple(model_state[k].shape)}

    # Loading (Relaxed Mode)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # Print/record loading reports
    report = {
        "ckpt_path": weight_path,
        "loaded_keys": len(filtered),
        "missing_in_ckpt_but_in_model": list(missing),
        "unexpected_in_ckpt": list(unexpected),
        "shape_mismatch": mismatched_shapes
    }
    try:
        out_dir, _, _ = create_output_dir()
        with open(os.path.join(out_dir, "ckpt_load_report.json"), "w") as f:
            import json; json.dump(report, f, indent=2)
    except Exception:
        pass


    print(f"[ckpt] loaded {report['loaded_keys']} tensors from: {weight_path}")
    if report["missing_in_ckpt_but_in_model"]:
        print(f"[ckpt] missing keys (kept pretrained for these): {len(report['missing_in_ckpt_but_in_model'])}")
    if report["shape_mismatch"]:
        print(f"[ckpt] shape-mismatch keys (skipped): {len(report['shape_mismatch'])}")
    if report["unexpected_in_ckpt"]:
        print(f"[ckpt] unexpected keys in ckpt (ignored): {len(report['unexpected_in_ckpt'])}")

    model.eval()
    return model


# ======================
# 6. motif Statistics and in-silico mutation
# ======================


def count_motif(seq: str, motif: str) -> int:
    return len(re.findall(f"(?={motif})", seq))  # Supports duplicate counting

def tail_A_fraction(seq: str, tail_len: int = 50) -> float:
    tail = seq[-tail_len:] if len(seq) >= tail_len else seq
    return (tail.count("A") / len(tail)) if len(tail) > 0 else 0.0

def make_motif_table(df_residuals: pd.DataFrame) -> pd.DataFrame:
    motifs = ["AUUUA", "AATAAA", "ATTAAA"]  # Note: If your sequence is DNA (containing T), please change ARE to ATTTA.
    records = []
    df_residuals = df_residuals.copy()
    df_residuals["tailA_frac_50"] = df_residuals["sequence"].apply(tail_A_fraction)
    for m in motifs:
        df_residuals[f"count_{m}"] = df_residuals["sequence"].apply(lambda s: count_motif(s, m))

    for col in [f"count_{m}" for m in motifs] + ["tailA_frac_50"]:
        v = df_residuals[col].values
        r = df_residuals["residual"].values
        try:
            p_corr, p_p = pearsonr(v, r)
            s_corr, s_p = spearmanr(v, r)
        except Exception:
            p_corr, p_p, s_corr, s_p = np.nan, np.nan, np.nan, np.nan
        records.append({
            "feature": col,
            "pearson": float(p_corr),
            "pearson_p": float(p_p) if p_p == p_p else np.nan,
            "spearman": float(s_corr),
            "spearman_p": float(s_p) if s_p == s_p else np.nan
        })
    return pd.DataFrame(records)

def mutate_once(seq: str, pos: int, old: str, new: str) -> str:
    if pos < 0 or pos + len(old) > len(seq):
        return None
    if seq[pos:pos + len(old)] != old:
        return None
    return seq[:pos] + new + seq[pos + len(old):]

@torch.no_grad()
def predict_sequence_list(model, tokenizer, config, device, seq_list):
    ds = MRNADataset(seq_list, targets=None)
    collate = partial(collate_fn_no_chunk, tokenizer=tokenizer, config=config)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate)
    out = predict_dataset(model, loader, device, loss_fn=None, has_target=False)
    return out["pred"]

def run_mutation_suite(model, tokenizer, config, device,
                       df_base: pd.DataFrame,
                       motifs=("AATAAA", "ATTAAA"),
                       replacements=("AGTAAA",),
                       sample_k=32) -> pd.DataFrame:
    """
    In-silico mutation:
      - When RunParams.MUTATION_MODE == "full": exhaustive enumeration (all samples containing motifs × all occurrence positions × all alternative sequences)
      - When RunParams.MUTATION_MODE == "per-motif": Independently sample at least sample_k (NUM_MUTATION_SAMPLES passed as outer parameter) samples containing each motif

    Return columns: sample_idx, motif, new, pos, base_pred, mut_pred, delta, sequence
    """
    rng = np.random.default_rng(2024)

    # One-to-one mapping (aligned by RunParams.MOTIFS with REPLACEMENTS)
    params = RunParams()
    motif_to_repls = build_paired_mapping(params)
    motifs = list(motif_to_repls.keys())

    # Performed solely on sequences containing any target motif (to avoid invalid samples)
    union_pat = re.compile("(?:%s)" % "|".join(map(re.escape, motifs)))
    has_any = df_base["sequence"].str.contains(union_pat)

    eligible_all = np.where(has_any.values)[0]
    if len(eligible_all) == 0:
        return pd.DataFrame(columns=["sample_idx","motif","new","pos","base_pred","mut_pred","delta","sequence"])

    # Compute/cache base_pred (only for sequences that will be used)
    def batch_predict_base(idx_list: list[int]) -> dict[int, float]:
        seq_list = [df_base.iloc[i]["sequence"] for i in idx_list]
        preds = predict_sequence_list(model, tokenizer, config, device, seq_list)
        return {i: float(p) for i, p in zip(idx_list, preds)}

    records = []
    params = RunParams()
    max_pos_per_seq = params.MAX_POS_PER_SEQ_PER_MOTIF

    if getattr(params, "MUTATION_MODE", "full") == "full":
        # ========= Full Enumeration =========
        # For all sequences containing any motif, first obtain base_pred in one go
        base_pred_map = batch_predict_base(list(eligible_all))

        # Sequence processing by motif: For each occurrence of a motif within the sequence, construct all possible substitutions and perform batch prediction.
        for idx in tqdm(eligible_all, desc="In-silico mutation (full)", leave=False):
            seq = df_base.iloc[idx]["sequence"]
            base_pred = base_pred_map[idx]

            mut_tasks = []   # (motif, new, pos, mut_seq)
            for motif in motifs:
                positions = [m.start() for m in re.finditer(f"(?={motif})", seq)]
                if max_pos_per_seq is not None and len(positions) > max_pos_per_seq:
                    # To limit the number of positions, randomly select a specified number.
                    positions = list(rng.choice(positions, size=max_pos_per_seq, replace=False))
                for pos in positions:
                    for newmotif in motif_to_repls.get(motif, []):
                        mut_seq = mutate_once(seq, pos, motif, newmotif)
                        if mut_seq is not None:
                            mut_tasks.append((motif, newmotif, pos, mut_seq))

            if not mut_tasks:
                continue

            # A single-step prediction of all mutations in this sequence
            mut_preds = predict_sequence_list(model, tokenizer, config, device, [t[3] for t in mut_tasks])
            for (motif, newmotif, pos, _), mp in zip(mut_tasks, mut_preds):
                records.append({
                    "sample_idx": int(idx),
                    "motif": motif,
                    "new": newmotif,
                    "pos": int(pos),
                    "base_pred": float(base_pred),
                    "mut_pred": float(mp),
                    "delta": float(mp - base_pred),
                    "sequence": seq
                })

    else:
        # ========= per-motif sampling =========
        per_quota = int(sample_k)  # The outer layer receives RunParams.NUM_MUTATION_SAMPLES
        # Identify the sample sets containing each motif and perform quota sampling.
        motif_to_indices: dict[str, np.ndarray] = {}
        for motif in motifs:
            m_pat = f"(?={re.escape(motif)})"
            has_m = df_base["sequence"].str.contains(m_pat, regex=True)
            idxs = np.where(has_m.values)[0]
            if len(idxs) == 0:
                motif_to_indices[motif] = np.array([], dtype=int)
                continue
            take = min(per_quota, len(idxs))
            motif_to_indices[motif] = rng.choice(idxs, size=take, replace=False)

        # The unique sample index set requiring base_pred
        uniq_idxs = sorted(set(int(i) for arr in motif_to_indices.values() for i in arr))
        if len(uniq_idxs) == 0:
            return pd.DataFrame(columns=["sample_idx","motif","new","pos","base_pred","mut_pred","delta","sequence"])
        base_pred_map = batch_predict_base(uniq_idxs)

        # Enumerate alternatives per motif, per sample, per position; perform "batch mutation prediction" for each sample.
        for motif, idx_arr in motif_to_indices.items():
            if len(idx_arr) == 0:
                continue
            for idx in tqdm(idx_arr, desc=f"In-silico mutation (per-motif:{motif})", leave=False):
                seq = df_base.iloc[int(idx)]["sequence"]
                base_pred = base_pred_map[int(idx)]
                positions = [m.start() for m in re.finditer(f"(?={re.escape(motif)})", seq)]
                if max_pos_per_seq is not None and len(positions) > max_pos_per_seq:
                    positions = list(rng.choice(positions, size=max_pos_per_seq, replace=False))

                mut_tasks = []
                for pos in positions:
                    for newmotif in motif_to_repls.get(motif, []):
                        mut_seq = mutate_once(seq, pos, motif, newmotif)
                        if mut_seq is not None:
                            mut_tasks.append((motif, newmotif, pos, mut_seq))
                if not mut_tasks:
                    continue

                mut_preds = predict_sequence_list(model, tokenizer, config, device, [t[3] for t in mut_tasks])
                for (motif, newmotif, pos, _), mp in zip(mut_tasks, mut_preds):
                    records.append({
                        "sample_idx": int(idx),
                        "motif": motif,
                        "new": newmotif,
                        "pos": int(pos),
                        "base_pred": float(base_pred),
                        "mut_pred": float(mp),
                        "delta": float(mp - base_pred),
                        "sequence": seq
                    })

    return pd.DataFrame(records)


# ======================
# 7. Main process
# ======================
def main():
    # Read built-in parameters
    params = RunParams()
    exp_dir = os.path.abspath(params.EXP_DIR)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"EXP_DIR does not exist or is not a folder:{exp_dir}")

    # The remainder shall remain consistent with the main programme.
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)
    device = get_device()

    out_dir, tb_dir, project_root = create_output_dir()
    writer = SummaryWriter(log_dir=tb_dir)

    # Environmental Records
    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "random_seed": cfg.RANDOM_SEED,
        "exp_dir": exp_dir,
        "num_mutation_samples": int(params.NUM_MUTATION_SAMPLES),
        "motifs": (params.MOTIFS if isinstance(params.MOTIFS, list) else [m.strip() for m in str(params.MOTIFS).split(",") if m.strip()]),
        "replacements": (params.REPLACEMENTS if isinstance(params.REPLACEMENTS, list) else [m.strip() for m in str(params.REPLACEMENTS).split(",") if m.strip()])
    }
    with open(os.path.join(out_dir, "interpret_env.json"), "w") as f:
        json.dump(env, f, indent=4)

    # Loading tokenizer
    from multimolecule import RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(cfg.PRETRAINED_MODEL_NAME, trust_remote_code=True)
    collate = partial(collate_fn_no_chunk, tokenizer=tokenizer, config=cfg)

    # Load trained model
    model = load_trained_model(exp_dir, cfg, device)

    # Retrieve predictions from the main experiment's test set; if unavailable, fall back to splitting and predicting according to the main programme.
    test_pred_path = os.path.join(exp_dir, "final_test_predictions.csv")
    if os.path.exists(test_pred_path):
        df_base = pd.read_csv(test_pred_path)
        if not {"sequence", "true", "pred"}.issubset(df_base.columns):
            df_base = None
    else:
        df_base = None

    if df_base is None:
        data_path = os.path.join(project_root, cfg.DATA_PATH)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found：{data_path}")
        df_all = pd.read_csv(data_path).dropna(subset=["sequence", "Isoform Half-Life"]).copy()
        df_all["target"] = np.log1p(df_all["Isoform Half-Life"])
        df_train_val, df_test = train_test_split(df_all, test_size=0.2, random_state=cfg.RANDOM_SEED)

        test_ds = MRNADataset(df_test["sequence"].values, df_test["target"].values)
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate)
        loss_fn = nn.MSELoss()
        metrics = predict_dataset(model, test_loader, device, loss_fn=loss_fn, has_target=True)
        df_base = pd.DataFrame({
            "sequence": metrics["sequence"],
            "true": metrics["true"],
            "pred": metrics["pred"]
        })

    # Residual table
    df_base["residual"] = df_base["true"] - df_base["pred"]
    df_base.to_csv(os.path.join(out_dir, "residuals.csv"), index=False)

    # Write test metrics (using existing pred/true)
    try:
        r2 = r2_score(df_base["true"], df_base["pred"])
        mse = mean_squared_error(df_base["true"], df_base["pred"])
        p, _ = pearsonr(df_base["true"], df_base["pred"])
        s, _ = spearmanr(df_base["true"], df_base["pred"])
        writer.add_scalar("Test/R2", r2, 0)
        writer.add_scalar("Test/MSE", mse, 0)
        writer.add_scalar("Test/Pearson", p, 0)
        writer.add_scalar("Test/Spearman", s, 0)
    except Exception:
        pass

    # Motif and residual correlation
    df_corr = make_motif_table(df_base.copy())
    df_corr.to_csv(os.path.join(out_dir, "motif_corr.csv"), index=False)
    for _, row in df_corr.iterrows():
        writer.add_scalar(f"MotifCorr/{row['feature']}_pearson", row["pearson"], 0)
        writer.add_scalar(f"MotifCorr/{row['feature']}_spearman", row["spearman"], 0)

    # in-silico mutation
    motifs = env["motifs"]
    replacements = env["replacements"]
    df_mut = run_mutation_suite(
        model, tokenizer, cfg, device,
        df_base=df_base,
        motifs=tuple(motifs),
        replacements=tuple(replacements),
        sample_k=max(1, int(params.NUM_MUTATION_SAMPLES))
    )
    df_mut.to_csv(os.path.join(out_dir, "mutation_results.csv"), index=False)

    # Summary
    summary = {
        "n_test": int(len(df_base)),
        "test_R2": float(r2_score(df_base["true"], df_base["pred"])),
        "test_Pearson": float(pearsonr(df_base["true"], df_base["pred"])[0]),
        "test_Spearman": float(spearmanr(df_base["true"], df_base["pred"])[0]),
        "num_motif_features": int(len(df_corr)),
        "num_mutation_rows": int(len(df_mut)),
        "motifs": motifs,
        "replacements": replacements,
        "out_dir": out_dir
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Histogram written to TB
    try:
        writer.add_histogram("Residuals/hist", (df_base["residual"].values.astype(np.float32)), 0)
        if len(df_mut) > 0:
            writer.add_histogram("Mutation/delta_hist", (df_mut["delta"].values.astype(np.float32)), 0)
    except Exception:
        pass

    writer.close()
    print("\n  The explanatory analysis has been completed, with results saved in:", out_dir)
    print("  - residuals.csv（Residual table）")
    print("  - motif_corr.csv（motif related to residuals）")
    print("  - mutation_results.csv（in-silico mutation）")
    print("  - summary.json / interpret_env.json")
    print("  - TensorBoard：", os.path.join(out_dir, "tensorboard-log", "interpretability"))

if __name__ == "__main__":
    main()
