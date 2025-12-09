# -*- coding: utf-8 -*-
"""
run_interpretability.py —— 与主程序对齐的可解释性分析脚本（内置参数版，无需命令行）

如何使用：
1) 仅修改下方 RunParams 中的 4 个字段：
   - EXP_DIR：你的最终训练目录（含 best_model_final.pth；如果有 final_test_predictions.csv 会优先读）
   - NUM_MUTATION_SAMPLES：in-silico 突变抽样序列数
   - MOTIFS：需要检测/做突变的 motif 列表（逗号分隔或 list）
   - REPLACEMENTS：对应的替代序列候选（逗号分隔或 list；长度要与 motif 相同）

2) 运行：python run_interpretability.py
   结果输出到：项目根目录 / result / interpretability_result / 时间戳 / ...
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
# 0. 仅修改这里（内置参数）
# ======================
class RunParams:
    # 你的最终训练目录（含 best_model_final.pth；若含 final_test_predictions.csv 会直接使用）
    EXP_DIR: str = "/ABSOLUTE/OR/RELATIVE/PATH/TO/your_final_run_dir"

    # === 新增：变更 in-silico 架构 ===
    # "full"：全量枚举（对含 motif 的所有序列、所有出现位置、所有替代序列逐一评估）
    # "per-motif"：按 motif 独立抽样（每个 motif 至少抽 NUM_MUTATION_SAMPLES 条“含该 motif”的序列）
    MUTATION_MODE: str = "full"   # 可设为 "per-motif"

    # in-silico 突变抽样的“配额基准”
    # - 当 MUTATION_MODE="full" 时，此参数会被忽略（全量枚举）
    # - 当 MUTATION_MODE="per-motif" 时，此参数作为“每个 motif 的最少样本配额”
    NUM_MUTATION_SAMPLES: int = 2125

    # 需要统计/突变的 motifs（RNA 字母表，示例为你给的清单）
    MOTIFS =       "GAGGU,GCACU,CACCA,ACCAC,CCUAA,UCACC,CGAAU,AGAAG,UCUUG,GUGUA,CACUU,AAAGU,UUGUAU,AUAAUU,AUGCA,UUUAUG,GGUGU,AAAUGA,AUUUA,UUAUUU,CCCCC,GCGCGC"

    # 替代序列（与上面 MOTIFS 等长；可以是 1:多 的“候选集合”，每个 motif 都会逐一尝试）
    REPLACEMENTS = "GAAGU,GCGCU,CAACA,ACAAC,CCGAA,UCGCC,CGGAU,AGGAG,UCGUG,GUAUA,CAAUU,AAGGU,UUAUAU,AUGAUU,AUACA,UUGAUG,GGGGU,AAGUGA,AUAUA,UUGUGU,CUAUA,GUAUAC"


    # （可选）限制“每条序列、每个 motif”的枚举位置数量，避免极端长序列组合爆炸；None 表示不限制
    MAX_POS_PER_SEQ_PER_MOTIF: int | None = None


# ======================
# 1. 与主程序一致的配置
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
# 2. 实用函数
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
    # 输出到：脚本所在上一层 / result / interpretability_result / <timestamp>
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
    """把 'A,B,C' 或 list 转成大写 list[str]，去掉空项。"""
    if isinstance(x, (list, tuple)):
        return [str(s).strip().upper() for s in x if str(s).strip()]
    return [s.strip().upper() for s in str(x).split(",") if s.strip()]

def build_paired_mapping(params) -> dict[str, list[str]]:
    """
    生成一一对应映射：{ motif -> [replacement, ...] }
    - MOTIFS 与 REPLACEMENTS 必须等长（位置对齐）；
    - 自动跳过“长度不等”的无效配对；
    - 同一 motif 如出现多次，可对应多个替代（聚合到 list）。
    """
    motifs = _parse_items_list(params.MOTIFS)
    repls  = _parse_items_list(params.REPLACEMENTS)
    if len(motifs) != len(repls):
        raise ValueError(f"MOTIFS 与 REPLACEMENTS 数量不一致：{len(motifs)} vs {len(repls)}")
    mapping: dict[str, list[str]] = {}
    for m, r in zip(motifs, repls):
        if len(m) != len(r):
            print(f"[warn] 跳过长度不等的配对: {m} vs {r}")
            continue
        mapping.setdefault(m, [])
        if r not in mapping[m]:
            mapping[m].append(r)
    if not mapping:
        raise ValueError("没有有效的 motif→replacement 配对（请检查长度是否一致）")
    return mapping

# ======================
# 3. 数据集与 collate（无 chunk，返回 sequence）
# ======================
class MRNADataset(Dataset):
    def __init__(self, sequences, targets=None):
        self.sequences = sequences
        self.targets = targets  # 评估时可为 None

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
# 4. 模型（RNA-FM + TokenTransformerHead）
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
# 5. 推理与评估（log 空间→线性空间）
# ======================
@torch.no_grad()
def predict_dataset(model, loader, device, loss_fn=None, has_target=True):
    model.eval()
    total_loss = 0.0
    y_true_log, y_pred_log, seqs = [], [], []
    for batch in tqdm(loader, desc="预测中", leave=False):
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
    """尽可能从各种常见格式里取出真正的 state_dict，并剥离常见前缀。"""
    obj = torch.load(ckpt_path, map_location=device)

    # 1) 从常见外壳里解嵌套
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "ema_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}")

    state = obj

    # 2) 连续剥常见前缀
    def strip_prefix(d, prefix):
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

    for pref in ["_orig_mod.", "module.", "model.", "net."]:
        state = strip_prefix(state, pref)

    return state


def load_trained_model(exp_dir, config: Config, device):
    """更健壮的权重加载：能对上的就加载，对不上的保持预训练权重。"""
    # 允许你仍用 best_model_final.pth；若不存在，可自己改成你目录里的 ckpt 名
    weight_path = os.path.join(exp_dir, "best_model_final.pth")
    if not os.path.exists(weight_path):
        # 兜底：找目录里第一个 .pth
        cand = [f for f in os.listdir(exp_dir) if f.endswith(".pth")]
        if not cand:
            raise FileNotFoundError(f"未找到权重文件：{weight_path}，且目录下也没有 .pth。")
        weight_path = os.path.join(exp_dir, cand[0])

    # 构建模型（先加载预训练 RNA-FM）
    model = ChunkingMRNATransformer(config).to(device)

    # 取出 checkpoint 的 state_dict 并做键名清洗
    raw_state = _read_state_dict_flex(weight_path, device)

    # === 新增：把 ckpt 的 bert.* / 纯 HF 键 映射到 wrapper 的 backbone.* ===
    def remap_backbone_prefix(state: dict) -> dict:
        """
        将 ckpt 的 bert.* / 纯 HF 键(embeddings./encoder./pooler.) 映射为 wrapper 的 backbone.*
        不动 token_head.* 等其它键；并清理可能的 backbone.backbone.* 重复前缀。
        """
        keys = list(state.keys())
        has_backbone = any(k.startswith("backbone.") for k in keys)
        has_bert = any(k.startswith("bert.") for k in keys)

        # 情况1：ckpt 用的是 bert.* —— 映射到 backbone.*
        if has_bert and not has_backbone:
            state = {("backbone." + k[5:] if k.startswith("bert.") else k): v for k, v in state.items()}
            keys = list(state.keys())
            has_backbone = any(k.startswith("backbone.") for k in keys)

        # 情况2：ckpt 是纯 HF 键（无前缀）：embeddings./encoder./pooler.
        has_hf_root = any(k.startswith(("embeddings.", "encoder.", "pooler.")) for k in keys)
        if has_hf_root and not has_backbone:
            state = {("backbone." + k if k.startswith(("embeddings.", "encoder.", "pooler.")) else k): v
                     for k, v in state.items()}

        # 清理可能的重复前缀
        state = {(k.replace("backbone.backbone.", "backbone.") if k.startswith("backbone.backbone.") else k): v
                 for k, v in state.items()}
        return state

    raw_state = remap_backbone_prefix(raw_state)

    model_state = model.state_dict()

    # 只保留“键存在且 shape 一致”的条目
    filtered = {}
    mismatched_shapes = {}
    for k, v in raw_state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        elif k in model_state:
            mismatched_shapes[k] = {"ckpt": tuple(v.shape), "model": tuple(model_state[k].shape)}

    # 加载（宽松模式）
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # 打印/记录加载报告（方便你审计）
    report = {
        "ckpt_path": weight_path,
        "loaded_keys": len(filtered),
        "missing_in_ckpt_but_in_model": list(missing),      # 模型需要但 ckpt 没有（例如 backbone.*）
        "unexpected_in_ckpt": list(unexpected),             # ckpt 里有但模型没有
        "shape_mismatch": mismatched_shapes                 # 键同名但 shape 不同
    }
    try:
        out_dir, _, _ = create_output_dir()
        with open(os.path.join(out_dir, "ckpt_load_report.json"), "w") as f:
            import json; json.dump(report, f, indent=2)
    except Exception:
        pass

    # 友好提示
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
# 6. motif 统计与 in-silico mutation
# ======================


def count_motif(seq: str, motif: str) -> int:
    return len(re.findall(f"(?={motif})", seq))  # 支持重叠计数

def tail_A_fraction(seq: str, tail_len: int = 50) -> float:
    tail = seq[-tail_len:] if len(seq) >= tail_len else seq
    return (tail.count("A") / len(tail)) if len(tail) > 0 else 0.0

def make_motif_table(df_residuals: pd.DataFrame) -> pd.DataFrame:
    motifs = ["AUUUA", "AATAAA", "ATTAAA"]  # 注意：如果你的序列是 DNA（含T），ARE 请改成 ATTTA
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
    In-silico 突变：
      - 当 RunParams.MUTATION_MODE == "full" 时：全量枚举（所有含 motif 的样本 × 所有出现位置 × 所有替代序列）
      - 当 RunParams.MUTATION_MODE == "per-motif" 时：对每个 motif 独立抽样，至少 sample_k（外层传入的 NUM_MUTATION_SAMPLES）条含该 motif 的样本

    返回列：sample_idx, motif, new, pos, base_pred, mut_pred, delta, sequence
    """
    rng = np.random.default_rng(2024)

    # 一一对应映射（按 RunParams.MOTIFS 与 REPLACEMENTS 对齐）
    params = RunParams()
    motif_to_repls = build_paired_mapping(params)
    motifs = list(motif_to_repls.keys())

    # 仅在“含任一目标 motif 的序列”上进行（避免无效样本）
    union_pat = re.compile("(?:%s)" % "|".join(map(re.escape, motifs)))
    has_any = df_base["sequence"].str.contains(union_pat)

    eligible_all = np.where(has_any.values)[0]
    if len(eligible_all) == 0:
        return pd.DataFrame(columns=["sample_idx","motif","new","pos","base_pred","mut_pred","delta","sequence"])

    # 计算/缓存 base_pred（只对会被用到的序列）
    def batch_predict_base(idx_list: list[int]) -> dict[int, float]:
        seq_list = [df_base.iloc[i]["sequence"] for i in idx_list]
        preds = predict_sequence_list(model, tokenizer, config, device, seq_list)
        return {i: float(p) for i, p in zip(idx_list, preds)}

    records = []
    params = RunParams()
    max_pos_per_seq = params.MAX_POS_PER_SEQ_PER_MOTIF

    if getattr(params, "MUTATION_MODE", "full") == "full":
        # ========= 全量枚举 =========
        # 对所有“含任一 motif”的序列，先一次性拿到 base_pred
        base_pred_map = batch_predict_base(list(eligible_all))

        # 逐条序列处理：对该序列中每个 motif 的每个出现位置，构造全部替代，批量预测
        for idx in tqdm(eligible_all, desc="In-silico mutation (full)", leave=False):
            seq = df_base.iloc[idx]["sequence"]
            base_pred = base_pred_map[idx]

            mut_tasks = []   # (motif, new, pos, mut_seq)
            for motif in motifs:
                positions = [m.start() for m in re.finditer(f"(?={motif})", seq)]
                if max_pos_per_seq is not None and len(positions) > max_pos_per_seq:
                    # 如需限制位置数量，随机抽指定个数
                    positions = list(rng.choice(positions, size=max_pos_per_seq, replace=False))
                for pos in positions:
                    for newmotif in motif_to_repls.get(motif, []):
                        mut_seq = mutate_once(seq, pos, motif, newmotif)
                        if mut_seq is not None:
                            mut_tasks.append((motif, newmotif, pos, mut_seq))

            if not mut_tasks:
                continue

            # 对该序列的所有突变一次性预测
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
        # ========= per-motif 抽样 =========
        per_quota = int(sample_k)  # 外层传入的是 RunParams.NUM_MUTATION_SAMPLES
        # 为每个 motif 找出“含该 motif”的样本集合，并按配额抽样
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

        # 需要 base_pred 的唯一样本索引集合
        uniq_idxs = sorted(set(int(i) for arr in motif_to_indices.values() for i in arr))
        if len(uniq_idxs) == 0:
            return pd.DataFrame(columns=["sample_idx","motif","new","pos","base_pred","mut_pred","delta","sequence"])
        base_pred_map = batch_predict_base(uniq_idxs)

        # 逐 motif、逐样本、逐位置 枚举替代；对每个样本做“批量突变预测”
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
# 7. 主流程（不再使用命令行）
# ======================
def main():
    # 读取内置参数
    params = RunParams()
    exp_dir = os.path.abspath(params.EXP_DIR)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"EXP_DIR 不存在或不是文件夹：{exp_dir}")

    # 其余保持与主程序一致
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)
    device = get_device()

    out_dir, tb_dir, project_root = create_output_dir()
    writer = SummaryWriter(log_dir=tb_dir)

    # 环境记录
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

    # 载入 tokenizer
    from multimolecule import RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(cfg.PRETRAINED_MODEL_NAME, trust_remote_code=True)
    collate = partial(collate_fn_no_chunk, tokenizer=tokenizer, config=cfg)

    # 载入已训练模型
    model = load_trained_model(exp_dir, cfg, device)

    # 读取主实验的测试集预测，若无则回退到按主程序拆分并预测
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
            raise FileNotFoundError(f"数据文件未找到：{data_path}")
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

    # 残差表
    df_base["residual"] = df_base["true"] - df_base["pred"]
    df_base.to_csv(os.path.join(out_dir, "residuals.csv"), index=False)

    # 写入测试指标（使用已有 pred/true）
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

    # motif 与残差相关
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

    # 总结
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

    # 直方图写入TB
    try:
        writer.add_histogram("Residuals/hist", (df_base["residual"].values.astype(np.float32)), 0)
        if len(df_mut) > 0:
            writer.add_histogram("Mutation/delta_hist", (df_mut["delta"].values.astype(np.float32)), 0)
    except Exception:
        pass

    writer.close()
    print("\n✅ 解释性分析完成，结果保存在：", out_dir)
    print("  - residuals.csv（残差表）")
    print("  - motif_corr.csv（motif 与残差相关）")
    print("  - mutation_results.csv（in-silico mutation）")
    print("  - summary.json / interpret_env.json")
    print("  - TensorBoard：", os.path.join(out_dir, "tensorboard-log", "interpretability"))

if __name__ == "__main__":
    main()
