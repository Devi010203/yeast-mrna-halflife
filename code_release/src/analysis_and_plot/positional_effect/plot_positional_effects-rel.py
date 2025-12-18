# plots/plot_positional_effects_relative_fp32_cached.py
# -*- coding: utf-8 -*-
"""
Position Effect (Relative) Curve: rΔ = (baseline − occluded) / baseline
 Full precision (FP32),  Full-process caching,  Checkpointing,  Aggregation-only mode
 Modified to **chunk-free** version: no longer relies on collate_fn_chunking / chunks_per_sample,
   model forward unified to model(input_ids, attention_mask), prioritising reuse of collate_fn_no_chunk from main script.

Output to:
  - Cache: <project>/result/cache/positional_effects_rel/<RUN_NAME>/
      baseline.csv (baseline prediction + length + md5 for each sample)
      occl/seq_XXXXXX.csv (start, centre, occ_pred, delta, rel for each window of this sample)
      meta.json     (snapshot of key configurations)
  - Visualisations: <project>/result/plot/positional_effects_rel/<timestamp>/
      Curves/Bar charts/Top-K heatmaps (PNG+SVG, dpi=400) + corresponding CSV
"""

import os, sys, math, re, json, hashlib, importlib.util, types, time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


import torch
from torch.utils.data import Dataset

class _SeqDS(Dataset):
    """Top-level Dataset, Windows multi-process pickleable; used for chunkless inference"""
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        # Inference does not require a real target, but to reuse the training collate, provide a placeholder 0.0.
        return {"sequence": self.seqs[idx],
                "target": torch.tensor(0.0, dtype=torch.float)}


# ==== Offline & CUDA Memory Strategy ====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========= Please fill in manually here=========
CONFIG = {
    # Your main training script (containing Config, ChunkingMRNATransformer, get_device; if collate_fn_no_chunk is
    # included, it will be prioritised)
    "MODEL_CODE_PATH": r"F:\mRNA_Project\3UTR\Paper\script\5f_full_head_v3.py",

    # Complete training output directory (must contain final_test_predictions.csv -> providing sequences)
    "RUN_DIR": r"F:\mRNA_Project\3UTR\Paper\result\5f_full_head_v3_20251024_01",

    # Task Weight (.pth)
    "CKPT_PATH": r"F:\mRNA_Project\3UTR\Paper\result\5f_full_head_v3_20251024_01\best_model_final.pth",

    # Local tokeniser / RnaFmModel directory
    "LOCAL_TOKENIZER_DIR": r"F:\mRNA_Project\3UTR\Paper\script\model\rna-fm",
    "LOCAL_RNAFM_DIR": r"F:\mRNA_Project\3UTR\Paper\script\model\rna-fm",

    # Optional root directory (for resolving relative paths; if left blank, the parent directory of the script is used)
    "BASE_DIR": r"",

    # Mode: "all" (compute + aggregate output), "compute_only" (compute cache only), "aggregate_only" (read-only cache output)
    "MODE": "all",

    # Cache Settings
    "CACHE_ROOT": r"result/cache/positional_effects_rel",
    "RUN_NAME": "",    # If left blank, it will be automatically combined into W{W}_S{S}_full/n{N}

    # Occlusion Settings
    "SUBSET_N":200,        # None=Full volume; a larger number may also be set, for example 5000/10000
    "WINDOW_SIZE": 15,
    "WINDOW_STEP": 5,
    "FILL_CHAR": "N",

    # Deduction
    "BATCH_SIZE": 16,        # Graphics memory can be increased if sufficient; Scripts encountering OOM will automatically halve and retry.
    "SEED": 42,

    # Normalised binning
    "NORM_POS_BINS": 50,

    # Top-K Heatmap (representative sample)
    "TOP_HEATMAP_K": 12,

    # (Optional) Length stratification
    "DO_LENGTH_STRATA": False,
    "LENGTH_BINS_ABS": None,
    "LENGTH_QUANTILES": [0.0, 0.33, 0.66, 1.0],

    # Plotting
    "DPI": 400,
    "FIGSIZE_CURVE": (6.4, 4.6),
    "FIGSIZE_HEATMAP": (8.0, 4.8),
    "GRID_ALPHA": 0.35,
    "SAVE_SUBDIR": "positional_effects_rel",
}
# =================================


matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans CJK SC"],
    "font.style": "normal",
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
})

# ---------- Path Normalisation ----------
def _is_windows_style(p: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", p or ""))

def _abs_path(p: str) -> str:
    if not p: return ""
    if os.name != "nt" and _is_windows_style(p):
        raise RuntimeError(f"Windows path detected: {p}. Please change to an absolute Linux path on the server.")
    q = Path(p)
    if q.is_absolute():
        return str(q)
    base = Path(CONFIG.get("BASE_DIR") or Path(__file__).resolve().parents[1])
    return str((base / q).resolve())

def _normalize_config_paths():
    for key in ["MODEL_CODE_PATH", "RUN_DIR", "CKPT_PATH", "LOCAL_TOKENIZER_DIR", "LOCAL_RNAFM_DIR", "CACHE_ROOT"]:
        CONFIG[key] = _abs_path(CONFIG[key])

# ---------- Output directory ----------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_plot_outdir(sub: str) -> str:
    outdir = _project_root() / "result" / "plot" / sub / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)

def _save_dual(fig, out_base: str, dpi: int):
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- Cache directory ----------
def _ensure_cache_dir() -> str:
    root = Path(CONFIG["CACHE_ROOT"])
    run_name = CONFIG["RUN_NAME"].strip()
    if not run_name:
        tag = f"W{CONFIG['WINDOW_SIZE']}_S{CONFIG['WINDOW_STEP']}"
        tag2 = "full" if CONFIG["SUBSET_N"] is None else f"n{CONFIG['SUBSET_N']}"
        run_name = f"{tag}_{tag2}"
    cache_dir = root / run_name
    (cache_dir / "occl").mkdir(parents=True, exist_ok=True)
    return str(cache_dir)

# ---------- Tools ----------
def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _write_json(fp: Path, obj: dict):
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _bootstrap_ci_mean(y: np.ndarray, n_boot=1000, alpha=0.05, seed=20251016):
    r = np.random.RandomState(seed)
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = y.size
    if n <= 1: return (np.nan, np.nan)
    idx = np.arange(n)
    means = []
    for _ in range(n_boot):
        samp = r.choice(idx, size=n, replace=True)
        means.append(float(np.mean(y[samp])))
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1 - alpha/2))

def _bin_index(center_pos: float, bins: int) -> int:
    i = int(math.floor(center_pos * bins))
    return max(0, min(bins - 1, i))

def _interp_to_bins(x_pos: np.ndarray, x_val: np.ndarray, bins: int) -> np.ndarray:
    if x_pos.size == 0: return np.zeros(bins, dtype=float)
    grid = (np.arange(bins) + 0.5) / bins
    return np.interp(grid, x_pos, x_val, left=x_val[0], right=x_val[-1])

# ---------- Dynamically import your training script ----------
def _import_model_module(py_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("user_model_code", py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # Register for subsequent access
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---------- Load tokeniser locally only ----------
def _load_tokenizer_locally(RnaTokenizer, config, model_code_path: str, explicit_dir: Optional[str] = None):
    candidates = []
    if explicit_dir: candidates.append(Path(explicit_dir))
    candidates.append(Path(model_code_path).parent / str(config.PRETRAINED_MODEL_NAME))
    candidates.append(Path(str(config.PRETRAINED_MODEL_NAME)))
    last_err = None
    for p in candidates:
        if p and p.is_dir():
            try:
                tok = RnaTokenizer.from_pretrained(str(p), trust_remote_code=True, local_files_only=True)
                print(f"[Tokenizer] Already loaded from local：{p}")
                return tok
            except Exception as e:
                last_err = e
    raise RuntimeError(
        "Failed to load tokeniser locally. Please set LOCAL_TOKENIZER_DIR or ensure PRETRAINED_MODEL_NAME points to a local directory.\n"
        f"Final error：{last_err}"
    )

# ---------- Weight Key Name Prefix Remapping ----------
def _prepare_state_dict_for_model(state_obj, model) -> OrderedDict:
    sd = state_obj
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model_keys = list(model.state_dict().keys())
    target_prefix = None
    for pref in ("bert.", "backbone.", "fm."):
        if any(k.startswith(pref) for k in model_keys):
            target_prefix = pref
            break
    if target_prefix is None:
        target_prefix = "bert."

    def strip_wrap_prefix(k: str) -> str:
        for p in ("module.", "model."):
            if k.startswith(p): return k[len(p):]
        return k

    def detect_src_prefix(keys):
        for p in ("fm.", "backbone.", "bert."):
            if any(k.startswith(p) for k in keys): return p
        return None

    src_prefix = detect_src_prefix(sd.keys())
    remapped = OrderedDict()
    for k, v in sd.items():
        k2 = strip_wrap_prefix(k)
        if src_prefix and k2.startswith(src_prefix) and src_prefix != target_prefix:
            k2 = target_prefix + k2[len(src_prefix):]
        remapped[k2] = v
    return remapped

# ---------- collate of no-chunk (preferred reuse of main script; otherwise local implementation) ----------
def _build_nochunk_collate(mod, tokenizer, config):
    """
    Prioritise the collate_fn_no_chunk from the main script; if absent, construct an equivalent no-chunk collate locally.
    Returns: A collate_fn suitable for direct use with DataLoader.
    """
    fn = getattr(mod, "collate_fn_no_chunk", None)
    if fn is not None:
        from functools import partial
        return partial(fn, tokenizer=tokenizer, config=config)

    import torch
    def _local_collate(batch):
        sequences = [item["sequence"] for item in batch]
        targets = torch.stack([item.get("target", torch.tensor(0.0, dtype=torch.float)) for item in batch])
        tokenized = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=getattr(config, "MODEL_MAX_LENGTH", 1024),
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "targets": targets,
            "sequence": sequences,
        }
    return _local_collate

# ---------- Full-precision inference (adaptive mini-batches, no AMP/TF32), no chunked forward passes ----------
def _predict_batch(model, tokenizer, config, device, sequences: List[str], batch_size: int) -> np.ndarray:
    """
    Full-precision inference; upon encountering an out-of-memory error, automatically halves the batch_size and retries.
Forward unification: model(input_ids, attention_mask); does not utilise chunks_per_sample.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader

    class _TmpDS(Dataset):
        def __init__(self, seqs): self.seqs = seqs
        def __len__(self): return len(self.seqs)
        def __getitem__(self, idx):
            return {"sequence": self.seqs[idx], "target": torch.tensor(0.0, dtype=torch.float)}

    collate = getattr(config, "_COLLATE_NOCHUNK", None)
    if collate is None:
        collate = _build_nochunk_collate(sys.modules["user_model_code"], tokenizer, config)
        config._COLLATE_NOCHUNK = collate

    preds_all = []
    i = 0
    N = len(sequences)
    cur_bs = max(1, int(batch_size))

    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    with torch.inference_mode():
        while i < N:
            end = min(N, i + max(cur_bs, 1))
            subset = sequences[i:end]

            # Prioritise disabling multi-process on Windows; maintain 2 on Linux.
            workers = 0 if os.name == "nt" else 2
            persist = (workers > 0)

            ds = _SeqDS(subset)
            dl = DataLoader(
                ds,
                batch_size=cur_bs,
                shuffle=False,
                collate_fn=collate,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=persist,
            )

            # Compatibility fallback: Should pickling errors persist, automatically revert to single-process mode.
            try:
                iterator = iter(dl)
            except Exception as e:
                if "pickle" in str(e).lower() or "attributeerror" in str(e).lower():
                    workers = 0
                    persist = False
                    dl = DataLoader(
                        ds,
                        batch_size=cur_bs,
                        shuffle=False,
                        collate_fn=collate,
                        num_workers=workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=persist,
                    )
                else:
                    raise

            ok = True
            part = []
            try:
                for batch in dl:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    # No AMP, FP32 forward pass; output is log1p half-life
                    out = model(input_ids, attention_mask)
                    part.append(np.expm1(out.detach().cpu().numpy()))  # Restore to original scale
                    del out, input_ids, attention_mask
            except torch.cuda.OutOfMemoryError:
                ok = False
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                print(f"[OOM] reduce batch_size → {cur_bs} and retry（i={i}）")

            if ok:
                preds_all.append(np.concatenate(part, axis=0) if part else np.zeros((0,), dtype=float))
                i = end

    return np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0,), dtype=float)

# ---------- Read/Prepare Sequence ----------
def _load_sequences() -> List[str]:
    fp = os.path.join(CONFIG["RUN_DIR"], "final_test_predictions.csv")
    if not Path(fp).is_file():
        raise FileNotFoundError(f"Not found {fp}")
    df_all = pd.read_csv(fp)
    if "sequence" not in df_all.columns:
        raise ValueError(f"{fp} The 'sequence' column is missing.")
    df_all = df_all.dropna(subset=["sequence"]).copy()
    df_all["sequence"] = df_all["sequence"].astype(str)
    seqs_all = df_all["sequence"].tolist()

    rng = np.random.RandomState(CONFIG["SEED"])
    if CONFIG["SUBSET_N"] is None:
        seqs = seqs_all
    elif len(seqs_all) > int(CONFIG["SUBSET_N"]):
        idx = rng.choice(np.arange(len(seqs_all)), size=int(CONFIG["SUBSET_N"]), replace=False)
        seqs = [seqs_all[i] for i in idx]
    else:
        seqs = seqs_all
    return seqs

# ---------- Computing + Caching (without chunking) ----------
def compute_and_cache(cache_dir: str):
    print("[cache dir]", cache_dir)
    # Import your training module
    mod = _import_model_module(CONFIG["MODEL_CODE_PATH"])
    Config = mod.Config
    ChunkingMRNATransformer = mod.ChunkingMRNATransformer
    get_device = mod.get_device
    try:
        from multimolecule import RnaTokenizer
    except Exception as e:
        raise ImportError("Unable to import RnaTokenizer from multimolecule. Please run in the same environment.") from e

    # Initialise device/model
    import torch
    config = Config()
    device = get_device()
    tokenizer = _load_tokenizer_locally(
        RnaTokenizer=RnaTokenizer, config=config,
        model_code_path=CONFIG["MODEL_CODE_PATH"],
        explicit_dir=(CONFIG.get("LOCAL_TOKENIZER_DIR") or None)
    )
    # Direct PRETRAINED_MODEL_NAME to the local RnaFmModel directory
    local_model_dir = CONFIG.get("LOCAL_RNAFM_DIR") or ""
    if not Path(local_model_dir).is_dir():
        cand = Path(CONFIG["MODEL_CODE_PATH"]).parent / str(config.PRETRAINED_MODEL_NAME)
        if cand.is_dir():
            local_model_dir = str(cand.resolve())
        else:
            raise RuntimeError("The local RnaFmModel directory was not found. Please set LOCAL_RNAFM_DIR.")
    config.PRETRAINED_MODEL_NAME = local_model_dir
    print(f"[Model] Using the local RnaFmModel directory:{config.PRETRAINED_MODEL_NAME}")

    config._COLLATE_NOCHUNK = _build_nochunk_collate(mod, tokenizer, config)
    print("[collate] Prepared: no-chunk version")

    model = ChunkingMRNATransformer(config).to(device)
    ckpt = CONFIG["CKPT_PATH"]
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Weight not found：{ckpt}")
    state = torch.load(ckpt, map_location=device)
    cleaned = _prepare_state_dict_for_model(state, model)
    model.load_state_dict(cleaned, strict=False)
    print(f"[OK] Weights loaded (post-compatibility mapping):{ckpt}")

    # Read sequence
    seqs = _load_sequences()
    print(f"[Sample size] Sequence for occlusion:{len(seqs)}")

    # masking characters
    fill_char = str(CONFIG["FILL_CHAR"]).upper()
    allowed = set("ACGTUNRYSMWKBDHVX.-*I")  # IUPAC + Common extensions
    if len(fill_char) != 1 or fill_char not in allowed:
        raise ValueError(f"Illegal FILL_CHAR='{fill_char}'；Please use one of the IUPAC letters (such as N, A, C, G, U, etc.)")
    print(f"[masking characters] Use：{fill_char}")

    # meta Save
    meta = {
        "WINDOW_SIZE": CONFIG["WINDOW_SIZE"],
        "WINDOW_STEP": CONFIG["WINDOW_STEP"],
        "BATCH_SIZE": CONFIG["BATCH_SIZE"],
        "SEED": CONFIG["SEED"],
        "FILL_CHAR": fill_char,
        "SUBSET_N": CONFIG["SUBSET_N"],
        "NORM_POS_BINS": CONFIG["NORM_POS_BINS"],
        "MODEL_CODE_PATH": CONFIG["MODEL_CODE_PATH"],
        "CKPT_PATH": CONFIG["CKPT_PATH"],
        "LOCAL_RNAFM_DIR": CONFIG["LOCAL_RNAFM_DIR"],
        "LOCAL_TOKENIZER_DIR": CONFIG["LOCAL_TOKENIZER_DIR"],
        "NO_CHUNK": True,
    }
    _write_json(Path(cache_dir) / "meta.json", meta)

    # Baseline forecast (reuses baseline.csv if present)
    baseline_fp = Path(cache_dir) / "baseline.csv"
    if baseline_fp.exists():
        dfb = pd.read_csv(baseline_fp)
        base_pred = dfb["base_pred"].values.astype(float)
        print(f"[baseline] Reuse existing:{baseline_fp} (n={len(dfb)})")
    else:
        base_pred = _predict_batch(model, tokenizer, config, device, seqs, CONFIG["BATCH_SIZE"])
        dfb = pd.DataFrame({
            "seq_idx": np.arange(len(seqs), dtype=int),
            "length": [len(s) for s in seqs],
            "seq_md5": [ _md5(s) for s in seqs ],
            "base_pred": base_pred,
            # If the original sequence is to be retained, uncomment the next line (note that this will result in a larger file size).
            # "sequence": seqs,
        })
        dfb.to_csv(baseline_fp, index=False)
        print(f"[baseline] Saved：{baseline_fp}")

    # Window prediction and relative effects for each sample (processed sequentially, with support for resuming from breakpoints)
    W = int(CONFIG["WINDOW_SIZE"]); S = int(CONFIG["WINDOW_STEP"])
    occl_dir = Path(cache_dir) / "occl"
    for i, seq in enumerate(seqs):
        occ_i = occl_dir / f"seq_{i:06d}.csv"
        if occ_i.exists():
            continue  # Resume from breakpoint: Skip if already present
        L = len(seq)
        if L < W:
            pd.DataFrame(columns=["start","center","occ_pred","delta","rel"]).to_csv(occ_i, index=False)
            continue
        starts = list(range(0, L - W + 1, S))
        centers = np.array([(st + W/2)/L for st in starts], dtype=float)
        occ_seqs = [seq[:st] + fill_char*W + seq[st+W:] for st in starts]
        occ_pred = _predict_batch(model, tokenizer, config, device, occ_seqs, CONFIG["BATCH_SIZE"])
        b = float(base_pred[i])
        if b <= 1e-9:
            rel = np.full_like(occ_pred, np.nan, dtype=float)
        else:
            rel = (b - occ_pred) / b
        delta = (b - occ_pred)
        dfo = pd.DataFrame({
            "start": starts,
            "center": centers,
            "occ_pred": occ_pred,
            "delta": delta,
            "rel": rel,
        })
        dfo.to_csv(occ_i, index=False)

    print("[compute] Occlusion results for all samples have been cached:", cache_dir)

# ---------- Aggregation & Image Generation (read from cache, no inference performed) ----------
def aggregate_and_plot_from_cache(cache_dir: str):
    print("[aggregate] Read cache：", cache_dir)
    meta = json.loads(Path(cache_dir, "meta.json").read_text(encoding="utf-8"))
    W = int(meta["WINDOW_SIZE"]); S = int(meta["WINDOW_STEP"])
    BINS = int(CONFIG["NORM_POS_BINS"])
    outdir = _ensure_plot_outdir(CONFIG["SAVE_SUBDIR"])
    print("[Output directory]", outdir)

    # Read baseline
    dfb = pd.read_csv(Path(cache_dir) / "baseline.csv")
    N = len(dfb)
    print(f"[baseline] n={N}")

    # Aggregated into bins (relative effect)
    bin_rel_effects: List[List[float]] = [[] for _ in range(BINS)]
    # Also construct a Top-K matrix (interpolated to fixed bins)
    rows_interp = []
    lengths = dfb["length"].values.astype(int)

    for i in range(N):
        occ_i = Path(cache_dir) / "occl" / f"seq_{i:06d}.csv"
        if not occ_i.exists():
            continue
        dfo = pd.read_csv(occ_i)
        if dfo.shape[0] == 0:
            continue
        centers = dfo["center"].values.astype(float)
        rel = dfo["rel"].values.astype(float)
        # Write to the bin container
        for r, c in zip(rel, centers):
            if not np.isnan(r):
                bin_rel_effects[_bin_index(float(c), BINS)].append(float(r))
        # For use with heatmaps: Interpolate to BINS
        rows_interp.append(_interp_to_bins(centers, np.nan_to_num(rel, nan=0.0), BINS))
    mat = np.vstack(rows_interp) if len(rows_interp) else np.zeros((0, BINS), dtype=float)

    # === All rΔ curves ===
    x_centers = (np.arange(BINS) + 0.5) / BINS
    mean_rel, lo_rel, hi_rel = [], [], []
    for b in range(BINS):
        arr = np.array(bin_rel_effects[b], dtype=float)
        m = float(np.nanmean(arr)) if arr.size else np.nan
        mean_rel.append(m)
        l, h = _bootstrap_ci_mean(arr, n_boot=1000, alpha=0.05, seed=CONFIG["SEED"]) if arr.size else (np.nan, np.nan)
        lo_rel.append(l); hi_rel.append(h)

    def _save_curve_csv_plot(x_centers, mean_y, lo, hi, outdir, stem, ylabel, title):
        df = pd.DataFrame({"center": x_centers, "mean": mean_y, "ci_lo": lo, "ci_hi": hi})
        df.to_csv(os.path.join(outdir, stem + ".csv"), index=False)
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_CURVE"])
        ax.plot(x_centers, mean_y, linewidth=1.8, marker="o", markersize=3)
        ax.fill_between(x_centers, lo, hi, alpha=0.25)
        ax.set_xlabel("Normalized position (5'→3')")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
        fig.savefig(os.path.join(outdir, stem + ".png"), dpi=CONFIG["DPI"], bbox_inches="tight")
        fig.savefig(os.path.join(outdir, stem + ".svg"), dpi=CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    _save_curve_csv_plot(
        x_centers, mean_rel, lo_rel, hi_rel, outdir,
        stem="positional_effect_curve_relative_fp32_from_cache",
        ylabel="Relative effect Δ \/ baseline",
        title=f"Relative positional effect (FP32, W={W}, step={S}, n={N})"
    )

    # === Segment Mean Bar Chart (with 95% Confidence Interval and Significance) ===
    seg_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    seg_labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    seg_vals = [[] for _ in range(len(seg_labels))]
    for b in range(BINS):
        c = x_centers[b]
        idx = np.searchsorted(seg_edges, c, side="right") - 1
        idx = max(0, min(idx, len(seg_labels)-1))
        seg_vals[idx].extend(bin_rel_effects[b])
    seg_means, seg_lo, seg_hi = [], [], []
    for vals in seg_vals:
        arr = np.array(vals, dtype=float)
        seg_means.append(float(np.nanmean(arr)) if arr.size else np.nan)
        l, h = _bootstrap_ci_mean(arr, n_boot=1000, alpha=0.05, seed=CONFIG["SEED"]) if arr.size else (np.nan, np.nan)
        seg_lo.append(l); seg_hi.append(h)

    def _bar_with_ci_and_sig(labels: List[str], means: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                             title: str, ylabel: str, out_base: str, dpi: int):
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_CURVE"])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[np.array(means)-np.array(lo), np.array(hi)-np.array(means)], capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
        for i, (l, h) in enumerate(zip(lo, hi)):
            if not (np.isnan(l) or np.isnan(h)) and (l > 0 or h < 0):
                ax.text(i, max(0, means[i]) + (hi[i]-means[i])*0.15 + 0.02, "*",
                        ha="center", va="bottom", fontsize=14)
        fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    _bar_with_ci_and_sig(
        labels=seg_labels,
        means=np.array(seg_means, dtype=float),
        lo=np.array(seg_lo, dtype=float),
        hi=np.array(seg_hi, dtype=float),
        title="Relative effect by region (FP32, from cache)",
        ylabel="Δ / baseline",
        out_base=os.path.join(outdir, "positional_effect_relative_segments_fp32_from_cache"),
        dpi=CONFIG["DPI"]
    )
    pd.DataFrame({"segment": seg_labels, "mean": seg_means, "ci_lo": seg_lo, "ci_hi": seg_hi}).to_csv(
        os.path.join(outdir, "positional_effect_relative_segments_fp32_from_cache.csv"), index=False
    )

    # === Top-K rΔ Heatmap ===
    if mat.size:
        row_score = np.nanmean(mat, axis=1)
        order = np.argsort(-row_score)
        K = min(CONFIG["TOP_HEATMAP_K"], len(order))
        mat_top = mat[order[:K], :]
        pd.DataFrame(mat_top, columns=[f"bin_{i}" for i in range(BINS)]).to_csv(
            os.path.join(outdir, "positional_effect_heatmap_relative_topK_fp32_from_cache.csv"), index=False
        )
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_HEATMAP"])
        im = ax.imshow(mat_top, aspect="auto", origin="lower")
        ax.set_xlabel("Normalized position bins (5'→3')")
        ax.set_ylabel("Top-K sequences")
        ax.set_title(f"Relative positional effect — heatmap of top {K} sequences (FP32, from cache)")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ / baseline")
        fig.savefig(os.path.join(outdir, "positional_effect_heatmap_relative_topK_fp32_from_cache.png"),
                    dpi=CONFIG["DPI"], bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "positional_effect_heatmap_relative_topK_fp32_from_cache.svg"),
                    dpi=CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    # === (Optional) Length stratification ===
    if CONFIG.get("DO_LENGTH_STRATA", False) and mat.size:
        # Establishing a threshold
        bins_abs = CONFIG.get("LENGTH_BINS_ABS", None)
        if bins_abs is None:
            qs = CONFIG.get("LENGTH_QUANTILES", [0.0, 0.33, 0.66, 1.0])
            edges = np.quantile(lengths, qs)
        else:
            edges = np.array(bins_abs, dtype=float)
        edges = np.unique(edges)
        labels = [f"[{int(edges[j])},{int(edges[j+1])})" for j in range(len(edges)-1)]

        for j in range(len(edges)-1):
            lo_e, hi_e = edges[j], edges[j+1]
            mask = (lengths >= lo_e) & (lengths < hi_e)
            if not np.any(mask): continue
            sub = mat[mask, :]
            mean_col = np.nanmean(sub, axis=0)

            lo_ci, hi_ci = [], []
            r = np.random.RandomState(CONFIG["SEED"])
            idx = np.arange(sub.shape[0])
            if sub.shape[0] > 1:
                for c in range(BINS):
                    col = sub[:, c]
                    boots = []
                    for _ in range(1000):
                        samp = r.choice(idx, size=len(idx), replace=True)
                        boots.append(float(np.nanmean(col[samp])))
                    lo_ci.append(float(np.quantile(boots, 0.025)))
                    hi_ci.append(float(np.quantile(boots, 0.975)))
            else:
                lo_ci = [np.nan]*BINS; hi_ci = [np.nan]*BINS

            stem = f"positional_effect_curve_relative_lenbin{j+1}_fp32_from_cache"
            df = pd.DataFrame({"center": x_centers, "mean": mean_col, "ci_lo": lo_ci, "ci_hi": hi_ci})
            df.to_csv(os.path.join(outdir, stem + ".csv"), index=False)
            fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_CURVE"])
            ax.plot(x_centers, mean_col, linewidth=1.8, marker="o", markersize=3)
            ax.fill_between(x_centers, lo_ci, hi_ci, alpha=0.25)
            ax.set_xlabel("Normalized position (5'→3')")
            ax.set_ylabel("Δ / baseline")
            ax.set_title(f"Relative positional effect by length {labels[j]} (FP32, from cache, n={int(mask.sum())})")
            ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
            fig.savefig(os.path.join(outdir, stem + ".png"), dpi=CONFIG["DPI"], bbox_inches="tight")
            fig.savefig(os.path.join(outdir, stem + ".svg"), dpi=CONFIG["DPI"], bbox_inches="tight")
            plt.close(fig)

    print("[Completed] The image has been generated from the cached merge.：", outdir)

# ---------- Main process ----------
def main():
    _normalize_config_paths()
    cache_dir = _ensure_cache_dir()
    mode = CONFIG["MODE"].lower()

    if mode in ("all", "compute_only"):
        compute_and_cache(cache_dir)
    if mode in ("all", "aggregate_only"):
        aggregate_and_plot_from_cache(cache_dir)

if __name__ == "__main__":
    main()
