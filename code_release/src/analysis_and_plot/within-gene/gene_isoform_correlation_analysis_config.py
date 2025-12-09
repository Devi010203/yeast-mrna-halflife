#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gene/isoform-level post hoc analysis for mRNA half-life predictions (config version).

Edit the CONFIG block below to set your file paths, column names, and options.
Then simply run (no CLI args needed):
    python gene_isoform_correlation_analysis_config.py

Outputs:
  - analysis_per_isoform.csv : per-isoform table with derived fields
  - per_gene_summary.csv     : per-gene medians and per-gene correlations
  - metrics_summary.json     : global summary metrics
  - cross_gene_median.csv    : gene-level truth/pred medians (NEW)
  - matching_log.txt         : matching/quality diagnostics
  - Optional: basic scatter plots (PNG + SVG) if CONFIG["plots"] is True
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ========================== CONFIG ==========================
# Set your paths and options here.
CONFIG = {
    # File paths
    "dataset": "F:/mRNA_Project/3UTR/Paper/data/mRNA_half_life_dataset.csv",
    "predictions": "F:/mRNA_Project/3UTR/Paper/result/5f_full_head_v3_20251024_01/final_test_predictions.csv",
    "outdir": "result/gene_isoform_analysis-2",

    # Column names (set to None to auto-detect)
    "dataset_seq_col": None,    # e.g., "sequence"
    "dataset_gene_col": None,   # e.g., "systematic_name"
    "dataset_true_col": None,   # e.g., "half_life"
    "pred_seq_col": None,       # e.g., "sequence"
    "pred_pred_col": "pred",    # e.g., "y_pred"
    "pred_true_col": "true",    # if predictions CSV already includes the true label; else taken from dataset

    # Matching & normalization
    "map_u_to_t": True,         # Normalize sequences by mapping U->T

    # Plots
    "plots": True,              # Save basic scatter plots (PNG + SVG)
}
# ============================================================


# ------------------------- Utilities -------------------------

def normalize_seq(s: str,
                  map_u_to_t: bool = True,
                  remove_whitespace: bool = True) -> str:
    """Normalize a nucleotide sequence string for robust matching.
    - Upper-case
    - Optional: map U->T (RNA to DNA)
    - Optional: remove whitespace
    """
    if pd.isna(s):
        return ""
    s = str(s)
    if remove_whitespace:
        s = "".join(s.split())
    s = s.upper()
    if map_u_to_t:
        s = s.replace("U", "T")
    return s


def _pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k in low: return low[k]
    # 宽松匹配
    for c in cols:
        lc = c.lower()
        if any(tag in lc for tag in candidates):
            return c
    return None

def export_cross_gene_median(df_merged: pd.DataFrame, outdir: Path):
    """
    df_merged 需至少含：基因列、真实列、预测列。
    保存到 outdir / 'cross_gene_median.csv'，列为：
      gene, ref_real, ref_pred, n
    并打印 R²、斜率/截距（ŷ = a·x + b）。
    """
    cols = df_merged.columns

    gene_col = _pick_col(cols, ["gene","systematic_name","systematic","gene_name","geneid","sys","orf","name"])
    true_col = _pick_col(cols, ["true","y_true","label","target","half_life","halflife","half-life","t_half","decay","hl","obs","real","ground_truth"])
    pred_col = _pick_col(cols, ["pred","y_pred","prediction","predicted","pred_half_life","half_life_pred","yhat","estimate"])

    if gene_col is None or true_col is None or pred_col is None:
        raise ValueError(f"缺少必要列：gene/true/pred。检测到的列：{list(cols)}")

    d = df_merged[[gene_col, true_col, pred_col]].copy()
    d = d.replace([np.inf,-np.inf], np.nan).dropna()
    g = (d.groupby(gene_col, as_index=False)
           .agg(ref_real=(true_col, "median"),
                ref_pred=(pred_col, "median"),
                n=(true_col, "size")))

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "cross_gene_median.csv"
    g.to_csv(out_csv, index=False)

    # 计算基因层面的拟合指标（用于角标/文字）
    x = g["ref_real"].to_numpy()
    y = g["ref_pred"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]  # ŷ = a·x + b
    ss_res = float(np.sum((y - (a*x + b))**2))
    ss_tot = float(np.sum((x - np.mean(x))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    print(f"[cross-gene median] saved: {out_csv}")
    print(f"[cross-gene median] R²≈{r2:.3f}, slope≈{a:.3f}, intercept≈{b:.3f}, genes={g.shape[0]}")


def autodetect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first column name found in 'cols' that matches any of the candidate
    strings (case-insensitive, substring-friendly)."""
    lower_cols = [c.lower() for c in cols]
    for cand in candidates:
        cand_l = cand.lower()
        for idx, lc in enumerate(lower_cols):
            if cand_l == lc or cand_l in lc or lc in cand_l:
                return cols[idx]
    return None


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2. Returns NaN if degenerate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


@dataclass
class CorrResult:
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    r2: float
    n: int

def safe_corr(x: np.ndarray, y: np.ndarray) -> CorrResult:
    """Compute Pearson, Spearman, and R^2 with safety checks (NaNs, constant inputs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)

    if n < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return CorrResult(np.nan, np.nan, np.nan, np.nan, np.nan, n)

    try:
        pr, pp = pearsonr(x, y)
    except Exception:
        pr, pp = (np.nan, np.nan)

    try:
        sr, sp = spearmanr(x, y)
        if isinstance(sr, np.ndarray):  # edge case
            sr = float(sr)
    except Exception:
        sr, sp = (np.nan, np.nan)

    r2 = r2_score(y, x)  # Align with y_true=y, y_pred=x for R^2
    return CorrResult(float(pr), float(pp), float(sr), float(sp), float(r2), n)


@dataclass
class SignResult:
    pearson_r: float
    spearman_rho: float
    agreement: float
    n: int

def sign_stats(x: np.ndarray, y: np.ndarray) -> SignResult:
    """Compute sign-based Pearson/Spearman on {-1,0,+1} plus exact sign agreement rate."""
    sx = np.sign(np.asarray(x, dtype=float))
    sy = np.sign(np.asarray(y, dtype=float))
    mask = np.isfinite(sx) & np.isfinite(sy)
    sx = sx[mask]
    sy = sy[mask]
    n = int(sx.size)
    agr = float(np.mean(sx == sy)) if n > 0 else np.nan

    if n < 2 or np.allclose(np.std(sx), 0) or np.allclose(np.std(sy), 0):
        return SignResult(np.nan, np.nan, agr, n)

    try:
        pr, _ = pearsonr(sx, sy)
    except Exception:
        pr = np.nan
    try:
        sr, _ = spearmanr(sx, sy)
        if isinstance(sr, np.ndarray):
            sr = float(sr)
    except Exception:
        sr = np.nan
    return SignResult(float(pr), float(sr), agr, n)


def fisher_z_mean(rs: List[float], ns: List[int]) -> float:
    """Fisher z-transformed mean of correlation coefficients (Pearson), weighted by n-3."""
    vals = []
    wts = []
    for r, n in zip(rs, ns):
        if np.isfinite(r) and n is not None and n >= 4 and abs(r) < 1:
            z = np.arctanh(r)
            w = max(n - 3, 1)
            vals.append(z)
            wts.append(w)
    if not vals:
        return np.nan
    z_mean = np.average(vals, weights=wts)
    return float(np.tanh(z_mean))


def _canon(s: str) -> str:
    """把列名做宽松规范化，用于后缀/大小写/下划线不一致时的模糊匹配。"""
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch == "_")

def _resolve_merged_col(df: pd.DataFrame, base: Optional[str], prefer: str = "either") -> Optional[str]:
    """
    在 merge 之后解析列名：可能是 base、本体；或被 pandas 加了后缀 base_pred/base_data。
    prefer: "pred" | "data" | "either"  指定当两边都有时优先选择哪一侧。
    """
    if base is None:
        return None
    cols = list(df.columns)
    # 先按优先级尝试精确匹配
    candidates = [base]
    if prefer in ("either", "pred"):
        candidates.append(f"{base}_pred")
    if prefer in ("either", "data"):
        candidates.append(f"{base}_data")
    for c in candidates:
        if c in df.columns:
            return c
    # 再做宽松匹配（去掉后缀、大小写/下划线容忍）
    base_core = _canon(base.replace("_pred", "").replace("_data", ""))
    best = None
    for c in cols:
        cc = _canon(c.replace("_pred", "").replace("_data", ""))
        if base_core == cc or base_core in cc or cc in base_core:
            if prefer == "pred" and c.endswith("_pred"):
                return c
            if prefer == "data" and c.endswith("_data"):
                return c
            if best is None:
                best = c
    return best


# ------------------------- Core Analysis -------------------------

def match_predictions(
    df_data: pd.DataFrame,
    df_pred: pd.DataFrame,
    data_seq_col: str,
    pred_seq_col: str,
    map_u_to_t: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """Normalize sequences and merge predictions to dataset on normalized sequence key.
    Returns merged dataframe, diagnostics, and detected gene column name.
    """
    df_data = df_data.copy()
    df_pred = df_pred.copy()
    df_data["_seq_norm"] = df_data[data_seq_col].map(lambda s: normalize_seq(s, map_u_to_t=map_u_to_t))
    df_pred["_seq_norm"] = df_pred[pred_seq_col].map(lambda s: normalize_seq(s, map_u_to_t=map_u_to_t))

    # Detect gene column
    gene_cols_candidates = ["systematic_name", "systematic", "gene", "gene_name", "orf", "name"]
    gene_col_found = None
    for gc in gene_cols_candidates:
        col = autodetect_col(list(df_data.columns), [gc])
        if col is not None:
            gene_col_found = col
            break
    if gene_col_found is None:
        raise ValueError("Could not autodetect a gene name column in dataset. Please set CONFIG['dataset_gene_col'].")

    # Ambiguity check: does a normalized sequence map to multiple genes?
    seq_to_genes = df_data.groupby("_seq_norm")[gene_col_found].nunique()
    ambiguous_keys = set(seq_to_genes[seq_to_genes > 1].index.tolist())

    # Merge (first occurrence per sequence on dataset side)
    df_data_first = df_data.sort_values(by=[gene_col_found]).drop_duplicates(subset=["_seq_norm"], keep="first")
    df_merged = pd.merge(df_pred, df_data_first, on="_seq_norm", how="left", suffixes=("_pred", "_data"))
    df_merged["ambiguous_seq"] = df_merged["_seq_norm"].isin(ambiguous_keys)

    diag = {
        "n_pred_rows": int(len(df_pred)),
        "n_pred_seq_unique": int(df_pred["_seq_norm"].nunique()),
        "n_data_rows": int(len(df_data)),
        "n_data_seq_unique": int(df_data["_seq_norm"].nunique()),
        "n_ambiguous_sequences_in_data": int(len(ambiguous_keys)),
        "n_matched": int(df_merged[gene_col_found].notna().sum()),
        "n_unmatched": int(df_merged[gene_col_found].isna().sum()),
        "n_matched_ambiguous": int(df_merged["ambiguous_seq"].sum()),
    }
    return df_merged, diag, gene_col_found


def run_analysis(config: Dict) -> str:
    # Paths
    dataset_path = os.path.expanduser(config["dataset"])
    pred_path = os.path.expanduser(config["predictions"])
    outdir = os.path.expanduser(config["outdir"])
    os.makedirs(outdir, exist_ok=True)

    # Load CSVs
    try:
        df_data = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df_data = pd.read_csv(dataset_path, encoding="utf-8-sig")
    try:
        df_pred = pd.read_csv(pred_path, encoding="utf-8")
    except UnicodeDecodeError:
        df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")

    # Column resolution (before merge)
    dataset_seq_col = config.get("dataset_seq_col")
    dataset_gene_col = config.get("dataset_gene_col")
    dataset_true_col = config.get("dataset_true_col")
    pred_seq_col = config.get("pred_seq_col")
    pred_pred_col = config.get("pred_pred_col")
    pred_true_col = config.get("pred_true_col")

    if dataset_seq_col is None:
        dataset_seq_col = autodetect_col(list(df_data.columns), ["sequence", "seq", "utr", "utr3", "utr_3", "rna", "dna"])
    if dataset_gene_col is None:
        dataset_gene_col = autodetect_col(list(df_data.columns), ["systematic_name", "systematic", "gene", "gene_name", "orf", "name"])
    if dataset_true_col is None and pred_true_col is None:
        dataset_true_col = autodetect_col(list(df_data.columns), ["half_life", "halflife", "half-life", "t_half", "decay", "hl", "y_true", "label", "target"])

    if pred_seq_col is None:
        pred_seq_col = autodetect_col(list(df_pred.columns), ["sequence", "seq", "utr", "utr3", "utr_3", "rna", "dna"])
    if pred_pred_col is None:
        pred_pred_col = autodetect_col(list(df_pred.columns), ["y_pred", "pred", "prediction", "predicted", "half_life_pred", "pred_half_life"])
    if pred_true_col is None:
        pred_true_col = autodetect_col(list(df_pred.columns), ["y_true", "label", "target", "half_life", "halflife", "half-life", "t_half", "decay", "hl"])

    # Validate
    missing = []
    if dataset_seq_col is None: missing.append("dataset sequence")
    if dataset_gene_col is None: missing.append("dataset gene")
    if dataset_true_col is None and pred_true_col is None: missing.append("real half-life (dataset_true or pred_true)")
    if pred_seq_col is None: missing.append("pred sequence")
    if pred_pred_col is None: missing.append("pred predicted value")
    if missing:
        raise RuntimeError(
            "Auto-detect failed for required columns: " + ", ".join(missing) +
            f"\nDataset columns: {list(df_data.columns)}" +
            f"\nPredictions columns: {list(df_pred.columns)}"
        )

    real_from_dataset = dataset_true_col is not None

    # Matching (merge may introduce _pred/_data suffixes)
    merged, diag, gene_col_found = match_predictions(
        df_data=df_data,
        df_pred=df_pred,
        data_seq_col=dataset_seq_col,
        pred_seq_col=pred_seq_col,
        map_u_to_t=bool(config.get("map_u_to_t", True)),
    )

    # === 关键解析：合并后解析列名（处理 _pred/_data 后缀） ===
    pred_seq_col_m    = _resolve_merged_col(merged, pred_seq_col,    prefer="pred")
    dataset_seq_col_m = _resolve_merged_col(merged, dataset_seq_col, prefer="data")

    # Gene 列解析（优先用数据集一侧）
    gene_col = dataset_gene_col if dataset_gene_col in merged.columns else None
    if gene_col is None:
        gene_col = _resolve_merged_col(merged, dataset_gene_col or gene_col_found, prefer="data")
    if gene_col is None:
        gene_col = autodetect_col(list(merged.columns), ["systematic_name", "systematic", "gene", "gene_name", "orf", "name"])
    if gene_col is None:
        raise RuntimeError("Gene column not found after merge.")

    # Real 值列解析（来自数据集或预测文件）
    if real_from_dataset:
        real_col_m = _resolve_merged_col(merged, dataset_true_col, prefer="data")
    else:
        real_col_m = _resolve_merged_col(merged, pred_true_col, prefer="pred")
    if real_col_m is None:
        raise RuntimeError("Could not resolve real (true) half-life column after merge.")

    # Pred 值列解析（预测侧）
    pred_col_m = _resolve_merged_col(merged, pred_pred_col, prefer="pred")
    if pred_col_m is None:
        raise RuntimeError("Could not resolve predicted half-life column after merge.")

    # === NEW: 导出 cross_gene_median.csv，用于补充图 Sy ===
    try:
        export_cross_gene_median(merged, Path(outdir))
    except Exception as e:
        print(f"[WARN] export_cross_gene_median failed: {e}")

    # Clean per-isoform table
    df = pd.DataFrame({
        "sequence_pred": merged[pred_seq_col_m] if pred_seq_col_m in merged.columns else np.nan,
        "sequence_data": merged[dataset_seq_col_m] if (dataset_seq_col_m is not None and dataset_seq_col_m in merged.columns) else np.nan,
        "seq_norm": merged["_seq_norm"],
        "gene": merged[gene_col],
        "ios_real": merged[real_col_m],
        "ios_pre": merged[pred_col_m],
        "ambiguous_seq": merged["ambiguous_seq"].astype(bool)
    })

    df_before = len(df)
    df = df.dropna(subset=["gene", "ios_real", "ios_pre"])
    dropped_missing = df_before - len(df)

    # Per-gene medians
    ref = df.groupby("gene", dropna=False).agg(
        ref_real=("ios_real", "median"),
        ref_pre=("ios_pre", "median"),
        n_isoforms=("ios_real", "size")
    ).reset_index()
    ref["delta_ref"] = ref["ref_pre"] - ref["ref_real"]

    # Map back
    df = df.merge(ref[["gene", "ref_real", "ref_pre", "delta_ref", "n_isoforms"]], on="gene", how="left")

    # Deltas / demeaned values
    df["delta_iso"] = df["ios_pre"] - df["ios_real"]
    df["d_real"] = df["ios_real"] - df["ref_real"]
    df["d_pre"]  = df["ios_pre"]  - df["ref_pre"]

    # Signs
    df["sign_delta_ref"] = np.sign(df["delta_ref"])
    df["sign_delta_iso"] = np.sign(df["delta_iso"])
    df["sign_d_real"] = np.sign(df["d_real"])
    df["sign_d_pre"]  = np.sign(df["d_pre"])

    # Metrics
    metrics = {}
    overall = safe_corr(df["ios_pre"].values, df["ios_real"].values)
    metrics["overall"] = overall.__dict__

    cross_gene = safe_corr(ref["ref_pre"].values, ref["ref_real"].values)
    metrics["cross_gene"] = cross_gene.__dict__

    within_pooled = safe_corr(df["d_pre"].values, df["d_real"].values)
    metrics["within_gene_pooled"] = within_pooled.__dict__

    bias_vs_iso = safe_corr(df["delta_ref"].values, df["delta_iso"].values)
    metrics["bias_vs_iso"] = bias_vs_iso.__dict__

    # Sign-based
    metrics["sign_overall"] = sign_stats(df["ios_pre"].values, df["ios_real"].values).__dict__
    metrics["sign_cross_gene"] = sign_stats(ref["ref_pre"].values, ref["ref_real"].values).__dict__
    metrics["sign_within_gene_pooled"] = sign_stats(df["d_pre"].values, df["d_real"].values).__dict__
    metrics["sign_bias_vs_iso"] = sign_stats(df["delta_ref"].values, df["delta_iso"].values).__dict__

    # Per-gene correlations (>=2 isoforms)
    rows = []
    for g, gdf in df.groupby("gene"):
        gdf2 = gdf.dropna(subset=["d_pre", "d_real"])
        n = len(gdf2)
        if n >= 2 and (gdf2["d_pre"].std() > 0) and (gdf2["d_real"].std() > 0):
            cres = safe_corr(gdf2["d_pre"].values, gdf2["d_real"].values)
            rows.append({
                "gene": g,
                "n_isoforms": int(gdf["n_isoforms"].iloc[0]),
                "pearson_r": cres.pearson_r,
                "spearman_rho": cres.spearman_rho,
                "r2": cres.r2
            })
        else:
            rows.append({
                "gene": g,
                "n_isoforms": int(gdf["n_isoforms"].iloc[0]),
                "pearson_r": np.nan,
                "spearman_rho": np.nan,
                "r2": np.nan
            })
    per_gene_corr = pd.DataFrame(rows)

    # Macro / Fisher-z means
    valid = per_gene_corr["pearson_r"].replace([np.inf, -np.inf], np.nan).dropna()
    macro_mean = float(valid.mean()) if not valid.empty else float('nan')
    macro_median = float(valid.median()) if not valid.empty else float('nan')
    fisher_weighted = fisher_z_mean(
        rs=per_gene_corr["pearson_r"].tolist(),
        ns=per_gene_corr["n_isoforms"].tolist()
    )
    metrics["per_gene_pearson_macro_mean"] = macro_mean
    metrics["per_gene_pearson_macro_median"] = macro_median
    metrics["per_gene_pearson_fisher_weighted"] = fisher_weighted

    # Save outputs
    per_isoform_out = os.path.join(outdir, "analysis_per_isoform.csv")
    per_gene_out = os.path.join(outdir, "per_gene_summary.csv")
    metrics_out = os.path.join(outdir, "metrics_summary.json")
    log_out = os.path.join(outdir, "matching_log.txt")

    cols = [
        "gene", "sequence_pred", "sequence_data", "seq_norm",
        "ios_real", "ios_pre",
        "ref_real", "ref_pre", "delta_ref",
        "d_real", "d_pre", "delta_iso",
        "sign_delta_ref", "sign_delta_iso", "sign_d_real", "sign_d_pre",
        "n_isoforms", "ambiguous_seq"
    ]
    df[cols].to_csv(per_isoform_out, index=False, encoding="utf-8")
    ref.merge(per_gene_corr, on="gene", how="left").to_csv(per_gene_out, index=False, encoding="utf-8")

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(log_out, "w", encoding="utf-8") as f:
        print("=== Matching & Data Quality ===", file=f)
        for k, v in diag.items():
            print(f"{k}: {v}", file=f)
        print(f"dropped_missing_after_merge: {dropped_missing}", file=f)
        print("Resolved columns after merge:", file=f)
        print(f"  pred_seq_col -> {pred_seq_col} -> {pred_seq_col_m}", file=f)
        print(f"  dataset_seq_col -> {dataset_seq_col} -> {dataset_seq_col_m}", file=f)
        print(f"  gene_col -> {gene_col}", file=f)
        print(f"  real_col -> {real_col_m}", file=f)
        print(f"  pred_col -> {pred_col_m}", file=f)
        print("Note: If a sequence maps to multiple genes in dataset, we kept the first occurrence "
              "and flagged 'ambiguous_seq=True'. Consider filtering these rows if needed.", file=f)

    # Optional plots
    if config.get("plots", False):
        try:
            import matplotlib.pyplot as plt

            fig1 = plt.figure()
            plt.scatter(ref["ref_real"], ref["ref_pre"], s=6, alpha=0.7)
            plt.xlabel("Gene median REAL (ref_real)")
            plt.ylabel("Gene median PRED (ref_pre)")
            plt.title("Cross-gene median scatter")
            fig1.savefig(os.path.join(outdir, "scatter_cross_gene.png"), dpi=400, bbox_inches="tight")
            fig1.savefig(os.path.join(outdir, "scatter_cross_gene.svg"), bbox_inches="tight")
            plt.close(fig1)

            fig2 = plt.figure()
            plt.scatter(df["d_real"], df["d_pre"], s=3, alpha=0.5)
            plt.xlabel("d_real = ios_real - ref_real")
            plt.ylabel("d_pre  = ios_pre  - ref_pre")
            plt.title("Within-gene (demeaned) scatter")
            fig2.savefig(os.path.join(outdir, "scatter_within_gene.png"), dpi=400, bbox_inches="tight")
            fig2.savefig(os.path.join(outdir, "scatter_within_gene.svg"), bbox_inches="tight")
            plt.close(fig2)

            fig3 = plt.figure()
            plt.scatter(df["delta_ref"], df["delta_iso"], s=3, alpha=0.5)
            plt.xlabel("Δ_ref = ref_pre - ref_real (gene bias)")
            plt.ylabel("Δ_iso = ios_pre - ios_real (isoform error)")
            plt.title("Gene bias vs Isoform error")
            fig3.savefig(os.path.join(outdir, "scatter_bias_vs_iso.png"), dpi=400, bbox_inches="tight")
            fig3.savefig(os.path.join(outdir, "scatter_bias_vs_iso.svg"), bbox_inches="tight")
            plt.close(fig3)

        except Exception as e:
            with open(log_out, "a", encoding="utf-8") as f:
                print(f"[WARN] Plotting failed: {e}", file=f)

    return outdir



if __name__ == "__main__":
    outdir = run_analysis(CONFIG)
    print(f"Done. Outputs written to: {outdir}")
