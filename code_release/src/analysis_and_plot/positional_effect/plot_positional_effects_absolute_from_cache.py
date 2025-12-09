# plots/plot_positional_effects_absolute_from_cache.py
# -*- coding: utf-8 -*-
"""
从已有缓存聚合绘制“绝对效应（Δ = baseline − occluded）”：
  1) 位置曲线（均值 ± 95%CI）
  2) 区段均值条形图（含95%CI与显著性）
  3) Top-K 热图
  4) （可选）长度分层曲线
仅读取缓存（CPU 即可，不占 GPU），输出到：
  result/plot/positional_effects_abs/<timestamp>/   （PNG+SVG, dpi=400 + CSV）
缓存来源：你之前跑的“带缓存”脚本（relative 版本），每条样本在 occl/seq_xxxxxx.csv 里已包含 delta 列。
"""

import os, json, math, re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial","DejaVu Sans",  "Liberation Sans", "Noto Sans CJK SC"],
    "font.style": "normal",
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    # 字号相关
    "font.size":17,
    "axes.titlesize":20,
    "axes.labelsize":19,
    "xtick.labelsize":15,
    "ytick.labelsize":15,
    "legend.fontsize":16,
    "figure.titlesize":17,
    # "axes.titleweight": "bold",  # 图标题
    # "axes.labelweight": "bold",  # x / y 轴标签
})
# ========== 在此手动填写 ==========
CONFIG = {
    # 指向你之前“带缓存”脚本生成的目录： result/cache/positional_effects_rel/<RUN_NAME>/
    "CACHE_DIR": r"F:\mRNA_Project\3UTR\Paper\plots\result\cache\positional_effects_rel\W15_S5_full",

    # 归一化位置分箱数（应与计算阶段一致）
    "NORM_POS_BINS": 50,

    # 图像输出
    "DPI": 400,
    "FIGSIZE_CURVE": (6.4, 4.6),
    "FIGSIZE_HEATMAP": (8.0, 4.8),
    "GRID_ALPHA": 0.35,

    # Top-K 热图的 K
    "TOP_HEATMAP_K": 12,

    # （可选）长度分层
    "DO_LENGTH_STRATA": False,
    "LENGTH_BINS_ABS": [0, 500, 1000, 10**9],   # 或者设为 None 并使用分位
    "LENGTH_QUANTILES": [0.0, 0.33, 0.66, 1.0],
}
# =================================

# ---- 统一字体：非斜体 ----
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans CJK SC"],
    "font.style": "normal",
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
})

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_outdir() -> Path:
    outdir = _project_root() / "result" / "plot" / "positional_effects_abs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _save_dual(fig, out_base: Path, dpi: int):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _bootstrap_ci_mean(y: np.ndarray, n_boot=1000, alpha=0.05, seed=20251016) -> Tuple[float, float]:
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

def _bar_with_ci_and_sig(labels: List[str], means: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         title: str, ylabel: str, out_base: Path, dpi: int):
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_CURVE"])
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[means - lo, hi - means], capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    # CI 不含 0 → 加星
    for i, (m, l, h) in enumerate(zip(means, lo, hi)):
        if not (np.isnan(l) or np.isnan(h)) and (l > 0 or h < 0):
            ax.text(i, max(0, m) + (h - m) * 0.15 + 0.02, "*", ha="center", va="bottom", fontsize=14)
    _save_dual(fig, out_base, dpi)

def main():
    cache_dir = Path(CONFIG["CACHE_DIR"])
    occl_dir = cache_dir / "occl"
    meta_fp = cache_dir / "meta.json"
    base_fp = cache_dir / "baseline.csv"
    assert meta_fp.is_file(), f"缺少 meta.json：{meta_fp}"
    assert base_fp.is_file(), f"缺少 baseline.csv：{base_fp}"
    assert occl_dir.is_dir(), f"缺少 occl 目录：{occl_dir}"

    meta = json.loads(meta_fp.read_text(encoding="utf-8"))
    W, S = int(meta["WINDOW_SIZE"]), int(meta["WINDOW_STEP"])
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    dfb = pd.read_csv(base_fp)
    lengths = dfb["length"].values.astype(int)
    N = len(dfb)
    print(f"[baseline] n={N}")

    # 聚合：把每个样本的 delta 归入位置 bin；同时构建热图矩阵（插值到固定 bins）
    BINS = int(CONFIG["NORM_POS_BINS"])
    bin_delta: List[List[float]] = [[] for _ in range(BINS)]
    rows_interp = []

    for i in range(N):
        p = occl_dir / f"seq_{i:06d}.csv"
        if not p.exists():
            continue
        dfo = pd.read_csv(p)
        if dfo.shape[0] == 0:
            continue
        centers = dfo["center"].values.astype(float)
        delta = dfo["delta"].values.astype(float)  # ← 绝对效应
        for d, c in zip(delta, centers):
            if not np.isnan(d):
                bin_delta[_bin_index(float(c), BINS)].append(float(d))
        rows_interp.append(_interp_to_bins(centers, np.nan_to_num(delta, nan=0.0), BINS))
    mat = np.vstack(rows_interp) if len(rows_interp) else np.zeros((0, BINS), dtype=float)

    x_centers = (np.arange(BINS) + 0.5) / BINS

    # === 1) 位置曲线（Δ） ===
    mean_y, lo_ci, hi_ci = [], [], []
    for b in range(BINS):
        arr = np.array(bin_delta[b], dtype=float)
        m = float(np.nanmean(arr)) if arr.size else np.nan
        mean_y.append(m)
        l, h = _bootstrap_ci_mean(arr, n_boot=1000, alpha=0.05, seed=20251016) if arr.size else (np.nan, np.nan)
        lo_ci.append(l); hi_ci.append(h)

    df_curve = pd.DataFrame({"center": x_centers, "mean": mean_y, "ci_lo": lo_ci, "ci_hi": hi_ci})
    df_curve.to_csv(outdir / "positional_effect_curve_absolute.csv", index=False)
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_CURVE"])
    ax.plot(x_centers, mean_y, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(x_centers, lo_ci, hi_ci, alpha=0.25)
    ax.set_xlabel("Normalized position ($5'→ 3'$)")
    ax.set_ylabel("Absolute effect $\Delta$")
    # ax.set_title(f"Absolute positional effect (W={W}, step={S}, n={N})")
    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    _save_dual(fig, outdir / "positional_effect_curve_absolute", CONFIG["DPI"])

    # === 2) 区段均值条形图（0–0.2 … 0.8–1.0） ===
    seg_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    seg_labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    seg_vals = [[] for _ in range(len(seg_labels))]
    for b in range(BINS):
        c = x_centers[b]
        idx = np.searchsorted(seg_edges, c, side="right") - 1
        idx = max(0, min(idx, len(seg_labels)-1))
        seg_vals[idx].extend(bin_delta[b])
    seg_means, seg_lo, seg_hi = [], [], []
    for vals in seg_vals:
        arr = np.array(vals, dtype=float)
        seg_means.append(float(np.nanmean(arr)) if arr.size else np.nan)
        l, h = _bootstrap_ci_mean(arr, n_boot=1000, alpha=0.05, seed=20251016) if arr.size else (np.nan, np.nan)
        seg_lo.append(l); seg_hi.append(h)

    pd.DataFrame({"segment": seg_labels, "mean": seg_means, "ci_lo": seg_lo, "ci_hi": seg_hi}).to_csv(
        outdir / "positional_effect_absolute_segments.csv", index=False
    )
    _bar_with_ci_and_sig(
        labels=seg_labels,
        means=np.array(seg_means, dtype=float),
        lo=np.array(seg_lo, dtype=float),
        hi=np.array(seg_hi, dtype=float),
        title="Absolute effect by region",
        ylabel="$\Delta$",
        out_base=outdir / "positional_effect_absolute_segments",
        dpi=CONFIG["DPI"]
    )

    # === 3) Top-K 热图（Δ） ===
    if mat.size:
        row_score = np.nanmean(mat, axis=1)
        order = np.argsort(-row_score)
        K = min(CONFIG["TOP_HEATMAP_K"], len(order))
        mat_top = mat[order[:K], :]
        pd.DataFrame(mat_top, columns=[f"bin_{i}" for i in range(BINS)]).to_csv(
            outdir / "positional_effect_heatmap_absolute_topK.csv", index=False
        )
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_HEATMAP"])
        im = ax.imshow(mat_top, aspect="auto", origin="lower")
        ax.set_xlabel("Normalized position bins ($5'→ 3'$)")
        ax.set_ylabel("Top-K sequences")
        # ax.set_title(f"Absolute positional effect — heatmap of top {K} sequences")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("$\Delta$")
        _save_dual(fig, outdir / "positional_effect_heatmap_absolute_topK", CONFIG["DPI"])

    # === 4) （可选）长度分层（Δ） ===
    if CONFIG.get("DO_LENGTH_STRATA", False) and mat.size:
        # 准备阈值
        bins_abs = CONFIG.get("LENGTH_BINS_ABS", None)
        if bins_abs is None:
            qs = CONFIG.get("LENGTH_QUANTILES", [0.0, 0.33, 0.66, 1.0])
            edges = np.quantile(lengths, qs)
        else:
            edges = np.array(bins_abs, dtype=float)
        edges = np.unique(edges)
        labels = [f"[{int(edges[j])},{int(edges[j+1])})" for j in range(len(edges)-1)]

        # 用插值矩阵 mat（每行一个样本、BINS 列）在“样本维度”上做均值与 bootstrap CI
        x_centers = (np.arange(BINS) + 0.5) / BINS
        for j in range(len(edges)-1):
            lo_e, hi_e = edges[j], edges[j+1]
            mask = (lengths >= lo_e) & (lengths < hi_e)
            if not np.any(mask):
                continue
            sub = mat[mask, :]
            mean_col = np.nanmean(sub, axis=0)
            lo_ci, hi_ci = [], []
            r = np.random.RandomState(20251016)
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

            stem = f"positional_effect_curve_absolute_lenbin{j+1}"
            df = pd.DataFrame({"center": x_centers, "mean": mean_col, "ci_lo": lo_ci, "ci_hi": hi_ci})
            df.to_csv(outdir / f"{stem}.csv", index=False)

            fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_CURVE"])
            ax.plot(x_centers, mean_col, linewidth=1.8, marker="o", markersize=3)
            ax.fill_between(x_centers, lo_ci, hi_ci, alpha=0.25)
            ax.set_xlabel("Normalized position (5'→3')")
            ax.set_ylabel("$\Delta$Δ")
            # ax.set_title(f"Absolute positional effect by length {labels[j]} (n={int(mask.sum())})")
            ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
            _save_dual(fig, outdir / stem, CONFIG["DPI"])

    print("[完成] 绝对效应（Δ）图/表输出至：", outdir)

if __name__ == "__main__":
    main()
