# plots/plot_error_2d_heatmap.py
# -*- coding: utf-8 -*-
"""
误差二维热图：长度 × GC
生成：
  - 等宽分箱：MAE / Bias / Count 热图 + CSV
  - 分位分箱：MAE / Bias / Count 热图 + CSV
输出：result/plot/error_2d_heatmap/<时间戳>/   （PNG+SVG, dpi=400）
"""

import re, math, json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
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

# ========= 在此手动填写（不走命令行）=========
RUN_DIR    = r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01"     # ← 改为你的完整训练输出目录
INPUT_FILE = "final_test_predictions.csv"         # ← 若文件名不同请修改
DPI        = 400

# 分箱设置（等宽 & 分位）
LEN_BINS_EQUALWIDTH = 10
GC_BINS_EQUALWIDTH  = 10
LEN_BINS_QUANTILE   = 10
GC_BINS_QUANTILE    = 10

# 可视化样式
CMAP_MAE  = "viridis"
CMAP_BIAS = "coolwarm"   # 发散色带
GRID_ALPHA = 0.35

# 轴刻度最多显示多少个tick，避免重叠
MAX_TICKS_PER_AXIS = 10
# ============================================

# 全局非斜体字体
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
    outdir = _project_root() / "result" / "plot" / "error_2d_heatmap" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _save_dual(fig, out_base: Path, dpi: int):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _auto_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cand_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    yt = None; yp = None
    for c in df.columns:
        if c.lower() in cand_true: yt = c; break
    for c in df.columns:
        if c.lower() in cand_pred: yp = c; break
    if yt is None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["true","target","label","ground","half"]):
                yt = c; break
    if yp is None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["pred","hat","predict"]):
                yp = c; break
    if yt is None or yp is None:
        raise ValueError(f"无法识别真值/预测列：{list(df.columns)}")
    return yt, yp

def _seq_len_gc(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    lens, gcs = [], []
    for s in series.astype(str):
        s2 = re.sub(r"[^ACGTNacgtn]", "", s)
        L  = len(s2)
        lens.append(L)
        if L == 0:
            gcs.append(np.nan)
        else:
            su = s2.upper()
            gc = su.count("G") + su.count("C")
            at = su.count("A") + su.count("T")
            tot = gc + at
            gcs.append(gc / tot if tot > 0 else np.nan)
    return np.array(lens, int), np.array(gcs, float)

def _ensure_len_gc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sequence" in out.columns:
        L, G = _seq_len_gc(out["sequence"])
        out["utr_len"] = L
        out["gc_frac"] = G
        return out
    # fallback: 直接使用已有列
    len_col = None; gc_col = None
    for c in out.columns:
        cl = c.lower()
        if len_col is None and ("len" in cl or "length" in cl):
            len_col = c
        if gc_col is None and (cl in ("gc","gc_content","gc_fraction") or ("gc" in cl and "frac" in cl)):
            gc_col = c
    if len_col is None or gc_col is None:
        raise ValueError("缺少 sequence 或 length/gc 列，无法计算二维热图。")
    out["utr_len"] = out[len_col].astype(int).values
    out["gc_frac"] = out[gc_col].astype(float).values
    return out

def _digitize(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, edges, right=True) - 1  # 左闭右开，最后一格包含右端
    return np.clip(idx, 0, len(edges)-2)

def _grid_agg(len_vals: np.ndarray, gc_vals: np.ndarray, resid: np.ndarray,
              len_edges: np.ndarray, gc_edges: np.ndarray) -> pd.DataFrame:
    nL = len(len_edges) - 1
    nG = len(gc_edges) - 1
    # 预分配
    cnt  = np.zeros((nL, nG), dtype=int)
    mae  = np.full((nL, nG), np.nan, float)
    rmse = np.full((nL, nG), np.nan, float)
    bias = np.full((nL, nG), np.nan, float)

    il = _digitize(len_vals, len_edges)
    ig = _digitize(gc_vals, gc_edges)

    # 为了速度，用分组聚合
    key = il.astype(np.int64) * 10_000 + ig.astype(np.int64)
    df = pd.DataFrame({"key": key, "resid": resid})
    grp = df.groupby("key")["resid"]

    for k, vals in grp:
        li = int(k // 10_000); gi = int(k % 10_000)
        arr = vals.values.astype(float)
        cnt[li, gi]  = arr.size
        bias[li, gi] = float(np.mean(arr))
        mae[li, gi]  = float(np.mean(np.abs(arr)))
        rmse[li, gi] = float(np.sqrt(np.mean(arr**2)))

    # 展平为表，并附上边界
    rows = []
    for li in range(nL):
        for gi in range(nG):
            rows.append({
                "len_lo": len_edges[li], "len_hi": len_edges[li+1],
                "gc_lo":  gc_edges[gi], "gc_hi":  gc_edges[gi+1],
                "count":  int(cnt[li, gi]),
                "mae":    mae[li, gi],
                "rmse":   rmse[li, gi],
                "bias":   bias[li, gi],
                "len_bin": li, "gc_bin": gi,
            })
    return pd.DataFrame(rows), (cnt, mae, rmse, bias)

def _tick_from_edges(edges: np.ndarray, max_ticks: int) -> Tuple[np.ndarray, List[str]]:
    centers = 0.5*(edges[:-1] + edges[1:])
    n = centers.size
    if n <= max_ticks:
        idx = np.arange(n)
    else:
        step = math.ceil(n / max_ticks)
        idx = np.arange(0, n, step)
    labs = [f"{centers[i]:.0f}" if centers.max() > 1.5 else f"{centers[i]:.2f}" for i in idx]
    return idx, labs

def _plot_heat(mat: np.ndarray, len_edges: np.ndarray, gc_edges: np.ndarray,
               title: str, cmap: str, diverging: bool, out_base: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    # 选择色尺
    if diverging:
        vmax = np.nanmax(np.abs(mat))
        vmin = -vmax
    else:
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    im = ax.imshow(mat.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    # x/y 刻度（用 bin 索引 + 部分中心值标签，避免长区间字符串重叠）
    ix, xlabs = _tick_from_edges(len_edges, MAX_TICKS_PER_AXIS)
    iy, ylabs = _tick_from_edges(gc_edges,  MAX_TICKS_PER_AXIS)
    ax.set_xticks(ix); ax.set_xticklabels(xlabs, rotation=0)
    ax.set_yticks(iy); ax.set_yticklabels(ylabs, rotation=0)
    ax.set_xlabel("3'UTR length (nt) — bin centers")
    ax.set_ylabel("GC fraction — bin centers")
    # ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.set_ylabel("Value")
    ax.grid(False)
    _save_dual(fig, out_base, DPI)

def main():
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    fp = Path(RUN_DIR) / INPUT_FILE
    if not fp.is_file():
        raise FileNotFoundError(f"未找到输入文件：{fp}")

    df = pd.read_csv(fp).dropna(how="all").copy()
    y_true_col, y_pred_col = _auto_cols(df)
    df = _ensure_len_gc(df)

    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values
    resid  = y_pred - y_true
    L      = df["utr_len"].astype(float).values
    GC     = df["gc_frac"].astype(float).values

    # ============ 等宽分箱 =============
    len_min, len_max = float(np.nanmin(L)), float(np.nanmax(L))
    gc_min,  gc_max  = float(np.nanmin(GC)), float(np.nanmax(GC))
    len_edges_w = np.linspace(len_min, len_max, LEN_BINS_EQUALWIDTH + 1)
    gc_edges_w  = np.linspace(gc_min, gc_max,  GC_BINS_EQUALWIDTH  + 1)

    grid_w, (cnt_w, mae_w, rmse_w, bias_w) = _grid_agg(L, GC, resid, len_edges_w, gc_edges_w)
    grid_w.to_csv(outdir / "grid_equalwidth.csv", index=False)

    _plot_heat(mae_w,  len_edges_w, gc_edges_w,  "MAE heatmap (equal-width bins)", CMAP_MAE,  diverging=False, out_base=outdir / "heat_mae_equalwidth")
    _plot_heat(bias_w, len_edges_w, gc_edges_w,  "Bias heatmap (equal-width bins)", CMAP_BIAS, diverging=True,  out_base=outdir / "heat_bias_equalwidth")
    _plot_heat(cnt_w.astype(float),  len_edges_w, gc_edges_w, "Count heatmap (equal-width bins)", "Greys",    diverging=False, out_base=outdir / "heat_count_equalwidth")

    # ============ 分位分箱 =============
    qsL = np.linspace(0, 1, LEN_BINS_QUANTILE + 1)
    qsG = np.linspace(0, 1, GC_BINS_QUANTILE  + 1)
    len_edges_q = np.quantile(L, qsL); len_edges_q[0] = len_min; len_edges_q[-1] = len_max
    gc_edges_q  = np.quantile(GC, qsG); gc_edges_q[0]  = gc_min;  gc_edges_q[-1]  = gc_max

    grid_q, (cnt_q, mae_q, rmse_q, bias_q) = _grid_agg(L, GC, resid, len_edges_q, gc_edges_q)
    grid_q.to_csv(outdir / "grid_quantile.csv", index=False)

    _plot_heat(mae_q,  len_edges_q, gc_edges_q,  "MAE heatmap (quantile bins)", CMAP_MAE,  diverging=False, out_base=outdir / "heat_mae_quantile")
    _plot_heat(bias_q, len_edges_q, gc_edges_q,  "Bias heatmap (quantile bins)", CMAP_BIAS, diverging=True,  out_base=outdir / "heat_bias_quantile")
    _plot_heat(cnt_q.astype(float),  len_edges_q, gc_edges_q, "Count heatmap (quantile bins)", "Greys",    diverging=False, out_base=outdir / "heat_count_quantile")

    # 保存一次配置快照
    with open(outdir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({
            "RUN_DIR": RUN_DIR, "INPUT_FILE": INPUT_FILE,
            "LEN_BINS_EQUALWIDTH": LEN_BINS_EQUALWIDTH, "GC_BINS_EQUALWIDTH": GC_BINS_EQUALWIDTH,
            "LEN_BINS_QUANTILE": LEN_BINS_QUANTILE, "GC_BINS_QUANTILE": GC_BINS_QUANTILE,
            "CMAP_MAE": CMAP_MAE, "CMAP_BIAS": CMAP_BIAS
        }, f, ensure_ascii=False, indent=2)

    print("[完成] 输出目录：", outdir)

if __name__ == "__main__":
    main()
