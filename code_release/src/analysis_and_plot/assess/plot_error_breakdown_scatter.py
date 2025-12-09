# plots/plot_error_breakdown_scatter.py
# -*- coding: utf-8 -*-
"""
误差分解（长度/GC 相关）可视化：
  1) y_pred vs 3'UTR length（散点 + 分位平滑 + 95%CI）
  2) residual vs 3'UTR length（散点 + 分位平滑 + 95%CI）
  3) y_pred vs GC fraction（散点 + 分位平滑 + 95%CI）
  4) residual vs GC fraction（散点 + 分位平滑 + 95%CI）
  5) MAE by length（等宽 & 分位）柱状
  6) MAE by GC（等宽 & 分位）柱状
输入：final_test_predictions.csv（含 prediction / truth；最好含 sequence 列以计算 length/GC）
输出：<项目根>/result/plot/error_breakdown_scatter/<时间戳>/  （PNG+SVG, dpi=400 + CSV）
"""

import os, re, math, json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import scienceplots

plt.style.use('science')

# ========== 在此手动填写（不使用命令行）==========
RUN_DIR = r"F:/mRNA_Project/3UTR/Paper/result/5f_full_head_v3_20251024_01"   # ← 改成你的完整训练输出目录
INPUT_FILE = "final_test_predictions.csv"    # 如有不同文件名可改
DPI = 400

# 散点与平滑
POINT_SIZE = 8
POINT_ALPHA = 0.25
SMOOTH_QUANTILES = np.linspace(0.05, 0.95, 19)  # 分位平滑的节点（可改稠密或稀疏）
N_BOOT = 1000                                   # 平滑均值的 bootstrap 次数（95%CI）

# 分箱参数
LEN_BINS_EQUALWIDTH = 8     # 长度等宽箱数
LEN_BINS_QUANTILES  = 8     # 长度分位箱数
GC_BINS_EQUALWIDTH  = 8     # GC 等宽箱数
GC_BINS_QUANTILES   = 8     # GC 分位箱数
# =================================================

# ---- 全局字体：非斜体 ----
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
    outdir = _project_root() / "result" / "plot" / "error_breakdown_scatter" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _save_dual(fig, out_base: Path, dpi: int):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _auto_pick_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """自动识别真值列与预测列；失败则抛错。"""
    cand_y_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_y_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    y_true_col = None; y_pred_col = None
    # 直接命中
    for c in df.columns:
        if c.lower() in cand_y_true: y_true_col = c; break
    for c in df.columns:
        if c.lower() in cand_y_pred: y_pred_col = c; break
    # 包含关系兜底
    if y_true_col is None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["true","target","label","ground","half"]):
                y_true_col = c; break
    if y_pred_col is None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["pred","hat","predict"]):
                y_pred_col = c; break
    if y_true_col is None or y_pred_col is None:
        raise ValueError(f"无法自动识别真值/预测列，请检查列名：{list(df.columns)}")
    return y_true_col, y_pred_col

def _seq_len_gc(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """从 sequence 计算长度与 GC 比例（忽略非 ACGTN 的字符）。"""
    lens = []
    gcs  = []
    for s in series.astype(str).tolist():
        s2 = re.sub(r"[^ACGTNacgtn]", "", s)
        L = len(s2)
        lens.append(L)
        if L == 0:
            gcs.append(np.nan)
        else:
            su = s2.upper()
            gc = su.count("G") + su.count("C")
            at = su.count("A") + su.count("T")
            total = gc + at
            gcs.append(gc / total if total > 0 else np.nan)
    return np.array(lens, dtype=int), np.array(gcs, dtype=float)

def _ensure_len_gc(df: pd.DataFrame) -> pd.DataFrame:
    """优先从 sequence 计算；若无 sequence，则尝试已有 length/gc 列；都无则报错。"""
    out = df.copy()
    have_seq = "sequence" in out.columns
    if have_seq:
        lens, gcs = _seq_len_gc(out["sequence"])
        out["utr_len"] = lens
        out["gc_frac"] = gcs
    else:
        # 兜底列名
        len_col = None; gc_col = None
        for c in out.columns:
            cl = c.lower()
            if len_col is None and ("len" in cl or "length" in cl):
                len_col = c
            if gc_col is None and ("gc" in cl and "frac" in cl) or cl in ("gc","gc_content","gc_fraction"):
                gc_col = c
        if len_col is None or gc_col is None:
            raise ValueError("未找到 sequence 列，且缺少可用的长度/GC 列。请提供 sequence 或 length/gc 列。")
        out["utr_len"] = out[len_col].astype(int).values
        out["gc_frac"] = out[gc_col].astype(float).values
    return out

def _summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2))
    pr = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    sr = float(stats.spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pr, "Spearman": sr}

def _quantile_smooth(x: np.ndarray, y: np.ndarray, qs: np.ndarray, n_boot=1000, seed=20251016):
    """按 x 的分位点做分箱平滑，输出 (x_mid, mean_y, ci_lo, ci_hi)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.quantile(x, qs)
    means, lo, hi = [], [], []
    idxs = []
    for j in range(len(qs)-1):
        lo_q, hi_q = xq[j], xq[j+1]
        sel = (x >= lo_q) & (x <= hi_q) if j == 0 else (x > lo_q) & (x <= hi_q)
        idxs.append(np.where(sel)[0])
        vals = y[sel]
        if vals.size == 0:
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        m = float(np.mean(vals)); means.append(m)
        # bootstrap 95%CI
        ridx = np.arange(vals.size)
        rng = np.random.RandomState(seed)
        boots = [float(np.mean(vals[rng.choice(ridx, size=ridx.size, replace=True)])) for _ in range(n_boot)]
        lo.append(float(np.quantile(boots, 0.025)))
        hi.append(float(np.quantile(boots, 0.975)))
    x_mid = 0.5*(xq[:-1] + xq[1:])
    return x_mid, np.array(means), np.array(lo), np.array(hi)

def _bar_with_ci(labels: List[str], vals: np.ndarray, title: str, ylabel: str, out_base: Path):
    import re
    # 1) 压缩标签："[100,200]" -> "100–200"；"[0.12,0.25]" -> "0.12–0.25"
    def _shorten(l: str) -> str:
        s = str(l)
        s = re.sub(r"[\[\]\s]", "", s)   # 去括号和空格
        s = s.replace(",", "–")          # 用 en dash 连接
        return s

    short = [_shorten(l) for l in labels]

    # 2) 画布更宽一些，给旋转刻度留空间
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    x = np.arange(len(short))
    ax.bar(x, vals)

    # 3) 自动抽稀：标签多时只显示部分刻度（最多 ~14 个）
    max_ticks = 14
    if len(short) > max_ticks:
        step = int(np.ceil(len(short) / max_ticks))
        tick_idx = np.arange(len(short))[::step]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([short[i] for i in tick_idx], rotation=30, ha="right", fontsize=9)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)

    # 4) 额外留白，避免左右贴边
    ax.margins(x=0.02)
    ax.tick_params(axis="x", pad=6)
    fig.subplots_adjust(bottom=0.26)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    _save_dual(fig, out_base, DPI)


def _mae_by_bins(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, edges: np.ndarray, labels: List[str]) -> np.ndarray:
    mae_vals = []
    for j in range(len(edges)-1):
        lo_e, hi_e = edges[j], edges[j+1]
        sel = (x >= lo_e) & (x <= hi_e) if j == 0 else (x > lo_e) & (x <= hi_e)
        if np.any(sel):
            mae_vals.append(float(np.mean(np.abs(y_pred[sel] - y_true[sel]))))
        else:
            mae_vals.append(np.nan)
    return np.array(mae_vals, dtype=float)

def main():
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    fp = Path(RUN_DIR) / INPUT_FILE
    if not fp.is_file():
        raise FileNotFoundError(f"未找到输入文件：{fp}")

    df = pd.read_csv(fp).dropna(how="all").copy()
    y_true_col, y_pred_col = _auto_pick_columns(df)
    print(f"[列识别] y_true={y_true_col} | y_pred={y_pred_col}")

    df = _ensure_len_gc(df)
    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values
    resid  = y_pred - y_true
    utr_len = df["utr_len"].astype(float).values
    gc_frac = df["gc_frac"].astype(float).values

    # 概览指标
    summ = _summary_metrics(y_true, y_pred)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summ.items():
            f.write(f"{k}: {v:.6g}\n")
    pd.DataFrame({"metric": list(summ.keys()), "value": list(summ.values())}).to_csv(outdir / "summary.csv", index=False)

    # ===== 1) y_pred vs length（平滑） =====
    xm, ym, ylo, yhi = _quantile_smooth(utr_len, y_pred, SMOOTH_QUANTILES, n_boot=N_BOOT)
    pd.DataFrame({"x_mid_len": xm, "mean_pred": ym, "ci_lo": ylo, "ci_hi": yhi}).to_csv(outdir / "pred_vs_len_smooth.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6,4.6))
    ax.scatter(utr_len, y_pred, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0)
    ax.plot(xm, ym, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(xm, ylo, yhi, alpha=0.25)
    ax.set_xlabel("3'UTR length (nt)")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction vs 3'UTR length")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "pred_vs_len", DPI)

    # ===== 2) residual vs length（平滑） =====
    xm2, ym2, ylo2, yhi2 = _quantile_smooth(utr_len, resid, SMOOTH_QUANTILES, n_boot=N_BOOT)
    pd.DataFrame({"x_mid_len": xm2, "mean_resid": ym2, "ci_lo": ylo2, "ci_hi": yhi2}).to_csv(outdir / "resid_vs_len_smooth.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6,4.6))
    ax.scatter(utr_len, resid, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0)
    ax.plot(xm2, ym2, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(xm2, ylo2, yhi2, alpha=0.25)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("3'UTR length (nt)")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residual vs 3'UTR length")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "resid_vs_len", DPI)

    # ===== 3) y_pred vs GC（平滑） =====
    xm3, ym3, ylo3, yhi3 = _quantile_smooth(gc_frac, y_pred, SMOOTH_QUANTILES, n_boot=N_BOOT)
    pd.DataFrame({"x_mid_gc": xm3, "mean_pred": ym3, "ci_lo": ylo3, "ci_hi": yhi3}).to_csv(outdir / "pred_vs_gc_smooth.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6,4.6))
    ax.scatter(gc_frac, y_pred, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0)
    ax.plot(xm3, ym3, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(xm3, ylo3, yhi3, alpha=0.25)
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction vs GC fraction")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "pred_vs_gc", DPI)

    # ===== 4) residual vs GC（平滑） =====
    xm4, ym4, ylo4, yhi4 = _quantile_smooth(gc_frac, resid, SMOOTH_QUANTILES, n_boot=N_BOOT)
    pd.DataFrame({"x_mid_gc": xm4, "mean_resid": ym4, "ci_lo": ylo4, "ci_hi": yhi4}).to_csv(outdir / "resid_vs_gc_smooth.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6,4.6))
    ax.scatter(gc_frac, resid, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0)
    ax.plot(xm4, ym4, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(xm4, ylo4, yhi4, alpha=0.25)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residual vs GC fraction")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "resid_vs_gc", DPI)

    # ===== 5) MAE by length：等宽 & 分位 =====
    # 等宽
    len_min, len_max = float(np.nanmin(utr_len)), float(np.nanmax(utr_len))
    len_edges_w = np.linspace(len_min, len_max, LEN_BINS_EQUALWIDTH+1)
    len_labels_w = [f"[{int(len_edges_w[j])},{int(len_edges_w[j+1])}]" for j in range(LEN_BINS_EQUALWIDTH)]
    mae_w = _mae_by_bins(utr_len, y_true, y_pred, len_edges_w, len_labels_w)
    pd.DataFrame({"length_bin": len_labels_w, "mae": mae_w}).to_csv(outdir / "mae_by_len_equalwidth.csv", index=False)
    _bar_with_ci(len_labels_w, mae_w, "MAE by 3'UTR length (equal-width)", "MAE", outdir / "mae_by_len_equalwidth")

    # 分位
    qs_len = np.linspace(0.0, 1.0, LEN_BINS_QUANTILES+1)
    len_edges_q = np.quantile(utr_len, qs_len)
    len_edges_q[0] = len_min; len_edges_q[-1] = len_max  # 扩到端点
    len_labels_q = [f"[{int(len_edges_q[j])},{int(len_edges_q[j+1])}]" for j in range(LEN_BINS_QUANTILES)]
    mae_q = _mae_by_bins(utr_len, y_true, y_pred, len_edges_q, len_labels_q)
    pd.DataFrame({"length_bin": len_labels_q, "mae": mae_q}).to_csv(outdir / "mae_by_len_quantile.csv", index=False)
    _bar_with_ci(len_labels_q, mae_q, "MAE by 3'UTR length (quantile)", "MAE", outdir / "mae_by_len_quantile")

    # ===== 6) MAE by GC：等宽 & 分位 =====
    # 等宽
    gc_min, gc_max = float(np.nanmin(gc_frac)), float(np.nanmax(gc_frac))
    gc_edges_w = np.linspace(gc_min, gc_max, GC_BINS_EQUALWIDTH+1)
    gc_labels_w = [f"[{gc_edges_w[j]:.2f},{gc_edges_w[j+1]:.2f}]" for j in range(GC_BINS_EQUALWIDTH)]
    mae_gw = _mae_by_bins(gc_frac, y_true, y_pred, gc_edges_w, gc_labels_w)
    pd.DataFrame({"gc_bin": gc_labels_w, "mae": mae_gw}).to_csv(outdir / "mae_by_gc_equalwidth.csv", index=False)
    _bar_with_ci(gc_labels_w, mae_gw, "MAE by GC fraction (equal-width)", "MAE", outdir / "mae_by_gc_equalwidth")

    # 分位
    qs_gc = np.linspace(0.0, 1.0, GC_BINS_QUANTILES+1)
    gc_edges_q = np.quantile(gc_frac, qs_gc)
    gc_edges_q[0] = gc_min; gc_edges_q[-1] = gc_max
    gc_labels_q = [f"[{gc_edges_q[j]:.2f},{gc_edges_q[j+1]:.2f}]" for j in range(GC_BINS_QUANTILES)]
    mae_gq = _mae_by_bins(gc_frac, y_true, y_pred, gc_edges_q, gc_labels_q)
    pd.DataFrame({"gc_bin": gc_labels_q, "mae": mae_gq}).to_csv(outdir / "mae_by_gc_quantile.csv", index=False)
    _bar_with_ci(gc_labels_q, mae_gq, "MAE by GC fraction (quantile)", "MAE", outdir / "mae_by_gc_quantile")

    # 保存一次配置快照
    with open(outdir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({
            "RUN_DIR": RUN_DIR,
            "INPUT_FILE": INPUT_FILE,
            "SMOOTH_QUANTILES": SMOOTH_QUANTILES.tolist(),
            "N_BOOT": N_BOOT,
            "LEN_BINS_EQUALWIDTH": LEN_BINS_EQUALWIDTH,
            "LEN_BINS_QUANTILES": LEN_BINS_QUANTILES,
            "GC_BINS_EQUALWIDTH": GC_BINS_EQUALWIDTH,
            "GC_BINS_QUANTILES": GC_BINS_QUANTILES,
        }, f, ensure_ascii=False, indent=2)

    print("[完成] 图表与 CSV 输出至：", outdir)

if __name__ == "__main__":
    main()
