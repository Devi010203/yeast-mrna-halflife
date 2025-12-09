# plots/plot_residual_diagnostics.py
# -*- coding: utf-8 -*-
"""
全量测试集 残差诊断图
生成：
  1) residual vs prediction（带分位平滑 + 95%CI）
  2) residual vs truth（带分位平滑 + 95%CI）
  3) 残差直方图（线性）
  4) QQ plot（线性残差）
  5) MAE by prediction decile（异方差检查）
  6) （可选）log1p 残差的 1–3
输出：<项目根>/result/plot/fulltrain_plot/residual_diagnostics/<时间戳>/
图片：PNG+SVG, dpi=400；同时导出对应 CSV 和一个 summary.txt
"""

import os, re, math
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
RUN_DIR = r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01"  # ← 改成你的完整训练输出目录（包含 final_test_predictions.csv）
INPUT_FILE = "final_test_predictions.csv"   # 如你有别名就改这里
DPI = 400
DO_LOG1P = True          # 是否额外生成 log1p 残差版本的 1–3
SMOOTH_QUANTILES = np.linspace(0.0, 1.0, 11)  # 分位平滑的横坐标分位点
NBINS_HIST = 60          # 直方图箱数
POINT_ALPHA = 0.25       # 散点透明度，避免遮挡
SCATTER_X_Q = (0.01, 0.99)   # 只画横坐标（预测值）在 1%–99% 分位范围内的点
SCATTER_Y_Q = (0.01, 0.99)   # 只画纵坐标（残差）在 1%–99% 分位范围内的点

# ===============================



# ---- 工具 ----
def _ensure_outdir() -> Path:
    outdir = Path(__file__).resolve().parent.parent / "result" / "plot" / "fulltrain_plot" / "residual_diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _save_dual(fig, out_base: Path, dpi: int):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _auto_pick_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """自动识别 真值列 与 预测列；如失败抛错。"""
    cols = [c.lower() for c in df.columns]
    cand_y_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_y_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    y_true_col = None
    y_pred_col = None
    for c in df.columns:
        if c.lower() in cand_y_true:
            y_true_col = c; break
    for c in df.columns:
        if c.lower() in cand_y_pred:
            y_pred_col = c; break
    # 容错：若没有简单命名，试试包含关系
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
        raise ValueError(f"无法自动识别列名，请检查：{df.columns.tolist()}")
    return y_true_col, y_pred_col

def _summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2))
    pr = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    sr = float(stats.spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pr, "Spearman": sr}

def _quantile_smooth(x: np.ndarray, y: np.ndarray, qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 x 的分位点做分箱平滑；返回 (x_q, mean_y, 95%CI)"""
    n = len(x)
    xq = np.quantile(x, qs)
    mean_y, lo, hi = [], [], []
    for j in range(len(qs)-1):
        lo_q, hi_q = xq[j], xq[j+1]
        if j == 0:
            sel = (x >= lo_q) & (x <= hi_q)
        else:
            sel = (x >  lo_q) & (x <= hi_q)
        vals = y[sel]
        if vals.size == 0:
            mean_y.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            m = float(np.mean(vals))
            mean_y.append(m)
            # bootstrap CI
            idx = np.arange(vals.size)
            boots = []
            rng = np.random.RandomState(20251016)
            for _ in range(1000):
                samp = rng.choice(idx, size=idx.size, replace=True)
                boots.append(float(np.mean(vals[samp])))
            lo.append(float(np.quantile(boots, 0.025)))
            hi.append(float(np.quantile(boots, 0.975)))
    # 以分箱中心当作 x 坐标
    x_mid = 0.5*(xq[:-1] + xq[1:])
    return x_mid, np.array(mean_y), np.array(lo), np.array(hi)

def _mae_by_decile(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Tuple[List[str], np.ndarray]:
    """按预测分位数分十等份，计算每份 MAE。"""
    qs = np.linspace(0.0, 1.0, k+1)
    edges = np.quantile(y_pred, qs)
    labels = [f"D{j+1}" for j in range(k)]
    mae_vals = []
    for j in range(k):
        lo_e, hi_e = edges[j], edges[j+1]
        sel = (y_pred >= lo_e) & (y_pred <= hi_e) if j == 0 else (y_pred > lo_e) & (y_pred <= hi_e)
        if np.any(sel):
            mae_vals.append(float(np.mean(np.abs(y_pred[sel] - y_true[sel]))))
        else:
            mae_vals.append(np.nan)
    return labels, np.array(mae_vals)

def main():
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    fp = Path(RUN_DIR) / INPUT_FILE
    if not fp.is_file():
        raise FileNotFoundError(f"未找到输入文件：{fp}")

    df = pd.read_csv(fp)
    y_true_col, y_pred_col = _auto_pick_columns(df)
    print(f"[列识别] y_true={y_true_col} | y_pred={y_pred_col}")

    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values
    resid = y_pred - y_true

    # ---- 概览指标并写入 summary.txt ----
    summ = _summary_metrics(y_true, y_pred)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summ.items():
            f.write(f"{k}: {v:.6g}\n")
    pd.DataFrame({"metric": list(summ.keys()), "value": list(summ.values())}).to_csv(outdir / "summary.csv", index=False)

    # 1) residual vs prediction（线性）
    x = y_pred.copy()
    y = resid.copy()

    # 用全部数据做分位平滑和 95%CI（统计不截断）
    x_mid, y_mean, y_lo, y_hi = _quantile_smooth(x, y, SMOOTH_QUANTILES)
    pd.DataFrame(
        {"x_mid_pred": x_mid, "mean_resid": y_mean, "ci_lo": y_lo, "ci_hi": y_hi}
    ).to_csv(outdir / "resid_vs_pred_smooth.csv", index=False)

    # 只对散点做分位截断，避免少数极端点拉伸坐标轴
    x_q_lo, x_q_hi = np.quantile(x, SCATTER_X_Q)
    y_q_lo, y_q_hi = np.quantile(y, SCATTER_Y_Q)
    mask_scatter = (x >= x_q_lo) & (x <= x_q_hi) & (y >= y_q_lo) & (y <= y_q_hi)
    x_plot = x[mask_scatter]
    y_plot = y[mask_scatter]
    n_dropped = int((~mask_scatter).sum())
    print(f"[resid_vs_pred] 截断散点 {n_dropped} 个极端点（仅影响可视化）")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(x_plot, y_plot, s=8, alpha=POINT_ALPHA, linewidths=0)
    ax.plot(x_mid, y_mean, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(x_mid, y_lo, y_hi, alpha=0.25)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Residual (pred − true)")
    # ax.set_title("Residual vs Prediction")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "residual_vs_prediction", DPI)

    # 2) residual vs truth（线性）
    x2 = y_true.copy()
    y2 = resid.copy()

    # 用全部数据做分位平滑和 95%CI
    x_mid2, y_mean2, y_lo2, y_hi2 = _quantile_smooth(x2, y2, SMOOTH_QUANTILES)
    pd.DataFrame(
        {"x_mid_true": x_mid2, "mean_resid": y_mean2, "ci_lo": y_lo2, "ci_hi": y_hi2}
    ).to_csv(outdir / "resid_vs_true_smooth.csv", index=False)

    # 只对散点做分位截断
    x2_q_lo, x2_q_hi = np.quantile(x2, SCATTER_X_Q)
    y2_q_lo, y2_q_hi = np.quantile(y2, SCATTER_Y_Q)
    mask_scatter2 = (x2 >= x2_q_lo) & (x2 <= x2_q_hi) & (y2 >= y2_q_lo) & (y2 <= y2_q_hi)
    x2_plot = x2[mask_scatter2]
    y2_plot = y2[mask_scatter2]
    n_dropped2 = int((~mask_scatter2).sum())
    print(f"[resid_vs_true] 截断散点 {n_dropped2} 个极端点（仅影响可视化）")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(x2_plot, y2_plot, s=8, alpha=POINT_ALPHA, linewidth=0)
    ax.plot(x_mid2, y_mean2, linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(x_mid2, y_lo2, y_hi2, alpha=0.25)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residual vs Truth")
    ax.grid(True, linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "residual_vs_truth", DPI)

    # 4) QQ plot（线性残差）
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("QQ plot of residuals")
    _save_dual(fig, outdir / "residual_qqplot", DPI)

    # 5) MAE by prediction decile
    labels, mae_vals = _mae_by_decile(y_true, y_pred, k=10)
    pd.DataFrame({"decile": labels, "mae": mae_vals}).to_csv(outdir / "mae_by_pred_decile.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x_idx = np.arange(len(labels))
    ax.bar(x_idx, mae_vals)
    ax.set_xticks(x_idx); ax.set_xticklabels(labels)
    ax.set_xlabel("Prediction deciles")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by prediction decile")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "mae_by_prediction_decile", DPI)

    # 6) （可选）log1p 残差版本
    if DO_LOG1P:
        y_true_l = np.log1p(y_true)
        y_pred_l = np.log1p(y_pred)
        resid_l = y_pred_l - y_true_l

        # vs prediction(log)
        xm, ym, ylo, yhi = _quantile_smooth(y_pred_l, resid_l, SMOOTH_QUANTILES)
        pd.DataFrame({"x_mid_pred_log1p": xm, "mean_resid_log1p": ym, "ci_lo": ylo, "ci_hi": yhi}).to_csv(outdir / "resid_vs_pred_log1p_smooth.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(y_pred_l, resid_l, s=8, alpha=POINT_ALPHA, linewidth=0)
        ax.plot(xm, ym, linewidth=1.8, marker="o", markersize=3)
        ax.fill_between(xm, ylo, yhi, alpha=0.25)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Prediction (log1p)")
        ax.set_ylabel("Residual (pred − true) in log1p")
        ax.set_title("Residual vs Prediction (log1p)")
        ax.grid(True, linestyle="--", alpha=0.35)
        _save_dual(fig, outdir / "residual_vs_prediction_log1p", DPI)

        # vs truth(log)
        xm2, ym2, ylo2, yhi2 = _quantile_smooth(y_true_l, resid_l, SMOOTH_QUANTILES)
        pd.DataFrame({"x_mid_true_log1p": xm2, "mean_resid_log1p": ym2, "ci_lo": ylo2, "ci_hi": yhi2}).to_csv(outdir / "resid_vs_true_log1p_smooth.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(y_true_l, resid_l, s=8, alpha=POINT_ALPHA, linewidth=0)
        ax.plot(xm2, ym2, linewidth=1.8, marker="o", markersize=3)
        ax.fill_between(xm2, ylo2, yhi2, alpha=0.25)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Truth (log1p)")
        ax.set_ylabel("Residual (pred − true) in log1p")
        ax.set_title("Residual vs Truth (log1p)")
        ax.grid(True, linestyle="--", alpha=0.35)
        _save_dual(fig, outdir / "residual_vs_truth_log1p", DPI)

        # 直方图（log1p 残差）
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.hist(resid_l, bins=NBINS_HIST)
        ax.set_xlabel("Residual (pred − true) in log1p")
        ax.set_ylabel("Count")
        ax.set_title("Residual histogram (log1p)")
        ax.grid(True, linestyle="--", alpha=0.35)
        _save_dual(fig, outdir / "residual_hist_log1p", DPI)

    print("[完成] 残差诊断图与 CSV 已输出到：", outdir)

if __name__ == "__main__":
    main()
