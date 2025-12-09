# plots/plot_bland_altman.py
# -*- coding: utf-8 -*-
"""
Bland–Altman 分析（全量测试集）
输出目录：result/plot/bland_altman/<时间戳>/
图像：PNG+SVG，dpi=400；坐标文字非斜体

生成：
  1) Bland–Altman（线性尺度）
  2) Bland–Altman（log1p 尺度）
  3) MAE by mean-decile（辅助异方差判断）
  4) summary.csv / summary.txt（偏差、SD、LoA、相关性等）
"""

import os, math
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

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

# ========= 在此手动填写（不使用命令行）=========
RUN_DIR   = r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01"   # ← 改成你的完整训练输出目录
INPUT_CSV = "final_test_predictions.csv"       # ← 改成实际文件名（需包含预测与真值）
DPI       = 400
POINT_SIZE   = 10
POINT_ALPHA  = 0.25
NBINS_DECILE = 10

# 坐标裁剪与居中显示
QTRIM_X = (0.0, 1.0)   # x 轴按分位数裁剪显示范围（去掉极端点的视觉影响）
QTRIM_Y = (0.0, 1.0)   # y 轴按分位数裁剪显示范围
Y_ZERO_CENTER = True     # 让 y=0 横线位于图像中部（对称上下界）
LIM_PAD_FRAC = 0.1      # 两端额外留白比例

# ===========================================

def _ensure_outdir() -> Path:
    outdir = Path(__file__).resolve().parent.parent / "result" / "plot" / "bland_altman" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _save_dual(fig, out_base: Path, dpi: int):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _auto_cols(df: pd.DataFrame) -> Tuple[str,str]:
    cand_y_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_y_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    yt = yp = None
    for c in df.columns:
        if c.lower() in cand_y_true: yt = c; break
    for c in df.columns:
        if c.lower() in cand_y_pred: yp = c; break
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


def _limits_from_quantiles(arr: np.ndarray, qlo: float, qhi: float, pad: float = 0.03) -> tuple[float, float]:
    """根据分位数给出显示范围，并加少量边距。"""
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    lo, hi = np.quantile(arr, [qlo, qhi])
    span = max(1e-12, hi - lo)
    return float(lo - pad*span), float(hi + pad*span)


def _loa_stats(diff: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """均值差、SD、±1.96SD；同时给出均值与LoA的近似95%CI（Bland & Altman 1986 简化）"""
    n = diff.size
    mean_d = float(np.mean(diff))
    sd_d   = float(np.std(diff, ddof=1))
    z = 1.96
    # 平均差CI
    se_mean = sd_d / math.sqrt(n)
    mean_lo = mean_d - z * se_mean
    mean_hi = mean_d + z * se_mean
    # LoA 近似CI（常见近似公式）
    se_loa  = sd_d * math.sqrt(1/n + (z**2) / (2*(n-1)))
    loa_lo  = mean_d - z*sd_d
    loa_hi  = mean_d + z*sd_d
    loa_lo_ci = (loa_lo - z*se_loa, loa_lo + z*se_loa)
    loa_hi_ci = (loa_hi - z*se_loa, loa_hi + z*se_loa)
    return {
        "bias": mean_d, "sd": sd_d,
        "loa_lo": loa_lo, "loa_hi": loa_hi,
        "bias_ci_lo": mean_lo, "bias_ci_hi": mean_hi,
        "loa_lo_ci_lo": loa_lo_ci[0], "loa_lo_ci_hi": loa_lo_ci[1],
        "loa_hi_ci_lo": loa_hi_ci[0], "loa_hi_ci_hi": loa_hi_ci[1],
    }

def _fit_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float,float,float]:
    """简单线性回归 y = a + b x，返回 a, b, p-value（检验斜率是否为0）"""
    slope, intercept, r, p, se = stats.linregress(x, y)
    return intercept, slope, p

def _decile_mae(x_mean: np.ndarray, diff: np.ndarray, k: int = 10) -> pd.DataFrame:
    qs = np.linspace(0,1,k+1)
    edges = np.quantile(x_mean, qs)
    labs = [f"D{j+1}" for j in range(k)]
    mae_vals = []
    for j in range(k):
        lo, hi = edges[j], edges[j+1]
        sel = (x_mean >= lo) & (x_mean <= hi) if j==0 else (x_mean > lo) & (x_mean <= hi)
        if np.any(sel):
            mae_vals.append(float(np.mean(np.abs(diff[sel]))))
        else:
            mae_vals.append(np.nan)
    return pd.DataFrame({"decile": labs, "mae": mae_vals})

def _plot_ba(x_mean: np.ndarray, diff: np.ndarray, title: str, out_base: Path):
    st = _loa_stats(diff)               # 统计仍用全量数据；如需连统计也裁剪，可改为 diff_mask 后再算
    a, b, p = _fit_trend(x_mean, diff)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(x_mean, diff, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0)

    # === 新增：先设置显示范围（分位数裁剪） ===
    # x 轴按分位裁剪
    xlo, xhi = _limits_from_quantiles(x_mean, QTRIM_X[0], QTRIM_X[1], pad=LIM_PAD_FRAC)
    ax.set_xlim(xlo, xhi)

    # y 轴按分位裁剪；如需 0 居中，则对称到 |max(|qlo|, |qhi|)|
    ylo_q, yhi_q = np.quantile(diff[np.isfinite(diff)], [QTRIM_Y[0], QTRIM_Y[1]])
    if Y_ZERO_CENTER:
        ymax = max(abs(ylo_q), abs(yhi_q))
        span = ymax * (1 + LIM_PAD_FRAC)
        ax.set_ylim(-span, span)
    else:
        ylo, yhi = _limits_from_quantiles(diff, QTRIM_Y[0], QTRIM_Y[1], pad=LIM_PAD_FRAC)
        ax.set_ylim(ylo, yhi)

    # 均值差 & 一致性界限（此时 xlim 已确定，阴影能正确覆盖全宽）
    ax.axhline(st["bias"], color="k", linestyle="-", linewidth=1.5, label=f"Bias = {st['bias']:.3g}")
    ax.axhline(st["loa_lo"], color="k", linestyle="--", linewidth=1.0, label=f"LoA = {st['bias']:.3g} ± 1.96·SD")
    ax.axhline(st["loa_hi"], color="k", linestyle="--", linewidth=1.0)

    # 给出均值差与 LoA 的近似 CI（浅色带）
    xl0, xl1 = ax.get_xlim()
    ax.fill_between([xl0, xl1], st["bias_ci_lo"], st["bias_ci_hi"], color="gray", alpha=0.15, step="pre", label="Bias 95% CI")
    ax.fill_between([xl0, xl1], st["loa_lo_ci_lo"], st["loa_lo_ci_hi"], color="gray", alpha=0.08, step="pre")
    ax.fill_between([xl0, xl1], st["loa_hi_ci_lo"], st["loa_hi_ci_hi"], color="gray", alpha=0.08, step="pre")

    # 趋势拟合（检验比例性偏差）
    ax.plot([xl0, xl1], [a + b*xl0, a + b*xl1], color="C1", linewidth=1.6,
            label=f"Trend: diff = {a:.2g} + {b:.2g}·mean (p={p:.2g})")

    ax.set_xlabel("Mean of prediction and truth")
    ax.set_ylabel("Difference (prediction − truth)")
    # ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right")
    _save_dual(fig, out_base, DPI)

    return st, (a, b, p)


def main():
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    fp = Path(RUN_DIR) / INPUT_CSV
    if not fp.is_file():
        raise FileNotFoundError(f"未找到输入：{fp}")

    df = pd.read_csv(fp).dropna(how="all").copy()
    y_true_col, y_pred_col = _auto_cols(df)
    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values

    # 线性尺度
    mean_lin = 0.5*(y_true + y_pred)
    diff_lin = y_pred - y_true
    st_lin, trend_lin = _plot_ba(mean_lin, diff_lin, "Bland–Altman (linear scale)", outdir / "bland_altman_linear")


    # log1p 尺度（看比例性偏差更稳）
    y_true_l = np.log1p(y_true)
    y_pred_l = np.log1p(y_pred)
    mean_log = 0.5*(y_true_l + y_pred_l)
    diff_log = y_pred_l - y_true_l
    st_log, trend_log = _plot_ba(mean_log, diff_log, "Bland–Altman (log1p scale)", outdir / "bland_altman_log1p")

    # MAE by mean-decile（线性）
    dec = _decile_mae(mean_lin, diff_lin, k=NBINS_DECILE)
    dec.to_csv(outdir / "mae_by_mean_decile.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(dec))
    ax.bar(x, dec["mae"].values)
    ax.set_xticks(x); ax.set_xticklabels(dec["decile"].tolist(), rotation=0)
    ax.set_xlabel("Mean deciles")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by mean (linear scale)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    _save_dual(fig, outdir / "mae_by_mean_decile", DPI)

    # 汇总指标
    def _metrics(y_t, y_p):
        r = {}
        resid = y_p - y_t
        r["MAE"] = float(np.mean(np.abs(resid)))
        r["RMSE"] = float(np.sqrt(np.mean(resid**2)))
        r["R2"] = float(1.0 - np.sum(resid**2) / np.sum((y_t - np.mean(y_t))**2))
        r["Pearson"] = float(stats.pearsonr(y_t, y_p)[0]) if len(y_t) > 1 else np.nan
        r["Spearman"] = float(stats.spearmanr(y_t, y_p)[0]) if len(y_t) > 1 else np.nan
        return r

    summ = _metrics(y_true, y_pred)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summ.items():
            f.write(f"{k}: {v:.6g}\n")
        f.write("\n[Linear BA]\n")
        for k, v in st_lin.items(): f.write(f"{k}: {v:.6g}\n")
        f.write(f"trend_a: {trend_lin[0]:.6g}, trend_b: {trend_lin[1]:.6g}, trend_p: {trend_lin[2]:.6g}\n")
        f.write("\n[log1p BA]\n")
        for k, v in st_log.items(): f.write(f"{k}: {v:.6g}\n")
        f.write(f"trend_a: {trend_log[0]:.6g}, trend_b: {trend_log[1]:.6g}, trend_p: {trend_log[2]:.6g}\n")

    pd.DataFrame({
        "metric": list(summ.keys()) + [f"lin_{k}" for k in st_lin.keys()] + [f"log_{k}" for k in st_log.keys()],
        "value": list(summ.values()) + list(st_lin.values()) + list(st_log.values()),
    }).to_csv(outdir / "summary.csv", index=False)

    print("[完成] 输出到：", outdir)

if __name__ == "__main__":
    main()
