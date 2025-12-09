# plots/plot_5fold_residual_distribution.py
# -*- coding: utf-8 -*-
"""
5 折残差分布 & 正态性分析：
  - 每折：残差直方图（叠加匹配均值/方差的正态曲线）
  - 每折：QQ 图（对标准正态）
  - 汇总：小提琴 + 箱线（各折 + All）
  - 导出：summary_per_fold.csv（MAE, RMSE, mean, sd, skew, kurt, KS/Normaltest p）
输出目录：<项目根>/result/plot/residual_distribution/<时间戳>/
图像：PNG+SVG（dpi=400）；坐标文字非斜体
"""

import os, math
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

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

# ========== 在此手动填写（不使用命令行）==========
# 为每折指定 CSV 路径（只需包含“真值列 + 预测列”，列名可自动识别）
FOLD_FILES: Dict[str, str] = {
    "Fold1": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01\val_predictions_fold1.csv",
    "Fold2": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01\val_predictions_fold2.csv",
    "Fold3": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01\val_predictions_fold3.csv",
    "Fold4": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01\val_predictions_fold4.csv",
    "Fold5": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01\val_predictions_fold5.csv",
}
# 直方图设置
NBINS = 60
POINT_ALPHA = 0.25
DPI = 400

DO_LOG1P = True          # 是否额外生成 log1p 残差版本的 1–3
SMOOTH_QUANTILES = np.linspace(0.0, 1.0, 11)  # 分位平滑的横坐标分位点
NBINS_HIST = 60          # 直方图箱数
POINT_ALPHA = 0.25       # 散点透明度，避免遮挡

# 新增：用于限制散点显示范围的分位数（只影响散点，不影响统计）
VIOLIN_ABS_Q = 0.99   # 小提琴/箱线图仅展示 |residual| 的 99% 分位范围（可调）


# ===============================================


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_outdirs():
    root = _project_root() / "result" / "plot" / "residual_distribution" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (root / "hist").mkdir(parents=True, exist_ok=True)
    (root / "qq").mkdir(parents=True, exist_ok=True)
    (root / "violin").mkdir(parents=True, exist_ok=True)
    return root

def _save_dual(fig, out_base: Path):
    fig.savefig(str(out_base) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def _auto_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cand_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    yt = yp = None
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

def _read_residuals(fp: Path) -> np.ndarray:
    df = pd.read_csv(fp).dropna(how="all")
    yt, yp = _auto_cols(df)
    y_true = df[yt].astype(float).values
    y_pred = df[yp].astype(float).values
    resid = y_pred - y_true
    resid = resid[np.isfinite(resid)]
    return resid

def _plot_hist_with_gauss(resid: np.ndarray, title: str, out_base: Path):
    mu = float(np.mean(resid))
    sd = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(resid, bins=NBINS, density=True, alpha=0.75, color="C0", edgecolor="white", linewidth=0.4)
    if sd > 0:
        xs = np.linspace(mu - 4*sd, mu + 4*sd, 400)
        pdf = (1.0/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mu)/sd)**2)
        ax.plot(xs, pdf, color="C1", linewidth=2.0, label=f"Normal($\\mu$={mu:.3g}, $\\sigma$={sd:.3g})")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Residual (prediction − truth)")
    ax.set_ylabel("Density")
    # ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    if sd > 0: ax.legend(frameon=False)
    _save_dual(fig, out_base)

def _plot_qq(resid: np.ndarray, title: str, out_base: Path):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm", sparams=(), fit=True)
    ax.scatter(osm, osr, s=12, alpha=0.6, linewidth=0)
    # 理论直线
    xx = np.array([np.min(osm), np.max(osm)], dtype=float)
    ax.plot(xx, intercept + slope*xx, color="C1", linewidth=2.0, label=f"fit: y={intercept:.2g}+{slope:.2g}x")
    ax.set_xlabel("Theoretical quantiles (Normal)")
    ax.set_ylabel("Ordered residuals")
    # ax.set_title(title + f"\n(R = {r:.3f})")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="best")
    _save_dual(fig, out_base)

def _summary_stats(resid: np.ndarray) -> Dict[str, float]:
    if resid.size == 0:
        return {k: np.nan for k in ["N","MAE","RMSE","mean","sd","skew","kurt","ks_p","normal_p"]}
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mean = float(np.mean(resid))
    sd = float(np.std(resid, ddof=1)) if resid.size > 1 else np.nan
    skew = float(stats.skew(resid, bias=False)) if resid.size > 2 else np.nan
    kurt = float(stats.kurtosis(resid, fisher=True, bias=False)) if resid.size > 3 else np.nan
    # 正态性检验
    ks_p = np.nan
    try:
        # KS 需给定均值和方差后的正态分布
        if np.isfinite(sd) and sd > 0:
            ks_p = float(stats.kstest(resid, "norm", args=(mean, sd)).pvalue)
    except Exception:
        pass
    normal_p = np.nan
    try:
        if resid.size >= 20:
            normal_p = float(stats.normaltest(resid).pvalue)  # D'Agostino K^2
    except Exception:
        pass
    return {"N": resid.size, "MAE": mae, "RMSE": rmse, "mean": mean, "sd": sd, "skew": skew, "kurt": kurt, "ks_p": ks_p, "normal_p": normal_p}

def main():
    outdir = _ensure_outdirs()
    print("[输出目录]", outdir)

    per_fold_resid: Dict[str, np.ndarray] = {}
    rows = []

    # 逐折读取与绘图
    for fold, path in FOLD_FILES.items():
        fp = Path(path)
        if not fp.is_file():
            print(f"[WARN] 未找到文件：{fp}，跳过 {fold}")
            continue
        resid = _read_residuals(fp)
        per_fold_resid[fold] = resid

        # 直方图+高斯曲线
        _plot_hist_with_gauss(resid, f"Residual histogram — {fold}", outdir / "hist" / f"hist_{fold}")
        # QQ 图
        _plot_qq(resid, f"QQ plot — {fold}", outdir / "qq" / f"qq_{fold}")
        # 统计
        st = _summary_stats(resid)
        st["fold"] = fold
        rows.append(st)

    if len(per_fold_resid) == 0:
        raise RuntimeError("没有任何有效折的数据，请在 FOLD_FILES 中填写正确的 CSV 路径。")

    # 汇总 All
    all_resid = np.concatenate(list(per_fold_resid.values()))
    _plot_hist_with_gauss(all_resid, "Residual histogram — All folds", outdir / "hist" / "hist_all")
    _plot_qq(all_resid, "QQ plot — All folds", outdir / "qq" / "qq_all")
    st_all = _summary_stats(all_resid); st_all["fold"] = "All"
    rows.append(st_all)

    # 小提琴 + 箱线（各折 + All）
    labels = list(per_fold_resid.keys()) + ["All"]
    data = [per_fold_resid[k] for k in per_fold_resid.keys()] + [all_resid]

    # ---- 计算全局 |residual| 的分位数，用于控制 y 轴范围 ----
    all_concat = np.concatenate(data)
    if all_concat.size > 0 and np.any(np.isfinite(all_concat)):
        max_abs = float(np.quantile(np.abs(all_concat), VIOLIN_ABS_Q))
    else:
        max_abs = 0.0

    # 为了避免极端点把 violin 形状拉扁，这里可以把每折数据 clip 一下（仅用于图）
    if max_abs > 0:
        data_violin = [np.clip(d, -max_abs, max_abs) for d in data]
    else:
        data_violin = data

    # 小提琴
    fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(labels)), 4.8))
    parts = ax.violinplot(data_violin, showmeans=True, showextrema=False, widths=0.9)
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Residual (prediction − truth)")
    # ax.set_title("Residual distribution — violin (per fold)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    if max_abs > 0:
        ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    _save_dual(fig, outdir / "violin" / "violin_per_fold")

    # 箱线（同样使用 clip 后的数据 + 同一 y 轴范围）
    if max_abs > 0:
        data_box = data_violin
    else:
        data_box = data

    fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(labels)), 4.8))
    ax.boxplot(data_box, showmeans=True, meanline=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Residual (prediction − truth)")
    # ax.set_title("Residual distribution — boxplot (per fold)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    if max_abs > 0:
        ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    _save_dual(fig, outdir / "violin" / "box_per_fold")

    # 导出统计
    cols = ["fold","N","MAE","RMSE","mean","sd","skew","kurt","ks_p","normal_p"]
    pd.DataFrame(rows)[cols].to_csv(outdir / "summary_per_fold.csv", index=False)

    print("[完成] 图像与统计输出至：", outdir)

if __name__ == "__main__":
    main()
