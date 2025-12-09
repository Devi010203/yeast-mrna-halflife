# plots/plot_test_error_breakdown.py
# -*- coding: utf-8 -*-
"""
测试集误差剖析与可靠性评估（论文用图）
输出目录：脚本同级的上一级  result/plot/test_error_breakdown/<时间戳>/   （自动创建）
每张图同时导出 .png + .svg（dpi=400）

生成图表：
  1) binned_metrics_{MAE,RMSE}_by_true_bins        # 按真实值分位数分箱的性能条形图（含样本数）
  2) calibration_deciles_ci                         # 10-bin 校准（按真值分位），均值±95%CI（对 pred 的均值做CI）
     同时导出 calibration_deciles.csv
  3) parity_hexbin / parity_density                 # 整体 Parity（Pred vs True），附 R² / Pearson / Spearman / 斜率截距 / MAE
  4) （保留）bland_altman_test                      # Bland–Altman（差异-均值）图
  5) （保留）error_vs_length / error_vs_gc          # 误差随序列长度/GC变化
  6) （新增）error_bins_bar                         # MAE & RMSE 合并条形（双轴，可选）

数据输入（在 CONFIG 顶部手动填写）：
  - RUN_DIR/final_test_predictions.csv             # 必需：含真实值/预测列（脚本可自动识别常见列名）
  - DATA_CSV（可选）                               # 总数据表（含 sequence 与 Isoform Half-Life），用于补充序列、计算GC与长度

注意：仅使用 matplotlib；不依赖 seaborn。分箱优先分位数等频分箱（qcut），失败时退回等宽分箱。
"""

import os, math, json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

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
    # 完整训练输出目录（里面有 final_test_predictions.csv）
    "RUN_DIR": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01",

    # （可选）总数据 CSV，包含至少 'sequence' 与 'Isoform Half-Life'
    "DATA_CSV": r"F:\mRNA_Project\3UTR\data\processed\mRNA_half_life_dataset_RNA.csv",

    # 输出子目录名（位于 脚本同级的上一级 /result/plot/ 下）
    "SAVE_SUBDIR": "test_error_breakdown",

    # 分箱参数
    "NUM_BINS_TRUE": 10,          # 真实值分箱个数（性能条形图/误差柱状）
    "NUM_BINS_FEATURE": 10,       # 特征分箱（长度/GC）
    "CALIB_BINS": 10,             # 校准分箱（按真实值分位：deciles）

    "ALLOW_DUP_DROP": True,       # qcut duplicates='drop' 以应对重复值

    # 置信区间（bootstrap）
    "BOOTSTRAP_N": 1000,
    "BOOTSTRAP_SEED": 20251016,
    "CI_ALPHA": 0.05,             # 95% CI

    # Parity 图配置
    "PARITY_KIND": "scatter",      # "hexbin" 或 "scatter"（"density" 将被视作 "scatter"）
    "PARITY_HEX_GRIDSIZE": 40,    # hexbin 网格密度
    "PARITY_Q_LIMITS": (0.01, 0.99),  # 坐标分位裁剪，避免极端点造成大空白
    "PARITY_PAD_FRAC": 0.03,

    # 作图风格
    "DPI": 400,
    "FIGSIZE": (6.0, 4.5),
    "GRID_ALPHA": 0.35,
}
# =================================


# ---------------- 基础工具 ----------------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent  # 脚本上一级为项目根

def _ensure_outdir() -> str:
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = _project_root() / "result" / "plot" / CONFIG["SAVE_SUBDIR"] / t
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)

def _save_dual(fig, out_base: str):
    fig.savefig(out_base + ".png", dpi=CONFIG["DPI"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", dpi=CONFIG["DPI"], bbox_inches="tight")
    plt.close(fig)

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        # 尝试自动识别编码
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
    print(f"[提示] 找不到文件：{path}")
    return None

def _compute_gc(seq: str) -> float:
    s = str(seq).upper()
    if not s: return np.nan
    n = sum(c in "ACGTU" for c in s)
    if n == 0: return np.nan
    gc = sum(c in "GC" for c in s)
    return gc / n

def _qcut_safe(x: pd.Series, q: int, allow_drop=True):
    try:
        return pd.qcut(x, q=q, duplicates="drop" if allow_drop else None)
    except Exception:
        # 回退等宽分箱
        return pd.cut(x, bins=q, include_lowest=True)

def _quantile_limits_xy(x: np.ndarray, y: np.ndarray, qlo: float = 0.01, qhi: float = 0.99, pad_frac: float = 0.03):
    """根据 x,y 的联合分位数给出紧凑的对角可读坐标范围，并留少量边距。"""
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (0.0, 1.0), (0.0, 1.0)
    x_lo, x_hi = np.quantile(x, [qlo, qhi])
    y_lo, y_hi = np.quantile(y, [qlo, qhi])
    lo = float(min(x_lo, y_lo)); hi = float(max(x_hi, y_hi))
    span = max(1e-12, hi - lo)
    lo -= pad_frac * span; hi += pad_frac * span
    return (lo, hi), (lo, hi)

# ---------------- 列名自动识别 ----------------
_POSS_TRUE = ["true", "y_true", "label", "target", "half_life", "halflife", "half-life", "y", "obs", "real"]
_POSS_PRED = ["pred", "y_pred", "prediction", "predicted", "pred_half_life", "half_life_pred", "yhat", "preds"]

def _autodetect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for key in candidates:
        if key in lower:
            return lower[key]
    # 宽松包含匹配
    for c in cols:
        lc = c.lower()
        for key in candidates:
            if key in lc or lc in key:
                return c
    return None


# ---------------- 数据加载 ----------------
def _load_test_predictions(run_dir: str, data_csv: Optional[str]) -> pd.DataFrame:
    test_csv = os.path.join(run_dir, "final_test_predictions.csv")
    dft = _safe_read_csv(test_csv)
    if dft is None:
        raise FileNotFoundError(f"缺少 {test_csv}")

    # 真实/预测列名自动识别
    t_col = _autodetect_col(list(dft.columns), _POSS_TRUE)
    p_col = _autodetect_col(list(dft.columns), _POSS_PRED)
    if t_col is None or p_col is None:
        raise ValueError(f"无法在 {test_csv} 中识别真实/预测列名，请检查列名：{list(dft.columns)}")

    dft = dft.copy()
    dft.rename(columns={t_col: "true", p_col: "pred"}, inplace=True)
    dft["true"] = pd.to_numeric(dft["true"], errors="coerce")
    dft["pred"] = pd.to_numeric(dft["pred"], errors="coerce")
    dft = dft.replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "pred"])

    # 序列获取与特征（尽力而为）
    if "sequence" not in dft.columns and data_csv:
        df_all = _safe_read_csv(data_csv)
        if df_all is not None and "sequence" in df_all.columns:
            # 没有稳定 key 就不强行 merge，避免误匹配（保持与你原逻辑一致）
            pass

    if "sequence" in dft.columns:
        dft["sequence"] = dft["sequence"].astype(str)
        dft["seq_len"] = dft["sequence"].map(len)
        dft["gc_frac"] = dft["sequence"].map(_compute_gc)
    else:
        dft["seq_len"] = np.nan
        dft["gc_frac"] = np.nan

    dft["residual"] = dft["true"] - dft["pred"]
    dft["abs_error"] = np.abs(dft["residual"])
    return dft


# ---------------- 1) 分箱性能条形图 ----------------
def plot_binned_metrics_by_true(dft: pd.DataFrame, outdir: str):
    bins = _qcut_safe(dft["true"], q=CONFIG["NUM_BINS_TRUE"], allow_drop=CONFIG["ALLOW_DUP_DROP"])
    grp = dft.groupby(bins, observed=True).agg(
        true_mean=("true", "mean"),
        mae=("abs_error", "mean"),
        rmse=("residual", lambda z: math.sqrt(np.mean(np.square(z)))),
        n=("true", "size")
    ).reset_index(drop=True)

    # MAE
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    ax.bar(np.arange(len(grp)), grp["mae"].to_numpy())
    ax.set_xlabel("True bins (quantiles)")
    ax.set_ylabel("MAE")
    ax.set_title("Test — MAE by true-value bins")
    ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    for i, (y, n) in enumerate(zip(grp["mae"], grp["n"])):
        ax.text(i, y, str(int(n)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks([])
    _save_dual(fig, os.path.join(outdir, "binned_metrics_MAE_by_true_bins"))

    # RMSE
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    ax.bar(np.arange(len(grp)), grp["rmse"].to_numpy())
    ax.set_xlabel("True bins (quantiles)")
    ax.set_ylabel("RMSE")
    ax.set_title("Test — RMSE by true-value bins")
    ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    for i, (y, n) in enumerate(zip(grp["rmse"], grp["n"])):
        ax.text(i, y, str(int(n)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks([])
    _save_dual(fig, os.path.join(outdir, "binned_metrics_RMSE_by_true_bins"))

    # （新增）合并条形（双轴）—— 文件名：error_bins_bar
    x = np.arange(len(grp))
    fig, ax1 = plt.subplots(figsize=CONFIG["FIGSIZE"])
    w = 0.4
    ax1.bar(x - w/2, grp["mae"], width=w, label="MAE", color="#4e79a7")
    ax1.set_ylabel("MAE")
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, grp["rmse"], width=w, label="RMSE", color="#f28e2b")
    ax2.set_ylabel("RMSE")
    ax1.set_xlabel("True bins (quantiles)")
    ax1.set_title("Error by true-value bins (MAE \& RMSE)")
    ax1.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    ax1.set_xticks([])
    # 样本量盖在上层
    for i, n in enumerate(grp["n"]):
        ax1.text(i - w/2, grp["mae"][i], str(int(n)), ha="center", va="bottom", fontsize=7)
    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    _save_dual(fig, os.path.join(outdir, "error_bins_bar"))


# ---------------- 2) 按真值分位的 10-bin 校准（均值±95%CI） ----------------
def _bootstrap_ci_mean(y: np.ndarray, n_boot: int, rng: np.random.RandomState, alpha: float) -> tuple[float, float]:
    """对给定样本 y 的均值做 bootstrap 置信区间（双侧，1-alpha）。"""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n <= 1:
        return (np.nan, np.nan)
    idx = np.arange(n)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        means.append(float(np.mean(y[samp])))
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1 - alpha/2))
    return (lo, hi)

def plot_calibration_true_deciles_with_ci(dft: pd.DataFrame, outdir: str):
    """
    校准曲线（10-bin，按“真值”分位分箱）：
    x = 每个 bin 的 true 均值；y = 该 bin 的 pred 均值；误差条 = pred 均值的 95%CI（bootstrap）
    输出：calibration_deciles.csv + calibration_deciles_ci.{png,svg}
    """
    # 以 true 做分位分箱
    bins = _qcut_safe(dft["true"], q=CONFIG["CALIB_BINS"], allow_drop=CONFIG["ALLOW_DUP_DROP"])
    rng = np.random.RandomState(CONFIG.get("BOOTSTRAP_SEED", 20251016))

    rows = []
    for _, g in dft.groupby(bins, observed=True):
        if len(g) == 0:
            continue
        x_true = g["true"].to_numpy(dtype=float)
        y_pred = g["pred"].to_numpy(dtype=float)
        ci_lo, ci_hi = _bootstrap_ci_mean(
            y=y_pred,
            n_boot=int(CONFIG.get("BOOTSTRAP_N", 1000)),
            rng=rng,
            alpha=float(CONFIG.get("CI_ALPHA", 0.05))
        )
        rows.append({
            "true_mean": float(np.mean(x_true)),
            "pred_mean": float(np.mean(y_pred)),
            "count": int(len(y_pred)),
            "ci_lo": ci_lo,    # pred_mean 的CI
            "ci_hi": ci_hi
        })

    if not rows:
        print("[跳过] 校准曲线：分箱后没有有效数据。")
        return

    calib = pd.DataFrame(rows).sort_values("true_mean").reset_index(drop=True)
    calib.to_csv(os.path.join(outdir, "calibration_deciles.csv"), index=False)

    # ===== 绘图：1:1 比例，保证点和 CI 都在范围内 =====
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    x_mean = calib["true_mean"].to_numpy()
    y_mean = calib["pred_mean"].to_numpy()
    ci_lo = calib["ci_lo"].to_numpy()
    ci_hi = calib["ci_hi"].to_numpy()

    ax.plot(x_mean, y_mean, marker="o", linewidth=1.5, label="mean per bin")

    # y 方向误差条（对 pred 的均值）
    yerr = np.vstack([
        y_mean - ci_lo,
        ci_hi - y_mean,
    ])
    ax.errorbar(x_mean, y_mean, yerr=yerr, fmt="none", linewidth=1.0, alpha=0.85)

    # 统一坐标范围：考虑 true_mean / pred_mean / CI 三者，避免裁掉误差条
    lo_raw = float(
        min(
            np.nanmin(x_mean),
            np.nanmin(y_mean),
            np.nanmin(ci_lo),
        )
    )
    hi_raw = float(
        max(
            np.nanmax(x_mean),
            np.nanmax(y_mean),
            np.nanmax(ci_hi),
        )
    )
    span = max(1e-12, hi_raw - lo_raw)
    lo = lo_raw - 0.04 * span
    hi = hi_raw + 0.04 * span

    # 理想参考线 y = x
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="y = x")

    # 标注每箱样本数
    for x0, y0, n in zip(x_mean, y_mean, calib["count"]):
        ax.annotate(str(int(n)), (x0, y0), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")  # 坐标尺度 1:1

    ax.set_xlabel("Bin Mean True")
    ax.set_ylabel("Bin Mean Prediction")
    # ax.set_title(f"Calibration (true deciles) with 95% CI (bins={CONFIG['CALIB_BINS']})")
    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    _save_dual(fig, os.path.join(outdir, "calibration_deciles_ci"))



# ---------------- 3) 整体 Parity（附指标，hexbin/散点） ----------------
def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    # 无穷依赖：用 pandas 排名 + 皮尔逊
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return _pearsonr(xr, yr)

def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ux, uy = float(np.mean(x)), float(np.mean(y))
    num = float(np.sum((x - ux) * (y - uy)))
    den = float(np.sum((x - ux) ** 2))
    if den <= 0: return (np.nan, np.nan)
    a = num / den
    b = uy - a * ux
    return (float(a), float(b))

def plot_parity(dft: pd.DataFrame, outdir: str):
    x = dft["true"].to_numpy(dtype=float)
    y = dft["pred"].to_numpy(dtype=float)

    # 指标
    mae = float(np.mean(np.abs(y - x)))
    # R² = 1 - SS_res/SS_tot
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pear = _pearsonr(x, y)
    spear = _spearmanr(x, y)
    slope, intercept = _ols_slope_intercept(x, y)

    # 坐标分位裁剪以去掉右上角空白
    qlo, qhi = CONFIG.get("PARITY_Q_LIMITS", (0.01, 0.99))
    pad_frac = CONFIG.get("PARITY_PAD_FRAC", 0.03)
    (xlim, ylim) = _quantile_limits_xy(x, y, qlo=qlo, qhi=qhi, pad_frac=pad_frac)

    kind = (CONFIG.get("PARITY_KIND", "hexbin") or "hexbin").lower()
    if kind == "density":  # 兼容写法
        kind = "scatter"

    # ===== 正方形画布 =====
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    if kind == "hexbin":
        hb = ax.hexbin(
            x,
            y,
            gridsize=int(CONFIG.get("PARITY_HEX_GRIDSIZE", 40)),
            cmap="viridis",
            mincnt=1,
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("count")
        out_name = "parity_hexbin"
    else:
        ax.scatter(x, y, s=8, alpha=0.6)
        out_name = "parity_density"

    # y = x 参考线
    ax.plot(
        [xlim[0], xlim[1]],
        [xlim[0], xlim[1]],
        linestyle="--",
        linewidth=1.2,
        color="k",
        alpha=0.8,
        label="y = x",
    )

    # 拟合线
    if np.isfinite(slope) and np.isfinite(intercept):
        xs_line = np.array([xlim[0], xlim[1]])
        ax.plot(
            xs_line,
            slope * xs_line + intercept,
            linewidth=1.4,
            color="#d62728",
            alpha=0.9,
            label=f"fit: y={slope:.2f}x+{intercept:.2f}",
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")  # 坐标尺度 1:1

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # ax.set_title("Test Parity")

    # 角标注（文本框）
    text = (
        f"R² = {r2:.3f}\n"
        f"Pearson = {pear:.3f}\n"
        f"Spearman = {spear:.3f}\n"
        f"slope = {slope:.3f}, intercept = {intercept:.3f}\n"
        f"MAE = {mae:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
    )

    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    # ✅ 确保只调用一次 legend，因此只会有一个图例框
    ax.legend(loc="lower right", fontsize=11, frameon=False)

    fig.tight_layout()
    _save_dual(fig, os.path.join(outdir, out_name))



# ---------------- 4) Bland–Altman（保留） ----------------
def plot_bland_altman(dft: pd.DataFrame, outdir: str):
    mean_vals = 0.5 * (dft["true"].to_numpy() + dft["pred"].to_numpy())
    diff_vals = (dft["true"] - dft["pred"]).to_numpy()
    bias = float(np.mean(diff_vals))
    sd = float(np.std(diff_vals, ddof=1))
    loA = bias - 1.96 * sd
    hiA = bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    ax.scatter(mean_vals, diff_vals, s=10, alpha=0.6)
    ax.axhline(bias, color="k", linestyle="-", linewidth=1.2, label=f"bias={bias:.3f}")
    ax.axhline(loA, color="k", linestyle="--", linewidth=1.0, label=f"LoA={loA:.3f}")
    ax.axhline(hiA, color="k", linestyle="--", linewidth=1.0, label=f"HiA={hiA:.3f}")
    ax.set_xlabel("Mean of True and Pred")
    ax.set_ylabel("True - Pred")
    ax.set_title("Bland–Altman (Test)")
    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    ax.legend()
    _save_dual(fig, os.path.join(outdir, "bland_altman_test"))


# ---------------- 5) 误差 vs 序列长度/GC（保留） ----------------
def _plot_error_vs_feature(dft: pd.DataFrame, feat: str, outpath: str, ylabel="Mean |Error|"):
    if feat not in dft.columns or dft[feat].isna().all():
        print(f"[跳过] 缺少特征列：{feat}")
        return
    try:
        bins = pd.qcut(dft[feat], q=CONFIG["NUM_BINS_FEATURE"], duplicates="drop" if CONFIG["ALLOW_DUP_DROP"] else None)
    except Exception:
        bins = pd.cut(dft[feat], bins=CONFIG["NUM_BINS_FEATURE"])
    grp = dft.groupby(bins, observed=True).agg(
        feat_mean=(feat, "mean"),
        mean_abs_err=("abs_error", "mean"),
        n=("abs_error", "size")
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    ax.plot(grp["feat_mean"], grp["mean_abs_err"], marker="o", linewidth=1.5)
    for x, y, n in zip(grp["feat_mean"], grp["mean_abs_err"], grp["n"]):
        ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel(feat)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Test — {ylabel} vs {feat}")
    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    _save_dual(fig, outpath)


# ---------------- 主流程 ----------------
def main():
    outdir = _ensure_outdir()

    run_dir = CONFIG["RUN_DIR"]
    if not run_dir or not os.path.isdir(run_dir):
        raise NotADirectoryError("请在 CONFIG['RUN_DIR'] 填写完整训练输出目录。")

    dft = _load_test_predictions(run_dir, CONFIG.get("DATA_CSV"))

    # 1) 分箱性能（按真实值）
    plot_binned_metrics_by_true(dft, outdir)

    # 2) 校准（按真值分位，10-bin，均值±95%CI）
    plot_calibration_true_deciles_with_ci(dft, outdir)

    # 3) 整体 Parity（hexbin/散点，附指标）
    plot_parity(dft, outdir)

    # 4) Bland–Altman（保留）
    plot_bland_altman(dft, outdir)

    # 5) 误差 vs 序列长度/GC（若有序列）
    _plot_error_vs_feature(dft, "seq_len", os.path.join(outdir, "error_vs_length"))
    _plot_error_vs_feature(dft, "gc_frac", os.path.join(outdir, "error_vs_gc"))

    print(f"[OK] 测试集误差剖析图已生成：{outdir}")

if __name__ == "__main__":
    main()
