# plots/plot_test_error_breakdown.py
# -*- coding: utf-8 -*-
"""
Test Set Error Breakdown and Reliability Assessment (Paper Figures)
Output Directory: Parent directory of script level  result/plot/test_error_breakdown/<timestamp>/   (Automatically created)
Each figure exported as .png + .svg (dpi=400)

Generated Charts:
  1) binned_metrics_{MAE,RMSE}_by_true_bins        # Performance bar chart binned by true value deciles (includes sample counts)
  2) calibration_deciles_ci                         # 10-bin calibration (by true value deciles), mean ± 95% CI (CI for mean of pred)
     Simultaneously export calibration_deciles.csv
  3) parity_hexbin / parity_density                 # Overall Parity (Pred vs True), with R² / Pearson / Spearman / slope-intercept / MAE
  4) (Retained) bland_altman_test                      # Bland–Altman (difference-mean) plot
  5) (Retained) error_vs_length / error_vs_gc          # Error vs. sequence length / GC content
  6) (New) error_bins_bar                            # Combined MAE & RMSE bar chart (dual axes, optional)

Data Input (manually entered at top of CONFIG):
  - RUN_DIR/final_test_predictions.csv             # Required: Contains true/predicted columns (script automatically recognizes common column names)
  - DATA_CSV (optional)                               # Master data table (includes sequence and Isoform Half-Life), used for sequence supplementation, GC and length calculation

Note: Uses only matplotlib; does not depend on seaborn. Prioritizes quantile-based equal-frequency binning (qcut), falls back to equal-width binning if qcut fails.
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

    "font.size":17,
    "axes.titlesize":20,
    "axes.labelsize":19,
    "xtick.labelsize":15,
    "ytick.labelsize":15,
    "legend.fontsize":16,
    "figure.titlesize":17,
    # "axes.titleweight": "bold",
    # "axes.labelweight": "bold",
})

# ========== Please fill in manually here ==========
CONFIG = {
    # Complete training output directory (containing final_test_predictions.csv)
    "RUN_DIR": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01",

    # (Optional) Aggregate data CSV containing at least the ‘sequence’ with the 'Isoform Half-Life'
    "DATA_CSV": r"F:\mRNA_Project\3UTR\data\processed\mRNA_half_life_dataset_RNA.csv",

    # Output subdirectory name (located in the parent directory of the script, under `/result/plot/`)
    "SAVE_SUBDIR": "test_error_breakdown",

    # Boxing Parameters
    "NUM_BINS_TRUE": 10,          # Number of bins for actual values (Performance histogram/error bars)
    "NUM_BINS_FEATURE": 10,       # Feature Binning (Length/GC)
    "CALIB_BINS": 10,             # Calibration compartments (in bits of true value: deciles)

    "ALLOW_DUP_DROP": True,       # qcut duplicates='drop' To address duplicate values

    # Confidence Interval (Bootstrap)
    "BOOTSTRAP_N": 1000,
    "BOOTSTRAP_SEED": 20251016,
    "CI_ALPHA": 0.05,             # 95% CI

    # Parity Figure Placement
    "PARITY_KIND": "scatter",      # "hexbin" or “scattering” (‘density’ will be treated as “scattering”)
    "PARITY_HEX_GRIDSIZE": 40,    # hexbin Grid Density
    "PARITY_Q_LIMITS": (0.01, 0.99),  # Coordinate quantile trimming to prevent large gaps caused by outliers
    "PARITY_PAD_FRAC": 0.03,

    # Graphic Style
    "DPI": 400,
    "FIGSIZE": (6.0, 4.5),
    "GRID_ALPHA": 0.35,
}
# =================================


# ---------------- Basic Tools ----------------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent  # The script's parent level is the project root.

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
        # Attempt automatic encoding recognition
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
    print(f"[Notice] File not found:{path}")
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
        # Rollback of fixed-width partitioning
        return pd.cut(x, bins=q, include_lowest=True)

def _quantile_limits_xy(x: np.ndarray, y: np.ndarray, qlo: float = 0.01, qhi: float = 0.99, pad_frac: float = 0.03):
    """Provide a compact diagonal range of readable coordinates based on the joint quantiles of x and y, leaving a small margin."""
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (0.0, 1.0), (0.0, 1.0)
    x_lo, x_hi = np.quantile(x, [qlo, qhi])
    y_lo, y_hi = np.quantile(y, [qlo, qhi])
    lo = float(min(x_lo, y_lo)); hi = float(max(x_hi, y_hi))
    span = max(1e-12, hi - lo)
    lo -= pad_frac * span; hi += pad_frac * span
    return (lo, hi), (lo, hi)

# ---------------- List-based Automatic Recognition ----------------
_POSS_TRUE = ["true", "y_true", "label", "target", "half_life", "halflife", "half-life", "y", "obs", "real"]
_POSS_PRED = ["pred", "y_pred", "prediction", "predicted", "pred_half_life", "half_life_pred", "yhat", "preds"]

def _autodetect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for key in candidates:
        if key in lower:
            return lower[key]
    # Loose containment matching
    for c in cols:
        lc = c.lower()
        for key in candidates:
            if key in lc or lc in key:
                return c
    return None


# ---------------- Data loading ----------------
def _load_test_predictions(run_dir: str, data_csv: Optional[str]) -> pd.DataFrame:
    test_csv = os.path.join(run_dir, "final_test_predictions.csv")
    dft = _safe_read_csv(test_csv)
    if dft is None:
        raise FileNotFoundError(f"Lacking {test_csv}")

    # Automatic Identification of Actual/Predicted Column Names
    t_col = _autodetect_col(list(dft.columns), _POSS_TRUE)
    p_col = _autodetect_col(list(dft.columns), _POSS_PRED)
    if t_col is None or p_col is None:
        raise ValueError(f"Unable to recognize actual/predicted column names in {test_csv}. Please check the column names:{list(dft.columns)}")

    dft = dft.copy()
    dft.rename(columns={t_col: "true", p_col: "pred"}, inplace=True)
    dft["true"] = pd.to_numeric(dft["true"], errors="coerce")
    dft["pred"] = pd.to_numeric(dft["pred"], errors="coerce")
    dft = dft.replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "pred"])

    # Sequence Acquisition and Features
    if "sequence" not in dft.columns and data_csv:
        df_all = _safe_read_csv(data_csv)
        if df_all is not None and "sequence" in df_all.columns:
            # Without a stable key, do not force a merge to avoid incorrect matching.
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


# ---------------- 1) Box-and-Whisker Plot for Sorting Performance ----------------
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

    # (New) Merged Bar Chart (Dual-Axis) — File Name:error_bins_bar
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
    # Sample size is covered in the upper layer.
    for i, n in enumerate(grp["n"]):
        ax1.text(i - w/2, grp["mae"][i], str(int(n)), ha="center", va="bottom", fontsize=7)
    # Combined Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    _save_dual(fig, os.path.join(outdir, "error_bins_bar"))


# ---------------- 2) 10-bin calibration by true value bin (mean ± 95% CI) ----------------
def _bootstrap_ci_mean(y: np.ndarray, n_boot: int, rng: np.random.RandomState, alpha: float) -> tuple[float, float]:
    """Estimate the bootstrap confidence interval (two-tailed, 1-alpha) for the mean of the given sample y."""
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
    Calibration curve (10-bin, decile binning by "true value"):
    x = true mean for each bin; y = predicted mean for that bin; error bars = 95% CI (bootstrap) of predicted mean
    Output: calibration_deciles.csv + calibration_deciles_ci.{png,svg}
    """
    # Partition into bins using true as the cutoff value
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
            "ci_lo": ci_lo,    # Confidence interval for pred_mean
            "ci_hi": ci_hi
        })

    if not rows:
        print("[Skip] Calibration curve: No valid data after binning.")
        return

    calib = pd.DataFrame(rows).sort_values("true_mean").reset_index(drop=True)
    calib.to_csv(os.path.join(outdir, "calibration_deciles.csv"), index=False)

    # ===== Drawing: 1:1 scale, ensuring points and CI are within range =====
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    x_mean = calib["true_mean"].to_numpy()
    y_mean = calib["pred_mean"].to_numpy()
    ci_lo = calib["ci_lo"].to_numpy()
    ci_hi = calib["ci_hi"].to_numpy()

    ax.plot(x_mean, y_mean, marker="o", linewidth=1.5, label="mean per bin")

    # y Direction Error Bar (Mean of pred)
    yerr = np.vstack([
        y_mean - ci_lo,
        ci_hi - y_mean,
    ])
    ax.errorbar(x_mean, y_mean, yerr=yerr, fmt="none", linewidth=1.0, alpha=0.85)

    # Unify coordinate range: Consider true_mean, pred_mean, and CI together to avoid clipping error bars.
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

    # Ideal Reference Line y = x
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="y = x")

    # Indicate the number of samples per box
    for x0, y0, n in zip(x_mean, y_mean, calib["count"]):
        ax.annotate(str(int(n)), (x0, y0), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")  # Coordinate scale 1:1

    ax.set_xlabel("Bin Mean True")
    ax.set_ylabel("Bin Mean Prediction")
    # ax.set_title(f"Calibration (true deciles) with 95% CI (bins={CONFIG['CALIB_BINS']})")
    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    _save_dual(fig, os.path.join(outdir, "calibration_deciles_ci"))



# ---------------- 3) Overall Parity (with Metrics, Hexbin/Scatter Plot) ----------------
def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    # Infinite Dependence: Ranking with pandas + Pearson
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

    # Indicator
    mae = float(np.mean(np.abs(y - x)))
    # R² = 1 - SS_res/SS_tot
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pear = _pearsonr(x, y)
    spear = _spearmanr(x, y)
    slope, intercept = _ols_slope_intercept(x, y)

    # Coordinate-based cropping to remove the upper-right corner blank space
    qlo, qhi = CONFIG.get("PARITY_Q_LIMITS", (0.01, 0.99))
    pad_frac = CONFIG.get("PARITY_PAD_FRAC", 0.03)
    (xlim, ylim) = _quantile_limits_xy(x, y, qlo=qlo, qhi=qhi, pad_frac=pad_frac)

    kind = (CONFIG.get("PARITY_KIND", "hexbin") or "hexbin").lower()
    if kind == "density":  # Compatible Writing Style
        kind = "scatter"

    # ===== Square canvas =====
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

    # y = x Reference line
    ax.plot(
        [xlim[0], xlim[1]],
        [xlim[0], xlim[1]],
        linestyle="--",
        linewidth=1.2,
        color="k",
        alpha=0.8,
        label="y = x",
    )

    # Fitting line
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
    ax.set_aspect("equal", adjustable="box")  # Coordinate scale 1:1

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # ax.set_title("Test Parity")

    # Superscript annotation (text box)
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
    # Ensure that the legend is called only once, so that only one legend box will appear.
    ax.legend(loc="lower right", fontsize=11, frameon=False)

    fig.tight_layout()
    _save_dual(fig, os.path.join(outdir, out_name))



# ---------------- 4) Bland–Altman（Retain） ----------------
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


# ---------------- 5) Error vs. Sequence Length/GC (Retained) ----------------
def _plot_error_vs_feature(dft: pd.DataFrame, feat: str, outpath: str, ylabel="Mean |Error|"):
    if feat not in dft.columns or dft[feat].isna().all():
        print(f"[Skip] Missing feature column:{feat}")
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


# ---------------- Main Process ----------------
def main():
    outdir = _ensure_outdir()

    run_dir = CONFIG["RUN_DIR"]
    if not run_dir or not os.path.isdir(run_dir):
        raise NotADirectoryError("Please enter the full training output directory in CONFIG['RUN_DIR'].")

    dft = _load_test_predictions(run_dir, CONFIG.get("DATA_CSV"))

    # 1) Boxing Performance (Actual Values)
    plot_binned_metrics_by_true(dft, outdir)

    # 2) Calibration (by true value bin, 10-bin, mean ± 95% CI)
    plot_calibration_true_deciles_with_ci(dft, outdir)

    # 3) Overall Parity (hexbin/scatter plot with indicators)
    plot_parity(dft, outdir)

    # 4) Bland–Altman(Retain)
    plot_bland_altman(dft, outdir)

    # 5) Error vs. Sequence Length/GC Content (if sequence available)
    _plot_error_vs_feature(dft, "seq_len", os.path.join(outdir, "error_vs_length"))
    _plot_error_vs_feature(dft, "gc_frac", os.path.join(outdir, "error_vs_gc"))

    print(f"[OK] Test set error analysis chart has been generated:{outdir}")

if __name__ == "__main__":
    main()
