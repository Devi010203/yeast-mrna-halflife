# plots/plot_cv.py
# -*- coding: utf-8 -*-
"""
Five-fold cross-validation plotting (strictly aligned with main program output; paths manually specified in code):
- Output location: result/plot/<SAVE_SUBDIR>/ (automatically created) in the same directory as the script
- Manually specify RUN_DIR (output directory for the current main program run) in CONFIG
- Automatically detects main program's actual outputs:
    * RUN_DIR/cv_summary.csv
    * RUN_DIR/training_log.json
    * RUN_DIR/val_predictions_fold*.csv
    * RUN_DIR/learning_rate_schedule_fold*.csv (optional)
- Visualizations:
    1) R² Bar Chart (with dashed mean line; dynamically calculated from val_predictions_* if needed)
    2) R² Boxplot (one of three modes per configuration, default bootstrap: one box per fold)
       - aggregate   : Combines R² from all 5 folds into a single overall box (single box)
       - per_epoch: Uses val_r2 from each epoch within a fold as its distribution (5 boxes)
       - bootstrap: Obtains R² distribution via bootstrap sampling for each fold's validation set (5 boxes, recommended)
    3) Val R² learning curve (per fold)
    4) Loss learning curve (per fold) — Outputs separately:
       - Training set: cv_train_loss_learning_curves.{png,svg}
       - Validation set: cv_val_loss_learning_curves.{png,svg}
       - Combined plot: cv_loss_learning_curves_combined.{png,svg}
    5) Fold-wise validation scatter plot collage: cv_val_scatter_folds.{png,svg}
    6) Independent calibration curve per fold
    7) (Optional) Learning rate curve per fold
- Export each image as both PNG and SVG, dpi=400
"""
import os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


from plot_utils import (
    ensure_dir, savefig_dual,
    safe_read_csv, read_json_or_jsonl,
    scatter_true_pred, calibration_curve  # Retain the original import (without altering any other logic)
)
from sklearn.metrics import r2_score  # Used to back-calculate R² from val_predictions_*

# ========= Manually enter your input and output configurations here. =========
CONFIG = {
    # Output directory from a specific run of the main program (containing files such as training_log.json, cv_summary.csv, val_predictions_fold*.csv, etc.)
    # For example:"/home/zdl4/mRNA/python/3UTR/runs_transformer_accumulation/test_withsavedata_20251007_01"
    "RUN_DIR": r"Path\To\Your\Training\Output\Directory",

    # Output subdirectory names -> result/plot/<SAVE_SUBDIR>/
    "SAVE_SUBDIR": "K-Fold-CrossValidation",

    # Number of calibrated compartments
    "CALIB_BINS": 20,

    # R² Box Plot Pattern: "bootstrap" | "per_epoch" | "aggregate"
    "R2_BOX_MODE": "bootstrap",

    # bootstrap Parameters (only valid when R2_BOX_MODE="bootstrap")
    "BOOT_N": 1000,          # Number of self-service samples per fold
    "BOOT_SEED": 20251015,   # Random seed
    "JITTER_MAX_POINTS": 300 # Maximum number of points for stacked jitter scatter plots（Prevent overcrowding of figures）
}

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

# ---------- Unify graphic dimensions & fold colors/markers ----------

# Loss Learning Curve Chart at 4:3 Aspect Ratio
FIGSIZE_LOSS = (6.4, 4.8)

# The side length (in inches) of each scatter subgraph, ensuring the subgraph is 1:1.
FIGSIZE_SCATTER_SUB = 4.0

# Each fold corresponds to a fixed color & marker, ensuring consistent style across different diagrams.
# FOLD_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
FOLD_MARKERS = ["o"]
FOLD_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def _style_for_fold(fold_idx, idx_fallback=0):
    """
    Return (color, marker) based on the fold index, ensuring consistent scatter plot style for that fold across different plots.
When fold_idx cannot be parsed, fall back to idx_fallback.
    """
    if fold_idx is None:
        i = idx_fallback
    else:
        try:
            i = int(fold_idx)-1
        except Exception:
            i = idx_fallback
    color = FOLD_COLORS[i % len(FOLD_COLORS)]
    marker = FOLD_MARKERS[i % len(FOLD_MARKERS)]
    return color, marker

# ---------- gadget ----------
def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:  # Exact match
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:     # Ignore case matching
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def _auto_files(run_dir: str):
    """Based on the actual output of the main program, automatically collect the required files."""
    files = {}
    files["cv_summary"] = os.path.join(run_dir, "cv_summary.csv")
    files["training_log"] = os.path.join(run_dir, "training_log.json")
    files["val_fold_files"] = sorted(glob.glob(os.path.join(run_dir, "val_predictions_fold*.csv")))
    files["lr_fold_files"] = sorted(glob.glob(os.path.join(run_dir, "learning_rate_schedule_fold*.csv")))
    return files

def _infer_fold_idx(fp: str):
    """Extract the folder number from the filename:val_predictions_fold{N}.csv"""
    name = os.path.basename(fp)
    for token in name.replace(".csv","").split("_"):
        if token.lower().startswith("fold"):
            try:
                return int(token.lower().replace("fold",""))
            except Exception:
                return None
    return None

def _fallback_cv_summary_from_preds(val_fold_files, save_to=None):
    """When cv_summary.csv is missing, calculate R² from the val_predictions_* files for each fold and (optionally) save the completed CSV."""
    rows = []
    for fp in val_fold_files:
        df = safe_read_csv(fp)
        if df.empty or not {"true","pred"}.issubset(df.columns):
            continue
        fold = _infer_fold_idx(fp)
        r2 = r2_score(df["true"].values, df["pred"].values)
        rows.append({"fold": fold, "val_r2": r2})
    if not rows:
        return pd.DataFrame()
    df_sum = pd.DataFrame(rows).sort_values("fold")
    df_sum["mean_r2"] = df_sum["val_r2"].mean()
    if save_to:
        try: df_sum.to_csv(save_to, index=False)
        except Exception: pass
    return df_sum

# ----------【Compact Coordinates and Calibration】----------
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

def calibration_curve_tight(df: pd.DataFrame, n_bins: int, out_basepath: str, title: str,
                            q_limits=(0.01, 0.99), pad_frac=0.03, min_per_bin: int = 10):
    """Plot the calibration curve in bins only within the predicted percentile range of the samples,
    with a compact coordinate range and the upper-right corner blank area removed."""
    df = df[["true","pred"]].replace([np.inf,-np.inf], np.nan).dropna()
    if df.empty:
        return
    y = df["true"].values.astype(float)
    p = df["pred"].values.astype(float)

    # Use the quantile range of the "predicted value" as the visible range (to avoid areas with no data).
    qlo, qhi = q_limits
    p_finite = p[np.isfinite(p)]
    plo, phi = np.quantile(p_finite, [qlo, qhi])
    span = max(1e-12, phi - plo)
    plo -= pad_frac * span; phi += pad_frac * span

    # Equal-width binning, retaining only bins with sufficient sample size
    edges = np.linspace(plo, phi, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i+1]) & np.isfinite(y)
        cnt = int(np.sum(m))
        if cnt >= min_per_bin:
            xs.append(float(np.mean(p[m])))
            ys.append(float(np.mean(y[m])))
            ns.append(cnt)
    if len(xs) < 2:
        return

    xs = np.array(xs); ys = np.array(ys)

    # Y-axis range aligns with scatter plot: combined quantiles + padding; X-axis dominated by prediction range.
    (xlim_joint, ylim_joint) = _quantile_limits_xy(y, p, qlo, qhi, pad_frac)
    xlim = (float(plo), float(phi))
    ylim = (min(ylim_joint[0], xlim[0]), max(ylim_joint[1], xlim[1]))

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "--", lw=1.0, alpha=0.6, color="k", label="ideal")
    ax.plot(xs, ys, "-o", lw=1.6, ms=3.5, alpha=0.95, label="calibration")
    for xi, yi, ni in zip(xs, ys, ns):
        ax.text(xi, yi, f"{ni}", fontsize=7, ha="center", va="bottom", alpha=0.7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.grid(True, alpha=0.35); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_dual(fig, out_basepath, dpi=400)
    plt.close(fig)

# ---------- Box Plot: Three Patterns ----------
def plot_cv_box_aggregate(cv_csv: pd.DataFrame, outdir: str):
    """Combine the 50% R² into a single overall bin (single bin)"""
    r2_col = _find_col(cv_csv, ["val_r2","r2","valR2","Val_R2"])
    if r2_col is None or cv_csv.empty: return
    vals = cv_csv[r2_col].astype(float).values
    mean_r2 = float(np.mean(vals)); median_r2 = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.boxplot([vals], vert=True, patch_artist=False, showmeans=True, meanline=True, widths=0.5)
    xj = np.random.normal(loc=1.0, scale=0.03, size=len(vals))
    ax.scatter(xj, vals, s=18, alpha=0.8)
    ax.set_xticks([1]); ax.set_xticklabels(["Val R² (folds)"])
    ax.set_ylabel("Val. R²"); ax.set_title("5-fold val. R² (aggregate)")
    ax.grid(True)
    ax.text(1.16, mean_r2, f"mean={mean_r2:.3f}", va="center", fontsize=8)
    ax.text(0.84, median_r2, f"median={median_r2:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_box_aggregate"), dpi=400)
    plt.close(fig)

def plot_cv_box_per_epoch(trainlog_items, outdir: str):
    """
    Use the val_r2 values from each epoch within each fold as the "distribution," yielding 5 bins.
Note: Epochs exhibit strong correlation, resulting in weaker statistical significance compared to bootstrap sampling.
    """
    if not trainlog_items: return
    df = pd.DataFrame(trainlog_items)
    vcol = _find_col(df, ["val_r2","valR2","Val_R2"])
    fcol = _find_col(df, ["fold","Fold"])
    if vcol is None or fcol is None or df.empty: return

    groups = []
    labels = []
    for f in sorted(df[fcol].dropna().unique()):
        sub = df[df[fcol]==f][vcol].dropna().astype(float).values
        if len(sub) >= 2:
            groups.append(sub)
            labels.append(f"fold{int(f)}")

    if not groups: return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot(groups, vert=True, patch_artist=False, showmeans=True, meanline=True)
    # Overlay jitter scatter plot (limited points)
    m = CONFIG.get("JITTER_MAX_POINTS", 300)
    for i, g in enumerate(groups, start=1):
        g = np.array(g)
        if len(g) > m:
            idx = np.linspace(0, len(g)-1, m, dtype=int)
            g = g[idx]
        xj = np.random.normal(loc=i, scale=0.05, size=len(g))
        ax.scatter(xj, g, s=10, alpha=0.5)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel("Val. R²"); ax.set_title("5-fold val. R² (per-epoch)")
    ax.grid(True)
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_box_per_epoch"), dpi=400)
    plt.close(fig)

def _bootstrap_r2_for_fold(df_pred: pd.DataFrame, n_boot: int, rng: np.random.RandomState):
    """Perform bootstrap sampling on the validation set for a single fold, returning an R² list."""
    df = df_pred[["true","pred"]].dropna()
    y = df["true"].values; yhat = df["pred"].values
    n = len(y)
    if n < 3:  # Too few to provide a stable estimate
        return []
    idx = np.arange(n)
    r2s = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        r2s.append(r2_score(y[samp], yhat[samp]))
    return r2s

def plot_cv_box_bootstrap(val_fold_files, outdir: str, n_boot: int, seed: int):
    """
    Use bootstrap sampling within each fold to obtain the R² distribution for each fold → 5 bins.
Simultaneously assign fixed colors and markers to the scatter plots for each fold's jitter, facilitating correspondence with the scatter plots.
    """
    if not val_fold_files: return
    rng = np.random.RandomState(seed)
    groups = []
    labels = []
    fold_ids = []

    for fp in sorted(val_fold_files, key=lambda x: (_infer_fold_idx(x) or 9999)):
        df = safe_read_csv(fp)
        if df.empty or not {"true","pred"}.issubset(df.columns):
            continue
        fold_idx = _infer_fold_idx(fp)
        r2s = _bootstrap_r2_for_fold(df, n_boot=n_boot, rng=rng)
        if len(r2s) >= 2:
            groups.append(np.array(r2s))
            labels.append(f"Fold{fold_idx if fold_idx is not None else '?'}")
            fold_ids.append(fold_idx)

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot(groups, vert=True, patch_artist=False, showmeans=True, meanline=True)
    # Overlay jitter scatter points (with a point limit), using colors/markers consistent with the scatter plot.
    m = CONFIG.get("JITTER_MAX_POINTS", 300)
    for i, (g, fold_idx) in enumerate(zip(groups, fold_ids), start=1):
        g = np.array(g)
        if len(g) > m:
            idx = np.linspace(0, len(g)-1, m, dtype=int)
            g = g[idx]
        xj = np.random.normal(loc=i, scale=0.05, size=len(g))
        color, marker = _style_for_fold(fold_idx, idx_fallback=i-1)
        ax.scatter(xj, g, s=8, alpha=0.35, color=color, marker=marker)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel("Val. R²")
    # ax.set_title(f"5-fold val. R² (bootstrap, n={n_boot})")
    ax.grid(True)
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_box_bootstrap"), dpi=400)
    plt.close(fig)

# ---------- Other images ----------
def plot_cv_bar(cv_csv: pd.DataFrame, outdir: str):
    """The bar chart displays the Val R² for each fold, with the dotted line representing the mean."""
    fold_col = _find_col(cv_csv, ["fold", "Fold"])
    r2_col   = _find_col(cv_csv, ["val_r2", "r2", "valR2", "Val_R2"])
    if fold_col is None or r2_col is None or cv_csv.empty:
        return
    folds = cv_csv[fold_col].astype(int).values
    vals  = cv_csv[r2_col].astype(float).values
    mean_r2 = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(x) for x in folds], vals, width=0.65)
    ax.axhline(mean_r2, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label=f"mean = {mean_r2:.3f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("Val. R²"); ax.set_title("5-fold val. R²")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_bar"), dpi=400)
    plt.close(fig)

def plot_cv_learning_curves(trainlog_items, outdir: str):
    """Training Log: Val R² curve per fold, Loss curve (train/val split + combined)"""
    if not trainlog_items: return
    df = pd.DataFrame(trainlog_items)
    if df.empty: return
    epoch_col = _find_col(df, ["epoch","Epoch"])
    fold_col  = _find_col(df, ["fold","Fold"])
    vcol      = _find_col(df, ["val_r2","valR2","Val_R2"])
    tloss_col = _find_col(df, ["train_loss","Train_Loss"])
    vloss_col = _find_col(df, ["val_loss","Val_Loss"])
    if fold_col is None or epoch_col is None: return

    # Val R²
    if vcol is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[vcol], linewidth=1.5, label=f"fold {int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val. R²"); ax.set_title("Val. R² by epoch (per fold)")
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_valR2_learning_curves"), dpi=400)
        plt.close(fig)

    # Train loss (single plot, 4:3 aspect ratio, vertical axis shows raw loss, linear scale)
    if tloss_col is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LOSS)
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[tloss_col], linewidth=1.2, label=f"fold {int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss")
        # ax.set_title("Train loss by epoch (per fold)")
        ax.set_yscale("linear")  # Use actual numerical values explicitly (without logarithmic transformation).
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_train_loss_learning_curves"), dpi=400)
        plt.close(fig)

    # Val loss (single plot, 4:3 aspect ratio, vertical axis represents raw loss, linear coordinates)
    if vloss_col is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LOSS)
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[vloss_col], linewidth=1.4, label=f"fold {int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss")
        # ax.set_title("Val loss by epoch (per fold)")
        ax.set_yscale("linear")  # Use actual numerical values explicitly (without logarithmic transformation).
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_val_loss_learning_curves"), dpi=400)
        plt.close(fig)

    # Combined Version: train/val same image (maintain original size configuration)
    if tloss_col is not None and vloss_col is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[tloss_col], alpha=0.8, linewidth=1.0, label=f"train f{int(f)}")
            ax.plot(sub[epoch_col], sub[vloss_col], alpha=0.95, linewidth=1.5, label=f"val f{int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss by epoch (per fold)")
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_loss_learning_curves_combined"), dpi=400)
        plt.close(fig)

def plot_cv_scatter_and_calibration(val_fold_files, outdir: str, n_bins: int = 10):
    """
    Scatter plots per fold (imposition) + Independent calibration curves per fold (single sheet)
    - cv_val_scatter_folds.png: Color/marker for each fold's scatter plot matches cv_r2_box_bootstrap
    - Subplots at 1:1 scale with unified x/y axis ranges (based on combined quantiles across all folds)
    """
    fps = sorted(val_fold_files, key=lambda x: (_infer_fold_idx(x) or 9999))
    if not fps:
        return

    # First, traverse once to obtain all fold true/pred values for unifying the coordinate range.
    all_true_list = []
    all_pred_list = []
    for fp in fps:
        df = safe_read_csv(fp)
        if df.empty or not {"true", "pred"}.issubset(df.columns):
            continue
        all_true_list.append(df["true"].values)
        all_pred_list.append(df["pred"].values)
    if not all_true_list:
        return
    all_true = np.concatenate(all_true_list)
    all_pred = np.concatenate(all_pred_list)
    global_xlim, global_ylim = _quantile_limits_xy(
        all_true, all_pred,
        qlo=0.01, qhi=0.99, pad_frac=0.03
    )

    n = len(fps)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * FIGSIZE_SCATTER_SUB, rows * FIGSIZE_SCATTER_SUB),
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for i, fp in enumerate(fps):
        df = safe_read_csv(fp)
        if df.empty or not {"true", "pred"}.issubset(df.columns):
            continue
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("on")

        fold_idx = _infer_fold_idx(fp)
        color, marker = _style_for_fold(fold_idx, idx_fallback=i)

        ax.scatter(df["true"], df["pred"], s=6, alpha=0.6, color=color, marker=marker)
        ax.plot(
            [global_xlim[0], global_xlim[1]],
            [global_ylim[0], global_ylim[1]],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="k",
        )
        ax.set_xlim(*global_xlim)
        ax.set_ylim(*global_ylim)
        ax.set_aspect("equal", adjustable="box")  # Subfigure 1:1 scale

        ax.tick_params(labelsize=19)

        # name = os.path.basename(fp).replace(".csv", "")
        fold_idx = _infer_fold_idx(fp)
        if fold_idx is None:
            # If the folder cannot be found, revert to the filename.
            title = os.path.basename(fp).replace(".csv", "")
        else:
            title = f"Fold {fold_idx}"
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True)

    # fig.suptitle("Val. scatter per fold", y=0.90, fontsize=21)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_dual(fig, os.path.join(outdir, "cv_val_scatter_folds"), dpi=400)
    plt.close(fig)

    # Independent calibration curves for each fold (tight version)
    for fp in fps:
        df = safe_read_csv(fp)
        if df.empty or not {"true","pred"}.issubset(df.columns):
            continue
        name = os.path.splitext(os.path.basename(fp))[0]
        calibration_curve_tight(
            df, n_bins=n_bins,
            out_basepath=os.path.join(outdir, f"{name}_calibration"),
            title=f"Calibration: {name}",
            q_limits=(0.01, 0.99), pad_frac=0.03, min_per_bin=10
        )

# ---------- Main Process ----------
def main():
    # Output root directory: Script directory's peer level result/plot/<SAVE_SUBDIR>/
    script_dir = Path(__file__).resolve().parent
    save_subdir = CONFIG.get("SAVE_SUBDIR", "5foldplot")
    outdir = ensure_dir(script_dir.parent / "result" / "plot" / save_subdir)

    run_dir = CONFIG.get("RUN_DIR", "").strip()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("Please specify the output directory path for the main program's current run in CONFIG['RUN_DIR'] (including cv and all fold CSV files).")

    files = _auto_files(run_dir)

    # --- Column Chart + (Single-Box) Aggregated Box Plot ---
    cv_csv = safe_read_csv(files["cv_summary"])
    if cv_csv.empty:
        cv_csv = _fallback_cv_summary_from_preds(files["val_fold_files"],
                                                 save_to=os.path.join(outdir, "cv_summary_from_preds.csv"))
    if not cv_csv.empty:
        plot_cv_bar(cv_csv, outdir)
        plot_cv_box_aggregate(cv_csv, outdir)

    # --- Learning Curve (Includes: Val R²; Train loss; Val loss; Combined version) ---
    trainlog_items = read_json_or_jsonl(files["training_log"])
    if trainlog_items:
        plot_cv_learning_curves(trainlog_items, outdir)

    # --- Verification Scatter & Calibration per Fold ---
    if files["val_fold_files"]:
        plot_cv_scatter_and_calibration(files["val_fold_files"], outdir, n_bins=int(CONFIG.get("CALIB_BINS", 10)))

    # --- R² 5 boxes: Select the optimal available mode based on configuration ---
    mode = (CONFIG.get("R2_BOX_MODE") or "bootstrap").lower()
    if mode == "bootstrap" and files["val_fold_files"]:
        plot_cv_box_bootstrap(
            files["val_fold_files"], outdir,
            n_boot=int(CONFIG.get("BOOT_N", 1000)),
            seed=int(CONFIG.get("BOOT_SEED", 20251015))
        )
    elif mode == "per_epoch" and trainlog_items:
        plot_cv_box_per_epoch(trainlog_items, outdir)
    else:
        # If data for the selected mode is insufficient, automatically attempt another available mode.
        if files["val_fold_files"]:
            plot_cv_box_bootstrap(
                files["val_fold_files"], outdir,
                n_boot=int(CONFIG.get("BOOT_N", 1000)),
                seed=int(CONFIG.get("BOOT_SEED", 20251015))
            )
        elif trainlog_items:
            plot_cv_box_per_epoch(trainlog_items, outdir)
        # If none are available, an existing aggregate version can serve as an alternative.

    # --- (Optional) Learning Rate Curve per Fold ---
    if files["lr_fold_files"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for fp in files["lr_fold_files"]:
            df = safe_read_csv(fp)
            if df.empty or "epoch" not in df.columns or "lr" not in df.columns:
                continue
            name = os.path.basename(fp).replace(".csv","")
            ax.plot(df["epoch"], df["lr"], linewidth=1.3, label=name.split("_")[-1])
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning rate per fold")
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_lr_schedules"), dpi=400)
        plt.close(fig)

    print(f"[OK] CV Image generated：{outdir}")

if __name__ == "__main__":
    main()
