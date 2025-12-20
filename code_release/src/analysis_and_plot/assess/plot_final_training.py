# -*- coding: utf-8 -*-
"""
plot_final_training.py

Purpose:
  Generate charts required for the paper from output files of the "final training" phase, saving them uniformly as PNG and SVG files (dpi=400).

Input files (manually specify the directory EXP_DIR below in INPUT):
  - training_curve_final.csv                # Train/val metrics per epoch
  - learning_rate_schedule_final.csv        # Learning rate per epoch
  - val_predictions_final.csv               # Optimal model predictions on validation set (if successfully saved by main programme)
  - final_test_predictions.csv              # Optimal model predictions on test set (includes sequence/true/pred)
  - final_test_metrics.json                 # Aggregated test set metrics (if available)

Output directory structure (automatically generated):
  <project root>/result/plot/finaltrain_plot/<timestamp>/*.png|*.svg
  Additionally exports selected intermediate statistics CSV files (e.g., bin calibration, error-sequence length analysis).

Note:
  1) Do not use command-line arguments; manually specify EXP_DIR within INPUT.
  2) Loss curves for train and val are plotted separately (as per your previous requirement).
  3) Calibration plots default to 10 equally spaced bins, adjustable via CONFIG.
"""

import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============== Users must manually specify ==============
INPUT = {
    # Please replace this section with the output directory for your "final training phase" round of experiments (including files such as training_curve_final.csv).
    "EXP_DIR": r"Path\To\Your\Training\Output\Directory"
}
# ==========================================


# ============== General Configuration ==============
CONFIG = {
    "dpi": 400,
    "deciles": 20,             # Calibration charts and length bins are divided into 10 segments by default.
    "scatter_alpha": 0.6,
    "figsize": (6, 4.5),
    "bins_hist": 30            # Number of residual histogram bars
}
# =====================================


def _project_root_from_script() -> str:
    """The directory immediately above the script's location serves as the project root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def _ensure_outdir(subname: str = "finaltrain_plot") -> str:
    """
    Create a directory named `finaltrain_plot/<timestamp>` under `<project root>/result/plot/`.
    Return the path to this timestamp directory.
    """
    project_root = _project_root_from_script()
    base = os.path.join(project_root, "result", "plot", subname)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _save_figure(fig: plt.Figure, outdir: str, name: str, dpi: int):
    """Simultaneously save as PNG and SVG (dpi=400)."""
    png_path = os.path.join(outdir, f"{name}.png")
    svg_path = os.path.join(outdir, f"{name}.svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[Warning] File does not exist, skipping:{path}")
    return None


def _safe_read_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[Warning] File does not exist, skipping:{path}")
    return None


def plot_losses_separate(curve: pd.DataFrame, outdir: str, dpi: int):
    """Plot train_loss and val_loss separately (single graph, single curve)."""
    if not {"epoch", "train_loss", "val_loss"}.issubset(curve.columns):
        print("[Skip] training_curve_final.csv Missing required columns：epoch/train_loss/val_loss")
        return

    # Train loss（Single image）
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(curve["epoch"], curve["train_loss"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title("Final Training — Train Loss per Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_train_loss_per_epoch", dpi)

    # Val loss（Single image）
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(curve["epoch"], curve["val_loss"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_title("Final Training — Validation Loss per Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_val_loss_per_epoch", dpi)


def plot_val_metrics(curve: pd.DataFrame, outdir: str, dpi: int):
    """Plotting various metrics for the validation set（R2、MSE、correlation coefficient）。"""
    # R2
    if {"epoch", "val_r2"}.issubset(curve.columns):
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        ax.plot(curve["epoch"], curve["val_r2"], marker="o", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation R²")
        ax.set_title("Final Training — Validation R² per Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save_figure(fig, outdir, "final_val_r2_per_epoch", dpi)

    # MSE
    if {"epoch", "val_mse"}.issubset(curve.columns):
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        ax.plot(curve["epoch"], curve["val_mse"], marker="o", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.set_title("Final Training — Validation MSE per Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save_figure(fig, outdir, "final_val_mse_per_epoch", dpi)

    # Pearson / Spearman
    has_pearson = {"epoch", "val_pearson"}.issubset(curve.columns)
    has_spearman = {"epoch", "val_spearman"}.issubset(curve.columns)
    if has_pearson or has_spearman:
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        if has_pearson:
            ax.plot(curve["epoch"], curve["val_pearson"], marker="o", linewidth=1.5, alpha=0.9, label="Pearson")
        if has_spearman:
            ax.plot(curve["epoch"], curve["val_spearman"], marker="s", linewidth=1.5, alpha=0.9, label="Spearman")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Correlation")
        ax.set_title("Final Training — Validation Correlations per Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        _save_figure(fig, outdir, "final_val_correlations_per_epoch", dpi)


def plot_lr_schedule(lr_df: pd.DataFrame, outdir: str, dpi: int):
    if not {"epoch", "lr"}.issubset(lr_df.columns):
        print("[Skip] learning_rate_schedule_final.csv Missing column epoch/lr")
        return
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(lr_df["epoch"], lr_df["lr"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Final Training — Learning Rate Schedule")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_learning_rate_schedule", dpi)


def _scatter_parity(true_y: np.ndarray, pred_y: np.ndarray, title: str, outpath_prefix: str, dpi: int):
    """General: Actual vs Forecast Scatter Plot + y=x Reference Line + Basic Statistics."""
    # Statistics
    resid = pred_y - true_y
    mae = np.mean(np.abs(resid))
    rmse = math.sqrt(np.mean(resid**2))
    r2 = 1.0 - np.sum((true_y - pred_y) ** 2) / np.sum((true_y - np.mean(true_y)) ** 2)

    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.scatter(true_y, pred_y, s=12, alpha=CONFIG["scatter_alpha"])
    lo = min(np.min(true_y), np.min(pred_y))
    hi = max(np.max(true_y), np.max(pred_y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)  # y=x
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    # Corner annotation
    ax.text(0.04, 0.96, f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, os.path.dirname(outpath_prefix), os.path.basename(outpath_prefix), dpi)


def plot_val_parity(val_pred_csv: str, outdir: str, dpi: int):
    df = _safe_read_csv(val_pred_csv)
    if df is None:
        return
    # Compatible with both cases: containing sequence and not containing sequence
    needed = {"true", "pred"}
    if not needed.issubset(df.columns):
        print(f"[Skip] {val_pred_csv} Excludes the true/pred column")
        return
    _scatter_parity(
        df["true"].to_numpy(dtype=float),
        df["pred"].to_numpy(dtype=float),
        "Final Training — Validation Parity (True vs Predicted)",
        os.path.join(outdir, "final_val_parity"),
        dpi
    )


def plot_test_parity_and_residuals(test_pred_csv: str, outdir: str, dpi: int):
    df = _safe_read_csv(test_pred_csv)
    if df is None:
        return
    if not {"true", "pred"}.issubset(df.columns):
        print(f"[Skip] {test_pred_csv} Excludes the true/pred column")
        return

    # 1) Parity
    _scatter_parity(
        df["true"].to_numpy(dtype=float),
        df["pred"].to_numpy(dtype=float),
        "Final Training — Test Parity (True vs Predicted)",
        os.path.join(outdir, "final_test_parity"),
        dpi
    )

    # 2) Residual histogram (consistent with the definition of residual in the main programme):true - pred）
    df["residual"] = df["true"].astype(float) - df["pred"].astype(float)
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.hist(df["residual"].to_numpy(), bins=CONFIG["bins_hist"])
    ax.set_xlabel("Residual (True - Pred)")
    ax.set_ylabel("Count")
    ax.set_title("Final Training — Test Residuals")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_test_residual_hist", dpi)

    # 3) Error-Sequence Length Relationship (if sequence exists)
    if "sequence" in df.columns:
        df["seq_len"] = df["sequence"].astype(str).map(len)
        df["abs_error"] = np.abs(df["residual"])
        # Partition into equal-sized bins (default 10)
        try:
            df["len_bin"] = pd.qcut(df["seq_len"], q=CONFIG["deciles"], duplicates="drop")
        except ValueError:
            # Insufficient sample size or length repetition resulted in an inability to qcut，Then revert to equal-width boxes.
            df["len_bin"] = pd.cut(df["seq_len"], bins=CONFIG["deciles"])
        grp = df.groupby("len_bin", observed=True).agg(
            mean_len=("seq_len", "mean"),
            mean_abs_err=("abs_error", "mean"),
            count=("abs_error", "size")
        ).reset_index(drop=True)
        # Save table
        grp.to_csv(os.path.join(outdir, "final_test_error_vs_length.csv"), index=False)

        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        ax.plot(grp["mean_len"], grp["mean_abs_err"], marker="o", linewidth=1.5)
        for x, y, n in zip(grp["mean_len"], grp["mean_abs_err"], grp["count"]):
            ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Sequence Length (bin mean)")
        ax.set_ylabel("Mean |Error|")
        ax.set_title("Final Training — Test Error vs Sequence Length")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save_figure(fig, outdir, "final_test_error_vs_length", dpi)

    # 4) 测试集真实值分布
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.hist(df["true"].to_numpy(dtype=float), bins=CONFIG["bins_hist"])
    ax.set_xlabel("True Half-life")
    ax.set_ylabel("Count")
    ax.set_title("Final Training — Test True Distribution")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_test_true_distribution", dpi)


def plot_test_calibration(test_pred_csv: str, outdir: str, dpi: int):
    """Calibration curve based on 10 equal-partition bins of the test set: x = bin average prediction, y = bin average actual, reference line y = x."""
    df = _safe_read_csv(test_pred_csv)
    if df is None:
        return
    if not {"true", "pred"}.issubset(df.columns):
        print(f"[Skip] {test_pred_csv} Excludes the true/pred column")
        return

    # 10-partition box (adjustable in CONFIG)
    try:
        df["bin"] = pd.qcut(df["pred"].astype(float), q=CONFIG["deciles"], labels=False, duplicates="drop")
    except ValueError:
        print("[Presentation] When the sample size is too small or predicted values cluster, switch to equal-width binning.")
        df["bin"] = pd.cut(df["pred"].astype(float), bins=CONFIG["deciles"], labels=False, include_lowest=True)

    calib = df.groupby("bin", observed=True).agg(
        pred_mean=("pred", "mean"),
        true_mean=("true", "mean"),
        count=("true", "size"),
        mae=("pred", lambda x: np.mean(np.abs(x.to_numpy() - df.loc[x.index, "true"].to_numpy())))
    ).reset_index(drop=True)

    # Prevent division by zero
    true_vals = df["true"].astype(float).to_numpy()
    pred_vals = df["pred"].astype(float).to_numpy()
    eps = 1e-12
    calib["mape"] = df.groupby("bin", observed=True).apply(
        lambda g: float(np.mean(np.abs((g["pred"] - g["true"]) / (g["true"] + eps))))
    ).reset_index(drop=True)

    calib.to_csv(os.path.join(outdir, "final_test_calibration_deciles.csv"), index=False)

    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(calib["pred_mean"], calib["true_mean"], marker="o", linewidth=1.5)
    lo = float(min(calib["pred_mean"].min(), calib["true_mean"].min()))
    hi = float(max(calib["pred_mean"].max(), calib["true_mean"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)  # y=x
    for x, y, n in zip(calib["pred_mean"], calib["true_mean"], calib["count"]):
        ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Bin Mean Prediction")
    ax.set_ylabel("Bin Mean True")
    ax.set_title(f"Final Training — Test Calibration ({CONFIG['deciles']} bins)")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_test_calibration_deciles", dpi)


def main():
    exp_dir = os.path.normpath(INPUT["EXP_DIR"])
    if not os.path.isdir(exp_dir):
        raise NotADirectoryError(f"EXP_DIR Does not exist or is not a folder:{exp_dir}")

    outdir = _ensure_outdir("finaltrain_plot")
    print(f"[Output Directory] {outdir}")

    # -------- Read file path --------
    curve_csv = os.path.join(exp_dir, "training_curve_final.csv")
    lr_csv    = os.path.join(exp_dir, "learning_rate_schedule_final.csv")
    val_csv   = os.path.join(exp_dir, "val_predictions_final.csv")
    test_csv  = os.path.join(exp_dir, "final_test_predictions.csv")
    test_json = os.path.join(exp_dir, "final_test_metrics.json")  # Optional

    # -------- Training/Validation Curve --------
    curve_df = _safe_read_csv(curve_csv)
    if curve_df is not None and "epoch" in curve_df.columns:
        plot_losses_separate(curve_df, outdir, CONFIG["dpi"])
        plot_val_metrics(curve_df, outdir, CONFIG["dpi"])
    else:
        print("[Skip] Unable to plot loss/val metric curves (missing or no epoch column)")

    # -------- Learning rate curve --------
    lr_df = _safe_read_csv(lr_csv)
    if lr_df is not None:
        plot_lr_schedule(lr_df, outdir, CONFIG["dpi"])

    # -------- Validation Set Parity (if applicable)--------
    if os.path.exists(val_csv):
        plot_val_parity(val_csv, outdir, CONFIG["dpi"])

    # -------- Test Set Parity/Residual/Distribution/Length Error --------
    if os.path.exists(test_csv):
        plot_test_parity_and_residuals(test_csv, outdir, CONFIG["dpi"])
        plot_test_calibration(test_csv, outdir, CONFIG["dpi"])
    else:
        print("[Skip] final_test_predictions.csv was not found; the test set correlation plot cannot be generated.")

    # -------- Record test metrics JSON (if present)--------
    metrics = _safe_read_json(test_json)
    if metrics:
        # Generate a simple metric bar chart (for easy comparison in the paper)
        keys = ["test_r2", "test_pearson", "test_spearman"]
        present = [k for k in keys if k in metrics]
        if present:
            fig, ax = plt.subplots(figsize=CONFIG["figsize"])
            ax.bar(present, [metrics[k] for k in present])
            ax.set_title("Final Training — Test Summary Metrics")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            _save_figure(fig, outdir, "final_test_summary_metrics_bar", CONFIG["dpi"])

    print("[Completed] The complete training phase chart has been output.。")


if __name__ == "__main__":
    main()
