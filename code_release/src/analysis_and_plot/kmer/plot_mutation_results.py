#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_mutation_results.py

Usage:
1) Modify MUTATION_CSV and OUT_DIR in CONFIG
2) Run: python analyze_mutation_results.py
Output: CSV summary + figures (PNG≥400dpi + SVG)

Statistical metrics:
- Main effects measured in "Δ per sample mean" (averaged across multiple sites per transcript), then statistically analyzed and tested by (motif, new) and by motif.
- Significance: Wilcoxon signed-rank test (Δ vs 0), multiple correction: Benjamini–Hochberg (FDR).
- Position effects: Δ vs relative position (motif center / sequence length), equal-width binning.
"""

import os, json, math, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import wilcoxon
import matplotlib
import scienceplots
import matplotlib.gridspec as gridspec

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

# ============== Placement ==============
@dataclass
class CONFIG:
    # The path to your mutation_results.csv file (output from run_interpretability.py)
    MUTATION_CSV: str = r"Path\to\mutation_results.csv"
    # Output Directory (Automatically Created)
    OUT_DIR: str = r"Path\to\mutation_plot"
    # Top-KPlot after sorting by |mean_delta|
    TOPK: int = 18
    # Number of Positioned Boxes
    N_POS_BINS: int = 20
    # bootstrap Number of times
    N_BOOT: int = 2000
    # PNG Resolution & Save SVG
    DPI: int = 400
    SAVE_SVG: bool = True
    # Maximum number of columns per row when drawing a polyhedron diagram
    NCOLS: int = 4
    # Random seed（bootstrap Sampling）
    SEED: int = 42
    # Relative change threshold, e.g., 0.10 = Δt/t0 > 10%, is recorded as "strong response."”
    BIG_FRAC_THRESH: float = 0.10
    # Number of bins in baseline half-life stratification (equally divided by base_pred quantiles)
    N_T0_BINS: int = 3
    # residuals.csv Path; if left blank, defaults to residuals.csv in the same directory as MUTATION_CSV.
    RESIDUALS_CSV: str = ""
    # Whether to perform separate statistics for samples with smaller residuals
    USE_RESIDUAL_FILTER: bool = True
    # residual Absolute value threshold; automatically takes the median of abs_residual when <=0
    RESIDUAL_ABS_THRESH: float = 0.0

CFG = CONFIG()

# ============== Utility functions ==============
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def bootstrap_ci(x, n_boot=2000, ci=95, seed=2025):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(np.mean(x[idx]))
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(lo), float(hi)

def bh_fdr(pvals):
    """Benjamini–Hochberg FDR correction. Returns q-values (np.array) of the same length as pvals."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n-1, -1, -1):
        val = ranked[i] * n / (i+1)
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = q
    return out

def safe_wilcoxon_zero(x):
    """Δ vs 0 Symbol rank test; returns the p-value. Returns 1.0 in case of an exception."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.allclose(x, 0):
        return 1.0
    try:
        stat, p = wilcoxon(x, alternative='two-sided', zero_method='wilcox')
        return float(p)
    except Exception:
        return 1.0

def savefig(fig, path_base):
    png = f"{path_base}.png"
    fig.savefig(png, dpi=CFG.DPI, bbox_inches='tight')
    if CFG.SAVE_SVG:
        svg = f"{path_base}.svg"
        fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)

def compute_effect_stats(sub: pd.DataFrame):
    """
    Perform uniform statistical calculations (including mean, median, quantiles, proportions, etc.)
    on a batch of per-sequence records (containing delta and base_pred).
    """
    n = int(len(sub))
    if n == 0:
        return {
            "n_samples": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "q90_delta": np.nan,
            "ci95_low_delta": np.nan,
            "ci95_high_delta": np.nan,
            "wilcoxon_p": np.nan,
            "mean_frac_delta": np.nan,
            "median_frac_delta": np.nan,
            "q90_frac_delta": np.nan,
            "prop_delta_pos": np.nan,
            "prop_frac_gt_big": np.nan,
            "prop_frac_lt_-0.1": np.nan,
        }

    # Absolute Δ
    arr = sub["delta"].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        mean = median = q90 = lo = hi = p = prop_pos = np.nan
    else:
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        q90 = float(np.quantile(arr, 0.9))
        lo, hi = bootstrap_ci(arr, n_boot=CFG.N_BOOT, seed=CFG.SEED)
        p = safe_wilcoxon_zero(arr)
        prop_pos = float(np.mean(arr > 0))

    # Relative change Δ / base_pred
    frac_arr = (sub["delta"] / sub["base_pred"]).replace(
        [np.inf, -np.inf], np.nan
    ).to_numpy(dtype=float)
    frac_arr = frac_arr[np.isfinite(frac_arr)]
    if frac_arr.size == 0:
        mean_frac = median_frac = q90_frac = prop_big = prop_neg_big = np.nan
    else:
        mean_frac = float(np.mean(frac_arr))
        median_frac = float(np.median(frac_arr))
        q90_frac = float(np.quantile(frac_arr, 0.9))
        # Δ/t0 > +BIG_FRAC_THRESH ratio (strong elevation)
        prop_big = float(np.mean(frac_arr > CFG.BIG_FRAC_THRESH))
        # Δ/t0 < -BIG_FRAC_THRESH ratio (strong reduction)
        prop_neg_big = float(np.mean(frac_arr < -CFG.BIG_FRAC_THRESH))

    return {
        "n_samples": n,
        "mean_delta": mean,
        "median_delta": median,
        "q90_delta": q90,
        "ci95_low_delta": lo,
        "ci95_high_delta": hi,
        "wilcoxon_p": p,
        "mean_frac_delta": mean_frac,
        "median_frac_delta": median_frac,
        "q90_frac_delta": q90_frac,
        "prop_delta_pos": prop_pos,
        "prop_frac_gt_big": prop_big,
        "prop_frac_lt_-0.1": prop_neg_big,
    }


def select_motifs_for_plot(df_motif: pd.DataFrame, topk: int):
    """
    Unify the selection of motifs to be displayed across various plots:
    1) First, select the Top-K motifs by descending order of |mean_delta|
    2) Within this Top-K set, sort by descending order of mean_delta
    """
    if df_motif is None or df_motif.empty:
        return []

    if topk is None or topk <= 0:
        # No quantity restrictions; include all motifs. Sort them directly by mean_delta in descending order.
        tmp = df_motif.copy()
        tmp = tmp.sort_values("mean_delta", ascending=False)
        return tmp["motif"].tolist()

    tmp = df_motif.copy()
    tmp = tmp.assign(
        rank_abs=lambda d: d["mean_delta"].abs().rank(
            method="first", ascending=False
        )
    )
    # First, retrieve the Top-K based on rank_abs.
    top = tmp.sort_values("rank_abs").head(topk)
    # Sort the Top-K entries in descending order by mean_delta to determine the final display sequence.
    top = top.sort_values("mean_delta", ascending=False)
    motifs = top["motif"].tolist()
    return motifs


# ============== Main Process ==============
def plot_motif_triptych_c1(df_motif: pd.DataFrame,
                           per_seq: pd.DataFrame,
                           df_pos: pd.DataFrame,
                           motifs_sorted,
                           out_dir: str):
    """Triple-panel Figure Version 1 (using Scheme C1):
    (a) Left: Horizontal bar chart—mean effect + 95% CI for each motif (sorted from positive to negative by mean_delta)
    (b) Middle: Position effect—one row per motif, Δ vs. relative position (12×1 scatter plot grid)
    (c) Right: Boxplot + jittered points — Per-sample Δ distribution for each motif (colors correspond to (a)/(b))
    """
    motifs_sorted = list(motifs_sorted) if motifs_sorted is not None else []
    if len(motifs_sorted) == 0:
        return
    if df_motif.empty or per_seq.empty or df_pos.empty:
        return

    # First filter once using the original motifs_sorted to ensure only the motifs you care about are retained.
    motifs_candidate = [m for m in motifs_sorted if m in df_motif["motif"].values]
    if not motifs_candidate:
        return

    # Extract these motifs from df_motif, corresponding to their rows, and sort them in descending order by mean_delta.
    sub_motif = (
        df_motif[df_motif["motif"].isin(motifs_candidate)]
        .set_index("motif")
        .loc[motifs_candidate]          # Retrieve in the original sorted order of motifs_sorted
        .reset_index()
    )

    if "mean_delta" not in sub_motif.columns:
        print("[WARN] df_motif The mean_delta column is missing, preventing sorting. motif")
        return

    motifs = sub_motif["motif"].tolist()

    # Color: Δ > 0 Orange, Δ < 0 Green (shared scatter plot for panel1 / panel3 / panel2)
    mu = sub_motif["mean_delta"].to_numpy(dtype=float)
    colors = np.where(mu >= 0, "#d95f02", "#1b9e77")
    color_map = {m: col for m, col in zip(motifs, colors)}

    # -------- Prepare per-sample Δ data for box plots (in the same order as the motifs) --------
    data_per_motif = []
    for m in motifs:
        vals = per_seq.loc[per_seq["motif"] == m, "delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            data_per_motif.append(np.array([np.nan]))
        else:
            data_per_motif.append(vals)

    # Box Plot Y-axis Range: Percentiles based on per-sample Δ
    valid_arrays = [v for v in data_per_motif if np.isfinite(v).any()]
    if valid_arrays:
        all_vals = np.concatenate(valid_arrays)
        q_lo, q_hi = np.nanquantile(all_vals, [0.02, 0.98])
        span = max(1e-12, q_hi - q_lo)
        y_lo = q_lo - 0.1 * span
        y_hi = q_hi + 0.1 * span
    else:
        y_lo, y_hi = -0.1, 0.1

    # -------- List of motifs in the positional effect subgraph (in the same order as the motifs) --------
    motifs_pos = [m for m in motifs if m in df_pos["motif"].values]
    if not motifs_pos:
        return
    n_pos = len(motifs_pos)

    # ===== Overall figure: 3 columns, (a) left, (b) center, (c) right =====
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.1, 1.8, 1.3],   # Center position effect panel slightly wider
        wspace=0.35,
        figure=fig
    )

    # ---------- (a) Left：bar, motif mean Δ ----------
    ax_bar = fig.add_subplot(gs[0, 0])
    idxs = np.arange(len(sub_motif))
    err = np.vstack([
        mu - sub_motif["ci95_low_delta"].to_numpy(dtype=float),
        sub_motif["ci95_high_delta"].to_numpy(dtype=float) - mu,
    ])
    bars = ax_bar.barh(idxs, mu, xerr=err, align="center", alpha=0.9)

    for bar, col in zip(bars, colors):
        bar.set_color(col)

    # Y-axis label: Significant increase *
    labels = []
    for _, row in sub_motif.iterrows():
        label = row["motif"]
        if "fdr" in sub_motif.columns and not pd.isna(row.get("fdr", np.nan)) and row["fdr"] <= 0.05:
            label += " *"
        labels.append(label)
    ax_bar.set_yticks(idxs)
    ax_bar.set_yticklabels(labels)

    ax_bar.axvline(0, color="k", lw=0.8)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.set_xlabel("Mean Δprediction (mut - base)")
    # ax_bar.set_title("Mean effect per motif")
    ax_bar.invert_yaxis()  # The motif with index 0 (maximum mean_delta) is at the top.
    ax_bar.text(0.02, 0.98, "(a)", transform=ax_bar.transAxes,
                ha="left", va="top", fontsize=18, fontweight="bold")

    # ---------- (c) Right: Box plot + outliers ----------
    ax_box = fig.add_subplot(gs[0, 2])
    positions = np.arange(1, len(motifs) + 1)

    bp = ax_box.boxplot(
        data_per_motif,
        positions=positions,
        widths=0.6,
        vert=True,
        showfliers=False,
        patch_artist=True,
    )

    # Color the box (matching the color of panel1)
    for box, m in zip(bp["boxes"], motifs):
        box.set_facecolor(color_map.get(m, "#cccccc"))
        box.set_edgecolor("black")
        box.set_alpha(0.7)

    # Median line thickened
    for med in bp["medians"]:
        med.set_linewidth(2.0)

    # Jitter scatter: Color matches the bar/position line color of this motif
    max_points = 400
    rng = np.random.RandomState(20251120)
    for pos, m, vals in zip(positions, motifs, data_per_motif):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > max_points:
            idx = rng.choice(vals.size, size=max_points, replace=False)
            vals = vals[idx]
        x_jitter = rng.normal(loc=pos, scale=0.06, size=vals.size)
        ax_box.scatter(
            x_jitter,
            vals,
            s=6,
            alpha=0.25,
            linewidths=0,
            color=color_map.get(m, "black"),
        )

    ax_box.axhline(0, color="k", lw=0.8)
    ax_box.set_xlim(0.5, len(motifs) + 0.5)
    ax_box.set_ylim(y_lo, y_hi)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(motifs, rotation=45, ha="right", fontsize=11, va="top")
    ax_box.set_ylabel("Δprediction (per sample)")
    # ax_box.set_title("Per-sample Δ distribution")
    ax_box.grid(axis="y", linestyle="--", alpha=0.4)
    ax_box.text(0.02, 0.98, "(c)", transform=ax_box.transAxes,
                ha="left", va="top", fontsize=18, fontweight="bold")

    # ---------- (b) Position effect, motif: one broken line per row ----------
    nrows_pos = n_pos
    gs_pos = gridspec.GridSpecFromSubplotSpec(
        nrows_pos, 1, subplot_spec=gs[0, 1], hspace=0.05
    )

    shared_ax = None
    for i, m in enumerate(motifs_pos):
        ax = fig.add_subplot(gs_pos[i, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        sub = df_pos[df_pos["motif"] == m].sort_values("pos_rel_center")
        if sub.empty:
            ax.axis("off")
            continue

        col = color_map.get(m, "#1f77b4")
        ax.plot(sub["pos_rel_center"], sub["mean_delta"], marker="o", lw=1.0, color=col)
        ax.fill_between(
            sub["pos_rel_center"],
            sub["ci95_low_delta"],
            sub["ci95_high_delta"],
            alpha=0.25,
            color=col,
        )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(0, 1)

        # Draw the x-axis labels on the bottom row
        if i == nrows_pos - 1:
            ax.set_xlabel("Relative position in 3'UTR")
        else:
            ax.set_xticklabels([])

        # The y-axis displays only the 0 tick mark without numerical labels.
        ax.set_yticks([0])
        ax.set_yticklabels([])

        # Add the label "panel (b)" to the topmost row.
        if i == 0:
            ax.text(0.02, 0.98, "(b)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=18, fontweight="bold")

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "motif_effect_triptych_C1"))



def plot_motif_triptych_c2(df_motif: pd.DataFrame,
                           per_seq: pd.DataFrame,
                           df_pos: pd.DataFrame,
                           motifs_sorted,
                           out_dir: str):
    """Triple-Plot Version 2: Position Effects with Motif×Position Heatmap (C2)
(a) Horizontal bar chart: Mean effect + 95% CI for each motif
(b) Boxplot + jittered points: Per-sample Δ distribution for each motif
(c) Position effects: Matrix heatmap of motif×position_bin
    """
    motifs_sorted = list(motifs_sorted) if motifs_sorted is not None else []
    if len(motifs_sorted) == 0:
        return
    if df_motif.empty or per_seq.empty or df_pos.empty:
        return

    motifs = [m for m in motifs_sorted if m in df_motif["motif"].values]
    if not motifs:
        return

    sub_motif = df_motif[df_motif["motif"].isin(motifs)].set_index("motif").loc[motifs].reset_index()

    # Color mapping (consistent with C1)
    mu = sub_motif["mean_delta"].to_numpy(dtype=float)
    colors = np.where(mu >= 0, "#d95f02", "#1b9e77")
    color_map = {m: col for m, col in zip(sub_motif["motif"], colors)}

    # per-sample Δ Distribution
    data_per_motif = []
    for m in motifs:
        vals = per_seq.loc[per_seq["motif"] == m, "delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            data_per_motif.append(np.array([np.nan]))
        else:
            data_per_motif.append(vals)

    all_vals = np.concatenate([v for v in data_per_motif if np.isfinite(v).any()])
    if all_vals.size == 0:
        y_lo, y_hi = -0.1, 0.1
    else:
        q_lo, q_hi = np.nanquantile(all_vals, [0.02, 0.98])
        span = max(1e-12, q_hi - q_lo)
        y_lo = q_lo - 0.1 * span
        y_hi = q_hi + 0.1 * span

    # Location Effect Heatmap Data: By motif×pos_bin pivot
    df_heat = df_pos[df_pos["motif"].isin(motifs)].copy()
    if df_heat.empty:
        return
    # Ensure pos_bin is numeric and ordered.
    if "pos_bin" in df_heat.columns:
        df_heat["pos_bin"] = df_heat["pos_bin"].astype(int)
        bins_sorted = sorted(df_heat["pos_bin"].unique())
    else:
        # If pos_bin is not available, sort by pos_rel_center and assign manual numbers.
        df_heat = df_heat.sort_values("pos_rel_center")
        df_heat["pos_bin"] = pd.factorize(df_heat["pos_rel_center"])[0]
        bins_sorted = sorted(df_heat["pos_bin"].unique())

    heat_table = (
        df_heat
        .pivot_table(index="motif", columns="pos_bin", values="mean_delta", aggfunc="mean")
        .reindex(index=motifs, columns=bins_sorted)
    )
    heat_vals = heat_table.to_numpy(dtype=float)

    # Symmetrical Color Axis
    if np.all(np.isnan(heat_vals)):
        v_max = 1.0
    else:
        v_max = float(np.nanmax(np.abs(heat_vals)))
        if not np.isfinite(v_max) or v_max == 0:
            v_max = 1.0

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.3, 1.6], wspace=0.35, figure=fig)

    # ---------- (a) bar ----------
    ax_bar = fig.add_subplot(gs[0, 0])
    idxs = np.arange(len(sub_motif))
    err = np.vstack([
        mu - sub_motif["ci95_low_delta"].to_numpy(dtype=float),
        sub_motif["ci95_high_delta"].to_numpy(dtype=float) - mu,
    ])
    bars = ax_bar.barh(idxs, mu, xerr=err, align="center", alpha=0.9)
    for bar, col in zip(bars, colors):
        bar.set_color(col)
    labels = []
    for _, row in sub_motif.iterrows():
        label = row["motif"]
        if "fdr" in sub_motif.columns and not pd.isna(row.get("fdr", np.nan)) and row["fdr"] <= 0.05:
            label += " *"
        labels.append(label)
    ax_bar.set_yticks(idxs)
    ax_bar.set_yticklabels(labels)
    ax_bar.axvline(0, color="k", lw=0.8)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.set_xlabel("Mean Δprediction (mut - base)")
    ax_bar.set_title("Mean effect per motif")
    ax_bar.invert_yaxis()
    ax_bar.text(0.02, 0.98, "(a)", transform=ax_bar.transAxes,
                ha="left", va="top", fontsize=18, fontweight="bold")

    # ---------- (b) box + jitter ----------
    ax_box = fig.add_subplot(gs[0, 1])
    positions = np.arange(1, len(motifs) + 1)
    bp = ax_box.boxplot(
        data_per_motif,
        positions=positions,
        widths=0.6,
        vert=True,
        showfliers=False,
        patch_artist=True,
    )
    for box, m in zip(bp["boxes"], motifs):
        box.set_facecolor(color_map.get(m, "#cccccc"))
        box.set_edgecolor("black")
        box.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_linewidth(2.0)
    max_points = 400
    rng = np.random.RandomState(20251120)
    for pos, m, vals in zip(positions, motifs, data_per_motif):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > max_points:
            idx = rng.choice(vals.size, size=max_points, replace=False)
            vals = vals[idx]
        x_jitter = rng.normal(loc=pos, scale=0.06, size=vals.size)
        ax_box.scatter(x_jitter, vals, s=6, alpha=0.25, linewidths=0, color="black")

    ax_box.axhline(0, color="k", lw=0.8)
    ax_box.set_xlim(0.5, len(motifs) + 0.5)
    ax_box.set_ylim(y_lo, y_hi)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(motifs, rotation=45, ha="right")
    ax_box.set_ylabel("Δprediction (per sample)")
    ax_box.set_title("Per-sample Δ distribution")
    ax_box.grid(axis="y", linestyle="--", alpha=0.4)
    ax_box.text(0.02, 0.98, "(b)", transform=ax_box.transAxes,
                ha="left", va="top", fontsize=18, fontweight="bold")

    # ---------- (c2) Position Effect Heat Map ----------
    ax_heat = fig.add_subplot(gs[0, 2])
    im = ax_heat.imshow(
        heat_vals,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="coolwarm",
        vmin=-v_max,
        vmax=v_max,
    )
    ax_heat.set_yticks(np.arange(len(motifs)))
    ax_heat.set_yticklabels(motifs)
    ax_heat.set_xlabel("Relative position in 3'UTR")
    ax_heat.set_title("Position effect (per motif)")

    # x-axis tick: mapped to relative positions between 0 and 1
    n_bins = heat_vals.shape[1]
    if n_bins > 1:
        xticks = np.linspace(0, n_bins - 1, 5)
        xlabels = [f"{t:.2f}" for t in np.linspace(0.0, 1.0, 5)]
        ax_heat.set_xticks(xticks)
        ax_heat.set_xticklabels(xlabels)
    else:
        ax_heat.set_xticks([0])
        ax_heat.set_xticklabels(["0.5"])

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Δprediction (per site)")

    ax_heat.text(0.02, 0.98, "(c)", transform=ax_heat.transAxes,
                 ha="left", va="top", fontsize=18, fontweight="bold")

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "motif_effect_triptych_C2"))

def plot_motif_tail_prop(df_motif: pd.DataFrame,
                         motifs_focus,
                         out_dir: str,
                         title_suffix: str = ""):
    """
    Plot motif tail response proportion bar chart:
    - Two bars per motif: Proportion of Δt/t0 > +BIG_FRAC_THRESH and Δt/t0 < -BIG_FRAC_THRESH
    - Upper half = Positive strong response proportion, lower half = Negative strong response proportion
    """
    motifs_focus = list(motifs_focus) if motifs_focus is not None else []
    if df_motif is None or df_motif.empty or len(motifs_focus) == 0:
        return

    df_sub = df_motif[df_motif["motif"].isin(motifs_focus)].copy()
    if df_sub.empty:
        return

    # Ensure the sequence matches motifs_focus.
    df_sub = df_sub.set_index("motif").loc[motifs_focus].reset_index()

    gt = df_sub["prop_frac_gt_big"].to_numpy(dtype=float)
    lt = df_sub["prop_frac_lt_-0.1"].to_numpy(dtype=float)
    motifs = df_sub["motif"].tolist()

    x = np.arange(len(motifs))
    fig_width = max(6.0, 0.6 * len(motifs) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    bar_width = 0.6
    ax.bar(x, gt, width=bar_width, color="#d95f02", alpha=0.9, label="Δt/t0 > +10%")
    ax.bar(x, -lt, width=bar_width, color="#1b9e77", alpha=0.9, label="Δt/t0 < -10%")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(motifs, rotation=45, ha="right")
    ax.set_ylabel("Fraction of sequences")

    base_title = "Motif tail response fractions"
    if title_suffix:
        ax.set_title(base_title + " " + title_suffix)
    else:
        ax.set_title(base_title)

    # Symmetric y-axis range
    both = np.concatenate([gt, lt]) if len(gt) > 0 else np.array([0.0])
    max_val = float(np.nanmax(np.abs(both)))
    if max_val <= 0 or not np.isfinite(max_val):
        max_val = 0.1
    ax.set_ylim(-max_val * 1.1, max_val * 1.1)

    ax.legend(loc="upper right", fontsize=10, frameon=False)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "motif_tail_prop"))

def plot_motif_t0bin_heatmap(df_motif_t0: pd.DataFrame,
                             motifs_focus,
                             out_dir: str):
    """
    Plot the mean_frac_delta heatmap for motif × t0_bin.
    Rows: motif; Columns: t0_bin (T0_bin1/2/3...); Colors: mean_frac_delta.
    """
    motifs_focus = list(motifs_focus) if motifs_focus is not None else []
    if df_motif_t0 is None or df_motif_t0.empty or len(motifs_focus) == 0:
        return

    df_sub = df_motif_t0[df_motif_t0["motif"].isin(motifs_focus)].copy()
    if df_sub.empty:
        return

    # Unify bin order
    df_sub["t0_bin"] = df_sub["t0_bin"].astype(str)
    bins_order = sorted(df_sub["t0_bin"].unique())

    df_piv = (df_sub
              .pivot(index="motif", columns="t0_bin", values="mean_frac_delta")
              .reindex(index=motifs_focus, columns=bins_order))

    data = df_piv.to_numpy(dtype=float)

    fig_height = 0.4 * len(motifs_focus) + 1.8
    fig, ax = plt.subplots(figsize=(4.5, fig_height))

    # Symmetrical Color Axis
    if np.isfinite(data).any():
        vmax = float(np.nanmax(np.abs(data)))
    else:
        vmax = 0.1
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 0.1

    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(len(motifs_focus)))
    ax.set_yticklabels(motifs_focus)
    ax.set_xticks(np.arange(len(bins_order)))
    ax.set_xticklabels(bins_order)
    ax.set_xlabel("Baseline half-life bin (t0_bin)")
    ax.set_ylabel("Motif")
    ax.set_title("Mean Δt/t0 per motif and t0 bin")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean_frac_delta (Δt/t0)", rotation=90)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "heatmap_motif_t0bin_mean_frac_delta"))

def plot_motif_tail_and_heatmap(
    df_motif: pd.DataFrame,
    df_motif_t0: pd.DataFrame,
    motifs: list[str],
    out_dir: str,
):
    """
    (a) Top: Positive/negative ratio of motif tails (vertical bar chart, motifs on x-axis)
    (b) Bottom: Motif × t0_bin heatmap (motifs on x-axis, t0_bin on y-axis)

    motifs: A sorted list of motifs (e.g., motifs_sorted)
    Both subplots share the same x-axis scale and order.
    """
    if df_motif is None or df_motif.empty:
        return
    if df_motif_t0 is None or df_motif_t0.empty:
        return

    # Only retain motifs present in both DataFrames, sorted according to the input motifs.
    motifs = [
        m for m in motifs
        if (m in df_motif["motif"].values) and (m in df_motif_t0["motif"].values)
    ]
    if not motifs:
        return

    # ---------- Above image: tail data ----------
    sub_tail = (
        df_motif[df_motif["motif"].isin(motifs)]
        .set_index("motif")
        .loc[motifs]  # Ensure consistent order
        .reset_index()
    )
    gt = sub_tail["prop_frac_gt_big"].to_numpy(float)
    lt = sub_tail["prop_frac_lt_-0.1"].to_numpy(float)

    # ---------- Figure below: Heatmap data ----------
    sub_t0 = df_motif_t0[df_motif_t0["motif"].isin(motifs)].copy()
    if sub_t0.empty:
        return

    sub_t0["t0_bin"] = sub_t0["t0_bin"].astype(str)
    bins_order = sorted(sub_t0["t0_bin"].unique())

    # pivot: rows = t0_bin, columns = motif → heatmap the horizontal axis is the motif
    df_piv = (
        sub_t0
        .pivot(index="t0_bin", columns="motif", values="mean_frac_delta")
        .reindex(index=bins_order, columns=motifs)
    )
    heat = df_piv.to_numpy(float)

    # ---------- Plot: Two subplots sharing the same x-axis ----------
    n = len(motifs)
    fig_width = max(8.0, 0.6 * n + 2.0)   # Automatically relax based on motif count
    fig_height = 6.5

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        sharex=True,  # Shared x-axis
        gridspec_kw={"height_ratios": [1.0, 1.2]}
    )

    # ====== (a) Top: Tail vertical bar chart (same x-axis, one above the other) ======
    x = np.arange(n)
    bar_w = 0.45  # It can be a bit wider.

    # Positive: Δt/t0 > +10%, up
    ax_top.bar(x, gt, width=bar_w,
               color="#d95f02", alpha=0.9, label="Δt/t0 > +10%")

    # Negative: Δt/t0 < -10%, downward (indicated by negative height)
    ax_top.bar(x, -lt, width=bar_w,
               color="#1b9e77", alpha=0.9, label="Δt/t0 < -10%")

    # Share the x-axis tick mark position, but do not display text in the upper figure; display it only in the lower figure.
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])

    ax_top.axhline(0, color="black", linewidth=1)

    # ax_top.set_ylabel("Fraction of sequences")

    # Y-axis asymmetry: The zero line is slightly offset downward.
    both = np.concatenate([gt, lt]) if len(gt) else np.array([0.0])
    max_val = float(np.nanmax(both)) if np.isfinite(both).any() else 0.1
    if max_val <= 0 or not np.isfinite(max_val):
        max_val = 0.1

    neg_scale = 0.8
    pos_scale = 1.6
    ax_top.set_ylim(-max_val * neg_scale, max_val * pos_scale)

    ax_top.legend(loc="upper right", fontsize=9, frameon=False)

    # Add to the subgraph (a)
    ax_top.text(
        0.03, 0.91,
        "(a)",
        transform=ax_top.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # ====== (b) Down：motif × t0_bin heatmap ======
    if np.isfinite(heat).any():
        vmax = float(np.nanmax(np.abs(heat)))
    else:
        vmax = 0.1
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 0.1

    im = ax_bottom.imshow(
        heat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax_bottom.set_xticks(x)
    # ax_bottom.set_xticklabels(motifs, rotation=45, ha="right")
    labels = ax_bottom.set_xticklabels(motifs, rotation=45, ha="right")

    # Apply offset to each label
    for label in labels:
        # Create an offset transformation: Offset 2 points in the x-direction, leaving the y-direction unchanged.
        offset = matplotlib.transforms.ScaledTranslation(10 / 72, 0, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    ax_bottom.set_yticks(np.arange(len(bins_order)))
    label_map = {
        "T0_bin1": "Bin1",
        "T0_bin2": "Bin2",
        "T0_bin3": "Bin3",
    }
    tick_labels = [label_map.get(b, b) for b in bins_order]
    ax_bottom.set_yticklabels(tick_labels)

    ax_bottom.set_xlabel("Motif")
    # ax_bottom.set_ylabel("Baseline half-life bin")

    # The colorbar is only anchored to the bottom axis, with the same size as before.
    cbar = fig.colorbar(im, ax=ax_bottom, fraction=0.046, pad=0.04)
    # cbar.set_label("mean_frac_delta (Δt/t0)")

    ax_bottom.text(
        0.03, 0.92,
        "(b)",
        transform=ax_bottom.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    ax_bottom.set_xlim(-0.5, n - 0.5)
    fig.tight_layout()
    pos_bottom = ax_bottom.get_position()
    pos_top = ax_top.get_position()
    ax_top.set_position([pos_bottom.x0, pos_top.y0, pos_bottom.width, pos_top.height])

    savefig(fig, os.path.join(out_dir, "motif_tail_and_t0bin_heatmap"))
    plt.close(fig)

def plot_motif_t0bin_bars_detail(df_motif_t0: pd.DataFrame,
                                 motifs_detail,
                                 out_dir: str):
    """
    Plot t0_bin hierarchical bar charts for several representative motifs (e.g., CCUAA/UUAUUU/CCCCC/GCGCGC).
Each motif is displayed as a subplot, with the x-axis representing t0_bin and the y-axis representing mean_frac_delta.
    """
    motifs_detail = [m for m in (motifs_detail or [])]
    if df_motif_t0 is None or df_motif_t0.empty or len(motifs_detail) == 0:
        return

    # Filter out motifs that actually exist
    motifs_exist = [m for m in motifs_detail if m in df_motif_t0["motif"].values]
    if len(motifs_exist) == 0:
        return

    bins_order = sorted(df_motif_t0["t0_bin"].astype(str).unique())

    n = len(motifs_exist)
    ncols = min(CFG.NCOLS, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.2 * nrows),
        squeeze=False
    )

    for i, motif in enumerate(motifs_exist):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = (df_motif_t0[df_motif_t0["motif"] == motif]
               .assign(t0_bin=lambda d: d["t0_bin"].astype(str)))

        vals = []
        for b in bins_order:
            row = sub.loc[sub["t0_bin"] == b]
            if row.empty:
                vals.append(np.nan)
            else:
                vals.append(float(row["mean_frac_delta"].iloc[0]))

        x = np.arange(len(bins_order))
        ax.bar(x, vals, width=0.6)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(bins_order)
        ax.set_ylabel("mean Δt/t0")
        ax.set_title(motif)

    # Hide redundant subgraphs
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "bar_motif_t0bins_detail"))



def main():
    ensure_dir(CFG.OUT_DIR)
    df = pd.read_csv(CFG.MUTATION_CSV)

    required_cols = {"sample_idx","motif","new","pos","base_pred","mut_pred","delta","sequence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Unified Type
    df["sample_idx"] = df["sample_idx"].astype(int)
    for c in ["motif","new","sequence"]:
        df[c] = df[c].astype(str)
    df["pos"] = df["pos"].astype(int)
    for c in ["base_pred","mut_pred","delta"]:
        df[c] = df[c].astype(float)

    # Relative Position (motif center/sequence length)
    df["seq_len"] = df["sequence"].str.len()
    df["motif_len"] = df["motif"].str.len()
    df["pos_rel"] = (df["pos"] + df["motif_len"]/2.0) / df["seq_len"]

    # -------- Aggregate per sample (to prevent amplified weights at multiple loci within the same sequence)--------
    # This is an extra sequence, so that if residuals.csv doesn't have a sample_idx, it can be aligned with sequence instead.
    per_seq = (
        df.groupby(["motif", "new", "sample_idx"], as_index=False)
        .agg(
            base_pred=("base_pred", "mean"),
            mut_pred=("mut_pred", "mean"),
            delta=("delta", "mean"),
            sequence=("sequence", "first"),
        )
    )

    # (Optional) Merge residual information for analysis of the "better-predicted" subset.
    residual_path = CFG.RESIDUALS_CSV.strip()
    if residual_path == "":
        residual_path = os.path.join(os.path.dirname(CFG.MUTATION_CSV), "residuals.csv")
    have_residual = False
    residual_threshold_used = None

    if os.path.isfile(residual_path):
        try:
            df_resid = pd.read_csv(residual_path)

            # First determine the residual column: Prioritize using existing residuals, then use the difference between true and predicted values.
            resid_col = None
            if "residual" in df_resid.columns:
                resid_col = "residual"
            elif {"true", "pred"}.issubset(df_resid.columns):
                df_resid["residual"] = df_resid["true"] - df_resid["pred"]
                resid_col = "residual"

            if resid_col is None:
                print(
                    f"[WARN] residuals Document {residual_path} If the residual or true/pred column is not found, skip residual-related analysis.")
            else:
                # Preferred approach: Align by sample_idx
                if "sample_idx" in df_resid.columns and "sample_idx" in per_seq.columns:
                    df_resid["sample_idx"] = df_resid["sample_idx"].astype(int)
                    df_resid_small = df_resid[["sample_idx", "residual"]].drop_duplicates(subset="sample_idx")
                    per_seq = per_seq.merge(df_resid_small, on="sample_idx", how="left")
                    have_residual = True

                # Alternative: if there is no sample_idx, but there is a sequence in both residuals.csv and per_seq, then align by sequence
                elif "sequence" in df_resid.columns and "sequence" in per_seq.columns:
                    df_resid["sequence"] = df_resid["sequence"].astype(str)
                    df_resid_small = df_resid[["sequence", "residual"]].drop_duplicates(subset="sequence")
                    per_seq = per_seq.merge(df_resid_small, on="sequence", how="left")
                    have_residual = True
                else:
                    print(
                        f"[WARN] residuals Document {residual_path} There is no sample_idx，"
                        f"There is also no sequence column that can be aligned with per_seq, skipping the residual correlation analysis."
                    )

                # If the residuals merge successfully, calculate the absolute residuals as well.
                if have_residual:
                    per_seq["abs_residual"] = per_seq["residual"].abs()

        except Exception as e:
            print(f"[WARN] Read residuals Document {residual_path} Failure：{e}")
    else:
        print(f"[INFO] Not found residuals Document {residual_path},The residual-related subset analysis will be skipped.")

    # Baseline half-life stratification (using quantiles of base_pred)
    have_t0_bins = False
    if per_seq["base_pred"].notna().any():
        try:
            base_vals = per_seq["base_pred"].to_numpy(dtype=float)
            base_vals = base_vals[np.isfinite(base_vals)]
            if base_vals.size >= CFG.N_T0_BINS:
                q = np.linspace(0.0, 1.0, CFG.N_T0_BINS + 1)
                edges = np.quantile(base_vals, q)
                # Prevent identical boundaries → Slightly expand
                edges[0] = -np.inf
                edges[-1] = np.inf
                labels = [f"T0_bin{i+1}" for i in range(CFG.N_T0_BINS)]
                per_seq["t0_bin"] = pd.cut(per_seq["base_pred"], bins=edges,
                                           labels=labels, include_lowest=True)
                have_t0_bins = True
            else:
                print("[WARN] The number of valid base_pred samples is too small to partition into T0 bins; baseline stratification analysis is skipped.")
        except Exception as e:
            print(f"[WARN] Calculation of baseline half-life stratification failed：{e}")

    # ====== Statistics: By (motif, new) ======
    rows_pair = []
    for (m,n), sub in per_seq.groupby(["motif","new"]):
        stats = compute_effect_stats(sub)
        rows_pair.append({
            "motif": m,
            "replacement": n,
            **stats
        })
    df_pair = pd.DataFrame(rows_pair)
    if not df_pair.empty:
        df_pair["fdr"] = bh_fdr(df_pair["wilcoxon_p"].values)
    df_pair = df_pair.sort_values(["fdr","mean_delta"]).reset_index(drop=True)

    # ====== Statistics: by motif (merged across replacements) ======
    rows_m = []
    for m, sub in per_seq.groupby("motif"):
        stats = compute_effect_stats(sub)
        rows_m.append({
            "motif": m,
            **stats
        })
    df_motif = pd.DataFrame(rows_m)
    if not df_motif.empty:
        df_motif["fdr"] = bh_fdr(df_motif["wilcoxon_p"].values)
    df_motif = df_motif.sort_values(["fdr","mean_delta"]).reset_index(drop=True)

    # ====== Baseline Half-Life Stratified Statistics (by t0_bin) ======
    df_pair_t0 = pd.DataFrame()
    df_motif_t0 = pd.DataFrame()
    if have_t0_bins and "t0_bin" in per_seq.columns:
        rows_pair_t0 = []
        rows_m_t0 = []

        tmp = per_seq.dropna(subset=["t0_bin"])
        for (m, n, bin_label), sub in tmp.groupby(["motif","new","t0_bin"], observed=False):
            stats = compute_effect_stats(sub)
            rows_pair_t0.append({
                "motif": m,
                "replacement": n,
                "t0_bin": str(bin_label),
                **stats
            })
        df_pair_t0 = pd.DataFrame(rows_pair_t0)

        for (m, bin_label), sub in tmp.groupby(["motif","t0_bin"], observed=False):
            stats = compute_effect_stats(sub)
            rows_m_t0.append({
                "motif": m,
                "t0_bin": str(bin_label),
                **stats
            })
        df_motif_t0 = pd.DataFrame(rows_m_t0)

    # ====== residual Subset Statistics (Predicting Better Samples)======
    df_pair_good = pd.DataFrame()
    df_motif_good = pd.DataFrame()
    if have_residual and CFG.USE_RESIDUAL_FILTER and "abs_residual" in per_seq.columns:
        valid_abs = per_seq["abs_residual"].to_numpy(dtype=float)
        valid_abs = valid_abs[np.isfinite(valid_abs)]
        if valid_abs.size > 0:
            if CFG.RESIDUAL_ABS_THRESH > 0:
                residual_threshold_used = float(CFG.RESIDUAL_ABS_THRESH)
            else:
                residual_threshold_used = float(np.median(valid_abs))
            good_mask = per_seq["abs_residual"] <= residual_threshold_used
            per_seq_good = per_seq[good_mask].copy()
            if len(per_seq_good) > 0:
                rows_pair_good = []
                rows_m_good = []
                for (m, n), sub in per_seq_good.groupby(["motif","new"]):
                    stats = compute_effect_stats(sub)
                    rows_pair_good.append({
                        "motif": m,
                        "replacement": n,
                        **stats
                    })
                df_pair_good = pd.DataFrame(rows_pair_good)
                if not df_pair_good.empty:
                    df_pair_good["fdr"] = bh_fdr(df_pair_good["wilcoxon_p"].values)

                for m, sub in per_seq_good.groupby("motif"):
                    stats = compute_effect_stats(sub)
                    rows_m_good.append({
                        "motif": m,
                        **stats
                    })
                df_motif_good = pd.DataFrame(rows_m_good)
                if not df_motif_good.empty:
                    df_motif_good["fdr"] = bh_fdr(df_motif_good["wilcoxon_p"].values)
        else:
            print("[WARN] No valid values found in abs_residual; skipping residual subset statistics.")

    # ====== Position Effect (By Site, Grouped)======
    pos_bins = np.linspace(0, 1, CFG.N_POS_BINS+1)
    df["pos_bin"] = pd.cut(df["pos_rel"], bins=pos_bins, labels=False, include_lowest=True)
    pos_rows = []
    for m, sub in df.groupby("motif"):
        for b, sb in sub.groupby("pos_bin"):
            if len(sb) == 0:
                continue
            arr = sb["delta"].values.astype(float)
            mean = float(np.mean(arr))
            lo, hi = bootstrap_ci(arr, n_boot=CFG.N_BOOT, seed=CFG.SEED)
            center = (pos_bins[int(b)] + pos_bins[int(b)+1]) / 2.0
            pos_rows.append({
                "motif": m, "pos_bin": int(b),
                "pos_rel_center": float(center),
                "mean_delta": mean, "ci95_low_delta": lo, "ci95_high_delta": hi,
                "n_sites": int(len(arr))
            })
    df_pos = pd.DataFrame(pos_rows).sort_values(["motif","pos_bin"]).reset_index(drop=True)

    # ====== Export CSV ======
    out_pair = os.path.join(CFG.OUT_DIR, "mutation_stats_by_motif_new.csv")
    out_motif = os.path.join(CFG.OUT_DIR, "mutation_stats_by_motif.csv")
    out_pos = os.path.join(CFG.OUT_DIR, "mutation_position_effect.csv")
    out_perseq = os.path.join(CFG.OUT_DIR, "per_sequence_delta.csv")
    out_pair_t0 = os.path.join(CFG.OUT_DIR, "mutation_stats_by_motif_new_t0bins.csv")
    out_motif_t0 = os.path.join(CFG.OUT_DIR, "mutation_stats_by_motif_t0bins.csv")
    out_pair_good = os.path.join(CFG.OUT_DIR, "mutation_stats_by_motif_new_goodfit.csv")
    out_motif_good = os.path.join(CFG.OUT_DIR, "mutation_stats_by_motif_goodfit.csv")

    per_seq.to_csv(out_perseq, index=False)
    df_pair.to_csv(out_pair, index=False)
    df_motif.to_csv(out_motif, index=False)
    df_pos.to_csv(out_pos, index=False)
    if not df_pair_t0.empty:
        df_pair_t0.to_csv(out_pair_t0, index=False)
    if not df_motif_t0.empty:
        df_motif_t0.to_csv(out_motif_t0, index=False)
    if not df_pair_good.empty:
        df_pair_good.to_csv(out_pair_good, index=False)
    if not df_motif_good.empty:
        df_motif_good.to_csv(out_motif_good, index=False)


    # ====== Plotting ======
    # Unify the selection of motifs to be displayed in various plots (length controlled by CFG.TOPK)
    motifs_sorted = select_motifs_for_plot(df_motif, CFG.TOPK)

    # 0a) motif Tail Response Ratio Chart (Prioritize goodfit Subset)
    df_for_tail = df_motif_good if not df_motif_good.empty else df_motif
    if df_for_tail is not None and not df_for_tail.empty and len(motifs_sorted) > 0:
        # Only retain motifs that are indeed statistically significant in df_for_tail.
        motifs_for_tail = [m for m in motifs_sorted if m in df_for_tail["motif"].values]
        if len(motifs_for_tail) > 0:
            print("[INFO] Plot the motif_tail_prop graph using the following DataFrame:",
                  "goodfit subset" if df_for_tail is df_motif_good else "All samples")
            plot_motif_tail_prop(df_for_tail, motifs_for_tail, CFG.OUT_DIR)

    # 0b) motif×t0_bin Heatmap + t0-based hierarchical bar chart representing motif
    if df_motif_t0 is not None and not df_motif_t0.empty and len(motifs_sorted) > 0:
        # Only retain motifs present in the t0 stratified statistics, maintaining the order specified in motifs_sorted.
        motifs_for_t0 = [m for m in motifs_sorted if m in df_motif_t0["motif"].values]
        if len(motifs_for_t0) > 0:
            plot_motif_t0bin_heatmap(df_motif_t0, motifs_for_t0, CFG.OUT_DIR)

        # Detailed bar charts can still be generated by selecting only CCUAA/UUAUUU/CCCCC/GCGCGC sequences that appear in both motifs_for_t0 and the statistical data.
        motifs_detail = [
            m for m in ["CCUAA", "UUAUUU", "CCCCC", "GCGCGC"]
            if m in motifs_for_t0
        ]
        if len(motifs_detail) > 0:
            plot_motif_t0bin_bars_detail(df_motif_t0, motifs_detail, CFG.OUT_DIR)


    # 0c) Combined Chart: Left tail, Right t0_bin Heatmap
    if (df_motif is not None and not df_motif.empty and
        df_motif_t0 is not None and not df_motif_t0.empty and
        len(motifs_sorted) > 0):

        motifs_for_combo = [
            m for m in motifs_sorted
            if (m in df_motif["motif"].values
                and m in df_motif_t0["motif"].values)
        ]
        if len(motifs_for_combo) > 0:
            plot_motif_tail_and_heatmap(
                df_motif=df_motif,
                df_motif_t0=df_motif_t0,
                motifs=motifs_for_combo,
                out_dir=CFG.OUT_DIR,
            )


    # 1) Top-K Horizontal Bar Chart (Motif-merged, with 95% CI, distinguishing positive and negative Δ)
    if len(motifs_sorted) > 0:
        sub = df_motif[df_motif["motif"].isin(motifs_sorted)].copy()
        sub = sub.set_index("motif").loc[motifs_sorted].reset_index()

        # Set colors based on the positive or negative value of mean_delta
        mu = sub["mean_delta"].values
        lo = sub["ci95_low_delta"].values
        hi = sub["ci95_high_delta"].values
        err = np.vstack([mu - lo, hi - mu])
        colors = np.where(mu >= 0, "#d95f02", "#1b9e77")  # Δ>0 Orange, Δ<0 Green

        # Draw
        fig_height = 0.45 * len(sub) + 1.5
        fig, ax = plt.subplots(figsize=(7, fig_height))
        y = np.arange(len(sub))
        bars = ax.barh(y, mu, xerr=err, align="center", alpha=0.9)

        # Color each bar
        for bar, col in zip(bars, colors):
            bar.set_color(col)

        # Y-axis label: Significant increase *
        labels = []
        for _, row in sub.iterrows():
            label = row["motif"]
            if "fdr" in sub.columns and row["fdr"] <= 0.05:
                label += " *"
            labels.append(label)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        ax.axvline(0, color="k", lw=0.8)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_xlabel("Mean Δprediction (mut - base)")
        ax.set_title(f"Top-{CFG.TOPK} motifs by |Δ| (per-sample mean, 95% CI)")

        # Place the largest |Δ| at the top (more in line with reading habits)
        ax.invert_yaxis()

        fig.tight_layout()
        savefig(fig, os.path.join(CFG.OUT_DIR, "bar_topk_motif_mean_delta"))

    # 2) Box plot (Δ per sample)
    if len(motifs_sorted) > 0:
        n = len(motifs_sorted)
        ncols = min(CFG.NCOLS, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), squeeze=False)
        for i, m in enumerate(motifs_sorted):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            arr = per_seq.loc[per_seq["motif"]==m, "delta"].values
            ax.boxplot(arr, vert=True, tick_labels=[m], whis=1.5, showfliers=False)
            ax.axhline(0, color='k', lw=1)
            ax.set_ylabel("Δprediction (per sample)", labelpad=2, fontsize=16)
            ax.set_title(f"{m} (n={len(arr)})")
        # Hide redundant subgraphs
        for j in range(i+1, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
        fig.tight_layout()
        savefig(fig, os.path.join(CFG.OUT_DIR, "box_per_motif_delta"))

    # 3) Position Effect Curve (Δ vs Relative Position, Remove Duplicate Labels to Prevent Subfigure Text Overlap)
    if not df_pos.empty:
        motifs_for_pos = motifs_sorted if len(motifs_sorted) > 0 else df_pos["motif"].unique().tolist()
        n = len(motifs_for_pos)
        ncols = min(CFG.NCOLS, n if n > 0 else 1)
        nrows = int(math.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.8 * ncols, 3.2 * nrows),
            sharex=True,
            squeeze=False
        )

        for idx, m in enumerate(motifs_for_pos):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            sub = df_pos[df_pos["motif"] == m].sort_values("pos_rel_center")

            ax.plot(sub["pos_rel_center"], sub["mean_delta"], marker="o", lw=1)
            ax.fill_between(sub["pos_rel_center"], sub["ci95_low_delta"], sub["ci95_high_delta"], alpha=0.25)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_title(f"{m} (per-site, binned)", fontsize=18)

            # Draw an x label only on the bottom row.
            if r == nrows - 1:
                ax.set_xlabel("Relative position in 3'UTR", fontsize=16)
            # Plot the y-label only in the first column
            if c == 0:
                ax.set_ylabel("Δprediction (per site)", fontsize=16)

        # Hide redundant subgraphs
        total_plots = len(motifs_for_pos)
        for j in range(total_plots, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig.tight_layout()
        savefig(fig, os.path.join(CFG.OUT_DIR, "position_effect_per_motif"))

        # 3b) Comprehensive Motif Effect Triad (C1: Position Line Chart, C2: Position Heatmap)
    if len(motifs_sorted) > 0 and not df_pos.empty:
        try:
            plot_motif_triptych_c1(df_motif, per_seq, df_pos, motifs_sorted, CFG.OUT_DIR)
            plot_motif_triptych_c2(df_motif, per_seq, df_pos, motifs_sorted, CFG.OUT_DIR)
        except Exception as e:
            print("[WARN] Failed to generate motif triplet diagram:", e)

        # 4) Pairwise Scatter Plot (Base vs. Mut, by Sample Mean)
    if len(motifs_sorted) > 0:
        for m in motifs_sorted:
            sub = per_seq[per_seq["motif"]==m].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.scatter(sub["base_pred"], sub["mut_pred"], s=8, alpha=0.5)
            lo = min(sub["base_pred"].min(), sub["mut_pred"].min())
            hi = max(sub["base_pred"].max(), sub["mut_pred"].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Base prediction (per sample mean)")
            ax.set_ylabel("Mutated prediction (per sample mean)")
            ax.set_title(f"Paired scatter: {m}")
            savefig(fig, os.path.join(CFG.OUT_DIR, f"scatter_base_vs_mut_{m}"))

    # 5) Volcano Plot (Grouped by Motif)
    if not df_motif.empty:
        sub = df_motif.copy()
        x = sub["mean_delta"].values
        y = -np.log10(np.maximum(sub["fdr"].values, 1e-300)) if "fdr" in sub.columns else -np.log10(np.maximum(sub["wilcoxon_p"].values, 1e-300))
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.scatter(x, y, s=20, alpha=0.7)
        ax.axvline(0, color='k', lw=1)
        if "fdr" in sub.columns:
            ax.axhline(-np.log10(0.05), color='r', lw=1, linestyle='--')
        ax.set_xlabel("Mean Δprediction (motif-merged)")
        ax.set_ylabel("-log10(FDR)" if "fdr" in sub.columns else "-log10(p)")
        ax.set_title("Volcano (by motif)")
        # Top-K Annotation
        idx = np.argsort(np.abs(x))[-CFG.TOPK:]
        for i in idx:
            ax.text(x[i], y[i], sub["motif"].iloc[i], fontsize=9)
        savefig(fig, os.path.join(CFG.OUT_DIR, "volcano_by_motif"))

    # ====== Write summary.json ======
    summary = {
        "input_csv": CFG.MUTATION_CSV,
        "n_rows": int(len(df)),
        "n_unique_motifs": int(df["motif"].nunique()),
        "n_unique_pairs": int(df[["motif","new"]].drop_duplicates().shape[0]),
        "per_sequence_rows": int(len(per_seq)),
        "topk": CFG.TOPK,
        "pos_bins": CFG.N_POS_BINS,
        "n_boot": CFG.N_BOOT,
        "have_t0_bins": bool(have_t0_bins),
        "have_residual": bool(have_residual),
        "residual_threshold_used": residual_threshold_used,
        "big_frac_thresh": CFG.BIG_FRAC_THRESH,
        "outputs": {
            "by_motif_csv": out_motif,
            "by_motif_new_csv": out_pair,
            "by_motif_t0bins_csv": out_motif_t0 if not df_motif_t0.empty else "",
            "by_motif_new_t0bins_csv": out_pair_t0 if not df_pair_t0.empty else "",
            "by_motif_goodfit_csv": out_motif_good if not df_motif_good.empty else "",
            "by_motif_new_goodfit_csv": out_pair_good if not df_pair_good.empty else "",
            "position_csv": out_pos,
            "per_sequence_csv": out_perseq,
            "figs": [
                "motif_tail_prop.(png/svg)",
                "heatmap_motif_t0bin_mean_frac_delta.(png/svg)",
                "bar_motif_t0bins_detail.(png/svg)",
                "bar_topk_motif_mean_delta.(png/svg)",
                "box_per_motif_delta.(png/svg)",
                "position_effect_per_motif.(png/svg)",
                "motif_effect_triptych_C1.(png/svg)",
                "motif_effect_triptych_C2.(png/svg)",
                "scatter_base_vs_mut_<motif>.(png/svg)",
                "volcano_by_motif.(png/svg)"
            ]

        }
    }
    with open(os.path.join(CFG.OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Analysis complete. Results saved to:", CFG.OUT_DIR)



if __name__ == "__main__":
    main()
