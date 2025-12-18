# plots/plot_utils.py
# -*- coding: utf-8 -*-
"""
General Drawing Tools (Paper Style):
- Uniform styling (font size, line width, grid, scale direction)
- Simultaneous PNG/SVG saving (dpi=400)
- Secure CSV/JSON/JSONL reading
- Common reference lines: scatter plots, calibration, y=x
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Server/No Display Environment
import matplotlib.pyplot as plt

# ========= Style and Save =========
def set_paper_style():
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Select common non-italic Western fonts, with Chinese alternatives included to prevent substitution with italic variants.
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans CJK SC"],
        "font.style": "normal",            # Key: Global Upright
        "mathtext.default": "regular",     # Key: Use upright rather than italicized variables in mathematical text.
        "mathtext.fontset": "dejavusans",  # Use upright sans-serif typefaces
        "axes.unicode_minus": False,

        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.linestyle": ":",
        "grid.alpha": 0.3,
        "savefig.transparent": False,
        "figure.autolayout": False,
    })


def _fisher_two_sided_p(a, b, c, d):
    """
    2x2 Fisher Exact test (two-tailed), based on the cumulative hypergeometric distribution.
    Implementing log combinator using only math.lgamma to avoid overflow.
    Table:
              present  absent
      top        a       b
      bottom     c       d
    """
    import math
    a = int(a); b = int(b); c = int(c); d = int(d)
    row1 = a + b; row2 = c + d
    col1 = a + c; col2 = b + d
    N = row1 + row2
    if N == 0:
        return 1.0

    def logC(n, k):
        if k < 0 or k > n:
            return -float("inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    # Hypergeometric PMF：P(X=x | row1, col1, N)
    def logpmf(x):
        return logC(col1, x) + logC(col2, row1 - x) - logC(N, row1)

    # Observed probability
    p_obs = math.exp(logpmf(a))

    # x The range of values
    x_min = max(0, row1 - col2)
    x_max = min(row1, col1)

    # Two-tailed p: Sum all probabilities where P(X) ≤ P(obs)
    p = 0.0
    for x in range(x_min, x_max + 1):
        px = math.exp(logpmf(x))
        if px <= p_obs + 1e-15:
            p += px
    return min(max(p, 0.0), 1.0)



def ensure_dir(path_like) -> str:
    p = Path(path_like)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def savefig_dual(fig, out_basepath: str, dpi: int = 400):
    """
    out_basepath Without file extension; save simultaneously as *.png and *.svg
    """
    fig.savefig(f"{out_basepath}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_basepath}.svg", dpi=dpi, bbox_inches="tight")

# ========= Reading and writing =========
def safe_read_csv(fp: str) -> pd.DataFrame:
    return pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()

def read_json_or_jsonl(fp: str):
    """
    Simultaneously compatible with JSON (list or dict) and JSONL (one object per line)
Returns: list (if dict, wrapped in a list)
    """
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return []
            # Simple Determination of JSONL (Line-by-Line Object)
            if "\n" in txt and txt.lstrip().startswith("{"):
                items = []
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
                return items
            # Ordinary JSON
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            return [obj]
    except Exception:
        return []

# ========= Basic Visual Functions =========
def identity_line(ax, data=None):
    """
    Adaptive plot y=x dashed line; data if provided (concatenated true/pred), used for estimating range
    """
    if data is not None and len(data) > 0:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        lo, hi = np.floor(vmin), np.ceil(vmax)
    else:
        lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.6)

def scatter_true_pred(df: pd.DataFrame, title: str, out_basepath: str):
    """
    Required column：true, pred
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["true"], df["pred"], s=6, alpha=0.6)
    identity_line(ax, np.r_[df["true"].values, df["pred"].values])
    ax.set_xlabel("True half-life")
    ax.set_ylabel("Predicted half-life")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    savefig_dual(fig, out_basepath, dpi=400)
    plt.close(fig)

def calibration_curve(df: pd.DataFrame, n_bins: int, out_basepath: str, title="Calibration (by predicted)"):
    """
    Use bins with equal predicted values to plot the calibration curve
Required columns: true, pred
    """
    df = df[["true","pred"]].dropna().sort_values("pred").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return
    n_bins = max(3, min(n_bins, n))  # Reasonable restrictions
    bins = np.array_split(df, n_bins)
    x_bin = [b["pred"].mean() for b in bins]
    y_bin = [b["true"].mean() for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x_bin, y_bin, marker="o", linewidth=1.5)
    identity_line(ax, np.r_[df["true"].values, df["pred"].values])
    ax.set_xlabel("Predicted (bin mean)")
    ax.set_ylabel("Observed (bin mean)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    savefig_dual(fig, out_basepath, dpi=400)
    plt.close(fig)

def add_panel_title(fig, txt: str):
    fig.suptitle(txt, y=0.98, fontsize=11)
