# plots/plot_utils.py
# -*- coding: utf-8 -*-
"""
通用绘图工具（论文风格）：
- 统一样式（字号、线宽、网格、刻度方向）
- 同时保存 PNG/SVG（dpi=400）
- 安全读取 csv/json/jsonl
- 常用的散点、校准、y=x 参考线等
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境
import matplotlib.pyplot as plt

# ========= 样式与保存 =========
def set_paper_style():
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        # 选择常见的非斜体西文字体，并含中文的备选以防替换成斜体变体
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans CJK SC"],
        "font.style": "normal",            # 关键：全局直立
        "mathtext.default": "regular",     # 关键：数学文本用直立，而不是斜体变量
        "mathtext.fontset": "dejavusans",  # 用 sans 字体系的直立体
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
    2x2 Fisher 精确检验（双侧），基于超几何分布累积概率。
    仅依赖 math.lgamma 实现 log 组合数，避免溢出。
    表格：
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

    # 超几何 pmf：P(X=x | row1, col1, N)
    def logpmf(x):
        return logC(col1, x) + logC(col2, row1 - x) - logC(N, row1)

    # 观测到的概率
    p_obs = math.exp(logpmf(a))

    # x 的取值范围
    x_min = max(0, row1 - col2)
    x_max = min(row1, col1)

    # 双侧 p：把所有 P(X) ≤ P(obs) 的概率加起来
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
    out_basepath 不带扩展名；同时保存为 *.png 和 *.svg
    """
    fig.savefig(f"{out_basepath}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_basepath}.svg", dpi=dpi, bbox_inches="tight")

# ========= 读写 =========
def safe_read_csv(fp: str) -> pd.DataFrame:
    return pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()

def read_json_or_jsonl(fp: str):
    """
    同时兼容 JSON（list或dict）和 JSONL（每行一个对象）
    返回：list（如是dict则包一层list）
    """
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return []
            # 简单判断 JSONL（逐行对象）
            if "\n" in txt and txt.lstrip().startswith("{"):
                items = []
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
                return items
            # 普通 JSON
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            return [obj]
    except Exception:
        return []

# ========= 基础可视函数 =========
def identity_line(ax, data=None):
    """
    自适应画 y=x 虚线；data 若给出（true/pred拼接），用于估计范围
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
    需要列：true, pred
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
    使用按预测值等分的分箱，画校准曲线
    需要列：true, pred
    """
    df = df[["true","pred"]].dropna().sort_values("pred").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return
    n_bins = max(3, min(n_bins, n))  # 合理限制
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
