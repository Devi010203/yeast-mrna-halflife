# plots/plot_dataset_overview.py
# -*- coding: utf-8 -*-
"""
数据集概览（按 Train/Val/Test 等划分）：
  1) 目标变量（mRNA 半衰期）直方图 + KDE（各划分叠加）
  2) 3'UTR 长度分布直方图 + KDE（各划分叠加）
  3) GC 含量分布直方图 + KDE（各划分叠加）
  4) 长度 × GC 的 hexbin 面板（各划分一图）
  5) 统计表：每划分 N、target（均值/中位/IQR/最值/偏度/峰度）、length/gc（均值±SD）
输出：result/plot/dataset_overview/<时间戳>/   （PNG+SVG，dpi=400；坐标文字非斜体）
"""

import re, math, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

import scienceplots

plt.style.use(['science', 'no-latex'])

# ============== 在此手动填写（不使用命令行）=============
# 每个划分可以是“单个CSV路径”或“多个CSV路径”组成的列表；脚本会自动合并
SPLITS: Dict[str, List[str]] = {
    "Train": [r"F:\mRNA_Project\3UTR\Paper\data_splits\data_splits_20251127_173725\final_split\train_set.csv"],
    # ← 改成你的文件
    "Val.": [r"F:\mRNA_Project\3UTR\Paper\data_splits\data_splits_20251127_173725\final_split\val_set.csv"],  # ← 改成你的文件
    "Test": [r"F:\mRNA_Project\3UTR\Paper\data_splits\data_splits_20251127_173725\final_split\test_set.csv"],
    # ← 改成你的文件
}

# 直方图全局 bin 数（对三个分布分别采用全局一致的 bin）
BINS_TARGET = 60
BINS_LEN = 60
BINS_GC = 50

# Hexbin 网格密度
HEX_GRIDSIZE = 60

# 图像参数
DPI = 400
# 叠加分布：4:3 比例，所有三个图保持一致
FIGSIZE_OVERLAY = (6.4, 4.8)
FIGSIZE_HEXGRID = (6.0, 5.0)  # 单个 hexbin
MAX_TICKS_PER_AXIS = 10  # 避免刻度重叠
ALPHA_HIST = 0.35  # 直方图透明度
LINEWIDTH_KDE = 1.8
# =======================================================

# ---- 全局非斜体字体 ----
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
    "xtick.labelsize":17,
    "ytick.labelsize":17,
    "legend.fontsize":16,
    "figure.titlesize":17,
    # "axes.titleweight": "bold",  # 图标题
    # "axes.labelweight": "bold",  # x / y 轴标签

})


def _project_root() -> Path:
    # 本脚本位于 <项目根>/plots/，所以 parent.parent 是项目根
    return Path(__file__).resolve().parent.parent


def _ensure_outdir() -> Path:
    outdir = _project_root() / "result" / "plot" / "dataset_overview" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_dual(fig, out_base: Path):
    fig.savefig(str(out_base) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------- 数据读取 & 预处理 ----------
def _auto_target_col(df: pd.DataFrame) -> str:
    cand = ["half_life", "halflife", "y_true", "true", "target", "label", "ground_truth"]
    for c in df.columns:
        if c.lower() in cand:
            return c
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["half", "true", "target", "label", "ground"]):
            return c
    raise ValueError(f"无法识别目标列（半衰期/真值）；可接受列名示例：{cand}；当前列：{list(df.columns)}")


def _ensure_len_gc(df: pd.DataFrame) -> pd.DataFrame:
    """优先使用 sequence 计算长度与GC；否则回退到已有列"""
    out = df.copy()
    # 优先用 sequence
    if "sequence" in out.columns:
        Ls, Gs = [], []
        for s in out["sequence"].astype(str):
            s2 = re.sub(r"[^ACGTUacgtu]", "", s)  # 只保留 A/C/G/T/U
            s2 = s2.upper().replace("U", "T")  # RNA U 当成 T
            L = len(s2)
            Ls.append(L)
            if L == 0:
                Gs.append(np.nan)
            else:
                gc = s2.count("G") + s2.count("C")
                at = s2.count("A") + s2.count("T")
                tot = gc + at
                Gs.append(gc / tot if tot > 0 else np.nan)
        out["utr_len"] = np.array(Ls, dtype=int)
        out["gc_frac"] = np.array(Gs, dtype=float)
        return out
    # 其次尝试已有的 length/gc 列
    len_col = None
    gc_col = None
    for c in out.columns:
        cl = c.lower()
        if len_col is None and (cl == "length" or cl.endswith("_len") or "length" in cl or cl == "len"):
            len_col = c
        if gc_col is None and (
            cl in ("gc", "gc_content", "gc_fraction") or ("gc" in cl and ("frac" in cl or "content" in cl))
        ):
            gc_col = c
    if len_col is None or gc_col is None:
        raise ValueError("缺少 sequence 列，且未找到 length/gc 列；无法计算二维图所需的长度与GC。")
    out["utr_len"] = out[len_col].astype(int).values
    out["gc_frac"] = out[gc_col].astype(float).values
    return out


def _read_split(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            print(f"[WARN] 未找到文件：{fp}（跳过）")
            continue
        d = pd.read_csv(fp).dropna(how="all")
        dfs.append(d)
    if len(dfs) == 0:
        return pd.DataFrame()
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # 识别目标列
    tgt_col = _auto_target_col(df)
    df = _ensure_len_gc(df)
    df = df.rename(columns={tgt_col: "target"})
    return df[["target", "utr_len", "gc_frac"]].copy()


# ---------- 统计 & KDE ----------
def _iqr(a: np.ndarray) -> float:
    q1, q3 = np.nanpercentile(a, [25, 75])
    return float(q3 - q1)


def _skew_kurt(a: np.ndarray) -> Tuple[float, float]:
    if SCIPY_OK and np.sum(np.isfinite(a)) >= 4:
        return float(stats.skew(a, bias=False)), float(stats.kurtosis(a, fisher=True, bias=False))
    # 简单近似或返回 NaN
    return np.nan, np.nan


def _freedman_diaconis_bins(a: np.ndarray) -> int:
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 10
    iqr = _iqr(a)
    if iqr == 0:
        return 10
    bw = 2 * iqr * (a.size ** (-1 / 3))
    if bw <= 0:
        return 10
    k = int(np.ceil((np.nanmax(a) - np.nanmin(a)) / bw))
    return max(10, min(200, k))


def _kde_line(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    if SCIPY_OK:
        try:
            kde = stats.gaussian_kde(x)
            return kde(grid)
        except Exception:
            pass
    # fallback: 简易核密度（高斯核）
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(grid, np.nan, dtype=float)
    bw = 1.06 * sd * (x.size ** (-1 / 5))
    z = (grid.reshape(-1, 1) - x.reshape(1, -1)) / (bw + 1e-12)
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens


# ---------- 画图工具 ----------
def _limited_ticks(ax, axis: str = "x", max_ticks: int = 10):
    if axis == "x":
        locs = ax.get_xticks()
        if len(locs) > max_ticks:
            step = int(math.ceil(len(locs) / max_ticks))
            ax.set_xticks(locs[::step])
    else:
        locs = ax.get_yticks()
        if len(locs) > max_ticks:
            step = int(math.ceil(len(locs) / max_ticks))
            ax.set_yticks(locs[::step])


def _overlay_hist_kde(
    data: Dict[str, np.ndarray],
    bins: int,
    xlabel: str,
    out_base: Path,
    panel_label: str | None = None,
    legend_anchor: Tuple[float, float] | None = None,
):
    """叠加直方图 + KDE；不显示标题，仅在左上角加 (a)/(b)/(c) 等标记"""
    # 全局范围 & 网格
    all_vals = np.concatenate([v[np.isfinite(v)] for v in data.values() if v is not None])
    x_min, x_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if x_max <= x_min:
        x_max = x_min + 1e-6
    grid = np.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots(figsize=FIGSIZE_OVERLAY)
    # 统一 bins
    edges = np.linspace(x_min, x_max, bins + 1)
    # 绘制
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, vals) in enumerate(data.items()):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            continue
        color = colors[i % len(colors)]
        ax.hist(
            v,
            bins=edges,
            density=True,
            alpha=ALPHA_HIST,
            color=color,
            label=f"{name} (N={v.size})",
            edgecolor="white",
            linewidth=0.3,
        )
        dens = _kde_line(v, grid)
        if np.any(np.isfinite(dens)):
            ax.plot(grid, dens, color=color, linewidth=LINEWIDTH_KDE)

    # 不再设置标题，只保留坐标轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.35)
    if legend_anchor is None:
        # 默认位置（另外两个图还是走这一支）
        ax.legend(frameon=False)
    else:
        # 使用你指定的位置
        ax.legend(
            frameon=False,
            loc="upper right",  # 以右上角为参考点
            bbox_to_anchor=legend_anchor,  # (x, y)，y<1 就是“往下挪一点”
        )

    _limited_ticks(ax, "x", MAX_TICKS_PER_AXIS)

    # 在方框内部左上角添加 (a)/(b)/(c)
    if panel_label is not None:
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
        )

    _save_dual(fig, out_base)


def _hexbin_panels(Ls: Dict[str, np.ndarray], GCs: Dict[str, np.ndarray], outdir: Path,  panel_label: str | None = None, ):
    names = list(Ls.keys())
    k = len(names)
    # 统一坐标范围
    all_L = np.concatenate([Ls[n][np.isfinite(Ls[n])] for n in names])
    all_G = np.concatenate([GCs[n][np.isfinite(GCs[n])] for n in names])
    xmin, xmax = float(np.nanmin(all_L)), float(np.nanmax(all_L))
    ymin, ymax = float(np.nanmin(all_G)), float(np.nanmax(all_G))
    pad_x = 0.02 * (xmax - xmin + 1e-9)
    pad_y = 0.02 * (ymax - ymin + 1e-9)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    # 计算面板行列（尽量接近正方形）
    nrow = int(math.floor(math.sqrt(k)))
    ncol = int(math.ceil(k / max(1, nrow)))
    if nrow * ncol < k:
        nrow = int(math.ceil(k / ncol))

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(FIGSIZE_HEXGRID[0] * ncol, FIGSIZE_HEXGRID[1] * nrow),
        squeeze=False,
    )
    # 在整张图的左上角加 (d)，使用 figure 坐标
    if panel_label is not None:
        fig.text(
            0.01, 0.99,           # (x, y) 在 figure 坐标中的位置，(0,0) 左下，(1,1) 右上
            panel_label,
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
        )

    for i, name in enumerate(names):
        r = i // ncol
        c = i % ncol
        ax = axes[r][c]
        L = Ls[name]
        G = GCs[name]
        m = np.isfinite(L) & np.isfinite(G)
        hb = ax.hexbin(
            L[m],
            G[m],
            gridsize=HEX_GRIDSIZE,
            cmap="viridis",
            mincnt=1,
            extent=(xmin, xmax, ymin, ymax),
        )
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Count")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(name)
        ax.set_xlabel("3'UTR length (nt)")
        if c == 0:
            ax.set_ylabel("GC fraction")
        ax.grid(False)

    # 关闭多余子图
    for j in range(k, nrow * ncol):
        r = j // ncol
        c = j % ncol
        axes[r][c].axis("off")

    # fig.suptitle("Length × GC — hexbin by split", y=0.995, fontsize=12)
    fig.tight_layout()
    _save_dual(fig, outdir / "hexbin_length_gc")


# ---------- 主流程 ----------
def main():
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    # 读取各划分
    split_data = {}
    for name, paths in SPLITS.items():
        if isinstance(paths, (str, Path)):
            paths = [str(paths)]
        df = _read_split(paths)
        if df.empty:
            print(f"[WARN] 划分 {name} 没有有效数据，跳过。")
            continue
        split_data[name] = df

    if len(split_data) == 0:
        raise RuntimeError("没有任何划分的数据，请在 SPLITS 中填写正确的 CSV 路径。")

    # 汇总统计表
    rows = []
    for name, df in split_data.items():
        t = df["target"].astype(float).values
        L = df["utr_len"].astype(float).values
        G = df["gc_frac"].astype(float).values

        t_f = t[np.isfinite(t)]
        L_f = L[np.isfinite(L)]
        G_f = G[np.isfinite(G)]

        mean = float(np.nanmean(t_f)) if t_f.size else np.nan
        median = float(np.nanmedian(t_f)) if t_f.size else np.nan
        iqr = _iqr(t_f) if t_f.size else np.nan
        tmin = float(np.nanmin(t_f)) if t_f.size else np.nan
        tmax = float(np.nanmax(t_f)) if t_f.size else np.nan
        skew, kurt = _skew_kurt(t_f)

        L_mean = float(np.nanmean(L_f)) if L_f.size else np.nan
        L_sd = float(np.nanstd(L_f, ddof=1)) if L_f.size > 1 else np.nan
        G_mean = float(np.nanmean(G_f)) if G_f.size else np.nan
        G_sd = float(np.nanstd(G_f, ddof=1)) if G_f.size > 1 else np.nan

        rows.append(
            {
                "split": name,
                "N": int(t_f.size),
                "target_mean": mean,
                "target_median": median,
                "target_IQR": iqr,
                "target_min": tmin,
                "target_max": tmax,
                "target_skew": skew,
                "target_kurtosis": kurt,
                "len_mean": L_mean,
                "len_sd": L_sd,
                "gc_mean": G_mean,
                "gc_sd": G_sd,
            }
        )

    pd.DataFrame(rows).to_csv(outdir / "summary_by_split.csv", index=False)

    # ---- 分布叠加：目标 / 长度 / GC ----
    # 目标变量
    target_dict = {name: df["target"].values.astype(float) for name, df in split_data.items()}
    bins_t = BINS_TARGET
    # 若想自适应，也可改：bins_t = _freedman_diaconis_bins(np.concatenate(list(target_dict.values())))
    _overlay_hist_kde(
        target_dict,
        bins_t,
        "Half-life",
        outdir / "dist_target_overlay",
        panel_label="(a)",
        legend_anchor=(0.52, 0.9),  # 这里决定“向下移动多少”
    )

    # 长度
    len_dict = {name: df["utr_len"].values.astype(float) for name, df in split_data.items()}
    _overlay_hist_kde(len_dict, BINS_LEN, "3'UTR length (nt)", outdir / "dist_length_overlay", panel_label="(b)")

    # GC
    gc_dict = {name: df["gc_frac"].values.astype(float) for name, df in split_data.items()}
    _overlay_hist_kde(gc_dict, BINS_GC, "GC fraction", outdir / "dist_gc_overlay", panel_label="(c)")

    # ---- 长度 × GC hexbin 面板 ----
    Ls = {name: df["utr_len"].values.astype(float) for name, df in split_data.items()}
    GCs = {name: df["gc_frac"].values.astype(float) for name, df in split_data.items()}
    _hexbin_panels(Ls, GCs, outdir, panel_label="(d)")


    # 配置快照
    with open(outdir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "SPLITS": SPLITS,
                "BINS_TARGET": BINS_TARGET,
                "BINS_LEN": BINS_LEN,
                "BINS_GC": BINS_GC,
                "HEX_GRIDSIZE": HEX_GRIDSIZE,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[完成] 输出目录：", outdir)


if __name__ == "__main__":
    main()
