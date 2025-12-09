# -*- coding: utf-8 -*-
# plot_within_gene_supplement.py  (final robust version)
#
# 作用：
#   Sx：逐基因相关分布（小提琴+箱线）
#       - 优先读 per_gene_summary.csv（pearson_r/spearman_rho）
#       - 若无则用 analysis_per_isoform.csv 的 d_real/d_pre 现场按基因重算
#       - 角标同时给出：宏平均/中位数、Fisher-z 加权均值、pooled within-gene Pearson
#   Sy：基因层中位数散点（ref_real vs ref_pred）
#       - 优先读 cross_gene_median.csv
#       - 若无则从 per_gene_summary.csv 或 analysis_per_isoform.csv 现场计算
#
# 运行：python plot_within_gene_supplement.py
# 仅需修改顶部路径常量即可，无需命令行参数。

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots



# ====== 路径配置（按需修改；当前写成你的常用路径） ======
PER_GENE_CSV = Path(r"F:\mRNA_Project\3UTR\Paper\script\result\gene_isoform_analysis-2\per_gene_summary.csv")
CROSS_GENE_CSV = Path(r"F:\mRNA_Project\3UTR\Paper\script\result\gene_isoform_analysis-2\cross_gene_median.csv")
PER_ISOFORM_CSV = PER_GENE_CSV.parent / "analysis_per_isoform.csv"

OUTDIR = Path(r"F:\mRNA_Project\3UTR\Paper\plots\result\plot\supp_within_gene-2")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
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
    "figure.dpi": 400
    # "axes.titleweight": "bold",  # 图标题
    # "axes.labelweight": "bold",  # x / y 轴标签
})


# ---------- 小工具 ----------
def pick_col(cols, candidates):
    """宽松匹配列名（大小写/下划线/子串都容忍）"""
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k in low: return low[k]
    def canon(s): return "".join(ch for ch in s.lower() if ch.isalnum())
    candz = [canon(k) for k in candidates]
    for c in cols:
        cc = canon(c)
        for ck in candz:
            if ck in cc or cc in ck:
                return c
    return None

def ensure_numeric(df, col):
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def qtrim_limits(x, y=None, q=(0.01, 0.99), pad=0.03):
    """分位数裁剪坐标范围，减少极端点带来的空白。"""
    arr = np.asarray(x, float); arr = arr[np.isfinite(arr)]
    if y is not None:
        arr2 = np.asarray(y, float); arr2 = arr2[np.isfinite(arr2)]
        arr = np.r_[arr, arr2]
    if arr.size == 0:
        return (0.0, 1.0)
    lo, hi = np.quantile(arr, q)
    span = max(1e-12, hi - lo)
    return (float(lo - pad*span), float(hi + pad*span))

def corr_pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def corr_spearman(x, y):
    xr = pd.Series(x).rank().to_numpy(dtype=float)
    yr = pd.Series(y).rank().to_numpy(dtype=float)
    m = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[m], yr[m]
    if xr.size < 2 or np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])

def fisher_z_mean(rs, ns):
    """Fisher-z 加权均值（权重 ~ n-3）。"""
    vals, wts = [], []
    for r, n in zip(rs, ns):
        if pd.notna(r) and np.isfinite(r) and abs(r) < 1 and pd.notna(n):
            if n >= 4:
                vals.append(np.arctanh(r))
                wts.append(max(int(n) - 3, 1))
    if not vals:
        return np.nan
    zbar = np.average(vals, weights=wts)
    return float(np.tanh(zbar))


# ---------- 读取 per_gene_summary.csv -> 直接取 per-gene 系列 ----------
def try_load_per_gene_series(per_gene_csv: Path):
    if not per_gene_csv.exists():
        return None

    df = pd.read_csv(per_gene_csv)
    cols = list(df.columns)

    pear_col  = pick_col(cols, ["pearson_r","pearson","r_pearson"])
    spear_col = pick_col(cols, ["spearman_rho","spearman","rho"])
    # 你文件里常见 n_isoforms_x / n_isoforms_y，这里优先 _x
    n_col     = pick_col(cols, ["n_isoforms_x","n_isoforms","n_isoforms_y","n","count","num_isoforms","size"])

    for c in [pear_col, spear_col, n_col]:
        df = ensure_numeric(df, c)

    # 先尝试 n>=2 过滤；若过滤后空则撤销（防误杀）
    df_orig = df.copy()
    if n_col and n_col in df.columns:
        df = df[df[n_col] >= 3].copy()
        if df.empty:
            df = df_orig

    # 收集 per-gene 序列
    series_list, labels, stat_lines = [], [], []
    # Pearson per-gene
    if pear_col and pear_col in df.columns:
        s = df[pear_col].replace([np.inf,-np.inf], np.nan).dropna()
        if not s.empty:
            series_list.append(s.to_numpy())
            labels.append("Pearson (per-gene)")
            stat_lines.append(f"Pearson: mean={s.mean():.3f}, median={s.median():.3f}")
    # Spearman per-gene
    if spear_col and spear_col in df.columns:
        s = df[spear_col].replace([np.inf,-np.inf], np.nan).dropna()
        if not s.empty:
            series_list.append(s.to_numpy())
            labels.append("Spearman (per-gene)")
            stat_lines.append(f"Spearman: mean={s.mean():.3f}, median={s.median():.3f}")

    # 追加 Fisher-z 加权（基于 per_gene_summary）
    if pear_col and n_col and pear_col in df.columns and n_col in df.columns:
        r_list = pd.to_numeric(df[pear_col], errors="coerce").to_numpy()
        n_list = pd.to_numeric(df[n_col], errors="coerce").to_numpy()
        fz = fisher_z_mean(r_list, n_list)
        if np.isfinite(fz):
            stat_lines.append(rf"Fisher-z weighted Pearson $\approx$ {fz:.3f}")

    # 追加 pooled within-gene（需要 analysis_per_isoform.csv）
    if PER_ISOFORM_CSV.exists():
        dfi = pd.read_csv(PER_ISOFORM_CSV)
        gcol  = pick_col(dfi.columns, ["gene","systematic_name","gene_name","orf","name"])
        drcol = pick_col(dfi.columns, ["d_real","dreal","real_delta_demean"])
        dpcol = pick_col(dfi.columns, ["d_pre","dpre","pred_delta_demean"])
        if gcol and drcol and dpcol:
            x = pd.to_numeric(dfi[drcol], errors="coerce").to_numpy()
            y = pd.to_numeric(dfi[dpcol], errors="coerce").to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
                pooled = float(np.corrcoef(x, y)[0, 1])
                stat_lines.append(rf"Pooled within-gene Pearson $\approx$ {pooled:.3f}")

    if series_list:
        return series_list, labels, stat_lines
    return None


# ---------- 从 analysis_per_isoform.csv 现场计算 per-gene 系列 ----------
def compute_series_from_isoform(per_isoform_csv: Path):
    if not per_isoform_csv.exists():
        return None

    dfi = pd.read_csv(per_isoform_csv)
    cols = list(dfi.columns)

    gene_col  = pick_col(cols, ["gene","systematic_name","gene_name","orf","name"])
    dreal_col = pick_col(cols, ["d_real","dreal","real_delta_demean"])
    dpre_col  = pick_col(cols, ["d_pre","dpre","pred_delta_demean"])

    if not (gene_col and dreal_col and dpre_col):
        return None

    dfi = dfi.dropna(subset=[gene_col, dreal_col, dpre_col]).copy()

    pears, speas = [], []
    for _, g in dfi.groupby(gene_col, dropna=True):
        x = g[dreal_col].to_numpy()
        y = g[dpre_col].to_numpy()
        if x.size >= 3 and np.std(x) > 0 and np.std(y) > 0:
            pears.append(corr_pearson(x, y))
            speas.append(corr_spearman(x, y))

    series_list, labels, stat_lines = [], [], []
    s = pd.Series(pears).replace([np.inf,-np.inf], np.nan).dropna()
    if not s.empty:
        series_list.append(s.to_numpy())
        labels.append("Pearson (per-gene, computed)")
        stat_lines.append(f"Pearson: mean={s.mean():.3f}, median={s.median():.3f}")
    s2 = pd.Series(speas).replace([np.inf,-np.inf], np.nan).dropna()
    if not s2.empty:
        series_list.append(s2.to_numpy())
        labels.append("Spearman (per-gene, computed)")
        stat_lines.append(f"Spearman: mean={s2.mean():.3f}, median={s2.median():.3f}")

    # 追加 pooled within-gene（同一次读取直接算）
    x_all = pd.to_numeric(dfi[dreal_col], errors="coerce").to_numpy()
    y_all = pd.to_numeric(dfi[dpre_col], errors="coerce").to_numpy()
    m = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all = x_all[m], y_all[m]
    if x_all.size >= 2 and np.std(x_all) > 0 and np.std(y_all) > 0:
        pooled = float(np.corrcoef(x_all, y_all)[0, 1])
        stat_lines.append(rf"Pooled within-gene Pearson $\approx$≈ {pooled:.3f}")

    if series_list:
        return series_list, labels, stat_lines
    return None


# ---------- Sy：交叉基因中位数散点 ----------
def plot_cross_gene_from_df(g_df: pd.DataFrame, outdir: Path, title_suffix=""):
    # 需要列：ref_real / ref_pred
    if not {"ref_real", "ref_pred"}.issubset(set(g_df.columns)):
        return
    x = pd.to_numeric(g_df["ref_real"], errors="coerce").to_numpy()
    y = pd.to_numeric(g_df["ref_pred"], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return

    # 最小二乘 y = a x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a*x + b
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((x - np.mean(x))**2))
    # r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan
    r2 =  1.0 - np.sum((y - x)**2) / np.sum((y - np.mean(y))**2)
    lo, hi = qtrim_limits(x, y, q=(0.01, 0.99), pad=0.03)
    xs = np.array([lo, hi])

    # 散点图：图像物理比例 1:1，坐标范围也对称
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    ax2.scatter(x, y, s=10, alpha=0.6, linewidth=0)
    ax2.plot(xs, xs, "--", lw=1.2, label="y=x")
    ax2.plot(xs, a * xs + b, lw=1.5, label=f"fit: y={a:.2f}x+{b:.2f}")
    ax2.set_xlim(lo, hi);
    ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal", adjustable="box")  # 确保 x/y 轴刻度比例一致
    ax2.set_xlabel("Gene-level median (truth)")
    ax2.set_ylabel("Gene-level median (pred)")
    # ax2.set_title(rf"Cross-gene baseline{title_suffix}  ($R^2 \approx {r2:.3f}$)")
    ax2.legend(frameon=False, loc="lower right", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / "cross_gene_median_scatter.png")
    fig2.savefig(outdir / "cross_gene_median_scatter.svg")


def try_plot_cross_gene(per_gene_csv: Path, cross_csv: Path, per_isoform_csv: Path, outdir: Path):
    # 1) 优先用 cross_gene_median.csv
    if cross_csv.exists():
        d2 = pd.read_csv(cross_csv).dropna()
        xcol = pick_col(d2.columns, ["ref_real","median_real","real_ref","truth_median"])
        ycol = pick_col(d2.columns, ["ref_pred","median_pred","pred_ref","prediction_median"])
        if xcol and ycol:
            g = d2.rename(columns={xcol: "ref_real", ycol: "ref_pred"})
            plot_cross_gene_from_df(g, outdir, title_suffix="")
            return

    # 2) 若 per_gene_summary.csv 含 ref_real/ref_pre，直接用
    if per_gene_csv.exists():
        df = pd.read_csv(per_gene_csv)
        if {"ref_real", "ref_pre"}.issubset(set(df.columns)):
            g = df[["ref_real","ref_pre"]].dropna().rename(columns={"ref_pre":"ref_pred"})
            if not g.empty:
                plot_cross_gene_from_df(g, outdir, title_suffix=" (from per_gene_summary)")
                return

    # 3) 兜底：从 analysis_per_isoform.csv 现场计算基因中位数
    if per_isoform_csv.exists():
        dfi = pd.read_csv(per_isoform_csv)
        gene_col  = pick_col(dfi.columns, ["gene","systematic_name","gene_name","orf","name"])
        ios_r_col = pick_col(dfi.columns, ["ios_real","real","true","y_true","half_life","halflife","half-life"])
        ios_p_col = pick_col(dfi.columns, ["ios_pre","pred","y_pred","prediction","half_life_pred"])
        if gene_col and ios_r_col and ios_p_col:
            g = (dfi.groupby(gene_col, as_index=False)
                   .agg(ref_real=(ios_r_col, "median"),
                        ref_pred=(ios_p_col, "median"),
                        n=("gene" if "gene" in dfi.columns else gene_col, "size")))
            if not g.empty:
                plot_cross_gene_from_df(g, outdir, title_suffix=" (computed)")


# ---------- 主流程 ----------
def main():
    # ====== Sx：逐基因相关分布 ======
    series_pack = try_load_per_gene_series(PER_GENE_CSV)
    if series_pack is None:
        series_pack = compute_series_from_isoform(PER_ISOFORM_CSV)

    if series_pack is None:
        raise ValueError(
            "无法从 per_gene_summary.csv 或 analysis_per_isoform.csv 获取逐基因 Pearson/Spearman 数据。\n"
            f"请检查：\n"
            f"1) {PER_GENE_CSV} 是否包含 'pearson_r' / 'spearman_rho' 数值列；或\n"
            f"2) {PER_ISOFORM_CSV} 是否包含 'gene'、'd_real'、'd_pre' 数值列。"
        )

    series_list, labels, stat_lines = series_pack

    # ====== Sx：per-gene 相关分布（violin）======
    # 小提琴图：4:3 比例
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.violinplot(series_list, showmeans=False, showmedians=False, showextrema=False)
    ax.boxplot(series_list, widths=0.2, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    ax.set_ylabel("Correlation")
    # ax.set_title("Per-gene correlation distribution (within-gene)")
    if stat_lines:
        ax.text(0.29, 0.02, "\n".join(stat_lines), transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "per_gene_corr_violin.png")
    fig.savefig(OUTDIR / "per_gene_corr_violin.svg")

    # ====== Sx：per-gene 相关分布（箱线图 + 抖动散点）======
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
    positions = np.arange(1, len(series_list) + 1)

    # 箱线图本体
    ax2.boxplot(
        series_list,
        positions=positions,
        widths=0.4,
        vert=True,
        showfliers=False,
        patch_artist=False,
    )

    # 叠加抖动散点，显示每个基因的相关系数
    max_points = 400
    for i, vals in enumerate(series_list, start=1):
        vals_arr = np.asarray(vals, dtype=float)
        vals_arr = vals_arr[np.isfinite(vals_arr)]
        if vals_arr.size == 0:
            continue
        if vals_arr.size > max_points:
            idx = np.linspace(0, vals_arr.size - 1, max_points, dtype=int)
            vals_arr = vals_arr[idx]
        x_jitter = np.random.normal(loc=i, scale=0.04, size=vals_arr.size)
        ax2.scatter(x_jitter, vals_arr, s=6, alpha=0.25, linewidths=0)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Correlation")
    # ax2.set_title("Per-gene correlation distribution (within-gene)")
    if stat_lines:
        ax2.text(0.3, 0.1, "\n".join(stat_lines), transform=ax2.transAxes,
                 fontsize=9, va="bottom", ha="left")
    ax2.grid(True, linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "per_gene_corr_box_jitter.png")
    fig2.savefig(OUTDIR / "per_gene_corr_box_jitter.svg")

    # ====== Sy：交叉基因中位数散点 ======
    try_plot_cross_gene(PER_GENE_CSV, CROSS_GENE_CSV, PER_ISOFORM_CSV, OUTDIR)

    print(f"[OK] 输出已保存到：{OUTDIR}")

if __name__ == "__main__":
    main()
