# -*- coding: utf-8 -*-
"""
plot_kmer_error_source_summary_v2.py

升级点：
- 若存在 `kmer_enrichment_results_all.csv`，优先用它汇总（k、tag、log2fc、q、top_rate、bot_rate…），
  无需再依赖分文件；如不存在，则回退到各单独CSV。
- 支持 k=6 的位置热图（positional_heatmap_k6*.csv）：若提供，则 k=6 面板也按峰值bin上色；
  否则k=6统一灰色。
- 输出仍为：result/plot/kmer_error_source_summary/<timestamp>/ 下：
  - kmer_error_source_bubble_grid.png/svg（2×3气泡图）
  - kmer_sig_counts_bar.png/svg（显著计数条形图）
  - sig_counts_summary.csv（显著计数汇总）
  - top_each_cell.csv（各格TopN表，含峰值bin/出现率）（k=5/6均支持）

使用：
1) 在“手动配置”区域填写你的项目根目录与文件路径（特别是：
   - `PATH_RESULTS_ALL`（可选）
   - k=5/6 的位置热图 CSV：`PATH_POS_K5_*` 与 `PATH_POS_K6_*`
2) 运行：python plot_kmer_error_source_summary_v2.py
"""

from __future__ import annotations
import os, math, datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============== 手动配置（按你的实际路径修改） ==================
PROJECT_ROOT = Path(r"F:\mRNA_Project\3UTR\Paper")   # ← 改成你的项目根
OUT_PARENT   = PROJECT_ROOT / "result" / "plot" / "kmer_error_source_summary"

# （可选，优先使用）总汇表：由 plot_kmer_enrichment.py 生成
PATH_RESULTS_ALL = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_results_all.csv"

# 回退用的分文件路径（如没 results_all 才用）
PATH_K5_ABS    = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_k5.csv"
PATH_K5_POS    = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_k5_posres.csv"
PATH_K5_NEG    = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_k5_negres.csv"
PATH_K6_ABS    = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_k6.csv"
PATH_K6_POS    = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_k6_posres.csv"
PATH_K6_NEG    = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "kmer_enrichment_k6_negres.csv"

# 位置热图（用于峰值bin上色）：k=5 与 k=6 都支持（给哪个就上哪个的色）
PATH_POS_K5_ALL = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "positional_heatmap_k5.csv"
PATH_POS_K5_POS = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "positional_heatmap_k5_posres.csv"
PATH_POS_K5_NEG = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "positional_heatmap_k5_negres.csv"
PATH_POS_K6_ALL = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "positional_heatmap_k6.csv"
PATH_POS_K6_POS = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "positional_heatmap_k6_posres.csv"
PATH_POS_K6_NEG = PROJECT_ROOT / "result" / "plot" / "kmer_enrichment" / "positional_heatmap_k6_negres.csv"

# 可视化与统计
FDR_Q      = 0.05
TOP_LABELS = 12      # 每格标注数量
DPI        = 400
SEED       = 2025

# 字号（可以按需要微调）
FONT_SIZE_BASE   = 12   # 全局基础字体
FONT_SIZE_TITLE  = 14   # 图标题
FONT_SIZE_LABEL  = 12   # 坐标轴标题
FONT_SIZE_TICK   = 11   # 坐标刻度
FONT_SIZE_LEGEND = 11   # 图例
FONT_SIZE_CBAR   = 11   # 色标（colorbar）刻度/标题
# ================================================================


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def _prep_cols(df: pd.DataFrame) -> pd.DataFrame:
    """统一列：kmer/log2fc/q/top_rate/bot_rate 以及 rate_diff（若缺则NaN）"""
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in df.columns: return c
            lc = c.lower()
            if lc in cols: return cols[lc]
        return None
    kmer     = pick("kmer")
    log2fc   = pick("log2fc")
    q        = pick("q", "fdr", "q_value")
    top_rate = pick("top_rate", "top_presence", "top_prop")
    bot_rate = pick("bot_rate", "bottom_rate", "bot_prop")
    keep = [x for x in [kmer, log2fc, q, top_rate, bot_rate] if x]
    out = df[keep].rename(columns={
        kmer: "kmer", log2fc: "log2fc", q: "q",
        top_rate: "top_rate", bot_rate: "bot_rate"
    })
    out["rate_diff"] = (out["top_rate"] - out["bot_rate"]).abs() if "top_rate" in out.columns and "bot_rate" in out.columns else np.nan
    return out

def _read_peaks(pos_csv: Path) -> Optional[pd.DataFrame]:
    """从位置热图CSV计算峰值bin与峰值出现率"""
    df = _read_csv(pos_csv)
    if df is None or df.empty:
        return None
    bins = [c for c in df.columns if str(c).startswith("bin_")]
    first = df.columns[0]
    arr = df[bins].to_numpy()
    peak_idx = arr.argmax(axis=1)
    peak_val = arr.max(axis=1)
    return pd.DataFrame({"kmer": df[first], "peak_bin": peak_idx.astype(int), "peak_rate": peak_val})

def _merge_peak(df: Optional[pd.DataFrame], peaks: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None: return None
    if peaks is None or peaks.empty:
        dft = df.copy()
        dft["peak_bin"] = np.nan; dft["peak_rate"] = np.nan
        return dft
    return df.merge(peaks, on="kmer", how="left")

def _sig_counts(df: Optional[pd.DataFrame], q_th=0.05) -> Dict[str, int]:
    if df is None or df.empty:
        return {"sig_total":0, "sig_pos":0, "sig_neg":0}
    m = df["q"] < q_th
    return {
        "sig_total": int(m.sum()),
        "sig_pos": int(((df["log2fc"]>0) & m).sum()),
        "sig_neg": int(((df["log2fc"]<0) & m).sum()),
    }

def _ythr(q):
    return -math.log10(max(q, 1e-300))

def main():
    np.random.seed(SEED)
    outdir = OUT_PARENT / _ts()
    _ensure_dir(outdir)

    # 全局字体：非斜体
    plt.rcParams.update({
        "font.style": "normal",
        "font.size": 10,
        "svg.fonttype": "none",
        "axes.unicode_minus": False
    })

    # 1) 读 results_all 或回退到分文件
    using_all = False
    sets: Dict[Tuple[str,str], Optional[pd.DataFrame]] = { }  # key=(k, subset)

    res_all = _read_csv(PATH_RESULTS_ALL)
    if res_all is not None and not res_all.empty:
        # 期待至少有列：k, tag, kmer, log2fc, q, top_rate, bot_rate
        using_all = True
        res_all["k"] = res_all["k"].astype(str).str.replace(".0","", regex=False)
        for (k, subset) in [("5","abs"),("5","posres"),("5","negres"),
                            ("6","abs"),("6","posres"),("6","negres")]:
            sub = res_all[(res_all["k"].astype(str)==k) & (res_all["tag"]==subset)].copy()
            sets[(f"k{k}", "abs" if subset=="abs" else ("pos" if subset=="posres" else "neg"))] = _prep_cols(sub)
    else:
        # 回退：逐文件读取
        def rd(p):
            df = _read_csv(p);
            return _prep_cols(df) if df is not None else None
        sets[("k5","abs")] = rd(PATH_K5_ABS)
        sets[("k5","pos")] = rd(PATH_K5_POS)
        sets[("k5","neg")] = rd(PATH_K5_NEG)
        sets[("k6","abs")] = rd(PATH_K6_ABS)
        sets[("k6","pos")] = rd(PATH_K6_POS)
        sets[("k6","neg")] = rd(PATH_K6_NEG)

    # 2) 读位置热图峰值（k=5/k=6 各自独立）
    peaks = {
        ("k5","abs"): _read_peaks(PATH_POS_K5_ALL),
        ("k5","pos"): _read_peaks(PATH_POS_K5_POS),
        ("k5","neg"): _read_peaks(PATH_POS_K5_NEG),
        ("k6","abs"): _read_peaks(PATH_POS_K6_ALL),
        ("k6","pos"): _read_peaks(PATH_POS_K6_POS),
        ("k6","neg"): _read_peaks(PATH_POS_K6_NEG),
    }

    # 合并峰值
    for key in list(sets.keys()):
        sets[key] = _merge_peak(sets[key], peaks.get(key))

    # 3) 显著计数汇总
    rows = []
    for (k, subset), df in sets.items():
        cnt = _sig_counts(df, q_th=FDR_Q)
        rows.append({"k":k, "subset":subset, **cnt})
    counts = pd.DataFrame(rows)
    counts.to_csv(outdir / "sig_counts_summary.csv", index=False, encoding="utf-8-sig")

    # 4) 取各格Top表
    def pick_top(df: Optional[pd.DataFrame], k: str, subset: str, topn=TOP_LABELS):
        if df is None or df.empty:
            return pd.DataFrame(columns=["k","subset","kmer","log2fc","q","top_rate","bot_rate","rate_diff","peak_bin","peak_rate"])
        t = df.sort_values(["q", df["log2fc"].abs().name], ascending=[True, False]).head(topn).copy()
        t.insert(0, "subset", subset); t.insert(0, "k", k)
        return t

    top_tables = [pick_top(df, k, subset, TOP_LABELS) for (k, subset), df in sets.items()]
    top_all = pd.concat(top_tables, ignore_index=True)
    top_all.to_csv(outdir / "top_each_cell.csv", index=False, encoding="utf-8-sig")

    # 5) 2×3 气泡图
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    row_keys = ["k5","k6"]
    col_keys = [("abs","绝对误差 abs"), ("pos","低估 posres"), ("neg","高估 negres")]
    y_thr = _ythr(FDR_Q)

    for i, kmer_k in enumerate(row_keys):
        for j, (subset, title) in enumerate(col_keys):
            ax = axes[i, j]
            ax.axhline(y=y_thr, lw=1, ls="--")
            ax.axvline(x=0, lw=1, ls="--")
            ax.set_title(f"{kmer_k.upper()} • {title}")
            ax.set_xlabel("log2FC (top/bottom)")
            ax.set_ylabel("-log10(FDR q)" if j==0 else "")

            df = sets.get((kmer_k, subset))
            if df is None or df.empty:
                ax.text(0.5,0.5,"无数据", ha="center", va="center", transform=ax.transAxes)
                continue

            size = df["rate_diff"].fillna(0.0).values
            size = 100.0 * (size/size.max()) + 10.0 if np.nanmax(size)>0 else np.full(len(df), 30.0)

            # 若该(k,subset)有峰值bin，就上色；否则灰色
            if "peak_bin" in df.columns and df["peak_bin"].notna().any():
                sc = ax.scatter(df["log2fc"], -np.log10(np.clip(df["q"].values,1e-300,1.0)),
                                s=size, c=df["peak_bin"].fillna(-1).values, cmap="viridis", alpha=0.7, edgecolor="none")
                # 每一行的第一个子图放一个色条
                if j == 0:
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.set_label("peak_bin")
            else:
                ax.scatter(df["log2fc"], -np.log10(np.clip(df["q"].values,1e-300,1.0)),
                           s=size, color="0.5", alpha=0.6, edgecolor="none")

            # 标注前 TOP_LABELS
            t = df.sort_values(["q", df["log2fc"].abs().name], ascending=[True, False]).head(TOP_LABELS)
            for _, r in t.iterrows():
                ax.text(r["log2fc"], _ythr(r["q"]), r["kmer"], fontsize=7, ha="center", va="bottom")

            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_locator(MaxNLocator(5))

    for suf in ["png","svg"]:
        fig.savefig(outdir / f"kmer_error_source_bubble_grid.{suf}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # 6) 显著计数条形图（方向分开）
    fig2, ax2 = plt.subplots(figsize=(8, 4), constrained_layout=True)
    plot_rows = []
    for (k, subset), df in sets.items():
        cnt = _sig_counts(df, q_th=FDR_Q)
        plot_rows.append({"group": f"{k}-{subset}", "pos": cnt["sig_pos"], "neg": cnt["sig_neg"]})
    bar_df = pd.DataFrame(plot_rows)
    x = np.arange(len(bar_df))
    ax2.bar(x-0.2, bar_df["pos"], width=0.4, label="正向富集（log2FC>0）")
    ax2.bar(x+0.2, bar_df["neg"], width=0.4, label="负向富集（log2FC<0）")
    ax2.set_xticks(x); ax2.set_xticklabels(bar_df["group"])
    ax2.set_ylabel("显著k-mer数量 (q<%.2g)" % FDR_Q)
    ax2.set_title("显著k-mer数量（方向分开）"); ax2.legend()
    for suf in ["png","svg"]:
        fig2.savefig(outdir / f"kmer_sig_counts_bar.{suf}", dpi=DPI, bbox_inches="tight")
    plt.close(fig2)

    print("[OK] 输出目录：", outdir)
    print(" - 2×3气泡图：kmer_error_source_bubble_grid.png/svg")
    print(" - 显著计数：kmer_sig_counts_bar.png/svg")
    print(" - 汇总：sig_counts_summary.csv, top_each_cell.csv")
    print(" - 数据来源：", "results_all（优先）" if using_all else "分文件（回退）")


if __name__ == "__main__":
    main()
