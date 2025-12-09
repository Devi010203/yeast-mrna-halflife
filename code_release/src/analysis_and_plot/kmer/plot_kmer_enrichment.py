# plots/plot_kmer_enrichment.py
# -*- coding: utf-8 -*-
"""
误差驱动与方向性（高估/低估）k-mer 富集分析（论文用图）
- 读取 RUN_DIR/final_test_predictions.csv（需包含：sequence, true, pred）
- 计算 residual = true - pred，abs_error = |residual|
- 三类富集：
  1) 绝对误差 |error|：top vs bottom
  2) 方向性“低估”（正残差）：在 residual > 0 的子集内，按 |residual| 切分 top vs bottom
  3) 方向性“高估”（负残差）：在 residual < 0 的子集内，按 |residual| 切分 top vs bottom
- 每类均输出：火山图、含/不含 |error| 条形图、归一化位置热图；及 CSV 结果

使用：
1) 修改 CONFIG["RUN_DIR"]（含 final_test_predictions.csv）
2) 运行：python plots/plot_kmer_enrichment.py
3) 输出：<项目根>/result/plot/kmer_enrichment/ 下的 .png/.svg（dpi=400）与若干 CSV
"""

import os, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')


# ========== 手动填写 ==========
CONFIG = {
    "RUN_DIR": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01",  # 里面要有 final_test_predictions.csv
    "SAVE_SUBDIR": "kmer_enrichment",

    # k-mer 设置
    "K_LIST": [5, 6],            # 同时跑 5-mer 与 6-mer
    "TOP_PCT": 0.10,             # |error| 前 10% 视为 top，后 10% 为 bottom（方向性子集也用这个比例）
    "ALPHABET": "ACGU",         # RNA/UTR 常见字符；有 T 也没关系，非字母表字符会被忽略
    "MIN_OCC": 20,               # 过滤总出现过少的 kmer（在当前比较的样本集合中统计）

    # 显著性与可视化
    "FDR_Q": 0.05,
    "VOLCANO_TOPN": 15,          # 火山图标注 top N
    "BAR_TOPN": 12,              # 含/不含对比图展示 top N
    "NORM_POS_BINS": 20,         # 位置热图：序列归一化切成多少段
    "HEATMAP_TOPN": 12,          # 位置热图展示的基序个数

    # 方向性富集开关
    "DO_DIRECTIONAL": True,      # True 则额外做 posres/negres 两组

    # 作图风格
    "DPI": 400,
    "FIGSIZE": (6.0, 4.5),
    "GRID_ALPHA": 0.35,
}
# ============================

# ---------- 基础工具 ----------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_outdir() -> str:
    outdir = _project_root() / "result" / "plot" / CONFIG["SAVE_SUBDIR"]
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)

def _save_dual(fig, out_base: str):
    fig.savefig(out_base + ".png", dpi=CONFIG["DPI"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", dpi=CONFIG["DPI"], bbox_inches="tight")
    plt.close(fig)

def _set_fonts():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans CJK SC"],
        "font.style": "normal",
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
    })

def _safe_read_test(run_dir: str) -> pd.DataFrame:
    fp = os.path.join(run_dir, "final_test_predictions.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"未找到 {fp}")
    df = pd.read_csv(fp)
    need = {"sequence", "true", "pred"}
    if not need.issubset(df.columns):
        raise ValueError(f"{fp} 需要包含列：{need}")
    df = df.dropna(subset=["sequence","true","pred"]).copy()
    df["sequence"] = df["sequence"].astype(str)
    df["true"] = df["true"].astype(float)
    df["pred"] = df["pred"].astype(float)
    df["residual"] = df["true"] - df["pred"]
    df["abs_error"] = np.abs(df["residual"])
    return df

# ---------- 组合学与统计 ----------
def _scan_kmers(seq: str, k: int, alphabet: set) -> set:
    """返回序列中所有出现过的 k-mer（去重，用于 presence 统计）"""
    s = seq.upper()
    found = set()
    for i in range(0, len(s) - k + 1):
        kmer = s[i:i+k]
        if set(kmer) <= alphabet:
            found.add(kmer)
    return found

def _count_kmers_presence(df_sub: pd.DataFrame, k: int, alphabet: str):
    """在给定子集 df_sub 中统计 presence；返回 (presence_rows, vocab_set)"""
    alpha = set(alphabet)
    vocab = {}
    rows = []
    for _, row in df_sub.iterrows():
        kmers = _scan_kmers(row["sequence"], k, alpha)
        rows.append(kmers)
        for km in kmers:
            vocab[km] = vocab.get(km, 0) + 1
    vocab = {km:c for km,c in vocab.items() if c >= CONFIG["MIN_OCC"]}
    return rows, set(vocab.keys())

def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR"""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=float); ranks[order] = np.arange(1, n+1)
    q = pvals * n / ranks
    for i in range(n-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return np.clip(q, 0, 1)

def _fisher_two_sided_p(a, b, c, d):
    """
    2x2 Fisher 精确检验（双侧）
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

    def logpmf(x):
        return logC(col1, x) + logC(col2, row1 - x) - logC(N, row1)

    p_obs = math.exp(logpmf(a))
    x_min = max(0, row1 - col2)
    x_max = min(row1, col1)
    p = 0.0
    for x in range(x_min, x_max + 1):
        px = math.exp(logpmf(x))
        if px <= p_obs + 1e-15:
            p += px
    return min(max(p, 0.0), 1.0)

# ---------- 富集核心：对任意“子集 df_sub”做 top vs bottom ----------
def _enrichment_for_k_core(df_sub: pd.DataFrame, k: int, alphabet: str, top_pct: float) -> pd.DataFrame:
    """
    在给定子集 df_sub 内做 presence 富集（top vs bottom，按 abs_error 排序）
    - 不改变 df_sub.index（用原始索引判断分组，避免错位）
    - 显著性：Fisher 双侧；多重：BH-FDR
    返回列：["k","kmer","a","b","c","d","top_rate","bot_rate","log2fc","p","q"]
    """
    n = len(df_sub)
    if n == 0:
        return pd.DataFrame(columns=["k","kmer","a","b","c","d","top_rate","bot_rate","log2fc","p","q"])

    top_n = max(1, int(round(top_pct * n)))
    bottom_n = top_n

    # 不 reset_index，保持原索引坐标系
    df_desc = df_sub.sort_values("abs_error", ascending=False)
    df_asc  = df_sub.sort_values("abs_error", ascending=True)
    top_idx = set(df_desc.index[:top_n].tolist())
    bot_idx = set(df_asc.index[:bottom_n].tolist())

    # 仅在当前子集上统计 presence 与词表
    presence_rows, vocab = _count_kmers_presence(df_sub, k, alphabet)

    results = []
    for km in sorted(vocab):
        a = b = c = d = 0
        for orig_idx, kmers in zip(df_sub.index, presence_rows):
            has = (km in kmers)
            if orig_idx in top_idx:
                if has: a += 1
                else:   b += 1
            elif orig_idx in bot_idx:
                if has: c += 1
                else:   d += 1

        # 过滤无信息/过少（在子集内）
        if (a + c) < CONFIG["MIN_OCC"]:
            continue

        top_rate = a / max(1, (a + b))
        bot_rate = c / max(1, (c + d))
        eps = 1e-9
        log2fc = math.log2((top_rate + eps) / (bot_rate + eps))
        p = _fisher_two_sided_p(a, b, c, d)
        results.append([k, km, a, b, c, d, top_rate, bot_rate, log2fc, p])

    if not results:
        return pd.DataFrame(columns=["k","kmer","a","b","c","d","top_rate","bot_rate","log2fc","p","q"])

    out = pd.DataFrame(results, columns=["k","kmer","a","b","c","d","top_rate","bot_rate","log2fc","p"])
    out["q"] = _fdr_bh(out["p"].to_numpy())
    out = out.sort_values(["q", "log2fc"], ascending=[True, False]).reset_index(drop=True)
    return out

# 便捷封装：三种场景的 df_sub
def _enrich_abs(df_all: pd.DataFrame, k: int, alphabet: str) -> pd.DataFrame:
    return _enrichment_for_k_core(df_all, k, alphabet, CONFIG["TOP_PCT"])

def _enrich_posres(df_all: pd.DataFrame, k: int, alphabet: str) -> pd.DataFrame:
    df_sub = df_all[df_all["residual"] > 0]
    return _enrichment_for_k_core(df_sub, k, alphabet, CONFIG["TOP_PCT"])

def _enrich_negres(df_all: pd.DataFrame, k: int, alphabet: str) -> pd.DataFrame:
    df_sub = df_all[df_all["residual"] < 0]
    return _enrichment_for_k_core(df_sub, k, alphabet, CONFIG["TOP_PCT"])

# ---------- 作图 ----------
def _volcano_plot(dfk: pd.DataFrame, outdir: str, k: int, tag: str = ""):
    if dfk.empty: return
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    x = dfk["log2fc"].to_numpy()
    y = -np.log10(np.clip(dfk["q"].to_numpy(), 1e-300, 1.0))
    ax.scatter(x, y, s=14, alpha=0.7)
    ax.axhline(-math.log10(CONFIG["FDR_Q"]), linestyle="--", linewidth=1.0, color="k")
    ax.set_xlabel("log2FC (presence rate: top vs bottom)")
    ttl = f"K={k} k-mer enrichment (volcano)"
    if tag: ttl += f" — {tag}"
    ax.set_title(ttl)
    ax.set_ylabel("-log10(FDR)")
    ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    # 标注 topN
    topn = min(CONFIG["VOLCANO_TOPN"], len(dfk))
    for _, r in dfk.head(topn).iterrows():
        ax.annotate(r["kmer"], (r["log2fc"], -math.log10(max(r["q"], 1e-300))),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    suffix = f"_k{k}{('_' + tag) if tag else ''}"
    _save_dual(fig, os.path.join(outdir, "volcano" + suffix))

def _bar_error_diff(df_sub: pd.DataFrame, dfk: pd.DataFrame, outdir: str, k: int, tag: str = ""):
    """对 topN 基序，比较“含/不含”的 |error| 均值差异（在 df_sub 上计算）"""
    if dfk.empty or df_sub.empty: return
    topN = min(CONFIG["BAR_TOPN"], len(dfk))
    kmers = dfk.head(topN)["kmer"].tolist()
    means_in, means_out, counts_in, counts_out = [], [], [], []
    for km in kmers:
        mask = df_sub["sequence"].str.contains(km)
        ae_in = df_sub.loc[mask, "abs_error"].to_numpy()
        ae_out = df_sub.loc[~mask, "abs_error"].to_numpy()
        means_in.append(np.mean(ae_in) if len(ae_in) else np.nan)
        means_out.append(np.mean(ae_out) if len(ae_out) else np.nan)
        counts_in.append(int(len(ae_in)))
        counts_out.append(int(len(ae_out)))
    x = np.arange(len(kmers))
    width = 0.45
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    ax.bar(x - width/2, means_in, width, label="contains", alpha=0.9)
    ax.bar(x + width/2, means_out, width, label="not contains", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(kmers, rotation=40, ha="right")
    ax.set_ylabel("Mean |error|")
    ttl = f"K={k} top motifs — |error| by presence"
    if tag: ttl += f" ({tag})"
    ax.set_title(ttl)
    ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    ax.legend()
    for i in range(len(kmers)):
        if not np.isnan(means_in[i]):  ax.text(x[i]-width/2, means_in[i], str(counts_in[i]), ha="center", va="bottom", fontsize=8)
        if not np.isnan(means_out[i]): ax.text(x[i]+width/2, means_out[i], str(counts_out[i]), ha="center", va="bottom", fontsize=8)
    suffix = f"_k{k}{('_' + tag) if tag else ''}"
    _save_dual(fig, os.path.join(outdir, "error_bar_presence" + suffix))

def _positional_heatmap(df_sub: pd.DataFrame, dfk: pd.DataFrame, outdir: str, k: int, tag: str = ""):
    """对 topN 基序，在 df_sub 上画归一化位置出现率热图"""
    if dfk.empty or df_sub.empty: return
    topN = min(CONFIG["HEATMAP_TOPN"], len(dfk))
    kmers = dfk.head(topN)["kmer"].tolist()
    B = int(CONFIG["NORM_POS_BINS"])
    mat = np.zeros((topN, B), dtype=float)
    cnt = np.zeros(B, dtype=int)
    for seq in df_sub["sequence"].astype(str):
        L = len(seq)
        if L < k:
            continue
        seen = {km: np.zeros(B, dtype=int) for km in kmers}
        for i in range(L - k + 1):
            km = seq[i:i+k]
            if km in seen:
                center = (i + k/2) / L
                b = int(min(B-1, max(0, math.floor(center * B))))
                seen[km][b] = 1
        for j, km in enumerate(kmers):
            mat[j] += seen[km]
        cnt += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = mat / np.maximum(1, cnt[None, :])
    pd.DataFrame(rate, index=kmers, columns=[f"bin_{i}" for i in range(B)]).to_csv(
        os.path.join(outdir, f"positional_heatmap_k{k}{('_' + tag) if tag else ''}.csv"), index=True
    )
    fig, ax = plt.subplots(figsize=(max(6.5, B*0.3), max(4.0, topN*0.25)))
    im = ax.imshow(rate, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(topN)); ax.set_yticklabels(kmers)
    ax.set_xticks(np.arange(B)); ax.set_xticklabels([str(i+1) for i in range(B)], rotation=0)
    ax.set_xlabel("Normalized position bins (5'→3')")
    ax.set_ylabel("k-mer")
    ttl = f"K={k} top motifs — positional occurrence rate"
    if tag: ttl += f" ({tag})"
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rate")
    suffix = f"_k{k}{('_' + tag) if tag else ''}"
    _save_dual(fig, os.path.join(outdir, "positional_heatmap" + suffix))

# ---------- 主流程 ----------
def main():
    _set_fonts()
    outdir = _ensure_outdir()
    df_all = _safe_read_test(CONFIG["RUN_DIR"])

    all_results = []

    for k in CONFIG["K_LIST"]:
        # 1) 绝对误差富集
        res_abs = _enrich_abs(df_all, k, CONFIG["ALPHABET"])
        res_abs.to_csv(os.path.join(outdir, f"kmer_enrichment_k{k}.csv"), index=False)
        all_results.append(res_abs.assign(k=k, tag="abs"))
        _volcano_plot(res_abs, outdir, k, tag="")
        _bar_error_diff(df_all, res_abs, outdir, k, tag="")
        _positional_heatmap(df_all, res_abs, outdir, k, tag="")

        # 2/3) 方向性富集（可选）
        if CONFIG.get("DO_DIRECTIONAL", True):
            # 正残差：true > pred（模型低估）
            df_pos = df_all[df_all["residual"] > 0]
            res_pos = _enrich_posres(df_all, k, CONFIG["ALPHABET"])
            res_pos.to_csv(os.path.join(outdir, f"kmer_enrichment_k{k}_posres.csv"), index=False)
            all_results.append(res_pos.assign(k=k, tag="posres"))
            _volcano_plot(res_pos, outdir, k, tag="posres (underestimation)")
            _bar_error_diff(df_pos, res_pos, outdir, k, tag="posres")
            _positional_heatmap(df_pos, res_pos, outdir, k, tag="posres")

            # 负残差：true < pred（模型高估）
            df_neg = df_all[df_all["residual"] < 0]
            res_neg = _enrich_negres(df_all, k, CONFIG["ALPHABET"])
            res_neg.to_csv(os.path.join(outdir, f"kmer_enrichment_k{k}_negres.csv"), index=False)
            all_results.append(res_neg.assign(k=k, tag="negres"))
            _volcano_plot(res_neg, outdir, k, tag="negres (overestimation)")
            _bar_error_diff(df_neg, res_neg, outdir, k, tag="negres")
            _positional_heatmap(df_neg, res_neg, outdir, k, tag="negres")

    # 汇总表
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            os.path.join(outdir, "kmer_enrichment_results_all.csv"), index=False
        )
    print(f"[OK] k-mer 富集图已生成：{outdir}")

if __name__ == "__main__":
    main()
