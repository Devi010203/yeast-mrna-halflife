#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_kmers.py
功能：

- 从 k-mer 富集汇总表（*_all.csv）中筛选“可用 k-mer”
- 支持阈值过滤（q、|log2FC|、支持度 a+c）
- 去冗余（反向互补 + 汉明≤1 近邻）
- 生成等长、中心1bp“温和去功能”的替代序列（用于 in-silico 突变）
- 导出筛选结果与便于 run_interpretability.py 直接使用的 motifs/replacements 文本

使用：
1) 修改下方 CONFIG 中的路径与阈值
2) 运行：  python select_kmers_from_all.py
"""

import os
import math
import json
import pandas as pd
import numpy as np

# =========================
# 配置（全部在此修改）
# =========================
CONFIG = dict(
    # 输入的 k-mer 富集总表（例：你那份 *_all.csv）
    ALL_CSV = r"F:\mRNA_Project\3UTR\Paper\plots\result\plot\kmer_enrichment\kmer_enrichment_results_all.csv",

    # 输出目录（会自动创建）
    OUT_DIR = "F:/mRNA_Project/3UTR/Paper/plots/result/plot/select_kmers",

    # 基本筛选阈值
    Q_MAX = 0.10,          # FDR 阈值
    ABS_LOG2FC_MIN = 0.50, # 效应量阈值（≈1.5x 富集）
    SUPPORT_MIN = 15,      # 支持度阈值 (a+c)；若多为6-mer，可酌情用10

    # 参与筛选的标签（abs/posres/negres）
    TAGS = ["abs", "posres", "negres"],

    # 去冗余策略："none" / "revcomp" / "revcomp+hamming1"
    DEDUP = "revcomp+hamming1",

    # 为每个 tag 导出的 TopN
    TOP_PER_TAG = 20,

    # 当筛后条目过多（>此阈值）时仅做 RC 去重，跳过汉明去重以提速
    HAMMING_ON_MAX = 1500
)

# =========================
# 工具函数
# =========================
DNA = set("ACGT")
DNAU = set("ACGTU")

def normalize_kmer(s: str) -> str:
    return s.upper()

def has_only_acgtu(s: str) -> bool:
    return set(s.upper()) <= DNAU

def revcomp(kmer: str) -> str:
    s = kmer.upper()
    if 'U' in s and 'T' not in s:
        # RNA 互补
        comp = str.maketrans({'A':'U','U':'A','C':'G','G':'C'})
    else:
        # DNA 互补（兼容含 T 的情况）
        comp = str.maketrans({'A':'T','T':'A','C':'G','G':'C'})
    return s.translate(comp)[::-1]


def canonical_rc(kmer: str) -> str:
    k = normalize_kmer(kmer)
    rc = revcomp(k)
    return min(k, rc)

def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x!=y for x,y in zip(a,b))

def score_row(row) -> float:
    """综合得分：|log2fc| * -log10(q) * log(1+support)"""
    q = float(row["q"])
    q = max(q, 1e-300)
    l2 = abs(float(row["log2fc"]))
    support = float(row["a"]) + float(row["c"])
    return l2 * (-math.log10(q)) * math.log(1.0 + support)

def suggest_replacement(kmer: str) -> str:
    s = kmer.upper()
    k = len(s)
    pos = (k - 1) // 2
    base = s[pos]

    # A/U/T 看作“弱”一类；G/C 为“强”一类
    if base in ('A','U','T'):
        new, alt = ('G', 'C')
    else:  # G or C
        # 优先用 A；若这是 RNA（含U且不含T），备用改成 U，否则备用 T
        new = 'A'
        alt = 'U' if ('U' in s and 'T' not in s) else 'T'

    mut = s[:pos] + new + s[pos+1:]
    if mut == s:
        mut = s[:pos] + alt + s[pos+1:]
    return mut


def dedup_by_rc_and_hamming(df: pd.DataFrame, use_hamming: bool) -> pd.DataFrame:
    """先按反向互补合并，再在每个(tag,k)内做汉明≤1的贪心去重（按分数保留代表）。"""
    if df.empty:
        return df.copy()
    df = df.copy()
    df["canonical"] = df["kmer"].apply(canonical_rc)
    df["score"] = df.apply(score_row, axis=1)
    # 反向互补合并：保留最高分
    df = df.sort_values("score", ascending=False)
    df = df.groupby(["tag","k","canonical"], as_index=False).first()

    if not use_hamming:
        return df.drop(columns=["canonical"])

    kept_chunks = []
    for (tag, k), sub in df.groupby(["tag","k"]):
        sub = sub.sort_values("score", ascending=False).copy()
        chosen_idx = []
        removed = set()
        kmers = sub["kmer"].tolist()
        for i, km in enumerate(kmers):
            if km in removed:
                continue
            chosen_idx.append(i)
            # 标记与之汉明<=1的近邻
            for j in range(i+1, len(kmers)):
                kn = kmers[j]
                if kn in removed:
                    continue
                if len(km)==len(kn) and hamming(km, kn) <= 1:
                    removed.add(kn)
        kept_chunks.append(sub.iloc[chosen_idx])
    out = pd.concat(kept_chunks, axis=0).drop(columns=["canonical"])
    return out.sort_values(["tag","k","score"], ascending=[True,True,False])

# =========================
# 主流程
# =========================
def main():
    cfg = CONFIG
    all_csv = cfg["ALL_CSV"]
    out_dir = cfg["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    # 读取
    df = pd.read_csv(all_csv)
    need_cols = {"k","kmer","a","b","c","d","log2fc","p","q","tag"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"输入缺少必要列: {miss}")

    # 类型与清洗
    for col in ["a","b","c","d"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["log2fc","p","q","k"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 只保留合法字母
    df = df[df["kmer"].astype(str).apply(has_only_acgtu)].copy()
    df["kmer"] = df["kmer"].astype(str).str.upper()
    # 过滤 tag
    tag_set = set(cfg["TAGS"])
    df = df[df["tag"].isin(tag_set)].copy()
    # 支持度
    df["support"] = df["a"] + df["c"]

    # 基本阈值过滤
    filt = (
        (df["q"] <= cfg["Q_MAX"]) &
        (df["log2fc"].abs() >= cfg["ABS_LOG2FC_MIN"]) &
        (df["support"] >= cfg["SUPPORT_MIN"])
    )
    df_filt = df[filt].copy()
    if df_filt.empty:
        # 导出空模板并写summary
        empty_path = os.path.join(out_dir, "filtered_kmers.csv")
        df_filt.to_csv(empty_path, index=False)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({
                "note": "no rows matched thresholds",
                "config": cfg,
                "input_rows": int(len(df)),
                "after_basic_filter": 0
            }, f, indent=2)
        print(f"[WARN] 按阈值未筛到任何 k-mer，已导出空文件：{empty_path}")
        return

    # 去冗余
    dedup_mode = cfg["DEDUP"]
    if dedup_mode == "none":
        df_sel = df_filt.copy()
    elif dedup_mode == "revcomp":
        df_sel = dedup_by_rc_and_hamming(df_filt, use_hamming=False)
    else:
        use_hamming = len(df_filt) <= cfg["HAMMING_ON_MAX"]
        df_sel = dedup_by_rc_and_hamming(df_filt, use_hamming=use_hamming)

    # 分数、排序与替代序列
    df_sel["score"] = df_sel.apply(score_row, axis=1)
    df_sel = df_sel.sort_values(["tag","k","score"], ascending=[True,True,False]).copy()
    df_sel["replacement_suggest"] = df_sel["kmer"].apply(suggest_replacement)

    # 导出全集
    all_out = os.path.join(out_dir, "filtered_kmers.csv")
    df_sel.to_csv(all_out, index=False)

    # 各 tag 的 TopN + 汇总 TopN
    tops = []
    topN = int(cfg["TOP_PER_TAG"])
    for tag, sub in df_sel.groupby("tag"):
        topn = sub.head(topN).copy()
        top_path = os.path.join(out_dir, f"top{topN}_{tag}.csv")
        topn.to_csv(top_path, index=False)
        tops.append(topn)
    df_tops = pd.concat(tops, axis=0) if len(tops)>0 else pd.DataFrame(columns=df_sel.columns)
    df_tops.to_csv(os.path.join(out_dir, f"top{topN}_alltags.csv"), index=False)

    # 生成 CLI 参数串（若你仍想用到 run_interpretability.py 的命令行）
    motifs = df_tops["kmer"].tolist()
    repls  = df_tops["replacement_suggest"].tolist()
    motifs_str = ",".join(motifs)
    repls_str  = ",".join(repls)
    with open(os.path.join(out_dir, "motifs_for_cli.txt"), "w") as f:
        f.write(motifs_str + "\n")
    with open(os.path.join(out_dir, "replacements_for_cli.txt"), "w") as f:
        f.write(repls_str + "\n")

    # 导出映射表（便于人工校对/改写）
    map_csv = os.path.join(out_dir, "motif_replacement_map.csv")
    pd.DataFrame({
        "motif": motifs,
        "replacement": repls,
        "tag": df_tops["tag"].tolist(),
        "k": df_tops["k"].tolist(),
        "score": np.round(df_tops["score"].values, 4),
        "log2fc": np.round(df_tops["log2fc"].values, 3),
        "q": df_tops["q"].values,
        "support": df_tops["support"].values
    }).to_csv(map_csv, index=False)

    # 写 summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "config": cfg,
            "input_rows": int(len(df)),
            "after_basic_filter": int(len(df_filt)),
            "after_dedup": int(len(df_sel)),
            "top_per_tag": topN,
            "top_rows": int(len(df_tops)),
            "outputs": {
                "filtered_kmers.csv": all_out,
                f"top{topN}_<tag>.csv": os.path.join(out_dir, f"top{topN}_<tag>.csv"),
                f"top{topN}_alltags.csv": os.path.join(out_dir, f"top{topN}_alltags.csv"),
                "motifs_for_cli.txt": os.path.join(out_dir, "motifs_for_cli.txt"),
                "replacements_for_cli.txt": os.path.join(out_dir, "replacements_for_cli.txt"),
                "motif_replacement_map.csv": map_csv
            }
        }, f, indent=2)

    print("\n✅ 完成筛选与导出")
    print("  - 全部筛选结果:", all_out)
    print(f"  - 每个 tag 的前 {topN}: {os.path.join(out_dir, f'top{topN}_<tag>.csv')}")
    print("  - 汇总 TopN:", os.path.join(out_dir, f"top{topN}_alltags.csv"))
    print("  - --motifs 参数串:", os.path.join(out_dir, "motifs_for_cli.txt"))
    print("  - --replacements 参数串:", os.path.join(out_dir, "replacements_for_cli.txt"))
    print("  - 映射表:", map_csv)
    print("  - 概览:", os.path.join(out_dir, "summary.json"))

if __name__ == "__main__":
    main()
