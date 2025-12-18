#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_kmers.py
Functionality:

- Filters "usable k-mers" from the k-mer enrichment summary table (*_all.csv)
- Supports threshold filtering (q, |log2FC|, support a+c)
- Removes redundancy (reverse complementary + nearest neighbours with Hamming ≤ 1)
- Generates equally sized, centre-1bp "mildly de-functionalised" replacement sequences (for in-silico mutation)
- Exports filtered results and motifs/replacements text files for direct use by run_interpretability.py

Usage:
1) Modify paths and thresholds in CONFIG below
2) Run: python select_kmers_from_all.py
"""

import os
import math
import json
import pandas as pd
import numpy as np

# =========================
# Configuration
# =========================
CONFIG = dict(
    # Input k-mer enrichment summary table (e.g., your *_all.csv file)
    ALL_CSV = r"F:\mRNA_Project\3UTR\Paper\plots\result\plot\kmer_enrichment\kmer_enrichment_results_all.csv",

    # Output directory (will be created automatically)
    OUT_DIR = "F:/mRNA_Project/3UTR/Paper/plots/result/plot/select_kmers",

    # Basic screening threshold
    Q_MAX = 0.10,          # FDR threshold
    ABS_LOG2FC_MIN = 0.50, # Effect size threshold
    SUPPORT_MIN = 15,      # Support threshold (a+c); where 6-mers predominate, 10 may be used as appropriate.

    # Tags involved in screening (abs/posres/negres)
    TAGS = ["abs", "posres", "negres"],

    # Redundancy elimination strategy:"none" / "revcomp" / "revcomp+hamming1"
    DEDUP = "revcomp+hamming1",

    # Top N exported for each tag
    TOP_PER_TAG = 20,

    # When the number of post-filtering entries exceeds this threshold, perform only RC deduplication and bypass Hamming deduplication to accelerate processing.
    HAMMING_ON_MAX = 1500
)

# =========================
# Utility functions
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
        # RNA Complementary
        comp = str.maketrans({'A':'U','U':'A','C':'G','G':'C'})
    else:
        # DNA Complementarity (including cases involving T)
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
    """Overall score:|log2fc| * -log10(q) * log(1+support)"""
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

    # A/U/T is classified as the 'weak' category; G/C is classified as the 'strong' category.
    if base in ('A','U','T'):
        new, alt = ('G', 'C')
    else:  # G or C
        # Prioritise A; if this is RNA (containing U and lacking T), substitute with U; otherwise substitute with T.
        new = 'A'
        alt = 'U' if ('U' in s and 'T' not in s) else 'T'

    mut = s[:pos] + new + s[pos+1:]
    if mut == s:
        mut = s[:pos] + alt + s[pos+1:]
    return mut


def dedup_by_rc_and_hamming(df: pd.DataFrame, use_hamming: bool) -> pd.DataFrame:
    """First perform reverse complementary merging, then within each (tag, k) pair, apply a greedy deduplication step
    where the Hamming distance is ≤1 (retaining the representative based on score)."""
    if df.empty:
        return df.copy()
    df = df.copy()
    df["canonical"] = df["kmer"].apply(canonical_rc)
    df["score"] = df.apply(score_row, axis=1)
    # Reverse complementary merging: Retain the highest score
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
            # Mark the nearest neighbour with a Hanming value <= 1
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
# Main process
# =========================
def main():
    cfg = CONFIG
    all_csv = cfg["ALL_CSV"]
    out_dir = cfg["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    # Read
    df = pd.read_csv(all_csv)
    need_cols = {"k","kmer","a","b","c","d","log2fc","p","q","tag"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"Input is missing required columns: {miss}")

    # Type and Cleaning
    for col in ["a","b","c","d"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["log2fc","p","q","k"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Only retain legal letters
    df = df[df["kmer"].astype(str).apply(has_only_acgtu)].copy()
    df["kmer"] = df["kmer"].astype(str).str.upper()
    # Filter tag
    tag_set = set(cfg["TAGS"])
    df = df[df["tag"].isin(tag_set)].copy()
    # support rating
    df["support"] = df["a"] + df["c"]

    # Basic threshold filtering
    filt = (
        (df["q"] <= cfg["Q_MAX"]) &
        (df["log2fc"].abs() >= cfg["ABS_LOG2FC_MIN"]) &
        (df["support"] >= cfg["SUPPORT_MIN"])
    )
    df_filt = df[filt].copy()
    if df_filt.empty:
        # Export empty template and write summary
        empty_path = os.path.join(out_dir, "filtered_kmers.csv")
        df_filt.to_csv(empty_path, index=False)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({
                "note": "no rows matched thresholds",
                "config": cfg,
                "input_rows": int(len(df)),
                "after_basic_filter": 0
            }, f, indent=2)
        print(f"[WARN] No k-mers were filtered at the threshold; an empty file has been exported.{empty_path}")
        return

    # Eliminate redundancy
    dedup_mode = cfg["DEDUP"]
    if dedup_mode == "none":
        df_sel = df_filt.copy()
    elif dedup_mode == "revcomp":
        df_sel = dedup_by_rc_and_hamming(df_filt, use_hamming=False)
    else:
        use_hamming = len(df_filt) <= cfg["HAMMING_ON_MAX"]
        df_sel = dedup_by_rc_and_hamming(df_filt, use_hamming=use_hamming)

    # Scores, Ordering and Substitution Sequences
    df_sel["score"] = df_sel.apply(score_row, axis=1)
    df_sel = df_sel.sort_values(["tag","k","score"], ascending=[True,True,False]).copy()
    df_sel["replacement_suggest"] = df_sel["kmer"].apply(suggest_replacement)

    # Export the complete collection
    all_out = os.path.join(out_dir, "filtered_kmers.csv")
    df_sel.to_csv(all_out, index=False)

    # Top N for each tag + aggregated Top N
    tops = []
    topN = int(cfg["TOP_PER_TAG"])
    for tag, sub in df_sel.groupby("tag"):
        topn = sub.head(topN).copy()
        top_path = os.path.join(out_dir, f"top{topN}_{tag}.csv")
        topn.to_csv(top_path, index=False)
        tops.append(topn)
    df_tops = pd.concat(tops, axis=0) if len(tops)>0 else pd.DataFrame(columns=df_sel.columns)
    df_tops.to_csv(os.path.join(out_dir, f"top{topN}_alltags.csv"), index=False)

    motifs = df_tops["kmer"].tolist()
    repls  = df_tops["replacement_suggest"].tolist()
    motifs_str = ",".join(motifs)
    repls_str  = ",".join(repls)
    with open(os.path.join(out_dir, "motifs_for_cli.txt"), "w") as f:
        f.write(motifs_str + "\n")
    with open(os.path.join(out_dir, "replacements_for_cli.txt"), "w") as f:
        f.write(repls_str + "\n")

    # Export mapping table (for manual proofreading/rewriting)
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

    # Write summary
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

    print("\n Complete filtering and exporting")
    print("  - All filtered results:", all_out)
    print(f"  - Each tag's prefix {topN}: {os.path.join(out_dir, f'top{topN}_<tag>.csv')}")
    print("  - aggregate TopN:", os.path.join(out_dir, f"top{topN}_alltags.csv"))
    print("  - --motifs Parameter string:", os.path.join(out_dir, "motifs_for_cli.txt"))
    print("  - --replacements parameter string:", os.path.join(out_dir, "replacements_for_cli.txt"))
    print("  - Mapping table:", map_csv)
    print("  - skim through:", os.path.join(out_dir, "summary.json"))

if __name__ == "__main__":
    main()
