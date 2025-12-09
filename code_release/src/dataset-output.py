# -*- coding: utf-8 -*-
"""
One-click mRNA half-life dataset construction + DNA→RNA sequence conversion.

Function overview:
1. Read half-life raw table (Excel/CSV, default ../data/raw/mmc2.xlsx)
2. Extracts 3′UTR sequences for each isoform from the genomic chromosome FASTA (default ../data/processed/sacCer2_chromFa)
3. Calculates simple sequence characteristics (length, GC content, single-base/dinucleotide ratio)
4. Output:
   - ../data/processed/mRNA_half_life_dataset.csv
   - ../data/sequences/sequences_with_half_life.csv
5. Additional step: Convert the `sequence` column in both CSV files from DNA format (T) to RNA format (U), generating:
   - ../data/processed/mRNA_half_life_dataset_RNA.csv
   - ../data/sequences/sequences_with_half_life_RNA.csv

Should you only wish to alter the paths, modify the CONFIG section below.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


# ===================== Configuration Area =====================

HALF_LIFE_FILE = '../data/raw/mmc2.xlsx'              # Original Half-Life Table
CHROMOSOME_DIR = '../data/processed/sacCer2_chromFa'  # Chromosome FASTA Directory
OUTPUT_DIR_PROCESSED = '../data/processed'
OUTPUT_DIR_SEQUENCES = '../data/sequences'

PROCESSED_DATASET_FILE = os.path.join(
    OUTPUT_DIR_PROCESSED, 'mRNA_half_life_dataset.csv'
)
SEQUENCES_FILE = os.path.join(
    OUTPUT_DIR_SEQUENCES, 'sequences_with_half_life.csv'
)

SEQUENCE_COLUMN_NAME = "sequence"     # Column names requiring DNA→RNA conversion
RNA_OUTPUT_SUFFIX = "_RNA"            # RNA version filename suffix


# ===================== Directory, IO Utility Functions =====================

def setup_directories(processed_dir, sequences_dir):
    """Create output directory"""
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(sequences_dir, exist_ok=True)
    print(f"[Directory] Ensure the directory exists: {processed_dir}, {sequences_dir}")


def load_half_life_data(file_path):
    """Reading half-life data from Excel or CSV files"""
    if not os.path.exists(file_path):
        print(f"[Error] Half-life file not found -> {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
        print(f"[Reading] Successfully loaded half-life data for {len(df)} isoforms from Excel.")
    except Exception:
        try:
            df = pd.read_csv(file_path)
            print(f"[Reading] Successfully loaded half-life data for {len(df)} isoforms from CSV.")
        except Exception as e:
            print(f"[Error] Failed to read Excel/CSV file: {e}")
            return None
    print("[Read] Data column:", df.columns.tolist())
    return df


# ===================== Chromosome & Sequence Extraction =====================

def extract_chromosome_sequences(chrom_dir):
    """Extract all chromosome sequences from the FASTA file directory into the dictionary"""
    chrom_sequences = {}
    if not os.path.isdir(chrom_dir):
        print(f"[Error] Chromosome directory not found -> {chrom_dir}")
        return chrom_sequences

    print(f"[FASTA] Loading chromosome sequences from {chrom_dir}...")
    for filename in os.listdir(chrom_dir):
        if filename.endswith('.fa'):
            filepath = os.path.join(chrom_dir, filename)
            try:
                for record in SeqIO.parse(filepath, "fasta"):
                    chrom_sequences[record.id] = str(record.seq).upper()
                    print(f"  - Loaded: {record.id} (Length: {len(record.seq)})")
            except Exception as e:
                print(f"[Warning] An error occurred while processing the file {filepath}: {e}")
    return chrom_sequences


def get_isoform_sequences(df, chrom_sequences):
    """Extract the 3′UTR sequence (excluding the poly(A) tail) for each isoform from the chromosome sequence based on coordinates."""
    sequences = []

    # Required column: chrom、strand、cdsStart、cdsEnd、Absolute Peak Coordinate
    required = ['chrom', 'strand', 'cdsStart', 'cdsEnd', 'Absolute Peak Coordinate']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[Error] Required columns are missing: {missing}")
        return df.assign(sequence=[None] * len(df))

    # Chromosome name compatibility mapping:chr1~chr16 -> chrI~chrXVI
    arabic_to_roman = {
        f'chr{i}': f'chr{r}' for i, r in zip(
            range(1, 17),
            ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
             'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
        )
    }

    for idx, row in df.iterrows():
        chrom_name = str(row['chrom'])
        chrom_key = arabic_to_roman.get(chrom_name, chrom_name)

        if chrom_key not in chrom_sequences:
            # Attempt to remove the prefix 'chr'
            alt = chrom_key.replace('chr', '')
            if alt in chrom_sequences:
                chrom_key = alt
            else:
                print(f"[Warning] Chromosome {idx + 2}, chromosome '{chrom_key}' not found in FASTA.")
                sequences.append(None)
                continue

        strand = str(row['strand']).strip()
        cds_start = int(row['cdsStart'])
        cds_end = int(row['cdsEnd'])
        peak = int(row['Absolute Peak Coordinate'])
        chrom_seq = chrom_sequences[chrom_key]

        seq = None
        try:
            if strand == '+':
                # 3′UTR: (cdsEnd, peak] —— Python slicing excludes the right end; it's an exact length peak - cdsEnd
                if peak < cds_end:
                    print(f"[Warning] Line {idx + 2} (+)，peak({peak}) < cdsEnd({cds_end})Skip.")
                else:
                    seq = chrom_seq[cds_end:peak]
            elif strand == '-':
                # 3′UTR: [peak, cdsStart) —— After extraction, perform reverse complementary pairing. Length cdsStart - peak
                if peak > cds_start:
                    print(f"[Warning] Line {idx + 2} (-)，peak({peak}) > cdsStart({cds_start})Skip.")
                else:
                    raw = chrom_seq[peak:cds_start]
                    seq = str(Seq(raw).reverse_complement())
            else:
                print(f"[Warning] Line {idx + 2} Unknown chain information: {strand}")
        except Exception as e:
            print(f"[Warning] Line {idx + 2} Slicing failed: {e}")

        sequences.append(seq)

    df = df.copy()
    df['sequence'] = sequences
    return df


# ===================== 特征提取 =====================

def extract_sequence_features(df):
    """从序列中提取用于机器学习的特征"""
    features = []
    for seq in df['sequence']:
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            features.append([np.nan] * 8)
            continue

        seq_len = len(seq)
        feature_vector = [
            seq_len,
            (seq.count('G') + seq.count('C')) / seq_len,  # GC 含量
            seq.count('A') / seq_len,
            seq.count('T') / seq_len,
            seq.count('G') / seq_len,
            seq.count('C') / seq_len,
            seq.count('AT') / seq_len,
            seq.count('GC') / seq_len,
        ]
        features.append(feature_vector)

    feature_columns = [
        'length', 'gc_content',
        'A_content', 'T_content', 'G_content', 'C_content',
        'AT_dinuc', 'GC_dinuc'
    ]
    feature_df = pd.DataFrame(features, columns=feature_columns, index=df.index)
    return pd.concat([df, feature_df], axis=1)


# ===================== DNA→RNA 转换相关 =====================

def dna_to_rna_str(x):
    """仅把 T/t 换成 U/u；其他字符保持不变。x 可能为 NaN 或非字符串。"""
    if pd.isna(x):
        return x
    s = str(x)
    return s.translate(str.maketrans({'T': 'U', 't': 'u'}))


def convert_sequence_column_to_rna(input_csv_path, seq_col="sequence",
                                   output_suffix="_RNA"):
    """
    将指定 CSV 文件中的 seq_col 列从 DNA (T) 转为 RNA (U)。
    其余列保持不变，在同目录下生成追加后缀的新文件。
    """
    in_path = Path(input_csv_path)
    if not in_path.exists():
        print(f"[RNA] 找不到输入文件：{in_path}，跳过 RNA 转换。")
        return None

    out_path = in_path.with_name(in_path.stem + output_suffix + in_path.suffix)

    # 读取时尽量保证 sequence 列当作字符串，避免 'NA' 被当作缺失
    try:
        df = pd.read_csv(in_path, dtype={seq_col: "string"},
                         keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(in_path)

    if seq_col not in df.columns:
        print(f"[RNA] 列 '{seq_col}' 不存在，现有列：{list(df.columns)}，跳过 {in_path.name}")
        return None

    seq_before = df[seq_col].astype("string")
    num_upper_t = seq_before.fillna("").str.count("T").sum()
    num_lower_t = seq_before.fillna("").str.count("t").sum()

    df[seq_col] = seq_before.apply(dna_to_rna_str)

    # 保留 UTF-8 BOM 以兼容 Excel，index=False 防止生成索引列
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[RNA] 已生成：{out_path}")
    print(f"[RNA] 在列 '{seq_col}' 中，共发现 T={int(num_upper_t)}、t={int(num_lower_t)}（均已替换为 U/u）")
    print("[RNA] 其它列未做任何修改。")
    return out_path


# ===================== 主流程 =====================

def main():
    print("=== mRNA 半衰期预测数据处理 & DNA→RNA 转换 ===")

    # 1. 准备目录
    setup_directories(OUTPUT_DIR_PROCESSED, OUTPUT_DIR_SEQUENCES)

    # 2. 读取半衰期表
    half_life_df = load_half_life_data(HALF_LIFE_FILE)
    if half_life_df is None:
        return

    # 3. 读取染色体FASTA
    chrom_sequences = extract_chromosome_sequences(CHROMOSOME_DIR)
    if not chrom_sequences:
        print("[终止] 未加载到任何染色体序列。")
        return

    # 4. 提取 3′UTR 序列
    df_with_sequences = get_isoform_sequences(half_life_df, chrom_sequences)

    # 5. 提取序列特征
    final_df = extract_sequence_features(df_with_sequences)

    # 尝试识别半衰期列名（兼容不同表头）
    half_life_col = next(
        (col for col in ['Isoform Half-Life (min)', 'Isoform Half-Life']
         if col in final_df.columns),
        None
    )

    # 6. 保存主数据集
    final_df.to_csv(PROCESSED_DATASET_FILE, index=False)
    print(f"\n[输出] 完整数据集保存至: {PROCESSED_DATASET_FILE}")

    # 7. 保存“序列 + 半衰期”子集
    if half_life_col:
        name_col = 'systematic name' if 'systematic name' in final_df.columns else final_df.columns[0]
        seq_df = final_df[[name_col, 'sequence', half_life_col]]
        seq_df.to_csv(SEQUENCES_FILE, index=False)
        print(f"[输出] 序列与半衰期文件保存至: {SEQUENCES_FILE}")

        # 一些简单统计
        stats_df = final_df.dropna(subset=[half_life_col, 'length'])
        if not stats_df.empty:
            print("\n[统计] 基本统计信息：")
            print(f"  - 半衰期范围: {stats_df[half_life_col].min():.2f} - "
                  f"{stats_df[half_life_col].max():.2f} 分钟")
            print(f"  - 平均半衰期: {stats_df[half_life_col].mean():.2f} 分钟")
            print(f"  - 序列平均长度: {stats_df['length'].mean():.2f}")
        else:
            print("\n[统计] 未能计算统计信息，因为没有有效的序列或半衰期数据。")
    else:
        print("\n[警告] 未找到半衰期列，跳过统计与简化输出。")

    # 8. 进行 DNA→RNA 转换（生成额外的 *_RNA.csv 文件）
    print("\n=== 开始 DNA→RNA 序列转换 ===")
    convert_sequence_column_to_rna(
        PROCESSED_DATASET_FILE,
        seq_col=SEQUENCE_COLUMN_NAME,
        output_suffix=RNA_OUTPUT_SUFFIX
    )

    if os.path.exists(SEQUENCES_FILE):
        convert_sequence_column_to_rna(
            SEQUENCES_FILE,
            seq_col=SEQUENCE_COLUMN_NAME,
            output_suffix=RNA_OUTPUT_SUFFIX
        )

    print("\n[完成] 全部流程结束。")


if __name__ == "__main__":
    main()
