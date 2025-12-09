# -*- coding: utf-8 -*-
"""
一键完成 mRNA 半衰期数据集构建 + DNA→RNA 序列转换。

功能概览：
1. 读取半衰期原始表（Excel/CSV，默认 ../data/raw/mmc2.xlsx）
2. 从基因组染色体 FASTA（默认 ../data/processed/sacCer2_chromFa）中提取每个 isoform 的 3′UTR 序列
3. 计算简单序列特征（长度、GC 含量、单碱基/二核苷酸比例）
4. 输出：
   - ../data/processed/mRNA_half_life_dataset.csv
   - ../data/sequences/sequences_with_half_life.csv
5. 额外步骤：将上述两个 CSV 中的 `sequence` 列由 DNA 格式 (T) 转为 RNA 格式 (U)，生成：
   - ../data/processed/mRNA_half_life_dataset_RNA.csv
   - ../data/sequences/sequences_with_half_life_RNA.csv

如果你只想改路径，修改下面 CONFIG 区域即可。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


# ===================== 配置区域 =====================

HALF_LIFE_FILE = '../data/raw/mmc2.xlsx'              # 原始半衰期表格
CHROMOSOME_DIR = '../data/processed/sacCer2_chromFa'  # 染色体FASTA目录
OUTPUT_DIR_PROCESSED = '../data/processed'
OUTPUT_DIR_SEQUENCES = '../data/sequences'

PROCESSED_DATASET_FILE = os.path.join(
    OUTPUT_DIR_PROCESSED, 'mRNA_half_life_dataset.csv'
)
SEQUENCES_FILE = os.path.join(
    OUTPUT_DIR_SEQUENCES, 'sequences_with_half_life.csv'
)

SEQUENCE_COLUMN_NAME = "sequence"     # 需要进行 DNA→RNA 转换的列名
RNA_OUTPUT_SUFFIX = "_RNA"            # RNA 版本文件名后缀


# ===================== 目录、IO 辅助函数 =====================

def setup_directories(processed_dir, sequences_dir):
    """创建输出目录"""
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(sequences_dir, exist_ok=True)
    print(f"[目录] 确保目录存在: {processed_dir}, {sequences_dir}")


def load_half_life_data(file_path):
    """从Excel或CSV文件读取半衰期数据"""
    if not os.path.exists(file_path):
        print(f"[错误] 半衰期文件未找到 -> {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
        print(f"[读取] 成功从 Excel 加载了 {len(df)} 个 isoform 的半衰期数据。")
    except Exception:
        try:
            df = pd.read_csv(file_path)
            print(f"[读取] 成功从 CSV 加载了 {len(df)} 个 isoform 的半衰期数据。")
        except Exception as e:
            print(f"[错误] 读取 Excel/CSV 文件失败: {e}")
            return None
    print("[读取] 数据列:", df.columns.tolist())
    return df


# ===================== 染色体 & 序列提取 =====================

def extract_chromosome_sequences(chrom_dir):
    """从FASTA文件目录中提取所有染色体序列到字典"""
    chrom_sequences = {}
    if not os.path.isdir(chrom_dir):
        print(f"[错误] 找不到染色体目录 -> {chrom_dir}")
        return chrom_sequences

    print(f"[FASTA] 从 {chrom_dir} 加载染色体序列...")
    for filename in os.listdir(chrom_dir):
        if filename.endswith('.fa'):
            filepath = os.path.join(chrom_dir, filename)
            try:
                for record in SeqIO.parse(filepath, "fasta"):
                    chrom_sequences[record.id] = str(record.seq).upper()
                    print(f"  - 已加载: {record.id} (长度: {len(record.seq)})")
            except Exception as e:
                print(f"[警告] 处理文件 {filepath} 时出错: {e}")
    return chrom_sequences


def get_isoform_sequences(df, chrom_sequences):
    """根据坐标从染色体序列中提取每个isoform的3′UTR序列（不含poly(A)）"""
    sequences = []

    # 必要列：chrom、strand、cdsStart、cdsEnd、Absolute Peak Coordinate
    required = ['chrom', 'strand', 'cdsStart', 'cdsEnd', 'Absolute Peak Coordinate']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[错误] 缺少必要列: {missing}")
        return df.assign(sequence=[None] * len(df))

    # 染色体名兼容映射：chr1~chr16 -> chrI~chrXVI
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
            # 尝试去掉前缀 'chr'
            alt = chrom_key.replace('chr', '')
            if alt in chrom_sequences:
                chrom_key = alt
            else:
                print(f"[警告] 行 {idx + 2}, 染色体 '{chrom_key}' 未在FASTA中找到。")
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
                # 3′UTR: (cdsEnd, peak] —— Python 切片右端不含，正好长度 peak - cdsEnd
                if peak < cds_end:
                    print(f"[警告] 行 {idx + 2} (+)，peak({peak}) < cdsEnd({cds_end})，跳过。")
                else:
                    seq = chrom_seq[cds_end:peak]
            elif strand == '-':
                # 3′UTR: [peak, cdsStart) —— 取出后再反向互补，长度 cdsStart - peak
                if peak > cds_start:
                    print(f"[警告] 行 {idx + 2} (-)，peak({peak}) > cdsStart({cds_start})，跳过。")
                else:
                    raw = chrom_seq[peak:cds_start]
                    seq = str(Seq(raw).reverse_complement())
            else:
                print(f"[警告] 行 {idx + 2} 未知链信息: {strand}")
        except Exception as e:
            print(f"[警告] 行 {idx + 2} 切片失败: {e}")

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
