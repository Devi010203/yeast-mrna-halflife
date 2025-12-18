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


# ===================== Feature extraction =====================

def extract_sequence_features(df):
    """Extracting features from sequences for machine learning"""
    features = []
    for seq in df['sequence']:
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            features.append([np.nan] * 8)
            continue

        seq_len = len(seq)
        feature_vector = [
            seq_len,
            (seq.count('G') + seq.count('C')) / seq_len,  # GC content
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


# ===================== DNA→RNA Conversion-related =====================

def dna_to_rna_str(x):
    """Replace only T/t with U/u; all other characters remain unchanged. x may be NaN or a non-string value."""
    if pd.isna(x):
        return x
    s = str(x)
    return s.translate(str.maketrans({'T': 'U', 't': 'u'}))


def convert_sequence_column_to_rna(input_csv_path, seq_col="sequence",
                                   output_suffix="_RNA"):
    """
    Convert the seq_col column in the specified CSV file from DNA (T) to RNA (U).
    All other columns remain unchanged. Generate a new file with a suffix appended to the original filename in the same directory.
    """
    in_path = Path(input_csv_path)
    if not in_path.exists():
        print(f"[RNA] Input file not found:{in_path}Skip RNA conversion.")
        return None

    out_path = in_path.with_name(in_path.stem + output_suffix + in_path.suffix)

    # When reading data, ensure the sequence column is treated as a string to prevent 'NA' from being interpreted as missing values.
    try:
        df = pd.read_csv(in_path, dtype={seq_col: "string"},
                         keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(in_path)

    if seq_col not in df.columns:
        print(f"[RNA] column '{seq_col}' Does not exist, existing columns:{list(df.columns)}，Skip {in_path.name}")
        return None

    seq_before = df[seq_col].astype("string")
    num_upper_t = seq_before.fillna("").str.count("T").sum()
    num_lower_t = seq_before.fillna("").str.count("t").sum()

    df[seq_col] = seq_before.apply(dna_to_rna_str)

    # Retain UTF-8 BOM for Excel compatibility; set index=False to prevent generating index columns.
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[RNA] Generated:{out_path}")
    print(f"[RNA] on the list '{seq_col}' In total, T={int(num_upper_t)}、t={int(num_lower_t)}(All have been replaced with U/u)")
    print("[RNA] No changes were made to the other columns.")
    return out_path


# ===================== Main Process =====================

def main():
    print("=== mRNA Half-Life Prediction Data Processing & DNA→RNA Conversion ===")

    # 1.Prepare the directory
    setup_directories(OUTPUT_DIR_PROCESSED, OUTPUT_DIR_SEQUENCES)

    # 2. Reading the Half-Life Table
    half_life_df = load_half_life_data(HALF_LIFE_FILE)
    if half_life_df is None:
        return

    # 3. Read chromosome FASTA
    chrom_sequences = extract_chromosome_sequences(CHROMOSOME_DIR)
    if not chrom_sequences:
        print("[Termination] Not loaded onto any chromosome sequence.")
        return

    # 4. Extract the 3′UTR sequence
    df_with_sequences = get_isoform_sequences(half_life_df, chrom_sequences)

    # 5. Extract sequence features
    final_df = extract_sequence_features(df_with_sequences)

    # Attempt to identify half-life column names (compatible with different headers)
    half_life_col = next(
        (col for col in ['Isoform Half-Life (min)', 'Isoform Half-Life']
         if col in final_df.columns),
        None
    )

    # 6. Save the master dataset
    final_df.to_csv(PROCESSED_DATASET_FILE, index=False)
    print(f"\n[Output] Complete dataset saved to: {PROCESSED_DATASET_FILE}")

    # 7. Save the "Sequence + Half-Life" subset
    if half_life_col:
        name_col = 'systematic name' if 'systematic name' in final_df.columns else final_df.columns[0]
        seq_df = final_df[[name_col, 'sequence', half_life_col]]
        seq_df.to_csv(SEQUENCES_FILE, index=False)
        print(f"[Output] Sequence and half-life files saved to: {SEQUENCES_FILE}")

        # Some simple statistics
        stats_df = final_df.dropna(subset=[half_life_col, 'length'])
        if not stats_df.empty:
            print("\n[Statistics] Basic Statistical Information:")
            print(f"  - Half-life range: {stats_df[half_life_col].min():.2f} - "
                  f"{stats_df[half_life_col].max():.2f} minute")
            print(f"  - Average half-life: {stats_df[half_life_col].mean():.2f} minute")
            print(f"  - Sequence Average Length: {stats_df['length'].mean():.2f}")
        else:
            print("\n[Statistics] Statistical information could not be calculated due to the absence of valid sequence or half-life data.")
    else:
        print("\n[Warning] Half-life column not found; skipping statistics and simplifying output.")

    # 8. Perform DNA→RNA conversion (generate additional *_RNA.csv files)
    print("\n=== Initiate DNA-to-RNA Sequence Conversion ===")
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

    print("\n[Complete] The entire process is finished.")


if __name__ == "__main__":
    main()
