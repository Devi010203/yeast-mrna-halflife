# data_release Description

This directory is used to make public the **processed dataset** and the corresponding **model predictions** used in the paper for this project, and to reserve the **pre-trained model storage location** for users of the replication experiments.

## Directory structure

- `processed/`    Stores the **processed mRNA half-life dataset**, including:
  - Master dataset automatically generated from raw half-life tables with genomic sequences (containing 3′UTR sequences and underlying sequence features, etc.);
  - A version expressed as an RNA alphabet to facilitate direct input into downstream models.

- `predictions/'    Stores the results of the test set generated after the **main program has been run**, typically in a number of CSV tables:
  - Each file typically contains the **true half-life** and **model predictions** of the samples in the test set (along with the necessary identifying information);
  - These files correspond to the performance of the independent test set in the paper and can be used to reproduce the experimental results or for secondary analysis.

- `model/`    Reserved directory for the storage of **pre-trained model files**:
  - This project relies on pre-trained model weights such as RNA-FM, but due to size and licensing reasons, they are not directly distributed with the repository;
  - Users are requested to download the corresponding RNA-FM pre-training models and tokenizers from the official channels, and place the files in this directory (or according to the path description in the main program/configuration file), so as to reproduce the training and inference process locally.

## Description of data sources and processing flow

The `processed/` dataset** in this directory is not a direct copy of the original experimental data**, but is derived data automatically generated through the following open resources and programmatic processing:

1. **Raw half-life data**     - From the supplemental table accompanying the public paper (Global Analysis of mRNA Isoform Half-Lives Reveals Stabilizing and Destabilizing Elements in Yeast) (`mmc2.xlsx` supplemental data file provided by the authors);
2. **Genomic and Sequence Information**     - Using publicly available standard genome sequences (chromosome FASTA files for the yeast genome sacCer2) as reference sequences;
3. **Python scripts for automated construction of the dataset**     - The following steps were automated by Python data processing scripts in the repository:
  
     - Read isoform level half-life information from the supplemental table;
  
     - Extract 3′UTR sequences for each isoform from genomic FASTA based on chromosomal coordinates (remove poly(A));
  
     - Calculate the sequence length, GC content and other basic sequence characteristics;
  
     - The above information was combined into a structured dataset and exported as a CSV file;
  
     - Convert the sequence from DNA form (T) to RNA form (U) and generate the corresponding `_RNA` version file. :contentReference[oaicite:0]{index=0}
  

Therefore, the files in `processed/` are derived results based on **public raw data + standard genome sequences + Python programming**, which is convenient for readers to directly use and reproduce the training and evaluation process in the paper.

## Tips for use

- For complete reproduction of the experiment from scratch, you can:
  1. Synchronize the dependencies using `uv` as per the README of the main repository;
  2. Place the downloaded RNA-FM pre-trained model files into the `data_release/model/` directory;
  3. run training and evaluation using the data in `data_release/processed/` and `data_release/predictions/` according to the paths configured in the main program (e.g., `main.py` or training script).

- For downstream analysis only (e.g., replotting, additional statistics), it is usually sufficient to read the CSV files in `processed/` and `predictions/` without reconstructing the original dataset.

---

### Original paper: ``Global Analysis of mRNA Isoform Half-Lives Reveals Stabilizing and Destabilizing Elements in Yeast`''
### Authors: `Geisberg`, `Joseph V.`, et al.
### doi:`10.1016/j.cell.2013.12.026'
