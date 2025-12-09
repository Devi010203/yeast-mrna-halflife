## 此文件夹下的程序都需要依赖于主程序运行后产物，请确保你已经运行完主程序，并在相关脚本中修改相关文件路径

### select_kmers.py

- **功能作用**  
  从 `kmer_enrichment_results_all.csv` 这类 k-mer 富集汇总表中自动筛选“可用 k-mer”：  
  按 FDR（q 值）、|log2FC| 和支持度 (a+c) 做阈值过滤，并按反向互补＋汉明距离≤1 去冗余；  
  同时为每个 k-mer 生成等长、中心 1bp“温和去功能”的替代序列，导出为文本文件，便于直接作为 `run_interpretability.py` 的 motif/replacement 输入。

- **主要输入**  
  - `CONFIG["ALL_CSV"]`：k-mer 富集总表（通常是 `kmer_enrichment_results_all.csv`），需包含 `k,kmer,a,b,c,d,log2fc,p,q,tag` 等列。  
  - `CONFIG["Q_MAX"]`、`ABS_LOG2FC_MIN`、`SUPPORT_MIN`：FDR、效应量和支持度筛选阈值。  
  - `CONFIG["TAGS"]`：参与筛选的标签（如 `["abs","posres","negres"]`）。  
  - `CONFIG["DEDUP"]`：去冗余策略（`none` / `revcomp` / `revcomp+hamming1`）。  
  - `CONFIG["OUT_DIR"]`：输出目录。

- **主要输出（写入 `OUT_DIR`）**  
  - `filtered_kmers.csv`：通过阈值和去冗余后的 k-mer 全部列表。  
  - `top{topN}_<tag>.csv`、`top{topN}_alltags.csv`：按 |log2FC| 排序的各 tag 及合并 TopN k-mer。  
  - `motifs_for_cli.txt`：一行包含所有筛选 motif 的逗号分隔串，可直接给命令行 `--motifs` 使用。  
  - `replacements_for_cli.txt`：对应的替代序列串，可给 `--replacements` 使用。  
  - `motif_replacement_map.csv`：motif 与其替代序列映射表。  
  - `summary.json`：记录输入文件、阈值、筛选后数量等摘要信息。


---

### run_interpretability.py

- **功能作用**  
  加载最终训练好的 RNA-FM + TokenTransformer 模型，对含指定 motif 的序列做系统性 in-silico 突变：  
  在测试集或指定样本中枚举（或按 motif 抽样）替换位点，将 motif 替换为给定替代序列，比较突变前后预测半衰期，得到每个 motif / 替代序列的效应分布，并输出残差表与简单性能统计，用于后续可解释性分析与绘图。

- **主要输入**  
  - `RunParams.EXP_DIR`：最终训练目录，需包含 `best_model_final.pth`，若存在 `final_test_predictions.csv` 会优先直接读取。  
  - `RunParams.MUTATION_MODE`：`"full"`（含 motif 的所有序列 × 所有出现位置 × 所有替代序列全量评估）或 `"per-motif"`（按 motif 抽样每个至少 N 条）。  
  - `RunParams.NUM_MUTATION_SAMPLES`：`"per-motif"` 模式下，每个 motif 最少抽样的序列数。  
  - `RunParams.MOTIFS`：要检测/突变的 motif 列表（RNA 字母表）。  
  - `RunParams.REPLACEMENTS`：对应 motif 的替代序列集合（与 motif 等长）。  
  - 与主模型一致的 `Config`：数据集路径、RNA-FM 预训练模型路径、最大长度、batch size 等。

- **主要输出（写入 `EXP_DIR/../result/interpretability_result/<时间戳>/`）**  
  - `residuals.csv`：测试集样本级真实值、预测值与残差。  
  - `motif_corr.csv`：在原始残差上计算的 motif 出现与残差相关性概览。  
  - `mutation_results.csv`：in-silico 突变结果，包含每条样本、motif、替代序列、突变位置、突变前/后预测值及 `delta`（mut − base）。  
  - `summary.json`：测试集性能（R²、Pearson、Spearman）和突变配置等摘要。  
  - `interpret_env.json`：记录使用的 RunParams / Config 等环境参数。  
  - `tensorboard-log/interpretability/`：TensorBoard 日志（残差直方图、delta 分布等）。


---

### analyze_mutation_results.py

- **功能作用**  
  对 `run_interpretability.py` 生成的 `mutation_results.csv` 做系统统计与绘图：  
  先在每个样本内对多位点 Δ 取均值，得到“每样本均值 Δ”；再按 (motif,new) 对样本进行汇总，计算均值、中位数、bootstrap 置信区间、强响应比例，并使用 Wilcoxon 符号秩检验（Δ vs 0）＋ BH-FDR 校正；  
  同时分析 Δ 与相对位置（motif 中心 / 序列长度）、基线半衰期（base_pred）分层之间的关系，输出多张论文级图。

- **主要输入**  
  - `CONFIG.MUTATION_CSV`：`mutation_results.csv` 路径。  
  - `CONFIG.OUT_DIR`：分析结果输出目录。  
  - 统计/作图参数：  
    - `TOPK`：按 |mean_delta| 排序后展示的 motif / (motif,new) 数量。  
    - `N_POS_BINS`：位置分箱数。  
    - `N_BOOT`：bootstrap 次数。  
    - `DPI`、`SAVE_SVG`：图像分辨率和是否保存 SVG。  
    - `BIG_FRAC_THRESH`：相对变化阈值（如 Δt/t₀ > 10% 视为“强响应”）。  
    - `N_T0_BINS`：按基线预测值分层的 bin 数。  
    - `RESIDUALS_CSV`（可选）：残差文件路径，留空则默认同目录 `residuals.csv`。  
    - `USE_RESIDUAL_FILTER`、`RESIDUAL_ABS_THRESH`：是否仅在低残差样本上做额外统计及其阈值。

- **主要输出（写入 `OUT_DIR`）**  
  - 表格类：  
    - `per_sequence_delta.csv`：每个样本（序列）级的均值 Δ。  
    - `mutation_stats_by_motif.csv` / `mutation_stats_by_motif_new.csv`：按 motif 或 (motif,new) 聚合的主效应统计（均值、中位数、CI、p 值、FDR、强响应比例等）。  
    - `mutation_stats_by_motif*_goodfit.csv`：仅在低残差样本子集上的同类统计。  
    - `mutation_stats_by_motif*_t0bins.csv`：在基线半衰期分层上的统计。  
    - `mutation_position_effect.csv`：位置效应相关统计（Δ vs 相对位置）。  
    - `residuals.csv`：如有 residual 过滤，会导出实际使用的残差信息。  
  - 图像（PNG≥400dpi，且可选 SVG）：  
    - 主效应条形图（按 |mean_delta| 排序）。  
    - 每个 motif/(motif,new) 的 Δ 箱线＋抖动图。  
    - Δ 随相对位置变化的折线图或热图。  
    - Δ 在不同基线半衰期分层下的效应对比图。  
    - 按 motif 聚合的“火山图”等。  
  - `summary.json`：输入文件、样本数、motif 数量、TopK、分箱数等整体摘要。


---

### plot_kmer_error_source_summary_v2.py

- **功能作用**  
  对 k-mer 富集结果做“误差来源总览”可视化：  
  读取 `kmer_enrichment_results_all.csv`（若不存在则回退单独的 k=5/k=6 结果文件），统计在绝对误差、正残差（模型低估）、负残差（模型高估）三个子集下显著富集的 k-mer 数量和效应；  
  结合位置热图（如 `positional_heatmap_k5*.csv`, `positional_heatmap_k6*.csv`），为每个 k-mer 估计峰值 bin，用颜色/气泡大小编码效应和位置，生成 2×3 气泡图和显著计数条形图。

- **主要输入**  
  - `PATH_RESULTS_ALL`（可选）：`kmer_enrichment_results_all.csv` 总表；若存在则优先使用。  
  - 回退路径（在无总表时使用）：  
    - `PATH_K5_ABS / POS / NEG`：k=5 的绝对误差、正残差、负残差富集结果 CSV。  
    - `PATH_K6_ABS / POS / NEG`：k=6 的同类 CSV。  
  - 位置热图：  
    - `PATH_POS_K5_ABS / POS / NEG`：k=5 的位置热图 CSV。  
    - `PATH_POS_K6_ABS / POS / NEG`：k=6 的位置热图 CSV（若缺失，则 k=6 气泡统一灰色）。  
  - `OUT_PARENT`：输出父目录（脚本内部会按时间戳新建子目录）。

- **主要输出（写入 `OUT_PARENT/<时间戳>/`）**  
  - `kmer_error_source_bubble_grid.png/svg`：2×3 气泡图（行区分 k=5/6，列区分 abs/posres/negres），点大小/颜色编码显著性和效应、并可按峰值位置上色。  
  - `kmer_sig_counts_bar.png/svg`：各面板显著 k-mer 数量（总计 / 正向 / 负向）的条形图。  
  - `sig_counts_summary.csv`：每个面板显著计数和阈值统计。  
  - `top_each_cell.csv`：每个面板 TopN k-mer 的详细表（含 log2FC、q、top/bot 出现率、峰值 bin 等）。


---

### plots/plot_kmer_enrichment.py

- **功能作用**  
  以模型在测试集上的预测误差为驱动，对 5-mer/6-mer 做富集分析：  
  先计算 `residual = true - pred` 和 `abs_error`，再构造三类子集：  
  1）按 |error| 的 top vs bottom；  
  2）在残差>0 的“低估”子集内，按 |residual| top vs bottom；  
  3）在残差<0 的“高估”子集内，按 |residual| top vs bottom。  
  对每类、每个 k 分别做 presence-based 富集，计算 log2FC、Fisher 精确检验 p 值及 FDR，并输出火山图、条形图和归一化位置热图，同时生成后续脚本使用的 k-mer 富集 CSV 与汇总表。

- **主要输入**  
  - `CONFIG["RUN_DIR"]`：模型运行目录，需包含 `final_test_predictions.csv`，其中至少包括列：  
    - `sequence`：3′UTR 序列；  
    - `true`：真实半衰期（或转化后的目标）；  
    - `pred`：模型预测值。  
  - `CONFIG["SAVE_SUBDIR"]`：输出子目录名（如 `"kmer_enrichment"`）。  
  - k-mer 与作图参数：  
    - `K_LIST`：k 值列表（如 `[5,6]`）。  
    - `TOP_PCT`：划分 top/bottom 所用的误差百分位（如前后 10%）。  
    - `ALPHABET`：允许的字母表（如 `"ACGU"`）。  
    - `MIN_OCC`：过滤总出现次数过少的 k-mer。  
    - `FDR_Q`：显著性阈值。  
    - `VOLCANO_TOPN`、`BAR_TOPN`：火山图和条形图中标注/展示的 TopN 数量。  
    - `NORM_POS_BINS`：位置热图的归一化分段数。  
    - `HEATMAP_TOPN`：位置热图中展示的 k-mer 数量。

- **主要输出（写入 `<项目根>/result/plot/kmer_enrichment/`）**  
  - CSV：  
    - `kmer_enrichment_k{k}.csv`：按绝对误差分组的富集结果。  
    - `kmer_enrichment_k{k}_posres.csv`：在“低估”子集上的富集结果。  
    - `kmer_enrichment_k{k}_negres.csv`：在“高估”子集上的富集结果。  
    - `kmer_enrichment_results_all.csv`：整合 k、tag、log2FC、q、top/bot 出现率等信息的总汇表（供后续脚本使用）。  
    - `positional_heatmap_k{k}.csv` 及带 `_posres` / `_negres` 后缀的变体：归一化位置热图的底层数据。  
  - 图像（PNG, dpi=400，同时保存 SVG）：  
    - 每个 k、每种误差类型对应的火山图。  
    - 对比“含/不含 k-mer”误差分布的条形图。  
    - 归一化位置热图，显示富集 k-mer 在 3′UTR 上的位置分布。
