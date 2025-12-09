## 此文件夹下的程序都需要依赖于主程序运行后产物，请确保你已经运行完主程序，并在相关脚本中修改相关文件路径

### gene_isoform_correlation_analysis_config.py
- **功能作用**：  
  基于测试集预测结果，按序列将预测文件与原始 mRNA 半衰期数据集自动匹配，统一到“基因–异构体”框架下进行后验分析。脚本会在异构体层面整理真实值与预测值、计算每个基因的中位数半衰期与预测偏差，并构建 overall / cross-gene / within-gene / bias-vs-iso 等多种相关性与 R² 指标，为文中“基因内去中心化相关”“交叉基因中位数基线”等分析及后续绘图脚本提供标准输入。:contentReference[oaicite:0]{index=0}  

- **主要输入**：  
  - `CONFIG["dataset"]`：原始 mRNA 半衰期数据集（CSV），包含基因 ID、3′UTR 序列和真实半衰期等列（列名可自动识别或在 CONFIG 中显式指定）。  
  - `CONFIG["predictions"]`：模型在独立测试集上的预测结果（CSV），包含序列列和预测值列，可选包含真实值列。  
  - `CONFIG` 中的列名与选项：  
    - `dataset_seq_col` / `pred_seq_col`：用于匹配的序列列（为空时自动检测）。  
    - `dataset_gene_col`：基因标识列（为空时自动检测）。  
    - `dataset_true_col` / `pred_true_col`：真实半衰期列来源。  
    - `pred_pred_col`：预测值列名。  
    - `map_u_to_t`：是否在匹配前将 U→T 统一为 DNA 格式。  
    - `outdir`：结果输出目录。  
    - `plots`：是否额外生成基础散点图（PNG+SVG）。

- **主要输出**（写入 `CONFIG["outdir"]` 目录）：  
  - `analysis_per_isoform.csv`：  
    - 按异构体（序列）整理的表格，包含：  
      - `gene`、`ios_real`（真实值）、`ios_pre`（预测值）  
      - 基因层中位数 `ref_real` / `ref_pre` 及其差值 `delta_ref`  
      - 去中心化偏差 `d_real` / `d_pre`、异构体误差 `delta_iso`  
      - 若序列在原数据集中对应多个基因，则标记 `ambiguous_seq`。  
  - `per_gene_summary.csv`：  
    - 按基因聚合后的表格，给出每个基因的真实/预测中位数、异构体数量以及基因内 Pearson / Spearman / R² 等指标。  
  - `cross_gene_median.csv`：  
    - 每个基因的真实中位数 `ref_real` 与预测中位数 `ref_pred` 以及观测数 `n`，用于构建交叉基线散点及 R² 拟合。  
  - `metrics_summary.json`：  
    - 全局汇总指标（overall、cross-gene、within-gene pooled、bias vs isoform，以及 per-gene 相关的宏平均、中位数、Fisher-z 加权均值等）。  
  - `matching_log.txt`：  
    - 匹配与数据质量诊断（行数统计、模糊列名解析结果、丢失情况、歧义序列数等）。  
  - 若 `CONFIG["plots"] = True`，还会生成：  
    - `scatter_cross_gene.(png/svg)`：基因层真实/预测中位数散点图。  
    - `scatter_within_gene.(png/svg)`：去中心化 within-gene 散点图。  
    - `scatter_bias_vs_iso.(png/svg)`：基因偏差 vs 异构体误差散点图。

- **使用方式**：  
  1. 根据实际文件路径与列名修改脚本顶部的 `CONFIG` 字典。  
  2. 在终端中运行：  
     ```bash
     python gene_isoform_correlation_analysis_config.py
     ```  
  3. 所有结果将输出到 `CONFIG["outdir"]` 指定的目录中，可作为后续绘图和论文分析的数据输入。


---

### plot_within_gene_supplement.py
- **功能作用**：  
  读取 `gene_isoform_correlation_analysis_config.py` 生成的结果文件，在补充材料中自动绘制两类图像：  
  - **Sx：逐基因相关分布图**（小提琴 + 箱线 + 抖动散点），展示基因内去中心化 Pearson/Spearman 相关系数在所有基因上的分布，并在角标同时给出宏平均、中位数、Fisher-z 加权均值以及 pooled within-gene Pearson。  
  - **Sy：交叉基因中位数散点图**，展示基因层真实中位数 vs 预测中位数的关系，并叠加 y=x 基准线与线性拟合直线，用于展示 cross-gene baseline 拟合程度。:contentReference[oaicite:1]{index=1}  

- **主要输入**（通过脚本顶部常量指定）：  
  - `PER_GENE_CSV`：`per_gene_summary.csv`，优先从中读取逐基因 Pearson/Spearman 相关系数及异构体数量。  
  - `CROSS_GENE_CSV`：`cross_gene_median.csv`，优先从中读取基因层真实/预测中位数。  
  - `PER_ISOFORM_CSV`：`analysis_per_isoform.csv`，在缺少上述信息时用于现场按基因重算 within-gene 相关或基因中位数。  
  - `OUTDIR`：图像输出目录。

- **主要输出**（写入 `OUTDIR` 目录）：  
  - `per_gene_corr_violin.(png/svg)`：  
    - 逐基因 Pearson/Spearman 相关分布的小提琴 + 箱线图，并在图内文字标注宏平均、宏中位数、Fisher-z 加权均值及 pooled within-gene Pearson。  
  - `per_gene_corr_box_jitter.(png/svg)`：  
    - 逐基因相关分布的箱线图叠加抖动散点，每个点对应一个基因的相关系数，用于更直观展示基因层分布。  
  - `cross_gene_median_scatter.(png/svg)`：  
    - 基因层真实中位数 vs 预测中位数散点图，绘制 y=x 参考线和最小二乘拟合直线，并在 1:1 比例坐标系下显示，用于补充 cross-gene 基线拟合情况。

- **使用方式**：  
  1. 将 `PER_GENE_CSV`、`CROSS_GENE_CSV`、`PER_ISOFORM_CSV` 修改为实际生成文件的路径，将 `OUTDIR` 修改为期望的输出目录。  
  2. 确保已安装 `matplotlib` 与 `scienceplots` 等依赖。  
  3. 在终端中运行：  
     ```bash
     python plot_within_gene_supplement.py
     ```  
  4. 脚本会自动在 `OUTDIR` 中生成补充材料所需的 Sx/Sy 图像文件。
