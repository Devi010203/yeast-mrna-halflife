## 此文件夹下的程序都需要依赖于主程序运行后产物，请确保你已经运行完主程序，并在相关脚本中修改相关文件路径

### plot_positional_effects-rel.py（relative FP32 cached 版本）

- **功能作用**：  
  以最终训练好的 mRNA 半衰期预测模型为基础，对测试集序列进行滑窗遮挡（occlusion）实验，计算相对位置效应  
  \[
  r\Delta = \frac{\text{baseline} - \text{occluded}}{\text{baseline}}
  \]  
  全程使用 FP32 精度，并将基线预测与每个窗口的遮挡结果完整缓存到 `result/cache/positional_effects_rel/<RUN_NAME>/` 中，支持断点续跑和“仅聚合绘图”模式；随后聚合得到整体相对位置效应曲线、分段条形图、Top-K 热图及可选的长度分层曲线。

- **主要输入**：  
  - 模型与数据路径（在 `CONFIG` 中指定）：  
    - `MODEL_CODE_PATH`：主训练脚本路径（提供 `Config`、模型类和 `get_device`，如有 `collate_fn_no_chunk` 会优先复用）；  
    - `RUN_DIR`：训练结果目录（需包含 `final_test_predictions.csv`，提供 `sequence` 列）；  
    - `CKPT_PATH`：训练好权重文件（如 `best_model_final.pth`）；  
    - `LOCAL_TOKENIZER_DIR`、`LOCAL_RNAFM_DIR`：本地 RNA-FM tokenizer 和模型目录。  
  - 遮挡与采样设置：如 `SUBSET_N`（参与遮挡的序列数）、`WINDOW_SIZE`、`WINDOW_STEP`、`FILL_CHAR` 等；  
  - 推理与作图参数：如 `BATCH_SIZE`、`NORM_POS_BINS`、`TOP_HEATMAP_K`、`DO_LENGTH_STRATA`、`LENGTH_BINS_ABS` / `LENGTH_QUANTILES`、`DPI`、`FIGSIZE_CURVE`、`FIGSIZE_HEATMAP` 等；  
  - 运行模式：`MODE` 可选 `"all"`（计算+绘图）、`"compute_only"`（只生成/补全缓存）、`"aggregate_only"`（仅从缓存聚合并绘图）。

- **主要输出**：  
  1. **缓存（`result/cache/positional_effects_rel/<RUN_NAME>/`）**  
     - `meta.json`：记录窗口大小、步长、分箱数、遮挡字符、batch 大小、模型路径等关键信息；  
     - `baseline.csv`：每条样本的基线预测值、序列长度、序列 md5 等；  
     - `occl/seq_XXXXXX.csv`：每条样本所有遮挡窗口的 `start`、`center`、`occ_pred`、`delta` 与相对效应 `rel`。

  2. **聚合结果与图像（`result/plot/positional_effects_rel/<timestamp>/`）**  
     - 相对位置效应整体曲线（均值 ± 95%CI）及对应 CSV；  
     - 0–0.2…0.8–1.0 分段均值条形图及 CSV（带显著性星号）；  
     - Top-K 序列的相对效应位置热图及 CSV；  
     - 若启用长度分层，还会输出不同长度层的相对位置效应曲线及对应 CSV。


---
### plot_positional_effects_absolute_from_cache.py

- **功能作用**：  
  从已有的遮挡缓存结果中（`result/cache/positional_effects_rel/<RUN_NAME>/`），在**不再调用模型、只用 CPU** 的前提下，聚合各样本在不同位置的遮挡效应 `Δ = baseline − occluded`，绘制整体“绝对位置效应”曲线、分段均值条形图、Top-K 序列热图，以及可选的按 3′UTR 长度分层的曲线，用于展示在不同相对位置遮挡对预测半衰期的平均影响大小。

- **主要输入**：  
  - `CONFIG["CACHE_DIR"]`：relative 版本脚本产生的缓存目录，内含：
    - `meta.json`：窗口大小、步长、分箱数等元信息；
    - `baseline.csv`：每条序列的基线预测及长度；
    - `occl/seq_XXXXXX.csv`：每条序列各窗口的中心位置 `center` 与 `delta` 等信息。
  - 位置与作图参数：如 `NORM_POS_BINS`、`TOP_HEATMAP_K`、`DO_LENGTH_STRATA`、`LENGTH_BINS_ABS` / `LENGTH_QUANTILES`、`DPI`、`FIGSIZE_CURVE`、`FIGSIZE_HEATMAP` 等。

- **主要输出**（默认写入 `result/plot/positional_effects_abs/<timestamp>/`）：  
  - 绝对位置效应整体曲线及对应 CSV（均值 ± 95%CI）；  
  - 0–0.2…0.8–1.0 分段均值条形图及 CSV（显著区段带星号标记）；  
  - Top-K 序列的绝对效应位置热图及 CSV；  
  - 若启用长度分层，还会输出不同长度层的绝对位置效应曲线及对应 CSV。
