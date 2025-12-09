### 此文件夹下的程序绝大部分都需要依赖于主程序运行后产物，请确保你已经运行完主程序，并在对应脚本中修改相关路径



### `plot_5fold_residual_distribution.py`

- **功能作用**：  
  基于 5 折交叉验证的验证集预测结果，自动读取每折的 `val_predictions_fold*.csv`，计算残差 `pred − true`，为每一折绘制残差直方图（叠加匹配均值/方差的正态曲线）和 QQ 图，并汇总生成各折及全体的残差小提琴图/箱线图，同时导出包含 MAE、RMSE、均值、标准差、偏度、峰度以及正态性检验 p 值等指标的 `summary_per_fold.csv`，用于评估残差分布与正态性假设。:contentReference[oaicite:0]{index=0}  

- **主要输入**：  
  - 在脚本顶部 `FOLD_FILES` 字典中手动填写的各折验证预测文件路径（`val_predictions_fold1.csv` ~ `val_predictions_fold5.csv`），文件需至少包含真实值列和预测值列（列名自动识别）。  
  - 绘图与统计相关参数：如 `NBINS`（直方图箱数）、`DO_LOG1P`（是否额外生成 log1p 版本分析）、`VIOLIN_ABS_Q`（小提琴/箱线 y 轴的 |residual| 截断分位数）等。  

- **主要输出**（自动写入 `<项目根>/result/plot/residual_distribution/<时间戳>/`）：  
  - `hist/`：每折及 All 的残差直方图叠加高斯曲线（`hist_Fold1.(png|svg)` 等）。  
  - `qq/`：每折及 All 的正态 QQ 图（`qq_Fold1.(png|svg)` 等）。  
  - `violin/violin_per_fold.(png|svg)`：各折 + All 的残差分布小提琴图；同目录下还会生成箱线图文件。  
  - `summary_per_fold.csv`：逐折及 All 的统计汇总（N、MAE、RMSE、mean、sd、skew、kurt、KS p、Normaltest p 等）。  
  - 若开启 log1p 分析，会额外生成对应的图像/CSV 版本（命名中包含 log1p 标记）。  


---

### `plot_ablation.py`

- **功能作用**：  
  对一次消融实验（full/UTR_only/polyA_only/masked_polyA/shuffled 等模式）的测试结果进行系统可视化：读取 `ablation_results.csv` 和各模式的 `test_predictions_<mode>.csv`，绘制各模式测试指标柱状图、相对于 full 模式的 Δ 指标（配对 bootstrap 95% 置信区间），并可选地为每个模式生成校准曲线以及运行时间对比图，用于直观比较不同消融设置对模型性能与推理效率的影响。:contentReference[oaicite:1]{index=1}  

- **主要输入**：  
  - `CONFIG["ABLATION_DIR"]`：指向某次消融实验输出目录，内含：  
    - `ablation_results.csv`（如存在）：按模式汇总的 R²、Pearson、Spearman、MSE 等指标；  
    - `test_predictions_<mode>.csv`：各模式测试集预测结果（需有 `true` 与 `pred` 列，可选 `sequence` 列用于配对 bootstrap）；  
    - 可选 `test_metrics_<mode>.json`：补充 Loss、运行时长等信息。  
  - `CONFIG["SAVE_SUBDIR"]`：输出子目录名（位于 `result/plot/<SAVE_SUBDIR>/` 下）。  
  - Bootstrap 与绘图参数：`BOOTSTRAP_N`、`BOOTSTRAP_SEED`、`CI_ALPHA`、`PLOT_CALIB_PER_MODE`、`CALIB_BINS`、`DPI`、`FIGSIZE` 等。  

- **主要输出**（写入 `<项目根>/result/plot/<SAVE_SUBDIR>/<时间戳>/`）：  
  - `ablation_metrics_bar_[R2|Pearson|Spearman|MSE].(png|svg)`：各模式测试指标柱状图。  
  - `delta_vs_full_[R2|Pearson|Spearman|MSE]_with_CI.(png|svg)`：相对 full 模式的 Δ 指标柱状图（配对 bootstrap 95% CI）。  
  - 若 `PLOT_CALIB_PER_MODE=True`：`calibration_<mode>.(png|svg)`，每个模式独立的校准曲线。  
  - 若从 JSON/CSV 中解析到运行时间：`runtime_seconds_bar.(png|svg)`，模式间 runtime 对比。  
  - 过程中还会构建/补全内部 DataFrame 版本的 ablation 汇总结果，必要时从 `test_predictions_<mode>.csv` 动态计算指标。  


---

### `plot_bland_altman.py`

- **功能作用**：  
  针对最终独立测试集预测结果，执行 Bland–Altman 分析：从指定 `RUN_DIR` 下读取测试集预测 CSV，自动识别真实值与预测值列，构建预测与真值的均值–差值关系，分别在原始线性尺度和 log1p 尺度下绘制 Bland–Altman 图，同时计算偏差（bias）、标准差（SD）和一致性界限（LoA = bias ±1.96·SD）及其近似 95% 置信区间，并生成按均值分位分箱的 MAE 条形图和汇总统计文件，用于判断模型是否存在系统性偏差和异方差。:contentReference[oaicite:2]{index=2}  

- **主要输入**：  
  - `RUN_DIR`：一次完整训练的输出目录（需包含最终测试预测 CSV）。  
  - `INPUT_CSV`：在 `RUN_DIR` 中的预测文件名（如 `final_test_predictions.csv`），需包含真实值和预测值两列（列名自动识别）。  
  - 绘图与裁剪参数：  
    - `DPI`、`POINT_SIZE`、`POINT_ALPHA`；  
    - `NBINS_DECILE`：将均值划分为多少分位区间来统计 MAE；  
    - `QTRIM_X` / `QTRIM_Y`：按分位数裁剪 x/y 轴显示范围（去掉极端点影响）；  
    - `Y_ZERO_CENTER`、`LIM_PAD_FRAC`：是否让 y=0 居中以及坐标留白比例。  

- **主要输出**（写入 `<项目根>/result/plot/bland_altman/<时间戳>/`）：  
  - `bland_altman_linear.(png|svg)`：线性尺度的 Bland–Altman 图（均值 vs 差值），带 bias、LoA 线及其近似 95% CI 阴影区和线性趋势拟合。  
  - `bland_altman_log1p.(png|svg)`：log1p 尺度下的 Bland–Altman 图，更稳定地观察比例性偏差。  
  - `mae_by_mean_decile.(png|svg)` 与同名 `.csv`：按均值分位区间统计的 MAE 条形图及数值表。  
  - `summary.txt` 与 `summary.csv`：  
    - MAE、RMSE、R²、Pearson、Spearman 等整体指标；  
    - 线性与 log1p 尺度下的 bias、SD、LoA 及其 95% CI，和均值–差值线性趋势拟合结果（截距、斜率及 p 值）。  


---

### `plot_cv.py`

- **功能作用**：  
  针对五折交叉验证结果进行集中可视化和诊断：从主程序输出目录中自动发现 `cv_summary.csv`、`training_log.json`、`val_predictions_fold*.csv`（以及可选的 `learning_rate_schedule_fold*.csv`），绘制每折验证 R² 柱状图、R² 箱线图（支持 aggregate / per_epoch / bootstrap 三种模式）、Val R² 与 Loss 学习曲线、每折验证散点拼版图、每折校准曲线以及（可选）学习率曲线，用于系统展示模型在交叉验证过程中的稳定性和收敛行为。:contentReference[oaicite:3]{index=3}  

- **主要输入**：  
  - `CONFIG["RUN_DIR"]`：主训练程序某次运行的输出目录，需包含：  
    - `cv_summary.csv`（如缺失，则从 `val_predictions_fold*.csv` 反算各折验证 R²）；  
    - `training_log.json`（或 `.jsonl`）：含每个 epoch 的 train/val loss、val_r2、fold 等字段；  
    - `val_predictions_fold*.csv`：每折验证预测结果（至少有 `true`、`pred`，若含 `sequence` 可用于更复杂分析）；  
    - 可选 `learning_rate_schedule_fold*.csv`：记录每折的学习率随 epoch 的变化。  
  - `CONFIG["SAVE_SUBDIR"]`：绘图输出子目录名（如 `"5foldplot-2"`）。  
  - 其他配置项：  
    - `CALIB_BINS`：校准曲线分箱数；  
    - `R2_BOX_MODE`：R² 箱线图模式（`"bootstrap"` / `"per_epoch"` / `"aggregate"`）；  
    - `BOOT_N`、`BOOT_SEED`：bootstrap 箱线模式下的重复次数与随机种子；  
    - `JITTER_MAX_POINTS`：箱线图上叠加抖动散点的最大数量上限。  

- **主要输出**（写入 `<项目根>/result/plot/<SAVE_SUBDIR>/<时间戳>/`）：  
  - 交叉验证性能：  
    - `cv_r2_bar.(png|svg)`：各折验证 R² 柱状图（含均值虚线）。  
    - `cv_r2_box_aggregate.(png|svg)` / `cv_r2_box_per_epoch.(png|svg)` / `cv_r2_box_bootstrap.(png|svg)`：三种模式下的 R² 箱线图（实际根据 `R2_BOX_MODE` 选择）。  
  - 学习曲线：  
    - `cv_valR2_learning_curves.(png|svg)`：每折 Val R² 随 epoch 变化。  
    - `cv_train_loss_learning_curves.(png|svg)`：每折训练集 loss 曲线（4:3 比例，线性坐标）。  
    - `cv_val_loss_learning_curves.(png|svg)`：每折验证集 loss 曲线。  
    - `cv_loss_learning_curves_combined.(png|svg)`：train/val loss 合并在同一图中。  
  - 验证散点与校准：  
    - `cv_val_scatter_folds.(png|svg)`：拼版展示每折验证集 true vs pred 散点图，子图统一坐标、1:1 比例，折间颜色/marker 保持一致。  
    - 每折校准曲线图（文件名类似 `calib_fold*.png|svg`，由 `calibration_curve_tight` 生成，仅在数据充足时输出）。  
  - 若存在学习率文件：  
    - 每折学习率随 epoch 变化的曲线图（文件名中含 `lr` 标记）。  

### `plot_dataset_overview.py`

**功能作用：**  
对按 Train / Val / Test 等划分的数据集做整体分布概览，生成论文/报告中常用的“数据集描述”图表与统计表。主要包括：  
- 目标变量（mRNA 半衰期）的直方图 + KDE 曲线（各划分叠加对比）  
- 3′UTR 长度分布的直方图 + KDE（各划分叠加）  
- GC 含量分布的直方图 + KDE（各划分叠加）  
- 3′UTR 长度 × GC 含量的 hexbin 二维密度图（每个划分一幅子图）  
- 各划分的样本数量、目标分布（均值/中位数/IQR/最值/偏度/峰度）以及长度和 GC 的均值±标准差统计表  
:contentReference[oaicite:0]{index=0}  

**输入：**  
- 在脚本顶部 `SPLITS` 字典中手动指定的若干 CSV 文件路径，每个划分可以是单个或多个 CSV，脚本会自动合并：  
  - 例如：`"Train": [".../train_set.csv"]`、`"Val.": [".../val_set.csv"]`、`"Test": [".../test_set.csv"]`  
- CSV 中需要包含：  
  - 半衰期/真值列（脚本会自动在列名中识别，如 `half_life` / `target` / `true` 等）  
  - 序列列 `sequence`（优先用于计算长度与 GC），或已有的长度 / GC 列（如 `len` / `length` / `gc_frac` 等）

**输出：**  
- 目录：`<项目根>/result/plot/dataset_overview/<时间戳>/`  
- 图像（均为 PNG + SVG，dpi=400）：  
  - 目标分布叠加图  
  - 3′UTR 长度分布叠加图  
  - GC 含量分布叠加图  
  - 长度 × GC hexbin 面板图  
- 统计表：  
  - `summary_by_split.csv`：按划分的 N、目标分布统计量及长度/GC 的均值和标准差  


---

### `plot_error_2d_heatmap.py`

**功能作用：**  
在测试集上分析模型误差在「3′UTR 长度 × GC 含量」二维空间中的分布，生成 MAE / Bias / Count 的二维热图和对应的栅格统计表，用于揭示模型在哪些长度-GC 区间误差较大或存在系统偏差。:contentReference[oaicite:1]{index=1}  

**输入：**  
- 在脚本顶部手动指定：  
  - `RUN_DIR`：完整实验输出目录  
  - `INPUT_FILE`：通常为 `final_test_predictions.csv`  
- `final_test_predictions.csv` 中需要包含：  
  - 真值列（如 `true` / `target` / `half_life` 等，脚本自动识别）  
  - 预测值列（如 `pred` / `prediction` / `y_pred` 等，脚本自动识别）  
  - 序列列 `sequence`（优先用于计算长度与 GC），或现成的长度/GC 列（`len` / `length` / `gc_frac` 等）

**输出：**  
- 目录：`<项目根>/result/plot/error_2d_heatmap/<时间戳>/`  
- CSV：  
  - `grid_equalwidth.csv`：等宽分箱下，每个长度 × GC 网格的样本数、MAE、RMSE、Bias  
  - `grid_quantile.csv`：按分位数分箱下的同类统计  
- 图像（PNG + SVG，dpi=400）：  
  - `heat_mae_equalwidth` / `heat_bias_equalwidth` / `heat_count_equalwidth`  
  - `heat_mae_quantile` / `heat_bias_quantile` / `heat_count_quantile`  
  对应 MAE、Bias（pred−true 的均值）和样本计数的二维热图  


---

### `plot_error_breakdown_scatter.py`

**功能作用：**  
从测试集的预测结果出发，系统地分析“误差与 3′UTR 长度、GC 含量”的关系：  
- 预测值 vs 3′UTR 长度（散点 + 分位平滑 + 95% CI）  
- 残差 vs 3′UTR 长度（散点 + 分位平滑 + 95% CI）  
- 预测值 vs GC 含量（散点 + 分位平滑 + 95% CI）  
- 残差 vs GC 含量（散点 + 分位平滑 + 95% CI）  
- 按长度、GC 进行等宽和分位分箱后的 MAE 柱状图  
同时计算整体 MAE、RMSE、R²、Pearson/Spearman 相关系数等指标，便于写入论文结果部分。:contentReference[oaicite:2]{index=2}  

**输入：**  
- 在脚本顶部手动指定：  
  - `RUN_DIR`：完整实验输出目录  
  - `INPUT_FILE`：通常为 `final_test_predictions.csv`  
- `final_test_predictions.csv` 中需要包含：  
  - 真值列与预测列（自动识别，如 `true` / `pred` 等）  
  - `sequence` 列（优先用于计算 3′UTR 长度与 GC），或长度/GC 列（`len` / `length` / `gc_frac` 等）

**输出：**  
- 目录：`<项目根>/result/plot/error_breakdown_scatter/<时间戳>/`  
- 文本与表格：  
  - `summary.txt` / `summary.csv`：整体 MAE、RMSE、R²、Pearson、Spearman  
  - 若干 *\*_smooth.csv：分位平滑后的曲线数据（长度/GC vs 预测或残差）  
  - 若干 `mae_by_len_*.csv`、`mae_by_gc_*.csv`：不同分箱方式下的 MAE 统计  
- 图像（PNG + SVG，dpi=400）：  
  - 预测/残差 vs 长度、GC 的散点 + 平滑曲线图  
  - 长度/GC 等宽 & 分位分箱的 MAE 柱状图  


---

### `plot_final_training.py`

**功能作用：**  
面向“完整训练（final training）”阶段的一键绘图脚本：从最终一次实验输出目录中集中读取曲线和预测结果文件，生成论文中需要的训练过程与测试表现图，包括：  
- Train / Val loss 随 epoch 变化的单独曲线图  
- 验证集 R² / MSE / 相关系数随 epoch 的曲线  
- 学习率调度曲线  
- 验证集、测试集的 parity 图（True vs Predicted）  
- 测试集残差直方图  
- 若存在 `sequence`：测试集“误差 vs 序列长度”关系图  
- 测试集按预测值分箱的校准曲线（bin 均值预测 vs bin 均值真实）以及对应的分箱统计表  
:contentReference[oaicite:3]{index=3}  

**输入：**  
- 在脚本顶部 `INPUT["EXP_DIR"]` 中手动指定最终训练实验的输出目录，该目录下需包含：  
  - `training_curve_final.csv`：逐 epoch 的 train/val loss 和验证指标  
  - `learning_rate_schedule_final.csv`：逐 epoch 学习率  
  - `val_predictions_final.csv`：最优模型在验证集上的 true/pred  
  - `final_test_predictions.csv`：最优模型在测试集上的 true/pred（可含 sequence）  
  - `final_test_metrics.json`（可选）：测试集整体指标  

**输出：**  
- 目录：`<项目根>/result/plot/finaltrain_plot/<时间戳>/`  
- 图像（PNG + SVG，dpi=400）：  
  - 训练/验证 loss 曲线  
  - 验证集各指标随 epoch 曲线  
  - 学习率调度图  
  - 验证集、测试集 parity 图  
  - 测试集残差直方图  
  - 测试集误差 vs 序列长度图（如有 sequence）  
  - 测试集校准曲线图  
- 表格：  
  - `final_test_error_vs_length.csv`：长度分箱后的平均绝对误差等统计  
  - `final_test_calibration_deciles.csv`：按预测值分箱的校准统计（bin 内样本数、均值真值、均值预测、MAE、MAPE 等）  

---

### `plot_parity_stratified.py`  

**功能作用：**  
在完整测试集上，按 3′UTR 序列长度和 GC 含量的分位数进行分层，分别绘制各层的 Parity 图（真值 vs 预测），并在每一层给出线性拟合线和评估指标（N、MAE、RMSE、R²、斜率、截距），同时生成按层的 10-bin 校准折线，用于分析模型在不同长度/GC 区间的拟合偏差与可靠性。  

**输入：**  
- 在脚本顶部手动指定：  
  - `RUN_DIR`：一次完整训练的输出目录，内部包含测试集预测结果文件  
  - `INPUT_FILE`：如 `final_test_predictions.csv`  
- 输入 CSV 中需包含：  
  - 真值列（如 `true`/`target`/`half_life` 等，自动识别）  
  - 预测列（如 `pred`/`y_pred`/`prediction` 等，自动识别）  
  - 序列列 `sequence`（用于自动计算 3′UTR 长度和 GC），或已有长度/GC 列（如 `len`/`length` 和 `gc_frac` 等）

**输出：**  
- 目录：`<项目根>/result/plot/parity_stratified/<时间戳>/`，内部包含：  
  - `length/`：按长度分层的 Parity 面板图（Q1–Q4）、对应分层指标表 `summary_length_quartiles.csv`  
  - `gc/`：按 GC 分层的 Parity 面板图（Q1–Q4）、对应分层指标表  
  - `lines/`：长度和 GC 各自的分层校准折线图及其 CSV（每层 10-bin 的 mean truth / mean pred / count）  
- 所有图像均导出为 `.png` 与 `.svg`（dpi=400）  


---

### `plot_residual_diagnostics.py`  

**功能作用：**  
对完整测试集的残差（预测值 − 真值）进行系统诊断，绘制残差 vs 预测值、残差 vs 真值（均带分位平滑曲线与 95% 置信区间）、残差直方图、QQ 图以及“按预测分位数分组的 MAE”柱状图，并可选生成 log1p 空间下的对应图像，用于检查偏差、异方差性和接近正态的程度。  

**输入：**  
- 在脚本顶部手动指定：  
  - `RUN_DIR`：完整训练输出目录  
  - `INPUT_FILE`：如 `final_test_predictions.csv`  
  - 其他控制参数：是否生成 log1p 版本、平滑分位点、直方图箱数等  
- 输入 CSV 中需包含：  
  - 真值列（自动从常见列名中识别，如 `true`/`target`/`half_life` 等）  
  - 预测列（如 `pred`/`y_pred`/`prediction` 等）

**输出：**  
- 目录：`<项目根>/result/plot/fulltrain_plot/residual_diagnostics/<时间戳>/`  
- 文本与表格：  
  - `summary.txt` 与 `summary.csv`：整体 MAE、RMSE、R²、Pearson、Spearman  
  - `resid_vs_pred_smooth.csv`、`resid_vs_true_smooth.csv`：线性残差的分位平滑曲线与 95% CI 数据  
  - 若开启 `DO_LOG1P=True`，额外导出 log1p 残差版本的平滑曲线 CSV  
  - `mae_by_pred_decile.csv`：按预测值分位数（十等分）计算的 MAE  
- 图像（均为 PNG+SVG，dpi=400）：  
  - `residual_vs_prediction` / `residual_vs_truth`  
  - `residual_qqplot`（QQ 图）  
  - `mae_by_prediction_decile`（MAE vs 预测分位）  
  - 以及 log1p 版本的残差诊断图（如已启用）  


---

### `plot_results_overview_table.py`  
  
**功能作用：**  
汇总 5 折交叉验证和最终测试集的性能指标，生成“结果总览表”。包括：每一折的 MAE、RMSE、R²、Pearson、Spearman、样本数 N；5 折指标的“均值±标准差”；将 5 折样本拼接后的 Pooled 指标；以及最终测试集指标。同时导出 LaTeX 表格和一张适合直接放入论文或幻灯片的表格图片。  

**输入：**  
- 在脚本顶部手动指定：  
  - `FOLD_FILES`：一个字典，键为折名（如 `"fold1"`–`"fold5"`），值为各折验证集预测 CSV 的路径（如 `val_predictions_fold1.csv` 等）  
  - `FINAL_TEST_FILE`：完整测试集预测 CSV（如 `final_test_predictions.csv`）  
- 各 CSV 文件需包含：  
  - 真值列（自动识别）  
  - 预测列（自动识别）

**输出：**  
- 目录：`<项目根>/result/plot/overview_table/<时间戳>/`  
- 指标类 CSV：  
  - `cv_metrics_per_fold.csv`：各折 N、MAE、RMSE、R²、Pearson、Spearman  
  - `cv_summary.csv`：5 折指标的均值与标准差  
  - `cv_pooled.csv`：5 折样本拼接后的整体指标  
  - `final_test_metrics.csv`：最终测试集指标（若提供测试集文件）  
- 表格与图像：  
  - `overview_table.tex`：booktabs 风格的 LaTeX 表格源码  
  - `overview_table.png` / `overview_table.svg`：用 matplotlib 渲染的总览表图片（列为 Split / N / MAE / RMSE / R² / Pearson / Spearman）  


---

### `plot_test_error_breakdown.py`  

**功能作用：**  
对最终测试集的误差和可靠性进行全面剖析，生成论文用的多张关键图表：包括按真实值分位数分箱的 MAE/RMSE 条形图、带 95% 置信区间的 10-bin 校准曲线、整体 Parity 图（支持 hexbin 或散点，附带 R²、相关系数、斜率截距和 MAE 注释）、Bland–Altman 图，以及误差随序列长度与 GC 含量变化的分析图和“MAE & RMSE 合并条形图”，帮助系统评估模型在不同尺度上的表现与偏差模式。  

**输入：**  
- 在脚本顶部的 `CONFIG` 字典中手动指定：  
  - `"RUN_DIR"`：包含 `final_test_predictions.csv` 的完整训练输出目录  
  - `"DATA_CSV"`（可选）：包含全体样本及 `sequence`、`Isoform Half-Life` 等信息的总数据表，用于补充长度/GC 特征  
  - 分箱个数、bootstrap 次数、Paritiy 图类型（hexbin 或 scatter）、图像 DPI/尺寸等参数  
- `RUN_DIR/final_test_predictions.csv` 中需包含：  
  - 真值列（自动从常见列名识别）  
  - 预测列（自动识别）  
  - 可选 `sequence` 列（存在时用来计算长度和 GC）

**输出：**  
- 目录：`<项目根>/result/plot/test_error_breakdown/<时间戳>/`  
- 主要 CSV：  
  - `calibration_deciles.csv`：按真值分位分箱的校准数据（bin 内 true 均值、pred 均值、样本数、pred 均值的 95% CI）  
  - 若干中间表，用于记录按真实值/特征分箱后的 MAE、RMSE 及样本数等统计  
- 主要图像（全部 PNG+SVG，dpi=400）：  
  1. `binned_metrics_MAE_by_true_bins` / `binned_metrics_RMSE_by_true_bins`：按真实值分位数分箱的 MAE/RMSE 条形图  
  2. `error_bins_bar`：MAE 与 RMSE 合并的双轴条形图（同时标注各 bin 样本数）  
  3. `calibration_deciles_ci`：10-bin 校准曲线，带 pred 均值的 bootstrap 置信区间  
  4. `parity_hexbin` 或 `parity_density`：整体 Parity 图（附 R²、Pearson、Spearman、斜率、截距、MAE）  
  5. `bland_altman_test`：Bland–Altman 差异-均值图  
  6. `error_vs_length`、`error_vs_gc` 等误差随长度/GC 变化的图（若能获取序列特征）  
