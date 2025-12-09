### Most of the programs in this folder depend on the main program, so make sure you run the main program and change the paths in the corresponding scripts.



### `plot_5fold_residual_distribution.py`

- **Functionality**:
  
  Based on the prediction results of 5-fold cross validation, automatically read `val_predictions_fold*.csv` for each fold, calculate residuals `pred - true`, plot histogram of residuals (superimposed normal curves with matched mean/variance) and QQ plots for each fold and summarize them to generate residual violin/boxplot for each fold and all. We also export `summary_per_fold.csv` with MAE, RMSE, mean, standard deviation, skewness, kurtosis, and p-value of normality test for evaluating the distribution of residuals and assumption of normality. :contentReference[oaicite:0]{index=0}
  

- **Main Input**:
  
  - Path to each fold validation prediction file (`val_predictions_fold1.csv` ~ `val_predictions_fold5.csv`) manually filled in the `FOLD_FILES` dictionary at the top of the script, the file needs to contain at least the true value columns and the predicted value columns (the column names are automatically recognized).
  
  - Plotting and statistics related parameters: e.g. `NBINS` (number of histogram boxes), `DO_LOG1P` (whether or not to generate an additional log1p version of the analysis), `VIOLIN_ABS_Q` (|residual| truncation quartile for the y-axis of the violin/box line), etc.
  

- **Main output** (automatically written to `<project root>/result/plot/residual_distribution/<timestamp>/`):
  
  - `hist/`: histogram of residuals per fold and All overlaid with Gaussian curves (`hist_Fold1.(png|svg)` etc.).
  
  - `qq/`: normal QQ plots per fold and All (`qq_Fold1.(png|svg)` etc.).
  
  - `violin/violin_per_fold.(png|svg)`: violin plots of residual distributions for each fold + All; boxplot files are also generated in the same directory.
  
  - `summary_per_fold.csv`: summary statistics (N, MAE, RMSE, mean, sd, skew, kurt, KS p, Normaltest p, etc.) for each fold + All.
  
  - If log1p analysis is turned on, the corresponding image/CSV version is additionally generated (with log1p tag in the naming).
  


---

### `plot_ablation.py`

- **Functionality role**:
  
  Systematically visualize the test results of an ablation experiment (full/UTR_only/polyA_only/masked_polyA/shuffled modes): read `ablation_results.csv` and `test_predictions_<mode>.csv` for each mode. Histograms of test metrics for each mode, ∆ metrics relative to full mode (with paired bootstrap 95% confidence intervals), and optionally calibration curves for each mode, as well as runtime comparisons, are generated to visually compare the impact of different ablation settings on model performance and inference efficiency. :contentReference[oaicite:1]{index=1}
  

- **Primary Inputs**:
  
  - `CONFIG[“ABLATION_DIR”]`: points to the output directory of a particular ablation experiment, containing:
  
    - `ablation_results.csv` (if present): metrics such as R², Pearson, Spearman, MSE, etc. summarized by mode;
  
    - `test_predictions_<mode>.csv`: predictions for each mode test set (`true` and `pred` columns are required, with optional `sequence` columns for pairing bootstrap);
  
    - Optional `test_metrics_<mode>.json`: additional Loss, runtime, etc. information.
  
  - `CONFIG[“SAVE_SUBDIR”]`: output subdirectory name (under `result/plot/<SAVE_SUBDIR>/`).
  
  - Bootstrap and plotting parameters: `BOOTSTRAP_N`, `BOOTSTRAP_SEED`, `CI_ALPHA`, `PLOT_CALIB_PER_MODE`, `CALIB_BINS`, `DPI`, `FIGSIZE`, and so on.
  

- **Main output** (write to `<project root>/result/plot/<SAVE_SUBDIR>/<timestamp>/`):
  
  - `ablation_metrics_bar_[R2|Pearson|Spearman|MSE]. (png|svg)`: histogram of test metrics for each model.
  
  - `delta_vs_full_[R2|Pearson|Spearman|MSE]_with_CI.(png|svg)`: histogram of Δ metrics relative to the full model (paired bootstrap 95% CI).
  
  - If `PLOT_CALIB_PER_MODE=True`: `calibration_<mode>. (png|svg)`, independent calibration curves for each mode.
  
  - If parsing runtime from JSON/CSV: `runtime_seconds_bar.(png|svg)`, runtime comparison between modes.
  
  - The process also builds/completes the internal DataFrame version of the ablation aggregation, and dynamically calculates metrics from `test_predictions_<mode>.csv` if necessary.
  


---

### `plot_bland_altman.py`

- **Functionality**:
  
  Performs Bland-Altman analysis on the final independent test set predictions: reads the test set prediction CSV from the specified `RUN_DIR`, automatically identifies the true and predicted columns, constructs the mean-difference relationship between the predicted and true values, and plots the Bland-Altman plots on the original linear scale and log1p scale, respectively. Bland-Altman plots are plotted on the original linear scale and log1p scale, respectively, and the bias, standard deviation (SD), and consistency bounds (LoA = bias ±1.96-SD) and their approximate 95% confidence intervals are computed at the same time, and MAE bar charts and summary statistics files are generated by mean-quantile bins, which can be used to determine whether there are systematic biases and heteroskedasticity in the model. :contentReference[oaicite:2]{index=2}
  

- **Primary Inputs**:
  
  - `RUN_DIR`: output directory of one complete training (needs to contain the final test prediction CSV).
  
  - `INPUT_CSV`: the name of the predictions file in `RUN_DIR` (e.g. `final_test_predictions.csv`), which needs to contain both the true and predicted value columns (the column names are automatically recognized).
  
  - Plotting and cropping parameters:
  
    - `DPI`, `POINT_SIZE`, `POINT_ALPHA`;
  
    - `NBINS_DECILE`: how many quantile intervals to divide the mean into to count MAE;
  
    - `QTRIM_X` / `QTRIM_Y`: cropping the x/y axis display range by quartiles (removing the effect of extreme points);
  
    - `Y_ZERO_CENTER`, `LIM_PAD_FRAC`: whether or not to let y=0 center and the percentage of coordinate white space.
  

- **Main output** (write to `<project root>/result/plot/bland_altman/<timestamp>/`):
  
  - `bland_altman_linear.(png|svg)`: linear-scale Bland-Altman plot (mean vs. difference) with bias, LoA line and its approximate 95% CI shaded area, and linear trend fit.
  
  - `bland_altman_log1p.(png|svg)`: Bland-Altman plot at log1p scale, for a more stable view of proportionality bias.
  
  - `mae_by_mean_decile.(png|svg)` with the same name `.csv`: MAE bar chart and table of values by mean quantile interval.
  
  - `summary.txt` with `summary.csv`:
  
    - MAE, RMSE, R², Pearson, Spearman, and other overall metrics;
  
    - Linear vs. log1p scales for bias, SD, LoA and its 95% CI, and mean-difference linear trend fitting results (intercept, slope, and p-value).
  


---

### `plot_cv.py`

- **Functional role**:
  
  Centralized visualization and diagnostics for 50-fold cross-validation results: automatically discover `cv_summary.csv`, `training_log.json`, `val_predictions_fold*.csv` (and optionally `learning_rate_schedule_fold*.csv`) from the main program output directory, plot per-fold validation R² histograms, R² box-lines, and `plot_cv*.py`. fold*.csv`), plots per-fold validation R² histograms, R² box plots (supports aggregate / per_epoch / bootstrap modes), Val R² vs. Loss learning curves, per-fold validation scatter plots, per-fold calibration curves, and (optional) learning rate curves, which are used to systematically demonstrate the stability and convergence behavior of the model during the cross-validation process. and convergence behavior during cross-validation. :contentReference[oaicite:3]{index=3}
  

- **Primary Inputs**:
  
  - `CONFIG[“RUN_DIR”]`: the output directory of a particular run of the main training program to be included:
  
    - `cv_summary.csv` (if missing, back-calculate each fold validation R² from `val_predictions_fold*.csv`);
  
    - `training_log.json` (or `.jsonl`): contains fields for train/val loss, val_r2, fold, etc. for each epoch;
  
    - `val_predictions_fold*.csv`: per-fold validation predictions (at least `true`, `pred`, or `sequence` for more complex analysis);
  
    - Optional `learning_rate_schedule_fold*.csv`: record learning rate per fold over epoch.
  
  - `CONFIG[“SAVE_SUBDIR”]`: name of plot output subdirectory (e.g. `“5foldplot-2”`).
  
  - Other configuration items:
  
    - `CALIB_BINS`: number of calibration curve bins;
  
    - `R2_BOX_MODE`: R² box plot mode (`“bootstrap”` / `‘per_epoch’` / `“aggregate”`);
  
    - `BOOT_N`, `BOOT_SEED`: number of repetitions in bootstrap boxplot pattern with random seed;
  
    - `JITTER_MAX_POINTS`: upper limit on the maximum number of jitter scatters to be superimposed on the boxline plot.
  

- **Main output** (write to `<project root>/result/plot/<SAVE_SUBDIR>/<timestamp>/`):
  
  - Cross-validation performance:
  
    - `cv_r2_bar.(png|svg)`: each fold validation R² histogram (with mean dashed line).
  
    - `cv_r2_box_aggregate.(png|svg)` / `cv_r2_box_per_epoch.(png|svg)` / `cv_r2_box_bootstrap.(png|svg)`: R² boxplots in three modes (actually selected according to `R2_BOX_MODE`).
  
  - Learning curve:
  
    - `cv_valR2_learning_curves.(png|svg)`: per-fold Val R² with epoch.
  
    - `cv_train_loss_learning_curves.(png|svg)`: per-fold training set loss curves (4:3 scale, linear coordinates).
  
    - `cv_val_loss_learning_curves.(png|svg)`: per-fold validation set loss curves.
  
    - `cv_loss_learning_curves_combined.(png|svg)`: train/val losses combined in the same plot.
  
  - Validation scatter with calibration:
  
    - `cv_val_scatter_folds.(png|svg)`: collage showing per-fold validation set true vs pred scatter plots, with subplots in uniform coordinates, 1:1 scale, and consistent color/marker between folds.
  
    - Per-fold calibration curve plot (file name similar to `calib_fold*.png|svg`, generated by `calibration_curve_tight`, only output when there is enough data).
  
  - If a learning rate file exists:
  
    - Plot of learning rate per fold as a function of epoch (with `lr` tag in file name).
  

### `plot_dataset_overview.py`

**Functional role:**  
Provides an overview of the overall distribution of the dataset by Train / Val / Test, etc., and generates “dataset description” charts and statistical tables commonly used in papers/reports. This includes:
  
- Histogram of target variable (mRNA half-life) + KDE curve (stacked comparison of each division)
  
- Histogram of 3′UTR length distribution + KDE (stacked for each division)
  
- Histogram of GC content distribution + KDE (each division superimposed)
  
- hexbin 2D density plot of 3′UTR length × GC content (one subplot per division)
  
- Sample size, target distribution (mean/median/IQR/maximum/skewness/kurtosis) and mean±standard deviation statistics for length and GC for each division
  
:contentReference[oaicite:0]{index=0}
  

**Input:**  
- Paths to a number of CSV files manually specified in the `SPLITS` dictionary at the top of the script, each division can be a single or multiple CSVs, which are automatically merged by the script:
  
  - Example: `“Train”: ["... /train_set.csv“]`, `”Val.“: [”... /val_set.csv“]`, `‘Test’: [”... /test_set.csv"]`  
- CSV needs to be included:
  
  - Half-life/true columns (which the script automatically recognizes in the column names, e.g. `half_life` / `target` / `true`, etc.)
  
  - Sequence columns `sequence` (preferred for length and GC calculations), or existing length / GC columns (e.g. `len` / `length` / `gc_frac` etc.)

**Output:**  
- Directory: `<project root>/result/plot/dataset_overview/<timestamp>/`  
- Images (all PNG + SVG, dpi=400):
  
  - Target distribution overlay
  
  - 3′UTR length distribution overlay
  
  - GC content distribution overlay
  
  - Length × GC hexbin panel
  
- Statistical tables:
  
  - `summary_by_split.csv`: mean and standard deviation of N, target distribution statistic and length/GC by split
  


---.

### `plot_error_2d_heatmap.py`.

**Functional role:**  
Analyze the distribution of model error in the ‘3′ UTR length × GC content’ 2D space on the test set, and generate a 2D heatmap of MAE / Bias / Count and corresponding raster statistics, which can be used to reveal in which length-GC intervals the model has large error or systematic bias. :contentReference[oaicite:1]{index=1}
  

**INPUT:**  
- Specify manually at the top of the script:
  
  - `RUN_DIR`: full experiment output directory
  
  - `INPUT_FILE`: usually `final_test_predictions.csv`.  
- needs to be included in `final_test_predictions.csv`:
  
  - Truth columns (e.g. `true` / `target` / `half_life`, etc., automatically recognized by the script)
  
  - Prediction columns (e.g. `pred` / `prediction` / `y_pred`, etc., automatically recognized by the script)
  
  - Sequence columns `sequence` (preferred for calculating length with GC), or off-the-shelf length/GC columns (`len` / `length` / `gc_frac`, etc.)

**Output:**  
- Directory: `<project root>/result/plot/error_2d_heatmap/<timestamp>/`  
- CSV:
  
  - `grid_equalwidth.csv`: number of samples, MAE, RMSE, Bias for each length × GC grid under equal-width binning
  
  - `grid_quantile.csv`: like-for-like statistics under quartile binning
  
- Image (PNG + SVG, dpi=400):
  
  - `heat_mae_equalwidth` / `heat_bias_equalwidth` / `heat_count_equalwidth`    - `heat_mae_quantile` / `heat_bias_quantile` / `heat_count_quantile`    2D heatmap corresponding to MAE, Bias (mean of pred-true) and sample counts
  


---

### `plot_error_breakdown_scatter.py`

**Functional role:**  
Systematically analyze the relationship between “error and 3′UTR length, GC content” from the prediction results of the test set:
  
- Prediction vs 3′UTR length (scatter + quantile smoothing + 95% CI)
  
- Residual vs 3′UTR length (scatter + quantile smoothing + 95% CI)
  
- Predicted vs GC content (scatter + quantile smoothing + 95% CI)
  
- Residual vs GC content (scatter + quantile smoothing + 95% CI)
  
- MAE histogram after equal width and binning by length, GC
  
Overall MAE, RMSE, R², and Pearson/Spearman correlation coefficients were also calculated for inclusion in the results section of the paper. :contentReference[oaicite:2]{index=2}
  

**Input:**  
- Specify manually at the top of the script:
  
  - `RUN_DIR`: full experiment output directory
  
  - `INPUT_FILE`: usually `final_test_predictions.csv`.  
- needs to be included in `final_test_predictions.csv`:
  
  - True and predicted columns (automatically recognized, e.g. `true` / `pred` etc.)
  
  - `sequence` columns (preferred for calculating 3′UTR length with GC), or length / GC columns (`len` / `length` / `gc_frac` etc.)

**Output:**  
- Directory: `<project root>/result/plot/error_breakdown_scatter/<timestamp>/`  
- Text and tables:
  
  - `summary.txt` / `summary.csv`: overall MAE, RMSE, R², Pearson, Spearman
  
  - Several *\*_smooth.csv: quantile smoothed curve data (length/GC vs. prediction or residuals)
  
  - Several `mae_by_len_*.csv`, `mae_by_gc_*.csv`: MAE statistics for different binning methods
  
- Images (PNG + SVG, dpi=400):
  
  - Scatter + smoothed plot of prediction/residual vs length, GC
  
  - MAE histogram of length/GC equal width & binned bins
  


---

### `plot_final_training.py`

**Functional role:**  
One-click plotting script for the “final training” phase: centrally read the curve and prediction result files from the final experiment output directory, and generate the training process and test performance plots needed in the paper, including:
  
- Separate plots of Train / Val loss as a function of epoch.
  
- Validation set R² / MSE / correlation coefficients versus epoch
  
- Learning rate scheduling curve
  
- Parity plots for validation set, test set (True vs Predicted)
  
- Histogram of test set residuals
  
- If `sequence` is present: plot of “Error vs Sequence Length” for the test set.
  
- Calibration curves for the test set binned by prediction (bin mean prediction vs bin mean true) and the corresponding bin statistics.
  
:contentReference[oaicite:3]{index=3}
  

**Input: **  
- Manually specify the output directory of the final training experiment in `INPUT[“EXP_DIR”]` at the top of the script, which needs to be included:
  
  - `training_curve_final.csv`: epoch-by-epoch train/val loss and validation metrics
  
  - `learning_rate_schedule_final.csv`: epoch-by-epoch learning rate
  
  - `val_predictions_final.csv`: true/pred for the optimal model on the validation set.
  
  - `final_test_predictions.csv`: true/pred of the optimal model on the test set (can contain sequences)
  
  - `final_test_metrics.json` (optional): overall metrics for the test set
  

**Output:**  
- Directory: `<project root>/result/plot/finaltrain_plot/<timestamp>/`  
- Image (PNG + SVG, dpi=400):
  
  - Training/validation loss curves
  
  - Validation set metrics with epoch curve
  
  - Learning rate scheduling graph
  
  - Validation set, test set parity graph
  
  - Histogram of test set residuals
  
  - Test set error vs sequence length plot (if sequence)
  
  - Test set calibration curve
  
- Table:
  
  - `final_test_error_vs_length.csv`: mean absolute error after length binning and other statistics
  
  - `final_test_calibration_deciles.csv`: calibration statistics (number of samples in bin, mean true, mean predicted, MAE, MAPE, etc.) for bins by predicted values.
  

---

### `plot_parity_stratified.py`  

**Functionality: **  
On the full test set, stratify by quartiles of 3′UTR sequence length and GC content, plot Parity plots (true vs. predicted) for each stratum separately, and give linear fit lines and evaluation metrics (N, MAE, RMSE, R², slope, intercept) at each stratum, as well as generating 10-bin calibration folds by stratum for analyzing the model's fit bias and reliability.
  

**Input: **  
- Specify manually at the top of the script:
  
  - `RUN_DIR`: the output directory of a complete training session, which internally contains the test set prediction results file
  
  - `INPUT_FILE`: e.g. `final_test_predictions.csv`.  
- to be included in the input CSV:
  
  - Truth columns (e.g. `true`/`target`/`half_life`, etc., automatically recognized)
  
  - Prediction columns (e.g. `pred`/`y_pred`/`prediction`, etc., automatically recognized)
  
  - Sequence columns `sequence` (for automatic calculation of 3′UTR length and GC), or existing length/GC columns (e.g. `len`/`length` and `gc_frac`, etc.)

**Output:**  
- Directory: `<project root>/result/plot/parity_stratified/<timestamp>/`, internally contained:
  
  - `length/`: Parity panel plot (Q1-Q4) stratified by length, table of corresponding stratified metrics `summary_length_quartiles.csv`    - `gc/`: Parity panel (Q1-Q4) stratified by GC, corresponding stratified metrics table.
  
  - `lines/`: calibrated line plots of length and GC stratification and their CSVs (mean truth / mean pred / count for each 10-bin stratum)
  
- All images were exported as `.png` and `.svg` (dpi=400).
  


---

### `plot_residual_diagnostics.py`  

**Functionality: **  
Performs systematic diagnostics of residuals (predicted-true) for the complete test set, plotting residuals vs predicted, residuals vs true (with quantile smoothing and 95% confidence intervals), histograms of residuals, QQ-plots, and “MAE grouped by predicted quartile” histograms, and optionally generates Corresponding images in log1p space are optionally generated to check for bias, heteroskedasticity, and proximity to normality.
  

**Input: **  
- Specify manually at the top of the script:
  
  - `RUN_DIR`: full training output directory
  
  - `INPUT_FILE`: e.g. `final_test_predictions.csv`    - Other control parameters: whether to generate log1p version, smooth quantile, number of histogram bins, etc.
  
- Input CSV to be included:
  
  - Truth columns (automatically recognized from common column names, e.g. `true`/`target`/`half_life`, etc.)
  
  - Prediction columns (e.g. `pred`/`y_pred`/`prediction` etc.)

**Output:**  
- Directory: `<project root>/result/plot/fulltrain_plot/residual_diagnostics/<timestamp>/`  
- Text & Tables:
  
  - `summary.txt` & `summary.csv`: overall MAE, RMSE, R², Pearson, Spearman
  
  - `resid_vs_pred_smooth.csv`, `resid_vs_true_smooth.csv`: quantile smoothed curves of linear residuals with 95% CI data
  
  - If `DO_LOG1P=True` is turned on, additionally export the log1p residual version of the smoothed curve CSV
  
  - `mae_by_pred_decile.csv`: MAE by predicted value quartile (decile)
  
- images (all PNG+SVG, dpi=400):
  
  - `residual_vs_prediction` / `residual_vs_truth`    - `residual_qqplot` (QQ plot)
  
  - `mae_by_prediction_decile` (MAE vs prediction decile)
  
  - and the log1p version of the residual diagnostic plot (if enabled)
  


---

### `plot_results_overview_table.py`    
**Functional role:**  
Aggregates the performance metrics of the 5-fold cross-validation and final test set to generate a “results overview table”. Includes: MAE, RMSE, R², Pearson, Spearman, Sample Size N for each fold; Mean±Standard Deviation for the 5-fold metrics; Pooled metrics after splicing the 5-fold samples; and Final Test Set metrics. Also export the LaTeX table and a picture of the table suitable for direct inclusion in a paper or slide show.
  

**Enter:**  
- Specify manually at the top of the script:
  
  - `FOLD_FILES`: a dictionary with keys for the fold names (e.g. `“fold1”`-`“fold5”`) and values for the paths to the predictions CSVs for each fold validation set (e.g. `val_predictions_fold1.csv`, etc.)
  
  - `FINAL_TEST_FILE`: full test set prediction CSV (e.g. `final_test_predictions.csv`)
  
- Each CSV file needs to be included:
  
  - Truth columns (automatically recognized)
  
  - Predicted columns (auto-identification)

**Output: **  
- Directory: `<project root>/result/plot/overview_table/<timestamp>/`  
- Indicator Class CSV:
  
  - `cv_metrics_per_fold.csv`: each fold N, MAE, RMSE, R², Pearson, Spearman
  
  - `cv_summary.csv`: mean and standard deviation for 5-fold metrics
  
  - `cv_pooled.csv`: overall metrics after 5-fold sample splicing
  
  - `final_test_metrics.csv`: final test set metrics (if test set file is provided)
  
- Tables & Images:
  
  - `overview_table.tex`: booktabs-style LaTeX table source code
  
  - `overview_table.png` / `overview_table.svg`: overview table images rendered with matplotlib (listed as Split / N / MAE / RMSE / R² / Pearson / Spearman)
  


---

### `plot_test_error_breakdown.py`: rendered with matplotlib.  

**Functional role:**  
Provides a comprehensive analysis of the error and reliability of the final test set, generating multiple key charts for the paper: including MAE/RMSE bar charts binned by true value quartile, 10-bin calibration curves with 95% confidence intervals, overall Parity plots (supporting hexbin or scatter with R², correlation coefficients, slope intercepts, and MAE annotations), Bland -Altman plots, as well as plots analyzing error as a function of sequence length and GC content, and “Combined MAE & RMSE Bar Plots” to help systematically assess model performance and bias patterns at different scales.
  

**Input: **  
- Specify manually in the `CONFIG` dictionary at the top of the script:
  
  - `“RUN_DIR”`: the full training output directory containing `final_test_predictions.csv`
  
  - `“DATA_CSV”` (optional): a master data table containing all samples and information on `sequence`, `Isoform Half-Life`, etc., to complement the length/GC features
  
  - Number of bins, number of bootstrap times, Paritiy plot type (hexbin or scatter), image DPI/size, etc.
  
- To be included in `RUN_DIR/final_test_predictions.csv`:
  
  - True value columns (automatically recognized from common column names)
  
  - Predictions columns (automatically recognized)
  
  - Optional `sequence` column (used to calculate length and GC when present)

**Output:**  
- Directory: `<project root>/result/plot/test_error_breakdown/<timestamp>/`  
- Primary CSV:
  
  - `calibration_deciles.csv`: calibration data by true quantile bins (true mean, pred mean, number of samples, 95% CI of pred mean within bin)
  
  - Several intermediate tables for recording statistics such as MAE, RMSE, and sample size after binning by true value/characteristics.
  
- Main images (all PNG+SVG, dpi=400):
  
  1. `binned_metrics_MAE_by_true_bins` / `binned_metrics_RMSE_by_true_bins`: MAE/RMSE bar charts binned by true value quartiles.
  
  2. `error_bins_bar`: bi-axial bar plot of MAE and RMSE combined (with the number of samples in each bin)
  
  3. `calibration_deciles_ci`: 10-bin calibration curves, bootstrap confidence intervals with pred means
  
  4. `parity_hexbin` or `parity_density`: overall Parity plot (with R², Pearson, Spearman, slope, intercept, MAE)
  
  5. `bland_altman_test`: Bland-Altman Variance-Mean plot
  
  6. `error_vs_length`, `error_vs_gc`, etc. plots of error vs. length/GC (if sequence features are available)
  