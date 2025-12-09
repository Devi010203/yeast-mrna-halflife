# plots/plot_cv.py
# -*- coding: utf-8 -*-
"""
五折交叉验证绘图（与主程序输出严格对齐，路径在代码中手动填写）：
- 输出位置：脚本所在目录的同级 result/plot/<SAVE_SUBDIR>/   （自动创建）
- 在 CONFIG 中手动填写 RUN_DIR（主程序该次运行的输出目录）
- 自动发现主程序真实产物：
    * RUN_DIR/cv_summary.csv
    * RUN_DIR/training_log.json
    * RUN_DIR/val_predictions_fold*.csv
    * RUN_DIR/learning_rate_schedule_fold*.csv（可选）
- 图形：
    1) R² 柱状（带均值虚线；必要时由 val_predictions_* 动态计算）
    2) R² 箱线图（按配置三种模式其一，默认 bootstrap：每折一个箱）
       - aggregate   : 把5个折的R²合成一个总体箱（单箱）
       - per_epoch   : 每折用各epoch的 val_r2 作为该折分布（5箱）
       - bootstrap   : 对每折验证集做自助抽样得到R²分布（5箱，推荐）
    3) Val R² 学习曲线（每折）
    4) Loss 学习曲线（每折）——分别输出：
       - 训练集：cv_train_loss_learning_curves.{png,svg}
       - 验证集：cv_val_loss_learning_curves.{png,svg}
       - 合并图：cv_loss_learning_curves_combined.{png,svg}
    5) 每折验证散点拼版：cv_val_scatter_folds.{png,svg}
    6) 每折独立校准曲线
    7) （可选）每折学习率曲线
- 每张图同时导出 PNG 和 SVG，dpi=400
"""
import os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


from plot_utils import (
    ensure_dir, savefig_dual,
    safe_read_csv, read_json_or_jsonl,
    scatter_true_pred, calibration_curve  # 保留原有导入（未改变其余逻辑）
)
from sklearn.metrics import r2_score  # 用于从 val_predictions_* 反算 R²

# ========= 在此处手动填写你的输入与输出配置 =========
CONFIG = {
    # 主程序某次运行的输出目录（包含 training_log.json / cv_summary.csv / val_predictions_fold*.csv 等）
    # 例如："/home/zdl4/mRNA/python/3UTR/runs_transformer_accumulation/test_withsavedata_20251007_01"
    "RUN_DIR": r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01",

    # 输出子目录名 -> result/plot/<SAVE_SUBDIR>/
    "SAVE_SUBDIR": "5foldplot-2",

    # 校准分箱数
    "CALIB_BINS": 20,

    # R² 箱线图模式： "bootstrap" | "per_epoch" | "aggregate"
    "R2_BOX_MODE": "bootstrap",

    # bootstrap 参数（仅当 R2_BOX_MODE="bootstrap" 有效）
    "BOOT_N": 1000,          # 每折自助抽样次数
    "BOOT_SEED": 20251015,   # 随机种子
    "JITTER_MAX_POINTS": 300 # 叠加抖动散点的最大点数（防止图过密）
}

plt.style.use(['science', 'no-latex'])
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial","DejaVu Sans",  "Liberation Sans", "Noto Sans CJK SC"],
    "font.style": "normal",
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    # 字号相关
    "font.size":17,
    "axes.titlesize":20,
    "axes.labelsize":19,
    "xtick.labelsize":15,
    "ytick.labelsize":15,
    "legend.fontsize":16,
    "figure.titlesize":17,
    # "axes.titleweight": "bold",
    # "axes.labelweight": "bold",
})

# ---------- 统一图形尺寸 & fold 颜色/marker ----------

# 4:3 比例的 loss 学习曲线图
FIGSIZE_LOSS = (6.4, 4.8)

# 每个散点子图的边长（英寸），保证子图 1:1
FIGSIZE_SCATTER_SUB = 4.0

# 每个 fold 对应固定颜色 & marker，保证在不同图中风格一致
# FOLD_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
FOLD_MARKERS = ["o"]
FOLD_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def _style_for_fold(fold_idx, idx_fallback=0):
    """
    根据 fold 序号返回 (color, marker)，保证在不同图中该 fold 的散点风格一致。
    当无法解析 fold_idx 时，退回到 idx_fallback。
    """
    if fold_idx is None:
        i = idx_fallback
    else:
        try:
            i = int(fold_idx)-1
        except Exception:
            i = idx_fallback
    color = FOLD_COLORS[i % len(FOLD_COLORS)]
    marker = FOLD_MARKERS[i % len(FOLD_MARKERS)]
    return color, marker

# ---------- 小工具 ----------
def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:  # 精确匹配
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:     # 忽略大小写匹配
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def _auto_files(run_dir: str):
    """根据主程序真实输出，自动搜集所需文件"""
    files = {}
    files["cv_summary"] = os.path.join(run_dir, "cv_summary.csv")
    files["training_log"] = os.path.join(run_dir, "training_log.json")
    files["val_fold_files"] = sorted(glob.glob(os.path.join(run_dir, "val_predictions_fold*.csv")))
    files["lr_fold_files"] = sorted(glob.glob(os.path.join(run_dir, "learning_rate_schedule_fold*.csv")))
    return files

def _infer_fold_idx(fp: str):
    """从文件名提取 fold 序号：val_predictions_fold{N}.csv"""
    name = os.path.basename(fp)
    for token in name.replace(".csv","").split("_"):
        if token.lower().startswith("fold"):
            try:
                return int(token.lower().replace("fold",""))
            except Exception:
                return None
    return None

def _fallback_cv_summary_from_preds(val_fold_files, save_to=None):
    """缺少 cv_summary.csv 时，从各折 val_predictions_* 反算 R²，并（可选）保存补全版 CSV"""
    rows = []
    for fp in val_fold_files:
        df = safe_read_csv(fp)
        if df.empty or not {"true","pred"}.issubset(df.columns):
            continue
        fold = _infer_fold_idx(fp)
        r2 = r2_score(df["true"].values, df["pred"].values)
        rows.append({"fold": fold, "val_r2": r2})
    if not rows:
        return pd.DataFrame()
    df_sum = pd.DataFrame(rows).sort_values("fold")
    df_sum["mean_r2"] = df_sum["val_r2"].mean()
    if save_to:
        try: df_sum.to_csv(save_to, index=False)
        except Exception: pass
    return df_sum

# ----------【紧凑坐标与校准】----------
def _quantile_limits_xy(x: np.ndarray, y: np.ndarray, qlo: float = 0.01, qhi: float = 0.99, pad_frac: float = 0.03):
    """根据 x,y 的联合分位数给出紧凑的对角可读坐标范围，并留少量边距。"""
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (0.0, 1.0), (0.0, 1.0)
    x_lo, x_hi = np.quantile(x, [qlo, qhi])
    y_lo, y_hi = np.quantile(y, [qlo, qhi])
    lo = float(min(x_lo, y_lo)); hi = float(max(x_hi, y_hi))
    span = max(1e-12, hi - lo)
    lo -= pad_frac * span; hi += pad_frac * span
    return (lo, hi), (lo, hi)

def calibration_curve_tight(df: pd.DataFrame, n_bins: int, out_basepath: str, title: str,
                            q_limits=(0.01, 0.99), pad_frac=0.03, min_per_bin: int = 10):
    """仅在有样本的预测分位区间内分箱绘制校准曲线，坐标范围紧凑，去掉右上角空白。"""
    df = df[["true","pred"]].replace([np.inf,-np.inf], np.nan).dropna()
    if df.empty:
        return
    y = df["true"].values.astype(float)
    p = df["pred"].values.astype(float)

    # 以“预测值”的分位区间作为可视范围（避免无数据区）
    qlo, qhi = q_limits
    p_finite = p[np.isfinite(p)]
    plo, phi = np.quantile(p_finite, [qlo, qhi])
    span = max(1e-12, phi - plo)
    plo -= pad_frac * span; phi += pad_frac * span

    # 等宽分箱，仅保留样本数足够的箱
    edges = np.linspace(plo, phi, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i+1]) & np.isfinite(y)
        cnt = int(np.sum(m))
        if cnt >= min_per_bin:
            xs.append(float(np.mean(p[m])))
            ys.append(float(np.mean(y[m])))
            ns.append(cnt)
    if len(xs) < 2:
        return

    xs = np.array(xs); ys = np.array(ys)

    # y 轴范围与散点一致：联合分位 + padding；x 轴用预测范围主导
    (xlim_joint, ylim_joint) = _quantile_limits_xy(y, p, qlo, qhi, pad_frac)
    xlim = (float(plo), float(phi))
    ylim = (min(ylim_joint[0], xlim[0]), max(ylim_joint[1], xlim[1]))

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "--", lw=1.0, alpha=0.6, color="k", label="ideal")
    ax.plot(xs, ys, "-o", lw=1.6, ms=3.5, alpha=0.95, label="calibration")
    for xi, yi, ni in zip(xs, ys, ns):
        ax.text(xi, yi, f"{ni}", fontsize=7, ha="center", va="bottom", alpha=0.7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.grid(True, alpha=0.35); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_dual(fig, out_basepath, dpi=400)
    plt.close(fig)

# ---------- 箱线图：三种模式 ----------
def plot_cv_box_aggregate(cv_csv: pd.DataFrame, outdir: str):
    """把 5 折 R² 合成一个总体箱（单箱）"""
    r2_col = _find_col(cv_csv, ["val_r2","r2","valR2","Val_R2"])
    if r2_col is None or cv_csv.empty: return
    vals = cv_csv[r2_col].astype(float).values
    mean_r2 = float(np.mean(vals)); median_r2 = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.boxplot([vals], vert=True, patch_artist=False, showmeans=True, meanline=True, widths=0.5)
    xj = np.random.normal(loc=1.0, scale=0.03, size=len(vals))
    ax.scatter(xj, vals, s=18, alpha=0.8)
    ax.set_xticks([1]); ax.set_xticklabels(["Val R² (folds)"])
    ax.set_ylabel("Val. R²"); ax.set_title("5-fold val. R² (aggregate)")
    ax.grid(True)
    ax.text(1.16, mean_r2, f"mean={mean_r2:.3f}", va="center", fontsize=8)
    ax.text(0.84, median_r2, f"median={median_r2:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_box_aggregate"), dpi=400)
    plt.close(fig)

def plot_cv_box_per_epoch(trainlog_items, outdir: str):
    """
    用每折的各 epoch val_r2 值作为“分布”，得到 5 个箱
    注意：各epoch相关性强，统计含义弱于bootstrap
    """
    if not trainlog_items: return
    df = pd.DataFrame(trainlog_items)
    vcol = _find_col(df, ["val_r2","valR2","Val_R2"])
    fcol = _find_col(df, ["fold","Fold"])
    if vcol is None or fcol is None or df.empty: return

    groups = []
    labels = []
    for f in sorted(df[fcol].dropna().unique()):
        sub = df[df[fcol]==f][vcol].dropna().astype(float).values
        if len(sub) >= 2:
            groups.append(sub)
            labels.append(f"fold{int(f)}")

    if not groups: return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot(groups, vert=True, patch_artist=False, showmeans=True, meanline=True)
    # 叠加抖动散点（限制点数）
    m = CONFIG.get("JITTER_MAX_POINTS", 300)
    for i, g in enumerate(groups, start=1):
        g = np.array(g)
        if len(g) > m:
            idx = np.linspace(0, len(g)-1, m, dtype=int)
            g = g[idx]
        xj = np.random.normal(loc=i, scale=0.05, size=len(g))
        ax.scatter(xj, g, s=10, alpha=0.5)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel("Val. R²"); ax.set_title("5-fold val. R² (per-epoch)")
    ax.grid(True)
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_box_per_epoch"), dpi=400)
    plt.close(fig)

def _bootstrap_r2_for_fold(df_pred: pd.DataFrame, n_boot: int, rng: np.random.RandomState):
    """对单个折的验证集进行自助抽样，返回R²列表"""
    df = df_pred[["true","pred"]].dropna()
    y = df["true"].values; yhat = df["pred"].values
    n = len(y)
    if n < 3:  # 太少无法稳定估计
        return []
    idx = np.arange(n)
    r2s = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        r2s.append(r2_score(y[samp], yhat[samp]))
    return r2s

def plot_cv_box_bootstrap(val_fold_files, outdir: str, n_boot: int, seed: int):
    """
    用 bootstrap 在每折内采样，得到每折的 R² 分布 → 5 个箱
    同时为每个折的抖动散点指定固定颜色和 marker，便于与散点图对应。
    """
    if not val_fold_files: return
    rng = np.random.RandomState(seed)
    groups = []
    labels = []
    fold_ids = []

    for fp in sorted(val_fold_files, key=lambda x: (_infer_fold_idx(x) or 9999)):
        df = safe_read_csv(fp)
        if df.empty or not {"true","pred"}.issubset(df.columns):
            continue
        fold_idx = _infer_fold_idx(fp)
        r2s = _bootstrap_r2_for_fold(df, n_boot=n_boot, rng=rng)
        if len(r2s) >= 2:
            groups.append(np.array(r2s))
            labels.append(f"Fold{fold_idx if fold_idx is not None else '?'}")
            fold_ids.append(fold_idx)

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot(groups, vert=True, patch_artist=False, showmeans=True, meanline=True)
    # 叠加抖动散点（限制点数），并使用与 scatter 图一致的颜色/marker
    m = CONFIG.get("JITTER_MAX_POINTS", 300)
    for i, (g, fold_idx) in enumerate(zip(groups, fold_ids), start=1):
        g = np.array(g)
        if len(g) > m:
            idx = np.linspace(0, len(g)-1, m, dtype=int)
            g = g[idx]
        xj = np.random.normal(loc=i, scale=0.05, size=len(g))
        color, marker = _style_for_fold(fold_idx, idx_fallback=i-1)
        ax.scatter(xj, g, s=8, alpha=0.35, color=color, marker=marker)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel("Val. R²")
    # ax.set_title(f"5-fold val. R² (bootstrap, n={n_boot})")
    ax.grid(True)
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_box_bootstrap"), dpi=400)
    plt.close(fig)

# ---------- 其他图 ----------
def plot_cv_bar(cv_csv: pd.DataFrame, outdir: str):
    """柱状图展示各折 Val R²，虚线为均值"""
    fold_col = _find_col(cv_csv, ["fold", "Fold"])
    r2_col   = _find_col(cv_csv, ["val_r2", "r2", "valR2", "Val_R2"])
    if fold_col is None or r2_col is None or cv_csv.empty:
        return
    folds = cv_csv[fold_col].astype(int).values
    vals  = cv_csv[r2_col].astype(float).values
    mean_r2 = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(x) for x in folds], vals, width=0.65)
    ax.axhline(mean_r2, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label=f"mean = {mean_r2:.3f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("Val. R²"); ax.set_title("5-fold val. R²")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "cv_r2_bar"), dpi=400)
    plt.close(fig)

def plot_cv_learning_curves(trainlog_items, outdir: str):
    """训练日志：每折的 Val R² 曲线、Loss 曲线（train/val 分开 + 合并）"""
    if not trainlog_items: return
    df = pd.DataFrame(trainlog_items)
    if df.empty: return
    epoch_col = _find_col(df, ["epoch","Epoch"])
    fold_col  = _find_col(df, ["fold","Fold"])
    vcol      = _find_col(df, ["val_r2","valR2","Val_R2"])
    tloss_col = _find_col(df, ["train_loss","Train_Loss"])
    vloss_col = _find_col(df, ["val_loss","Val_Loss"])
    if fold_col is None or epoch_col is None: return

    # Val R²
    if vcol is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[vcol], linewidth=1.5, label=f"fold {int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val. R²"); ax.set_title("Val. R² by epoch (per fold)")
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_valR2_learning_curves"), dpi=400)
        plt.close(fig)

    # Train loss 单独一张（4:3，纵轴为原始 loss，线性坐标）
    if tloss_col is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LOSS)
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[tloss_col], linewidth=1.2, label=f"fold {int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss")
        # ax.set_title("Train loss by epoch (per fold)")
        ax.set_yscale("linear")  # 明确使用真实数值（不做对数变换）
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_train_loss_learning_curves"), dpi=400)
        plt.close(fig)

    # Val loss 单独一张（4:3，纵轴为原始 loss，线性坐标）
    if vloss_col is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LOSS)
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[vloss_col], linewidth=1.4, label=f"fold {int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss")
        # ax.set_title("Val loss by epoch (per fold)")
        ax.set_yscale("linear")  # 明确使用真实数值（不做对数变换）
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_val_loss_learning_curves"), dpi=400)
        plt.close(fig)

    # 合并版：train/val 同图（保持原尺寸配置）
    if tloss_col is not None and vloss_col is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in sorted(df[fold_col].unique()):
            sub = df[df[fold_col]==f].sort_values(epoch_col)
            ax.plot(sub[epoch_col], sub[tloss_col], alpha=0.8, linewidth=1.0, label=f"train f{int(f)}")
            ax.plot(sub[epoch_col], sub[vloss_col], alpha=0.95, linewidth=1.5, label=f"val f{int(f)}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss by epoch (per fold)")
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_loss_learning_curves_combined"), dpi=400)
        plt.close(fig)

def plot_cv_scatter_and_calibration(val_fold_files, outdir: str, n_bins: int = 10):
    """
    每折验证散点（拼版） + 每折独立校准曲线（单张）
    - cv_val_scatter_folds.png：每个折的散点颜色/marker 与 cv_r2_box_bootstrap 中一致
    - 各子图 1:1 比例，统一 x/y 轴范围（按所有折的联合分位数）
    """
    fps = sorted(val_fold_files, key=lambda x: (_infer_fold_idx(x) or 9999))
    if not fps:
        return

    # 先遍历一次，得到所有折的 true/pred，用于统一坐标范围
    all_true_list = []
    all_pred_list = []
    for fp in fps:
        df = safe_read_csv(fp)
        if df.empty or not {"true", "pred"}.issubset(df.columns):
            continue
        all_true_list.append(df["true"].values)
        all_pred_list.append(df["pred"].values)
    if not all_true_list:
        return
    all_true = np.concatenate(all_true_list)
    all_pred = np.concatenate(all_pred_list)
    global_xlim, global_ylim = _quantile_limits_xy(
        all_true, all_pred,
        qlo=0.01, qhi=0.99, pad_frac=0.03
    )

    n = len(fps)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * FIGSIZE_SCATTER_SUB, rows * FIGSIZE_SCATTER_SUB),
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for i, fp in enumerate(fps):
        df = safe_read_csv(fp)
        if df.empty or not {"true", "pred"}.issubset(df.columns):
            continue
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("on")

        fold_idx = _infer_fold_idx(fp)
        color, marker = _style_for_fold(fold_idx, idx_fallback=i)

        ax.scatter(df["true"], df["pred"], s=6, alpha=0.6, color=color, marker=marker)
        ax.plot(
            [global_xlim[0], global_xlim[1]],
            [global_ylim[0], global_ylim[1]],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="k",
        )
        ax.set_xlim(*global_xlim)
        ax.set_ylim(*global_ylim)
        ax.set_aspect("equal", adjustable="box")  # 子图 1:1 比例

        ax.tick_params(labelsize=19)

        # name = os.path.basename(fp).replace(".csv", "")
        fold_idx = _infer_fold_idx(fp)
        if fold_idx is None:
            # 找不到 fold 时退回文件名
            title = os.path.basename(fp).replace(".csv", "")
        else:
            title = f"Fold {fold_idx}"
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True)

    # fig.suptitle("Val. scatter per fold", y=0.90, fontsize=21)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_dual(fig, os.path.join(outdir, "cv_val_scatter_folds"), dpi=400)
    plt.close(fig)

    # 各折独立校准曲线（tight 版）
    for fp in fps:
        df = safe_read_csv(fp)
        if df.empty or not {"true","pred"}.issubset(df.columns):
            continue
        name = os.path.splitext(os.path.basename(fp))[0]
        calibration_curve_tight(
            df, n_bins=n_bins,
            out_basepath=os.path.join(outdir, f"{name}_calibration"),
            title=f"Calibration: {name}",
            q_limits=(0.01, 0.99), pad_frac=0.03, min_per_bin=10
        )

# ---------- 主流程 ----------
def main():
    # 输出根目录：脚本目录的同级 result/plot/<SAVE_SUBDIR>/
    script_dir = Path(__file__).resolve().parent
    save_subdir = CONFIG.get("SAVE_SUBDIR", "5foldplot")
    outdir = ensure_dir(script_dir.parent / "result" / "plot" / save_subdir)

    run_dir = CONFIG.get("RUN_DIR", "").strip()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("请在 CONFIG['RUN_DIR'] 中填写主程序该次运行的输出目录路径（包含 cv 与各折CSV）。")

    files = _auto_files(run_dir)

    # --- 柱状图 + （单箱）聚合箱线图 ---
    cv_csv = safe_read_csv(files["cv_summary"])
    if cv_csv.empty:
        cv_csv = _fallback_cv_summary_from_preds(files["val_fold_files"],
                                                 save_to=os.path.join(outdir, "cv_summary_from_preds.csv"))
    if not cv_csv.empty:
        plot_cv_bar(cv_csv, outdir)
        plot_cv_box_aggregate(cv_csv, outdir)

    # --- 学习曲线（含：Val R²；Train loss；Val loss；合并版） ---
    trainlog_items = read_json_or_jsonl(files["training_log"])
    if trainlog_items:
        plot_cv_learning_curves(trainlog_items, outdir)

    # --- 每折验证散点 & 校准 ---
    if files["val_fold_files"]:
        plot_cv_scatter_and_calibration(files["val_fold_files"], outdir, n_bins=int(CONFIG.get("CALIB_BINS", 10)))

    # --- R² 5箱：按配置选择最佳可用模式 ---
    mode = (CONFIG.get("R2_BOX_MODE") or "bootstrap").lower()
    if mode == "bootstrap" and files["val_fold_files"]:
        plot_cv_box_bootstrap(
            files["val_fold_files"], outdir,
            n_boot=int(CONFIG.get("BOOT_N", 1000)),
            seed=int(CONFIG.get("BOOT_SEED", 20251015))
        )
    elif mode == "per_epoch" and trainlog_items:
        plot_cv_box_per_epoch(trainlog_items, outdir)
    else:
        # 若所选模式数据不足，自动尝试另一个可用模式
        if files["val_fold_files"]:
            plot_cv_box_bootstrap(
                files["val_fold_files"], outdir,
                n_boot=int(CONFIG.get("BOOT_N", 1000)),
                seed=int(CONFIG.get("BOOT_SEED", 20251015))
            )
        elif trainlog_items:
            plot_cv_box_per_epoch(trainlog_items, outdir)
        # 若都不可用，则已有 aggregate 版本可作为备选

    # --- （可选）每折学习率曲线 ---
    if files["lr_fold_files"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for fp in files["lr_fold_files"]:
            df = safe_read_csv(fp)
            if df.empty or "epoch" not in df.columns or "lr" not in df.columns:
                continue
            name = os.path.basename(fp).replace(".csv","")
            ax.plot(df["epoch"], df["lr"], linewidth=1.3, label=name.split("_")[-1])
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning rate per fold")
        ax.grid(True); ax.legend(ncols=3, fontsize=8)
        fig.tight_layout()
        savefig_dual(fig, os.path.join(outdir, "cv_lr_schedules"), dpi=400)
        plt.close(fig)

    print(f"[OK] CV 图已生成：{outdir}")

if __name__ == "__main__":
    main()
