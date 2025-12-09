# -*- coding: utf-8 -*-
"""
plot_final_training.py

用途：
  从“完整训练（final training）”阶段的输出文件中绘制论文所需图表，并统一保存为 PNG 与 SVG（dpi=400）。

输入文件（请在下方 INPUT 手动指定所在目录 EXP_DIR）：
  - training_curve_final.csv                # 逐 epoch 的 train/val 指标
  - learning_rate_schedule_final.csv        # 逐 epoch 的学习率
  - val_predictions_final.csv               # 最优模型在验证集上的预测（若主程序成功保存）
  - final_test_predictions.csv              # 最优模型在测试集上的预测（含 sequence/true/pred）
  - final_test_metrics.json                 # 测试集汇总指标（若存在）

输出目录结构（自动创建）：
  <项目根>/result/plot/finaltrain_plot/<时间戳>/*.png|*.svg
  同时额外导出部分中间统计 csv（如分箱校准、误差-序列长度分析）。

注意：
  1) 不使用命令行传参；请直接在 INPUT 中手动填写 EXP_DIR。
  2) train 与 val 的 loss 曲线分别单图绘制（满足你之前的要求）。
  3) 校准图默认 10 个等分箱，可在 CONFIG 中调整。
"""

import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============== 用户需手动指定 ==============
INPUT = {
    # >>> 请将此处改为你“最终训练阶段”那一轮实验输出目录（包含 training_curve_final.csv 等）
    # 例如：r"D:\project\runs_transformer_accumulation\test_withsavedata_20251010_01\tensorboard-log\..\.."  # 示例
    "EXP_DIR": r"F:\mRNA_Project\3UTR\Paper\result\singlemain_v100_20251015_01"
}
# ==========================================


# ============== 通用配置 ==============
CONFIG = {
    "dpi": 400,
    "deciles": 20,             # 校准图与长度分箱默认分成 10 份
    "scatter_alpha": 0.6,
    "figsize": (6, 4.5),
    "bins_hist": 30            # 残差直方图柱数
}
# =====================================


def _project_root_from_script() -> str:
    """脚本所在目录的上一级作为项目根。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def _ensure_outdir(subname: str = "finaltrain_plot") -> str:
    """
    在 <项目根>/result/plot/ 下创建 finaltrain_plot/<时间戳> 目录。
    返回该时间戳目录路径。
    """
    project_root = _project_root_from_script()
    base = os.path.join(project_root, "result", "plot", subname)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _save_figure(fig: plt.Figure, outdir: str, name: str, dpi: int):
    """同时保存 PNG 和 SVG（dpi=400）。"""
    png_path = os.path.join(outdir, f"{name}.png")
    svg_path = os.path.join(outdir, f"{name}.svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[警告] 文件不存在，跳过：{path}")
    return None


def _safe_read_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[警告] 文件不存在，跳过：{path}")
    return None


def plot_losses_separate(curve: pd.DataFrame, outdir: str, dpi: int):
    """分别绘制 train_loss 和 val_loss（单图单曲线）。"""
    if not {"epoch", "train_loss", "val_loss"}.issubset(curve.columns):
        print("[跳过] training_curve_final.csv 缺少必需列：epoch/train_loss/val_loss")
        return

    # Train loss（单图）
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(curve["epoch"], curve["train_loss"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title("Final Training — Train Loss per Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_train_loss_per_epoch", dpi)

    # Val loss（单图）
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(curve["epoch"], curve["val_loss"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_title("Final Training — Validation Loss per Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_val_loss_per_epoch", dpi)


def plot_val_metrics(curve: pd.DataFrame, outdir: str, dpi: int):
    """绘制验证集各类指标（R2、MSE、相关系数）。"""
    # R2
    if {"epoch", "val_r2"}.issubset(curve.columns):
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        ax.plot(curve["epoch"], curve["val_r2"], marker="o", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation R²")
        ax.set_title("Final Training — Validation R² per Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save_figure(fig, outdir, "final_val_r2_per_epoch", dpi)

    # MSE
    if {"epoch", "val_mse"}.issubset(curve.columns):
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        ax.plot(curve["epoch"], curve["val_mse"], marker="o", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.set_title("Final Training — Validation MSE per Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save_figure(fig, outdir, "final_val_mse_per_epoch", dpi)

    # Pearson / Spearman
    has_pearson = {"epoch", "val_pearson"}.issubset(curve.columns)
    has_spearman = {"epoch", "val_spearman"}.issubset(curve.columns)
    if has_pearson or has_spearman:
        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        if has_pearson:
            ax.plot(curve["epoch"], curve["val_pearson"], marker="o", linewidth=1.5, alpha=0.9, label="Pearson")
        if has_spearman:
            ax.plot(curve["epoch"], curve["val_spearman"], marker="s", linewidth=1.5, alpha=0.9, label="Spearman")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Correlation")
        ax.set_title("Final Training — Validation Correlations per Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        _save_figure(fig, outdir, "final_val_correlations_per_epoch", dpi)


def plot_lr_schedule(lr_df: pd.DataFrame, outdir: str, dpi: int):
    if not {"epoch", "lr"}.issubset(lr_df.columns):
        print("[跳过] learning_rate_schedule_final.csv 缺少列 epoch/lr")
        return
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(lr_df["epoch"], lr_df["lr"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Final Training — Learning Rate Schedule")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_learning_rate_schedule", dpi)


def _scatter_parity(true_y: np.ndarray, pred_y: np.ndarray, title: str, outpath_prefix: str, dpi: int):
    """通用：真实 vs 预测 散点 + y=x 参考线 + 基本统计。"""
    # 统计
    resid = pred_y - true_y
    mae = np.mean(np.abs(resid))
    rmse = math.sqrt(np.mean(resid**2))
    r2 = 1.0 - np.sum((true_y - pred_y) ** 2) / np.sum((true_y - np.mean(true_y)) ** 2)

    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.scatter(true_y, pred_y, s=12, alpha=CONFIG["scatter_alpha"])
    lo = min(np.min(true_y), np.min(pred_y))
    hi = max(np.max(true_y), np.max(pred_y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)  # y=x
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    # 角落标注
    ax.text(0.04, 0.96, f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, os.path.dirname(outpath_prefix), os.path.basename(outpath_prefix), dpi)


def plot_val_parity(val_pred_csv: str, outdir: str, dpi: int):
    df = _safe_read_csv(val_pred_csv)
    if df is None:
        return
    # 兼容含/不含 sequence 的两种情况
    needed = {"true", "pred"}
    if not needed.issubset(df.columns):
        print(f"[跳过] {val_pred_csv} 不包含 true/pred 列")
        return
    _scatter_parity(
        df["true"].to_numpy(dtype=float),
        df["pred"].to_numpy(dtype=float),
        "Final Training — Validation Parity (True vs Predicted)",
        os.path.join(outdir, "final_val_parity"),
        dpi
    )


def plot_test_parity_and_residuals(test_pred_csv: str, outdir: str, dpi: int):
    df = _safe_read_csv(test_pred_csv)
    if df is None:
        return
    if not {"true", "pred"}.issubset(df.columns):
        print(f"[跳过] {test_pred_csv} 不包含 true/pred 列")
        return

    # 1) Parity
    _scatter_parity(
        df["true"].to_numpy(dtype=float),
        df["pred"].to_numpy(dtype=float),
        "Final Training — Test Parity (True vs Predicted)",
        os.path.join(outdir, "final_test_parity"),
        dpi
    )

    # 2) 残差直方图（与主程序 residual 定义一致：true - pred）
    df["residual"] = df["true"].astype(float) - df["pred"].astype(float)
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.hist(df["residual"].to_numpy(), bins=CONFIG["bins_hist"])
    ax.set_xlabel("Residual (True - Pred)")
    ax.set_ylabel("Count")
    ax.set_title("Final Training — Test Residuals")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_test_residual_hist", dpi)

    # 3) 误差-序列长度关系（若有 sequence）
    if "sequence" in df.columns:
        df["seq_len"] = df["sequence"].astype(str).map(len)
        df["abs_error"] = np.abs(df["residual"])
        # 用等分位分箱（默认 10）
        try:
            df["len_bin"] = pd.qcut(df["seq_len"], q=CONFIG["deciles"], duplicates="drop")
        except ValueError:
            # 样本过少或长度重复导致无法 qcut，则退回等宽分箱
            df["len_bin"] = pd.cut(df["seq_len"], bins=CONFIG["deciles"])
        grp = df.groupby("len_bin", observed=True).agg(
            mean_len=("seq_len", "mean"),
            mean_abs_err=("abs_error", "mean"),
            count=("abs_error", "size")
        ).reset_index(drop=True)
        # 保存表
        grp.to_csv(os.path.join(outdir, "final_test_error_vs_length.csv"), index=False)

        fig, ax = plt.subplots(figsize=CONFIG["figsize"])
        ax.plot(grp["mean_len"], grp["mean_abs_err"], marker="o", linewidth=1.5)
        for x, y, n in zip(grp["mean_len"], grp["mean_abs_err"], grp["count"]):
            ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Sequence Length (bin mean)")
        ax.set_ylabel("Mean |Error|")
        ax.set_title("Final Training — Test Error vs Sequence Length")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save_figure(fig, outdir, "final_test_error_vs_length", dpi)

    # 4) 测试集真实值分布
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.hist(df["true"].to_numpy(dtype=float), bins=CONFIG["bins_hist"])
    ax.set_xlabel("True Half-life")
    ax.set_ylabel("Count")
    ax.set_title("Final Training — Test True Distribution")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_test_true_distribution", dpi)


def plot_test_calibration(test_pred_csv: str, outdir: str, dpi: int):
    """基于测试集 10 等分箱的校准曲线：x=分箱平均预测，y=分箱平均真实，参考线 y=x。"""
    df = _safe_read_csv(test_pred_csv)
    if df is None:
        return
    if not {"true", "pred"}.issubset(df.columns):
        print(f"[跳过] {test_pred_csv} 不包含 true/pred 列")
        return

    # 10 等分箱（可在 CONFIG 中调整）
    try:
        df["bin"] = pd.qcut(df["pred"].astype(float), q=CONFIG["deciles"], labels=False, duplicates="drop")
    except ValueError:
        print("[提示] 样本过少或预测重复值集中，改用等宽分箱。")
        df["bin"] = pd.cut(df["pred"].astype(float), bins=CONFIG["deciles"], labels=False, include_lowest=True)

    calib = df.groupby("bin", observed=True).agg(
        pred_mean=("pred", "mean"),
        true_mean=("true", "mean"),
        count=("true", "size"),
        mae=("pred", lambda x: np.mean(np.abs(x.to_numpy() - df.loc[x.index, "true"].to_numpy())))
    ).reset_index(drop=True)

    # 防止零除
    true_vals = df["true"].astype(float).to_numpy()
    pred_vals = df["pred"].astype(float).to_numpy()
    eps = 1e-12
    calib["mape"] = df.groupby("bin", observed=True).apply(
        lambda g: float(np.mean(np.abs((g["pred"] - g["true"]) / (g["true"] + eps))))
    ).reset_index(drop=True)

    calib.to_csv(os.path.join(outdir, "final_test_calibration_deciles.csv"), index=False)

    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    ax.plot(calib["pred_mean"], calib["true_mean"], marker="o", linewidth=1.5)
    lo = float(min(calib["pred_mean"].min(), calib["true_mean"].min()))
    hi = float(max(calib["pred_mean"].max(), calib["true_mean"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)  # y=x
    for x, y, n in zip(calib["pred_mean"], calib["true_mean"], calib["count"]):
        ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Bin Mean Prediction")
    ax.set_ylabel("Bin Mean True")
    ax.set_title(f"Final Training — Test Calibration ({CONFIG['deciles']} bins)")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, outdir, "final_test_calibration_deciles", dpi)


def main():
    exp_dir = os.path.normpath(INPUT["EXP_DIR"])
    if not os.path.isdir(exp_dir):
        raise NotADirectoryError(f"EXP_DIR 不存在或不是文件夹：{exp_dir}")

    outdir = _ensure_outdir("finaltrain_plot")
    print(f"[输出目录] {outdir}")

    # -------- 读取文件路径 --------
    curve_csv = os.path.join(exp_dir, "training_curve_final.csv")
    lr_csv    = os.path.join(exp_dir, "learning_rate_schedule_final.csv")
    val_csv   = os.path.join(exp_dir, "val_predictions_final.csv")
    test_csv  = os.path.join(exp_dir, "final_test_predictions.csv")
    test_json = os.path.join(exp_dir, "final_test_metrics.json")  # 可选

    # -------- 训练/验证曲线 --------
    curve_df = _safe_read_csv(curve_csv)
    if curve_df is not None and "epoch" in curve_df.columns:
        plot_losses_separate(curve_df, outdir, CONFIG["dpi"])
        plot_val_metrics(curve_df, outdir, CONFIG["dpi"])
    else:
        print("[跳过] 无法绘制 loss/val 指标曲线（缺失或无 epoch 列）")

    # -------- 学习率曲线 --------
    lr_df = _safe_read_csv(lr_csv)
    if lr_df is not None:
        plot_lr_schedule(lr_df, outdir, CONFIG["dpi"])

    # -------- 验证集 Parity（若有）--------
    if os.path.exists(val_csv):
        plot_val_parity(val_csv, outdir, CONFIG["dpi"])

    # -------- 测试集 Parity/残差/分布/长度误差 --------
    if os.path.exists(test_csv):
        plot_test_parity_and_residuals(test_csv, outdir, CONFIG["dpi"])
        plot_test_calibration(test_csv, outdir, CONFIG["dpi"])
    else:
        print("[跳过] 未找到 final_test_predictions.csv，无法绘制测试集相关图。")

    # -------- 记录测试指标 JSON（若存在）--------
    metrics = _safe_read_json(test_json)
    if metrics:
        # 生成一个简单的指标条形图（便于论文中展示对比）
        keys = ["test_r2", "test_pearson", "test_spearman"]
        present = [k for k in keys if k in metrics]
        if present:
            fig, ax = plt.subplots(figsize=CONFIG["figsize"])
            ax.bar(present, [metrics[k] for k in present])
            ax.set_title("Final Training — Test Summary Metrics")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            _save_figure(fig, outdir, "final_test_summary_metrics_bar", CONFIG["dpi"])

    print("[完成] 完整训练阶段图表已输出。")


if __name__ == "__main__":
    main()
