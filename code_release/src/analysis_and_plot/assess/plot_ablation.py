# plots/plot_ablation.py
# -*- coding: utf-8 -*-
"""
Ablation（full/UTR_only/polyA_only/masked_polyA/shuffled）对比绘图
输入：在代码顶部 CONFIG 指定 ablation 结果目录（包含 ablation_results.csv / test_predictions_<mode>.csv）
输出：<项目根>/result/plot/ablation_plot/<时间戳>/ 下的 .png 与 .svg（dpi=400）

图表：
  1) ablation_metrics_bar_[R2|Pearson|Spearman|MSE]              # 各模式测试指标柱状图
  2) delta_vs_full_[R2|Pearson|Spearman|MSE]_with_CI             # 相对 full 的Δ指标（配对bootstrap 95%CI）
  3) calibration_<mode>                                         # 各模式测试集校准曲线（可选开关）
  4) runtime_seconds_bar                                         # 各模式运行时长对比（若 JSON/CSV 有）

说明：
  - 读取：<ABLATION_DIR>/ablation_results.csv（若缺失，按 test_predictions_<mode>.csv 现算）
  - 读取：<ABLATION_DIR>/test_predictions_<mode>.csv（含 sequence/true/pred）→ 用于配对bootstrap
  - 模式列表自动由文件扫描获得；若存在 full 模式，Δ与CI均以 full 为基准
  - 仅用 matplotlib，不用 seaborn
"""

import os, re, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ========== 手动填写 ==========
CONFIG = {
    # 指向一次 ablation 实验输出目录（包含 ablation_results.csv、test_predictions_<mode>.csv 等）
    # 例如：r"F:\mRNA_Project\3UTR\Paper\result\ablation_result\20251016_153012"
    "ABLATION_DIR": r"F:\mRNA_Project\3UTR\Paper\result\ablation_result\20251014_151251",

    # 输出子目录名（位于脚本上一级 <项目根>/result/plot/<SAVE_SUBDIR>/<时间戳>）
    "SAVE_SUBDIR": "ablation_plot",

    # 置信区间：bootstrap 次数 / 置信度
    "BOOTSTRAP_N": 2000,
    "BOOTSTRAP_SEED": 20251016,
    "CI_ALPHA": 0.05,  # 95% CI

    # 校准图参数（可选）
    "PLOT_CALIB_PER_MODE": True,
    "CALIB_BINS": 10,

    # 作图风格
    "DPI": 400,
    "FIGSIZE": (6.0, 4.5),
    "GRID_ALPHA": 0.35,
}
# ============================

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent  # 脚本上一级为项目根

def _ensure_outdir(subname: str) -> str:
    base = _project_root() / "result" / "plot" / subname
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = base / ts
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)

def _save_dual(fig, out_base: str, dpi: int):
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _auto_modes(ablation_dir: str) -> List[str]:
    modes = set()
    pat = re.compile(r"test_predictions_(.+)\.csv$")
    for fn in os.listdir(ablation_dir):
        m = pat.match(fn)
        if m:
            modes.add(m.group(1))
    return sorted(modes)

def _safe_read_csv(fp: str) -> Optional[pd.DataFrame]:
    if os.path.exists(fp):
        return pd.read_csv(fp)
    print(f"[提示] 文件不存在：{fp}")
    return None

def _calc_metrics_from_preds(df: pd.DataFrame) -> Dict[str, float]:
    y = df["true"].astype(float).to_numpy()
    yhat = df["pred"].astype(float).to_numpy()
    # R2
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else np.nan
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    # Pearson / Spearman
    def _corr(a, b, method="pearson"):
        a = a.astype(float); b = b.astype(float)
        if method == "pearson":
            return float(np.corrcoef(a, b)[0,1]) if len(a) > 1 else np.nan
        else:
            # 纯 python 版 spearman：秩相关
            ra = pd.Series(a).rank(method="average").to_numpy()
            rb = pd.Series(b).rank(method="average").to_numpy()
            return float(np.corrcoef(ra, rb)[0,1]) if len(a) > 1 else np.nan
    pearson = _corr(y, yhat, "pearson")
    spearman = _corr(y, yhat, "spearman")
    mse = float(np.mean((y - yhat) ** 2))
    return {"R2": r2, "Pearson": pearson, "Spearman": spearman, "MSE": mse}

def _load_or_build_results(ablation_dir: str, modes: List[str]) -> pd.DataFrame:
    csv_fp = os.path.join(ablation_dir, "ablation_results.csv")
    df = _safe_read_csv(csv_fp)
    if df is not None:
        # 有的实现会把模式作为 index，有的在列里；统一成 index=mode
        if "mode" in df.columns:
            df = df.set_index("mode")
        # 去掉 _summary 行
        if "_summary" in df.index:
            df = df.drop(index="_summary")
        return df
    # 若没有 summary CSV，就现算
    rows = []
    for m in modes:
        tp = os.path.join(ablation_dir, f"test_predictions_{m}.csv")
        d = _safe_read_csv(tp)
        if d is None or not {"true","pred"}.issubset(d.columns):
            continue
        met = _calc_metrics_from_preds(d)
        met["mode"] = m
        # 尝试从 per-mode json 里读取 Loss/runtime
        jf = os.path.join(ablation_dir, f"test_metrics_{m}.json")
        if os.path.exists(jf):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    jj = json.load(f)
                    met["Loss"] = jj.get("Loss", np.nan)
            except Exception:
                pass
        rows.append(met)
    if not rows:
        raise FileNotFoundError("既没有 ablation_results.csv，也没有可用的 test_predictions_<mode>.csv")
    dfb = pd.DataFrame(rows).set_index("mode")
    return dfb

def _paired_bootstrap_delta(full_df: pd.DataFrame, comp_df: pd.DataFrame, n_boot: int, rng: np.random.RandomState, alpha: float):
    """
    基于 sequence 进行配对 bootstrap。
    指标：R2 / Pearson / Spearman / MSE
    返回：dict -> {"R2": (delta, lo, hi), ...}，delta=comp - full
    """
    # 内连接对齐
    j = pd.merge(full_df[["sequence","true","pred"]].copy(),
                 comp_df[["sequence","true","pred"]].copy(),
                 on="sequence", suffixes=("_full","_comp"))
    j = j.dropna()
    if len(j) < 5:
        return None
    idx = np.arange(len(j))
    def _metrics(y, yhat):
        # 与 _calc_metrics_from_preds 同定义，避免漂移
        sse = np.sum((y - yhat) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else np.nan
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        pearson = np.corrcoef(y, yhat)[0,1] if len(y) > 1 else np.nan
        # spearman
        ra = pd.Series(y).rank(method="average").to_numpy()
        rb = pd.Series(yhat).rank(method="average").to_numpy()
        spearman = np.corrcoef(ra, rb)[0,1] if len(ra) > 1 else np.nan
        mse = np.mean((y - yhat) ** 2)
        return r2, pearson, spearman, mse

    deltas = {"R2": [], "Pearson": [], "Spearman": [], "MSE": []}
    for _ in range(n_boot):
        samp = rng.choice(idx, size=len(idx), replace=True)
        y_full = j["true_full"].to_numpy(dtype=float)[samp]
        p_full = j["pred_full"].to_numpy(dtype=float)[samp]
        y_comp = j["true_comp"].to_numpy(dtype=float)[samp]
        p_comp = j["pred_comp"].to_numpy(dtype=float)[samp]
        r2_f, pe_f, sp_f, mse_f = _metrics(y_full, p_full)
        r2_c, pe_c, sp_c, mse_c = _metrics(y_comp, p_comp)
        deltas["R2"].append(r2_c - r2_f)
        deltas["Pearson"].append(pe_c - pe_f)
        deltas["Spearman"].append(sp_c - sp_f)
        deltas["MSE"].append(mse_c - mse_f)

    out = {}
    for k, arr in deltas.items():
        arr = np.array(arr, dtype=float)
        lo = float(np.quantile(arr, alpha/2))
        hi = float(np.quantile(arr, 1 - alpha/2))
        out[k] = (float(np.mean(arr)), lo, hi)
    return out

def _bar_metrics(df: pd.DataFrame, outdir: str):
    metrics = ["R2","Pearson","Spearman","MSE"]
    for met in metrics:
        vals = df[met].dropna()
        if vals.empty:
            continue
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
        ax.bar(vals.index.tolist(), vals.values)
        ax.set_xlabel("Mode")
        ax.set_ylabel(met)
        ax.set_title(f"Ablation — Test {met} by Mode")
        ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
        plt.xticks(rotation=20)
        _save_dual(fig, os.path.join(outdir, f"ablation_metrics_bar_{met}"), CONFIG["DPI"])

def _delta_vs_full(full_df: pd.DataFrame, mode_to_df: Dict[str, pd.DataFrame], outdir: str):
    if "full" not in mode_to_df:
        print("[提示] 未找到 full 模式，跳过 Δ 对比。")
        return
    rng = np.random.RandomState(CONFIG["BOOTSTRAP_SEED"])
    n_boot = int(CONFIG["BOOTSTRAP_N"])
    alpha = float(CONFIG["CI_ALPHA"])

    full_preds = mode_to_df["full"]
    metrics = ["R2","Pearson","Spearman","MSE"]
    for met in metrics:
        xs, ys, lo, hi = [], [], [], []
        for m, dfm in mode_to_df.items():
            if m == "full":
                continue
            res = _paired_bootstrap_delta(full_preds, dfm, n_boot, rng, alpha)
            if res is None:
                continue
            d, l, h = res[met]
            xs.append(m); ys.append(d); lo.append(ys[-1] - l); hi.append(h - ys[-1])
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
        xpos = np.arange(len(xs))
        ax.bar(xpos, ys, yerr=[lo, hi], capsize=4)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k")
        ax.set_xticks(xpos); ax.set_xticklabels(xs, rotation=20)
        ax.set_ylabel(f"Δ{met} (mode - full)")
        ax.set_title(f"Ablation — Δ{met} vs full (paired bootstrap 95% CI)")
        ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
        _save_dual(fig, os.path.join(outdir, f"delta_vs_full_{met}_with_CI"), CONFIG["DPI"])

def _calibration_curve(df: pd.DataFrame, bins: int, outpath: str, title: str):
    # 等频分箱（预测为准），失败则退等宽
    try:
        bins_s = pd.qcut(df["pred"].astype(float), q=bins, labels=False, duplicates="drop")
    except Exception:
        bins_s = pd.cut(df["pred"].astype(float), bins=bins, labels=False, include_lowest=True)
    g = df.assign(bin=bins_s).groupby("bin", observed=True).agg(
        pred_mean=("pred","mean"),
        true_mean=("true","mean"),
        count=("true","size")
    ).reset_index(drop=True)
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    ax.plot(g["pred_mean"], g["true_mean"], marker="o", linewidth=1.5)
    lo = float(min(g["pred_mean"].min(), g["true_mean"].min()))
    hi = float(max(g["pred_mean"].max(), g["true_mean"].max()))
    ax.plot([lo,hi],[lo,hi], linestyle="--", linewidth=1.2)
    for x, y, n in zip(g["pred_mean"], g["true_mean"], g["count"]):
        ax.annotate(str(int(n)), (x,y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)
    ax.set_xlabel("Bin Mean Prediction"); ax.set_ylabel("Bin Mean True")
    ax.set_title(title); ax.grid(True, linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    _save_dual(fig, outpath, CONFIG["DPI"])

def main():
    ab_dir = os.path.normpath(CONFIG["ABLATION_DIR"])
    if not os.path.isdir(ab_dir):
        raise NotADirectoryError(f"ABLATION_DIR 不存在或不是文件夹：{ab_dir}")

    outdir = _ensure_outdir(CONFIG["SAVE_SUBDIR"])
    print(f"[输出目录] {outdir}")

    # 自动发现模式
    modes = _auto_modes(ab_dir)
    if not modes:
        raise FileNotFoundError("未在 ABLATION_DIR 下发现 test_predictions_<mode>.csv")
    print(f"[发现模式] {modes}")

    # 读取 summary 或就地计算
    df_sum = _load_or_build_results(ab_dir, modes)

    # 读取各模式预测
    mode_to_df: Dict[str, pd.DataFrame] = {}
    for m in modes:
        fp = os.path.join(ab_dir, f"test_predictions_{m}.csv")
        dfm = _safe_read_csv(fp)
        if dfm is None or not {"true","pred"}.issubset(dfm.columns):
            print(f"[跳过] {m} 缺 true/pred 列：{fp}")
            continue
        # 统一列类型
        dfm = dfm.dropna(subset=["true","pred"]).copy()
        if "sequence" not in dfm.columns:
            print(f"[提示] {m} 缺 sequence 列，Δbootstrap 只能做粗略非配对（此处将跳过Δ）")
        else:
            dfm["sequence"] = dfm["sequence"].astype(str)
        mode_to_df[m] = dfm

    # 1) 各模式测试指标柱状图
    _bar_metrics(df_sum, outdir)

    # 2) 与 full 的Δ（配对bootstrap 95%CI）
    _delta_vs_full(mode_to_df.get("full"), mode_to_df, outdir)

    # 3) 各模式校准曲线（可选）
    if CONFIG.get("PLOT_CALIB_PER_MODE", True):
        for m, dfm in mode_to_df.items():
            _calibration_curve(dfm, bins=int(CONFIG["CALIB_BINS"]),
                               outpath=os.path.join(outdir, f"calibration_{m}"),
                               title=f"Calibration — {m} (bins={CONFIG['CALIB_BINS']})")

    # 4) 运行时长（如果 summary 有 runtime_sec）
    if "runtime_sec" in df_sum.columns:
        fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
        vals = df_sum["runtime_sec"].dropna()
        ax.bar(vals.index.tolist(), vals.values)
        ax.set_xlabel("Mode"); ax.set_ylabel("Runtime (s)")
        ax.set_title("Ablation — Runtime by Mode")
        ax.grid(True, axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
        plt.xticks(rotation=20)
        _save_dual(fig, os.path.join(outdir, "runtime_seconds_bar"), CONFIG["DPI"])

    print("[完成] 消融结果图已输出。")

if __name__ == "__main__":
    main()
