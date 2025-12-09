# plots/plot_parity_stratified.py
# -*- coding: utf-8 -*-
"""
分层 Parity（按 3'UTR 长度 & GC 分位）：
  - 每层 parity：y=x 等式线 + 线性拟合线 + 指标注释（N/MAE/RMSE/R²/slope/intercept）
  - 面板图：长度 Q1–Q4、GC Q1–Q4（各一页 2×2）
  - 分层 10-bin 校准折线（E[pred] vs E[true]）
  - 输出分层指标 CSV & 折线 CSV
输出：result/plot/parity_stratified/<时间戳>/
图像：PNG+SVG，dpi=400；坐标文字非斜体
"""

import re, math, json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
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
    # "axes.titleweight": "bold",  # 图标题
    # "axes.labelweight": "bold",  # x / y 轴标签
})

# ========= 在此手动填写（不使用命令行）=========
RUN_DIR    = r"F:\mRNA_Project\3UTR\Paper\result\3utr_mrna_11.12\5f_full_head_v3_20251112_01"   # ← 改成你的完整训练输出目录
INPUT_FILE = "final_test_predictions.csv"       # ← 如有不同文件名请修改

DPI = 400
USE_HEXBIN = True    # True: hexbin 密度; False: 散点
HEX_GRIDSIZE = 60
POINT_SIZE = 6
POINT_ALPHA = 0.25

N_QUANT = 4          # 分层分位数（默认 4 分位=Q1..Q4）
CALIB_BINS = 10      # 每层校准折线的分箱数
# ==============================================

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_outdirs():
    root = _project_root() / "result" / "plot" / "parity_stratified" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (root / "length").mkdir(parents=True, exist_ok=True)
    (root / "gc").mkdir(parents=True, exist_ok=True)
    (root / "lines").mkdir(parents=True, exist_ok=True)
    return root

def _save_dual(fig, out_base: Path):
    fig.savefig(str(out_base) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def _auto_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cand_y_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_y_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    yt = yp = None
    for c in df.columns:
        if c.lower() in cand_y_true: yt = c; break
    for c in df.columns:
        if c.lower() in cand_y_pred: yp = c; break
    if yt is None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["true","target","label","ground","half"]):
                yt = c; break
    if yp is None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["pred","hat","predict"]):
                yp = c; break
    if yt is None or yp is None:
        raise ValueError(f"无法识别真值/预测列：{list(df.columns)}")
    return yt, yp

def _seq_len_gc(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    lens, gcs = [], []
    for s in series.astype(str):
        s2 = re.sub(r"[^ACGTNacgtn]", "", s)
        L  = len(s2)
        lens.append(L)
        if L == 0:
            gcs.append(np.nan)
        else:
            su = s2.upper()
            gc = su.count("G") + su.count("C")
            at = su.count("A") + su.count("T")
            tot = gc + at
            gcs.append(gc / tot if tot > 0 else np.nan)
    return np.array(lens, int), np.array(gcs, float)

def _ensure_len_gc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sequence" in out.columns:
        L, G = _seq_len_gc(out["sequence"])
        out["utr_len"] = L
        out["gc_frac"] = G
        return out
    # fallback: 使用已有 length/gc 列
    len_col = None; gc_col = None
    for c in out.columns:
        cl = c.lower()
        if len_col is None and ("len" in cl or "length" in cl):
            len_col = c
        if gc_col is None and (cl in ("gc","gc_content","gc_fraction") or ("gc" in cl and "frac" in cl)):
            gc_col = c
    if len_col is None or gc_col is None:
        raise ValueError("缺少 sequence 或 length/gc 列，无法分层。")
    out["utr_len"] = out[len_col].astype(int).values
    out["gc_frac"] = out[gc_col].astype(float).values
    return out

def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """最小二乘 y = a + b x；返回 a, b, R²"""
    x = x.astype(float); y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)   # 注意 np.polyfit 返回 [slope, intercept]
    yhat = a + b*x
    ssr = np.sum((y - yhat)**2)
    sst = np.sum((y - y.mean())**2)
    r2 = float(1 - ssr/sst) if sst > 0 else np.nan
    return float(a), float(b), r2

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {"MAE": mae, "RMSE": rmse}

def _labels_from_edges(edges: np.ndarray) -> List[str]:
    labs = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        if edges.max() <= 1.5:  # GC
            labs.append(f"Q{i+1}: [{lo:.2f},{hi:.2f}]")
        else:                   # length
            labs.append(f"Q{i+1}: [{int(lo)},{int(hi)}]")
    return labs

def _quantile_edges(x: np.ndarray, q: int) -> np.ndarray:
    qs = np.linspace(0,1,q+1)
    e = np.quantile(x, qs)
    # 去重处理（避免大量重复值时边界相同导致空层）
    for i in range(1, len(e)):
        if e[i] <= e[i-1]:
            e[i] = e[i-1] + 1e-9
    e[0] = np.nanmin(x); e[-1] = np.nanmax(x)
    return e

def _plot_panel(y_true: np.ndarray, y_pred: np.ndarray, group_idx: np.ndarray, labels: List[str],
                title: str, out_base: Path, limits: Tuple[float,float]):
    """2×2 面板（若 N_QUANT!=4，则自动生成行×列近似正方形网格）"""
    k = len(labels)
    # 计算网格行列
    nrow = int(math.floor(math.sqrt(k)))
    ncol = int(math.ceil(k / max(1, nrow)))
    if nrow * ncol < k:
        nrow = int(math.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6*ncol, 3.2*nrow), squeeze=False)
    x_min, x_max = limits
    y_min, y_max = limits

    records = []  # 用于 CSV 记录

    for i in range(k):
        r = i // ncol; c = i % ncol
        ax = axes[r][c]
        sel = (group_idx == i)
        xt = y_true[sel]; yp = y_pred[sel]
        a, b, r2 = _fit_line(xt, yp)
        met = _metrics(xt, yp)
        # 绘图
        if USE_HEXBIN:
            hb = ax.hexbin(xt, yp, gridsize=HEX_GRIDSIZE, cmap="viridis", mincnt=1, extent=(x_min, x_max, y_min, y_max))
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Count")
        else:
            ax.scatter(xt, yp, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0)
        # y=x
        ax.plot([x_min, x_max], [x_min, x_max], linestyle="--", color="k", linewidth=1.0, label="y = x")
        # 拟合线
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([x_min, x_max], [a + b*x_min, a + b*x_max], color="C1", linewidth=1.8, label=f"fit: y={a:.2g}+{b:.2g}x")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(labels[i])
        ax.grid(True, linestyle="--", alpha=0.35)
        if r == nrow-1: ax.set_xlabel("Truth")
        if c == 0:      ax.set_ylabel("Prediction")
        # 注释
        txt = f"N={xt.size}\nMAE={met['MAE']:.3g}\nRMSE={met['RMSE']:.3g}\nR²={r2:.3f}"
        if np.isfinite(b): txt += f"\nslope={b:.3g}"
        if np.isfinite(a): txt += f"\ninterc={a:.3g}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8))
        # 记录
        rec = {"group": labels[i], "N": int(xt.size), "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": r2, "slope": b, "intercept": a}
        records.append(rec)

    # 清空多余子图
    for j in range(k, nrow*ncol):
        r = j // ncol; c = j % ncol
        axes[r][c].axis("off")

    fig.suptitle(title, y=0.94, fontsize=20)
    fig.tight_layout()
    _save_dual(fig, out_base)
    return records

def _calib_lines(y_true: np.ndarray, y_pred: np.ndarray, group_idx: np.ndarray, labels: List[str],
                 out_base: Path):
    """每层 10-bin 校准折线（E[pred] vs E[true]），并导出 CSV"""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    all_rows = []
    for i, lab in enumerate(labels):
        sel = (group_idx == i)
        xt = y_true[sel]; yp = y_pred[sel]
        if xt.size < 3:
            continue
        # 按真值分位分箱
        qs = np.linspace(0,1,CALIB_BINS+1)
        edges = np.quantile(xt, qs)
        # 去重
        for j in range(1, len(edges)):
            if edges[j] <= edges[j-1]:
                edges[j] = edges[j-1] + 1e-9
        edges[0] = np.nanmin(xt); edges[-1] = np.nanmax(xt)
        # 计算每个箱的均值
        mx, my, n = [], [], []
        for j in range(CALIB_BINS):
            lo, hi = edges[j], edges[j+1]
            m = (xt >= lo) & (xt <= hi) if j==0 else (xt > lo) & (xt <= hi)
            if np.any(m):
                mx.append(float(np.mean(xt[m])))
                my.append(float(np.mean(yp[m])))
                n.append(int(np.sum(m)))
            else:
                mx.append(np.nan); my.append(np.nan); n.append(0)
        ax.plot(mx, my, marker="o", linewidth=1.6, label=lab)
        # 汇总 CSV
        for j in range(len(mx)):
            all_rows.append({"group": lab, "bin": j+1, "mean_truth": mx[j], "mean_pred": my[j], "count": n[j]})
    # 等式线
    x_all = np.array([r["mean_truth"] for r in all_rows if np.isfinite(r["mean_truth"])])
    if x_all.size > 0:
        x0, x1 = float(np.nanmin(x_all)), float(np.nanmax(x_all))
        ax.plot([x0, x1], [x0, x1], linestyle="--", color="k", linewidth=1.2, label="y = x")
    ax.set_xlabel("Mean truth (per bin)")
    ax.set_ylabel("Mean prediction (per bin)")
    ax.set_title("Calibration lines by stratum")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    _save_dual(fig, out_base)
    pd.DataFrame(all_rows).to_csv(str(out_base) + ".csv", index=False)

def main():
    outdir = _ensure_outdirs()
    print("[输出目录]", outdir)

    fp = Path(RUN_DIR) / INPUT_FILE
    if not fp.is_file():
        raise FileNotFoundError(f"未找到输入文件：{fp}")

    df = pd.read_csv(fp).dropna(how="all").copy()
    y_true_col, y_pred_col = _auto_cols(df)
    df = _ensure_len_gc(df)

    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values
    utr_len = df["utr_len"].astype(float).values
    gc_frac = df["gc_frac"].astype(float).values

    # 统一坐标范围（保证各层可比）
    all_min = float(np.nanmin([y_true.min(), y_pred.min()]))
    all_max = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = 0.02 * (all_max - all_min + 1e-9)
    limits = (all_min - pad, all_max + pad)

    # —— 按长度分层 ——
    len_edges = _quantile_edges(utr_len, N_QUANT)
    len_labels = _labels_from_edges(len_edges)
    len_idx = np.digitize(utr_len, len_edges, right=True) - 1
    len_idx = np.clip(len_idx, 0, len(len_edges)-2)

    rec_len = _plot_panel(y_true, y_pred, len_idx, len_labels,
                          "Parity by 3'UTR length (quantile strata)",
                          outdir / "length" / "parity_length_quartiles", limits)
    pd.DataFrame(rec_len).to_csv(outdir / "length" / "summary_length_quartiles.csv", index=False)
    _calib_lines(y_true, y_pred, len_idx, len_labels, outdir / "lines" / "calibration_length_quartiles")

    # —— 按 GC 分层 ——
    gc_edges = _quantile_edges(gc_frac, N_QUANT)
    gc_labels = _labels_from_edges(gc_edges)
    gc_idx = np.digitize(gc_frac, gc_edges, right=True) - 1
    gc_idx = np.clip(gc_idx, 0, len(gc_edges)-2)

    rec_gc = _plot_panel(y_true, y_pred, gc_idx, gc_labels,
                         "Parity by GC fraction (quantile strata)",
                         outdir / "gc" / "parity_gc_quartiles", limits)
    pd.DataFrame(rec_gc).to_csv(outdir / "gc" / "summary_gc_quartiles.csv", index=False)
    _calib_lines(y_true, y_pred, gc_idx, gc_labels, outdir / "lines" / "calibration_gc_quartiles")

    # 配置快照
    with open(outdir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({
            "RUN_DIR": RUN_DIR, "INPUT_FILE": INPUT_FILE,
            "USE_HEXBIN": USE_HEXBIN, "HEX_GRIDSIZE": HEX_GRIDSIZE,
            "N_QUANT": N_QUANT, "CALIB_BINS": CALIB_BINS
        }, f, ensure_ascii=False, indent=2)

    print("[完成] 输出目录：", outdir)

if __name__ == "__main__":
    main()
