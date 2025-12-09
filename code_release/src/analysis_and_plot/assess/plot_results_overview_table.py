# plots/plot_results_overview_table.py
# -*- coding: utf-8 -*-
"""
生成“结果总览表”：
 - 逐折指标（MAE, RMSE, R², Pearson, Spearman, N）
 - 5折汇总：均值±SD
 - Pooled（把5折拼一起）指标
 - 测试集（final_test_predictions.csv）指标
输出：result/plot/overview_table/<时间戳>/   （PNG+SVG, dpi=400）
并导出：cv_metrics_per_fold.csv / cv_summary.csv / cv_pooled.csv / final_test_metrics.csv / overview_table.tex
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import scienceplots
plt.style.use('science')

# ========== 在此手动填写 ==========
FOLD_FILES: Dict[str, str] = {
    "fold1": r"Path\to\val_predictions_fold1.csv",
    "fold2": r"Path\to\val_predictions_fold2.csv",
    "fold3": r"Path\to\val_predictions_fold3.csv",
    "fold4": r"Path\to\val_predictions_fold4.csv",
    "fold5": r"Path\to\val_predictions_fold5.csv",
}
FINAL_TEST_FILE = r"Path\to\final_test_predictions.csv"
DPI = 400
# =================================

# 全局非斜体字体
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Noto Sans CJK SC"],
    "font.style": "normal",
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
})

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_outdir() -> Path:
    outdir = _project_root() / "result" / "plot" / "overview_table" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _auto_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cand_true = ["true","target","label","y","y_true","ground_truth","half_life","halflife","halflife_true"]
    cand_pred = ["pred","prediction","y_pred","yhat","y_hat","predicted","prediction_mean"]
    yt = yp = None
    for c in df.columns:
        if c.lower() in cand_true: yt = c; break
    for c in df.columns:
        if c.lower() in cand_pred: yp = c; break
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

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float); y_pred = y_pred.astype(float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if y_true.size == 0:
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan, "Spearman": np.nan}
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    ssr = float(np.sum(resid**2))
    sst = float(np.sum((y_true - np.mean(y_true))**2))
    r2  = float(1.0 - ssr/sst) if sst > 1e-12 else np.nan
    # 手写相关（免 scipy）
    def _pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())
        return float((a*b).sum() / den) if den > 0 else np.nan
    def _spearman(a, b):
        ra = pd.Series(a).rank().values
        rb = pd.Series(b).rank().values
        return _pearson(ra, rb)
    return {"N": int(y_true.size), "MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": _pearson(y_true, y_pred), "Spearman": _spearman(y_true, y_pred)}

def _read_metrics_from_csv(path: str) -> Dict[str, float]:
    df = pd.read_csv(path).dropna(how="all")
    yt, yp = _auto_cols(df)
    return _metrics(df[yt].values, df[yp].values)

def _save_dual(fig, out_base: Path, dpi: int):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _latex_table(rows: list, col_names: list) -> str:
    # 简单 tabular 生成（booktabs 风格）
    def fmt(x):
        if isinstance(x, (int, np.integer)):
            return str(x)
        try:
            v = float(x)
            if np.isnan(v): return "--"
            return f"{v:.3f}"
        except Exception:
            return str(x)
    head = "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    head += "Split & N & MAE & RMSE & $R^2$ & Pearson & Spearman \\\\\n\\midrule\n"
    body = ""
    for r in rows:
        body += f"{r['Split']} & {r['N']} & {fmt(r['MAE'])} & {fmt(r['RMSE'])} & {fmt(r['R2'])} & {fmt(r['Pearson'])} & {fmt(r['Spearman'])} \\\\\n"
    tail = "\\bottomrule\n\\end{tabular}\n"
    return head + body + tail

def main():
    outdir = _ensure_outdir()
    print("[输出目录]", outdir)

    # 逐折
    per_fold = []
    pooled_y_true = []
    pooled_y_pred = []
    for name, p in FOLD_FILES.items():
        fp = Path(p)
        if not fp.is_file():
            print(f"[WARN] 未找到：{fp}，跳过 {name}")
            continue
        df = pd.read_csv(fp).dropna(how="all")
        yt, yp = _auto_cols(df)
        met = _metrics(df[yt].values, df[yp].values)
        met["Split"] = name
        per_fold.append(met)
        pooled_y_true.append(df[yt].values.astype(float))
        pooled_y_pred.append(df[yp].values.astype(float))
    if len(per_fold) == 0:
        raise RuntimeError("没有有效的折文件，请在 FOLD_FILES 中填写有效路径。")
    df_fold = pd.DataFrame(per_fold)[["Split","N","MAE","RMSE","R2","Pearson","Spearman"]]
    df_fold.to_csv(outdir / "cv_metrics_per_fold.csv", index=False)

    # 5折汇总：均值±SD
    stats_cols = ["MAE","RMSE","R2","Pearson","Spearman"]
    mean_vals = df_fold[stats_cols].mean(numeric_only=True)
    std_vals  = df_fold[stats_cols].std(numeric_only=True, ddof=1)
    df_cv = pd.DataFrame({"metric": stats_cols,
                          "mean": [mean_vals[c] for c in stats_cols],
                          "std":  [std_vals[c] for c in stats_cols]})
    df_cv.to_csv(outdir / "cv_summary.csv", index=False)

    # Pooled（把 5 折拼接）
    ytp = np.concatenate(pooled_y_true); ypp = np.concatenate(pooled_y_pred)
    pooled = _metrics(ytp, ypp)
    pd.DataFrame([pooled]).to_csv(outdir / "cv_pooled.csv", index=False)

    # 测试集
    test_metrics = {}
    if FINAL_TEST_FILE and Path(FINAL_TEST_FILE).is_file():
        test_metrics = _read_metrics_from_csv(FINAL_TEST_FILE)
        pd.DataFrame([test_metrics]).to_csv(outdir / "final_test_metrics.csv", index=False)
    else:
        print("[WARN] 未提供 FINAL_TEST_FILE 或文件不存在，跳过测试集指标。")

    # —— 画“总览表”成图片（便于放论文或幻灯）——
    # 组织行：Fold1..Fold5 + CV mean±SD + CV pooled + Test
    rows = []
    for _, r in df_fold.sort_values("Split").iterrows():
        rows.append({"Split": r["Split"], **{k: r[k] for k in ["N","MAE","RMSE","R2","Pearson","Spearman"]}})
    rows.append({"Split": "CV mean±SD",
                 "N": int(df_fold["N"].mean()),
                 "MAE": f"{mean_vals['MAE']:.3f} ± {std_vals['MAE']:.3f}",
                 "RMSE": f"{mean_vals['RMSE']:.3f} ± {std_vals['RMSE']:.3f}",
                 "R2": f"{mean_vals['R2']:.3f} ± {std_vals['R2']:.3f}",
                 "Pearson": f"{mean_vals['Pearson']:.3f} ± {std_vals['Pearson']:.3f}",
                 "Spearman": f"{mean_vals['Spearman']:.3f} ± {std_vals['Spearman']:.3f}"})
    rows.append({"Split": "CV pooled", **pooled})
    if test_metrics:
        rows.append({"Split": "Test", **test_metrics})

    # 导出 LaTeX（booktabs）
    tex = _latex_table(rows, ["Split","N","MAE","RMSE","R2","Pearson","Spearman"])
    (outdir / "overview_table.tex").write_text(tex, encoding="utf-8")

    # 生成 PNG/SVG 表格
    # 将数值统一为字符串格式
    def fmt_val(v):
        if isinstance(v, str): return v
        try:
            x = float(v)
            return f"{x:.3f}" if np.isfinite(x) else "--"
        except Exception:
            return str(v)
    table_cols = ["Split","N","MAE","RMSE","R2","Pearson","Spearman"]
    table_data = [[fmt_val(r.get(c, "")) for c in table_cols] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.35*len(table_data)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=table_cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    ax.set_title("Performance overview (5-fold CV and test)", pad=12)
    # 保存
    _save_dual(fig, outdir / "overview_table", DPI)

    print("[完成] 输出目录：", outdir)

if __name__ == "__main__":
    import pandas as pd, numpy as np
    main()
