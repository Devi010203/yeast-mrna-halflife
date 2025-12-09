<<<<<<< HEAD
# mRNA 3′UTR 半衰期预测项目
=======
# Yeast-mrna-halflife
Code and scripts for yeast mRNA half-life prediction from UTR sequences using RNA-FM and PyTorch.
>>>>>>> add6740321fc6c34ccd38b9819d531a83785d9de

本仓库包含一个基于 **RNA-FM 预训练模型 + Transformer** 的 mRNA 3′UTR 半衰期预测项目，  
包括主模型训练、五折交叉验证、一次完整训练，以及论文中使用的主要分析与绘图脚本。

---

## 1. 环境与依赖（uv 管理）

本项目使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依赖与虚拟环境：

- `pyproject.toml`：项目依赖与配置文件  
- `uv.lock`：依赖锁定文件（保证环境可复现）

### 快速开始

在项目根目录（包含 `pyproject.toml` 的目录）执行：

    # 安装/同步依赖并创建虚拟环境（.venv/）
    uv sync

    # 运行主程序（五折交叉验证 + 全数据完整训练/评估）
    uv run main.py

如需运行单独的分析/绘图脚本，例如交叉验证结果可视化：

    uv run Paper/plots/new-2/plot_cv.py

（脚本路径请根据实际目录结构自行调整。）

---

## 3. 主程序（main.py）

- `main.py` 为项目主入口，默认实现：
  - **五折交叉验证**：在训练集上进行 5-fold CV，输出每折验证指标、日志与中间结果；
  - **一次完整训练与评估**：使用全部训练数据重新训练最终模型，并在独立测试集上评估性能。

运行示例：

    uv run main.py

有关数据路径、输出目录等细节，请参考 `main.py` 内部注释和配置。

---

## 4. 数据与预训练模型（data_release）

`data_release/` 目录中包含：

- `processed/`：  
  由公开论文补充数据 + 标准基因组/测序文件，通过 Python 脚本自动处理得到的 **派生数据集**  
  （例如提取 3′UTR 序列、整合半衰期、计算基础特征等）。

- `predictions/`：  
  主程序在独立测试集上的 **真实值 + 模型预测值** 表格，可用于复现实验结果和绘图。
<<<<<<< HEAD
=======
  主程序最终的最佳训练模型文件。
>>>>>>> add6740321fc6c34ccd38b9819d531a83785d9de

- `model/`：  
  预留的 **RNA-FM 预训练模型存放位置**。  
   本仓库不直接提供权重文件。  
  请按 `data_release/README.md` 中的说明，从官方渠道下载对应的 RNA-FM 模型与 tokenizer，并放置在该目录下，以便完整复现实验。

> 更详细的数据来源与字段说明，请见 `data_release/README.md`。

---

## 5. 分析与绘图脚本

与论文主文和补充材料相关的大部分分析脚本位于：

- `code_release/src/analysis_and_plot`

典型用途包括（举例）：

- 交叉验证性能与学习曲线
- Bland–Altman 分析
- 残差分布与正态性检验
- k-mer 富集与误差来源分析
- 滑窗遮挡位置效应分析
- 可解释性（motif 替换、in-silico 突变）分析

通常可以在项目根目录下通过 uv 直接运行，例如：

    uv run plot_bland_altman.py

各脚本所需输入文件（数据集/预测结果）一般来自：

- `data_release/processed/`
- `data_release/predictions/`
- 主程序运行生成的 `result/` 目录  

具体请参考对应脚本文件内注释或子目录下的 README。

---

## 6. Citation（引用）

如果你在科研工作中使用了本仓库的代码或数据集，建议在论文中引用本项目/对应论文。  
在论文正式发表前，可以暂时使用类似格式：

> Xu J. *Deep learning–based prediction of mRNA 3′UTR half-life from sequence features* (manuscript in preparation).

（未来如有正式期刊信息和 DOI，可在此处更新为正式引用格式或 BibTeX。）

---

## 7. License（许可证）

本项目以 **MIT License** 开源发布。  
完整条款请见根目录下的 `LICENSE` 文件。
