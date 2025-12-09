# data_release 说明

本目录用于公开本项目在论文中使用的**处理后数据集**以及对应的**模型预测结果**，并为复现实验的用户预留**预训练模型存放位置**。

## 目录结构

- `processed/`  
  存放**已经处理好的 mRNA 半衰期数据集**和最终的最佳训练模型文件，包括：
  - 由原始半衰期表格与基因组序列自动生成的主数据集（包含 3′UTR 序列及基础序列特征等）；
  - 以 RNA 字母表表示的版本，方便直接输入到下游模型中。

- `predictions/`  
  存放**主程序运行后生成的测试集结果**，一般为若干 CSV 表格：
  - 每个文件通常包含测试集中样本的 **真实半衰期** 与 **模型预测值**（以及必要的标识信息）；
  - 这些文件对应论文中的独立测试集表现，可用于复现实验结果或进行二次分析。

- `model/`  
  预留的**预训练模型文件存放目录**：
  - 本项目依赖 RNA-FM 等预训练模型权重，但由于体积与授权原因，不直接随仓库分发；
  - 请用户从官方渠道下载对应的 RNA-FM 预训练模型与 tokenizer，将文件放置在本目录下（或按主程序/配置文件中的路径说明放置），即可在本地完整复现训练与推理过程。

## 数据来源与处理流程说明

本目录下的 `processed/` 数据集**并非原始实验数据的直接拷贝**，而是通过以下公开资源与程序化处理自动生成的派生数据：

1. **原始半衰期数据**  
   - 来自公开论文(Global Analysis of mRNA Isoform Half-Lives Reveals Stabilizing and Destabilizing Elements in Yeast)附带的补充表格（作者提供的 `mmc2.xlsx` 补充数据文件）；
2. **基因组与序列信息**  
   - 使用公开的标准基因组序列（酵母基因组 sacCer2 的染色体 FASTA 文件）作为参考序列；
3. **Python 脚本自动构建数据集**  
   - 通过仓库中的 Python 数据处理脚本自动完成如下步骤：  
     - 从补充表格中读取 isoform 级别的半衰期信息；  
     - 根据染色体坐标从基因组 FASTA 中提取每个 isoform 的 3′UTR 序列（去掉 poly(A)）；  
     - 计算序列长度、GC 含量等基础序列特征；  
     - 将上述信息合并为结构化数据集，并导出为 CSV 文件；  
     - 将序列从 DNA 形式（T）转换为 RNA 形式（U），生成对应的 `_RNA` 版本文件。:contentReference[oaicite:0]{index=0}  

因此，`processed/` 中的文件是基于**公开原始数据 + 标准基因组序列 + Python 程序化处理**得到的派生结果，方便读者直接使用与复现论文中的训练和评估过程。

## 使用提示

- 如需从头完全复现实验，可：
  1. 按主仓库的 README 使用 `uv` 同步依赖；
  2. 将下载好的 RNA-FM 预训练模型文件放入 `data_release/model/` 目录；
  3. 按主程序（如 `main.py` 或训练脚本）中配置的路径，使用 `data_release/processed/` 和 `data_release/predictions/` 中的数据运行训练与评估。

- 如只需进行下游分析（例如重新绘图、做附加统计），通常只需读取 `processed/` 和 `predictions/` 中的 CSV 文件即可，无需重新构建原始数据集。

---

### 原始论文：`Global Analysis of mRNA Isoform Half-Lives Reveals Stabilizing and Destabilizing Elements in Yeast`
### 作者：`Geisberg`、`Joseph V.`等
### doi:`10.1016/j.cell.2013.12.026`