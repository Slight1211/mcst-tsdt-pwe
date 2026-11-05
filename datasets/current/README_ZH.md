# 复合微板带隙与固定面积形状设计：数据与代码

**数据集 1.1.2** · [下载](https://github.com/Slight1211/mcst-tsdt-pwe/releases/tag/data-v1.1.2)

作者：Xin Chen、Jun Cai、Alex Kalamkarov、Abdolhamid Akbarzadeh。

本数据集包含九张图、九张表、五组优化界面及其支撑计算。研究以钢–环氧单胞为例，采用结合修正偶应力理论（MCST）和 Reddy 三阶板运动学的平面波模型。数据涵盖频谱、模态场、灵敏度核验、固定面积界面优化和三维模型比较。

## 查找数据

在 `06_provenance/FIGURE_TABLE_MAP.csv` 中，按图表编号查找对应数据、原始记录和处理代码。路径均相对于解压后的数据集根目录。

| 目录 | 内容 |
|---|---|
| `01_current_figures/` | 六张正文图、三张补充图及来源数据 |
| `02_current_tables/` | 两张正文表和七张补充表的 CSV |
| `03_optimized_shapes/` | 按材料内尺度分组的边界、频谱和参数 |
| `04_reproduction/` | 校验工具、处理脚本与数值代码 |
| `05_source_records/` | 按研究内容和算例分组的原始计算记录 |
| `06_provenance/` | 来源映射、文件身份与校验值 |

五组设计均采用 `a = 75 µm`、`h = 6 µm`、钢相面积分数 `0.09π`、`N = 15`、边界 `b = 0.24`、48 个界面顶点及六个形状坐标。在 `ell_um = 0、0.5、1、2、5` 时，153 个波矢点上的相对带隙分别为 34.58%、34.31%、33.22%、27.05% 和 2.96%。全精度摘要见 `03_optimized_shapes/summary.csv`。表 2 的板模型采用表注所述的 45 点集，实体比较采用 Γ、X、M 三点。

表 S5(a) 的六份实体–实体 MAC 记录位于 `02_current_tables/Table_S5/solid_solid_mac/`，按 `large_cell`、`small_cell` 和 `thick_plate` 分组。同级的 `solid_solid_mac_recheck.csv` 将从原记录计算的数值与表格显示值逐项对照；`06_provenance/MAC_RECORDS.csv` 给出文件身份和哈希。

名为 `displayed_values` 的表格 CSV 保留文章中的舍入精度，重新计算请使用映射中的未舍入记录。空白单元格和 `<` 上界保留原表中的含义。

## 校验与复现

解压到较短路径后，在数据集根目录运行：

```text
python 04_reproduction/verify_dataset.py --root .
python 04_reproduction/verify_mac_records.py --root .
```

这两个工具仅需 Python 标准库，用于校验文件完整性和已有 MAC 记录的表格计算。处理脚本见 `FIGURE_TABLE_MAP.csv`，所需原始输入见 `SOURCE_MAP.csv`。数组与绘图处理使用 NumPy、Matplotlib，PWE 计算还使用 SciPy；三维求解需要已授权的 COMSOL。运行各求解脚本时按所记录的设置配置，并将新计算写入独立输出目录。

## 单位与引用

`ell_um` 为环氧材料内尺度，单位 µm。顶点坐标已除以 `a`；`kx_a`、`ky_a` 和 `Omega` 为无量纲量，其中 `Ω = 2πaf / sqrt(E_B/ρ_B)`，`B` 表示环氧基体。标有 Hz、kHz 或 MHz 的频率列为有量纲量。相对带隙 `J = 200(U−L)/(U+L)` 以百分数表示，单位为 `pp` 的增益是百分点。能量参与率、对数灵敏度和 MAC 均无量纲。

引用本数据集请使用 `CITATION.cff`。原创数据、图件和文档采用 CC BY 4.0，原创可执行代码采用 MIT。许可范围和署名信息见 `LICENSES.md`。
