# Gmsh Pre-processing Tool

## 简介
这是一个基于 Gmsh SDK 的 Python 预处理脚本，用于自动化导入 STEP 几何文件、进行 2D 三角形网格划分、校正表面单元法线方向，并输出按区域分类的数据文件及通用的可视化文件。

## 功能特性
1.  **自动化流程**：自动读取 `tsv.stp` 几何文件，生成网格，无需人工干预。
2.  **法线校正 (BFS)**：使用广度优先搜索 (BFS) 算法，确保每个封闭曲面的法线方向一致（统一指向外部）。
3.  **多格式输出**：
    *   **数据文件**：专为数值计算程序设计的 `.txt` 格式网格数据。
    *   **可视化文件**：支持 `.vtk` (ParaView/Tecplot), `.msh` (Gmsh), `.stl`, `.cgns`, `.obj` 等多种通用格式。
4.  **实时日志**：捕获 Gmsh 内部日志并实时记录到 `out/log/process.log`。
5.  **命令行控制**：可通过命令行参数灵活控制网格划分精度。

## 运行环境
*   Python 3.x
*   Gmsh SDK (`pip install gmsh`)
*   NumPy (`pip install numpy`)

## 使用方法

### 基本运行
使用默认网格尺寸（1.0 ~ 10.0）运行（脚本将自动在当前目录下查找 `tsv.stp`、`tsv.igs`、`tsv.brep` 等文件）：
```bash
python ff.py
```

### 指定输入文件
通过 `--input` 参数指定几何文件：
```bash
# 导入 STEP 文件
python ff.py --input my_model.stp

# 导入 IGES 文件
python ff.py --input geometry.igs

# 导入 BREP 文件
python ff.py --input shape.brep
```
**注意**: 标准 Gmsh 构建通常不支持 `.sat` (ACIS) 格式。建议先将其转换为 STEP 或 IGES 格式。

### 指定网格精度
通过命令行参数控制网格的最大/最小尺寸：
```bash
# 设置最小和最大网格尺寸均为 50.0（此时网格更精细）
python ff.py --size_min 50.0 --size_max 50.0

# 仅指定最大尺寸
python ff.py --size_max 100.0
```

## 输出文件说明

脚本运行后会在 `out/` 目录下生成以下三个子目录：

### 1. `out/data/` (数值计算数据)
供后续求解器读取的核心网格拓扑数据。

| 文件名            | 内容描述       | 格式说明                                                                                        |
| :---------------- | :------------- | :---------------------------------------------------------------------------------------------- |
| **nodes.txt**     | 节点坐标       | 第1行：节点总数<br>后续行：`ID X Y Z`                                                           |
| **elements.txt**  | 外部边界单元   | 第1行：所有单元总数<br>第2行：内部交界面单元总数<br>后续行：`ID Node1 Node2 Node3 SurfID VolID` |
| **interface.txt** | 内部交界面单元 | 记录不同区域（Volume）接触面上的单元信息。<br>结构：`[SurfID] [VolID] [Count] [Elements...]`    |
| **inter.txt**     | 交界面索引     | 列出被判定为内部交界面的 Surface ID。                                                           |
| **zone.txt**      | 区域统计       | 第1行：子区域（Volume）总数。<br>后续行：每个区域包含的单元总数。                               |

### 2. `out/visual/` (可视化模型)
通用格式，可直接拖入 Tecplot, ParaView, MeshLab 等软件查看。

*   `visualization.vtk`: 最通用的科学数据格式。
*   `visualization.msh`: Gmsh 原生格式，包含最全的几何信息。
*   `visualization.stl`: 表面三角网格，适用于 CAD/3D 打印软件。
*   `visualization.cgns`: CFD 领域标准格式 (需 Gmsh 编译支持)。
*   `visualization.obj`: 3D 图形学通用模型格式。

### 3. `out/log/` (日志)
*   `process.log`: 包含 Python 脚本的运行日志以及 Gmsh 内核的实时输出日志。
