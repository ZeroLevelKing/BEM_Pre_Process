# Gmsh Pre-processing Tool

## 简介
这是一个基于 Gmsh SDK 的 Python 高性能预处理脚本，专为复杂装配体几何设计。它自动化导入 STEP/IGES 几何文件、执行布尔运算修复几何、进行并行化 2D 三角形网格划分、校正表面单元法线方向，并输出按区域分类的数据文件及通用的可视化文件。

## 核心特性
1.  **高性能几何处理**：
    *   **并行计算**：利用多核 CPU (`multiprocessing`) 加速网格生成。
    *   **批量布尔运算**：采用 OpenCASCADE 批量 `fragment` 操作，替代传统的 $O(N^2)$ 循环，极大提升了多实体几何（如成百上千个部件）的压印与分割速度。
    *   **Frontal-Delaunay 算法**：默认使用 Frontal-Delaunay 2D 算法 (ID=6)，生成质量更高、更均匀的三角形网格。
2.  **自动化流程**：自动识别 `tsv.stp`, `model.stp` 等常见几何文件，无需人工干预。
3.  **拓扑修复与法线校正 (BFS)**：
    *   **几何压印**：自动处理实体间的接触面，无需在 CAD 软件中预处理布尔运算。
    *   **法线统一**：使用广度优先搜索 (BFS) 算法，确保每个封闭曲面的网格法线方向一致指向外部。
4.  **灵活输出**：
    *   **按需导出**：通过 `--format` 参数控制输出格式，避免生成不必要的文件。
    *   **数值数据**：生成专为自研求解器设计的 `.txt` 拓扑数据 (`nodes.txt`, `elements.txt` 等)。
    *   **可视化文件**：支持 `.vtk`, `.msh`, `.stl`, `.cgns`, `.obj`。
5.  **性能监控**：详细记录几何加载、布尔运算、网格划分、法线校正等各阶段的耗时。

## 运行环境
*   Python 3.x
*   Gmsh SDK (`pip install gmsh`)
*   NumPy (`pip install numpy`)

## 使用方法

### 基本运行
自动查找目录下几何文件，使用默认网格尺寸（1.0 ~ 3.0），默认仅输出 `.vtk`：
```bash
python ff.py
```

### 命令行参数说明

| 参数         | 默认值     | 说明                                                       |
| :----------- | :--------- | :--------------------------------------------------------- |
| `--input`    | (自动查找) | 指定输入的几何文件路径 (`.step`, `.iges`, `.brep`)         |
| `--size_min` | `1.0`      | 最小网格单元尺寸                                           |
| `--size_max` | `3.0`      | 最大网格单元尺寸                                           |
| `--format`   | `stl`      | 输出的可视化格式，支持 `vtk, msh, stl, cgns, obj` 或 `all` |

### 示例

1.  **指定输入文件与网格尺寸**：
    ```bash
    python ff.py --input engine.stp --size_min 0.5 --size_max 2.0
    ```

2.  **导出多种格式**：
    ```bash
    # 同时导出 VTK 和 MSH 文件
    python ff.py --format "vtk, msh"
    ```

3.  **导出所有支持的格式**：
    ```bash
    python ff.py --format all
    ```

## 输出文件结构

脚本运行后会在 `out/` 目录下生成以下结构：

### 1. `out/data/` (数值计算拓扑数据)
供求解器读取的核心网格连接关系。

| 文件名            | 内容描述                                                       |
| :---------------- | :------------------------------------------------------------- |
| **nodes.txt**     | 节点坐标列表。格式：`ID X Y Z`                                 |
| **elements.txt**  | **外部边界**单元列表。包含经过法线校正的三角形单元。           |
| **interface.txt** | **内部交界面**单元列表。记录不同体（Volume）之间接触面的网格。 |
| **inter.txt**     | 内部交界面索引表。                                             |
| **zone.txt**      | 区域统计信息（各 Volume 的单元数量）。                         |

### 2. `out/visual/` (可视化模型)
根据 `--format` 参数生成，可直接拖入 Tecplot, ParaView, MeshLab 查看。
*   `visualization.vtk`: (默认) 最通用的科学可视化格式。
*   `visualization.stl`: 表面三角网格。
*   `visualization.msh`: Gmsh 原生格式。

### 3. `out/log/` (运行日志)
*   `process.log`: 记录 Python 脚本运行日志及 Gmsh 内核的详细输出。
