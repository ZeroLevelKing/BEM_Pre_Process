# BEM_Pre_Process

基于 Gmsh Python API 的边界元网格前处理程序。输入包含体实体的 STEP、IGES 或 BREP 几何文件，生成表面三角形网格，识别子域间界面，调整面元方向，并导出节点、面元和子域数据。

## 依赖

- Python 3
- Gmsh Python 包
- NumPy

```bash
python -m pip install gmsh numpy
```

## 运行

在项目目录中执行。指定单个几何文件和网格尺寸参数：

```bash
python main.py --input model.step --size_min 0.5 --size_max 2.0
```

不指定 `--input` 时，程序按文件名顺序处理当前工作目录中的 `.igs` 和 `.iges` 文件：

```bash
python main.py
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--input` | 当前目录中的 IGES 文件 | 指定单个 STEP、IGES 或 BREP 文件 |
| `--size_min` | `1.0` | Gmsh 最小网格尺寸参数 |
| `--size_max` | `3.0` | Gmsh 最大网格尺寸参数 |
| `--format` | `none` | 可选 `none`、`vtk`、`msh`、`cgns`、`obj` 或 `all`，多个格式用逗号分隔 |

网格尺寸参数采用输入几何的坐标单位。程序设置 Gmsh 的二维网格算法为 Frontal-Delaunay，并将 `General.NumThreads` 设置为 CPU 逻辑核心数。

同时导出 VTK 和 MSH 文件：

```bash
python main.py --input model.step --size_min 0.5 --size_max 2.0 --format vtk,msh
```

## 输出

输出位于当前工作目录的 `out/` 下。每个模型的数据目录和可视化文件名包含输入文件名、网格尺寸参数及时间戳。

| 路径 | 内容 |
| --- | --- |
| `out/data/<模型名>_min<最小尺寸>_max<最大尺寸>_<时间戳>/` | 五个求解器数据文件 |
| `out/visual/` | 由 `--format` 指定的可视化文件 |
| `out/log/process.log` | 运行日志和 Gmsh 日志 |

五个数据文件的格式如下。面元节点顺序经过 BFS 方向传播和方向判定处理。

| 文件 | 格式 |
| --- | --- |
| `nodes.txt` | 首行为节点数，随后每行为 `节点编号 x y z` |
| `elements.txt` | 前两行为外边界面元数加单侧界面面元数、单侧界面面元数。随后为外边界面元，每行为 `面元编号 节点1 节点2 节点3 曲面编号 子域编号` |
| `interface.txt` | 首行为界面记录组数，每组依次写入曲面编号、子域编号、面元数，再逐行写入 `面元编号 节点1 节点2 节点3 曲面编号`。共享界面分别记录两侧子域的面元 |
| `inter.txt` | 首行为共享界面数，随后每行一个界面曲面编号 |
| `zone.txt` | 首行为子域数，随后每行一个子域的面元数，最后每行一个子域处理顺序编号，默认顺序为 `1` 到子域数 |

例如，两个子域各包含 84 个面元时，`zone.txt` 为：

```text
2
84
84
1
2
```

边界条件和材料参数由求解器设置。使用输出文件前，需要核对求解器的编号规则、子域数量、界面配对及坐标单位。

导出时按体实体的遍历顺序，将 Gmsh 体编号映射为连续的 `1…N`。`elements.txt` 和 `interface.txt` 使用相同的子域编号映射，顺序与 `zone.txt` 中的面元计数一致。

## 源码目录

| 文件 | 职责 |
| --- | --- |
| `main.py` | 导入几何、划分网格并组织数据输出 |
| `src/geometry.py` | 几何计算和面元方向判定 |
| `src/mesh_processing.py` | BFS 面元方向传播及界面、外边界数据写出 |
| `src/export.py` | 可视化文件导出 |
| `src/logger.py` | 日志配置与 Gmsh 日志读取 |
| `tools/view_msh.py` | 网格查看工具 |
| `tools/benchmark_bfs_scaling.py` | 网格与 BFS 处理的规模测试 |
| `tools/run_single.py` | 指定网格尺寸的单次测试 |
