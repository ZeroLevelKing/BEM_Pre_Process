import os
import gmsh

def export_visualization(format_arg, output_dir='out/visual'):
    """
    Exports the current Gmsh model to various visualization formats based on the format_arg.
    Supported formats: 'vtk', 'msh', 'stl', 'cgns', 'obj' or 'all'.
    """
    # Ensure visualization directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting visualization files ({format_arg}) to '{output_dir}'...")

    export_formats = []
    if format_arg == 'all':
        export_formats = ['vtk', 'msh', 'stl', 'cgns', 'obj']
    else:
        # Allow comma separated input like "vtk,msh"
        export_formats = [fmt.strip() for fmt in format_arg.split(',')]

    # .vtk 文件通用性很强，Tecplot(通过插件) 和 ParaView 都能直接读取
    if 'vtk' in export_formats:
        gmsh.write(os.path.join(output_dir, 'visualization.vtk'))

    # .msh 是 Gmsh 原生格式，保留信息最全
    if 'msh' in export_formats:
        gmsh.write(os.path.join(output_dir, 'visualization.msh'))

    # .stl 表面网格通用格式，MeshLab/SolidWorks 等常用
    if 'stl' in export_formats:
        gmsh.write(os.path.join(output_dir, 'visualization.stl'))

    # .cgns CFD 通用格式，Tecplot 原生支持极佳
    if 'cgns' in export_formats:
        try:
            gmsh.write(os.path.join(output_dir, 'visualization.cgns'))
        except Exception:
            pass  # 忽略不支持 CGNS 的情况

    # .obj 通用 3D 模型格式
    if 'obj' in export_formats:
        gmsh.write(os.path.join(output_dir, 'visualization.obj'))
