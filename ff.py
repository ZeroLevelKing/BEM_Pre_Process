import gmsh
import numpy as np
import sys
import math
import os
import argparse
import logging
import threading
import time
import multiprocessing
from collections import deque
from typing import List, Tuple, Set, Dict, Optional


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging():
    log_dir = os.path.join("out", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "process.log")

    # Ensure standard handlers
    handler = FlushFileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)


def monitor_gmsh_logs(stop_event):
    while not stop_event.is_set():
        try:
            logs = gmsh.logger.get()
            for msg in logs:
                logging.info(f"Gmsh: {msg}")
        except Exception:
            pass
        time.sleep(0.5)


def find_max_x(surfaces):
    max_x_coordinate = -np.inf
    max_x_elem = None
    max_x_elem_coords = None
    max_x_elem_normal = None
    max_x_elem_center = None

    for sur in surfaces:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(
            sur[0], abs(sur[1]))

       # Check if elemNodeTag is empty or properly structured
        if not elemNodeTag or len(
                elemNodeTag) == 0 or len(elemNodeTag[0]) == 0:
            continue
        tags, coord, param = gmsh.model.mesh.getNodes(2, abs(sur[1]), True)
        num_nodes = 3
        node_coords_dict = {
            tag: coord for tag, coord in zip(
                tags, np.array(coord).reshape(
                    (-1, 3)))}
        elemNodeTag = np.array(elemNodeTag[0]).reshape((-1, num_nodes))

        for elem in elemNodeTag:
            x_coords = [node_coords_dict[node][0] for node in elem]
            max_x = min(x_coords)  # 保持找到最大的 x 坐标
            if max_x > max_x_coordinate:
                max_x_coordinate = max_x
                max_x_elem = elem
                max_x_elem_coords = [node_coords_dict[node] for node in elem]
                # Calculate normal vector
                v1 = np.array(max_x_elem_coords[1]) - \
                    np.array(max_x_elem_coords[0])
                v2 = np.array(max_x_elem_coords[2]) - \
                    np.array(max_x_elem_coords[0])
                max_x_elem_normal = np.cross(v1, v2)
                max_x_elem_normal = max_x_elem_normal / \
                    np.linalg.norm(max_x_elem_normal)  # Normalize the vector
                # Calculate center point
                max_x_elem_center = np.mean(max_x_elem_coords, axis=0)

    if max_x_elem is None or max_x_elem_center is None:
        return None, None, None, None, None

    max_x_coordinate += 10
    max_x_point = np.array(
        [max_x_coordinate, max_x_elem_center[1], max_x_elem_center[2]])
    return max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center


def determine_orientation(
        max_x_point,
        max_x_elem,
        max_x_elem_coords,
        max_x_elem_normal,
        max_x_elem_center):
    direction_vector = max_x_point - max_x_elem_center
    return np.dot(direction_vector, max_x_elem_normal) < 0


def get_shared_edge(tri1, tri2):
    # Create sets of edges for the first triangle
    edges_tri1 = {frozenset((tri1[i], tri1[(i + 1) % 3])) for i in range(3)}
    # Create sets of edges for the second triangle
    edges_tri2 = {frozenset((tri2[i], tri2[(i + 1) % 3])) for i in range(3)}
    # Find the intersection of the two sets of edges
    shared_edges = edges_tri1.intersection(edges_tri2)
    # If there is a shared edge, return it as a sorted tuple
    if shared_edges:
        edge = shared_edges.pop()
        return tuple(edge)
    # If no shared edge is found, return None
    return None


def adjust_triangle_orientation(reference_tri, tri, shared_edge):
    ref_idx0 = np.where(reference_tri == shared_edge[0])[0][0]
    ref_idx1 = np.where(reference_tri == shared_edge[1])[0][0]
    tri_idx0 = np.where(tri == shared_edge[0])[0][0]
    tri_idx1 = np.where(tri == shared_edge[1])[0][0]

    if (ref_idx0 + 1) % 3 == ref_idx1:
        if (tri_idx0 + 1) % 3 == tri_idx1:
            # 方向相同，需要反转
            return [tri[tri_idx1], tri[tri_idx0], tri[(tri_idx1 + 1) % 3]]
        else:
            # 方向相反，不需要调整
            return tri
    else:
        if (tri_idx1 + 1) % 3 == tri_idx0:
            # 方向相反，不需要调整
            return [tri[tri_idx0], tri[tri_idx1],
                    tri[(tri_idx0 + 1) % 3]]
        else:
            return tri


def bfs_fix_orientation(start_elem, surfaces):
    elem_nodes = []
    # 获取所有表面的元素
    for sur in surfaces:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(
            sur[0], abs(sur[1]))
        if len(elemNodeTag) > 0:
            elem_nodes.extend(np.array(elemNodeTag[0]).reshape((-1, 3)))

    # 将元素节点转换为列表形式，方便后续索引操作
    elem_nodes = [list(tri) for tri in elem_nodes]

    # 构建邻接表: 边 (min_id, max_id) -> [element_index, ...]
    adj_map = {}
    start_idx = -1
    start_elem_tuple = tuple(start_elem)

    for idx, tri in enumerate(elem_nodes):
        # 寻找起始点的索引
        # 注意: 传入的 start_elem 可能已经被 check_and_fix_orientation 翻转过
        # 所以我们需要匹配节点集合相同的那个原始单元，并将其强制更新为 start_elem 的顺序
        if start_idx == -1 and set(tri) == set(start_elem):
            start_idx = idx
            elem_nodes[idx] = list(start_elem) # 确保起始单元的方向与传入一致

        # 记录三条边
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]
        for edge in edges:
            if edge not in adj_map:
                adj_map[edge] = []
            adj_map[edge].append(idx)

    if start_idx == -1:
        logging.error("BFS Start element not found in mesh!")
        return []

    # BFS 初始化
    queue = deque([start_idx])
    visited_indices = {start_idx}
    corrected_elements = []

    while queue:
        curr_idx = queue.popleft()
        current_tri = elem_nodes[curr_idx]
        corrected_elements.append(current_tri)

        # 获取当前(已校正)单元的三条边
        current_edges = [
            tuple(sorted((current_tri[0], current_tri[1]))),
            tuple(sorted((current_tri[1], current_tri[2]))),
            tuple(sorted((current_tri[2], current_tri[0])))
        ]

        for edge in current_edges:
            # 查找共享此边的邻居索引
            neighbors = adj_map.get(edge, [])
            for nbr_idx in neighbors:
                if nbr_idx not in visited_indices:
                    # 获取待校正的邻居单元
                    nbr_tri = elem_nodes[nbr_idx]

                    # 使用现有逻辑校正方向
                    # get_shared_edge 返回的是排好序的 tuple，与 adj_map 的 key 一致
                    # 但为了保险起见，这里复用原不仅依赖 tuple 的 get_shared_edge 逻辑
                    shared_edge_tuple = get_shared_edge(current_tri, nbr_tri)

                    if shared_edge_tuple:
                        corrected_neighbor = adjust_triangle_orientation(
                            current_tri, nbr_tri, shared_edge_tuple)

                        # 更新节点列表中的邻居为校正后的版本，以便后续传播
                        elem_nodes[nbr_idx] = corrected_neighbor

                        visited_indices.add(nbr_idx)
                        queue.append(nbr_idx)

    return corrected_elements


def check_and_fix_orientation(e, finterface, felement, fzone, finter):
    surface = gmsh.model.getBoundary([e])
    max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center = find_max_x(
        surface)
    if max_x_elem is None:
        logging.error(f"Could not determine max x element for surface {surface}")
        return

    inward = determine_orientation(
        max_x_point,
        max_x_elem,
        max_x_elem_coords,
        max_x_elem_normal,
        max_x_elem_center)

    if inward:
        max_x_elem = max_x_elem[::-1]

    corrected_elements = bfs_fix_orientation(max_x_elem, surface)

    # Build a hash map for fast O(1) lookup of corrected element orientations
    # Key: frozenset of nodes (order-independent), Value: list of nodes (correct order)
    corrected_map = {frozenset(tri): tri for tri in corrected_elements}

    num_nodes = 3
    for sur in surface:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(
            sur[0], abs(sur[1]))
        tags, coord, param = gmsh.model.mesh.getNodes(2, abs(sur[1]), True)
        n = gmsh.model.getEntityName(sur[0], abs(sur[1]))

        if n == "interface":
            finterface.write(str(abs(sur[1])) + "\n")
            finterface.write(str(e[1]) + "\n")
            up, down = gmsh.model.getAdjacencies(sur[0], abs(sur[1]))
            if len(up) > 1:
                number = sum(len(i) for i in elemTag)
                finterface.write(str(number) + "\n")

            for tag, nodes in zip(
                elemTag[0], np.array(elemNodeTag[0]).reshape(
                    (-1, num_nodes))):

                # Optimize lookup using hash map
                corrected_elem = corrected_map.get(frozenset(nodes))

                final_nodes = corrected_elem if corrected_elem is not None else nodes
                nodes_str = ' '.join(str(i) for i in final_nodes)

                finterface.write(str(tag) +
                                 " " +
                                 str(nodes_str) +
                                 " " +
                                 str(abs(sur[1])) +
                                 "\n")
        else:
            for tag, nodes in zip(
                elemTag[0], np.array(elemNodeTag[0]).reshape(
                    (-1, num_nodes))):

                # Optimize lookup using hash map
                corrected_elem = corrected_map.get(frozenset(nodes))

                final_nodes = corrected_elem if corrected_elem is not None else nodes
                nodes_str = ' '.join(str(i) for i in final_nodes)

                felement.write(str(tag) +
                               " " +
                               str(nodes_str) +
                               " " +
                               str(abs(sur[1])) +
                               " " +
                               str(abs(e[1])) +
                               "\n")


def main():
    # Parse command line arguments for mesh sizing and input file
    parser = argparse.ArgumentParser(description="Gmsh Mesh Generator")
    parser.add_argument('--input', type=str, default=None, help='Input geometry file path (STEP, IGES, BREP)')
    parser.add_argument('--size_min', type=float, default=1.0, help='Minimum mesh element size')
    parser.add_argument('--size_max', type=float, default=3.0, help='Maximum mesh element size')
    parser.add_argument('--format', type=str, default='stl', help='Export format (vtk, msh, stl, cgns, obj, all). Default: vtk')

    # Use parse_known_args to avoid conflict if other args are passed (though we might filter sys.argv for gmsh)
    args, unknown_args = parser.parse_known_args()

    # Performance timing
    t0 = time.time()
    last_t = t0

    def print_duration(step_name):
        nonlocal last_t
        current_t = time.time()
        print(f"[{step_name}] took {current_t - last_t:.4f}s (Total: {current_t - t0:.4f}s)")
        last_t = current_t

    setup_logging()
    print("Initializing Gmsh...")
    # Initialize Gmsh with only the script name to avoid parsing our custom arguments
    gmsh.initialize([sys.argv[0]] + unknown_args)

    # Enable parallel computing
    num_threads = multiprocessing.cpu_count()
    print(f"Enabling parallel computing with {num_threads} threads.")
    gmsh.option.setNumber("General.NumThreads", num_threads)

    # 拦截 Gmsh 的终端输出，改为记录到日志
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.logger.start()

    stop_logging = threading.Event()
    log_thread = threading.Thread(
        target=monitor_gmsh_logs, args=(stop_logging,))
    log_thread.start()

    # 创建一个名为 "model_name" 的模型
    gmsh.model.add("model_name")

    print_duration("Gmsh Initialization")

    # Determine input file
    geometry_file = None
    if args.input:
        geometry_file = args.input
    else:
        # Default strategy: search for common geometry files in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Priority order: STEP > IGES > BREP
        potential_files = [
            os.path.join(script_dir, 'tsv.stp'),
            os.path.join(script_dir, 'tsv.step'),
            os.path.join(script_dir, 'model.stp'),
            os.path.join(script_dir, 'geom.iges'),
            os.path.join(script_dir, 'geom.igs'),
        ]

        # Also scan directory for any .stp/.step/.igs/.iges/.brep file if specific named ones aren't found
        if not any(os.path.exists(f) for f in potential_files):
            for file in os.listdir(script_dir):
                if file.lower().endswith(('.stp', '.step', '.igs', '.iges', '.brep')):
                    potential_files.append(os.path.join(script_dir, file))
                    break

        for f in potential_files:
            if os.path.exists(f):
                geometry_file = f
                break

    if geometry_file and os.path.exists(geometry_file):
        print(f"Loading geometry file: {geometry_file}")
        try:
            # importShapes supports STEP, IGES, BREP natively via OpenCASCADE
            # Note: .sat (ACIS) is generally NOT supported in open-source Gmsh builds.
            gmsh.model.occ.importShapes(geometry_file)
        except Exception as e:
            print(f"Error importing geometry: {e}")
            logging.error(f"Error importing geometry: {e}")
            # Fallback or exit could happen here
    else:
        if args.input:
             print(f"Error: Specified input file '{geometry_file}' not found.")
        else:
             print("Error: No suitable geometry file found in directory.")

        # Fallback for testing/demonstration if no file is found
        print("Using fallback internal box geometry for demonstration...")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)

    gmsh.model.occ.synchronize()
    print("Geometry loaded.")
    print_duration("Geometry Loading")

    # 获取所有实体
    entities = gmsh.model.getEntities(dim=3)

    print("Processing geometry fragments...")

    if len(entities) > 0:
        gmsh.model.occ.fragment(entities, [])

    gmsh.model.occ.synchronize()
    print_duration("Geometry Processing")
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    print(f"Setting Mesh Size: Min={args.size_min}, Max={args.size_max}")
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.size_min)
    print("Generating mesh...")
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.size_max)

    # Enable Frontal-Delaunay for 2D meshing (Algorithm ID=6) for faster and better quality
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.mesh.generate(2)
    gmsh.model.occ.synchronize()
    print_duration("Meshing")

    # Pre-calculation processing time estimation
    # Get all 2D elements to estimate workload
    _, all_elem_tags, _ = gmsh.model.mesh.getElements(2)
    total_2d_elements = sum(len(tags) for tags in all_elem_tags)
    print(f"Total mesh elements: {total_2d_elements}")

    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

    nodeCoords = np.array(nodeCoords).reshape((-1, 3))

    # Ensure out directory exists
    os.makedirs('out', exist_ok=True)
    # Ensure data sub-directory exists
    data_dir = os.path.join('out', 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("Writing nodes to file...")

    with open(os.path.join(data_dir, 'nodes.txt'), 'w') as fnode:
        fnode.write(str(len(nodeTags)) + "\n")
        for tag, xyz_e in zip(nodeTags, nodeCoords):
            xyz_e = ' '.join(str(i) for i in xyz_e)
            fnode.write(str(tag) + ' ' + str(xyz_e) + "\n")
    print_duration("Writing Nodes")

    entities = gmsh.model.getEntities(3)
    zone = sum(len(e) for e in entities)
    numElem = 0
    boundary = 0
    i = 0
    with open(os.path.join(data_dir, 'zone.txt'), 'w') as fzone, \
            open(os.path.join(data_dir, 'inter.txt'), 'w') as finter, \
            open(os.path.join(data_dir, 'interface.txt'), 'w') as finterface, \
            open(os.path.join(data_dir, 'elements.txt'), 'w') as felement:
        fzone.write(str(int((zone) / 2)) + "\n")
        for e in entities:
            dim = e[0]
            tag = e[1]
            surface = gmsh.model.getBoundary([e])
            Elem = 0
            for sur in surface:
                dim = sur[0]
                tag = sur[1]
                elemTypee, elemTagg, elemNodeTagg = gmsh.model.mesh.getElements(
                    sur[0], abs(sur[1]))
                Elem += sum(len(i) for i in elemTagg)
                numElem += sum(len(i) for i in elemTagg)
                up, down = gmsh.model.getAdjacencies(sur[0], abs(sur[1]))
                if len(up) > 1:
                    gmsh.model.setEntityName(2, sur[1], "interface")
                    boundary += sum(len(i) for i in elemTagg)
                    i += 1
            fzone.write(str(Elem) + "\n")
        finter.write(str(int(i / 2)) + "\n")
        entities = gmsh.model.getEntities(2)
        for sur in entities:
            dim = sur[0]
            tag = sur[1]
            elemTypee, elemTagg, elemNodeTagg = gmsh.model.mesh.getElements(
                sur[0], abs(sur[1]))
            up, down = gmsh.model.getAdjacencies(dim, tag)
            if len(up) > 1:
                finter.write(str(abs(sur[1])) + "\n")
        finterface.write(str(i) + "\n")
        felement.write(str(numElem - int(boundary / 2)) + "\n")
        felement.write(str(int(boundary / 2)) + "\n")

        print("Fixing element orientation and writing output files...")

        for e in gmsh.model.getEntities(3):
            check_and_fix_orientation(e, finterface, felement, fzone, finter)
    print_duration("Fixing Orientation & Writing Elements")

    # Stop logging thread
    stop_logging.set()
    log_thread.join()

    # Process any final logs
    for msg in gmsh.logger.get():
        logging.info(f"Gmsh: {msg}")
    gmsh.logger.stop()

    print("Process completed successfully. Check 'out/' for results.")

    # 导出通用的可视化格式
    print(f"Exporting visualization files ({args.format}) to 'out/visual'...")

    # Ensure visualization directory exists
    visual_dir = os.path.join('out', 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    export_formats = []
    if args.format == 'all':
        export_formats = ['vtk', 'msh', 'stl', 'cgns', 'obj']
    else:
        # Allow comma separated input like "vtk,msh"
        export_formats = [fmt.strip() for fmt in args.format.split(',')]

    # .vtk 文件通用性很强，Tecplot(通过插件) 和 ParaView 都能直接读取
    if 'vtk' in export_formats:
        gmsh.write(os.path.join(visual_dir, 'visualization.vtk'))

    # .msh 是 Gmsh 原生格式，保留信息最全
    if 'msh' in export_formats:
        gmsh.write(os.path.join(visual_dir, 'visualization.msh'))

    # .stl 表面网格通用格式，MeshLab/SolidWorks 等常用
    if 'stl' in export_formats:
        gmsh.write(os.path.join(visual_dir, 'visualization.stl'))

    # .cgns CFD 通用格式，Tecplot 原生支持极佳
    if 'cgns' in export_formats:
        try:
            gmsh.write(os.path.join(visual_dir, 'visualization.cgns'))
        except Exception:
            pass  # 忽略不支持 CGNS 的情况

    # .obj 通用 3D 模型格式
    if 'obj' in export_formats:
        gmsh.write(os.path.join(visual_dir, 'visualization.obj'))

    print_duration("Visualization Export")
    print_duration("Total Process")

    gmsh.finalize()


if __name__ == "__main__":
    main()
