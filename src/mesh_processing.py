import gmsh
import numpy as np
import logging
import os
from collections import deque
from src.geometry import (
    find_max_x,
    determine_orientation,
    get_shared_edge,
    adjust_triangle_orientation
)

def bfs_fix_orientation(start_elem, surfaces):
    """
    Propagates the orientation of the start_elem to all connected elements in surfaces using BFS.
    """
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

    # Pre-calculate set for fast check
    start_elem_set = set(start_elem)

    for idx, tri in enumerate(elem_nodes):
        # 寻找起始点的索引
        # 注意: 传入的 start_elem 可能已经被 check_and_fix_orientation 翻转过
        # 所以我们需要匹配节点集合相同的那个原始单元，并将其强制更新为 start_elem 的顺序
        if start_idx == -1 and set(tri) == start_elem_set:
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
                    shared_edge_tuple = get_shared_edge(current_tri, nbr_tri)

                    if shared_edge_tuple:
                        corrected_neighbor = adjust_triangle_orientation(
                            current_tri, nbr_tri, shared_edge_tuple)

                        # 更新节点列表中的邻居为校正后的版本，以便后续传播
                        elem_nodes[nbr_idx] = corrected_neighbor

                        visited_indices.add(nbr_idx)
                        queue.append(nbr_idx)

    return corrected_elements


def check_and_fix_orientation(e, finterface, felement, fzone, finter, data_dir=None):
    """
    Main logic to verify and correct mesh orientation for a given entity 'e'.
    Writes results securely to provided file handles.
    """
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

    # If the chosen element's normal points inwards (vs our reference point strategy), flip it
    if inward:
        max_x_elem = max_x_elem[::-1]

    corrected_elements = bfs_fix_orientation(max_x_elem, surface)

    # Build a hash map for fast O(1) lookup of corrected element orientations
    # Key: frozenset of nodes (order-independent), Value: list of nodes (correct order)
    corrected_map = {frozenset(tri): tri for tri in corrected_elements}

    num_nodes = 3
    for sur in surface:
        # Get raw elements from Gmsh
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(
            sur[0], abs(sur[1]))

        # Tags don't need reshaped
        # But nodes do
        tags = elemTag[0]
        nodes_flat = elemNodeTag[0]
        nodes_reshaped = np.array(nodes_flat).reshape((-1, num_nodes))

        # Check if interface
        n = gmsh.model.getEntityName(sur[0], abs(sur[1]))

        if n == "interface":
            finterface.write(str(abs(sur[1])) + "\n")
            finterface.write(str(e[1]) + "\n")
            up, down = gmsh.model.getAdjacencies(sur[0], abs(sur[1]))
            # If multiple connections, write total count
            if len(up) > 1:
                number = len(tags) # sum(len(i) for i in elemTag) if multiple types, but usually type 2 is dominant for 2D
                finterface.write(str(number) + "\n")

            for tag, nodes_arr in zip(tags, nodes_reshaped):
                # Optimize lookup using hash map
                corrected_elem = corrected_map.get(frozenset(nodes_arr))
                final_nodes = corrected_elem if corrected_elem is not None else nodes_arr

                nodes_str = ' '.join(str(i) for i in final_nodes)

                finterface.write(str(tag) +
                                 " " +
                                 str(nodes_str) +
                                 " " +
                                 str(abs(sur[1])) +
                                 "\n")
        else:
            # Regular element
            for tag, nodes_arr in zip(tags, nodes_reshaped):
                # Optimize lookup using hash map
                corrected_elem = corrected_map.get(frozenset(nodes_arr))
                final_nodes = corrected_elem if corrected_elem is not None else nodes_arr

                nodes_str = ' '.join(str(i) for i in final_nodes)

                # Format compatible with original ff.py
                # tag node1 node2 node3 surf_id vol_id
                felement.write(str(tag) +
                               " " +
                               str(nodes_str) +
                               " " +
                               str(abs(sur[1])) +
                               " " +
                               str(abs(e[1])) +
                               "\n")
