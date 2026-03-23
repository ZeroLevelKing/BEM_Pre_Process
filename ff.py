# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:17:14 2024

@author: HJY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:43:05 2024

@author: HJY
"""

import gmsh
import numpy as np
import sys
from collections import deque
import sys
import math
import os
import numpy as np
import logging

def setup_logging():
    log_file = os.path.join("out", "process.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def find_max_x(surfaces):
    max_x_coordinate = -np.inf
    max_x_elem = None
    max_x_elem_coords = None
    max_x_elem_normal = None
    max_x_elem_center = None

    for sur in surfaces:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(sur[0], abs(sur[1]))
        # Add a debug statement to check elemNodeTag
        logging.info(f"Surface: {sur}, elemNodeTag: {elemNodeTag}")

       # Check if elemNodeTag is empty or properly structured
        if not elemNodeTag or len(elemNodeTag) == 0 or len(elemNodeTag[0]) == 0:
           continue
        tags, coord, param = gmsh.model.mesh.getNodes(2, abs(sur[1]), True)
        num_nodes = 3
        node_coords_dict = {tag: coord for tag, coord in zip(tags, coord.reshape((-1, 3)))}
        elemNodeTag = np.array(elemNodeTag[0]).reshape((-1, num_nodes))

        for elem in elemNodeTag:
            x_coords = [node_coords_dict[node][0] for node in elem]
            max_x = min(x_coords)  # 保持找到最大的 x 坐标
            if max_x > max_x_coordinate:
                max_x_coordinate = max_x
                max_x_elem = elem
                max_x_elem_coords = [node_coords_dict[node] for node in elem]
                # Calculate normal vector
                v1 = np.array(max_x_elem_coords[1]) - np.array(max_x_elem_coords[0])
                v2 = np.array(max_x_elem_coords[2]) - np.array(max_x_elem_coords[0])
                max_x_elem_normal = np.cross(v1, v2)
                max_x_elem_normal = max_x_elem_normal / np.linalg.norm(max_x_elem_normal)  # Normalize the vector
                # Calculate center point
                max_x_elem_center = np.mean(max_x_elem_coords, axis=0)
    max_x_coordinate += 10
    max_x_point = np.array([max_x_coordinate, max_x_elem_center[1], max_x_elem_center[2]])
    return max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center

def determine_orientation(max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center):
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
    logging.info(f"gognxiang {ref_idx0} {ref_idx1} {tri_idx0} {tri_idx1}")
    if (ref_idx0 + 1) % 3 == ref_idx1:
       # print("调整1")
        if (tri_idx0 + 1) % 3 == tri_idx1:
            logging.info(f"调整1 {tri} {tri[tri_idx1]} {tri[tri_idx0]} {tri[(tri_idx1 + 1) % 3]}")
            return [tri[tri_idx1], tri[tri_idx0], tri[(tri_idx1 + 1) % 3]]
              # 方向相同，不需要调整
        else:
              # 方向相反，需要调整
            #  print("mei1111")
              return tri
    else:
       # print("调整2")
        if (tri_idx1 + 1) % 3 == tri_idx0:
        #if(tri_idx0 + 1) % 3 != tri_idx1:
            logging.info(f"调整2 {tri} {tri[tri_idx0]} {tri[tri_idx1]} {tri[(tri_idx0 + 1) % 3]}")
            return [tri[tri_idx0], tri[tri_idx1], tri[(tri_idx0 + 1) % 3]]  # 方向相反，需要调整

            #return tri  # 方向相同，不需要调整
        else:
           # print("mei222222")
            return tri



def bfs_fix_orientation(start_elem, surfaces):
    queue = deque([start_elem])
    visited = set()
    visited.add(tuple(start_elem))
    elem_nodes = []
    node_coords_dict = {}

    # 获取所有表面的元素和节点坐标
    for sur in surfaces:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(sur[0], abs(sur[1]))
        elem_nodes.extend(np.array(elemNodeTag[0]).reshape((-1, 3)))
        tags, coord, param = gmsh.model.mesh.getNodes(2, abs(sur[1]), True)
        for tag, coord in zip(tags, coord.reshape((-1, 3))):
            node_coords_dict[tag] = coord

    # 将元素节点转换为列表形式
    elem_nodes = [list(tri) for tri in elem_nodes]
    corrected_elements = []

    while queue:
       #print("queue:",queue)
        current = queue.popleft()
        corrected_elements.append(current)
        for i, neighbor in enumerate(elem_nodes):
            if tuple(neighbor) not in visited:
                shared_edge = get_shared_edge(current, neighbor)

                if shared_edge:
                    corrected_neighbor = adjust_triangle_orientation(current, neighbor, shared_edge)
                    queue.append(corrected_neighbor)
                    visited.add(tuple(corrected_neighbor))
                    elem_nodes[i] = corrected_neighbor

    return corrected_elements

def check_and_fix_orientation(e, finterface, felement, fzone, finter):
    surface = gmsh.model.getBoundary([e])
    max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center = find_max_x(surface)
    inward = determine_orientation(max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center)
    logging.info(f"max: {max_x_elem}")
    if inward:
        max_x_elem = max_x_elem[::-1]

    corrected_elements = bfs_fix_orientation(max_x_elem, surface)

    num_nodes = 3
    for sur in surface:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(sur[0], abs(sur[1]))
        tags, coord, param = gmsh.model.mesh.getNodes(2, abs(sur[1]), True)
        n = gmsh.model.getEntityName(sur[0], abs(sur[1]))

        if n == "interface":
            finterface.write(str(abs(sur[1])) + "\n")
            finterface.write(str(e[1]) + "\n")
            up, down = gmsh.model.getAdjacencies(sur[0], abs(sur[1]))
            if len(up) > 1:
                number = sum(len(i) for i in elemTag)
                finterface.write(str(number) + "\n")

            for tag, nodes in zip(elemTag[0], elemNodeTag[0].reshape((-1, num_nodes))):
                corrected_elem = next((ce for ce in corrected_elements if set(nodes) == set(ce)), None)
                if corrected_elem is not None:
                    nodes_str = ' '.join(str(i) for i in nodes)
                    finterface.write(str(tag) + " " + str(nodes_str) + " " + str(abs(sur[1])) + "\n")
                else:
                    nodes_str = ' '.join(str(i) for i in nodes)
                    finterface.write(str(tag) + " " + str(nodes_str) + " " + str(abs(sur[1])) + "\n")
        else:
            for tag, nodes in zip(elemTag[0], elemNodeTag[0].reshape((-1, num_nodes))):
                corrected_elem = next((ce for ce in corrected_elements if set(nodes) == set(ce)), None)
                if corrected_elem is not None:
                    nodes_str = ' '.join(str(i) for i in nodes)
                    felement.write(str(tag) + " " + str(nodes_str) + " " + str(abs(sur[1])) + " " + str(abs(e[1])) + "\n")
                else:
                    nodes_str = ' '.join(str(i) for i in nodes)
                    felement.write(str(tag) + " " + str(nodes_str) + " " + str(abs(sur[1])) + " " + str(abs(e[1])) + "\n")

    #fzone.write(str(len(surface)) + "\n")

def main():
   setup_logging()
   gmsh.initialize()

   # 创建一个名为 "model_name" 的模型
   gmsh.model.add("model_name")

   # Load a STEP file (using `importShapes' instead of `merge' allows to directly
   # retrieve the tags of the highest dimensional imported entities):
   path = os.path.dirname(os.path.abspath(__file__))

   # 【修改处】: 使用 importShapes 导入外部几何文件，而非创建 Box
   # 请将 'model_name.step' 替换为你实际的几何文件名 (支持 .step, .stp, .brep 等)
   geometry_file = os.path.join(path, 'tsv.stp')

   if os.path.exists(geometry_file):
       # importShapes 会读取文件并返回生成的实体标签
       gmsh.model.occ.importShapes(geometry_file)
   else:
       print(f"Error: Geometry file '{geometry_file}' not found.")
       # 为了防止后续代码报错，这里可以选择退出，或者你可以保留下面的测试用 Box 代码作为 fallback
       # return
       gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
       gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)

   gmsh.model.occ.synchronize()

   # 获取所有实体
   entities = gmsh.model.getEntities(dim=3)

   # 遍历所有面
   print(str(len(entities)))

    # 遍历所有实体进行分割
   for i in range(len(entities)):
          rest_entities = entities[i+1:]
          if rest_entities:
              gmsh.model.occ.fragment([entities[i]], rest_entities)
   gmsh.model.occ.synchronize()
   if '-nopopup' not in sys.argv:
       gmsh.fltk.run()
       #以上是注释内容

       # gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 30)
       # gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
   #gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
   gmsh.option.setNumber("Mesh.MeshSizeMin", 200)
   gmsh.option.setNumber("Mesh.MeshSizeMax", 200)
   gmsh.model.mesh.generate(2)
   gmsh.model.occ.synchronize()
   if '-nopopup' not in sys.argv:
       gmsh.fltk.run()


   nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

   # Ensure out directory exists
   os.makedirs('out', exist_ok=True)

   with open(os.path.join('out', 'nodes.txt'), 'w') as fnode:
        fnode.write(str(len(nodeTags)) + "\n")
        for tag, xyz_e in zip(nodeTags, nodeCoords):
            xyz_e = ' '.join(str(i) for i in xyz_e)
            fnode.write(str(tag) + ' ' + str(xyz_e) + "\n")

   entities = gmsh.model.getEntities(3)
   zone = sum(len(e) for e in entities)
   numElem = 0
   boundary = 0
   i = 0
   with open(os.path.join('out', 'zone.txt'), 'w') as fzone, \
         open(os.path.join('out', 'inter.txt'), 'w') as finter, \
         open(os.path.join('out', 'interface.txt'), 'w') as finterface, \
         open(os.path.join('out', 'elements.txt')', 'w') as finterface, \
         open('elements.txt', 'w') as felement:
        fzone.write(str(int((zone) / 2)) + "\n")
        for e in entities:
           dim = e[0]
           tag = e[1]
           surface = gmsh.model.getBoundary([e])
           Elem = 0
           for sur in surface:
               dim = sur[0]
               tag = sur[1]
               elemTypee, elemTagg, elemNodeTagg = gmsh.model.mesh.getElements(sur[0], abs(sur[1]))
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
           elemTypee, elemTagg, elemNodeTagg = gmsh.model.mesh.getElements(sur[0], abs(sur[1]))
           up, down = gmsh.model.getAdjacencies(dim, tag)
           if len(up) > 1:
               finter.write(str(abs(sur[1])) + "\n")
        finterface.write(str(i) + "\n")
        felement.write(str(numElem - int(boundary / 2)) + "\n")
        felement.write(str(int(boundary / 2)) + "\n")
        surface = gmsh.model.getBoundary([e])
        # max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center = find_max_x(surface)
        # inward = determine_orientation(max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center)
        # print("max:",max_x_elem)
        # if inward:
        #     max_x_elem = max_x_elem[::-1]

        # corrected_elements = bfs_fix_orientation(max_x_elem, surface)


        for e in gmsh.model.getEntities(3):
            check_and_fix_orientation(e, finterface, felement, fzone, finter)

   gmsh.finalize()

if __name__ == "__main__":
    main()
