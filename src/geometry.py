import numpy as np
import gmsh

def find_max_x(surfaces):
    """
    Finds the element with the maximum X coordinate in a given surface.
    Returns the element, its coordinates, normal vector, and center point.
    """
    max_x_coordinate = -np.inf
    max_x_elem = None
    max_x_elem_coords = None
    max_x_elem_normal = None
    max_x_elem_center = None

    for sur in surfaces:
        elemType, elemTag, elemNodeTag = gmsh.model.mesh.getElements(
            sur[0], abs(sur[1]))

        # Check if elemNodeTag is empty or properly structured
        if not elemNodeTag or len(elemNodeTag) == 0 or len(elemNodeTag[0]) == 0:
            continue

        tags, coord, param = gmsh.model.mesh.getNodes(2, abs(sur[1]), True)
        num_nodes = 3
        node_coords_dict = {
            tag: coord for tag, coord in zip(
                tags, np.array(coord).reshape(
                    (-1, 3)))}

        # Reshape element connectivities
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

                norm = np.linalg.norm(max_x_elem_normal)
                if norm != 0:
                    max_x_elem_normal = max_x_elem_normal / norm  # Normalize the vector

                # Calculate center point
                max_x_elem_center = np.mean(max_x_elem_coords, axis=0)

    if max_x_elem is None or max_x_elem_center is None:
        return None, None, None, None, None

    # Offset point slightly outside in X direction
    max_x_coordinate += 10
    max_x_point = np.array([max_x_coordinate, max_x_elem_center[1], max_x_elem_center[2]])

    return max_x_point, max_x_elem, max_x_elem_coords, max_x_elem_normal, max_x_elem_center


def determine_orientation(
        max_x_point,
        max_x_elem,
        max_x_elem_coords,
        max_x_elem_normal,
        max_x_elem_center):
    """
    Determines if the current normal points inward or outward relative to a reference point.
    """
    direction_vector = max_x_point - max_x_elem_center
    return np.dot(direction_vector, max_x_elem_normal) < 0


def get_shared_edge(tri1, tri2):
    """
    Finds the shared edge (2 nodes) between two triangles.
    Returns the edge tuple if found, otherwise None.
    """
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
    """
    Adjusts the orientation of 'tri' to be consistent with 'reference_tri' across 'shared_edge'.
    """
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
