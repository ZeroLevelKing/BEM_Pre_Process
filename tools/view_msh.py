import argparse
import os
import sys
from collections import Counter
from math import sqrt

try:
    import gmsh
except ImportError as exc:
    print("Error: Python package 'gmsh' is not installed in current environment.")
    print("Hint: conda run -n bem python -m pip install gmsh")
    raise SystemExit(1) from exc


def summarize_mesh() -> None:
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    print(f"Nodes: {len(node_tags)}")

    dim_entities = {dim: gmsh.model.getEntities(dim) for dim in (0, 1, 2, 3)}
    print(
        "Entities: "
        f"points={len(dim_entities[0])}, "
        f"curves={len(dim_entities[1])}, "
        f"surfaces={len(dim_entities[2])}, "
        f"volumes={len(dim_entities[3])}"
    )

    type_counter = Counter()
    total_elements = 0
    for dim in (0, 1, 2, 3):
        for _, entity_tag in dim_entities[dim]:
            element_types, element_tags, _ = gmsh.model.mesh.getElements(dim, entity_tag)
            for elem_type, tags in zip(element_types, element_tags):
                count = len(tags)
                total_elements += count
                elem_name, *_ = gmsh.model.mesh.getElementProperties(elem_type)
                type_counter[elem_name] += count

    print(f"Elements: {total_elements}")
    if type_counter:
        print("Element type breakdown:")
        for name, count in sorted(type_counter.items()):
            print(f"  - {name}: {count}")


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(v):
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _iter_surface_element_normals(node_map):
    for dim in (0, 1, 2, 3):
        for _, entity_tag in gmsh.model.getEntities(dim):
            element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(dim, entity_tag)
            for elem_type, flat_elem_tags, flat_nodes in zip(element_types, element_tags, element_node_tags):
                _, elem_dim, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
                if elem_dim != 2 or num_nodes < 3:
                    continue

                for i in range(0, len(flat_nodes), num_nodes):
                    elem_index = i // num_nodes
                    elem_tag = flat_elem_tags[elem_index]
                    elem_nodes = flat_nodes[i:i + num_nodes]
                    p1 = node_map.get(elem_nodes[0])
                    p2 = node_map.get(elem_nodes[1])
                    p3 = node_map.get(elem_nodes[2])
                    if p1 is None or p2 is None or p3 is None:
                        continue

                    v1 = _vec_sub(p2, p1)
                    v2 = _vec_sub(p3, p1)
                    n = _vec_cross(v1, v2)
                    mag = _vec_norm(n)
                    center = (
                        (p1[0] + p2[0] + p3[0]) / 3.0,
                        (p1[1] + p2[1] + p3[1]) / 3.0,
                        (p1[2] + p2[2] + p3[2]) / 3.0,
                    )

                    yield elem_tag, center, n, mag


def create_normal_vector_view(normal_scale: float = 0.02, max_vectors: int = 0) -> None:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = list(node_coords)
    node_map = {}
    for i, tag in enumerate(node_tags):
        idx = 3 * i
        node_map[tag] = (coords[idx], coords[idx + 1], coords[idx + 2])

    data = []
    degenerate = 0
    count = 0
    seen = set()

    for elem_tag, center, n, mag in _iter_surface_element_normals(node_map):
        if elem_tag in seen:
            continue
        seen.add(elem_tag)

        if mag < 1e-14:
            degenerate += 1
            continue

        inv = 1.0 / mag
        nx = n[0] * inv
        ny = n[1] * inv
        nz = n[2] * inv
        data.extend([center[0], center[1], center[2], nx, ny, nz])
        count += 1

        if max_vectors > 0 and count >= max_vectors:
            break

    if count == 0:
        print("No valid surface element normals were generated for display.")
        return

    view_tag = gmsh.view.add("ElementNormals")
    gmsh.view.addListData(view_tag, "VP", count, data)

    # Force arrow rendering to green.
    for color_name in ("Points", "Lines", "Triangles", "Vectors", "Intervals"):
        try:
            gmsh.view.option.setColor(view_tag, color_name, 0, 255, 0, 255)
        except Exception:
            pass

    gmsh.view.option.setNumber(view_tag, "ShowScale", 0)
    gmsh.view.option.setNumber(view_tag, "Visible", 1)
    print(
        f"Created normal vector view: {count} arrows, "
        f"degenerate skipped: {degenerate}, arrow length: 1"
    )


def configure_gui_style(surface_color: bool) -> None:
    def set_color(name: str, r: int, g: int, b: int, a: int = 255) -> None:
        try:
            gmsh.option.setColor(name, r, g, b, a)
        except Exception:
            pass

    # Use a fixed monochrome palette instead of entity rainbow colors.
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)
    set_color("Mesh.Color.Lines", 0, 0, 0)
    set_color("Mesh.Color.Points", 0, 0, 0)

    if surface_color:
        # Show gray shaded faces with black edges.
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
        gmsh.option.setNumber("Mesh.Lines", 1)
        gmsh.option.setNumber("Mesh.Points", 0)
        set_color("Mesh.Color.Triangles", 160, 160, 160)
        set_color("Mesh.Color.Quadrangles", 160, 160, 160)
    else:
        # Wireframe-like mode.
        gmsh.option.setNumber("Mesh.SurfaceFaces", 0)
        gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
        gmsh.option.setNumber("Mesh.Lines", 1)
        gmsh.option.setNumber("Mesh.Points", 0)


def report_computed_normals(sample: int = 5) -> None:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = list(node_coords)
    node_map = {}
    for i, tag in enumerate(node_tags):
        idx = 3 * i
        node_map[tag] = (coords[idx], coords[idx + 1], coords[idx + 2])

    total_surface_elems = 0
    degenerate = 0
    samples = []
    seen = set()

    for elem_tag, _, n, mag in _iter_surface_element_normals(node_map):
        if elem_tag in seen:
            continue
        seen.add(elem_tag)

        total_surface_elems += 1
        if mag < 1e-14:
            degenerate += 1
            continue

        if len(samples) < sample:
            inv = 1.0 / mag
            samples.append((n[0] * inv, n[1] * inv, n[2] * inv))

    print("Computed normals (from element node ordering):")
    print(f"  - Surface elements checked: {total_surface_elems}")
    print(f"  - Degenerate elements (|n|~0): {degenerate}")
    if samples:
        print("  - Sample unit normals:")
        for i, n in enumerate(samples, start=1):
            print(f"    {i}: ({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="View and inspect a .msh file with Gmsh")
    parser.add_argument("msh_file", help="Path to .msh file")
    normals_group = parser.add_mutually_exclusive_group()
    normals_group.add_argument(
        "--normals",
        action="store_true",
        help="Enable normal indicator display in Gmsh GUI",
    )
    normals_group.add_argument(
        "--no-normals",
        action="store_true",
        help="Disable normal indicator display in Gmsh GUI",
    )
    parser.add_argument(
        "--normal-report",
        action="store_true",
        help="Compute normals from element node ordering and print a report",
    )
    parser.add_argument(
        "--normal-sample",
        type=int,
        default=5,
        help="How many sample computed normals to print in --normal-report (default: 5)",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=0.02,
        help="Deprecated: ignored, normals are always unit length (1)",
    )
    parser.add_argument(
        "--normal-max",
        type=int,
        default=0,
        help="Maximum number of normal arrows to draw (0 means all)",
    )
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument(
        "--surface-color",
        action="store_true",
        help="Show colored mesh faces in GUI (default)",
    )
    style_group.add_argument(
        "--wireframe",
        action="store_true",
        help="Show wireframe-only mesh in GUI",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open interactive Gmsh GUI; print summary only",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip summary output")
    args = parser.parse_args()

    msh_file = os.path.abspath(args.msh_file)
    if not os.path.exists(msh_file):
        print(f"Error: file not found: {msh_file}")
        raise SystemExit(1)

    gmsh.initialize()
    try:
        gmsh.open(msh_file)
        print(f"Loaded: {msh_file}")

        if not args.no_summary:
            summarize_mesh()

        if args.normal_report:
            report_computed_normals(max(0, args.normal_sample))

        if not args.no_gui:
            use_surface_color = not args.wireframe
            if args.surface_color:
                use_surface_color = True
            configure_gui_style(use_surface_color)

            if args.normals:
                create_normal_vector_view(args.normal_scale, max(0, args.normal_max))
            elif args.no_normals:
                gmsh.option.setNumber("Mesh.Normals", 0)
                gmsh.option.setNumber("Geometry.Normals", 0)

            print("Opening Gmsh GUI. Close the window to exit.")
            gmsh.fltk.run()
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
