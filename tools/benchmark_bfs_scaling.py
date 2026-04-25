import argparse
import csv
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import time

import gmsh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.mesh_processing import run_bfs_orientation_pass
from src.logger import setup_logging


def flush_gmsh_logs():
    try:
        for msg in gmsh.logger.get():
            logging.info(f"Gmsh: {msg}")
    except Exception:
        pass


def parse_targets(value: str):
    targets = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        targets.append(int(item))
    if not targets:
        raise ValueError("No valid targets were provided.")
    return targets


def mesh_face_count():
    _, elem_tags, _ = gmsh.model.mesh.getElements(2)
    return sum(len(tags) for tags in elem_tags)


def import_and_mesh(geometry_file: str, mesh_size: float, model_name: str):
    gmsh.clear()
    flush_gmsh_logs()
    gmsh.model.add(model_name)
    gmsh.model.occ.importShapes(geometry_file)
    gmsh.model.occ.synchronize()

    entities = gmsh.model.getEntities(dim=3)
    if len(entities) > 1:
        gmsh.model.occ.fragment(entities, [])
        gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    flush_gmsh_logs()

    return mesh_face_count()


def find_size_for_target(
    geometry_file: str,
    target_faces: int,
    size_low: float,
    size_high: float,
    max_iter: int,
    rel_tol: float,
):
    model_name = f"tune_{target_faces}"
    history = []

    low_faces = import_and_mesh(geometry_file, size_low, model_name)
    history.append({
        "target_faces": target_faces,
        "phase": "bound",
        "iter_index": -2,
        "mesh_size": size_low,
        "faces": low_faces,
        "abs_error": abs(low_faces - target_faces),
        "rel_error": abs(low_faces - target_faces) / float(target_faces),
        "note": "size_low",
    })

    high_faces = import_and_mesh(geometry_file, size_high, model_name)
    history.append({
        "target_faces": target_faces,
        "phase": "bound",
        "iter_index": -1,
        "mesh_size": size_high,
        "faces": high_faces,
        "abs_error": abs(high_faces - target_faces),
        "rel_error": abs(high_faces - target_faces) / float(target_faces),
        "note": "size_high",
    })

    if low_faces < high_faces:
        raise RuntimeError(
            "Unexpected monotonicity: smaller size produced fewer faces. "
            "Try adjusting size_low/size_high bounds."
        )

    if target_faces > low_faces:
        return size_low, low_faces, "target_above_max_density", history

    if target_faces < high_faces:
        return size_high, high_faces, "target_below_min_density", history

    best_size = size_low
    best_faces = low_faces
    best_err = abs(low_faces - target_faces)

    lo = size_low
    hi = size_high

    for i in range(max_iter):
        mid = (lo + hi) * 0.5
        faces = import_and_mesh(geometry_file, mid, model_name)

        err = abs(faces - target_faces)
        rel_err = err / float(target_faces)
        history.append({
            "target_faces": target_faces,
            "phase": "binary",
            "iter_index": i,
            "mesh_size": mid,
            "faces": faces,
            "abs_error": err,
            "rel_error": rel_err,
            "note": "",
        })

        if err < best_err:
            best_err = err
            best_size = mid
            best_faces = faces
        if rel_err <= rel_tol:
            return best_size, best_faces, "ok", history

        if faces > target_faces:
            lo = mid
        else:
            hi = mid

    return best_size, best_faces, "max_iter_reached", history


def run_case_internal(geometry_file: str, target_faces: int, mesh_size: float):
    model_name = f"case_{target_faces}"
    t0 = time.perf_counter()
    actual_faces = import_and_mesh(geometry_file, mesh_size, model_name)
    bfs_stats = run_bfs_orientation_pass(gmsh.model.getEntities(3))
    total_seconds = time.perf_counter() - t0

    return {
        "target_faces": target_faces,
        "actual_faces": actual_faces,
        "mesh_size": mesh_size,
        "total_seconds": total_seconds,
        "bfs_seconds": bfs_stats["bfs_seconds"],
        "bfs_ratio": (bfs_stats["bfs_seconds"] / total_seconds) if total_seconds > 0 else 0.0,
        "volume_entities": bfs_stats["volume_entities"],
        "skipped_entities": bfs_stats["skipped_entities"],
        "corrected_elements": bfs_stats["corrected_elements"],
        "runner": "internal",
        "bfs_source": "pure_bfs",
    }


def _parse_main_output(output_text: str):
    faces_match = re.findall(r"Total mesh elements:\s*(\d+)", output_text)
    bfs_match = re.findall(r"\[.*Fixing Orientation & Writing Elements\]\s*took\s*([0-9.]+)s", output_text)

    actual_faces = int(faces_match[-1]) if faces_match else -1
    bfs_seconds = float(bfs_match[-1]) if bfs_match else float("nan")
    return actual_faces, bfs_seconds


def run_case_via_main(geometry_file: str, target_faces: int, mesh_size: float, show_gmsh_terminal: bool):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_py = os.path.join(root_dir, "main.py")

    cmd = [
        sys.executable,
        main_py,
        "--input",
        geometry_file,
        "--size_min",
        str(mesh_size),
        "--size_max",
        str(mesh_size),
        "--format",
        "msh",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=root_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    total_seconds = time.perf_counter() - t0

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if show_gmsh_terminal:
        print(combined)

    if proc.returncode != 0:
        raise RuntimeError(
            f"main.py runner failed for target {target_faces} with return code {proc.returncode}.\n"
            f"Output:\n{combined[-4000:]}"
        )

    actual_faces, bfs_seconds = _parse_main_output(combined)
    bfs_ratio = (bfs_seconds / total_seconds) if (total_seconds > 0 and bfs_seconds == bfs_seconds) else float("nan")

    return {
        "target_faces": target_faces,
        "actual_faces": actual_faces,
        "mesh_size": mesh_size,
        "total_seconds": total_seconds,
        "bfs_seconds": bfs_seconds,
        "bfs_ratio": bfs_ratio,
        "volume_entities": -1,
        "skipped_entities": -1,
        "corrected_elements": -1,
        "runner": "main",
        "bfs_source": "main_stage_with_io",
    }


def main():
    parser = argparse.ArgumentParser(description="Run BFS orientation scaling benchmarks by target face counts")
    parser.add_argument("--input", required=True, help="Input geometry file path (.iges/.igs/.step/.stp/.brep)")
    parser.add_argument(
        "--targets",
        default="50000,100000,500000,1000000,2500000",
        help="Comma-separated target face counts",
    )
    parser.add_argument("--size-low", type=float, default=0.1, help="Lower mesh size bound (smaller -> more faces)")
    parser.add_argument("--size-high", type=float, default=10.0, help="Upper mesh size bound (larger -> fewer faces)")
    parser.add_argument("--max-iter", type=int, default=14, help="Max binary-search iterations per target")
    parser.add_argument("--rel-tol", type=float, default=0.05, help="Relative tolerance for face count fitting")
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Gmsh thread count for internal runner (default: all CPU cores)",
    )
    parser.add_argument(
        "--runner",
        choices=("internal", "main"),
        default="internal",
        help="Timed-run backend: internal(pure BFS timing) or main(call main.py)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("out", "benchmarks", "bfs_scaling.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--output-search",
        default=os.path.join("out", "benchmarks", "bfs_scaling_search.csv"),
        help="CSV output path for all binary-search probe points",
    )
    parser.add_argument(
        "--show-gmsh-terminal",
        action="store_true",
        help="Show raw Gmsh terminal logs (default: intercept and write to out/log/process.log)",
    )
    args = parser.parse_args()

    geometry_file = os.path.abspath(args.input)
    if not os.path.exists(geometry_file):
        raise FileNotFoundError(f"Input geometry file not found: {geometry_file}")

    targets = parse_targets(args.targets)
    output_path = os.path.abspath(args.output)
    output_search_path = os.path.abspath(args.output_search)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_search_path), exist_ok=True)

    setup_logging()
    gmsh.initialize([sys.argv[0]])
    gmsh.option.setNumber("General.Terminal", 1 if args.show_gmsh_terminal else 0)
    gmsh.option.setNumber("General.NumThreads", max(1, args.threads))
    gmsh.logger.start()

    rows = []
    search_rows = []
    try:
        print("Starting BFS scaling benchmark...")
        print(f"Input: {geometry_file}")
        print(f"Targets: {targets}")

        for target in targets:
            print(f"\n[Target {target}] fitting mesh size...")
            size, fitted_faces, fit_status, fit_history = find_size_for_target(
                geometry_file=geometry_file,
                target_faces=target,
                size_low=args.size_low,
                size_high=args.size_high,
                max_iter=args.max_iter,
                rel_tol=args.rel_tol,
            )
            search_rows.extend(fit_history)
            print(
                f"[Target {target}] fit result: size={size:.6g}, "
                f"faces={fitted_faces}, status={fit_status}"
            )

            print(f"[Target {target}] running full timed case...")
            if args.runner == "main":
                row = run_case_via_main(geometry_file, target, size, args.show_gmsh_terminal)
            else:
                row = run_case_internal(geometry_file, target, size)
            row["fit_status"] = fit_status
            rows.append(row)
            print(
                f"[Target {target}] total={row['total_seconds']:.4f}s, "
                f"bfs={row['bfs_seconds']:.4f}s, faces={row['actual_faces']}"
            )

    finally:
        flush_gmsh_logs()
        gmsh.logger.stop()
        gmsh.finalize()

    fieldnames = [
        "target_faces",
        "actual_faces",
        "mesh_size",
        "fit_status",
        "total_seconds",
        "bfs_seconds",
        "bfs_ratio",
        "runner",
        "bfs_source",
        "volume_entities",
        "skipped_entities",
        "corrected_elements",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    search_fieldnames = [
        "target_faces",
        "phase",
        "iter_index",
        "mesh_size",
        "faces",
        "abs_error",
        "rel_error",
        "note",
    ]
    with open(output_search_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=search_fieldnames)
        writer.writeheader()
        writer.writerows(search_rows)

    print(f"\nBenchmark finished. CSV written to: {output_path}")
    print(f"Binary-search history CSV written to: {output_search_path}")


if __name__ == "__main__":
    main()
