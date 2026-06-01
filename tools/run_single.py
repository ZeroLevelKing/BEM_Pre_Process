"""Single-point benchmark runner.

Runs one mesh+orientation pass at a directly specified mesh size (no binary search)
and appends results to the same CSV files used by benchmark_bfs_scaling.py.

Use this when mesh generation alone takes tens of minutes — binary search would be
prohibitively expensive.

Example:
    python tools/run_single.py --input tsv10_10.iges --size 0.49
    python tools/run_single.py --input tsv10_10.iges --size 0.34 --label "100M-estimate"
"""

import argparse
import csv
import multiprocessing
import os
import sys
import time

import gmsh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_bfs_scaling import (
    flush_gmsh_logs,
    import_and_mesh,
    run_case_via_main,
    MemoryMonitor,
)
from src.mesh_processing import run_bfs_orientation_pass
from src.logger import setup_logging


def _resolve_fieldnames(output_path: str, default_fieldnames: list) -> list:
    """Return *default_fieldnames* if the file is new; otherwise read and
    return the existing header so appended rows stay column-aligned with
    older CSV versions."""
    if not os.path.exists(output_path):
        return list(default_fieldnames)
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing = next(reader, [])
    if not existing:
        return list(default_fieldnames)
    # Remove any trailing BOM / whitespace noise
    existing = [c.strip().lstrip("﻿") for c in existing]
    return existing


def append_result_row(output_path: str, row: dict, fieldnames: list):
    """Append a single result row to the main scaling CSV.

    Detects the existing header so that rows written by newer code are
    compatible with CSV files produced by older versions of the benchmark
    scripts (missing columns are silently dropped, new columns appear only
    when the header already contains them).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    actual_fieldnames = _resolve_fieldnames(output_path, fieldnames)
    write_header = actual_fieldnames == fieldnames and not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=actual_fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_search_row(output_search_path: str, row: dict, fieldnames: list):
    """Append a single search-history row to the search CSV."""
    os.makedirs(os.path.dirname(output_search_path), exist_ok=True)
    actual_fieldnames = _resolve_fieldnames(output_search_path, fieldnames)
    write_header = actual_fieldnames == fieldnames and not os.path.exists(output_search_path)
    with open(output_search_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=actual_fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-point BFS scaling benchmark at a specified mesh size"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input geometry file path (.iges/.igs/.step/.stp/.brep)",
    )
    parser.add_argument(
        "--size", type=float, required=True,
        help="Mesh size to use (set as both MeshSizeMin and MeshSizeMax)",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional label recorded as target_faces in CSV for identification",
    )
    parser.add_argument(
        "--runner", choices=("internal", "main"), default="internal",
        help="Timed-run backend: internal (pure BFS timing) or main (call main.py)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("out", "benchmarks", "bfs_scaling.csv"),
        help="CSV output path for result row",
    )
    parser.add_argument(
        "--output-search",
        default=os.path.join("out", "benchmarks", "bfs_scaling_search.csv"),
        help="CSV output path for search-history row",
    )
    parser.add_argument(
        "--threads", type=int, default=multiprocessing.cpu_count(),
        help="Gmsh thread count (default: all CPU cores)",
    )
    parser.add_argument(
        "--show-gmsh-terminal", action="store_true",
        help="Show raw Gmsh terminal logs",
    )
    args = parser.parse_args()

    geometry_file = os.path.abspath(args.input)
    if not os.path.exists(geometry_file):
        raise FileNotFoundError(f"Input geometry file not found: {geometry_file}")

    output_path = os.path.abspath(args.output)
    output_search_path = os.path.abspath(args.output_search)
    label = args.label if args.label is not None else f"size={args.size:.6g}"

    setup_logging()
    gmsh.initialize([sys.argv[0]])
    gmsh.option.setNumber("General.Terminal", 1 if args.show_gmsh_terminal else 0)
    gmsh.option.setNumber("General.NumThreads", max(1, args.threads))
    gmsh.logger.start()

    # Match benchmark_bfs_scaling.py fieldnames so CSVs stay mergeable
    result_fieldnames = [
        "target_faces", "actual_faces", "mesh_size", "fit_status",
        "total_seconds", "bfs_seconds", "bfs_ratio", "peak_rss_mb",
        "runner", "bfs_source",
        "volume_entities", "skipped_entities", "corrected_elements",
    ]
    search_fieldnames = [
        "target_faces", "phase", "iter_index", "mesh_size", "faces",
        "abs_error", "rel_error", "note",
    ]

    try:
        print(f"Single-point benchmark: size={args.size:.6g}, input={geometry_file}")
        print(f"Runner: {args.runner}")

        if args.runner == "main":
            # --- main runner: spawn main.py as subprocess (mesh + orient + I/O) ---
            row = run_case_via_main(geometry_file, label, args.size, args.show_gmsh_terminal)
            actual_faces = row["actual_faces"]
        else:
            # --- internal runner: mesh once, then pure BFS orientation pass ---
            model_name = f"single_{args.size:.6g}"
            monitor = MemoryMonitor()
            monitor.start()
            t0 = time.perf_counter()
            actual_faces = import_and_mesh(geometry_file, args.size, model_name)
            bfs_stats = run_bfs_orientation_pass(gmsh.model.getEntities(3))
            total_seconds = time.perf_counter() - t0
            peak_rss_mb = monitor.stop()
            print(f"Meshed {actual_faces} faces at mesh_size={args.size:.6g}")

            row = {
                "target_faces": label,
                "actual_faces": actual_faces,
                "mesh_size": args.size,
                "total_seconds": total_seconds,
                "bfs_seconds": bfs_stats["bfs_seconds"],
                "bfs_ratio": (
                    bfs_stats["bfs_seconds"] / total_seconds
                    if total_seconds > 0
                    else 0.0
                ),
                "peak_rss_mb": peak_rss_mb,
                "volume_entities": bfs_stats["volume_entities"],
                "skipped_entities": bfs_stats["skipped_entities"],
                "corrected_elements": bfs_stats["corrected_elements"],
                "runner": "internal",
                "bfs_source": "pure_bfs",
            }
        row["fit_status"] = "direct"

        # Step 4: Append result row
        append_result_row(output_path, row, result_fieldnames)
        print(
            f"Result → {output_path}: "
            f"total={row['total_seconds']:.2f}s, bfs={row['bfs_seconds']:.2f}s, "
            f"faces={row['actual_faces']}, peak_rss={row['peak_rss_mb']:.0f}MiB"
        )

        # Step 5: Append search-history row (single point, so one row)
        search_row = {
            "target_faces": label,
            "phase": "direct",
            "iter_index": 0,
            "mesh_size": args.size,
            "faces": actual_faces,
            "abs_error": "",
            "rel_error": "",
            "note": f"runner={args.runner}",
        }
        append_search_row(output_search_path, search_row, search_fieldnames)
        print(f"Search record → {output_search_path}")

        print("Done.")

    finally:
        flush_gmsh_logs()
        gmsh.logger.stop()
        gmsh.finalize()


if __name__ == "__main__":
    main()
