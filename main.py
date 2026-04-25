import gmsh
import numpy as np
import sys
import os
import argparse
import logging
import threading
import time
import multiprocessing

from src.logger import setup_logging, monitor_gmsh_logs
from src.geometry import find_max_x, determine_orientation
from src.mesh_processing import check_and_fix_orientation
from src.export import export_visualization

def main():
    # Parse command line arguments for mesh sizing and input file
    parser = argparse.ArgumentParser(description="Gmsh Mesh Generator")
    parser.add_argument('--input', type=str, default=None, help='Input geometry file path (STEP, IGES, BREP)')
    parser.add_argument('--size_min', type=float, default=1.0, help='Minimum mesh element size')
    parser.add_argument('--size_max', type=float, default=3.0, help='Maximum mesh element size')
    parser.add_argument('--format', type=str, default='msh', help='Export format (vtk, msh, cgns, obj, all). Default: msh')

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

    try:
        work_dir = os.getcwd()
        geometry_files = []

        if args.input:
            if os.path.exists(args.input):
                geometry_files = [os.path.abspath(args.input)]
            else:
                print(f"Error: Specified input file '{args.input}' not found.")
                return
        else:
            for file in sorted(os.listdir(work_dir)):
                if file.lower().endswith(('.igs', '.iges')):
                    geometry_files.append(os.path.join(work_dir, file))

        if not geometry_files:
            print("Error: No IGES files found in working directory.")
            return

        # Ensure output directories exist
        os.makedirs('out', exist_ok=True)
        visual_dir = os.path.join('out', 'visual')
        os.makedirs(visual_dir, exist_ok=True)
        data_root_dir = os.path.join('out', 'data')
        os.makedirs(data_root_dir, exist_ok=True)

        for geometry_file in geometry_files:
            model_name = os.path.splitext(os.path.basename(geometry_file))[0]
            print(f"\n=== Processing geometry: {os.path.basename(geometry_file)} ===")

            gmsh.clear()
            gmsh.model.add(model_name)

            print(f"Loading geometry file: {geometry_file}")
            try:
                # importShapes supports STEP, IGES, BREP natively via OpenCASCADE
                gmsh.model.occ.importShapes(geometry_file)
            except Exception as e:
                print(f"Error importing geometry '{geometry_file}': {e}")
                logging.error(f"Error importing geometry '{geometry_file}': {e}")
                continue

            gmsh.model.occ.synchronize()
            print("Geometry loaded.")
            print_duration(f"{model_name} Geometry Loading")

            # 获取所有实体
            entities = gmsh.model.getEntities(dim=3)

            print("Processing geometry fragments...")

            if len(entities) > 0:
                gmsh.model.occ.fragment(entities, [])

            gmsh.model.occ.synchronize()
            print_duration(f"{model_name} Geometry Processing")

            print(f"Setting Mesh Size: Min={args.size_min}, Max={args.size_max}")
            gmsh.option.setNumber("Mesh.MeshSizeMin", args.size_min)
            print("Generating mesh...")
            gmsh.option.setNumber("Mesh.MeshSizeMax", args.size_max)

            # Enable Frontal-Delaunay for 2D meshing (Algorithm ID=6) for faster and better quality
            gmsh.option.setNumber("Mesh.Algorithm", 6)

            gmsh.model.mesh.generate(2)
            gmsh.model.occ.synchronize()
            print_duration(f"{model_name} Meshing")

            # Pre-calculation processing time estimation
            # Get all 2D elements to estimate workload
            _, all_elem_tags, _ = gmsh.model.mesh.getElements(2)
            total_2d_elements = sum(len(tags) for tags in all_elem_tags)
            print(f"Total mesh elements: {total_2d_elements}")

            nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

            nodeCoords = np.array(nodeCoords).reshape((-1, 3))

            # Save data files in per-model directory to avoid overwriting
            data_dir = os.path.join(data_root_dir, model_name)
            os.makedirs(data_dir, exist_ok=True)

            print("Writing nodes to file...")

            with open(os.path.join(data_dir, 'nodes.txt'), 'w') as fnode:
                fnode.write(str(len(nodeTags)) + "\n")
                for tag, xyz_e in zip(nodeTags, nodeCoords):
                    xyz_e = ' '.join(str(i) for i in xyz_e)
                    fnode.write(str(tag) + ' ' + str(xyz_e) + "\n")
            print_duration(f"{model_name} Writing Nodes")

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

                # First Pass: Count elements and identify interfaces in one go?
                # The logic below mimics ff.py logic exactly for compatibility

                for e in entities:
                    dim = e[0]
                    tag = e[1]
                    surface = gmsh.model.getBoundary([e])
                    Elem = 0
                    for sur in surface:
                        dim = sur[0]
                        tag = sur[1]
                        # elemTypee here is typo in original, but unused.
                        # We need tags to count them.
                        elemTypee, elemTagg, elemNodeTagg = gmsh.model.mesh.getElements(
                            sur[0], abs(sur[1]))

                        count = sum(len(x) for x in elemTagg)
                        Elem += count
                        numElem += count

                        up, down = gmsh.model.getAdjacencies(sur[0], abs(sur[1]))
                        if len(up) > 1:
                            gmsh.model.setEntityName(2, sur[1], "interface")
                            boundary += count
                            i += 1 # Counting interface surfaces?
                    fzone.write(str(Elem) + "\n")

                finter.write(str(int(i / 2)) + "\n")

                # Second Pass: Write interface files
                surface_entities = gmsh.model.getEntities(2)
                for sur in surface_entities:
                    dim = sur[0]
                    tag = sur[1]
                    # Verify if it's an interface (has >1 adjacency)
                    up, down = gmsh.model.getAdjacencies(dim, tag)
                    if len(up) > 1:
                        finter.write(str(abs(sur[1])) + "\n")

                finterface.write(str(i) + "\n")
                felement.write(str(numElem - int(boundary / 2)) + "\n")
                felement.write(str(int(boundary / 2)) + "\n")

                print("Fixing element orientation and writing output files...")

                for e in gmsh.model.getEntities(3):
                    # Pass file handles to helper
                    check_and_fix_orientation(e, finterface, felement, fzone, finter)

            print_duration(f"{model_name} Fixing Orientation & Writing Elements")

            print("Process completed successfully. Check 'out/' for results.")

            # Export visualization with per-file basename, e.g. xxx.msh
            export_visualization(args.format, visual_dir, model_name)

            print_duration(f"{model_name} Visualization Export")

        print_duration("Total Process")
    finally:
        # Stop logging thread
        stop_logging.set()
        log_thread.join()

        # Process any final logs
        for msg in gmsh.logger.get():
            logging.info(f"Gmsh: {msg}")
        gmsh.logger.stop()
        gmsh.finalize()


if __name__ == "__main__":
    main()
