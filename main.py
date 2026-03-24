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
            gmsh.model.occ.importShapes(geometry_file)
        except Exception as e:
            print(f"Error importing geometry: {e}")
            logging.error(f"Error importing geometry: {e}")
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

    print_duration("Fixing Orientation & Writing Elements")

    # Stop logging thread
    stop_logging.set()
    log_thread.join()

    # Process any final logs
    for msg in gmsh.logger.get():
        logging.info(f"Gmsh: {msg}")
    gmsh.logger.stop()

    print("Process completed successfully. Check 'out/' for results.")

    # Apply visualization export
    visual_dir = os.path.join('out', 'visual')
    export_visualization(args.format, visual_dir)

    print_duration("Visualization Export")
    print_duration("Total Process")

    gmsh.finalize()


if __name__ == "__main__":
    main()
