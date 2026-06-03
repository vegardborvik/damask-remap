#!/usr/bin/env python3
"""
Crash-and-continue remapping for rolling.
Strategy A: Cube reset.
Runs DAMASK via apptainer to match the reference bash script.

Usage:
  python remapping_rolling.py

Set environment variables to override defaults:
  GRID_SIF=~/containers/damask-grid_latest.sif
  OMP_NUM_THREADS=8
  STRATEGY=cube   (or 'favg' for Strategy B)
"""

import os
import tempfile
import shutil
import re
import sys
import subprocess
import glob

import yaml
import numpy as np
import damask

grid = "rolling.geom"
mat = "material"
load = "rolling.load"

# Strategy: 'cube' for Strategy A, 'favg' for Strategy B
STRATEGY = os.environ.get('STRATEGY', 'favg')

# Apptainer settings
GRID_SIF = os.environ.get('GRID_SIF',
           os.path.expanduser('~/containers/damask-grid_latest.sif'))
OMP_THREADS = os.environ.get('OMP_NUM_THREADS', '8')

cwd = os.getcwd()
print(wd := tempfile.mkdtemp())
print(f"Strategy: {STRATEGY}")
print(f"Container: {GRID_SIF}")
print(f"OMP_NUM_THREADS: {OMP_THREADS}")


def run_damask(grid_file, load_file, mat_file):
    """Run DAMASK via apptainer, matching the reference bash script."""

    # Copy all input files into wd so apptainer can see them
    for src in [grid_file, load_file, mat_file]:
        dst = os.path.join(wd, os.path.basename(src))
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copyfile(src, dst)

    # Copy numerics.yaml if it exists
    numerics_src = f"{cwd}/numerics.yaml"
    if os.path.exists(numerics_src):
        shutil.copyfile(numerics_src, f"{wd}/numerics.yaml")

    g_base = os.path.basename(grid_file)
    l_base = os.path.basename(load_file)
    m_base = os.path.basename(mat_file)

    numerics_flag = "--numerics numerics.yaml" if os.path.exists(numerics_src) else ""

    cmd = (
        f'apptainer exec --cleanenv '
        f'--bind "{wd}:/wd" --pwd /wd '
        f'--env OMP_NUM_THREADS={OMP_THREADS} '
        f'--env OPENBLAS_NUM_THREADS=1 '
        f'--env MKL_NUM_THREADS=1 '
        f'--env NUMEXPR_NUM_THREADS=1 '
        f'"{GRID_SIF}" '
        f'DAMASK_grid --geom {g_base} --load {l_base} --material {m_base} '
        f'{numerics_flag}'
    )

    print(f"  Running: apptainer exec ... DAMASK_grid --geom {g_base} ...")

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_converged = None
    failed = False

    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
        sys.stdout.flush()
        inc_match = re.search(r'increment[:\s]+(\d+)', line, re.IGNORECASE)
        if inc_match and 'converged' in line.lower() and 'not' not in line.lower():
            last_converged = int(inc_match.group(1))
        if 'max number of cutbacks' in line.lower():
            failed = True

    process.wait()
    if process.returncode != 0:
        failed = True

    return failed, last_converged


def extract_orientations(result_file, mat_file, current_grid_file, out_mat, out_grid):
    """Extract orientations from last saved increment and create new grid."""
    r = damask.Result(result_file).view(increments=-1)
    O_flat = r.get('O').reshape(-1, 4, order='F')

    with open(mat_file, 'r') as f:
        data = yaml.safe_load(f)

    template = data['material'][0]['constituents'][0].copy()
    homog = data['material'][0]['homogenization']

    new_data = {
        'homogenization': data['homogenization'],
        'phase': data['phase'],
        'material': [{'homogenization': homog,
                       'constituents': [{**template, 'O': O_flat[i].tolist()}]}
                      for i in range(len(O_flat))]
    }

    with open(out_mat, 'w') as f:
        yaml.dump(new_data, f, default_flow_style=None, sort_keys=False)

    orig = damask.GeomGrid.load(f"{cwd}/{grid}.vti")
    cells = orig.cells

    if STRATEGY == 'favg':
        # Strategy B: deform grid using F_avg
        F_avg = np.average(r.place('F'), axis=0)
        current_geom = damask.GeomGrid.load(current_grid_file)
        new_size = F_avg @ current_geom.size
        print(f"  F_avg diag: [{F_avg[0,0]:.4f}, {F_avg[1,1]:.4f}, {F_avg[2,2]:.4f}]")
        print(f"  Old size: {current_geom.size}")
        print(f"  New size: {new_size}")
    else:
        # Strategy A: reset to original cube
        new_size = orig.size

    ids = np.arange(len(O_flat), dtype=int).reshape(cells, order='F')
    damask.GeomGrid(material=ids, size=new_size).save(out_grid)
    print(f"  Remapped {len(O_flat)} orientations ({STRATEGY})")


def make_continuation_load(original_load, remaining_time, output_file):
    """Copy original load case but with adjusted time."""
    with open(original_load, 'r') as f:
        data = yaml.safe_load(f)

    new_N = int(remaining_time / 0.5)

    data['loadstep'][0]['discretization']['t'] = float(remaining_time)
    data['loadstep'][0]['discretization']['N'] = new_N

    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


# Read original load to get total time
with open(f"{cwd}/{load}.yaml", 'r') as f:
    orig_load = yaml.safe_load(f)
total_time = orig_load['loadstep'][0]['discretization']['t']
print(f"Total simulation time: {total_time}s\n")

# Main loop
segment = 0
time_elapsed = 0.0
MAX_SEGMENTS = 50
VTK_INTERVAL = 1

while segment < MAX_SEGMENTS:
    print(f"\n{'='*60}")
    print(f"  Segment {segment} | Elapsed: {time_elapsed:.1f}s / {total_time:.1f}s")
    print(f"{'='*60}")

    # File paths
    if segment == 0:
        g_file = f"{cwd}/{grid}.vti"
        m_file = f"{cwd}/{mat}.yaml"
        l_file = f"{cwd}/{load}.yaml"
    else:
        g_file = f"{cwd}/{grid}_seg{segment}.vti"
        m_file = f"{cwd}/{mat}_seg{segment}.yaml"
        l_file = f"{cwd}/{load}_seg{segment}.yaml"

    # Result file name (DAMASK convention: grid_load_mat.hdf5)
    g_base = os.path.splitext(os.path.basename(g_file))[0]
    l_base = os.path.splitext(os.path.basename(l_file))[0]
    m_base = os.path.splitext(os.path.basename(m_file))[0]
    result_file = f"{wd}/{g_base}_{l_base}_{m_base}.hdf5"

    # Run
    failed, last_inc = run_damask(g_file, l_file, m_file)
    print(f"  Files in wd: {[f for f in os.listdir(wd) if f.endswith('.hdf5')]}")
    print(f"  Expected: {os.path.basename(result_file)}")

    if not failed:
        print(f"\n  Segment {segment} completed — simulation done!")
        shutil.copyfile(result_file, f"{cwd}/result_seg{segment}.hdf5")

        # Selective VTK export
        r = damask.Result(result_file)
        all_incs = r.increments
        selected = [all_incs[i] for i in range(0, len(all_incs), VTK_INTERVAL)]
        if all_incs[-1] not in selected:
            selected.append(all_incs[-1])
        r_sel = r.view(increments=selected)
        r_sel.add_IPF_color([0,0,1])
        r_sel.export_VTK(target_dir=cwd)
        break

    # Crashed
    if last_inc is None or last_inc < 3:
        print(f"  Crashed too early (inc {last_inc}). Aborting.")
        break
    if not os.path.exists(result_file):
        print(f"  Result file not found: {result_file}")
        print(f"  Files in wd: {os.listdir(wd)}")
        break

    # Use last SAVED increment
    r = damask.Result(result_file)
    last_saved_time = r.times[-1]
    last_saved_inc = r.increments[-1]
    time_elapsed += last_saved_time
    remaining = total_time - time_elapsed

    print(f"\n  Crashed at increment {last_inc}")
    print(f"  Last saved: {last_saved_inc} (t = {last_saved_time:.1f}s)")
    print(f"  Total elapsed: {time_elapsed:.1f}s")
    print(f"  Remaining: {remaining:.1f}s")

    # Save result
    shutil.copyfile(result_file, f"{cwd}/result_seg{segment}.hdf5")

    # Selective VTK export
    all_incs = r.increments
    selected = [all_incs[i] for i in range(0, len(all_incs), VTK_INTERVAL)]
    if all_incs[-1] not in selected:
        selected.append(all_incs[-1])
    r_sel = r.view(increments=selected)
    r_sel.add_IPF_color([0,0,1])
    r_sel.export_VTK(target_dir=cwd)

    if remaining < 1.0:
        print("  Close enough — done!")
        break

    # Remap
    print(f"\n  Remapping for segment {segment + 1}...")
    extract_orientations(result_file, m_file, g_file,
                         f"{cwd}/{mat}_seg{segment+1}.yaml",
                         f"{cwd}/{grid}_seg{segment+1}.vti")

    # Create continuation load case
    make_continuation_load(f"{cwd}/{load}.yaml", remaining,
                           f"{cwd}/{load}_seg{segment+1}.yaml")

    segment += 1

# Copy any remaining HDF5 files from temp
for f in glob.glob(f'{wd}/*.hdf5'):
    dst = f'{cwd}/{os.path.basename(f)}'
    if not os.path.exists(dst):
        shutil.copyfile(f, dst)

print(f"\nDone. Total segments: {segment + 1}")
print(f"Strategy: {STRATEGY}")
print(f"Time elapsed: {time_elapsed:.1f}s / {total_time:.1f}s")
print(f"Temp dir: {wd}")
