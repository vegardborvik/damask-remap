import os
import tempfile
import shutil
import re
import sys
import subprocess
import logging

import pandas as pd
import yaml
import damask
import numpy as np

extrusion_file = "COSMETEX_SYM_HexMesh_26B_350-45_Coulomb_S-plus0.05_001.pth"

grid = "rolling.geom"
mat = "material"

cwd = os.getcwd()
print(wd := tempfile.mkdtemp())

max_segments = 20
min_inc_per_segment = 5
TARGET_STRAIN_PER_INC = 0.001
MIN_N = 10
MAX_N = 500
VTK_INTERVAL = 100

# ============================================================
#  LOGGING
# ============================================================
log = logging.getLogger('remap')
log.setLevel(logging.DEBUG)

# File handler — detailed log
fh = logging.FileHandler('remap_log.txt', mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
log.addHandler(fh)

# Console handler — summary only
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(ch)

log.info(f"Working dir: {wd}")


def parse_extrusion(pth_file):
    log.info(f"Reading file: {pth_file}")
    df = pd.read_csv(pth_file, sep=r'\s+')

    if df['TIME'].iloc[0] < 0:
        df = df.iloc[::-1].reset_index(drop=True)
        df['TIME'] = df['TIME'] + abs(df['TIME'].iloc[0])

    t_data = df['TIME'].values

    L_data = np.zeros((len(df), 3, 3))

    L_data[:, 0, 0] = df['VEL_GRAD_XX'].values
    L_data[:, 0, 1] = df['VEL_GRAD_XY'].values
    L_data[:, 0, 2] = df['VEL_GRAD_XZ'].values
    L_data[:, 1, 0] = df['VEL_GRAD_YX'].values
    L_data[:, 1, 1] = df['VEL_GRAD_YY'].values
    L_data[:, 1, 2] = df['VEL_GRAD_YZ'].values
    L_data[:, 2, 0] = df['VEL_GRAD_ZX'].values
    L_data[:, 2, 1] = df['VEL_GRAD_ZY'].values
    L_data[:, 2, 2] = df['VEL_GRAD_ZZ'].values

    return L_data, t_data


def create_load_segment(L_data, t_data, output_file, start_loadstep=0, elapsed_time=0.0):
    L_segment = L_data[start_loadstep:]
    t_segment = t_data[start_loadstep:]

    n_steps = len(L_segment)
    t_adjusted = t_segment - elapsed_time

    load_case = {
        'solver': {
            'mechanical': 'spectral_basic'
        },
        'loadstep': []
    }

    total_incs = 0
    n_details = []

    for i in range(n_steps):
        L = L_segment[i]

        if i < n_steps - 1:
            t_step = t_adjusted[i+1] - t_adjusted[i]
        else:
            if i > 0:
                t_step = t_adjusted[i] - t_adjusted[i-1]
            else:
                t_step = 0.001
        if t_step < 1e-10:
            log.debug(f"  Skipping loadstep {start_loadstep + i}: t_step too small")
            continue

        L_norm = np.linalg.norm(L)
        total_strain = L_norm * t_step
        N = int(np.ceil(total_strain / TARGET_STRAIN_PER_INC))
        N = max(MIN_N, min(N, MAX_N))

        total_incs += N
        n_details.append((start_loadstep + i, N, L_norm, t_step))

        loadstep = {
            'boundary_conditions': {
                'mechanical': {
                    'L': L.tolist()
                }
            },
            'discretization': {
                't': float(t_step),
                'N': N
            },
            'f_out': 1
        }

        load_case['loadstep'].append(loadstep)

    with open(output_file, 'w') as f:
        yaml.dump(load_case, f, default_flow_style=None, sort_keys=False)

    log.info(f"  Load case: {len(load_case['loadstep'])} loadsteps, {total_incs} total increments")

    # Log adaptive N details
    log.debug(f"  Adaptive N breakdown (loadstep, N, ||L||, dt):")
    cum_inc = 0
    for ls_idx, n, l_norm, dt in n_details:
        cum_inc += n
        log.debug(f"    LS {ls_idx:4d}: N={n:4d}, ||L||={l_norm:8.2f}, dt={dt:.6e}, cum_inc={cum_inc}")

    return len(load_case['loadstep'])


def extract_orientations(result_file, original_mat, output_mat, output_grid, increment=-1):
    r = damask.Result(result_file).view(increments=increment)

    O_field = r.get('O')
    O_flat = O_field.reshape(-1, 4, order='F')
    n_points = len(O_flat)

    with open(original_mat, 'r') as f:
        original_data = yaml.safe_load(f)

    template = original_data['material'][0]['constituents'][0].copy()
    homog = original_data['material'][0]['homogenization']

    new_material_data = {
        'homogenization': original_data['homogenization'],
        'phase': original_data['phase'],
        'material': []
    }

    for i in range(n_points):
        constituent = template.copy()
        constituent['O'] = O_flat[i].tolist()
        new_material_data['material'].append({
            'homogenization': homog,
            'constituents': [constituent]
        })

    with open(output_mat, 'w') as f:
        yaml.dump(new_material_data, f, default_flow_style=None, sort_keys=False)
        log.info(f"  Created {output_mat} with {n_points} material entries")

    # Strategy A: reset to original cube
    original_geom = damask.GeomGrid.load(f"{cwd}/{grid}.vti")
    cells = original_geom.cells
    size = original_geom.size

    material_ids = np.arange(n_points).reshape(cells, order='F')
    new_geom = damask.GeomGrid(material=material_ids, size=size)
    new_geom.save(output_grid)
    log.info(f"  Created grid: {output_grid} (cube reset)")


def get_completed_loadsteps(result_file, L_data, t_data, start_loadstep, elapsed_time):
    """Figure out how many loadsteps were completed by matching time."""
    r = damask.Result(result_file)
    last_saved_time = r.times[-1]
    actual_time = elapsed_time + last_saved_time

    log.debug(f"  Time matching: last_saved={last_saved_time:.6e}, elapsed={elapsed_time:.6e}, actual={actual_time:.6e}")

    for i in range(start_loadstep, len(t_data)):
        if t_data[i] >= actual_time - 1e-12:
            completed = i - start_loadstep
            log.debug(f"  Matched t_data[{i}]={t_data[i]:.6e} >= {actual_time:.6e}, completed={completed}")
            return completed

    completed = len(t_data) - start_loadstep
    log.debug(f"  No match found, returning all remaining: {completed}")
    return completed


def run_segment(grid_file, load_file, material_file):
    cmd = f"DAMASK_grid -g {cwd}/{grid_file} -l {cwd}/{load_file} -m {cwd}/{material_file} -w {wd}"

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    last_converged = None
    failed = False

    for line in iter(process.stdout.readline, ''):
        if line:
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

    return {
        'failed': failed,
        'last_converged': last_converged
    }


L_data, t_data = parse_extrusion(extrusion_file)

# Preview adaptive N distribution
L_norms = np.linalg.norm(L_data.reshape(-1, 9), axis=1)
log.info(f"||L|| range: {L_norms.min():.2f} to {L_norms.max():.2f} s^-1")
log.info(f"Peak ||L|| at loadstep {np.argmax(L_norms)}")
log.info(f"Total loadsteps: {len(L_data)}")
log.info(f"Total time: {t_data[-1]:.6e} s")

segment = 0
completed_loadsteps = 0
total_loadsteps = len(L_data)
elapsed_time = 0.0
simulation_complete = False

while segment < max_segments and not simulation_complete:
    log.info(f"\n{'='*60}")
    log.info(f"  SEGMENT {segment} | Loadsteps: {completed_loadsteps}/{total_loadsteps} | Time: {elapsed_time:.6e}")
    log.info(f"{'='*60}")

    if segment == 0:
        grid_file = f'{grid}.vti'
        mat_file = f'{mat}.yaml'
        load_file = f'segment_{segment}_load.yaml'
        result_name = f'{grid}_segment_{segment}_load_{mat}'
    else:
        grid_file = f'{grid}_segment{segment}.vti'
        mat_file = f'{mat}_segment{segment}.yaml'
        load_file = f'segment_{segment}_load.yaml'
        result_name = f'{grid}_segment{segment}_segment_{segment}_load_{mat}_segment{segment}'

    n_steps = create_load_segment(L_data, t_data, load_file, start_loadstep=completed_loadsteps, elapsed_time=elapsed_time)

    if n_steps == 0:
        log.info("Error: No loadsteps")
        break

    result = run_segment(grid_file, load_file, mat_file)
    result_file = f"{wd}/{result_name}.hdf5"

    log.info(f"  DAMASK result: failed={result['failed']}, last_converged={result['last_converged']}")

    # Check result file exists
    if not os.path.exists(result_file):
        log.info(f"  Result file not found: {result_file}")
        log.info(f"  Files in wd: {[f for f in os.listdir(wd) if f.endswith('.hdf5')]}")
        break

    r = damask.Result(result_file)
    n_saved = len(r.increments)
    last_time = r.times[-1]
    log.info(f"  Result file: {n_saved} saved increments, last time={last_time:.6e}")
    log.debug(f"  First 5 increments: {r.increments[:5]}")
    log.debug(f"  Last 5 increments: {r.increments[-5:]}")

    if not result['failed']:
        log.info(f"  Segment {segment} completed without crash")

        if completed_loadsteps + n_steps >= total_loadsteps:
            simulation_complete = True
            log.info("  >>> FULL EXTRUSION PATH COMPLETED <<<")
        else:
            # Segment completed but more loadsteps remain
            # Since it didn't crash, ALL n_steps loadsteps completed
            loadsteps_completed = n_steps
            elapsed_time = t_data[completed_loadsteps + loadsteps_completed - 1]

            log.info(f"  Completed {loadsteps_completed} loadsteps this segment")
            log.info(f"  Total: {completed_loadsteps + loadsteps_completed}/{total_loadsteps}")
            log.info(f"  Elapsed time: {elapsed_time:.6e}")

            extract_orientations(
                result_file=result_file,
                original_mat=mat_file,
                output_mat=f"{cwd}/{mat}_segment{segment+1}.yaml",
                output_grid=f"{cwd}/{grid}_segment{segment+1}.vti",
                increment=-1
            )

            completed_loadsteps += loadsteps_completed
            segment += 1
    else:
        log.info(f"  Segment {segment} CRASHED")

        if result['last_converged'] is None or result['last_converged'] < min_inc_per_segment:
            log.info(f"  Failed too early (inc {result['last_converged']}). Aborting.")
            break

        last_saved_inc = r.increments[-1]
        log.info(f"  Last saved increment: {last_saved_inc}")

        extract_orientations(
            result_file=result_file,
            original_mat=mat_file,
            output_mat=f"{cwd}/{mat}_segment{segment+1}.yaml",
            output_grid=f"{cwd}/{grid}_segment{segment+1}.vti",
            increment=-1
        )

        # Time-based loadstep counting since N varies per loadstep
        loadsteps_completed = get_completed_loadsteps(
            result_file, L_data, t_data, completed_loadsteps, elapsed_time)

        log.info(f"  Completed {loadsteps_completed} loadsteps this segment")

        completed_loadsteps += loadsteps_completed
        elapsed_time = t_data[completed_loadsteps - 1]

        log.info(f"  Total: {completed_loadsteps}/{total_loadsteps}")
        log.info(f"  Elapsed time: {elapsed_time:.6e}")

        segment += 1

# Save all results and export selected increments to VTK
results_dir = f"{cwd}/results_extrusion_{grid}"
os.makedirs(results_dir, exist_ok=True)

hdf5_files = [f for f in os.listdir(wd) if f.endswith('.hdf5')]

for hdf5_file in hdf5_files:
    try:
        log.info(f"  Processing {hdf5_file}...")
        r = damask.Result(f'{wd}/{hdf5_file}')

        all_incs = r.increments
        selected = [all_incs[i] for i in range(0, len(all_incs), VTK_INTERVAL)]
        if all_incs[-1] not in selected:
            selected.append(all_incs[-1])

        r_selected = r.view(increments=selected)
        r_selected.add_IPF_color([0,0,1])
        r_selected.export_VTK(target_dir=results_dir)

        shutil.copyfile(f'{wd}/{hdf5_file}', f'{results_dir}/{hdf5_file}')

    except Exception as e:
        log.info(f"  Error processing {hdf5_file}: {e}")

log.info(f"{segment + 1} segments completed.")
log.info(f"Completed {completed_loadsteps} / {total_loadsteps} loadsteps")
log.info(f"Results in: {results_dir}")
log.info(f"Log file: remap_log.txt")
log.info(f"Temp dir: {wd}")