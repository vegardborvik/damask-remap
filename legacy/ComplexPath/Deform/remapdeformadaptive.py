import os
import tempfile
import shutil
import re
import sys
import subprocess

import pandas as pd
import yaml
import damask
import numpy as np

extrusion_file = "COSMETEX_SYM_HexMesh_26B_350-45_Coulomb_S-plus0.05_001.pth"

grid = "rolling.geom"
mat = "material"

cwd = os.getcwd()
print(wd := tempfile.mkdtemp())

#settings
max_segments = 20
min_inc_per_segment = 5
TARGET_STRAIN_PER_INC = 0.001  #For adaptive N when the magnitude of the strain rate gets too high (Not really needed)
MIN_N = 10                      
MAX_N = 500                    
VTK_INTERVAL = 100             #Deciding how many VTK files to export as f_out needs to be 1        


def parse_extrusion(pth_file):
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
            continue

        L_norm = np.linalg.norm(L)
        total_strain = L_norm * t_step
        N = int(np.ceil(total_strain / TARGET_STRAIN_PER_INC))
        N = max(MIN_N, min(N, MAX_N))

        total_incs += N

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

    return len(load_case['loadstep'])


def extract_orientations(result_file, current_grid_file, original_mat, output_mat, output_grid, increment=-1):
    r = damask.Result(result_file).view(increments=increment)

    O_field = r.get('O')
    O_flat = O_field.reshape(-1, 4, order='F')
    n_points = len(O_flat)

    F_avg = np.average(r.place('F'), axis=0)

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
        print(f"Created {output_mat} with {n_points} material entries")

    current_geom = damask.GeomGrid.load(current_grid_file)
    cells = current_geom.cells

    deformed_edges = F_avg @ np.diag(current_geom.size)
    new_size = np.linalg.norm(deformed_edges, axis=0)

    material_ids = np.arange(n_points).reshape(cells, order='F')
    new_geom = damask.GeomGrid(
        material=material_ids, 
        size=new_size)
    new_geom.save(output_grid)


def get_completed_loadsteps(result_file, L_data, t_data, start_loadstep, elapsed_time):

    r = damask.Result(result_file)
    last_saved_time = r.times[-1]
    actual_time = elapsed_time + last_saved_time

    for i in range(start_loadstep, len(t_data)):
        if t_data[i] >= actual_time - 1e-12:
            return i - start_loadstep
    return len(t_data) - start_loadstep


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
print(f"\n||L|| range: {L_norms.min():.2f} to {L_norms.max():.2f} s^-1")
print(f"Peak at loadstep {np.argmax(L_norms)}")

segment = 0
completed_loadsteps = 0
total_loadsteps = len(L_data)
elapsed_time = 0.0
simulation_complete = False

while segment < max_segments and not simulation_complete:
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
        print("Error: No loadsteps")
        break

    result = run_segment(grid_file, load_file, mat_file)

    result_file = f"{wd}/{result_name}.hdf5"

    if not result['failed']:
        print(f"Segment completed")
        if completed_loadsteps + n_steps >= total_loadsteps:
            simulation_complete = True
            print("Full extrusion path completed")
        else:
            # Use time-based loadstep counting since N varies per loadstep
            loadsteps_completed = get_completed_loadsteps(
                result_file, L_data, t_data, completed_loadsteps, elapsed_time)
            elapsed_time = t_data[completed_loadsteps + loadsteps_completed - 1]

            extract_orientations(
                result_file=result_file,
                current_grid_file=f"{cwd}/{grid_file}",
                original_mat=mat_file,
                output_mat=f"{cwd}/{mat}_segment{segment+1}.yaml",
                output_grid=f"{cwd}/{grid}_segment{segment+1}.vti",
                increment=-1
            )


            completed_loadsteps += loadsteps_completed
            segment += 1
    else:
        print("Segment failed!")
        if result['last_converged'] is None or result['last_converged'] < min_inc_per_segment:
            print("Error: failed too early")
            break

        # Use last saved increment and time-based loadstep counting
        r = damask.Result(result_file)
        last_saved_inc = r.increments[-1]

        extract_orientations(
            result_file=result_file,
            current_grid_file=f"{cwd}/{grid_file}",
            original_mat=mat_file,
            output_mat=f"{cwd}/{mat}_segment{segment+1}.yaml",
            output_grid=f"{cwd}/{grid}_segment{segment+1}.vti",
            increment=-1
        )

        # Use time-based loadstep counting since N varies per loadstep
        loadsteps_completed = get_completed_loadsteps(
            result_file, L_data, t_data, completed_loadsteps, elapsed_time)
        completed_loadsteps += loadsteps_completed
        elapsed_time = t_data[completed_loadsteps - 1]

        segment += 1

# Save all results and export selected increments to VTK
results_dir = f"{cwd}/results_extrusion_{grid}"
os.makedirs(results_dir, exist_ok=True)

hdf5_files = [f for f in os.listdir(wd) if f.endswith('.hdf5')]

for hdf5_file in hdf5_files:
    try:
        print(f"\n  Processing {hdf5_file}...")
        r = damask.Result(f'{wd}/{hdf5_file}')

        # Only export selected increments to VTK
        all_incs = r.increments
        selected = [all_incs[i] for i in range(0, len(all_incs), VTK_INTERVAL)]
        if all_incs[-1] not in selected:
            selected.append(all_incs[-1])

        r_selected = r.view(increments=selected)
        r_selected.add_IPF_color([0,0,1])
        r_selected.export_VTK(target_dir=results_dir)

        shutil.copyfile(f'{wd}/{hdf5_file}', f'{results_dir}/{hdf5_file}')

    except Exception as e:
        print(f"error processing {hdf5_file}: {e}")
