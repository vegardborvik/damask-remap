import tempfile
import shutil
import os

import yaml
import numpy as np
from matplotlib import pyplot as plt
import damask

load = "rolling.load"
grid = "rolling.geom"
mat = "material"

F = []
P = []

cwd = os.getcwd()
print(wd := tempfile.mkdtemp())

def extract_orientations(result_file, original_mat, output_mat, output_grid):
    r = damask.Result(result_file).view(increments=-1)

    #Get orientations
    O_field = r.get('O')
    O_flat = O_field.reshape(-1, 4, order='F')
    n_points = len(O_flat)

    #Get Deformation
    F_avg = np.average(r.view(increments=-1).place('F'),axis=0)

    #Load original material og kopier orienteringer
    with open(original_mat, 'r') as f:
        original_data = yaml.safe_load(f)

    template = original_data['material'][0]['constituents'][0].copy()
    homog = original_data['material'][0]['homogenization']

    new_material_data = {}

    new_material_data['homogenization'] = original_data['homogenization']
    new_material_data['phase'] = original_data['phase']
    new_material_data['material'] = []

    #Oppdater quaternions til nye (1:1 remapping)
    for i in range(n_points):
        constituents = template.copy()
        constituents['O'] = O_flat[i].tolist()

        material_entry = {
            'homogenization': homog,
            'constituents': [constituents]
        }

        new_material_data['material'].append(material_entry)

    #Lag ny material fil
    with open(output_mat, 'w') as f:
        yaml.dump(new_material_data, f, default_flow_style=None, sort_keys=False)

    #Lager ny grid med n_points for 1 til 1 remapping
    original_geom = damask.GeomGrid.load(f"{cwd}/{grid}.vti")
    cells = original_geom.cells
    original_geom.size = original_geom.size @ F_avg


    material_ids = np.arange(n_points, dtype=int).reshape(cells, order='F')

    new_geom = damask.GeomGrid(
        material=material_ids,
        size=original_geom.size
    )
    new_geom.save(output_grid)

#Første run
damask.util.run(f'DAMASK_grid -g {cwd}/{grid}.vti -l {cwd}/{load}.yaml -m {cwd}/{mat}.yaml -w {wd}')
r = damask.Result(f'{wd}/{grid}_{load}_{mat}.hdf5')
F.append([np.average(_, axis=0) for _ in r.place('F').values()])
P.append([np.average(_, axis=0) for _ in r.place('P').values()])
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)

#REMAPPING
extract_orientations(
    result_file=f'{wd}/{grid}_{load}_{mat}.hdf5',
    original_mat=f'{cwd}/{mat}.yaml',
    output_mat=f'{wd}/{mat}_remap1.yaml',
    output_grid=f"{wd}/{grid}_2.vti"
)

shutil.copyfile(f'{wd}/{grid}_{load}_{mat}.hdf5', f'{cwd}/{grid}_{load}_{mat}.hdf5')
#shutil.copyfile(f"{cwd}/{load}-2.yaml", f"{wd}/{load}-2.yaml")

#restart
damask.util.run(f'DAMASK_grid -g {wd}/{grid}_2.vti -l {cwd}/{load}.yaml -m {wd}/{mat}_remap1.yaml -w {wd}')
r = damask.Result(f'{wd}/{grid}_2_{load}_{mat}_remap1.hdf5')
F.append([np.average(_,axis=0) for _ in r.place('F').values()])
P.append([np.average(_,axis=0) for _ in r.place('P').values()])
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)

#Andre
extract_orientations(
    result_file=f'{wd}/{grid}_2_{load}_{mat}_remap1.hdf5',
    original_mat=f'{wd}/{mat}_remap1.yaml',
    output_mat=f'{wd}/{mat}_remap2.yaml',
    output_grid=f"{wd}/{grid}_3.vti"
)
shutil.copyfile(f'{wd}/{grid}_2_{load}_{mat}_remap1.hdf5', f'{cwd}/{grid}_2_{load}_{mat}_remap1.hdf5')

# damask.util.run(f'DAMASK_grid -g {wd}/{grid}_3.vti -l {cwd}/{load}.yaml -m {wd}/{mat}_remap2.yaml -w {wd}')
# r = damask.Result(f'{wd}/{grid}_3_{load}_{mat}_remap2.hdf5')
# F.append([np.average(_,axis=0) for _ in r.place('F').values()])
# P.append([np.average(_,axis=0) for _ in r.place('P').values()])
# r.add_IPF_color([0,0,1])
# r.export_VTK(target_dir=cwd)

# shutil.copyfile(f'{wd}/{grid}_3_{load}_{mat}_remap2.hdf5', f'{cwd}/{grid}_3_{load}_{mat}_remap2.hdf5')

#PLOT STRESS STRAIN
# F_ = np.concatenate([F[0], F[0][-1]@F[1], F[0][-1]@F[1][-1]@F[2]])
# epsilon_multiplicative = damask.mechanics.strain(F_,m=0.0,t='V')

# epsilon_ = [damask.mechanics.strain(_,m=0.0,t='V') for _ in F]
# epsilon_additive = np.concatenate([epsilon_[0],epsilon_[0][-1]+epsilon_[1],epsilon_[0][-1]+epsilon_[1][-1]+epsilon_[2]])

# sigma = damask.mechanics.stress_Cauchy(np.concatenate([P[0],P[1],P[2]]),
#                                        np.concatenate([F[0],F[1],F[2]]))

# fig, ax = plt.subplots()

# ax.set_xlabel('strain')
# ax.set_ylabel('Cauchy stress / Pa')
# ax.plot(epsilon_multiplicative[:,0,0],sigma[:,0,0],label='multiplicative')
# ax.plot(epsilon_additive[:,0,0],sigma[:,0,0],label='addititve')
# ax.legend()

# plt.show()