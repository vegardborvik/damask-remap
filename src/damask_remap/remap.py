import damask
import numpy as np

from damask_remap import generate as gen


def remap_orientations(result_file, original_grid_file, phase, *, deform: bool):
    r = damask.Result(result_file).view(increments=-1)

    O = r.get("O").reshape(-1, 4, order="F")
    orientations = damask.Rotation.from_quaternion(O)

    grid = damask.GeomGrid.load(original_grid_file)
    size = grid.size

    if deform:
        F_avg = np.asarray(np.average(r.place("F"), axis=0))
        size = size @ F_avg

    material_ids = np.arange(len(O)).reshape(grid.cells, order="F")
    new_grid = damask.GeomGrid(material=material_ids, size=size)
    new_material = gen.make_material(len(O), phase, orientations=orientations)

    return new_grid, new_material
