from pathlib import Path

import damask
import yaml
import shutil
import copy

from damask_remap import generate as gen
from damask_remap.remap import remap_orientations


def run_split(name: str, segments: int, *, deform: bool = False):
    case_dir = Path("outputs") / name
    grid_file, material_file, load_file = gen.resolve_inputs(case_dir)

    load_data = yaml.safe_load(load_file.read_text())
    disc = load_data["loadstep"][0]["discretization"]
    t_total, N_total = disc["t"], disc["N"]
    f_out = load_data["loadstep"][0]["f_out"]
    phase = next(iter(yaml.safe_load(material_file.read_text())["phase"]))

    if N_total % segments != 0:
        raise ValueError("Total increments")
    t_seg, N_seg = t_total // segments, N_total // segments
    if N_seg % f_out != 0:
        raise ValueError("f_out")

    seg_load = copy.deepcopy(load_data)
    seg_load["loadstep"][0]["discretization"].update(t=t_seg, N=N_seg)

    prev_result = None
    prev_grid = None
    for i in range(segments):
        seg_dir = case_dir / f"seg_{i:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        if i == 0:
            shutil.copyfile(
                f"{case_dir}/{material_file.name}", f"{seg_dir}/{material_file.name}"
            )
            shutil.copyfile(
                f"{case_dir}/{grid_file.name}", f"{seg_dir}/{grid_file.name}"
            )
        else:
            new_grid, new_material = remap_orientations(
                prev_result, prev_grid, phase, deform=deform
            )
            new_grid.save(seg_dir / grid_file.stem)
            new_material.save(seg_dir / material_file.name)

        damask.LoadcaseGrid(seg_load).save(f"{seg_dir}/{load_file.name}")

        g, m, l = gen.resolve_inputs(seg_dir)
        prev_result = run(m, g, l, seg_dir)
        prev_grid = g
    return prev_result


def run(mat: Path, grid: Path, load: Path, wd: Path):
    damask.util.run(
        f"DAMASK_grid -g {grid.name} -l {load.name} -m {mat.name}", wd=str(wd)
    )
    return wd / f"{grid.stem}_{load.stem}_{mat.stem}.hdf5"
