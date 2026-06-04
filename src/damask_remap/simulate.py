from pathlib import Path

import damask
import yaml

from damask_remap import generate as gen


def run_split(name: str, segments: int, *, deform: bool = False):
    case_dir = Path("outputs") / name
    grid_file, material_file, load_file = gen.resolve_inputs(case_dir)

    load_data = yaml.safe_load(load_file.read_text())
    disc = load_data["loadstep"][0]["discretization"]
    t_total, N_total = disc["t"], disc["N"]
    f_out = load_data["loadstep"][0]["f_out"]
    t_seg, N_seg = t_total // segments, N_total // segments

    phase = next(iter(yaml.safe_load(material_file.read_text())["phase"]))
    # print(f"Total time: {t_total}, Total increments: {N_total}")
    # print(f"Phase: {phase}")
    # print(f"Total segments: {segments}")
    # print(f"Time per segment: {t_seg}, Increments per segment: {N_seg}")
    print(f"f_out : {f_out}")


# def run(mat: Path, grid: Path, load: Path, wd: Path):
# damask.util.run(
# f"DAMASK_grid -g {grid.name} -l {load.name} -m {mat.name}", wd=str(wd)
# )
