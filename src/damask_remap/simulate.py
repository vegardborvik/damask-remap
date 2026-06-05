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
        prev_result, sta = run(m, g, l, seg_dir)
        prev_grid = g
        last, converged = read_status(sta)

        if last < N_seg or not converged:
            raise RuntimeError(
                f"seg_{i:02d} did not finish: reached increment {last}/{N_seg}\n"
                f"all_converged={converged}"
            )

    return prev_result


def run_automatic(name: str, *, max_segments: int = 50, deform: bool = False):
    case_dir = Path("outputs") / name
    grid_file, material_file, load_file = gen.resolve_inputs(case_dir)

    load_data = damask.LoadcaseGrid.load(load_file)
    disc = load_data["loadstep"][0]["discretization"]
    t_total, N_total = disc["t"], disc["N"]
    f_out = load_data["loadstep"][0]["f_out"]

    material_data = damask.ConfigMaterial.load(material_file)
    phase = next(iter(material_data["phase"]))

    t_elapsed = 0.0
    prev_result = None
    prev_grid = None
    segment = 0
    dt = t_total / N_total

    while segment < max_segments:
        remaining = t_total - t_elapsed
        N_seg = round(remaining / dt)
        if N_seg < 1:
            break

        seg_dir = case_dir / f"seg_{segment:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # First run
        if segment == 0:
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

        seg_load = copy.deepcopy(load_data)
        seg_load["loadstep"][0]["discretization"].update(t=remaining, N=N_seg)
        seg_load.save(seg_dir / load_file.name)

        g, m, l = gen.resolve_inputs(seg_dir)
        prev_result, sta = run(m, g, l, seg_dir)
        prev_grid = g
        last, converged = read_status(sta)

        if last >= N_seg and converged:
            return prev_result  # Done

        r = damask.Result(prev_result)
        last_saved_time = r.times[-1]

        if last_saved_time <= 0.0:
            raise RuntimeError(
                f"seg_{segment:02d} made no progress (Try a smaller f_out)"
            )
        t_elapsed += last_saved_time
        segment += 1

    raise RuntimeError(
        f"Maximum segments hit without completing. max_segments={max_segments}"
    )


def read_status(sta_file: Path):
    rows = [l.split() for l in sta_file.read_text().splitlines()[1:] if l.strip()]
    if not rows:
        return 0, False  # last_increment, all_converged
    last_increment = int(rows[-1][0])
    all_converged = all(r[3] == "T" for r in rows)  # Reads from STA file if converged
    return last_increment, all_converged


def run(mat: Path, grid: Path, load: Path, wd: Path):
    stem = f"{grid.stem}_{load.stem}_{mat.stem}"
    try:
        damask.util.run(
            f"DAMASK_grid -g {grid.name} -l {load.name} -m {mat.name}", wd=str(wd)
        )
    except RuntimeError as e:
        print(f"Solver exited with error: {e}")

    return wd / f"{stem}.hdf5", wd / f"{stem}.sta"
