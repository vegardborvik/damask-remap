import numpy as np
import damask
import yaml
from pathlib import Path

from damask_remap.config import GridConfig, LoadConfig, MaterialConfig, RunConfig

# Initializing standard material physics for Aluminium phase
# Parameters taken from the example file from DAMASK

HOMOGENIZATION = {
    "SX": {"N_constituents": 1, "mechanical": {"type": "pass"}},
}

PHASE_ALUMINUM = {
    "lattice": "cF",
    "mechanical": {
        "output": ["F", "P", "F_e", "F_p", "L_p", "O"],
        "elastic": {
            "type": "Hooke",
            "C_11": 106.75e9,
            "C_12": 60.41e9,
            "C_44": 28.34e9,
        },
        "plastic": {
            "type": "phenopowerlaw",
            "N_sl": [12],
            "a_sl": [2.25],
            "atol_xi": 1.0,
            "dot_gamma_0_sl": [0.001],
            "h_0_sl-sl": [75e6],
            "h_sl-sl": [1.0, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4],
            "n_sl": [20],
            "output": ["xi_sl"],
            "xi_0_sl": [31e6],
            "xi_inf_sl": [63e6],
        },
    },
}


def make_grid(grid: GridConfig):
    # The cells³ grid where every voxel is its own grain (Maximum 32x32x32 or the DAMASK_grid will fail)
    n_grains: int = grid.cells**3
    material_ids = np.arange(n_grains).reshape(grid.cells, grid.cells, grid.cells)
    return damask.GeomGrid(material=material_ids, size=grid.size)


def make_material(material: MaterialConfig, n_grains: int, orientations=None):
    # One material per grain, each with a random orientation.
    if orientations is None:
        orientations = damask.Rotation.from_random(n_grains, rng_seed=material.seed)

    config = damask.ConfigMaterial(
        homogenization=HOMOGENIZATION,
        phase={material.phase: PHASE_ALUMINUM},
    )
    return config.material_add(
        phase=material.phase, O=orientations, homogenization="SX"
    )


def make_loadcase(load: LoadConfig, output_file):

    if load.mode == "rolling":
        bc = [[load.rate, 0, 0], [0, 0, 0], [0, 0, -load.rate]]
        loadstep = {
            "boundary_conditions": {"mechanical": {"L": bc}},
            "discretization": {"t": load.t, "N": load.N},
            "f_out": load.f_out,
        }

    if load.solver == "basic":
        solver_input = {"mechanical": "spectral_basic"}

    load_case = {"solver": solver_input, "loadstep": [loadstep]}

    # with open(file=output_file, mode="w") as f:
    # yaml.dump(load_case, f, default_flow_style=None, sort_keys=False)

    return damask.LoadcaseGrid(load_case).save(f"{output_file}.yaml")


def generate_inputs(config: RunConfig):
    out_dir = Path("outputs") / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(config.grid)
    grid.save(out_dir / f"{config.load.mode}.grid")

    material = make_material(config.material, n_grains=config.grid.cells**3)
    material.save(out_dir / f"{config.load.mode}.material.yaml")

    load = make_loadcase(
        config.load,
        out_dir / f"{config.load.mode}.load",
    )
    generate_metadata(config, out_dir)
    return grid


def generate_metadata(config: RunConfig, out_dir: Path):
    run_record = yaml.safe_dump(config.model_dump(), sort_keys=False)
    (out_dir / "run.yaml").write_text(run_record)


def resolve_inputs(case_dir: Path):
    try:
        grid = next(case_dir.glob("*.grid.vti"))
        material = next(case_dir.glob("*.material.yaml"))
        load = next(case_dir.glob("*.load.yaml"))
    except StopIteration:
        raise FileNotFoundError(
            f"Missing one of grid/material/load files in {case_dir}"
        )
    return grid, material, load
