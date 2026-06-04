import numpy as np
import damask
import yaml
from pathlib import Path

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


def make_grid(cells: int, size: float):
    # The cells³ grid where every voxel is its own grain (Maximum 32x32x32 or the DAMASK_grid will fail)
    n_grains: int = cells**3
    material_ids = np.arange(n_grains).reshape(cells, cells, cells)
    return damask.GeomGrid(material=material_ids, size=[size, size, size])


def make_material(n_grains: int, phase: str, seed: int = None, orientations=None):
    # One material per grain, each with a random orientation.
    if orientations is None:
        orientations = damask.Rotation.from_random(n_grains, rng_seed=seed)

    config = damask.ConfigMaterial(
        homogenization=HOMOGENIZATION,
        phase={phase: PHASE_ALUMINUM},
    )
    return config.material_add(phase=phase, O=orientations, homogenization="SX")


def make_loadcase(
    mode: str,
    solver: str,
    t: int,
    N: int,
    rate: float,
    output_file: str,
    f_out: int = 10,
):

    if N % f_out != 0:
        raise ValueError(
            f"N ({N}) must be a multiple of f_out ({f_out}) so that the final increment is saved."
        )

    if mode == "rolling":
        bc = [[rate, 0, 0], [0, 0, 0], [0, 0, -rate]]
        loadstep = {
            "boundary_conditions": {"mechanical": {"L": bc}},
            "discretization": {"t": t, "N": N},
            "f_out": f_out,
        }

    if solver == "basic":
        solver_input = {"mechanical": "spectral_basic"}

    load_case = {"solver": solver_input, "loadstep": [loadstep]}

    # with open(file=output_file, mode="w") as f:
    # yaml.dump(load_case, f, default_flow_style=None, sort_keys=False)

    return damask.LoadcaseGrid(load_case).save(f"{output_file}.yaml")


def generate_inputs(
    cells,
    size,
    phase,
    seed,
    mode,
    solver,
    t,
    N,
    rate,
    name,
    f_out,
):
    out_dir = Path("outputs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(cells, size)
    grid.save(out_dir / f"{mode}.grid")

    material = make_material(cells**3, phase, seed)
    material.save(out_dir / f"{mode}.material.yaml")

    load = make_loadcase(mode, solver, t, N, rate, out_dir / f"{mode}.load")

    return grid


# TODO GENERATE METADATA OF INPUT FILES
def generate_metadata():
    return
