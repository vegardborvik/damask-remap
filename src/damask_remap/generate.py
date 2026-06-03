import numpy as np
import damask

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


def make_material(n_grains: int, phase: str, seed: int | None):
    # One material per grain, each with a random orientation.
    orientations = damask.Rotation.from_random(n_grains, rng_seed=seed)
    config = damask.ConfigMaterial(
        homogenization=HOMOGENIZATION,
        phase={phase: PHASE_ALUMINUM},
    )
    return config.material_add(phase=phase, O=orientations, homogenization="SX")


def generate_inputs(cells, size, phase, seed, out_grid, out_material):
    grid = make_grid(cells, size)
    grid.save(out_grid)

    material = make_material(cells**3, phase, seed)
    material.save(f"{out_material}.yaml")

    return grid
