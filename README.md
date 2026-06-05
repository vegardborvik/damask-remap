# damask-remap

Orientation-based remapping for DAMASK crystal-plasticity simulations, used to study texture
evolution at large plastic deformations.

The method is built for low-resolution, statistical microstructures where each voxel is a single
grain with its own orientation. DAMASK's spectral solver tends to lose convergence at large deformations, 
making it hard to reach large plastic deformations. This tool splits the simulation into segments 
and remaps the crystal orientations from one segment to the next, letting the simulation continue 
past where a continuous simulation would fail. It is a texture-oriented remap, only the orientations 
are transferred, and the mechanical state is not preserved.

## How it works

Two ways to decide where the segment boundaries go:

- `run-split` divides the load into a fixed number of equal segments.
- `run-auto` runs until the solver fails to converge, remaps from the last saved increment, and
  continues the simulation until the target deformation is reached.

Two choices for the geometry of the remapped box:

- *no-deform*: the box is reset to its original grid shape each segment.
- *deform*: the box carries the segment's average deformation gradient, so geometry accumulates
  across segments.

## Install

You need DAMASK 3.x — both the Python API and the `DAMASK_grid` solver on your `PATH`. See the
[DAMASK install docs](https://damask-multiphysics.org).

```bash
conda env create -f environment.yml
conda activate damask-remap
# install DAMASK (Python package + grid solver) into the env, per the DAMASK docs
pip install -e .
```

## Usage

```bash
# generate inputs for a 99 % rolling-reduction case (rate 1e-3 -> t=4605, N=9210)
damask-remap generate --name rolling99 --cells 16 --t 4605 --n 9210

# run with automatic remapping (deform geometry)
damask-remap run-auto --name rolling99 --deform

# or a fixed 4-segment split
damask-remap run-split --name rolling99 --segments 4 --deform
```

Results go to `outputs/<name>/`, one `seg_NN/` folder per segment (grid, material, load, and the
`.hdf5` result).

## Project structure

```
src/damask_remap/
  generate.py   # microstructure / load / material generation
  remap.py      # the orientation remap transform (deform flag)
  simulate.py   # run-split, run-auto, convergence detection
  cli.py        # command-line interface
  analyze.py    # result analysis (work in progress)
legacy/         # original thesis scripts, kept for reference (very hardcoded)
```

## Status
Still a work in progress

Working: input generation, both remapping strategies, deform / no-deform, automatic and segmented remapping methods. 
In progress: result analysis, plots, MTEX integration.

`legacy/` holds the original scripts from the bachelor thesis. They are kept for reference and
are not the current code path.

## Acknowledgements

The remapping approach is inspired by the adaptive-remeshing method of Sedighiani et al.:

> K. Sedighiani et al. "Large-deformation crystal plasticity simulation of microstructure and
> microtexture evolution through adaptive remeshing." *International Journal of Plasticity* 146
> (2021), 103078. https://doi.org/10.1016/j.ijplas.2021.103078

This project is based on my bachelor thesis, *Orientation-Based Remapping for Large-Strain
Texture Evolution Simulation in DAMASK* (NTNU, 2026), co-authored with Elias Bolz Gulema and
supervised by Assoc. Prof. Tomáš Mánik.

## License

MIT — see [LICENSE](LICENSE).
