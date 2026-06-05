"""Verify a completed segmented run.

Point it at an outputs/<name> directory and it checks the invariants that
must hold for a correct orientation remap:

  1. Orientation transfer is exact: each segment's input material orientations
     equal the previous segment's final-increment orientations (point-by-point).
  2. Geometry is consistent with a single mode across all segments:
       - nodeform: box size constant
       - deform:   box size chains as  next_size = prev_size @ F_avg
  3. (Informational) equivalent strain and lattice rotation per segment.

Usage:
    python tests/check_run.py outputs/test
    python tests/check_run.py outputs/test2

Exits non-zero if any invariant fails, so it can also gate CI later.
"""

import sys
from pathlib import Path

import numpy as np
import damask
import yaml

GRID = "rolling.grid.vti"
RESULT = "rolling.grid_rolling.load_rolling.material.hdf5"
MAT = "rolling.material.yaml"


def valid_segments(case_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in case_dir.glob("seg_*")
        if p.is_dir() and (p / GRID).exists() and (p / RESULT).exists()
    )


def final_orientations(seg: Path) -> np.ndarray:
    r = damask.Result(seg / RESULT).view(increments=-1)
    return r.get("O").reshape(-1, 4, order="F")


def input_orientations(seg: Path) -> np.ndarray:
    data = yaml.safe_load((seg / MAT).read_text())
    return np.array([m["constituents"][0]["O"] for m in data["material"]])


def avg_F(seg: Path) -> np.ndarray:
    r = damask.Result(seg / RESULT).view(increments=-1)
    return np.asarray(np.average(r.place("F"), axis=0))


def box_size(seg: Path) -> np.ndarray:
    return damask.GeomGrid.load(seg / GRID).size


def check_orientation_transfer(segs: list[Path]) -> bool:
    print("\n[1] Orientation transfer (seg[i] final == seg[i+1] input)")
    ok = True
    for a, b in zip(segs, segs[1:]):
        fa, ib = final_orientations(a), input_orientations(b)
        match = fa.shape == ib.shape and np.allclose(fa, ib, atol=1e-8)
        ok &= match
        print(f"    {a.name} -> {b.name}: {'PASS' if match else 'FAIL'}")
    if len(segs) < 2:
        print("    (single segment, nothing to chain)")
    return ok


def check_geometry(segs: list[Path]) -> bool:
    print("\n[2] Geometry consistency (single mode across all segments)")
    modes = []
    for a, b in zip(segs, segs[1:]):
        sa, sb = box_size(a), box_size(b)
        is_constant = np.allclose(sa, sb, rtol=1e-6, atol=1e-12)
        is_chained = np.allclose(sa @ avg_F(a), sb, rtol=1e-6, atol=1e-12)
        if is_constant:
            mode = "nodeform"
        elif is_chained:
            mode = "deform"
        else:
            mode = "BROKEN"
        modes.append(mode)
        print(f"    {a.name} -> {b.name}: {mode}  (size {sb})")
    if len(segs) < 2:
        print("    (single segment, nothing to chain)")
        return True
    ok = "BROKEN" not in modes and len(set(modes)) == 1
    print(f"    => {'PASS' if ok else 'FAIL'} (modes: {set(modes)})")
    return ok


def report_evolution(segs: list[Path]) -> None:
    print("\n[3] Per-segment strain & lattice rotation (informational)")
    for s in segs:
        r = damask.Result(s / RESULT)
        incs = r.increments
        F = np.asarray(np.average(r.view(increments=-1).place("F"), axis=0))
        eps = damask.mechanics.strain(F, m=0.0, t="V")
        o0 = damask.Orientation(
            rotation=damask.Rotation.from_quaternion(
                r.view(increments=incs[0]).get("O").reshape(-1, 4, order="F")
            ),
            family="cubic",
        )
        of = damask.Orientation(
            rotation=damask.Rotation.from_quaternion(
                r.view(increments=incs[-1]).get("O").reshape(-1, 4, order="F")
            ),
            family="cubic",
        )
        ang = np.degrees(o0.disorientation(of).as_axis_angle(pair=True)[1])
        print(
            f"    {s.name}: eps_zz={eps[2, 2]:+.4f}  "
            f"lattice rotation mean={ang.mean():.2f} deg max={ang.max():.2f} deg"
        )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    case_dir = Path(sys.argv[1])
    segs = valid_segments(case_dir)
    if not segs:
        print(f"No valid segments (with {GRID} and {RESULT}) found in {case_dir}")
        return 2

    print(f"Checking {case_dir}  ({len(segs)} valid segments)")
    ok = check_orientation_transfer(segs)
    ok &= check_geometry(segs)
    report_evolution(segs)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
