import typer
import yaml
from typing import Literal
from pathlib import Path

from damask_remap import generate as gen
from damask_remap import simulate as sim
from damask_remap import remap
from damask_remap.config import RunConfig

app = typer.Typer(help="Orientation-based remapping tool for DAMASK simulations.")


@app.command()
def generate(
    config: Path | None = None,
    cells: int | None = None,
    size: list[float] | None = None,
    phase: str | None = None,
    seed: int | None = None,
    mode: Literal["rolling"] | None = None,
    solver: Literal["basic"] | None = None,
    t: int | None = None,
    N: int | None = None,
    rate: float | None = None,
    name: str | None = None,
    f_out: int | None = None,
):
    "Generate microstructure, loadcase, and material files"
    data = yaml.safe_load(config.read_text()) if config else {}
    data.setdefault("grid", {})
    data.setdefault("load", {})
    data.setdefault("material", {})

    if name is not None:
        data["name"] = name
    if cells is not None:
        data["grid"]["cells"] = cells
    if size is not None:
        data["grid"]["size"] = size
    if phase is not None:
        data["material"]["phase"] = phase
    if seed is not None:
        data["material"]["seed"] = seed
    if mode is not None:
        data["load"]["mode"] = mode
    if solver is not None:
        data["load"]["solver"] = solver
    if t is not None:
        data["load"]["t"] = t
    if N is not None:
        data["load"]["N"] = N
    if rate is not None:
        data["load"]["rate"] = rate
    if f_out is not None:
        data["load"]["f_out"] = f_out

    cfg = RunConfig(**data)
    grid = gen.generate_inputs(cfg)
    typer.echo(grid)

    # if config is not None:
    #     cfg = RunConfig(**yaml.safe_load(config.read_text()))
    # else:
    #     cfg = RunConfig()
    # grid = gen.generate_inputs(cfg)
    # typer.echo(grid)


@app.command()
def run_split(
    segments: int = 4,
    name: str = "input_files",
    deform: bool = False,
):
    "Run a DAMASK simulation with a remapping method"
    case_dir = Path("outputs") / name
    sim.run_split(name, segments, deform=deform)
    typer.echo(f"[run] segments={segments}")
    typer.echo(f"Writing to {case_dir}")


@app.command()
def run_auto(
    name: str = "input_files",
    max_segments: int = 50,
    deform: bool = False,
):
    "Run a DAMASK simulation with an automatic remapping method."
    case_dir = Path("outputs") / name
    typer.echo(f"[run_auto] deform={deform}")
    typer.echo(f"Writing to {case_dir}")
    sim.run_automatic(name, max_segments=max_segments, deform=deform)


@app.command()
def analyze():
    "Analyze simulation results"
    typer.echo("[analyze]")


def main():
    app()


if __name__ == "__main__":
    main()
