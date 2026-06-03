import typer

from damask_remap import generate as gen

app = typer.Typer(help="Orientation-based remapping tool for DAMASK simulations.")


@app.command()
def generate(
    cells: int = 16,
    size: float = 1.0e-3,
    phase: str = "Aluminium",
    seed: int = 42,
    out_grid: str = "grid",
    out_material: str = "material",
):
    "Generate microstructure, loadcase, and material files"
    grid = gen.generate_inputs(cells, size, phase, seed, out_grid, out_material)
    typer.echo(f"Wrote {out_grid}.vti ({cells}³ = {cells**3} grains)")
    typer.echo(f"Wrote {out_material}.yaml")
    typer.echo(grid)


@app.command()
def run(segments: int = 4):
    "Run a DAMASK simulation with a remapping method"
    typer.echo(f"[run] segments={segments}")


@app.command()
def analyze():
    "Analyze simulation results"
    typer.echo("[analyze]")


def main():
    app()


if __name__ == "__main__":
    main()
