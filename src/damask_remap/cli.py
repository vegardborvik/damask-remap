import typer

from damask_remap import generate as gen

app = typer.Typer(help="Orientation-based remapping tool for DAMASK simulations.")


@app.command()
def generate(
    cells: int = 16,
    size: float = 1.0e-3,
    phase: str = "Aluminium",
    seed: int = 42,
    mode: str = "rolling",
    solver: str = "basic",
    t: int = 100,
    N: int = 200,
    rate: float = 1e-3,
    name: str = "input_files",
    f_out: int = 10,
):
    "Generate microstructure, loadcase, and material files"
    grid = gen.generate_inputs(
        cells, size, phase, seed, mode, solver, t, N, rate, name, f_out
    )
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
