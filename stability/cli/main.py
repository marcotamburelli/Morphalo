from pathlib import Path
import json
import typer

from stability.core.spec_loader import load_hocon_spec

app = typer.Typer(help="Stability CLI")


@app.command()
def dump(
    spec: str = typer.Argument(..., help="HOCON spec file"),
):
    """
    Parse a HOCON spec and print the resolved configuration as JSON.
    """
    cfg = load_hocon_spec(spec)
    typer.echo(json.dumps(cfg, indent=2, ensure_ascii=False))


@app.command()
def t2i(
    spec: str = typer.Argument(..., help="HOCON spec file"),
    out: str = typer.Option("outputs", help="Output directory"),
):
    """
    Text-to-image using HOCON spec.
    """
    print(spec)
    # cfg = load_hocon_spec(spec)
    # out_dir = Path(out)
    # out_dir.mkdir(parents=True, exist_ok=True)

    # # Qui chiami la tua funzione run_t2i
    # from stability.nodes.t2i import run_t2i
    # img_path = run_t2i(cfg, out_dir)

    # typer.echo(
    #     json.dumps(
    #         {
    #             "ok": True,
    #             "node": "t2i",
    #             "image": str(img_path),
    #         },
    #         ensure_ascii=False,
    #     )
    # )


@app.command()
def i2i(
    spec: str = typer.Argument(..., help="HOCON spec file"),
    image: str = typer.Argument(..., help="Input image"),
    out: str = typer.Option("outputs", help="Output directory"),
):
    """
    Image-to-image using HOCON spec.
    """
    print(spec)
    # cfg = load_hocon_spec(spec)
    # out_dir = Path(out)
    # out_dir.mkdir(parents=True, exist_ok=True)

    # from stability.nodes.i2i import run_i2i
    # img_path = run_i2i(cfg, image, out_dir)

    # typer.echo(
    #     json.dumps(
    #         {
    #             "ok": True,
    #             "node": "i2i",
    #             "image": str(img_path),
    #         },
    #         ensure_ascii=False,
    #     )
    # )


if __name__ == "__main__":
    app()
