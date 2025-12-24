from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Optional

import typer

from stability.core.spec_loader import load_hocon_spec
from stability.dag import DAG, DagRegistry
from stability.dag.runner import DAGRunner
from stability.dag.validation import validate_dag

app = typer.Typer(help='Stability CLI')


@app.command()
def dump(
    spec: str = typer.Argument(..., help='HOCON spec file'),
):
    """Parse a HOCON spec and print the resolved configuration as JSON."""
    cfg = load_hocon_spec(spec)
    typer.echo(json.dumps(cfg, indent=2, ensure_ascii=False))


@app.command('run-dags')
def run_dags(
    module: str = typer.Argument(
        ...,
        help='Python module that declares DAGs, e.g. dags.demo_depth_cn'
    ),
    dag: Optional[str] = typer.Option(
        None,
        '--dag',
        help='Run only the DAG with this name'
    )
):
    """
    Import a module that declares one or more DAGs and execute them.

    The module is expected to create DAGs at import time (Airflow-like).
    DAGs are collected through `DagRegistry`.

    If `--dag` is provided, only the matching DAG is executed.
    Otherwise, all discovered DAGs are executed in definition order.
    """
    DagRegistry.clear()
    importlib.import_module(module)

    dags = DagRegistry.all()
    if not dags:
        raise typer.BadParameter(
            f'No DAGs registered while importing module: {module!r}')

    if dag is not None:
        dags = [DagRegistry.get(dag)]
        if not dags:
            available = [d.name for d in DagRegistry.all()]
            raise typer.BadParameter(
                f'DAG {dag!r} not found. Available: {available}')

    for d in dags:
        # out_dir = Path(out or d.out_dir)
        # out_dir.mkdir(parents=True, exist_ok=True)

        DAGRunner(d).run()


if __name__ == '__main__':
    app()
