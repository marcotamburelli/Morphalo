from __future__ import annotations

import importlib
import json
from typing import Optional

import typer

from stability.core.spec_loader import load_hocon_spec
from stability.dag import DagRegistry
from stability.dag.runner import DAGRunner

app = typer.Typer(
    help='Stability CLI',
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)


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
    ),
    node: Optional[str] = typer.Option(
        None,
        '--node',
        help='Execute only this node id (requires cached upstream outputs)'
    ),
):
    """
    Import a module that declares one or more DAGs and execute them.

    The module is expected to create DAGs at import time (Airflow-like).
    DAGs are collected through `DagRegistry`.

    If `--dag` is provided, only the matching DAG is executed.
    Otherwise, all discovered DAGs are executed in definition order.

    If `--node` is provided, only that node is executed (via DAGRunner.run_node),
    using cached outputs from immediate upstream nodes. If multiple DAGs are
    discovered, `--dag` must be specified to disambiguate.
    """
    DagRegistry.clear()
    importlib.import_module(module)

    dags = DagRegistry.all()
    if not dags:
        raise typer.BadParameter(
            f'No DAGs registered while importing module: {module!r}')

    # If node is requested and multiple DAGs exist, require --dag to avoid ambiguity
    if node is not None and dag is None and len(dags) > 1:
        available = [d.name for d in dags]
        raise typer.BadParameter(
            f'--node requires --dag when multiple DAGs are present. Available: {available}'
        )

    if dag is not None:
        _d = DagRegistry.get(dag)
        if _d is None:
            available = [d.name for d in dags]
            raise typer.BadParameter(
                f'DAG {dag!r} not found. Available: {available}'
            )
        dags = [_d]

    for d in dags:
        runner = DAGRunner(d)
        if node is not None:
            runner.run_node(node)
        else:
            runner.run()


if __name__ == '__main__':
    app()
