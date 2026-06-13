from __future__ import annotations

import importlib
import json
from typing import Optional

import typer

from morphalo.core.spec_loader import load_hocon_spec
from morphalo.dag import DagRegistry
from morphalo.dag.runner import DAGRunner

app = typer.Typer(
    help='Morphalo CLI',
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
    downstream: bool = typer.Option(
        False,
        '--downstream',
        help='When used with --node, execute the node and all nodes reachable downstream'
    ),
    force_upstream: bool = typer.Option(
        False,
        '--force-upstream',
        help='When used with --node, execute missing upstream nodes to materialize inputs'
    ),
    cuda_chunk_size: int = typer.Option(
        8,
        '--cuda-chunk-size',
        min=1,
        help=(
            'Restart the node worker after this many CUDA node executions '
            'to limit CUDA context lifetime'
        )
    ),
):
    """
    Import a module that declares one or more DAGs and execute them.

    The specified Python module is imported dynamically. DAGs are expected to be
    instantiated at import time (Airflow-like pattern) and registered via
    ``DagRegistry``. All discovered DAGs are then executed according to the
    provided options.

    Execution modes
    ---------------
    Full DAG execution (default)
        If neither ``--node`` nor ``--dag`` are specified, all DAGs declared in
        the module are executed in definition order via :meth:`DAGRunner.run`.

    Single DAG execution
        If ``--dag <name>`` is provided, only the matching DAG is executed.
        An error is raised if the DAG name is not found.

    Single node execution
        If ``--node <id>`` is provided, only the specified node is executed
        within the selected DAG using :meth:`DAGRunner.run_node`.

        - By default, all immediate upstream dependencies of the node must
          already have cached outputs on disk.
        - If ``--force-upstream`` is set, missing upstream nodes are executed
          recursively (make-like behavior) to materialize required inputs.

    Downstream re-execution
        If ``--node`` and ``--downstream`` are both specified, the runner
        re-executes the downstream subgraph rooted at the given node via
        :meth:`DAGRunner.run_node_downstream`.

        - The downstream closure of the node is computed.
        - If ``--force-upstream`` is not set, lateral dependencies outside
          the closure must already be cached on disk.
        - If ``--force-upstream`` is set, uncached upstream dependencies of
          the downstream closure are included and executed as needed.

    Parameters
    ----------
    module : str
        Fully qualified Python module path that declares one or more DAGs
        (e.g. ``dags.demo_depth_cn``).
    dag : str, optional
        Name of a specific DAG to execute. If multiple DAGs are defined in
        the module and ``--node`` is used, this option is required to avoid
        ambiguity.
    node : str, optional
        Identifier of a specific node to execute instead of the full DAG.
    downstream : bool, optional
        When used together with ``--node``, execute the node and all nodes
        reachable downstream from it.
    force_upstream : bool, optional
        When used with ``--node`` (and optionally ``--downstream``), execute
        missing upstream nodes recursively to materialize required inputs.
    cuda_chunk_size : int, optional
        Maximum number of CUDA node executions per worker process. The worker
        restart destroys the current CUDA context before the next chunk, as a
        mitigation for observed low-level GPU/driver instability after long
        inference sequences. Must be greater than zero.

    Raises
    ------
    typer.BadParameter
        If no DAGs are registered in the imported module, if a specified DAG
        does not exist, if ``--node`` is ambiguous across multiple DAGs, or
        if ``--downstream`` is used without ``--node``.

    Notes
    -----
    - DAG discovery occurs at module import time via side effects.
    - Execution is deterministic and single-threaded.
    - CUDA work is divided into worker-process chunks to limit CUDA context
      lifetime. This is a defensive mitigation for observed driver/GSP-level
      instability, not a substitute for normal OOM handling.
    - All node execution relies on filesystem side effects for input/output
      materialization (JSON artifacts).
    - This command does not return values; results are persisted to disk.
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

    if downstream and node is None:
        raise typer.BadParameter(
            '--downstream can only be used together with --node'
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
        runner = DAGRunner(d, max_cuda_nodes=cuda_chunk_size)
        if node is not None:
            if downstream:
                runner.run_node_downstream(node, force_upstream=force_upstream)
            else:
                runner.run_node(node, force_upstream=force_upstream)
        else:
            runner.run()


if __name__ == '__main__':
    app()
