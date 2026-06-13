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
    group: Optional[str] = typer.Option(
        None,
        '--group',
        help='Execute all nodes in this injected node group'
    ),
    downstream: bool = typer.Option(
        False,
        '--downstream',
        help='With --node or --group, also execute all reachable downstream nodes'
    ),
    force_upstream: bool = typer.Option(
        False,
        '--force-upstream',
        help='With --node or --group, execute missing upstream nodes'
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
        If neither ``--node`` nor ``--group`` are specified, the selected DAGs
        are executed in definition order via :meth:`DAGRunner.run`.

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

    Node group execution
        If ``--group <id>`` is provided, every node belonging to the injected
        group is re-executed via :meth:`DAGRunner.run_group`.

        - Inputs from outside the group are loaded from cached outputs.
        - If ``--force-upstream`` is set, missing external prerequisites are
          executed recursively.
        - Nested groups can be selected by qualified id, e.g. ``outer.inner``.

    Downstream re-execution
        If ``--downstream`` is combined with ``--node`` or ``--group``, the
        selected target and all reachable downstream nodes are re-executed.

        - The downstream closure of the selected node or group is computed.
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
        the module and ``--node`` or ``--group`` is used, this option is
        required to avoid ambiguity.
    node : str, optional
        Identifier of a specific node to execute instead of the full DAG.
    group : str, optional
        Qualified identifier of an injected node group to re-execute.
    downstream : bool, optional
        Execute all nodes reachable downstream from the selected node or group.
    force_upstream : bool, optional
        Execute missing upstream nodes recursively for the selected node or
        group.
    cuda_chunk_size : int, optional
        Maximum number of CUDA node executions per worker process. The worker
        restart destroys the current CUDA context before the next chunk, as a
        mitigation for observed low-level GPU/driver instability after long
        inference sequences. Must be greater than zero.

    Raises
    ------
    typer.BadParameter
        If no DAGs are registered in the imported module, if a specified DAG
        does not exist, if a partial target is ambiguous across multiple DAGs,
        if ``--node`` and ``--group`` are combined, or if ``--downstream`` is
        used without a partial target.

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

    if node is not None and group is not None:
        raise typer.BadParameter(
            '--node and --group are mutually exclusive'
        )

    partial_target = node is not None or group is not None

    # Partial targets are ambiguous when a module registers multiple DAGs.
    if partial_target and dag is None and len(dags) > 1:
        available = [d.name for d in dags]
        raise typer.BadParameter(
            '--node/--group requires --dag when multiple DAGs are present. '
            f'Available: {available}'
        )

    if downstream and not partial_target:
        raise typer.BadParameter(
            '--downstream can only be used with --node or --group'
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
        elif group is not None:
            runner.run_group(
                group,
                downstream=downstream,
                force_upstream=force_upstream,
            )
        else:
            runner.run()


if __name__ == '__main__':
    app()
