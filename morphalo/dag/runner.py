from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from morphalo.core.config import MORPHALO_CUDA_CHUNK_SIZE
from morphalo.core.paths import load_latest_output
from morphalo.dag import NodeRef
from morphalo.dag.core import DAG, Edge, get_entry_nodes
from morphalo.dag.process_executor import ProcessNodeExecutor
from morphalo.dag.validation import DagValidationError, validate_dag
from morphalo.nodes.common.env import setup_env

Output = Dict[str, Any]


@dataclass
class Execution:
    edge: Edge
    output: Optional[Output] = None


def _execute(
    out_dir: str,
    entries: List[NodeRef],
    nodes: List[NodeRef],
    executions: List[Execution],
    load_output: Optional[Callable[[str], Optional[Output]]] = None,
    node_executor: Optional[ProcessNodeExecutor] = None,
):
    def _run_node(node: NodeRef, input_map: Dict[str, Output]) -> Output:
        if node_executor is not None:
            return node_executor.run(
                node=node,
                out_dir=out_dir,
                input_map=input_map,
            )

        out = node.run(out_dir, input=input_map)
        node.post_run()
        if out is None:
            raise DagValidationError(f"No output for node '{node.id}'")
        return out

    def _resolve_output(e: Execution):
        out = e.output
        if out is None and load_output:
            out = load_output(e.edge.node_from)
        return out

    nodes_map = {n.id: n for n in nodes}

    incoming: Dict[str, List[Execution]] = {}
    outgoing: Dict[str, List[Execution]] = {}

    for e in executions:
        edge = e.edge
        incoming.setdefault(edge.node_to, []).append(e)
        outgoing.setdefault(edge.node_from, []).append(e)

    executed: set[str] = set()

    # A node is executable only when all upstream outputs required by its
    # incoming edges have already been materialized, either in memory or
    # via `load_output`.
    while entries:
        pending: set[str] = set()
        to_execute: set[str] = set()
        executed_this_round: set[str] = set()
        missing_inputs: dict[str, list[Edge]] = {}

        for node in entries:
            node_id = node.id
            in_ups: Dict[str, Optional[Output]] = {}
            for execution in incoming.get(node_id, []):
                out = _resolve_output(execution)
                in_ups[execution.edge.input_id] = out
                if out is None:
                    missing_inputs.setdefault(node_id, []).append(
                        execution.edge
                    )

            if any(x is None for x in in_ups.values()):
                pending.add(node_id)
                continue

            out = _run_node(
                node,
                {input_id: value for input_id, value in in_ups.items()
                 if value is not None},
            )

            executed.add(node_id)
            executed_this_round.add(node_id)

            for e in outgoing.get(node_id, []):
                e.output = out
                to_execute.add(e.edge.node_to)

        if pending and not executed_this_round:
            details = []
            for node_id in sorted(pending):
                missing = ', '.join(
                    f'{edge.node_from!r} on input {edge.input_id!r}'
                    for edge in missing_inputs[node_id]
                )
                details.append(f'- {node_id!r} requires {missing}')
            raise DagValidationError(
                'Execution stalled because no node can make progress. '
                'Missing upstream outputs:\n'
                + '\n'.join(details)
            )

        entries = [nodes_map[n] for n in (pending | to_execute) - executed]


def _execute_subgraph(
    *,
    dag: DAG,
    executions: List[Execution],
    target_ids: set[str],
    downstream: bool,
    force_upstream: bool,
    load_output: Callable[[str], Optional[Output]],
    node_executor: Optional[ProcessNodeExecutor] = None,
) -> None:
    validate_dag(dag)

    nodes_by_id = {node.id: node for node in dag.nodes}
    unknown = target_ids - nodes_by_id.keys()
    if unknown:
        raise ValueError(f'Unknown node ids: {sorted(unknown)}')
    if not target_ids:
        raise ValueError('At least one target node id is required')

    down: dict[str, list[str]] = {}
    up: dict[str, list[str]] = {}
    for edge in dag.edges:
        down.setdefault(edge.node_from, []).append(edge.node_to)
        up.setdefault(edge.node_to, []).append(edge.node_from)

    execution_ids = set(target_ids)

    if downstream:
        stack = list(target_ids)
        while stack:
            node_id = stack.pop()
            for child_id in down.get(node_id, []):
                if child_id in execution_ids:
                    continue
                execution_ids.add(child_id)
                stack.append(child_id)

    if force_upstream:
        cached = {
            node.id for node in dag.nodes
            if load_output(node.id) is not None
        }

        stack = list(execution_ids)
        while stack:
            node_id = stack.pop()
            for upstream_id in up.get(node_id, []):
                if upstream_id in cached or upstream_id in execution_ids:
                    continue
                execution_ids.add(upstream_id)
                stack.append(upstream_id)

    nodes = [
        node for node in dag.nodes
        if node.id in execution_ids
    ]
    edges = [
        edge for edge in dag.edges
        if edge.node_from in execution_ids and edge.node_to in execution_ids
    ]

    # External incoming edges remain available for cache hydration.
    subgraph_executions = [
        execution for execution in executions
        if execution.edge.node_to in execution_ids
    ]

    def load_external_output(node_id: str) -> Optional[Output]:
        if node_id in execution_ids:
            return None
        return load_output(node_id)

    _execute(
        out_dir=dag.out_dir,
        entries=get_entry_nodes(nodes=nodes, edges=edges),
        executions=subgraph_executions,
        nodes=nodes,
        load_output=load_external_output,
        node_executor=node_executor,
    )


class SingleNodeRunner:
    """
    Execute a single node (and optionally its downstream subgraph) using cached upstream outputs.

    ``SingleNodeRunner`` is an internal helper used by :class:`~morphalo.dag.runner.DAGRunner`
    to support incremental, CLI-driven workflows such as:

    - running one specific node using upstream cached artifacts
    - re-running a node and then re-evaluating the downstream subgraph rooted at that node

    The class is designed around filesystem- or service-backed caching: instead of requiring
    upstream nodes to run in the current process, upstream outputs may be loaded on demand
    via a caller-provided ``load_output`` function.

    Key behaviors
    -------------
    - **Input resolution**: inputs are built as ``input_id -> upstream_output`` by scanning
      edge executions with ``edge.node_to == node_id``. For each incoming edge, the upstream
      output is resolved either from memory (``Execution.output``) or via ``load_output``.
    - **Make-like upstream build**: when ``force_upstream=True``, missing upstream cached
      outputs trigger recursive execution of upstream nodes to materialize prerequisites.
      A cycle guard prevents infinite recursion in case of invalid graphs or misuse.
    - **Downstream execution**: :meth:`run_downstream` computes the downstream closure of
      ``node_id`` and executes the induced subgraph in dependency order via ``_execute``.
      When ``force_upstream=True``, the execution set may be expanded with *uncached*
      upstream dependencies of the closure so that all required inputs can be resolved.

    Parameters
    ----------
    node_id : str
        Identifier of the target node to execute.
    dag : DAG
        DAG definition containing nodes and edges. The DAG is treated as immutable.
    executions : list[Execution]
        Edge execution records (edge + optional in-memory output). This allows partial runs
        to reuse already materialized outputs without reloading them.
    force_upstream : bool
        If True, missing upstream cached outputs are resolved by recursively executing
        upstream nodes. If False, missing upstream cached outputs cause an error.
    load_output : callable
        Function that returns a cached output dict for a given node id, or ``None`` if no
        cached output exists. This is intentionally injected to decouple execution logic
        from any specific caching implementation (filesystem, database, etc.).

    Notes
    -----
    - The DAG is validated on initialization for structural correctness (acyclic, valid edges,
      unique ``input_id`` per destination node). Semantic validation of node contracts remains
      the responsibility of node implementations.
    - This runner relies on side effects for caching (e.g., nodes writing JSON artifacts).
      The returned outputs from ``node.run`` are not persisted by this class; persistence is
      expected to be handled by the node itself (or by downstream infrastructure).
    - ``executions`` are mutable and may be shared with a parent runner. Tests should take
      care to avoid cross-test contamination by recreating execution records per test case.

    Raises
    ------
    ValueError
        If ``node_id`` does not correspond to any node in the DAG.
    RuntimeError
        If an upstream cached output is missing and ``force_upstream`` is False.
    RuntimeError
        If a cycle is detected during recursive upstream execution.
    RuntimeError
        If a node produces no output (returns ``None``).
    """

    def __init__(
        self,
        node_id: str,
        dag: DAG,
        executions: List[Execution],
        force_upstream: bool,
        load_output: Callable[[str], Optional[Output]],
        node_executor: Optional[ProcessNodeExecutor] = None,
    ):
        self.node_id = node_id
        self.__dag = dag
        self._executions = executions
        self.force_upstream = force_upstream
        self.load_output = load_output
        self.node_executor = node_executor

        self.stack: set[str] = set()

        validate_dag(self.__dag)

        self.nodes_by_id = {n.id: n for n in self.__dag.nodes}
        if node_id not in self.nodes_by_id:
            raise ValueError(f'Unknown node id: {node_id!r}')

    def _ensure_upstream(self, ex: Execution) -> Output:
        node_from = ex.edge.node_from

        out = self.load_output(node_from)
        if out is not None:
            return out

        if not self.force_upstream:
            raise RuntimeError(
                f'Missing cached output for upstream node {node_from!r} '
                f'(needed by {ex.edge.node_to!r} on input {ex.edge.input_id!r}).'
            )

        self._run(node_from)  # build upstream (cycle guard is inside _run)

        out = self.load_output(node_from)
        if out is None:
            raise RuntimeError(
                f'Upstream node {node_from!r} was executed but produced no cached output '
                f'(needed by {ex.edge.node_to!r} on input {ex.edge.input_id!r}).'
            )
        return out

    def _run(self, node_id: str) -> None:
        if node_id in self.stack:
            raise RuntimeError(
                f'Cycle detected while executing {node_id!r}'
            )
        self.stack.add(node_id)

        try:
            upstream_executions = [
                e for e in self._executions if e.edge.node_to == node_id
            ]

            # populate upstream outputs (from cache and/or forced upstream build)
            for ex in upstream_executions:
                ex.output = self._ensure_upstream(ex)

            input_map = {
                ex.edge.input_id: ex.output for ex in upstream_executions
            }

            node = self.nodes_by_id[node_id]

            if self.node_executor is not None:
                out = self.node_executor.run(
                    node=node,
                    out_dir=self.__dag.out_dir,
                    input_map=input_map,
                )
            else:
                out = node.run(
                    self.__dag.out_dir,
                    input=input_map
                )
                node.post_run()

            if out is None:
                raise RuntimeError(f'No output for node {node_id!r}')

        finally:
            # always clean up stack, even if node.run() raises
            self.stack.remove(node_id)

    def run(self):
        self._run(self.node_id)

    def run_downstream(self):
        _execute_subgraph(
            dag=self.__dag,
            executions=self._executions,
            target_ids={self.node_id},
            downstream=True,
            force_upstream=self.force_upstream,
            load_output=self.load_output,
            node_executor=self.node_executor,
        )


class DAGRunner:
    """
    Executes a DAG by resolving node dependencies and propagating outputs
    along edges until all nodes have been evaluated.

    The runner operates on a static DAG definition (nodes + edges) and
    performs the following steps:

    1. Validates the DAG structure (acyclic, valid edges).
    2. Tracks edge executions, where each edge carries the output produced
       by its source node.
    3. Executes nodes only when all their incoming edges have produced outputs.
    4. Propagates each node's output to all downstream edges.

    Execution model assumptions:
    - Each node produces a single primary output (a dict).
    - Outputs are immutable once produced.
    - Input for a node is built as a mapping:
          input_id -> upstream node output
    - Nodes with no incoming edges are considered source nodes and are
      executed with an empty input dictionary.

    The runner is deterministic and sequential. Node execution is delegated
    to a restartable worker process so CUDA contexts can be recycled after a
    bounded number of CUDA node executions. This is a defensive mitigation for
    observed low-level GPU/driver instability associated with long-lived CUDA
    contexts; it is not primarily an out-of-memory recovery mechanism.
    Parallelism, retries, and partial re-execution are intentionally
    out of scope.
    """

    def __init__(
        self,
        dag: DAG,
        *,
        max_cuda_nodes: int = MORPHALO_CUDA_CHUNK_SIZE,
    ):
        """
        Initializes a DAGRunner for a given DAG.

        This constructor does not execute the DAG. It only prepares
        the internal execution state by creating one execution record
        for each edge in the DAG.

        Each execution record tracks:
        - the edge itself
        - the output produced by the source node of that edge (initially None)

        Parameters
        ----------
        dag : DAG
            The directed acyclic graph to be executed. The DAG is assumed
            to be immutable during execution.
        max_cuda_nodes : int, optional
            Maximum number of successful CUDA node executions per worker
            process. CPU nodes do not increment the counter and may continue
            in the current worker after the limit is reached. Recycling the
            worker destroys its CUDA context before the next CUDA chunk.
        """

        setup_env()

        self.__dag = dag
        if max_cuda_nodes <= 0:
            raise ValueError(
                f'max_cuda_nodes must be > 0, got {max_cuda_nodes}'
            )
        self.max_cuda_nodes = int(max_cuda_nodes)
        self._executions = [Execution(edge=edge)
                            for edge in self.__dag.edges]

    def run(self):
        """
        Executes the DAG.

        Nodes are executed in dependency order: a node is executed only
        when all its incoming edges have produced an output. Source nodes
        (nodes with no incoming edges) are executed first with an empty
        input dictionary.

        For each executed node:
        - `node.run(output_dir, input=...)` is called.
        - The returned output is propagated to all outgoing edges.
        - Downstream nodes may become eligible for execution.

        If the execution reaches a state where no remaining node can be
        executed (i.e. missing inputs or cyclic dependencies), the method
        raises a RuntimeError.

        Raises
        ------
        RuntimeError
            If the DAG contains a cycle, unresolved dependencies, or if
            execution cannot make further progress.

        Notes
        -----
        - This method assumes the DAG has already been validated.
        - Worker failures are raised as process execution errors.
        - Execution is strictly sequential and deterministic.

        Returns
        -------
        None
            The runner does not return a value. Side effects are produced
            by node execution and propagated through the DAG.
        """

        # Validating the DAG
        validate_dag(self.__dag)

        with ProcessNodeExecutor(
            max_cuda_nodes=self.max_cuda_nodes
        ) as node_executor:
            _execute(
                out_dir=self.__dag.out_dir,
                entries=get_entry_nodes(
                    nodes=self.__dag.nodes,
                    edges=self.__dag.edges
                ),
                executions=self._executions,
                nodes=self.__dag.nodes,
                node_executor=node_executor,
            )

    def run_node(self, target_id: str, force_upstream: bool = False):
        """
        Execute a single node of the DAG using cached outputs from its upstream nodes.

        This method runs only the specified target node, without re-executing the
        entire DAG. Inputs for the target node are loaded from the filesystem by
        retrieving the most recent JSON outputs produced by its immediate upstream
        nodes.

        The method is intended for CLI-driven or incremental workflows, where node
        execution is side-effect based and outputs are materialized on disk rather
        than returned to the caller.

        Parameters
        ----------
        target_id : str
            Identifier of the node to execute.
        force_upstream : bool, optional
            If ``True``, missing upstream outputs trigger execution of upstream
            nodes (recursively) in order to materialize the required inputs.
            If ``False`` (default), the method fails if any required upstream
            output is not already available on disk.

        Raises
        ------
        ValueError
            If ``target_id`` does not correspond to any node in the DAG.
        RuntimeError
            If one or more upstream nodes have no cached output available on disk
            and ``force_upstream`` is ``False``.
        RuntimeError
            If an upstream node is executed but still produces no output.
        RuntimeError
            If the target node produces no output.

        Notes
        -----
        - The DAG structure is validated before execution.
        - Only *immediate* upstream dependencies of ``target_id`` are considered.
        Transitive upstream nodes are resolved on demand only when
        ``force_upstream`` is enabled.
        - Cached outputs are loaded via :func:`load_latest_output`, which selects
        the most recent JSON artifact in each upstream node's output directory.
        - When ``force_upstream`` is enabled, this method behaves similarly to a
        make-like build: missing dependencies are executed before the target
        node.
        - The execution relies entirely on filesystem side effects; no value is
        returned to the caller.
        """

        with ProcessNodeExecutor(
            max_cuda_nodes=self.max_cuda_nodes
        ) as node_executor:
            SingleNodeRunner(
                node_id=target_id,
                dag=self.__dag,
                executions=self._executions,
                force_upstream=force_upstream,
                load_output=lambda n: load_latest_output(
                    out_dir=self.__dag.out_dir,
                    node_id=n
                ),
                node_executor=node_executor,
            ).run()

    def run_node_downstream(self, target_id: str, force_upstream: bool = False):
        """
        Execute a node and re-evaluate the downstream subgraph rooted at that node.

        This method performs a partial DAG execution starting from ``target_id`` by
        computing the set of nodes reachable downstream (the *downstream closure*)
        and executing all nodes in that subgraph in dependency order.

        Execution is side-effect based: node outputs are expected to be materialized
        to disk (JSON sidecars), and upstream inputs for nodes can be resolved either
        from in-memory propagated outputs (within this run) or from cached outputs on
        disk, as implemented by :meth:`_execute_downstream`.

        The method supports two modes:

        - ``force_upstream=False`` (default):
        Only nodes in the downstream closure of ``target_id`` are eligible for
        execution. Any *lateral* dependency (an upstream node outside the closure)
        must already have a cached output available on disk; otherwise execution
        will stall or fail when inputs cannot be resolved.

        - ``force_upstream=True``:
        In addition to the downstream closure, the method expands the execution set
        with upstream dependencies of the closure that do not have cached outputs.
        This make-like behavior materializes missing prerequisites so that the
        downstream re-execution can proceed.

        Parameters
        ----------
        target_id : str
            Identifier of the node that defines the root of the downstream subgraph
            to re-execute.
        force_upstream : bool, optional
            If ``True``, include in the execution set any upstream dependencies
            of the downstream closure that are missing cached outputs on disk.
            If ``False``, dependencies outside the downstream closure are expected
            to be already cached.

        Raises
        ------
        DagValidationError
            If the DAG is invalid (e.g., cyclic, invalid edges) or if a node produces
            no output during execution.
        RuntimeError
            If execution cannot make progress because required inputs cannot be
            resolved (e.g., missing cached outputs for lateral dependencies when
            ``force_upstream`` is ``False``).

        Notes
        -----
        - The DAG is validated before any execution.
        - The downstream closure is computed via a DFS/BFS over the adjacency list
        built from DAG edges.
        - Cached status is determined by the presence of a latest JSON artifact
        retrievable via :func:`load_latest_output`.
        - When ``force_upstream`` is enabled, only *uncached* upstream nodes are
        added; cached upstream nodes remain external and will be read from disk
        as needed.
        - This method does not return a value; it relies on filesystem side effects
        and in-memory propagation through :meth:`_execute_downstream`.
        """

        with ProcessNodeExecutor(
            max_cuda_nodes=self.max_cuda_nodes
        ) as node_executor:
            SingleNodeRunner(
                node_id=target_id,
                dag=self.__dag,
                executions=self._executions,
                force_upstream=force_upstream,
                load_output=lambda n: load_latest_output(
                    out_dir=self.__dag.out_dir,
                    node_id=n
                ),
                node_executor=node_executor,
            ).run_downstream()

    def run_group(
        self,
        group_id: str,
        *,
        downstream: bool = False,
        force_upstream: bool = False,
    ):
        """
        Re-execute all nodes in an injected node group.

        External inputs are loaded from cache unless ``force_upstream`` is
        enabled, in which case missing prerequisites are executed recursively.
        When ``downstream`` is enabled, all nodes reachable from any node in
        the group are re-executed as well.
        """
        group = self.__dag.node_groups.get(group_id)
        if group is None:
            raise ValueError(
                f'Unknown node group {group_id!r}. '
                f'Available: {sorted(self.__dag.node_groups)}'
            )

        with ProcessNodeExecutor(
            max_cuda_nodes=self.max_cuda_nodes
        ) as node_executor:
            _execute_subgraph(
                dag=self.__dag,
                executions=self._executions,
                target_ids={node.id for node in group.nodes},
                downstream=downstream,
                force_upstream=force_upstream,
                load_output=lambda node_id: load_latest_output(
                    out_dir=self.__dag.out_dir,
                    node_id=node_id,
                ),
                node_executor=node_executor,
            )
