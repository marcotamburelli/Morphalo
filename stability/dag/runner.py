from typing import Dict, List, Optional

from stability.core.paths import load_latest_output
from stability.dag import *
from stability.dag.validation import DagValidationError, validate_dag


@dataclass
class Execution:
    edge: Edge
    output: Optional[Dict] = None


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

    The runner is deterministic and single-threaded.
    Parallelism, retries, and partial re-execution are intentionally
    out of scope.
    """

    _executions: List[Execution] = []

    def __init__(self, dag: DAG):
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
        """

        self.__dag = dag
        self._executions = [Execution(edge=edge)
                            for edge in self.__dag.edges]

    def _execute_downstream(self, node_to_execute: Dict[str, NodeRef], use_cache: bool = False):
        prev_pending_ids: Optional[frozenset[str]] = None

        while node_to_execute:
            pending: Dict[str, NodeRef] = {}
            next_to_execute: Dict[str, NodeRef] = {}

            for node in node_to_execute.values():
                source_executions = [
                    execution for execution in self._executions
                    if execution.edge.node_to == node.id
                ]

                is_pending = False
                has_next = False

                # Fill executions from memory or cache
                for execution in source_executions:
                    output = execution.output
                    if use_cache and output is None:
                        output = load_latest_output(
                            out_dir=self.__dag.out_dir,
                            node_id=execution.edge.node_from
                        )

                    if output is None:
                        is_pending = True
                        break
                    execution.output = output

                if is_pending:
                    pending[node.id] = node
                    continue

                input = {
                    execution.edge.input_id: execution.output
                    for execution in source_executions
                }

                out = node.run(self.__dag.out_dir, input=input)
                if not out:
                    raise DagValidationError(f"No output for node '{node.id}'")

                for execution in self._executions:
                    if execution.edge.node_from == node.id:
                        execution.output = out
                        has_next = True
                        # enqueue downstream
                        for n in self.__dag.nodes:
                            if n.id == execution.edge.node_to and n.id not in node_to_execute:
                                next_to_execute[n.id] = n

            pending_ids = frozenset(pending.keys())
            # Check to avoid potential loops (TODO Ensure it is really needed)
            if not has_next and pending_ids and pending_ids == prev_pending_ids:
                raise RuntimeError(
                    f"Execution stalled. Missing cached upstream outputs for: {sorted(pending_ids)}"
                )
            prev_pending_ids = pending_ids

            node_to_execute = {**pending, **next_to_execute}

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
        - The runner does not catch exceptions raised by node execution;
          failures propagate immediately.
        - Execution is strictly sequential and deterministic.

        Returns
        -------
        None
            The runner does not return a value. Side effects are produced
            by node execution and propagated through the DAG.
        """

        # Validating the DAG
        validate_dag(self.__dag)

        # Detect initial nodes
        node_to_execute: Dict[str, NodeRef] = {
            node.id: node for node in self.__dag.nodes if
            all(edge.node_to != node.id for edge in self.__dag.edges)
        }
        self._execute_downstream(node_to_execute)

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

        validate_dag(self.__dag)

        nodes_by_id = {n.id: n for n in self.__dag.nodes}
        if target_id not in nodes_by_id:
            raise ValueError(f"Unknown node id: {target_id!r}")

        stack: set[str] = set()

        def _load(node_from: str):
            return load_latest_output(out_dir=self.__dag.out_dir, node_id=node_from)

        def _ensure_upstream(ex: Execution) -> dict:
            """
            Ensure upstream output exists on disk; optionally build it.
            """

            node_from = ex.edge.node_from

            out = _load(node_from)
            if out is not None:
                return out

            if not force_upstream:
                raise RuntimeError(
                    f"Missing cached output for upstream node {node_from!r} "
                    f"(needed by {ex.edge.node_to!r} on input {ex.edge.input_id!r})."
                )

            _run(node_from)  # build upstream (cycle guard is inside _run)

            out = _load(node_from)
            if out is None:
                raise RuntimeError(
                    f"Upstream node {node_from!r} was executed but produced no cached output "
                    f"(needed by {ex.edge.node_to!r} on input {ex.edge.input_id!r})."
                )
            return out

        def _run(node_id: str) -> None:
            if node_id in stack:
                raise RuntimeError(
                    f"Cycle detected while executing {node_id!r}")
            stack.add(node_id)

            upstream_executions = [
                e for e in self._executions if e.edge.node_to == node_id
            ]

            # populate upstream outputs
            for ex in upstream_executions:
                ex.output = _ensure_upstream(ex)

            input_map = {
                ex.edge.input_id: ex.output for ex in upstream_executions}

            out = nodes_by_id[node_id].run(self.__dag.out_dir, input=input_map)
            if not out:
                raise RuntimeError(f"No output for node {node_id!r}")

            stack.remove(node_id)

        _run(target_id)

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
        validate_dag(self.__dag)

        # building adjacency lists
        down: dict[str, list[str]] = {}
        up: dict[str, list[str]] = {}
        for e in self.__dag.edges:
            down.setdefault(e.node_from, []).append(e.node_to)
            up.setdefault(e.node_to, []).append(e.node_from)

        down_closure = {target_id}

        stack = list(down.get(target_id, []))

        while stack:
            n = stack.pop()

            if n in down_closure:
                continue

            down_closure.add(n)
            stack.extend(down.get(n, []))

        R = set(down_closure)

        if force_upstream:
            cached = {
                n.id for n in self.__dag.nodes
                if load_latest_output(self.__dag.out_dir, n.id) is not None
            }

            # expand with uncached upstream
            q = list(down_closure)
            while q:
                r = q.pop()
                for u in up.get(r, []):
                    if u in cached or u in R:
                        continue
                    R.add(u)
                    q.append(u)

        node_to_execute = {
            n.id: n for n in self.__dag.nodes if n.id in R
        }
        self._execute_downstream(node_to_execute, use_cache=True)
