from morphalo.dag import DAG


class DagValidationError(RuntimeError):
    pass


def validate_dag(dag: DAG) -> None:
    """
    Validates the structural correctness of a DAG before execution.

    This function performs a series of static checks to ensure that the DAG
    can be safely executed by a DAGRunner. Validation is purely structural
    and does not execute any node logic.

    The following conditions are verified:

    1. All edges reference existing source and destination nodes.
    2. Each input identifier (``input_id``) is unique per destination node.
       Duplicate ``input_id`` values for the same node would cause input
       overwriting during execution.
    3. The graph is acyclic (i.e. it is a true Directed Acyclic Graph).

    If any of these conditions is violated, a ``DagValidationError`` is raised
    and execution must not proceed.

    Parameters
    ----------
    dag : DAG
        The directed acyclic graph to validate. The DAG is treated as
        immutable during validation.

    Raises
    ------
    DagValidationError
        If one or more of the following conditions hold:

        - An edge references a node that does not exist in the DAG.
        - Two or more edges target the same node with the same ``input_id``.
        - The graph contains at least one cycle.

    Returns
    -------
    None

    Notes
    -----
    This function should be invoked before executing the DAG. The runner
    assumes that a validated DAG is free of structural deadlocks and cyclic
    dependencies.

    Validation does not check semantic constraints of individual nodes
    (e.g. required inputs for a specific node type). Such checks should
    be implemented separately, either in node-specific validation logic
    or during execution.
    """
    node_ids = {n.id for n in dag.nodes}

    # 1) edge -> node existence
    for e in dag.edges:
        if e.node_from not in node_ids:
            raise DagValidationError(
                f"Edge refers to missing source node '{e.node_from}' to '{e.node_to}'"
            )
        if e.node_to not in node_ids:
            raise DagValidationError(
                f"Edge refers to missing destination node '{e.node_to}' from '{e.node_from}'"
            )

    # 2) duplicate input_id on same destination
    seen: set[tuple[str, str]] = set()
    for e in dag.edges:
        key = (e.node_to, e.input_id)
        if key in seen:
            raise DagValidationError(
                f"Duplicate input_id '{e.input_id}' for node '{e.node_to}'. "
                'This would overwrite inputs.'
            )
        seen.add(key)

    # 3) cycle detection
    _assert_acyclic(dag)


def _find_cycle(
    node_ids: list[str],
    out: dict[str, list[str]],
) -> list[str] | None:
    """
    Find one directed cycle in a graph.

    Parameters
    ----------
    node_ids : list of str
        Node identifiers to inspect.
    out : dict of str to list of str
        Adjacency list mapping each node to its outgoing neighbors.

    Returns
    -------
    list of str or None
        A concrete cycle path if one exists.

        The returned list repeats the first node as the last element, for example::

            ['a', 'b', 'c', 'a']

        If no cycle is found, returns ``None``.

    Notes
    -----
    This function uses DFS node coloring:

    - white / unseen: node has not been visited yet.
    - gray / visiting: node is currently on the DFS recursion stack.
    - black / done: node and its descendants have been fully explored.

    An edge pointing to a gray node is a back edge and identifies a cycle.
    """

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def visit(node_id: str) -> list[str] | None:
        visiting.add(node_id)
        stack_index[node_id] = len(stack)
        stack.append(node_id)

        for next_id in out.get(node_id, []):
            if next_id in visited:
                continue

            if next_id in visiting:
                i = stack_index[next_id]
                return stack[i:] + [next_id]

            cycle = visit(next_id)
            if cycle is not None:
                return cycle

        stack.pop()
        stack_index.pop(node_id, None)
        visiting.remove(node_id)
        visited.add(node_id)

        return None

    for node_id in node_ids:
        if node_id not in visited:
            cycle = visit(node_id)
            if cycle is not None:
                return cycle

    return None


def _assert_acyclic(dag: DAG) -> None:
    node_ids = [n.id for n in dag.nodes]

    indeg: dict[str, int] = {nid: 0 for nid in node_ids}
    out: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for e in dag.edges:
        out[e.node_from].append(e.node_to)
        indeg[e.node_to] += 1

    q = [nid for nid in node_ids if indeg[nid] == 0]
    visited = 0

    while q:
        n = q.pop()
        visited += 1
        for m in out[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)

    if visited != len(node_ids):
        cycle = _find_cycle(node_ids=node_ids, out=out)

        if cycle is None:
            cyclic = [nid for nid, d in indeg.items() if d > 0]
            raise DagValidationError(
                f'DAG contains a cycle. Remaining cyclic candidates: {cyclic}'
            )

        raise DagValidationError(
            'DAG contains a cycle: '
            + ' -> '.join(cycle)
        )
