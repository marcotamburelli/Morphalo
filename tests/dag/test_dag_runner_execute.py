from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pytest

from morphalo.dag import DAG, Edge, get_entry_nodes
from morphalo.dag.runner import Execution, _execute
from morphalo.dag.validation import DagValidationError
from tests.dag.nodes import MergeNode, NoneNode, PassNode, SourceNode


def _run_execute(dag: DAG, *, load_output: Optional[Callable[[str], Dict[str, Any]]] = None) -> list[Execution]:
    """
    Execute only the pure engine ``_execute`` for a given DAG.

    This helper builds one ``Execution`` per DAG edge, computes entry nodes,
    and calls ``_execute``.

    Parameters
    ----------
    dag : DAG
        The DAG to execute.
    load_output : callable, optional
        Function used by ``_execute`` to lazily load missing upstream outputs
        by source node id.

    Returns
    -------
    list[Execution]
        Execution records (edge + propagated output).
    """
    executions = [Execution(edge=e) for e in dag.edges]
    entries = get_entry_nodes(nodes=dag.nodes, edges=dag.edges)

    _execute(
        out_dir=str(dag.out_dir),
        entries=entries,
        nodes=dag.nodes,
        executions=executions,
        load_output=load_output,
    )
    return executions


def _exec_index(executions: list[Execution]) -> dict[tuple[str, str, str], Execution]:
    """
    Index executions by (node_from, node_to, input_id).

    This makes edge-level assertions stable and explicit.
    """
    return {(e.edge.node_from, e.edge.node_to, e.edge.input_id): e for e in executions}


def test_execute_linear_propagates_outputs(tmp_path):
    """
    Graph
    -----
    A -> B

    Assertions
    ----------
    - A runs once, B runs once
    - B receives A's output under input_id 'default'
    - The edge (A->B, default) carries A's output
    """
    with DAG('linear', out_dir=tmp_path) as dag:
        a = SourceNode(name='A', value=10)
        b = PassNode(name='B')
        a >> b

    executions = _run_execute(dag)
    idx = _exec_index(executions)

    assert a.calls == 1
    assert b.calls == 1

    assert b.last_input is not None
    assert set(b.last_input.keys()) == {'default'}
    assert b.last_input['default']['value'] == 10
    assert b.last_input['default']['from'] == a.id

    ex = idx[(a.id, b.id, 'default')]
    assert ex.output is not None
    assert ex.output['value'] == 10
    assert ex.output['from'] == a.id


def test_execute_fanout_propagates_to_multiple_downstream(tmp_path):
    """
    Graph
    -----
         -> B
    A --
         -> C

    Assertions
    ----------
    - A runs once, B runs once, C runs once
    - Both B and C receive A output
    - Two distinct edges carry the same payload from A
    """
    with DAG('fanout', out_dir=tmp_path) as dag:
        a = SourceNode(name='A', value=7)
        b = PassNode(name='B')
        c = PassNode(name='C')
        a >> [b, c]

    executions = _run_execute(dag)
    idx = _exec_index(executions)

    assert a.calls == 1
    assert b.calls == 1
    assert c.calls == 1

    assert b.last_input is not None
    assert c.last_input is not None

    assert b.last_input['default']['value'] == 7
    assert c.last_input['default']['value'] == 7

    ex_ab = idx[(a.id, b.id, 'default')]
    ex_ac = idx[(a.id, c.id, 'default')]

    assert ex_ab.output is not None
    assert ex_ac.output is not None
    assert ex_ab.output['value'] == 7
    assert ex_ac.output['value'] == 7


def test_execute_merge_waits_for_all_inputs_and_merges(tmp_path):
    """
    Graph
    -----
    A -> X -> C(attachment)
    B ------> C(default)

    Assertions
    ----------
    - C runs only after both incoming inputs are available
    - C receives exactly two inputs: 'default' and 'attachment'
    - merged contains both values keyed by input_id
    """
    with DAG('merge', out_dir=tmp_path) as dag:
        a = SourceNode(name='A', value=1)
        x = PassNode(name='X')
        a >> x

        b = SourceNode(name='B', value=2)
        c = MergeNode(name='C')

        b >> c
        x >> c.sink(name='X_to_C', input_id='attachment')

    _run_execute(dag)

    assert a.calls == 1
    assert x.calls == 1
    assert b.calls == 1
    assert c.calls == 1

    assert c.last_input is not None
    assert set(c.last_input.keys()) == {'default', 'attachment'}

    assert c.last_input['default']['value'] == 2
    assert c.last_input['attachment']['value'] == 1

    # Also validate the merge output shape
    assert c.outputs[-1]['merged'] == {'default': 2, 'attachment': 1}


def test_execute_raises_if_node_returns_none(tmp_path):
    """
    Graph
    -----
    N

    Assertions
    ----------
    - If a node returns None, _execute raises DagValidationError
    """
    with DAG('none_out', out_dir=tmp_path) as dag:
        n = NoneNode(name='N')

    with pytest.raises(DagValidationError, match='No output for node'):
        _run_execute(dag)


def test_execute_load_output_unblocks_pending_node(tmp_path):
    """
    Scenario
    --------
    We simulate a situation where a node is "ready" only after load_output provides
    a missing upstream output.

    Graph (manual)
    --------------
    A -> C(default)

    But A is not part of nodes executed in this run. Instead, load_output('A') returns
    a cached output dict, enabling C to run.

    Assertions
    ----------
    - C runs once
    - C receives input from load_output under input_id 'default'
    """
    with DAG('load_output_unblocks', out_dir=tmp_path) as dag:
        c = PassNode(name='C')

    # Manually create a single edge from a missing upstream id 'A' into 'C'
    edge = Edge(node_from='A', node_to=c.id, input_id='default')
    executions = [Execution(edge=edge)]
    entries = [c]

    def loader(node_id: str) -> Optional[Dict[str, Any]]:
        if node_id == 'A':
            return {'value': 42, 'from': 'A'}
        return None

    _execute(
        out_dir=str(tmp_path),
        entries=entries,
        nodes=[c],
        executions=executions,
        load_output=loader,
    )

    assert c.calls == 1
    assert c.last_input is not None
    assert c.last_input['default']['value'] == 42
    assert c.outputs[-1]['value'] == 42


def test_execute_stuck_fails_immediately_with_missing_input_details(tmp_path):
    """
    Scenario
    --------
    We construct an execution state where an entry node has an incoming edge
    whose output is never produced and never loadable.

    Graph (manual)
    --------------
    A -> C(default)

    Node A is missing and load_output always returns None, so C cannot make
    progress.

    Assertions
    ----------
    - _execute raises DagValidationError immediately
    - the error identifies the blocked node, upstream node, and input id
    """
    with DAG('stuck', out_dir=tmp_path) as dag:
        c = PassNode(name='C')

    edge = Edge(node_from='A', node_to=c.id, input_id='default')
    executions = [Execution(edge=edge)]
    entries = [c]

    def loader(_node_id: str) -> None:
        return None

    with pytest.raises(
        DagValidationError,
        match=(
            "Execution stalled because no node can make progress[\\s\\S]*"
            "'C' requires 'A' on input 'default'"
        ),
    ):
        _execute(
            out_dir=str(tmp_path),
            entries=entries,
            nodes=[c],
            executions=executions,
            load_output=loader,
        )


def test_execute_intricate_dag_schedules_correctly_and_propagates(tmp_path):
    """
    Intricate graph that mixes fan-out, fan-in, and an attachment input.

    Graph
    -----
           -> P1 -> \
    S1 ----          \
                     -> M -> E -> Z
    S2 -> P2(attach) /

    Side branch (independent)
    -------------------------
    S3 -> Q1 -> Q2

    Assertions
    ----------
    - Nodes in the main component run once: S1, P1, S2, P2, M, E, Z
    - Side branch runs once too: S3, Q1, Q2 (because it is an independent component)
    - Merge node receives both 'default' and 'attachment'
    - Z receives a value propagated from the merged path (through ExtractNode)
    """
    from tests.dag.nodes import \
        ExtractNode  # local import to keep test file self-contained

    with DAG('intricate', out_dir=tmp_path) as dag:
        # Main component
        s1 = SourceNode(name='S1', value=10)
        p1 = PassNode(name='P1')
        s1 >> p1

        s2 = SourceNode(name='S2', value=20)
        p2 = PassNode(name='P2')
        s2 >> p2

        m = MergeNode(name='M')
        p1 >> m
        p2 >> m.sink(name='P2_to_M', input_id='attachment')

        # Convert MergeNode output to a 'value' payload (pick merged['attachment'])
        e = ExtractNode(name='E', key='attachment')
        z = PassNode(name='Z')
        m >> e >> z

        # Independent side component (should also execute)
        s3 = SourceNode(name='S3', value=7)
        q1 = PassNode(name='Q1')
        q2 = PassNode(name='Q2')
        s3 >> q1 >> q2

    _run_execute(dag)

    # Main component executed
    assert s1.calls == 1
    assert p1.calls == 1
    assert s2.calls == 1
    assert p2.calls == 1
    assert m.calls == 1
    assert e.calls == 1
    assert z.calls == 1

    # Side component executed
    assert s3.calls == 1
    assert q1.calls == 1
    assert q2.calls == 1

    # Merge correctness
    assert m.last_input is not None
    assert set(m.last_input.keys()) == {'default', 'attachment'}
    assert m.last_input['default']['value'] == 10
    assert m.last_input['attachment']['value'] == 20
    assert m.outputs[-1]['merged'] == {'default': 10, 'attachment': 20}

    # ExtractNode picks merged['attachment'] -> value=20
    assert e.outputs[-1]['value'] == 20

    # Z receives value=20 from E
    assert z.last_input is not None
    assert z.last_input['default']['value'] == 20


def test_execute_smoke_with_nodegroups_executes_all_nodes_once(tmp_path):
    """
    Smoke test: build a DAG using NodeGroups and ensure _execute evaluates all nodes exactly once.

    Graph
    -----
    SRC -> G1 -> E -> G2 -> DST

    Where:
    - G1 has multiple entry nodes and merges into a single out node (MergeNode output is 'merged')
    - E (ExtractNode) adapts the merge payload to the standard {'value': ...} schema
    - G2 is a simple linear group that expects 'value' inputs (PassNode)

    Assertions
    ----------
    - All nodes run exactly once
    """
    from morphalo.dag import DAG, NodeGroup, get_entry_nodes
    from morphalo.dag.runner import Execution, _execute
    from tests.dag.nodes import ExtractNode, MergeNode, PassNode, SourceNode

    with DAG('execute_groups_smoke', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=5)
        dst = PassNode(name='DST')

        # Group 1: two entries -> merge -> single out (emits {'merged': ...})
        with NodeGroup('g1') as g1:
            a = PassNode(name='A')  # entry
            b = PassNode(name='B')  # entry
            m = MergeNode(name='M')
            a >> m
            b >> m.sink(name='b_to_m', input_id='attachment')

        # Adapter: convert merge output into a standard {'value': ...} payload.
        # We pick one branch deterministically; here 'default'.
        e = ExtractNode(name='E', key='default')

        # Group 2: simple chain (expects {'value': ...})
        with NodeGroup('g2') as g2:
            x = PassNode(name='X')
            y = PassNode(name='Y')
            x >> y

        # Wire groups through adapter
        src >> g1 >> e >> g2 >> dst

    executions = [Execution(edge=e_) for e_ in dag.edges]
    entries = get_entry_nodes(nodes=dag.nodes, edges=dag.edges)

    _execute(
        out_dir=str(dag.out_dir),
        entries=entries,
        nodes=dag.nodes,
        executions=executions,
    )

    # Root nodes
    assert src.calls == 1
    assert dst.calls == 1

    # Group g1 nodes
    assert a.calls == 1
    assert b.calls == 1
    assert m.calls == 1

    # Adapter node
    assert e.calls == 1
    assert e.outputs[-1]['value'] == 5

    # Group g2 nodes
    assert x.calls == 1
    assert y.calls == 1
