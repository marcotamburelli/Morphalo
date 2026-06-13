from __future__ import annotations

import pytest

from morphalo.dag import DAG, NodeGroup
from tests.dag.nodes import PassNode, SourceNode


def test_nodegroup_injection_is_idempotent(tmp_path):
    """
    Ensure injecting the same NodeGroup instance multiple times does not duplicate nodes/edges.

    We trigger injection via wiring:
      src1 >> g
      src2 >> g

    The group's nodes and internal edges must appear only once in the DAG.
    """
    with DAG('idempotent', out_dir=tmp_path) as dag:
        src1 = SourceNode(name='SRC1', value=1)
        src2 = SourceNode(name='SRC2', value=2)

        with NodeGroup('g') as g:
            n1 = PassNode(name='n1')
            n2 = PassNode(name='n2')
            n1 >> n2

        # Two separate wirings into the same group instance
        src1 >> g
        src2 >> g

    ids = [n.id for n in dag.nodes]

    # Group nodes must appear exactly once each
    assert ids.count('g.n1') == 1
    assert ids.count('g.n2') == 1

    # No duplicate node ids overall
    assert len(ids) == len(set(ids))

    edges = [(e.node_from, e.node_to, e.input_id) for e in dag.edges]

    # Internal edge appears once
    assert edges.count(('g.n1', 'g.n2', 'default')) == 1

    # Both srcs must connect to the group's entry node (g.n1)
    assert ('SRC1', 'g.n1', 'default') in edges
    assert ('SRC2', 'g.n1', 'default') in edges


def test_nodegroup_injection_raises_on_node_id_collision(tmp_path):
    """
    Ensure group injection fails fast if a node id would collide in the parent scope.

    We create a root node with id 'g.in' and then define a group 'g' that contains
    a node 'in' -> id 'g.in'. Injection must raise.
    """
    with DAG('collision', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)

        # Root node with an id that will collide with the group-internal node
        _collision = PassNode(name='g.in')

        with NodeGroup('g') as g:
            _in = PassNode(name='in')

        with pytest.raises(RuntimeError, match='node id collision'):
            src >> g


def test_injected_nested_groups_are_registered_by_qualified_id(tmp_path):
    with DAG('nested_groups', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)

        with NodeGroup('outer') as outer:
            outer_out = PassNode(name='out')

            with NodeGroup('inner') as inner:
                inner_in = PassNode(name='in')

            inner >> outer_out

        src >> outer

    assert dag.node_groups == {
        'outer': outer,
        'outer.inner': inner,
    }
    assert {node.id for node in outer.nodes} == {
        'outer.inner.in',
        'outer.out',
    }
    assert {node.id for node in inner.nodes} == {'outer.inner.in'}
