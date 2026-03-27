from __future__ import annotations

import pytest

from morphalo.dag import DAG, NodeGroup
from tests.dag.nodes import PassNode, SourceNode


def test_empty_group_has_no_entries_and_src_shift_raises(tmp_path):
    """
    A group with no nodes has no entry nodes, so `src >> group` must raise.
    """
    with DAG('empty_group', out_dir=tmp_path):
        src = SourceNode(name='SRC', value=1)
        with NodeGroup('g') as g:
            pass

        with pytest.raises(RuntimeError, match='has no entry nodes'):
            src >> g


def test_group_multi_entry_broadcasts_src_to_all_entries(tmp_path):
    """
    If a group has multiple entry nodes, `src >> group` must broadcast:
      src -> each entry node

    This is the policy implemented in NodeGroup.__rrshift__.
    """
    with DAG('multi_entry', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)

        with NodeGroup('g') as g:
            a = PassNode(name='A')  # entry
            b = PassNode(name='B')  # entry
            # No edges: both are entries

        src >> g

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}
    assert ('SRC', 'g.A', 'default') in edges
    assert ('SRC', 'g.B', 'default') in edges


def test_group_requires_single_out_node_for_group_shift(tmp_path):
    """
    If a group has more than one terminal node, `group >> dst` is ambiguous and must raise.
    """
    with DAG('multi_out', out_dir=tmp_path) as dag:
        dst = PassNode(name='DST')

        with NodeGroup('g') as g:
            a = PassNode(name='A')  # terminal
            b = PassNode(name='B')  # terminal
            # No edges: both are out nodes => group output is ambiguous

        with pytest.raises(RuntimeError, match='must have a single output node'):
            g >> dst


def test_group_port_wiring_src_to_specific_entry_then_to_dst(tmp_path):
    """
    Verify port wiring on a group:

        src >> group('IN') >> dst

    Expected:
    - src wires only to g.IN (not to all group entries)
    - g.IN -> g.OUT is preserved
    - g.OUT wires to dst (group output semantics)
    """
    with DAG('port_single', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)
        dst = PassNode(name='DST')

        with NodeGroup('g') as g:
            n_in = PassNode(name='IN')    # entry
            n_out = PassNode(name='OUT')  # single terminal
            n_in >> n_out

            # Register IN as a selectable port (must be called after internal wiring)
            g.register_ports(n_in)

        src >> g('IN') >> dst

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}

    assert ('SRC', 'g.IN', 'default') in edges
    assert ('g.IN', 'g.OUT', 'default') in edges
    assert ('g.OUT', 'DST', 'default') in edges

    # Ensure we did not accidentally broadcast to other nodes
    assert ('SRC', 'g.OUT', 'default') not in edges


def test_group_port_then_group_then_dst(tmp_path):
    """
    Verify chaining across groups:

        src >> g1('IN') >> g2 >> dst

    Expected:
    - src -> g1.IN
    - g1.IN -> g1.OUT
    - g1.OUT -> g2.E (broadcast to g2 entry nodes; here only one)
    - g2.E -> g2.OUT
    - g2.OUT -> dst
    """
    with DAG('port_then_group', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)
        dst = PassNode(name='DST')

        with NodeGroup('g1') as g1:
            g1_in = PassNode(name='IN')
            g1_out = PassNode(name='OUT')
            g1_in >> g1_out
            g1.register_ports(g1_in)

        with NodeGroup('g2') as g2:
            g2_e = PassNode(name='E')
            g2_out = PassNode(name='OUT')
            g2_e >> g2_out

        src >> g1('IN') >> g2 >> dst

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}

    assert ('SRC', 'g1.IN', 'default') in edges
    assert ('g1.IN', 'g1.OUT', 'default') in edges

    assert ('g1.OUT', 'g2.E', 'default') in edges
    assert ('g2.E', 'g2.OUT', 'default') in edges

    assert ('g2.OUT', 'DST', 'default') in edges


def test_group_port_to_group_port_to_dst(tmp_path):
    """
    Verify chaining with port selection on both groups:

        src >> g1('IN') >> g2('B') >> dst

    Here g2 has multiple entries (A and B), but we target only B.

    Expected:
    - g1.OUT wires only into g2.B (not g2.A)
    - g2.B -> g2.OUT
    - g2.OUT -> dst
    """
    with DAG('port_to_port', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)
        dst = PassNode(name='DST')

        with NodeGroup('g1') as g1:
            g1_in = PassNode(name='IN')
            g1_out = PassNode(name='OUT')
            g1_in >> g1_out
            g1.register_ports(g1_in)

        with NodeGroup('g2') as g2:
            g2_a = PassNode(name='A')    # entry
            g2_b = PassNode(name='B')    # entry
            g2_out = PassNode(name='OUT')

            # Make a single terminal node, otherwise `port >> dst` would be ambiguous
            g2_a >> g2_out
            g2_b >> g2_out

            # Register only B as a port
            g2.register_ports(g2_b)

        src >> g1('IN') >> g2('B') >> dst

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}

    assert ('SRC', 'g1.IN', 'default') in edges
    assert ('g1.IN', 'g1.OUT', 'default') in edges

    assert ('g1.OUT', 'g2.B', 'default') in edges
    assert ('g2.B', 'g2.OUT', 'default') in edges
    assert ('g2.OUT', 'DST', 'default') in edges

    # Ensure we did not wire into the other entry
    assert ('g1.OUT', 'g2.A', 'default') not in edges


def test_group_port_lookup_raises_for_unknown_port(tmp_path):
    """
    Calling group('X') for an unregistered port must raise.
    """
    with DAG('port_unknown', out_dir=tmp_path):
        src = SourceNode(name='SRC', value=1)

        with NodeGroup('g') as g:
            n_in = PassNode(name='IN')
            n_out = PassNode(name='OUT')
            n_in >> n_out
            g.register_ports(n_in)

        with pytest.raises(KeyError, match='Unknown port'):
            _ = g('DOES_NOT_EXIST')
