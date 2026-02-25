from __future__ import annotations

from stability.dag import DAG, NodeGroup
from tests.dag.nodes import PassNode, SourceNode


def test_nodegroup_nested_injection_builds_expected_ids_and_edges(tmp_path):
    """
    Validate DAG + nested NodeGroup composition (structure only).

    Semantics under test
    --------------------
    - Nesting affects id prefixing: nodes in g2 become 'g1.g2.*'.
    - A group's entry nodes are computed from *all nodes/edges within the injected group*.
      If a node (e.g. 'g1.entry') has an incoming edge, it is NOT an entry node.
    - Therefore, after wiring g2 -> g1.entry, the true entry of g1 becomes g2's entry
      ('g1.g2.A'), and SRC >> g1 wires to that node.

    Graph (conceptual)
    ------------------
      SRC -> g1.g2.A -> g1.g2.B -> g1.entry -> g1.out -> DST
    """
    from stability.dag import DAG, NodeGroup
    from tests.dag.nodes import PassNode, SourceNode

    with DAG('root', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)
        dst = PassNode(name='DST')

        with NodeGroup('g1') as g1:
            g1_entry = PassNode(name='entry')
            g1_out = PassNode(name='out')
            g1_entry >> g1_out

            with NodeGroup('g2') as g2:
                a = PassNode(name='A')
                b = PassNode(name='B')
                a >> b

            # g2 feeds g1.entry, making g1.entry NOT an entry node anymore
            g2 >> g1_entry

        src >> g1 >> dst

    ids = {n.id for n in dag.nodes}
    assert 'SRC' in ids
    assert 'DST' in ids
    assert 'g1.entry' in ids
    assert 'g1.out' in ids
    assert 'g1.g2.A' in ids
    assert 'g1.g2.B' in ids

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}

    # SRC wires to the true entry node of g1, which is g2's entry ('g1.g2.A')
    assert ('SRC', 'g1.g2.A', 'default') in edges

    # g2 internal wiring (nested ids)
    assert ('g1.g2.A', 'g1.g2.B', 'default') in edges

    # g2 out feeds g1.entry (thus g1.entry is not an entry node)
    assert ('g1.g2.B', 'g1.entry', 'default') in edges

    # g1 internal wiring and unique out
    assert ('g1.entry', 'g1.out', 'default') in edges
    assert ('g1.out', 'DST', 'default') in edges

    # Optional: make the semantic point explicit
    assert ('SRC', 'g1.entry', 'default') not in edges


def test_wiring_fanout_and_chained_nodegroups_builds_expected_structure(tmp_path):
    """
    Validate core DSL wiring behaviors (structure only).

    This test covers:
    - fan-out wiring: a >> [b, c]
    - NodeGroup injection and entry/out wiring
    - chaining groups: a >> group_1 >> group_2 >> b

    Graph (conceptual)
    ------------------
      A -> [B, C]

      A -> G1 -> G2 -> Z

    Where:
    - G1 is a group with a single entry node (g1.in) and a single output (g1.out)
    - G2 is a group with a single entry node (g2.in) and a single output (g2.out)

    Expectations
    ------------
    - Fan-out creates two edges: A->B and A->C.
    - A >> G1 wires A to G1 entry nodes (broadcast policy; here it's one entry).
    - G1 >> G2 wires from G1 out node to G2 entry nodes.
    - G2 >> Z wires from G2 out node to Z.
    - Group nodes are injected into the DAG exactly once, with scope-qualified ids.
    """
    with DAG('wire_groups', out_dir=tmp_path) as dag:
        a = SourceNode(name='A', value=1)
        b = PassNode(name='B')
        c = PassNode(name='C')

        # 1) fan-out
        a >> [b, c]

        # 2) group_1: single entry + single out
        with NodeGroup('g1') as g1:
            g1_in = PassNode(name='in')
            g1_out = PassNode(name='out')
            g1_in >> g1_out

        # 3) group_2: single entry + single out
        with NodeGroup('g2') as g2:
            g2_in = PassNode(name='in')
            g2_out = PassNode(name='out')
            g2_in >> g2_out

        z = PassNode(name='Z')

        # chaining groups
        a >> g1 >> g2 >> z

    ids = {n.id for n in dag.nodes}
    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}

    # Nodes created at DAG root
    assert 'A' in ids
    assert 'B' in ids
    assert 'C' in ids
    assert 'Z' in ids

    # Nodes created inside groups must be scope-qualified (prefix = group name)
    assert 'g1.in' in ids
    assert 'g1.out' in ids
    assert 'g2.in' in ids
    assert 'g2.out' in ids

    # Fan-out edges
    assert ('A', 'B', 'default') in edges
    assert ('A', 'C', 'default') in edges

    # A >> g1: entry node(s) of g1
    assert ('A', 'g1.in', 'default') in edges

    # internal g1 wiring
    assert ('g1.in', 'g1.out', 'default') in edges

    # g1 >> g2: from g1 out node to g2 entry node(s)
    assert ('g1.out', 'g2.in', 'default') in edges

    # internal g2 wiring
    assert ('g2.in', 'g2.out', 'default') in edges

    # g2 >> z: from g2 out node to Z
    assert ('g2.out', 'Z', 'default') in edges
