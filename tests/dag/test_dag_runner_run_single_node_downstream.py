from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

import morphalo.dag.runner as runner_mod
from morphalo.dag import DAG, get_entry_nodes
from morphalo.dag.runner import Execution, SingleNodeRunner, _execute
from tests.dag.nodes import MergeNode, PassNode, SourceNode


@dataclass
class CacheWritingSource(SourceNode):
    """
    Source node that writes its output into an in-memory cache.

    This emulates a node persisting its outputs (e.g. JSON artifacts)
    so that "cached" status can be controlled without filesystem I/O.
    """

    cache: Dict[str, Dict[str, Any]] = None

    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict[str, Any]:
        out = super().run(output_dir, input=input)
        assert self.cache is not None
        self.cache[self.id] = out
        return out


@dataclass
class CacheWritingPass(PassNode):
    """
    Pass-through node that persists its output into the in-memory test cache.
    """
    cache: Dict[str, Dict[str, Any]] = None

    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        out = super().run(output_dir, input=input)
        assert self.cache is not None
        self.cache[self.id] = out
        return out


@dataclass
class CacheWritingMerge(MergeNode):
    """
    Merge node that persists its output into the in-memory test cache.
    """
    cache: Dict[str, Dict[str, Any]] = None

    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        out = super().run(output_dir, input=input)
        assert self.cache is not None
        self.cache[self.id] = out
        return out


# SingleNodeRunner expects a prebuilt execution list.
# In production this is normally created by DAGRunner.


def _build_executions(dag: DAG) -> list[Execution]:
    """
    Build one Execution record per DAG edge.
    """
    return [Execution(edge=e) for e in dag.edges]


def _patch_load_latest_output(monkeypatch: pytest.MonkeyPatch, cache: Dict[str, Dict[str, Any]]) -> None:
    """
    Monkeypatch runner.load_latest_output to use an in-memory cache.

    The production code calls:
        load_latest_output(out_dir=..., node_id=...)
    so we mirror that signature.
    """

    def _fake_load_latest_output(out_dir: str, node_id: str) -> Optional[Dict[str, Any]]:
        return cache.get(node_id)

    monkeypatch.setattr(runner_mod, 'load_latest_output',
                        _fake_load_latest_output)


def test_run_downstream_executes_only_downstream_closure_and_pulls_uncached_upstream(tmp_path, monkeypatch):
    """
    Graph
    -----
    A -> B -> C
    A -> D -> E

    Target
    ------
    run_downstream(B, force_upstream=True)

    Expected execution set R
    ------------------------
    Downstream closure of B is {B, C}.
    With force_upstream=True and empty cache, upstream expansion should add A.

    Therefore R should be {A, B, C} and NOT include {D, E}.

    Assertions
    ----------
    - A, B, C run exactly once
    - D, E are not executed
    - B receives input from A (default)
    - C receives input from B (default)
    """
    cache: Dict[str, Dict[str, Any]] = {}
    _patch_load_latest_output(monkeypatch, cache)

    with DAG('downstream_simple', out_dir=tmp_path) as dag:
        a = CacheWritingSource(name='A', value=1, cache=cache)
        b = PassNode(name='B')
        c = PassNode(name='C')
        d = PassNode(name='D')
        e = PassNode(name='E')

        a >> b >> c
        a >> d >> e

    runner = SingleNodeRunner(
        node_id=b.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=True,
        load_output=lambda node_id: cache.get(node_id),
    )

    runner.run_downstream()

    assert a.calls == 1
    assert b.calls == 1
    assert c.calls == 1

    assert d.calls == 0
    assert e.calls == 0

    assert b.last_input is not None
    assert b.last_input['default']['value'] == 1
    assert b.last_input['default']['from'] == a.id

    assert c.last_input is not None
    assert c.last_input['default']['value'] == 1
    assert c.last_input['default']['from'] == b.id


def test_run_downstream_intricate_graph_executes_correct_subgraph(tmp_path, monkeypatch):
    """
    Intricate graph with fan-in, attachment sinks, and side branches.

    Graph
    -----
      S1 -> P1 -----
                  \
                   -> M -> E -> P3 -> P4
                  /
      S2 -> P2 --(attachment)

    Side branches (must NOT execute when targeting P2)
    --------------------------------------------------
      S1 -> X1 -> X2
      S2 -> Y1

    Target
    ------
    run_downstream(P2, force_upstream=True)

    Expected behavior
    -----------------
    Downstream closure of P2 is {P2, M, E, P3, P4}.
    With force_upstream=True and empty cache, upstream expansion should include:
      - S2 (upstream of P2)
      - P1 and S1 (because M depends also on P1)

    Therefore R should include: {S1, P1, S2, P2, M, E, P3, P4}
    and exclude: {X1, X2, Y1}

    Assertions
    ----------
    - All nodes in R run exactly once
    - Side branch nodes do not run
    - Merge node receives both 'default' and 'attachment'
    - Merge output contains both values keyed by input_id
    - Extract node converts merged payload into a standard 'value' payload
    - P3 and P4 run and receive the propagated value
    """
    from tests.dag.nodes import ExtractNode

    cache: Dict[str, Dict[str, Any]] = {}
    _patch_load_latest_output(monkeypatch, cache)

    with DAG('downstream_intricate', out_dir=tmp_path) as dag:
        s1 = CacheWritingSource(name='S1', value=10, cache=cache)
        p1 = PassNode(name='P1')
        s1 >> p1

        s2 = CacheWritingSource(name='S2', value=20, cache=cache)
        p2 = PassNode(name='P2')
        s2 >> p2

        m = MergeNode(name='M')
        p1 >> m  # default input_id
        p2 >> m.sink(name='P2_to_M', input_id='attachment')

        # Adapter: MergeNode emits {'merged': ...}, PassNode expects {'value': ...}
        e = ExtractNode(name='E', key='attachment')

        p3 = PassNode(name='P3')
        p4 = PassNode(name='P4')
        m >> e >> p3 >> p4

        # Side branches (should not execute)
        x1 = PassNode(name='X1')
        x2 = PassNode(name='X2')
        s1 >> x1 >> x2

        y1 = PassNode(name='Y1')
        s2 >> y1

    runner = SingleNodeRunner(
        node_id=p2.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=True,
        load_output=lambda node_id: cache.get(node_id),
    )

    runner.run_downstream()

    # Nodes expected to execute
    assert s1.calls == 1
    assert p1.calls == 1
    assert s2.calls == 1
    assert p2.calls == 1
    assert m.calls == 1
    assert e.calls == 1
    assert p3.calls == 1
    assert p4.calls == 1

    # Nodes expected NOT to execute
    assert x1.calls == 0
    assert x2.calls == 0
    assert y1.calls == 0

    # Merge correctness
    assert m.last_input is not None
    assert set(m.last_input.keys()) == {'default', 'attachment'}
    assert m.last_input['default']['value'] == 10
    assert m.last_input['attachment']['value'] == 20
    assert m.outputs[-1]['merged'] == {'default': 10, 'attachment': 20}

    # Extract correctness: pick the 'attachment' branch value (20)
    assert e.last_input is not None
    assert e


# Checks run_downstream for cached lateral dependencies
def test_run_downstream_uses_cached_lateral_dependencies_when_force_upstream_false(
    tmp_path,
    monkeypatch,
):
    """
    Graph
    -----
      S1 -> P1 -----
                  \
                   -> M -> E -> P3
                  /
      S2 -> P2 --(attachment)

    Target
    ------
    run_downstream(M, force_upstream=False)

    Setup
    -----
    P1 and P2 are precomputed in cache.

    Expected behavior
    -----------------
    The runner should:
    - execute M, E, P3
    - NOT execute S1/P1/S2/P2
    - use cached outputs for both inputs of M
    """
    from tests.dag.nodes import ExtractNode

    cache: Dict[str, Dict[str, Any]] = {}
    _patch_load_latest_output(monkeypatch, cache)

    with DAG('downstream_cached_lateral', out_dir=tmp_path) as dag:
        s1 = CacheWritingSource(name='S1', value=10, cache=cache)
        p1 = PassNode(name='P1')
        s1 >> p1

        s2 = CacheWritingSource(name='S2', value=20, cache=cache)
        p2 = PassNode(name='P2')
        s2 >> p2

        m = MergeNode(name='M')
        p1 >> m
        p2 >> m.sink(name='P2_to_M', input_id='attachment')

        e = ExtractNode(name='E', key='attachment')
        p3 = PassNode(name='P3')
        m >> e >> p3

    # Precompute both upstream branches, but do it manually so the runner
    # must consume them from cache rather than execute them again.
    s1_out = s1.run(tmp_path)
    p1_out = p1.run(tmp_path, input={'default': s1_out})
    s2_out = s2.run(tmp_path)
    p2_out = p2.run(tmp_path, input={'default': s2_out})

    cache[s1.id] = s1_out
    cache[p1.id] = p1_out
    cache[s2.id] = s2_out
    cache[p2.id] = p2_out

    runner = SingleNodeRunner(
        node_id=m.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=False,
        load_output=lambda node_id: cache.get(node_id),
    )

    runner.run_downstream()

    # Upstream nodes were only executed during manual precomputation.
    assert s1.calls == 1
    assert p1.calls == 1
    assert s2.calls == 1
    assert p2.calls == 1

    # Downstream execution starts at M.
    assert m.calls == 1
    assert e.calls == 1
    assert p3.calls == 1

    assert set(m.last_input.keys()) == {'default', 'attachment'}
    assert m.last_input['default']['value'] == 10
    assert m.last_input['attachment']['value'] == 20


# run_downstream should fail since one lateral dependency of M is missing
def test_run_downstream_fails_when_lateral_dependency_is_missing_and_force_upstream_false(
    tmp_path,
    monkeypatch,
):
    """
    Graph
    -----
      S1 -> P1 -----
                  \
                   -> M -> E -> P3
                  /
      S2 -> P2 --(attachment)

    Target
    ------
    run_downstream(M, force_upstream=False)

    Setup
    -----
    Only P2 is precomputed in cache.
    P1 is missing.

    Expected behavior
    -----------------
    The runner must fail because M requires both inputs:
    - default      <- P1
    - attachment   <- P2

    With force_upstream=False, the missing lateral dependency P1 must not be
    materialized automatically, so execution should fail when M is reached.
    """
    from tests.dag.nodes import ExtractNode

    cache: Dict[str, Dict[str, Any]] = {}
    _patch_load_latest_output(monkeypatch, cache)

    with DAG('downstream_missing_lateral', out_dir=tmp_path) as dag:
        s1 = CacheWritingSource(name='S1', value=10, cache=cache)
        p1 = PassNode(name='P1')
        s1 >> p1

        s2 = CacheWritingSource(name='S2', value=20, cache=cache)
        p2 = PassNode(name='P2')
        s2 >> p2

        m = MergeNode(name='M')
        p1 >> m
        p2 >> m.sink(name='P2_to_M', input_id='attachment')

        e = ExtractNode(name='E', key='attachment')
        p3 = PassNode(name='P3')
        m >> e >> p3

    # Precompute only the P2 branch.
    s2_out = s2.run(tmp_path)
    p2_out = p2.run(tmp_path, input={'default': s2_out})
    cache[s2.id] = s2_out
    cache[p2.id] = p2_out

    runner = SingleNodeRunner(
        node_id=m.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=False,
        load_output=lambda node_id: cache.get(node_id),
    )

    with pytest.raises(Exception):
        runner.run_downstream()

    # P1 branch must not be auto-materialized.
    assert s1.calls == 0
    assert p1.calls == 0

    # P2 branch was only executed during manual precomputation.
    assert s2.calls == 1
    assert p2.calls == 1

    # M should not complete successfully.
    assert e.calls == 0
    assert p3.calls == 0


# Checks run_downstream for materialized (force_upstream=True) lateral dependencies
def test_run_downstream_materializes_missing_lateral_dependency_when_force_upstream_true(
    tmp_path,
    monkeypatch,
):
    """
    Graph
    -----
      S1 -> P1 -----
                  \
                   -> M -> E -> P3
                  /
      S2 -> P2 --(attachment)

    Target
    ------
    run_downstream(M, force_upstream=True)

    Setup
    -----
    Only P2 is precomputed in cache.
    P1 is missing.

    Expected behavior
    -----------------
    The runner should detect that M also depends on P1, expand upstream
    requirements accordingly, execute S1/P1, and then complete execution of
    M -> E -> P3 successfully using:
    - cached P2
    - materialized P1
    """
    from tests.dag.nodes import ExtractNode

    cache: Dict[str, Dict[str, Any]] = {}
    _patch_load_latest_output(monkeypatch, cache)

    with DAG('downstream_force_lateral', out_dir=tmp_path) as dag:
        s1 = CacheWritingSource(name='S1', value=10, cache=cache)
        p1 = PassNode(name='P1')
        s1 >> p1

        s2 = CacheWritingSource(name='S2', value=20, cache=cache)
        p2 = PassNode(name='P2')
        s2 >> p2

        m = MergeNode(name='M')
        p1 >> m
        p2 >> m.sink(name='P2_to_M', input_id='attachment')

        e = ExtractNode(name='E', key='attachment')
        p3 = PassNode(name='P3')
        m >> e >> p3

    # Precompute only the P2 branch; P1 is intentionally missing.
    s2_out = s2.run(tmp_path)
    p2_out = p2.run(tmp_path, input={'default': s2_out})
    cache[s2.id] = s2_out
    cache[p2.id] = p2_out

    runner = SingleNodeRunner(
        node_id=m.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=True,
        load_output=lambda node_id: cache.get(node_id),
    )

    runner.run_downstream()

    # P1 branch must be materialized by the runner.
    assert s1.calls == 1
    assert p1.calls == 1

    # P2 branch must NOT be re-executed by the runner because it is already cached.
    assert s2.calls == 1
    assert p2.calls == 1

    assert m.calls == 1
    assert e.calls == 1
    assert p3.calls == 1

    assert set(m.last_input.keys()) == {'default', 'attachment'}
    assert m.last_input['default']['value'] == 10
    assert m.last_input['attachment']['value'] == 20


def _reset_tracking(*nodes) -> None:
    """
    Reset execution tracking fields between the first full run and the replay run.
    """
    for n in nodes:
        n.calls = 0
        n.last_input = None
        n.outputs = []


def test_run_downstream_replay_must_not_use_stale_cache_for_node_inside_current_subgraph(
    tmp_path,
):
    """
    Guard against stale-cache reads during downstream replay.

    Graph
    -----
      A -> B -> Y -> C -> M
      A -> X ---------(attachment)

    Replay target
    -------------
    run_downstream(A, force_upstream=False)

    Scenario
    --------
    1) Run the full DAG once to populate cache for every node.
    2) Change A.value.
    3) Run downstream replay rooted at A.

    Bug to detect
    -------------
    During replay, M must not consume the cached output of C from the previous run
    before C has been re-executed in the current run.

    Why this graph shape matters
    ----------------------------
    The side branch A -> X reaches M earlier than the longer branch
    A -> B -> Y -> C -> M.

    Therefore, if the runner incorrectly allows cached reads for nodes that belong
    to the *current* downstream subgraph, M may see:
    - attachment: fresh current-run output from X
    - default: stale previous-run output from C

    Expected behavior
    -----------------
    M must receive fresh current-run outputs on both inputs.
    """
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('downstream_replay_no_stale_internal_cache', out_dir=tmp_path) as dag:
        a = CacheWritingSource(name='A', value=1, cache=cache)
        b = CacheWritingPass(name='B', cache=cache)
        y = CacheWritingPass(name='Y', cache=cache)
        c = CacheWritingPass(name='C', cache=cache)
        x = CacheWritingPass(name='X', cache=cache)
        m = CacheWritingMerge(name='M', cache=cache)

        a >> b >> y >> c >> m
        a >> x >> m.sink(name='X_to_M', input_id='attachment')

    # First run: populate cache for the whole DAG.
    _execute(
        out_dir=dag.out_dir,
        entries=get_entry_nodes(dag.nodes, dag.edges),
        nodes=dag.nodes,
        executions=_build_executions(dag),
    )

    # Sanity check: cache really contains previous-run outputs.
    assert cache[a.id]['value'] == 1
    assert cache[c.id]['value'] == 1
    assert cache[x.id]['value'] == 1

    # Prepare replay:
    # - change the source value so fresh current-run outputs differ from stale cache
    # - reset node tracking so we can observe only the replay run
    a.value = 2
    _reset_tracking(a, b, y, c, x, m)

    # IMPORTANT:
    # Recreate executions for the replay run.
    # Execution objects are mutable and may retain outputs from earlier runs,
    # so reusing them would hide stale-cache behavior.
    replay_runner = SingleNodeRunner(
        node_id=a.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=False,
        load_output=lambda node_id: cache.get(node_id),
    )

    replay_runner.run_downstream()

    # All nodes in the replayed downstream subgraph should execute once.
    assert a.calls == 1
    assert b.calls == 1
    assert y.calls == 1
    assert c.calls == 1
    assert x.calls == 1
    assert m.calls == 1

    assert m.last_input is not None
    assert set(m.last_input.keys()) == {'default', 'attachment'}

    # Both branches must be fresh from the current run.
    #
    # If the bug is present, a likely bad state is:
    #   default.value == 1   (stale cached C from previous run)
    #   attachment.value == 2 (fresh X from current run)
    assert m.last_input['default']['value'] == 2
    assert m.last_input['attachment']['value'] == 2

    # Extra sanity: the default branch must really come from C, and attachment from X.
    assert m.last_input['default']['from'] == c.id
    assert m.last_input['attachment']['from'] == x.id
