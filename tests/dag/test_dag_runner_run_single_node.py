from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from stability.dag import DAG
from stability.dag.runner import Execution, SingleNodeRunner
from tests.dag.nodes import PassNode, SourceNode


@dataclass
class CacheWritingSource(SourceNode):
    """
    Source node that writes its output into an in-memory cache.

    This is used to simulate "materializing" cached outputs when
    SingleNodeRunner runs upstream nodes with force_upstream=True.
    """

    cache: Dict[str, Dict[str, Any]] = None

    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict[str, Any]:
        out = super().run(output_dir, input=input)

        # Emulate a node that persists its output (e.g. JSON sidecar on disk).
        # Here we store it in a dict to keep the test pure/in-memory.
        assert self.cache is not None
        self.cache[self.id] = out
        return out


def _build_executions(dag: DAG) -> list[Execution]:
    """
    Build edge execution records for a DAG.
    """
    return [Execution(edge=e) for e in dag.edges]


def test_run_uses_cached_upstream_outputs_when_available(tmp_path):
    """
    Graph
    -----
    A -> B

    Scenario
    --------
    - A is NOT executed in this run.
    - load_output('A') returns a cached payload.
    - B should execute successfully using that cached input.

    Assertions
    ----------
    - A.calls == 0 (not executed)
    - B.calls == 1
    - B.last_input contains 'default' with cached value
    """
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('single_node_cached', out_dir=tmp_path) as dag:
        a = SourceNode(name='A', value=123)
        b = PassNode(name='B')
        a >> b

    # Pre-populate cache for A (simulate a previous run)
    cache[a.id] = {'value': 123, 'from': a.id}

    def loader(node_id: str) -> Optional[Dict[str, Any]]:
        return cache.get(node_id)

    runner = SingleNodeRunner(
        node_id=b.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=False,
        load_output=loader,
    )

    runner.run()

    assert a.calls == 0
    assert b.calls == 1
    assert b.last_input is not None
    assert b.last_input['default']['value'] == 123
    assert b.last_input['default']['from'] == a.id


def test_run_raises_when_upstream_cache_missing_and_force_upstream_false(tmp_path):
    """
    Graph
    -----
    A -> B

    Scenario
    --------
    - load_output('A') returns None
    - force_upstream=False

    Assertions
    ----------
    - runner.run() raises RuntimeError complaining about missing cached output
    - neither A nor B should have executed
    """
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('single_node_missing_cache', out_dir=tmp_path) as dag:
        a = SourceNode(name='A', value=1)
        b = PassNode(name='B')
        a >> b

    def loader(node_id: str) -> Optional[Dict[str, Any]]:
        return cache.get(node_id)

    runner = SingleNodeRunner(
        node_id=b.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=False,
        load_output=loader,
    )

    with pytest.raises(RuntimeError, match='Missing cached output for upstream node'):
        runner.run()

    assert a.calls == 0
    assert b.calls == 0


def test_run_builds_upstream_when_missing_and_force_upstream_true(tmp_path):
    """
    Graph
    -----
    A -> B

    Scenario
    --------
    - cache initially empty
    - force_upstream=True
    - A, when executed, writes its output into cache (simulating persistence)
    - B should then execute using the newly cached A output

    Assertions
    ----------
    - A.calls == 1
    - B.calls == 1
    - cache contains A output
    - B.last_input['default'] matches A output
    """
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('single_node_force_upstream', out_dir=tmp_path) as dag:
        a = CacheWritingSource(name='A', value=77, cache=cache)
        b = PassNode(name='B')
        a >> b

    def loader(node_id: str) -> Optional[Dict[str, Any]]:
        return cache.get(node_id)

    runner = SingleNodeRunner(
        node_id=b.id,
        dag=dag,
        executions=_build_executions(dag),
        force_upstream=True,
        load_output=loader,
    )

    runner.run()

    assert a.calls == 1
    assert b.calls == 1

    assert a.id in cache
    assert cache[a.id]['value'] == 77

    assert b.last_input is not None
    assert b.last_input['default']['value'] == 77
    assert b.last_input['default']['from'] == a.id
