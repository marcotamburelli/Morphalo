from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

import morphalo.dag.runner as runner_mod
from morphalo.dag import DAG, NodeGroup
from morphalo.dag.runner import DAGRunner, Execution, _execute_subgraph
from morphalo.dag.validation import DagValidationError
from tests.dag.nodes import PassNode, SourceNode


@dataclass
class CacheWritingSource(SourceNode):
    cache: Dict[str, Dict[str, Any]] = None

    def run(self, output_dir, input=None) -> Dict[str, Any]:
        out = super().run(output_dir, input=input)
        assert self.cache is not None
        self.cache[self.id] = out
        return out


def _executions(dag: DAG) -> list[Execution]:
    return [Execution(edge=edge) for edge in dag.edges]


class InlineNodeExecutor:
    def __init__(self, *, max_cuda_nodes: int):
        self.max_cuda_nodes = max_cuda_nodes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def run(self, *, node, out_dir, input_map):
        out = node.run(out_dir, input=input_map)
        node.post_run()
        return out


def test_dag_runner_executes_group_nodes_and_uses_external_cache(
    tmp_path,
    monkeypatch,
):
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('group_cached_input', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=7)
        dst = PassNode(name='DST')

        with NodeGroup('group') as group:
            first = PassNode(name='first')
            second = PassNode(name='second')
            first >> second

        src >> group >> dst

    cache[src.id] = {'value': 7, 'from': src.id}

    monkeypatch.setattr(runner_mod, 'ProcessNodeExecutor', InlineNodeExecutor)
    monkeypatch.setattr(
        runner_mod,
        'load_latest_output',
        lambda *, out_dir, node_id: cache.get(node_id),
    )
    DAGRunner(dag).run_group(group.id)

    assert src.calls == 0
    assert first.calls == 1
    assert second.calls == 1
    assert dst.calls == 0
    assert first.last_input == {
        'default': {'value': 7, 'from': src.id}
    }


def test_group_run_can_include_downstream_nodes(tmp_path):
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('group_downstream', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=11)

        with NodeGroup('group') as group:
            first = PassNode(name='first')
            second = PassNode(name='second')
            first >> second

        dst = PassNode(name='DST')
        side = PassNode(name='SIDE')
        src >> group >> dst
        first >> side

    cache[src.id] = {'value': 11, 'from': src.id}

    _execute_subgraph(
        dag=dag,
        executions=_executions(dag),
        target_ids={node.id for node in group.nodes},
        downstream=True,
        force_upstream=False,
        load_output=lambda node_id: cache.get(node_id),
    )

    assert src.calls == 0
    assert first.calls == 1
    assert second.calls == 1
    assert dst.calls == 1
    assert side.calls == 1


def test_nested_group_can_be_selected_without_running_parent_nodes(tmp_path):
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('nested_group_selection', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=17)

        with NodeGroup('outer') as outer:
            outer_out = PassNode(name='out')

            with NodeGroup('inner') as inner:
                inner_node = PassNode(name='node')

            inner >> outer_out

        src >> outer

    cache[src.id] = {'value': 17, 'from': src.id}

    _execute_subgraph(
        dag=dag,
        executions=_executions(dag),
        target_ids={node.id for node in inner.nodes},
        downstream=False,
        force_upstream=False,
        load_output=lambda node_id: cache.get(node_id),
    )

    assert inner.id == 'outer.inner'
    assert src.calls == 0
    assert inner_node.calls == 1
    assert outer_out.calls == 0


def test_group_run_force_upstream_materializes_missing_input(tmp_path):
    cache: Dict[str, Dict[str, Any]] = {}

    with DAG('group_force_upstream', out_dir=tmp_path) as dag:
        src = CacheWritingSource(name='SRC', value=13, cache=cache)

        with NodeGroup('group') as group:
            first = PassNode(name='first')

        src >> group

    _execute_subgraph(
        dag=dag,
        executions=_executions(dag),
        target_ids={node.id for node in group.nodes},
        downstream=False,
        force_upstream=True,
        load_output=lambda node_id: cache.get(node_id),
    )

    assert src.calls == 1
    assert first.calls == 1
    assert first.last_input is not None
    assert first.last_input['default']['value'] == 13


def test_group_run_reports_missing_external_input(tmp_path):
    with DAG('group_missing_input', out_dir=tmp_path) as dag:
        src = SourceNode(name='SRC', value=1)

        with NodeGroup('group') as group:
            first = PassNode(name='first')

        src >> group

    with pytest.raises(
        DagValidationError,
        match="'group.first' requires 'SRC' on input 'default'",
    ):
        _execute_subgraph(
            dag=dag,
            executions=_executions(dag),
            target_ids={node.id for node in group.nodes},
            downstream=False,
            force_upstream=False,
            load_output=lambda _node_id: None,
        )

    assert src.calls == 0
    assert first.calls == 0
