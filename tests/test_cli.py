from __future__ import annotations

import pytest
import typer

import morphalo.cli as cli
from morphalo.dag import DAG, DagRegistry


class RecordingRunner:
    calls: list[tuple] = []

    def __init__(self, dag: DAG, *, max_cuda_nodes: int):
        self.dag = dag
        self.max_cuda_nodes = max_cuda_nodes

    def run(self):
        self.calls.append(('run', self.dag.name))

    def run_node(self, node_id: str, *, force_upstream: bool):
        self.calls.append(
            ('node', self.dag.name, node_id, force_upstream)
        )

    def run_node_downstream(self, node_id: str, *, force_upstream: bool):
        self.calls.append(
            ('node_downstream', self.dag.name, node_id, force_upstream)
        )

    def run_group(
        self,
        group_id: str,
        *,
        downstream: bool,
        force_upstream: bool,
    ):
        self.calls.append(
            (
                'group',
                self.dag.name,
                group_id,
                downstream,
                force_upstream,
            )
        )


def _patch_discovery(monkeypatch, *dags: DAG) -> None:
    def import_module(_module: str):
        for dag in dags:
            DagRegistry.add(dag)

    RecordingRunner.calls = []
    monkeypatch.setattr(cli.importlib, 'import_module', import_module)
    monkeypatch.setattr(cli, 'DAGRunner', RecordingRunner)


def test_run_dags_routes_group_options(monkeypatch, tmp_path):
    dag = DAG('selected', out_dir=tmp_path)
    _patch_discovery(monkeypatch, dag)

    cli.run_dags(
        module='example.dags',
        dag='selected',
        node=None,
        group='outer.inner',
        downstream=True,
        force_upstream=True,
        cuda_chunk_size=4,
    )

    assert RecordingRunner.calls == [
        ('group', 'selected', 'outer.inner', True, True)
    ]


def test_run_dags_rejects_node_and_group_together(monkeypatch, tmp_path):
    dag = DAG('selected', out_dir=tmp_path)
    _patch_discovery(monkeypatch, dag)

    with pytest.raises(
        typer.BadParameter,
        match='--node and --group are mutually exclusive',
    ):
        cli.run_dags(
            module='example.dags',
            dag='selected',
            node='node',
            group='group',
            downstream=False,
            force_upstream=False,
            cuda_chunk_size=8,
        )


def test_run_dags_requires_dag_for_group_with_multiple_dags(
    monkeypatch,
    tmp_path,
):
    first = DAG('first', out_dir=tmp_path)
    second = DAG('second', out_dir=tmp_path)
    _patch_discovery(monkeypatch, first, second)

    with pytest.raises(
        typer.BadParameter,
        match='--node/--group requires --dag',
    ):
        cli.run_dags(
            module='example.dags',
            dag=None,
            node=None,
            group='group',
            downstream=False,
            force_upstream=False,
            cuda_chunk_size=8,
        )


def test_run_dags_rejects_downstream_without_target(monkeypatch, tmp_path):
    dag = DAG('selected', out_dir=tmp_path)
    _patch_discovery(monkeypatch, dag)

    with pytest.raises(typer.BadParameter, match='--downstream'):
        cli.run_dags(
            module='example.dags',
            dag='selected',
            node=None,
            group=None,
            downstream=True,
            force_upstream=False,
            cuda_chunk_size=8,
        )
