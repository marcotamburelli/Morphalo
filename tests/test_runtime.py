from __future__ import annotations

import json

import pytest

from morphalo.core.paths import make_node_output_dir
from morphalo.dag import DAG, Runtime


class NodeLike:
    def __init__(self, node_id: str):
        self.id = node_id


def write_sidecar(out_dir, node_id: str, name: str, payload: dict):
    node_dir = make_node_output_dir(out_dir=out_dir, node_id=node_id)
    path = node_dir / name
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_runtime_reads_latest_output_from_current_dag(tmp_path):
    write_sidecar(
        tmp_path,
        'image',
        '2026-01-01_010101.json',
        {'params': {'width': 640, 'height': 960}},
    )
    write_sidecar(
        tmp_path,
        'image',
        '2026-01-01_010102.json',
        {'params': {'width': 768, 'height': 1024}},
    )

    with DAG('demo', out_dir=tmp_path):
        assert Runtime.get_out(NodeLike('image')) == {
            'params': {'width': 768, 'height': 1024},
        }
        assert Runtime.get_output_size(NodeLike('image')) == (768, 1024)


def test_runtime_output_size_fallbacks(tmp_path):
    write_sidecar(
        tmp_path,
        'resize',
        '2026-01-01_010101.json',
        {
            'output_size': [700, 900],
            'resize': {'resolved_size': [640, 832]},
        },
    )
    write_sidecar(
        tmp_path,
        'resize_only',
        '2026-01-01_010101.json',
        {'resize': {'resolved_size': [640, 832]}},
    )

    with DAG('demo', out_dir=tmp_path):
        assert Runtime.get_output_size('resize') == (700, 900)
        assert Runtime.get_output_size('resize_only') == (640, 832)


def test_runtime_input_size(tmp_path):
    write_sidecar(
        tmp_path,
        'resize',
        '2026-01-01_010101.json',
        {'input_size': [512, 768]},
    )

    with DAG('demo', out_dir=tmp_path):
        assert Runtime.get_input_size(NodeLike('resize')) == (512, 768)


def test_runtime_requires_active_dag(tmp_path):
    write_sidecar(
        tmp_path,
        'image',
        '2026-01-01_010101.json',
        {'params': {'width': 640, 'height': 960}},
    )

    with pytest.raises(RuntimeError, match='active DAG context'):
        Runtime.get_out('image')


def test_runtime_missing_output_raises_clear_error(tmp_path):
    with DAG('demo', out_dir=tmp_path):
        with pytest.raises(FileNotFoundError, match='missing'):
            Runtime.get_out('missing')
