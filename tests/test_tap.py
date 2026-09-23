import pytest

from morphalo.core.paths import load_latest_output
from morphalo.dag import DAG
from morphalo.dag.control_nodes import MaterializedPassthrough
from morphalo.nodes import Tap


def test_tap_inherits_materialized_passthrough_and_forwards_copy(tmp_path):
    with DAG('tap', out_dir=tmp_path):
        node = Tap(name='tap')

    upstream = {
        'image': '/tmp/image.png',
        'id': 'upstream',
        'node': 'fileimage',
        'metadata': '/tmp/upstream.json',
    }
    result = node.run(tmp_path, input={'default': upstream})

    assert isinstance(node, MaterializedPassthrough)
    assert result == {
        'image': '/tmp/image.png',
        'ok': True,
        'node': 'tap',
        'id': 'tap',
        'metadata': result['metadata'],
    }
    assert load_latest_output(tmp_path, node.id) == {
        'image': '/tmp/image.png',
        'ok': True,
        'node': 'tap',
        'id': 'tap',
    }
    assert upstream['id'] == 'upstream'
    assert upstream['metadata'] == '/tmp/upstream.json'


def test_non_strict_tap_materializes_minimal_output(tmp_path):
    with DAG('tap', out_dir=tmp_path):
        node = Tap(name='tap', strict=False)

    result = node.run(tmp_path)

    assert result == {
        'ok': True,
        'node': 'tap',
        'id': 'tap',
        'metadata': result['metadata'],
    }
    assert load_latest_output(tmp_path, node.id) == {
        'ok': True,
        'node': 'tap',
        'id': 'tap',
    }


def test_strict_tap_rejects_missing_and_none_input(tmp_path):
    with DAG('tap', out_dir=tmp_path):
        node = Tap(name='tap')

    with pytest.raises(RuntimeError, match='received no input'):
        node.run(tmp_path)
    with pytest.raises(RuntimeError, match='received None upstream output'):
        node.run(tmp_path, input={'default': None})
    with pytest.raises(RuntimeError, match='expects exactly one input'):
        node.run(tmp_path, input={'other': {}})
