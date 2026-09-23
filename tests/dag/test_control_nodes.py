import pytest

from morphalo.core.paths import load_latest_output
from morphalo.dag import DAG
from morphalo.dag.control_nodes import CudaCooldown


def test_cooldown_forwards_default_output(tmp_path):
    with DAG('cooldown', out_dir=tmp_path):
        node = CudaCooldown(name='cooldown')
    output = {
        'image': '/tmp/image.png',
        'prompt': 'test',
        'id': 'upstream',
        'node': 'FileImage',
        'metadata': '/tmp/upstream.json',
    }
    assert node.forces_cuda_cooldown is True
    assert node.uses_cuda is False
    result = node.run(tmp_path, input={'default': output})
    assert result == {
        'image': '/tmp/image.png',
        'prompt': 'test',
        'ok': True,
        'node': 'cudacooldown',
        'id': 'cooldown',
        'metadata': result['metadata'],
    }
    assert load_latest_output(tmp_path, node.id) == {
        'image': '/tmp/image.png',
        'prompt': 'test',
        'ok': True,
        'node': 'cudacooldown',
        'id': 'cooldown',
    }
    assert output['id'] == 'upstream'
    assert output['metadata'] == '/tmp/upstream.json'
    assert node.run(tmp_path) == {}
    assert node.run(tmp_path, input={}) == {}
    assert node.run(tmp_path, input={'default': None}) == {}
    with pytest.raises(ValueError, match='default input'):
        node.run(tmp_path, input={'other': output})
