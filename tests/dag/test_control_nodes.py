import pytest

from morphalo.dag import DAG
from morphalo.dag.control_nodes import CudaCooldown


def test_cooldown_forwards_default_output(tmp_path):
    with DAG('cooldown', out_dir=tmp_path):
        node = CudaCooldown(name='cooldown')
    output = {'image': '/tmp/image.png', 'prompt': 'test'}
    assert node.forces_cuda_cooldown is True
    assert node.uses_cuda is False
    assert node.run(tmp_path, input={'default': output}) is output
    assert node.run(tmp_path) == {}
    assert node.run(tmp_path, input={}) == {}
    with pytest.raises(ValueError, match='default input'):
        node.run(tmp_path, input={'other': output})
