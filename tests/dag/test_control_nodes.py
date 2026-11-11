import weakref

import pytest

from morphalo.cache import CacheKey, ModelCache
from morphalo.core import cuda_mem
from morphalo.core.paths import load_latest_output
from morphalo.dag import DAG, EvictModelCache
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


class CachedModel:
    pass


def test_evict_releases_cached_objects_and_forwards_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelCache, '_store', {})
    events = []
    monkeypatch.setattr(cuda_mem, 'synchronize_torch_cuda', lambda: events.append('sync'))
    monkeypatch.setattr(cuda_mem.torch.cuda, 'is_initialized', lambda: False)
    model = CachedModel()
    reference = weakref.ref(model)
    key = CacheKey(kind='test', ref='model', device='cpu', dtype='float32')
    ModelCache.put(key, model)
    del model

    original_clear = ModelCache.clear

    def clear():
        events.append('clear')
        original_clear()

    monkeypatch.setattr(ModelCache, 'clear', clear)
    with DAG('evict', out_dir=tmp_path):
        node = EvictModelCache(name='evict')
    upstream = {'image': '/tmp/image.png', 'seed': 123, 'id': 'source', 'metadata': 'old.json'}
    result = node.run(tmp_path, input={'default': upstream})

    assert events == ['sync', 'clear']
    assert ModelCache.get(key) is None
    assert reference() is None
    assert result['image'] == upstream['image']
    assert result['seed'] == 123
    assert result['id'] == 'evict'
    assert result['node'] == node.op
    assert load_latest_output(tmp_path, node.id)['image'] == upstream['image']
    assert upstream['metadata'] == 'old.json'
    assert node.forces_cuda_cooldown is False
    assert node.uses_cuda is False


@pytest.mark.parametrize('input_map', [None, {}, {'default': None}])
def test_evict_without_payload_still_clears_cache(tmp_path, monkeypatch, input_map):
    monkeypatch.setattr(ModelCache, '_store', {})
    events = []
    monkeypatch.setattr(cuda_mem, 'synchronize_torch_cuda', lambda: events.append('sync'))
    monkeypatch.setattr(cuda_mem, 'cleanup_torch_cuda', lambda: events.append('cleanup'))
    key = CacheKey(kind='test', ref='model', device='cpu', dtype='float32')
    ModelCache.put(key, object())
    with DAG('evict_empty', out_dir=tmp_path):
        node = EvictModelCache(name='evict')

    assert node.run(tmp_path, input=input_map) == {}
    assert ModelCache.get(key) is None
    assert events == ['sync', 'cleanup']


def test_evict_rejects_invalid_input_before_clearing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelCache, '_store', {})
    key = CacheKey(kind='test', ref='model', device='cpu', dtype='float32')
    model = ModelCache.put(key, object())
    with DAG('evict_invalid', out_dir=tmp_path):
        node = EvictModelCache(name='evict')

    with pytest.raises(ValueError, match='only the default input'):
        node.run(tmp_path, input={'other': {}})
    assert ModelCache.get(key) is model
