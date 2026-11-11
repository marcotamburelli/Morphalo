import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from morphalo.cache import ModelCache
from morphalo.cache import models
from morphalo.dag import DAG
from morphalo.nodes.foundation import QwenImage21, qwen_image21
from morphalo.nodes.foundation.wiring import ImageSequenceBundle


@pytest.fixture
def pipeline(monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new('RGBA', (8, 8), (10, 20, 30, 64))])

    monkeypatch.setattr(qwen_image21, 'get_qwen_image21_pipe', lambda **kwargs: generate)
    monkeypatch.setattr(qwen_image21, 'cuda_prerun', lambda device: None)
    monkeypatch.setattr(qwen_image21, 'cuda_sync', lambda device: None)
    monkeypatch.setattr(qwen_image21, 'cuda_mem_stats', lambda device: {})
    return calls


def make_node(tmp_path, spec=None):
    with DAG('qwen21', out_dir=tmp_path):
        return QwenImage21(name='out', spec=spec or {'prompt': 'A dragon sticker.', 'seed': 123})


def test_generation_preserves_output_alpha_and_metadata(pipeline, tmp_path):
    node = make_node(tmp_path)
    out = node.run(tmp_path)
    assert out['ok'] is True
    assert out['seed'] == 123
    assert 'image' not in pipeline[0]
    assert pipeline[0]['num_inference_steps'] == 40
    assert pipeline[0]['generator'].initial_seed() == 123
    assert 'guidance_scale' not in pipeline[0]
    with Image.open(out['image']) as image:
        assert image.mode == 'RGBA'
        assert image.getpixel((0, 0)) == (10, 20, 30, 64)
    metadata = json.loads(Path(out['metadata']).read_text())
    assert metadata['params']['num_input_images'] == 0
    assert metadata['model']['dtype'] == 'bfloat16'
    assert metadata['model']['device'] == 'cuda'
    assert metadata['model']['cpu_offload'] is True
    assert 'device_map' not in metadata['model']


def test_edit_preserves_alpha_order_and_prompt_override(pipeline, tmp_path):
    paths = []
    for alpha in (30, 60, 90):
        path = tmp_path / f'ref_{alpha}.png'
        Image.new('RGBA', (8, 8), (1, 2, 3, alpha)).save(path)
        paths.append(str(path))
    with DAG('qwen21_edit', out_dir=tmp_path):
        node = QwenImage21(name='out', spec={'prompt': 'Local prompt.', 'seed': 123})
        node.image(5)
        node.condition(1)
    out = node.run(tmp_path, input={
        'default': {'image': paths[0]},
        'image:5': {'path': paths[2]},
        'image:1': {'images': [paths[1]]},
        'prompt:default': {'prompt': 'Preserve these subjects.', 'negative_prompt': 'blur'},
    })
    assert pipeline[0]['prompt'] == 'Preserve these subjects.'
    assert pipeline[0]['negative_prompt'] == 'blur'
    assert [im.getpixel((0, 0))[3] for im in pipeline[0]['image']] == [30, 60, 90]
    assert [item['path'] for item in out['params']['input_images']] == paths
    # Other foundation nodes retain their RGB input contract.
    assert ImageSequenceBundle([], input={'default': {'image': paths[0]}}).images[0].mode == 'RGB'


def test_single_reference_and_turbo_defaults(pipeline, tmp_path):
    path = tmp_path / 'ref.png'
    Image.new('RGB', (8, 8)).save(path)
    node = make_node(tmp_path, {
        'model': {'id': qwen_image21.QWEN_IMAGE21_TURBO_MODEL_ID},
        'prompt': 'Make it blue.',
        'params': {'width': 512, 'height': 512, 'use_kv_cache': False, 'true_cfg_scale': 1},
    })
    node.run(tmp_path, input={'default': {'image': str(path)}})
    assert isinstance(pipeline[0]['image'], Image.Image)
    assert pipeline[0]['num_inference_steps'] == 8
    assert pipeline[0]['width'] == 512
    assert pipeline[0]['height'] == 512
    assert pipeline[0]['use_kv_cache'] is False


def test_rejects_more_than_ten_references_before_loading(monkeypatch, tmp_path):
    path = tmp_path / 'ref.png'
    Image.new('RGB', (8, 8)).save(path)
    monkeypatch.setattr(qwen_image21, 'get_qwen_image21_pipe', lambda **kw: pytest.fail('Loaded model'))
    with pytest.raises(ValueError, match='at most 10'):
        make_node(tmp_path).run(tmp_path, input={'default': {'images': [str(path)] * 11}})


@pytest.mark.parametrize(('spec', 'message'), [
    ({'prompt': ''}, 'non-empty prompt'),
    ({'prompt': 'x', 'params': {'guidance_scale': 4}}, 'true_cfg_scale'),
    ({'prompt': 'x', 'params': {'steps': 0}}, 'steps must be > 0'),
    ({'prompt': 'x', 'model': {'device_map': 'balanced'}}, 'model.device and model.cpu_offload'),
])
def test_invalid_configuration_fails_before_loading(tmp_path, spec, message):
    with pytest.raises(ValueError, match=message):
        make_node(tmp_path, spec).run(tmp_path)


def test_cache_reuse_isolation_and_eviction(monkeypatch):
    import diffusers

    loaded = []

    def load(model_id, **kwargs):
        pipe = SimpleNamespace(model_id=model_id, kwargs=kwargs, placements=[])
        pipe.vae = SimpleNamespace(enable_tiling=lambda: pipe.placements.append(('tiling',)))
        pipe.enable_model_cpu_offload = lambda **options: pipe.placements.append(('offload', options))

        def to(device):
            pipe.placements.append(('to', device))
            return pipe

        pipe.to = to
        loaded.append(pipe)
        return pipe

    monkeypatch.setattr(diffusers, 'QwenImage21Pipeline', SimpleNamespace(from_pretrained=load))
    # Isolate this test's cache without touching other test-owned entries.
    monkeypatch.setattr(ModelCache, '_store', {})
    cfg = {'model_id': qwen_image21.DEFAULT_QWEN_IMAGE21_MODEL_ID, 'dtype': torch.bfloat16}
    first = models.get_qwen_image21_pipe(**cfg)
    assert models.get_qwen_image21_pipe(**cfg) is first
    turbo = models.get_qwen_image21_pipe(**{**cfg, 'model_id': qwen_image21.QWEN_IMAGE21_TURBO_MODEL_ID})
    assert turbo is not first
    assert first.kwargs == {'torch_dtype': torch.bfloat16}
    assert first.placements == [('offload', {'device': 'cuda'})]

    tiled_cfg = {**cfg, 'vae_tiling': True}
    tiled = models.get_qwen_image21_pipe(**tiled_cfg)
    assert tiled is not first
    assert tiled.placements == [('tiling',), ('offload', {'device': 'cuda'})]
    assert models.get_qwen_image21_pipe(**tiled_cfg) is tiled
    assert models.evict_qwen_image21_pipe(**tiled_cfg) is tiled

    direct_cfg = {**cfg, 'cpu_offload': False}
    direct = models.get_qwen_image21_pipe(**direct_cfg)
    assert direct is not first
    assert direct.placements == [('to', 'cuda')]
    assert models.evict_qwen_image21_pipe(**direct_cfg) is direct

    cpu_cfg = {**cfg, 'device': 'cpu'}
    cpu = models.get_qwen_image21_pipe(**cpu_cfg)
    assert cpu.placements == [('to', 'cpu')]
    assert models.evict_qwen_image21_pipe(**cpu_cfg) is cpu

    indexed_cfg = {**cfg, 'device': 'cuda:1'}
    indexed = models.get_qwen_image21_pipe(**indexed_cfg)
    assert indexed.placements == [('offload', {'device': 'cuda:1'})]
    assert models.evict_qwen_image21_pipe(**indexed_cfg) is indexed
    assert models.evict_qwen_image21_pipe(**cfg) is first
    assert models.get_qwen_image21_pipe(**cfg) is not first
    assert len(loaded) == 7


@pytest.mark.parametrize('device', ['cpu', 'cuda:1'])
def test_node_uses_requested_device_for_loading_stats_and_eviction(monkeypatch, tmp_path, device):
    loaded = []
    evicted = []
    stats = []
    monkeypatch.setattr(qwen_image21, 'get_qwen_image21_pipe', lambda **kw: (
        loaded.append(kw) or (lambda **args: SimpleNamespace(images=[Image.new('RGBA', (8, 8))]))
    ))
    monkeypatch.setattr(qwen_image21, 'evict_qwen_image21_pipe', lambda **kw: evicted.append(kw) or object())
    monkeypatch.setattr(qwen_image21, 'cuda_prerun', stats.append)
    monkeypatch.setattr(qwen_image21, 'cuda_sync', lambda device: None)
    monkeypatch.setattr(qwen_image21, 'cuda_mem_stats', lambda device: {})
    monkeypatch.setattr(qwen_image21, 'synchronize_torch_cuda', lambda: None)
    monkeypatch.setattr(qwen_image21, 'cleanup_torch_cuda', lambda: None)
    node = make_node(tmp_path, {
        'model': {'device': device, 'cpu_offload': False, 'vae_tiling': True},
        'prompt': 'A sticker.',
    })
    node.evict_after_run = True
    out = node.run(tmp_path)
    node.post_run()
    assert loaded == evicted == [{
        'model_id': qwen_image21.DEFAULT_QWEN_IMAGE21_MODEL_ID,
        'dtype': torch.bfloat16,
        'device': device,
        'cpu_offload': False,
        'vae_tiling': True,
    }]
    assert out['model']['vae_tiling'] is True
    assert stats == [device]
    assert node.uses_cuda is device.startswith('cuda')


def test_post_run_cleans_up_even_when_eviction_fails(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(qwen_image21, 'synchronize_torch_cuda', lambda: events.append('sync'))
    monkeypatch.setattr(qwen_image21, 'cleanup_torch_cuda', lambda: events.append('cleanup'))

    def fail(**kwargs):
        raise RuntimeError('eviction failed')

    monkeypatch.setattr(qwen_image21, 'evict_qwen_image21_pipe', fail)
    node = make_node(tmp_path)
    node.evict_after_run = True
    with pytest.raises(RuntimeError, match='eviction failed'):
        node.post_run()
    assert events == ['sync', 'cleanup']


def test_call_parameters_match_installed_diffusers(pipeline, tmp_path):
    from diffusers import QwenImage21Pipeline

    make_node(tmp_path, {
        'prompt': 'A sticker.',
        'params': {'output_resolution': 2048, 'use_kv_cache': True, 'true_cfg_scale': 1.0},
    }).run(tmp_path)
    # Bind against the actual upstream signature without loading any weights.
    inspect.signature(QwenImage21Pipeline.__call__).bind(None, **pipeline[0])
