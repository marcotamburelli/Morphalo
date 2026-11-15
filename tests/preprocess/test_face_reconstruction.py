"""3DDFA-V3 parameter, alignment, rendering, and node contract checks."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from morphalo.cache.models import resolve_reconstruction_assets
from third_party.tddfa_v3.runtime import COEFFICIENT_SLICES, FaceReconstructionRuntime
from morphalo.nodes.preprocess.face_reconstruction import FaceReconstruction, _read_cfg
from morphalo.nodes.preprocess.utils.face_reconstruction_geometry import (
    AlignedFace, align_reconstruction_face, reconstruction_to_image,
    render_reconstruction, transfer_reconstruction_parameters,
)


def _alignment_fixture():
    reference = np.array([
        [-0.6, 0.6, 0], [0.6, 0.6, 0], [0, 0, 0.6],
        [-0.4, -0.5, 0], [0.4, -0.5, 0],
    ])
    points = np.zeros((478, 3))
    points[[33, 133], :2] = (60, 60)
    points[[362, 263], :2] = (140, 60)
    points[1, :2] = (100, 100)
    points[61, :2], points[291, :2] = (75, 140), (125, 140)
    return points, reference


def test_configuration_fixes_identity_pose_and_lighting_origins():
    model = {'face_landmarker_task': 'unused.task'}
    cfg = _read_cfg({'model': model}, 'reconstruction')
    assert cfg.preserve == ()
    assert cfg.occlusion_threshold == 0.
    assert cfg.color_transfer == 0.
    assert cfg.save_debug is False
    assert cfg.repo_id == 'Zidu-Wang/3DDFA-V3'
    alternative = _read_cfg({'model': {**model, 'id': 'alternative/reconstruction'}}, 'reconstruction')
    assert alternative.repo_id == 'alternative/reconstruction'
    for preserve in ([], ['albedo'], ['expression', 'albedo']):
        cfg = _read_cfg({'model': model, 'params': {'preserve': preserve}}, 'reconstruction')
        assert cfg.preserve == tuple(preserve)
    for preserve in (['identity'], ['rotation'], ['illumination'],
                     ['translation'], 'identity', [None]):
        with pytest.raises(ValueError, match='preserve'):
            _read_cfg({'model': model, 'params': {'preserve': preserve}}, 'reconstruction')
    with pytest.raises(ValueError, match='face_landmarker_task'):
        _read_cfg({}, 'reconstruction')
    with pytest.raises(ValueError, match='debug'):
        _read_cfg({'model': model, 'debug': False}, 'reconstruction')


@pytest.mark.parametrize('preserve', [
    (), ('expression',), ('albedo',), ('expression', 'albedo'),
])
def test_source_mean_transfers_only_selected_groups_without_changing_target(preserve):
    target = np.linspace(0, 1, 257, dtype=np.float32)
    before = target.copy()
    sources = [np.full(257, 2, np.float32), np.full(257, 4, np.float32)]
    mean, mixed = transfer_reconstruction_parameters(sources, target, preserve)
    np.testing.assert_array_equal(mean, np.full(257, 3, np.float32))
    for name, indices in COEFFICIENT_SLICES.items():
        expected = mean[indices] if name == 'identity' or name in preserve else target[indices]
        np.testing.assert_array_equal(mixed[indices], expected)
    np.testing.assert_array_equal(mixed[224:], target[224:])
    np.testing.assert_array_equal(target, before)
    for invalid in ([], [np.zeros(256)], [np.full(257, np.nan)]):
        with pytest.raises(ValueError):
            transfer_reconstruction_parameters(invalid, target, preserve)


def test_alignment_matrix_matches_actual_crop_image_and_round_trips():
    landmarks, reference = _alignment_fixture()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    locations = landmarks[[33, 263, 1, 61, 291], :2]
    for xy in locations:
        cv2.circle(image, tuple(xy.astype(int)), 3, (255, 0, 0), -1)
    alignment = align_reconstruction_face(image, landmarks, reference)
    assert alignment.rgb.shape == (224, 224, 3)
    original = np.column_stack((alignment.landmarks, np.ones(5)))
    cropped = original @ alignment.image_to_crop.T
    np.testing.assert_allclose(cropped @ alignment.crop_to_image.T, original, atol=1e-9)
    for x, y, _ in cropped:
        px, py = int(round(x)), int(round(y))
        assert alignment.rgb[py - 2:py + 3, px - 2:px + 3, 0].max() > 200
        assert alignment.rgb[py - 2:py + 3, px - 2:px + 3, 1:].max() == 0


def _decoded(xy, z, color):
    count = len(xy)
    return {
        'screen_vertices': np.asarray(xy, dtype=float),
        'camera_vertices': np.column_stack((np.zeros((count, 2)), z)),
        'colors': np.broadcast_to(color, (count, 3)).copy(),
        'gray_colors': np.full((count, 3), 0.75),
    }


def test_rasterizer_uses_nearest_depth_independently_of_winding_or_order():
    xy = [[1, 1], [10, 1], [1, 10]] * 2
    data = _decoded(xy, [5, 5, 5, 10, 10, 10], [1, 0, 0])
    data['colors'][:3] = [0, 0, 1]
    for triangles in (np.array([[3, 4, 5], [0, 1, 2]]), np.array([[2, 1, 0], [5, 4, 3]])):
        raster = render_reconstruction(data, triangles, size=224)
        np.testing.assert_array_equal(raster['color'][3, 3], [0, 0, 255])
        assert raster['depth'][3, 3] == pytest.approx(5)
        assert np.isnan(raster['depth'][20, 20])


def test_rasterizer_interpolates_attributes_in_perspective():
    data = _decoded([[0, 0], [4, 0], [0, 4]], [1, 2, 4], [0, 0, 0])
    data['colors'] = np.eye(3)
    raster = render_reconstruction(data, np.array([[0, 1, 2]]), size=224)
    weights = np.array([0.5, 0.25, 0.25]) / [1, 2, 4]
    expected = np.rint(weights / weights.sum() * 255).astype(np.uint8)
    np.testing.assert_array_equal(raster['color'][1, 1], expected)


def test_render_maps_back_using_the_alignment_transform():
    alignment = AlignedFace(np.zeros((224, 224, 3), np.uint8),
                            np.array([[2, 0, -10], [0, 2, -20], [0, 0, 1.]]),
                            np.eye(3), np.zeros((5, 2)))
    image = np.zeros((224, 224, 3), np.uint8)
    image[20:41, 20:41] = [255, 0, 0]
    raster = {'color': image, 'shape': image, 'mask': image[:, :, 0] > 0}
    mapped = reconstruction_to_image(raster, alignment, (100, 100, 3))
    # Crop [30, 30] corresponds to original [(30+10)/2, (30+20)/2].
    np.testing.assert_array_equal(mapped['color'][25, 20], [255, 0, 0])
    assert mapped['mask'][25, 20]
    assert not mapped['mask'][0, 0]


def test_runtime_normalizes_only_rgb_range_and_disables_gradients():
    import torch

    class Network(torch.nn.Module):
        def forward(self, image):
            assert not torch.is_grad_enabled()
            assert image.shape == (1, 3, 224, 224)
            assert image[0, 0, 0, 0] == 1
            assert image[0, 1, 0, 0] == 0
            return torch.zeros((1, 257))

    runtime = FaceReconstructionRuntime.__new__(FaceReconstructionRuntime)
    runtime.network = Network()
    runtime.device = 'cpu'
    rgb = np.zeros((224, 224, 3), np.uint8)
    rgb[:, :, 0] = 255
    assert runtime.infer(rgb).shape == (257,)
    with pytest.raises(ValueError, match='224'):
        runtime.infer(rgb[:10])


def test_assets_use_standard_hugging_face_cache_without_directory_overrides(monkeypatch):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return f"/standard-cache/{kwargs['filename']}"

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    paths = resolve_reconstruction_assets(repo_id='alternative/3DDFA-V3', revision='fixed-revision')
    assert set(paths) == {'face_model.npy', 'net_recon.pth', 'similarity_Lm3D_all.mat'}
    assert len(calls) == 3
    for call in calls:
        assert set(call) == {'repo_id', 'repo_type', 'revision', 'filename'}
        assert call['repo_id'] == 'alternative/3DDFA-V3'
        assert call['repo_type'] == 'dataset'
        assert call['revision'] == 'fixed-revision'


@pytest.mark.parametrize('with_occlusion', [False, True])
@pytest.mark.parametrize('save_debug', [None, False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda:1'])
@pytest.mark.parametrize('preserve', [[], ['albedo']])
def test_node_persists_source_target_and_mixed_renders_and_rejects_missing_faces(
    tmp_path, monkeypatch, save_debug, device, preserve, with_occlusion,
):
    import json
    from types import SimpleNamespace
    from morphalo.dag import DAG
    from morphalo.nodes.preprocess import face_reconstruction as module

    landmarks, reference = _alignment_fixture()
    target = np.linspace(0.01, 0.03, 257, dtype=np.float32)
    first, second = target.copy(), target.copy()
    first[:80], second[:80] = 1, 3
    first[144:224], second[144:224] = 4, 6
    remaining = [target.copy(), first.copy(), second.copy()]
    decoded_fields = []
    observed_coefficients = []
    statistics_calls = []
    clock = [100.]
    save_reconstruction = module._save_reconstruction

    def timed_save(*args, **kwargs):
        result = save_reconstruction(*args, **kwargs)
        # Give each completed diagnostic save a deterministic one-second cost.
        # Stopping the timer before the mixed save would lose the fourth second.
        clock[0] += 1.
        return result

    monkeypatch.setattr(module, '_save_reconstruction', timed_save)
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    memory = {'allocated_gb': 0.5, 'peak_allocated_gb': 0.8} if device != 'cpu' else {}

    def prerun(value):
        assert value == device and remaining
        statistics_calls.append('prerun')

    def sync(value):
        assert value == device and not remaining
        statistics_calls.append('sync')

    def mem_stats(value):
        assert value == device and statistics_calls == ['prerun', 'sync']
        statistics_calls.append('memory')
        return memory

    monkeypatch.setattr(module, 'cuda_prerun', prerun)
    monkeypatch.setattr(module, 'cuda_sync', sync)
    monkeypatch.setattr(module, 'cuda_mem_stats', mem_stats)

    class Runtime:
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        assets = {'face_model.npy': tmp_path / 'external.npy'}

        def infer(self, crop):
            assert crop.shape == (224, 224, 3)
            return remaining.pop(0)

        def decode(self, coefficients, *, render_fields=('color', 'shape')):
            decoded_fields.append(render_fields)
            observed_coefficients.append(coefficients.copy())
            width = 40 + coefficients[0]
            xy = [[112 - width, 60], [112 + width, 60],
                  [112 + width, 170], [112 - width, 170]]
            result = _decoded(xy, [10] * 4, [0.6, 0.4, 0.3])
            result['canonical_vertices'] = np.column_stack((xy, np.zeros(4)))
            result['albedo'] = result['colors'].copy()
            return result

    runtime = Runtime()
    runtime.reference = reference
    monkeypatch.setattr(module, 'get_face_reconstruction_model', lambda **kwargs: runtime)
    monkeypatch.setattr(module, 'get_mediapipe_face_landmarker', lambda **kwargs: object())

    def detect(img_rgb, **kwargs):
        if img_rgb[0, 0, 0] == 13:
            raise RuntimeError('MediaPipe found no face landmarks.')
        return landmarks.copy()

    monkeypatch.setattr(module, 'mp_face_landmarks_xyz', detect)
    files = []
    for name, value in (('target', 30), ('first', 20), ('missing', 13), ('second', 40)):
        path = tmp_path / f'{name}.png'
        Image.fromarray(np.full((200, 200, 3), value, np.uint8)).save(path)
        files.append(str(path))
    occlusion_path = tmp_path / 'occlusion.png'
    occlusion = np.zeros((100, 100), np.uint8)
    occlusion[55:60] = 128
    occlusion[60:] = 255
    if with_occlusion:
        Image.fromarray(occlusion).save(occlusion_path)
    color_transfer = 0.35 if with_occlusion and not preserve else 0.
    with DAG('face_reconstruction_test', out_dir=tmp_path) as dag:
        node = FaceReconstruction(name='reconstruct', spec={
            'model': {'face_landmarker_task': 'unused.task', 'device': device},
            'params': {'preserve': preserve, 'color_transfer': color_transfer, 'occlusion_threshold': 0.5},
            **({'debug': {'save_debug': save_debug}} if save_debug is not None else {}),
        })
        if with_occlusion:
            from morphalo.nodes import FileImage
            occlusion_node = FileImage(name='occluder', path=str(occlusion_path))
            occlusion_node >> node.occlusion()
    if with_occlusion:
        assert any(edge.node_from == 'occluder' and edge.node_to == 'reconstruct'
                   and edge.input_id == 'occlusion' for edge in dag.edges)
    payload = {'default': {'image': files[0]}, 'source': {'images': files[1:]}}
    if with_occlusion:
        payload['occlusion'] = {'image': str(occlusion_path)}
    out = node.run(tmp_path, input=payload)
    assert ('occlusion_image' in out) == with_occlusion
    if with_occlusion:
        assert out['occlusion_image'] == str(occlusion_path)
    assert remaining == []
    assert len(out['source_views']) == 2 and len(out['rejected_source_images']) == 1
    assert out['parameter_origins']['identity'] == 'source_mean'
    assert out['parameter_origins']['illumination'] == 'target'
    assert out['parameter_origins']['rotation'] == 'target'
    assert node.uses_cuda == (device != 'cpu')
    assert statistics_calls == ['prerun', 'sync', 'memory']
    assert out['timing']['seconds'] == (4. if save_debug else 0.)
    assert out.get('cuda_mem', {}) == memory
    assert ('cuda_mem' in out) == bool(memory)
    assert out['params'] == {'preserve': preserve, 'color_transfer': color_transfer, 'occlusion_threshold': 0.5}
    if color_transfer:
        assert out['composition']['color_transfer']['applied']
        assert out['composition']['color_transfer']['sample_pixels'] > 0
    assert out['output_kind'] == 'reconstruction_composite'
    assert out['composite'] == out['image']
    mixed = observed_coefficients[-1]
    np.testing.assert_array_equal(mixed[:80], [2.] * 80)
    expected_albedo = np.full(80, 5.) if 'albedo' in preserve else target[144:224]
    np.testing.assert_array_equal(mixed[144:224], expected_albedo)
    for group in ('rotation', 'translation', 'illumination', 'expression'):
        np.testing.assert_array_equal(mixed[COEFFICIENT_SLICES[group]], target[COEFFICIENT_SLICES[group]])
    card = json.loads(Path(out['metadata']).read_text())
    assert card['timing'] == out['timing']
    assert card.get('cuda_mem', {}) == memory
    debug_dir = Path(out['image']).with_name(Path(out['image']).stem + '_debug')
    assert all(set(view) == {'image'} for view in out['source_views'])
    assert 'warp' not in out['composition']
    assert not any(name in out for name in (
        'target', 'mixed', 'debug_directory', 'mean_source_coefficients',
    ))
    if save_debug:
        assert decoded_fields == [('color', 'shape')] * 4
        debug = out['debug']
        assert card['debug'] == debug
        assert debug['debug_directory'] == str(debug_dir)
        assert len(debug['source_views']) == 2
        assert [view['image'] for view in debug['source_views']] == out['source_images']
        assert debug['warp']['min_jacobian'] > 0
        for view in [*debug['source_views'], debug['target'], debug['mixed']]:
            for path in view['debug'].values():
                assert Path(path).is_file()
            with np.load(view['debug']['geometry']) as geometry:
                assert geometry['coefficients'].shape == (257,)
        assert debug['mean_source_coefficients']['identity'] == [2.] * 80
        assert len(list(debug_dir.glob('*.png'))) == (21 if with_occlusion else 20)
        if with_occlusion:
            with Image.open(debug['occlusion_mask']) as mask_image:
                expected_mask = np.repeat(np.repeat(occlusion, 2, axis=0), 2, axis=1)
                np.testing.assert_array_equal(np.array(mask_image), expected_mask)
        assert len(list(debug_dir.glob('*.npz'))) == 4
    else:
        assert decoded_fields == [(), ('color',)]
        assert not debug_dir.exists()
        assert 'debug' not in card
        assert 'coefficients' not in json.dumps(card)
        assert not any(name in out for name in ('target', 'mixed', 'debug_directory'))
        assert all(set(view) == {'image'} for view in out['source_views'])
        assert sorted(path.suffix for path in Path(out['image']).parent.iterdir()) == ['.json', '.png']
    with Image.open(out['image']) as image:
        assert image.size == (200, 200)
        assert image.mode == 'RGB'
        composite = np.array(image)
    if with_occlusion:
        np.testing.assert_array_equal(composite[120:], np.full((80, 200, 3), 30, np.uint8))
        assert np.any(composite[:120] != 30)
    if save_debug:
        # The diagnostic path must produce exactly the same composite.
        node.spec = {**node.spec, 'debug': {'save_debug': False}}
        remaining.extend([target.copy(), first.copy(), second.copy()])
        statistics_calls.clear()
        baseline = node.run(tmp_path / 'without_debug', input=payload)
        with Image.open(baseline['image']) as image:
            np.testing.assert_array_equal(composite, np.array(image))
    assert json.loads(Path(out['metadata']).read_text())['ok']



def test_optional_render_fields_preserve_visibility_and_color():
    data = _decoded([[20, 20], [100, 20], [20, 100]], [1, 2, 4], [0.6, 0.4, 0.2])
    triangles = np.array([[0, 1, 2]])
    complete = render_reconstruction(data, triangles, size=224)
    color = render_reconstruction(
        {key: value for key, value in data.items() if key != 'gray_colors'},
        triangles, size=224, render_fields=('color',),
    )
    visibility = render_reconstruction(
        {key: value for key, value in data.items() if key not in ('gray_colors', 'colors')},
        triangles, size=224, render_fields=(),
    )
    np.testing.assert_array_equal(color['color'], complete['color'])
    for raster in (color, visibility):
        np.testing.assert_array_equal(raster['mask'], complete['mask'])
        np.testing.assert_array_equal(raster['depth'], complete['depth'])
    assert set(visibility) == {'mask', 'depth'}
    alignment = AlignedFace(np.zeros((224, 224, 3), np.uint8), np.eye(3), np.eye(3), np.zeros((5, 2)))
    mapped = reconstruction_to_image(visibility, alignment, (224, 224, 3))
    assert set(mapped) == {'mask'}
    np.testing.assert_array_equal(mapped['mask'], complete['mask'])



def test_runtime_constructs_from_local_assets_without_cache_or_downloads(tmp_path, monkeypatch):
    import torch
    from scipy.io import savemat
    from third_party.tddfa_v3 import runtime as module

    data = {
        'u': np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32).reshape(-1),
        'id': np.zeros((9, 80), np.float32),
        'exp': np.zeros((9, 64), np.float32),
        'alb': np.zeros((9, 80), np.float32),
        'u_alb': np.full(9, 128, np.float32),
        'tri': np.array([[0, 1, 2]]),
        'point_buf': np.zeros((3, 1), dtype=np.int64),
    }
    np.save(tmp_path / 'face_model.npy', data)
    reference68 = np.arange(204, dtype=np.float32).reshape(68, 3)
    savemat(tmp_path / 'similarity_Lm3D_all.mat', {'lm': reference68})
    # A small checkpoint exercises strict loading without the full regressor.
    network = torch.nn.Linear(1, 257)
    torch.save({'net_recon': network.state_dict()}, tmp_path / 'net_recon.pth')
    monkeypatch.setattr(module, 'define_net_recon', lambda *args, **kw: torch.nn.Linear(1, 257))
    assets = {name: tmp_path / name for name in module.ASSET_FILENAMES}
    runtime = module.FaceReconstructionRuntime(assets=assets, device='cpu')
    assert runtime.assets == assets
    assert runtime.network.training is False
    for name, value in network.state_dict().items():
        torch.testing.assert_close(runtime.network.state_dict()[name], value)
    expected = np.stack((
        reference68[[36, 39]].mean(axis=0), reference68[[42, 45]].mean(axis=0),
        reference68[30], reference68[48], reference68[54],
    ))
    np.testing.assert_array_equal(runtime.reference, expected)
    np.testing.assert_array_equal(runtime.triangles, data['tri'])
    coefficients = np.zeros(257, np.float32)
    full = runtime.decode(coefficients)
    geometry = runtime.decode(coefficients, render_fields=())
    assert set(geometry) == {'canonical_vertices', 'camera_vertices', 'screen_vertices'}
    for key, value in geometry.items():
        np.testing.assert_array_equal(value, full[key])


def test_reconstruction_cache_uses_resolved_assets_device_and_file_signature(tmp_path, monkeypatch):
    from morphalo.cache import CacheKey
    from morphalo.cache import models
    from third_party.tddfa_v3 import runtime as module

    paths = {}
    for name in module.ASSET_FILENAMES:
        path = tmp_path / name
        path.write_bytes(b'initial')
        paths[name] = path
    resolved = []
    constructed = []
    cache = {}

    def resolve(**kwargs):
        resolved.append(kwargs)
        return paths

    def construct(**kwargs):
        constructed.append(kwargs)
        return object()

    monkeypatch.setattr(models, 'resolve_reconstruction_assets', resolve)
    monkeypatch.setattr(module, 'FaceReconstructionRuntime', construct)
    monkeypatch.setattr(models.ModelCache, '_store', cache)
    kwargs = {'repo_id': 'alternative/reconstruction', 'revision': 'fixed'}
    first = models.get_face_reconstruction_model(**kwargs, device='cpu')
    assert models.get_face_reconstruction_model(**kwargs, device='cpu') is first
    assert models.get_face_reconstruction_model(**kwargs, device='cuda:1') is not first
    paths['net_recon.pth'].write_bytes(b'updated checkpoint')
    assert models.get_face_reconstruction_model(**kwargs, device='cpu') is not first
    assert len(constructed) == 3
    assert all(call == kwargs for call in resolved)
    assert constructed[0] == {'assets': paths, 'device': 'cpu'}
    assert constructed[1]['device'] == 'cuda:1'
    assert all(isinstance(key, CacheKey) and key.kind == 'tddfa_v3' for key in cache)



def test_occlusion_loader_preserves_soft_values_when_resizing(tmp_path):
    from morphalo.nodes.preprocess.face_reconstruction import _load_occlusion

    values = np.array([[0, 127], [128, 255]], np.uint8)
    path = tmp_path / 'occlusion.png'
    Image.fromarray(values).save(path)
    actual = _load_occlusion(path, (4, 6, 3))
    expected = np.repeat(np.repeat(values.astype(np.float32) / 255., 2, axis=0), 3, axis=1)
    np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize('strength', [-0.1, 1.1, float('nan'), float('inf'), None, 'bad'])
def test_configuration_rejects_invalid_color_transfer(strength):
    with pytest.raises(ValueError, match='color_transfer'):
        _read_cfg({
            'model': {'face_landmarker_task': 'unused.task'},
            'params': {'color_transfer': strength},
        }, 'reconstruction')


def test_color_transfer_requires_target_albedo():
    model = {'face_landmarker_task': 'unused.task'}
    with pytest.raises(ValueError, match='incompatible'):
        _read_cfg({'model': model, 'params': {'preserve': ['albedo'], 'color_transfer': 0.5}}, 'reconstruction')
    cfg = _read_cfg({'model': model, 'params': {'preserve': ['expression'], 'color_transfer': 0.5}}, 'reconstruction')
    assert cfg.color_transfer == 0.5
    assert _read_cfg({'model': model, 'params': {'preserve': ['albedo'], 'color_transfer': 0}}, 'reconstruction').color_transfer == 0


@pytest.mark.parametrize('threshold', [-0.1, 1.1, float('nan'), float('inf'), None, 'bad'])
def test_configuration_rejects_invalid_occlusion_threshold(threshold):
    with pytest.raises(ValueError, match='occlusion_threshold'):
        _read_cfg({
            'model': {'face_landmarker_task': 'unused.task'},
            'params': {'occlusion_threshold': threshold},
        }, 'reconstruction')


@pytest.mark.parametrize('threshold', [0., 0.5, 1.])
def test_configuration_accepts_occlusion_threshold(threshold):
    cfg = _read_cfg({
        'model': {'face_landmarker_task': 'unused.task'},
        'params': {'occlusion_threshold': threshold},
    }, 'reconstruction')
    assert cfg.occlusion_threshold == threshold
