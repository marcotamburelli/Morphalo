from types import SimpleNamespace

import numpy as np
import pytest

from morphalo.cache.models import FaceSwapperRuntime
from morphalo.nodes.process.face_swap import (
    _align_face_crop,
    _make_paste_back_debug,
    _paste_back_face,
    _preprocess_target_crop,
    _read_cfg,
    _run_crossface_simswap,
    _run_simswap,
    _select_largest_face,
)


class Session:
    def __init__(self, inputs, output):
        self._inputs = [SimpleNamespace(name=name) for name in inputs]
        self.output = output
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feed):
        self.feed = feed
        return [self.output]


def test_read_cfg_defaults_and_validation():
    cfg = _read_cfg({}, 'swap')
    assert cfg.swapper == 'simswap_256'
    assert cfg.det_size == (640, 640)
    assert cfg.save_debug is False

    debug_cfg = _read_cfg({'debug': {'save_debug': True}}, 'swap')
    assert debug_cfg.save_debug is True

    with pytest.raises(ValueError, match='unsupported swapper'):
        _read_cfg({'model': {'swapper': 'unknown'}}, 'swap')


def test_select_largest_face_uses_bbox_area():
    small = SimpleNamespace(bbox=np.array([0, 0, 10, 10]))
    large = SimpleNamespace(bbox=np.array([2, 2, 22, 12]))
    assert _select_largest_face([small, large]) is large


def test_crossface_normalizes_converter_output():
    session = Session(['input'], np.array([[3.0, 4.0]], dtype=np.float32))
    result = _run_crossface_simswap(session, np.ones(512, dtype=np.float32))
    np.testing.assert_allclose(result, [[0.6, 0.8]])
    assert session.feed['input'].shape == (1, 512)


def test_preprocess_and_simswap_bind_by_semantic_name():
    runtime = FaceSwapperRuntime(
        converter=None,
        swapper=None,
        size=2,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    crop[:, :, 2] = 255
    tensor = _preprocess_target_crop(crop, runtime)
    np.testing.assert_allclose(tensor[0, 0], 1.0)
    np.testing.assert_allclose(tensor[0, 1:], 0.0)

    output = np.zeros((1, 3, 2, 2), dtype=np.float32)
    output[:, 0] = 1.0
    session = Session(['target', 'source'], output)
    result = _run_simswap(session, tensor, np.ones((1, 512), dtype=np.float32))
    assert set(session.feed) == {'source', 'target'}
    np.testing.assert_array_equal(result[:, :, 2], 255)


def test_alignment_and_paste_back_preserve_geometry():
    size = 112
    image = np.zeros((size, size, 3), dtype=np.uint8)
    face = SimpleNamespace(kps=np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32))
    crop, matrix = _align_face_crop(image, face, size)
    assert crop.shape == (size, size, 3)
    np.testing.assert_allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-3)

    swapped = np.full_like(crop, 255)
    result = _paste_back_face(image, swapped, matrix)
    assert result[size // 2, size // 2].min() > 240
    assert result[0, 0].max() == 0


def test_paste_back_debug_draws_transformed_crop_boundary():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    matrix = np.array([[1, 0, -8], [0, 1, -8]], dtype=np.float32)

    debug = _make_paste_back_debug(image, matrix, crop_size=16)

    np.testing.assert_array_equal(debug[8, 8], [0, 255, 255])
    np.testing.assert_array_equal(debug[23, 23], [0, 255, 255])
    np.testing.assert_array_equal(debug[16, 16], [0, 0, 0])
