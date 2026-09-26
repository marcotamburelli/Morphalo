import numpy as np
import pytest

from morphalo.nodes.preprocess.face_geometry_map import _read_cfg
from morphalo.nodes.vision.face_geometry import (
    _depth_yaw_weight,
    draw_luminance_map,
    _draw_landmarks,
    _estimate_landmark_similarity,
    _fuse_source_landmarks,
    _transform_landmarks,
    _yaw_degrees_from_transform,
    draw_geometry_alignment,
    face_mesh_mask,
)


def test_read_cfg_requires_face_landmarker_task():
    with pytest.raises(ValueError, match='face_landmarker_task required'):
        _read_cfg({}, 'transfer')

    cfg = _read_cfg(
        {
            'model': {
                'face_landmarker_task': '~/models/face_landmarker.task',
            },
            'params': {
                'processor': 'canny',
            },
        },
        'transfer',
    )

    assert cfg.device == 'cpu'
    assert cfg.face_landmarker_task == '~/models/face_landmarker.task'
    assert cfg.processor == 'canny'
    assert cfg.detail_gain == 0.0


def test_read_cfg_requires_supported_processor():
    model = {'face_landmarker_task': '~/models/face_landmarker.task'}

    with pytest.raises(ValueError, match='params.processor'):
        _read_cfg({'model': model}, 'transfer')

    cfg = _read_cfg(
        {
            'model': model,
            'params': {
                'processor': 'depth_midas',
                'processor_device': 'cuda',
                'detail_gain': 1.5,
            },
        },
        'transfer',
    )

    assert cfg.processor == 'depth_midas'
    assert cfg.processor_device == 'cuda'
    assert cfg.detail_gain == 1.5

    with pytest.raises(ValueError, match='params.processor'):
        _read_cfg(
            {'model': model, 'params': {'processor': 'unknown'}},
            'transfer',
        )

    with pytest.raises(ValueError, match='dwpose'):
        _read_cfg(
            {'model': model, 'params': {'processor': 'dwpose'}},
            'transfer',
        )

    with pytest.raises(ValueError, match='detail_gain'):
        _read_cfg(
            {
                'model': model,
                'params': {
                    'processor': 'canny',
                    'detail_gain': -0.1,
                },
            },
            'transfer',
        )


def test_similarity_reproduces_3d_scale_rotation_and_translation():
    source = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]],
        dtype=np.float32,
    )
    expected_matrix = np.eye(4, dtype=np.float32)
    expected_matrix[:3, :3] = np.array(
        [[2, 0, 0], [0, 0, -2], [0, 2, 0]],
        dtype=np.float32,
    )
    expected_matrix[:3, 3] = (10, 20, 30)
    target = _transform_landmarks(source, expected_matrix)

    matrix = _estimate_landmark_similarity(source, target)
    transformed = _transform_landmarks(source, matrix)

    np.testing.assert_allclose(matrix, expected_matrix, atol=1e-5)
    np.testing.assert_allclose(transformed, target, atol=1e-5)


def test_similarity_rejects_incompatible_landmark_sets():
    with pytest.raises(ValueError, match=r'matching \(N, 3\) shapes'):
        _estimate_landmark_similarity(
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
        )


def test_draw_landmarks_clips_points_outside_the_canvas():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    landmarks = np.array([[4, 5], [-1, 3], [20, 20]], dtype=np.float32)

    output = _draw_landmarks(image, landmarks, color=(255, 255, 255))

    np.testing.assert_array_equal(output[5, 4], [255, 255, 255])
    assert np.count_nonzero(output) > 0
    np.testing.assert_array_equal(image, np.zeros_like(image))


def test_geometry_alignment_draws_both_landmarks_and_contours():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    target_mask = np.zeros((32, 32), dtype=bool)
    synthetic_mask = np.zeros_like(target_mask)
    target_mask[4:20, 4:20] = True
    synthetic_mask[8:24, 8:24] = True

    output = draw_geometry_alignment(
        image,
        np.asarray(((10, 10),), dtype=np.float32),
        np.asarray(((14, 14),), dtype=np.float32),
        target_mask,
        synthetic_mask,
    )

    np.testing.assert_array_equal(output[10, 10], (0, 255, 255))
    np.testing.assert_array_equal(output[14, 14], (255, 0, 255))
    assert np.count_nonzero(output) > 0


def test_luminance_map_interpolates_shared_vertex_values():
    landmarks = np.array(
        [[1, 1, 0], [6, 1, 0], [1, 6, 0]],
        dtype=np.float32,
    )
    luminance = np.array([20, 120, 220], dtype=np.float32)

    output, mask = draw_luminance_map(
        (8, 8, 3),
        landmarks,
        luminance,
        triangles=[(0, 1, 2)],
    )

    assert output.dtype == np.uint8
    assert output[1, 1] == 20
    assert 20 < output[2, 2] < 220
    assert mask.shape == (8, 8)
    np.testing.assert_array_equal(mask, output > 0)


def test_face_mesh_mask_fills_enclosed_mesh_openings():
    landmarks = np.array(
        [
            [1, 1], [7, 1], [7, 7], [1, 7],
            [3, 3], [5, 3], [5, 5], [3, 5],
        ],
        dtype=np.float32,
    )
    triangles = [
        (0, 1, 4), (1, 5, 4), (1, 2, 5), (2, 6, 5),
        (2, 3, 6), (3, 7, 6), (3, 0, 7), (0, 4, 7),
    ]

    mask = face_mesh_mask(
        (9, 9, 3),
        landmarks,
        triangles=triangles,
    )

    assert mask[4, 4]
    assert not mask[0, 0]


@pytest.mark.parametrize(
    ('yaw', 'expected'),
    [(0, 0.1), (6, 0.1), (30, 0.5), (60, 1.0), (70, 0.5), (80, 0.0)],
)
def test_depth_yaw_weight(yaw, expected):
    assert _depth_yaw_weight(yaw) == pytest.approx(expected)
    assert _depth_yaw_weight(-yaw) == pytest.approx(expected)


def test_yaw_is_extracted_from_face_transform():
    yaw_radians = np.deg2rad(35.0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(
        [
            [np.cos(yaw_radians), 0, np.sin(yaw_radians)],
            [0, 1, 0],
            [-np.sin(yaw_radians), 0, np.cos(yaw_radians)],
        ]
    )

    assert _yaw_degrees_from_transform(transform) == pytest.approx(35.0)


def test_fusion_uses_most_frontal_xy_and_weighted_normalized_depth():
    frontal = np.array(
        [[0, 0, 0], [2, 0, 1], [0, 2, 2], [2, 2, 0]],
        dtype=np.float32,
    )
    oblique = frontal.copy()
    oblique[:, 2] += np.array([0, 1, 0, -1], dtype=np.float32)

    fused, normalized, base_index, weights = _fuse_source_landmarks(
        [oblique, frontal],
        [60.0, 5.0],
    )

    assert base_index == 1
    assert weights == pytest.approx([1.0, 0.1])
    np.testing.assert_array_equal(fused[:, :2], frontal[:, :2])
    np.testing.assert_allclose(
        fused[:, 2],
        np.average(
            np.stack([points[:, 2] for points in normalized]),
            axis=0,
            weights=weights,
        ),
    )


def test_single_source_fusion_preserves_the_base_geometry():
    source = np.array(
        [[0, 0, 0], [2, 0, 1], [0, 2, 2], [2, 2, 0]],
        dtype=np.float32,
    )

    fused, normalized, base_index, weights = _fuse_source_landmarks(
        [source],
        [25.0],
    )

    assert base_index == 0
    assert weights == pytest.approx([25.0 / 60.0])
    np.testing.assert_array_equal(fused, source)
    np.testing.assert_array_equal(normalized[0], source)
