"""Shared output-mask morphology and signed crop-radius configuration."""

from importlib import import_module

import cv2
import numpy as np
import pytest

from morphalo.nodes.preprocess.utils.mask_ops import prepare_output_mask


@pytest.mark.parametrize('radius', [-2, 0, 2])
def test_signed_radius_changes_mask_extent(radius):
    mask = np.zeros((21, 21), bool)
    mask[6:15, 6:15] = True
    result = prepare_output_mask(mask, dilate_radius=radius)
    y, x = np.nonzero(result)
    assert (x.min(), y.min(), x.max(), y.max()) == (
        6 - radius, 6 - radius, 14 + radius, 14 + radius,
    )
    assert result.dtype == np.uint8
    assert result[10, 10] == 255
    assert result[0, 0] == 0
    if radius == 0:
        np.testing.assert_array_equal(result, mask.astype(np.uint8) * 255)


def test_closing_precedes_erosion_and_smoothing_follows_it():
    mask = np.zeros((25, 25), np.uint8)
    mask[5:20, 5:20] = 255
    mask[12, 12] = 0
    result = prepare_output_mask(
        mask, close_radius=1, dilate_radius=-2, smoothing_radius=1,
    )
    filled = mask.copy()
    filled[12, 12] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expected = cv2.GaussianBlur(cv2.erode(filled, kernel), (3, 3), sigmaX=0, sigmaY=0)
    np.testing.assert_array_equal(result, expected)
    assert result[12, 12] == 255  # Closing fills the hole before erosion.
    assert 0 < result[6, 12] < 255  # Blur softens the eroded boundary.


@pytest.mark.parametrize(('module', 'reader'), [
    ('any_crop', '_read_any_crop_cfg'),
    ('face_crop', '_read_cfg'),
    ('subject_crop', '_read_cfg'),
    ('box_crop', '_read_cfg'),
    ('color_driven_crop', '_read_cfg'),
    ('sapiens2_segment_crop', '_read_cfg'),
    ('fashn_segment_crop', '_read_cfg'),
])
def test_crop_configuration_accepts_negative_dilation(module, reader):
    read = getattr(import_module(f'morphalo.nodes.preprocess.{module}'), reader)
    cfg = read({
        'model': {
            'face_landmarker_task': 'unused-face.task',
            'pose_landmarker_task': 'unused-pose.task',
        },
        'params': {'mode': 'mask', 'dilate_radius': -3},
    }, 'crop')
    assert cfg.dilate_radius == -3
