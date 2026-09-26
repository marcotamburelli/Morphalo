import numpy as np

from morphalo.nodes.preprocess.utils.face_luminance_compositing import (
    compare_face_masks,
    composite_face_luminance,
    gray_to_rgb,
    sample_landmark_luminance,
    sharpen_face_luminance,
    warp_face_luminance_background,
)


def test_compare_face_masks_reports_scale_and_center_offset():
    target = np.zeros((20, 30), dtype=bool)
    synthetic = np.zeros_like(target)
    target[3:15, 4:24] = True
    synthetic[5:13, 8:18] = True

    comparison = compare_face_masks(target, synthetic)

    assert comparison['target']['bbox'] == [4, 3, 23, 14]
    assert comparison['target']['width'] == 20
    assert comparison['target']['height'] == 12
    assert comparison['synthetic']['width'] == 10
    assert comparison['synthetic']['height'] == 8
    assert comparison['target_to_synthetic_width_ratio'] == 2.0
    assert comparison['target_to_synthetic_height_ratio'] == 1.5
    assert comparison['target_to_synthetic_area_ratio'] == 3.0
    assert comparison['synthetic_center_offset'] == [-1.0, 0.0]


def test_sample_landmark_luminance_returns_local_values():
    image = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    landmarks = np.asarray(((8, 12), (24, 20)), dtype=np.float32)

    sampled = sample_landmark_luminance(
        image,
        landmarks,
        blur_sigma=1.0,
    )

    np.testing.assert_allclose(sampled, (8, 24), atol=1)


def test_sharpen_face_luminance_enhances_detail_without_darkening_mask_edge():
    face = np.zeros((32, 32), dtype=np.uint8)
    mask = np.zeros_like(face, dtype=bool)
    mask[5:27, 5:27] = True
    face[mask] = 120
    face[13:19, 13:19] = 150

    sharpened = sharpen_face_luminance(
        face,
        mask,
        detail_gain=1.5,
    )

    assert sharpened[5, 16] == 120
    assert sharpened[13, 16] > face[13, 16]
    assert sharpened[0, 0] == 0


def test_warp_face_luminance_background_is_local_and_closes_gap():
    y, x = np.mgrid[:96, :96]
    image = np.rint(30.0 + x + y * 0.5).astype(np.uint8)
    erase_mask = (x - 48) ** 2 + (y - 48) ** 2 <= 32 ** 2
    insert_mask = (x - 48) ** 2 + (y - 48) ** 2 <= 22 ** 2

    warped, displacement, params = warp_face_luminance_background(
        image,
        erase_mask,
        insert_mask,
        face_span=64.0,
        feather_radius=3,
    )

    assert displacement[48, 70] > 0.0
    assert displacement[0, 0] == 0.0
    np.testing.assert_array_equal(warped[0, :], image[0, :])
    assert np.all(np.isfinite(displacement))
    assert params['influence_radius'] == 16
    assert params['control_count'] >= 32


def test_composite_face_luminance_keeps_background_and_softens_edge():
    background = np.zeros((31, 31), dtype=np.uint8)
    face = np.full((31, 31), 200, dtype=np.uint8)
    mask = np.zeros_like(background, dtype=bool)
    mask[5:26, 5:26] = True

    composed, alpha = composite_face_luminance(
        background,
        face,
        mask,
        feather_radius=4,
    )

    assert composed[0, 0] == 0
    assert 0 < composed[5, 15] < 200
    assert composed[15, 15] == 200
    assert 0.0 < alpha[5, 15] < 1.0
    assert alpha[15, 15] == 1.0


def test_gray_to_rgb_repeats_luminance_channels():
    gray = np.asarray(((0, 127, 255),), dtype=np.uint8)

    output = gray_to_rgb(gray)

    assert output.shape == (1, 3, 3)
    np.testing.assert_array_equal(output[0, 1], (127, 127, 127))
