import cv2
import numpy as np
import pytest

from morphalo.nodes.preprocess.utils.face_compositing import (
    compare_face_masks,
    composite_face,
    gray_to_rgb,
    sample_landmark_luminance,
    sharpen_face_luminance,
    warp_face_background,
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


def test_warp_face_background_is_local_and_closes_gap():
    y, x = np.mgrid[:96, :96]
    image = np.rint(30.0 + x + y * 0.5).astype(np.uint8)
    erase_mask = (x - 48) ** 2 + (y - 48) ** 2 <= 32 ** 2
    insert_mask = (x - 48) ** 2 + (y - 48) ** 2 <= 22 ** 2

    warped, displacement, params = warp_face_background(
        image,
        erase_mask,
        insert_mask,
        face_span=64.0,
        erosion_radius=3,
    )

    assert displacement[48, 70] > 0.0
    assert displacement[0, 0] == 0.0
    np.testing.assert_array_equal(warped[0, :], image[0, :])
    assert np.all(np.isfinite(displacement))
    assert params['influence_radius'] == 16
    assert params['ray_count'] >= 64
    assert params['min_jacobian'] > 0


def test_composite_face_keeps_background_and_softens_edge():
    background = np.zeros((31, 31), dtype=np.uint8)
    face = np.full((31, 31), 200, dtype=np.uint8)
    mask = np.zeros_like(background, dtype=bool)
    mask[5:26, 5:26] = True

    composed, alpha = composite_face(
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


def test_background_warp_without_diagnostics_preserves_pixels_and_skips_statistics(monkeypatch):
    y, x = np.mgrid[:96, :96]
    image = np.rint(30.0 + x + y * 0.5).astype(np.uint8)
    erase = (x - 48) ** 2 + (y - 48) ** 2 <= 32 ** 2
    insert = (x - 48) ** 2 + (y - 48) ** 2 <= 22 ** 2
    expected, _, _ = warp_face_background(
        image, erase, insert, face_span=64.0, erosion_radius=3,
    )

    def unexpected_gradient(*args, **kwargs):
        raise AssertionError('Diagnostic Jacobian was computed')

    monkeypatch.setattr(np, 'gradient', unexpected_gradient)
    actual, displacement, statistics = warp_face_background(
        image, erase, insert, face_span=64.0, erosion_radius=3,
        collect_diagnostics=False,
    )
    np.testing.assert_array_equal(actual, expected)
    assert displacement is None
    assert statistics == {}



@pytest.mark.parametrize('feather_radius', [0, 3])
@pytest.mark.parametrize('insert_radius', [16, 28])
@pytest.mark.parametrize('collect_diagnostics', [False, True])
def test_rgb_warp_and_blend_match_independent_channels(
    feather_radius, insert_radius, collect_diagnostics,
):
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:64, :80]
    image = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    face = rng.integers(0, 256, image.shape, dtype=np.uint8)
    erase = (x - 40) ** 2 + (y - 32) ** 2 <= 24 ** 2
    insert = (x - 40) ** 2 + (y - 32) ** 2 <= insert_radius ** 2
    warped, displacement, parameters = warp_face_background(
        image, erase, insert, face_span=48., erosion_radius=feather_radius,
        collect_diagnostics=collect_diagnostics,
    )
    composed, alpha = composite_face(warped, face, insert, feather_radius=feather_radius)
    assert warped.shape == composed.shape == image.shape
    assert alpha.shape == insert.shape
    for channel in range(3):
        expected, channel_displacement, channel_parameters = warp_face_background(
            image[:, :, channel], erase, insert,
            face_span=48., erosion_radius=feather_radius,
            collect_diagnostics=collect_diagnostics,
        )
        expected_composite, expected_alpha = composite_face(
            expected, face[:, :, channel], insert, feather_radius=feather_radius,
        )
        np.testing.assert_array_equal(warped[:, :, channel], expected)
        np.testing.assert_array_equal(composed[:, :, channel], expected_composite)
        np.testing.assert_array_equal(alpha, expected_alpha)
        assert parameters == channel_parameters
        if collect_diagnostics:
            assert displacement.shape == insert.shape
            np.testing.assert_array_equal(displacement, channel_displacement)
        else:
            assert displacement is channel_displacement is None


def test_rgb_warp_without_gap_keeps_image_and_spatial_diagnostics():
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    mask = np.zeros((8, 9), dtype=bool)
    mask[2:6, 2:6] = True
    warped, displacement, parameters = warp_face_background(
        image, mask, mask, face_span=4., erosion_radius=0,
    )
    np.testing.assert_array_equal(warped, image)
    assert not np.shares_memory(warped, image)
    np.testing.assert_array_equal(displacement, np.zeros(mask.shape))
    assert parameters == {}


@pytest.mark.parametrize('shape', [(8,), (8, 9, 1), (8, 9, 4)])
def test_compositing_rejects_unsupported_image_layouts(shape):
    image = np.zeros(shape, dtype=np.uint8)
    mask = np.zeros((8, 9), dtype=bool)
    with pytest.raises(ValueError, match='shape'):
        warp_face_background(image, mask, mask, face_span=4., erosion_radius=0)
    with pytest.raises(ValueError, match='shape'):
        composite_face(image, image, mask, feather_radius=0)


def test_compositing_rejects_channel_masks_and_mismatched_images():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    mask = np.zeros((8, 9), dtype=bool)
    with pytest.raises(ValueError, match='height and width'):
        warp_face_background(image, mask[:, :, None], mask, face_span=4., erosion_radius=0)
    with pytest.raises(ValueError, match='height and width'):
        composite_face(image, image, mask[:, :, None], feather_radius=0)
    with pytest.raises(ValueError, match='matching shapes'):
        composite_face(image, image[:, :, 0], mask, feather_radius=0)



@pytest.mark.parametrize(('original_radius', 'new_radius'), [(32, 22), (20, 32)])
def test_radial_warp_matches_face_boundaries_and_keeps_outer_anchor(
    original_radius, new_radius,
):
    y, x = np.mgrid[:144, :144]
    image = np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.uint8)
    original = (x - 72) ** 2 + (y - 72) ** 2 <= original_radius ** 2
    inserted = (x - 72) ** 2 + (y - 72) ** 2 <= new_radius ** 2
    warped, displacement, parameters = warp_face_background(
        image, original, inserted, face_span=64., erosion_radius=0,
    )
    assert parameters['center_x'] == parameters['center_y'] == 72.
    assert parameters['min_jacobian'] > 0
    # A destination-boundary pixel must sample the old boundary on the same ray.
    np.testing.assert_array_equal(warped[72, 72 + new_radius, :2], [72 + original_radius, 72])
    np.testing.assert_array_equal(warped[72 + new_radius, 72, :2], [72, 72 + original_radius])
    fixed_radius = max(original_radius, new_radius) + parameters['influence_radius']
    outside = (x - 72) ** 2 + (y - 72) ** 2 >= fixed_radius ** 2
    np.testing.assert_array_equal(warped[outside], image[outside])
    assert not np.any(displacement[outside])


@pytest.mark.parametrize(('old_axes', 'new_axes', 'new_center'), [
    ((30, 36), (19, 26), (80, 80)),
    ((19, 24), (33, 38), (80, 80)),
    ((30, 36), (23, 30), (85, 76)),
])
def test_radial_warp_hides_original_silhouette_under_inserted_face(
    old_axes, new_axes, new_center,
):
    original = np.zeros((160, 160), np.uint8)
    inserted = np.zeros_like(original)
    cv2.ellipse(original, (80, 80), old_axes, 0, 0, 360, 255, -1)
    cv2.ellipse(inserted, new_center, new_axes, 0, 0, 360, 255, -1)
    warped, _, parameters = warp_face_background(
        original, original > 0, inserted > 0, face_span=72., erosion_radius=2,
    )
    assert np.any(warped)
    assert not np.any(warped[inserted == 0])
    assert parameters['min_jacobian'] > 0
    composed, _ = composite_face(
        warped, np.zeros_like(original), inserted > 0, feather_radius=0,
    )
    # Painting over the new face leaves no old-face pixels anywhere in the image.
    assert not np.any(composed)


def test_radial_warp_rejects_shared_center_outside_a_face_mask():
    y, x = np.mgrid[:100, :140]
    original = (x - 35) ** 2 + (y - 50) ** 2 <= 20 ** 2
    inserted = (x - 100) ** 2 + (y - 50) ** 2 <= 20 ** 2
    with pytest.raises(ValueError, match='Shared center'):
        warp_face_background(
            np.zeros((100, 140), np.uint8), original, inserted,
            face_span=40., erosion_radius=1,
        )


def test_radial_warp_rejects_erosion_that_removes_shared_center():
    mask = np.zeros((32, 32), np.uint8)
    cv2.circle(mask, (16, 16), 2, 1, -1)
    with pytest.raises(ValueError, match='Eroded insertion mask'):
        warp_face_background(
            mask, mask, mask, face_span=4., erosion_radius=5,
        )


@pytest.mark.parametrize('concave_destination', [False, True])
def test_radial_warp_contains_face_despite_multiple_boundary_crossings(concave_destination):
    concave = np.zeros((100, 100), np.uint8)
    concave[15:85, 15:85] = 1
    # This notch makes some rays exit and reenter while leaving the center inside.
    concave[54:61, 60:85] = 0
    round_mask = np.zeros_like(concave)
    cv2.circle(round_mask, (50, 50), 25, 1, -1)
    original, inserted = (round_mask, concave) if concave_destination else (concave, round_mask)
    warped, _, parameters = warp_face_background(
        original * 255, original, inserted, face_span=70., erosion_radius=2,
    )
    assert parameters['min_jacobian'] > 0
    assert np.any(warped)
    # The solid old face fits under the replacement in both directions.
    # Bilinear filtering can leave fractional coverage at a grazing boundary,
    # so measure the opaque foreground rather than requiring zero filter tails.
    assert not np.any(warped[inserted == 0] == 255)


def test_reconstruction_feathering_does_not_change_warp_or_dilate_original(monkeypatch):
    from morphalo.nodes.preprocess.utils import face_reconstruction_geometry as geometry

    y, x = np.mgrid[:96, :96]
    original = (x - 48) ** 2 + (y - 48) ** 2 <= 30 ** 2
    inserted = (x - 48) ** 2 + (y - 48) ** 2 <= 22 ** 2
    target = np.stack((x, y, x + y), axis=-1).astype(np.uint8)
    face = {'mask': inserted, 'color': np.full_like(target, 180)}

    def unexpected_dilation(*args, **kwargs):
        raise AssertionError('Original mask must be passed without dilation')

    monkeypatch.setattr(cv2, 'dilate', unexpected_dilation)
    monkeypatch.setattr(geometry, 'FEATHER_RADIUS_FRACTION', 0.)
    narrow, first = geometry.compose_reconstructed_face(target, original, face)
    monkeypatch.setattr(geometry, 'FEATHER_RADIUS_FRACTION', 0.1)
    wide, second = geometry.compose_reconstructed_face(target, original, face)

    assert first['erosion_radius'] == second['erosion_radius'] == 2
    assert first['warp'] == second['warp']
    assert first['feather_radius'] == 2
    assert second['feather_radius'] == 6
    assert not np.array_equal(narrow, wide)


@pytest.mark.parametrize('deep_notch', [False, True])
def test_radial_intersections_select_outer_original_and_inner_destination(deep_notch):
    from morphalo.nodes.preprocess.utils.face_compositing import _radial_contour_radii

    contour = np.array([
        [-10, -10], [10, -10], [10, 3], [9, 3],
        [9, 4], [10, 4], [10, 10], [-10, 10],
    ], dtype=np.float64)
    if deep_notch:
        contour[[3, 4], 0] = 2
        contour[[2, 3], 1] = 0.5
    angle = np.array([np.arctan(0.42)])
    outer = _radial_contour_radii(contour, np.zeros(2), angle, outermost=True)
    inner = _radial_contour_radii(contour, np.zeros(2), angle, outermost=False)
    np.testing.assert_allclose(outer, [10 / np.cos(angle[0])])
    np.testing.assert_allclose(inner, [(2 if deep_notch else 9) / np.cos(angle[0])])


def test_radial_intersections_still_reject_a_ray_without_hits():
    from morphalo.nodes.preprocess.utils.face_compositing import _radial_contour_radii

    contour = np.array([[-3, -1], [-2, -1], [-2, 1], [-3, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match='without a positive boundary intersection'):
        _radial_contour_radii(contour, np.zeros(2), np.array([0.]), outermost=True)



def test_radial_intersections_include_tangent_vertex_hits():
    from morphalo.nodes.preprocess.utils.face_compositing import _radial_contour_radii

    contour = np.array([
        [-10, -10], [10, -10], [10, 2], [8, 0],
        [10, 4], [10, 10], [-10, 10],
    ], dtype=np.float64)
    # The ray touches the notch vertex at x=8, without crossing it, before
    # exiting the outer side at x=10. Both are boundary intersections.
    inner = _radial_contour_radii(contour, np.zeros(2), np.array([0.]), outermost=False)
    outer = _radial_contour_radii(contour, np.zeros(2), np.array([0.]), outermost=True)
    np.testing.assert_allclose(inner, [8.])
    np.testing.assert_allclose(outer, [10.])


def test_radial_intersections_include_collinear_boundary_endpoints():
    from morphalo.nodes.preprocess.utils.face_compositing import _radial_contour_radii

    contour = np.array([
        [-10, -10], [10, -10], [10, 0], [8, 0],
        [8, 2], [10, 2], [10, 10], [-10, 10],
    ], dtype=np.float64)
    inner = _radial_contour_radii(contour, np.zeros(2), np.array([0.]), outermost=False)
    outer = _radial_contour_radii(contour, np.zeros(2), np.array([0.]), outermost=True)
    np.testing.assert_allclose(inner, [8.])
    np.testing.assert_allclose(outer, [10.])
