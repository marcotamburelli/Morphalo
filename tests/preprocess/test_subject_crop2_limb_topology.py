import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from morphalo.nodes.preprocess.crop_debug import (
    CropDebugEdgeOverlay,
    CropDebugMaskOverlay,
    LimbTopologyDebugImage,
    write_limb_topology_debug_directory,
)
from morphalo.nodes.preprocess.subject_crop2 import (
    HumanParseContext,
    LimbCoreExtraction,
    LimbOverlapSubtraction,
    _apply_limb_overlap_subtraction,
    _build_limb_partition_context,
    _build_limb_workspace_relief_from_semantic_edges,
    _expand_regions_over_partition_barriers,
    _extend_limb_regions_proximally,
    _extract_internal_limb_core_regions,
    _find_edge_tube_intersections,
    _merge_limb_core_and_extension,
    _subtract_limb_core_overlaps,
)
from morphalo.nodes.preprocess.utils.canny_edge_ops import (
    CannyEdgeMap,
    CannyPairExtensionDebug,
    build_canny_edge_map,
    build_canny_partition_barriers,
    build_region_mosaic,
    connect_limb_edge_endpoints,
    internal_region_labels,
    prolong_endpoint,
    thin_binary_edges,
)
from morphalo.nodes.preprocess.utils.mask_geometry import (
    LimbCropGeometry,
    ResolvedLandmark,
    build_arm_crop_geometries,
)
from morphalo.nodes.preprocess.utils.sapiens2_seg import SAPIENS2_CLASSES


def _ctx(shape=(12, 12)):
    height, width = shape
    return HumanParseContext(
        node_id='test',
        cfg=SimpleNamespace(
            target='arms',
            expansion=1.0,
            shape_cleanup={},
        ),
        img_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        width=width,
        height=height,
        person_bbox=(0, 0, width, height),
        pose=None,
        segments=np.zeros(shape, dtype=np.int64),
        runtime_dtype=None,
    )


def _geometry(
    *,
    shape=(12, 12),
    crop_bbox=None,
    support=None,
    semantic_region=None,
    tube=None,
    skeleton=None,
    synthetic=None,
    cap=None,
    quadrant=None,
    side='anatomical-left',
    tube_radius=2.0,
):
    height, width = shape
    crop_bbox = crop_bbox or (0, 0, width, height)
    support = (
        np.zeros(shape, dtype=bool)
        if support is None
        else support.astype(bool)
    )
    semantic_region = (
        support.copy()
        if semantic_region is None
        else semantic_region.astype(bool)
    )
    tube = support.copy() if tube is None else tube.astype(bool)
    skeleton = np.zeros(shape, dtype=bool) if skeleton is None else skeleton
    synthetic = np.zeros(shape, dtype=bool) if synthetic is None else synthetic
    cap = np.zeros(shape, dtype=bool) if cap is None else cap
    quadrant = np.zeros(shape, dtype=bool) if quadrant is None else quadrant

    return LimbCropGeometry(
        side=side,
        proximal_name='shoulder',
        middle_name='elbow',
        distal_name='wrist',
        proximal_point=np.asarray([3.0, 3.0], dtype=np.float32),
        middle_point=np.asarray([5.0, 5.0], dtype=np.float32),
        distal_point=np.asarray([8.0, 8.0], dtype=np.float32),
        base_bbox=crop_bbox,
        crop_bbox=crop_bbox,
        semantic_region_mask=semantic_region.copy(),
        semantic_support_mask=support.copy(),
        tube_mask=tube.copy(),
        skeleton_mask=skeleton.astype(bool).copy(),
        synthetic_barrier_mask=synthetic.astype(bool).copy(),
        proximal_circle_mask=cap.astype(bool).copy(),
        proximal_quadrant_mask=quadrant.astype(bool).copy(),
        tube_radius=float(tube_radius),
    )


def _core(
    *,
    ctx,
    region_mask,
    expanded_mask=None,
    support=None,
    cap=None,
    quadrant=None,
):
    shape = (ctx.height, ctx.width)
    expanded_mask = region_mask if expanded_mask is None else expanded_mask
    support = np.ones(shape, dtype=bool) if support is None else support
    cap = np.zeros(shape, dtype=bool) if cap is None else cap
    quadrant = np.zeros(shape, dtype=bool) if quadrant is None else quadrant
    empty = np.zeros(shape, dtype=bool)
    edge_map = build_canny_edge_map(
        local_rgb=ctx.img_rgb,
        local_mask=empty,
        limb_chain_length=10.0,
    )
    barrier_set = build_canny_partition_barriers(
        visual_barrier_mask=empty,
    )
    mosaic = build_region_mosaic(
        workspace_mask=empty,
        barrier_mask=empty,
    )
    return LimbCoreExtraction(
        region_mosaic=mosaic,
        selected_region_labels=frozenset(),
        region_mask=region_mask.astype(bool).copy(),
        expanded_mask=expanded_mask.astype(bool).copy(),
        barrier_set=barrier_set,
        edge_map=edge_map,
        refined_edge_mask=empty.copy(),
        prior_mask=empty.copy(),
        workspace_mask=empty.copy(),
        workspace_relief_mask=empty.copy(),
        proximal_cap_mask=cap.astype(bool).copy(),
        proximal_half_plane_mask=quadrant.astype(bool).copy(),
        semantic_support_mask=support.astype(bool).copy(),
    )


def test_barrier_restoration_is_not_generic_dilation():
    region = np.zeros((7, 7), dtype=bool)
    region[3, 2] = True
    barrier = np.zeros_like(region)
    barrier[3, 3] = True
    allowed = np.ones_like(region)

    out = _expand_regions_over_partition_barriers(
        region_mask=region,
        barrier_mask=barrier,
        allowed_mask=allowed,
        dilation_radius=1,
    )

    assert out[3, 2]
    assert out[3, 3]
    assert not out[3, 1]
    assert not out[2, 2]
    assert int(np.count_nonzero(out)) == 2


def test_barrier_restoration_respects_allowed_mask():
    region = np.zeros((5, 5), dtype=bool)
    region[2, 1] = True
    barrier = np.zeros_like(region)
    barrier[2, 2] = True
    allowed = np.ones_like(region)
    allowed[2, 2] = False

    out = _expand_regions_over_partition_barriers(
        region_mask=region,
        barrier_mask=barrier,
        allowed_mask=allowed,
        dilation_radius=1,
    )

    assert out[2, 1]
    assert not out[2, 2]


def test_barrier_restoration_recovers_two_pixel_barrier():
    region = np.zeros((7, 8), dtype=bool)
    region[3, 1] = True
    barrier = np.zeros_like(region)
    barrier[3, 2:4] = True
    allowed = np.ones_like(region)

    out = _expand_regions_over_partition_barriers(
        region_mask=region,
        barrier_mask=barrier,
        allowed_mask=allowed,
        dilation_radius=2,
    )

    assert out[3, 1]
    assert out[3, 2]
    assert out[3, 3]
    assert not out[3, 4]
    assert int(np.count_nonzero(out)) == 3


def test_barrier_restoration_recovers_three_pixel_barrier():
    region = np.zeros((7, 9), dtype=bool)
    region[3, 1] = True
    barrier = np.zeros_like(region)
    barrier[3, 2:5] = True
    allowed = np.ones_like(region)

    out = _expand_regions_over_partition_barriers(
        region_mask=region,
        barrier_mask=barrier,
        allowed_mask=allowed,
        dilation_radius=3,
    )

    assert out[3, 1]
    assert out[3, 2]
    assert out[3, 3]
    assert out[3, 4]
    assert not out[3, 5]
    assert int(np.count_nonzero(out)) == 4


def test_barrier_restoration_output_dilation_respects_allowed_mask():
    region = np.zeros((5, 5), dtype=bool)
    region[2, 2] = True
    barrier = np.zeros_like(region)
    allowed = np.zeros_like(region)
    allowed[1:4, 1:4] = True
    allowed[2, 3] = False

    out = _expand_regions_over_partition_barriers(
        region_mask=region,
        barrier_mask=barrier,
        allowed_mask=allowed,
        dilation_radius=1,
        output_dilation_radius=1,
    )

    assert out[2, 2]
    assert out[1, 2]
    assert out[2, 1]
    assert out[3, 2]
    assert not out[2, 3]
    assert not out[0, 2]


def test_region_boundary_detection_matches_eight_connectivity():
    workspace = np.ones((3, 3), dtype=bool)
    workspace[0, 0] = False
    barriers = workspace.copy()
    barriers[1, 1] = False

    mosaic = build_region_mosaic(
        workspace_mask=workspace,
        barrier_mask=barriers,
    )

    assert mosaic.boundary_labels == frozenset({1})
    assert internal_region_labels(mosaic) == frozenset()


def test_thin_binary_edges_reduces_thick_connected_edge():
    edge = np.zeros((9, 9), dtype=bool)
    edge[2:7, 3:6] = True

    thinned = thin_binary_edges(edge)

    assert np.any(thinned)
    assert np.all(thinned <= edge)
    assert int(np.count_nonzero(thinned)) < int(np.count_nonzero(edge))


def test_edge_tube_intersections_accept_outside_boundary_contacts():
    tube = np.zeros((8, 8), dtype=bool)
    tube[2:6, 2:6] = True
    edge = np.zeros_like(tube)
    edge[1, 2:6] = True

    intersections = _find_edge_tube_intersections(
        edge_mask=edge,
        tube_mask=tube,
    )

    assert set(intersections.representative_points.values()) == {
        (1, 2),
        (1, 5),
    }


def test_workspace_relief_adds_semantic_bulge_outside_tube():
    semantic_region = np.zeros((18, 18), dtype=bool)
    semantic_region[3:14, 4:16] = True

    tube = np.zeros_like(semantic_region)
    tube[5:12, 5:12] = True
    edge = np.zeros_like(semantic_region)

    edge[6, 12:15] = True
    edge[7:10, 14] = True
    edge[10, 12:15] = True

    relief = _build_limb_workspace_relief_from_semantic_edges(
        semantic_region_mask=semantic_region,
        tube_mask=tube,
        base_workspace_mask=tube,
        skeleton_label_edge_mask=edge,
        tube_radius=20.0,
    )

    assert np.any(relief)
    assert not np.any(relief & tube)
    assert np.all(relief <= semantic_region)
    assert relief[8, 13]


def test_workspace_relief_rejects_patch_near_opposite_skeleton():
    semantic_region = np.zeros((18, 18), dtype=bool)
    semantic_region[3:14, 4:16] = True

    tube = np.zeros_like(semantic_region)
    tube[5:12, 5:12] = True
    edge = np.zeros_like(semantic_region)

    edge[6, 12:15] = True
    edge[7:10, 14] = True
    edge[10, 12:15] = True

    opposite_exclusion = np.zeros_like(semantic_region)
    opposite_exclusion[8, 13] = True
    rejected_components = np.zeros_like(semantic_region)

    relief = _build_limb_workspace_relief_from_semantic_edges(
        semantic_region_mask=semantic_region,
        tube_mask=tube,
        base_workspace_mask=tube,
        skeleton_label_edge_mask=edge,
        tube_radius=20.0,
        opposite_skeleton_exclusion_mask=opposite_exclusion,
        rejected_component_accumulator=rejected_components,
    )

    assert not np.any(relief)
    assert rejected_components[8, 13]


def test_workspace_relief_keeps_clean_components_near_opposite_skeleton():
    semantic_region = np.zeros((24, 24), dtype=bool)
    semantic_region[3:20, 4:18] = True

    tube = np.zeros_like(semantic_region)
    tube[5:18, 5:12] = True
    edge = np.zeros_like(semantic_region)

    edge[6, 12:15] = True
    edge[7:10, 14] = True
    edge[10, 12:15] = True

    edge[13, 12:15] = True
    edge[14:17, 14] = True
    edge[17, 12:15] = True

    opposite_exclusion = np.zeros_like(semantic_region)
    opposite_exclusion[8, 13] = True

    relief = _build_limb_workspace_relief_from_semantic_edges(
        semantic_region_mask=semantic_region,
        tube_mask=tube,
        base_workspace_mask=tube,
        skeleton_label_edge_mask=edge,
        tube_radius=20.0,
        opposite_skeleton_exclusion_mask=opposite_exclusion,
    )

    assert not relief[8, 13]
    assert relief[15, 13]


def test_workspace_relief_components_are_clipped_to_label_before_rejection():
    semantic_region = np.zeros((18, 18), dtype=bool)
    semantic_region[6:12, 12:15] = True
    semantic_region[8:10, 12:14] = False

    tube = np.zeros_like(semantic_region)
    tube[5:12, 5:12] = True
    edge = np.zeros_like(semantic_region)

    edge[6, 12:15] = True
    edge[7:10, 14] = True
    edge[10, 12:15] = True

    opposite_exclusion = np.zeros_like(semantic_region)
    opposite_exclusion[6, 13] = True
    rejected_components = np.zeros_like(semantic_region)

    relief = _build_limb_workspace_relief_from_semantic_edges(
        semantic_region_mask=semantic_region,
        tube_mask=tube,
        base_workspace_mask=tube,
        skeleton_label_edge_mask=edge,
        tube_radius=20.0,
        opposite_skeleton_exclusion_mask=opposite_exclusion,
        rejected_component_accumulator=rejected_components,
    )

    assert not np.any(relief & ~semantic_region)
    assert np.any(rejected_components)
    assert not np.any(rejected_components & ~semantic_region)


def test_workspace_relief_clips_path_too_far_from_tube():
    semantic_region = np.zeros((24, 24), dtype=bool)
    semantic_region[3:18, 4:22] = True

    tube = np.zeros_like(semantic_region)
    tube[5:12, 5:12] = True
    edge = np.zeros_like(semantic_region)

    edge[6, 12:21] = True
    edge[7:10, 20] = True
    edge[10, 12:21] = True

    relief = _build_limb_workspace_relief_from_semantic_edges(
        semantic_region_mask=semantic_region,
        tube_mask=tube,
        base_workspace_mask=tube,
        skeleton_label_edge_mask=edge,
        tube_radius=10.0,
    )

    tube_distance = cv2.distanceTransform(
        (~tube).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )

    assert np.any(relief)
    assert float(np.max(tube_distance[relief])) <= 3.5


def test_partition_context_re_admits_relief_into_core_prior():
    shape = (18, 18)
    semantic_region = np.zeros(shape, dtype=bool)
    tube = np.zeros(shape, dtype=bool)
    tube[5:12, 5:12] = True
    semantic_region |= tube
    semantic_region[6:11, 12:15] = True
    support = semantic_region & tube
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[6:11, 8] = True

    ctx = _ctx(shape)
    ctx.segments[semantic_region] = 8

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
            skeleton=skeleton,
            tube_radius=20.0,
        ),
    )

    assert np.any(partition.workspace_relief_mask)
    assert partition.core_prior_mask[8, 13]
    assert partition.semantic_support_mask[8, 13]
    assert not support[8, 13]


def test_workspace_relief_is_clipped_to_each_touched_label():
    shape = (18, 18)
    tube = np.zeros(shape, dtype=bool)
    tube[5:12, 5:12] = True

    left_label = np.zeros(shape, dtype=bool)
    left_label |= tube
    left_label[6:11, 12:15] = True

    right_label = np.zeros(shape, dtype=bool)
    right_label[6:11, 15:17] = True

    semantic_region = left_label | right_label
    support = semantic_region & tube
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[6:11, 8] = True

    ctx = _ctx(shape)
    ctx.segments[left_label] = 8
    ctx.segments[right_label] = 9

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
            skeleton=skeleton,
            tube_radius=20.0,
        ),
    )

    assert partition.core_prior_mask[8, 13]
    assert not partition.core_prior_mask[8, 16]


def test_partition_context_traces_label_boundary_before_localizing_relief():
    shape = (18, 18)
    semantic_region = np.zeros(shape, dtype=bool)
    tube = np.zeros(shape, dtype=bool)
    tube[5:12, 5:12] = True
    semantic_region |= tube
    semantic_region[6:11, 12:15] = True

    crop_bbox = (5, 7, 15, 10)
    support = semantic_region & tube
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[7:10, 8] = True

    ctx = _ctx(shape)
    ctx.segments[semantic_region] = 8

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            crop_bbox=crop_bbox,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
            skeleton=skeleton,
            tube_radius=20.0,
        ),
    )

    assert np.any(partition.workspace_relief_mask)
    assert partition.core_prior_mask[1, 8]
    assert not support[crop_bbox[1] + 1, crop_bbox[0] + 8]


def test_build_limb_partition_context_crops_existing_geometry():
    shape = (10, 10)
    ctx = _ctx(shape)
    support = np.zeros(shape, dtype=bool)
    support[2:8, 2:8] = True
    tube = np.zeros(shape, dtype=bool)
    tube[4:9, 4:9] = True
    cap = np.zeros(shape, dtype=bool)
    cap[2:5, 2:5] = True
    quadrant = np.zeros(shape, dtype=bool)
    quadrant[1:4, 1:4] = True
    synthetic = np.zeros(shape, dtype=bool)
    synthetic[2, 2:8] = True
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[4:7, 4] = True
    ctx.segments[support] = 8

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            crop_bbox=(2, 2, 9, 9),
            support=support,
            tube=tube,
            skeleton=skeleton,
            synthetic=synthetic,
            cap=cap,
            quadrant=quadrant,
        ),
    )

    assert partition.core_prior_mask.shape == (7, 7)
    expected_base_workspace = (
        tube[2:9, 2:9]
        | cap[2:9, 2:9]
    )
    expected_core_workspace = (
        expected_base_workspace
        | partition.workspace_relief_mask
    )
    expected_core_prior = support[2:9, 2:9] & expected_core_workspace
    assert np.array_equal(
        partition.core_prior_mask,
        expected_core_prior,
    )
    assert np.array_equal(
        partition.core_workspace_mask,
        expected_core_workspace,
    )
    assert np.all(partition.workspace_relief_mask <= support[2:9, 2:9])
    assert not np.any(partition.workspace_relief_mask & expected_base_workspace)
    assert np.any(partition.semantic_barrier_mask)
    assert np.all(partition.semantic_barrier_mask <= expected_core_workspace)
    assert np.array_equal(
        partition.skeleton_label_support_masks[0],
        support[2:9, 2:9],
    )
    assert len(partition.skeleton_label_support_masks) == 1
    assert np.any(partition.skeleton_label_edge_mask)
    assert np.all(
        partition.skeleton_label_edge_mask <= expected_core_workspace)
    assert np.array_equal(partition.proximal_cap_mask, cap[2:9, 2:9])
    assert np.array_equal(
        partition.proximal_half_plane_mask,
        quadrant[2:9, 2:9],
    )
    assert np.allclose(partition.proximal_point, [1.0, 1.0])


def test_semantic_barrier_uses_unclipped_semantic_region_boundary():
    shape = (10, 10)
    semantic_region = np.zeros(shape, dtype=bool)
    semantic_region[1:9, 1:9] = True
    tube = np.zeros(shape, dtype=bool)
    tube[3:7, 3:7] = True
    support = semantic_region & tube

    partition = _build_limb_partition_context(
        _ctx(shape),
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
        ),
    )

    assert np.array_equal(partition.core_prior_mask, support)
    assert np.array_equal(partition.core_workspace_mask, tube)
    assert not np.any(partition.semantic_barrier_mask)


def test_skeleton_label_edge_uses_unclipped_semantic_region_boundary():
    shape = (10, 10)
    semantic_region = np.zeros(shape, dtype=bool)
    semantic_region[1:9, 1:9] = True
    tube = np.zeros(shape, dtype=bool)
    tube[3:7, 3:7] = True
    support = semantic_region & tube
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[4:6, 4] = True
    ctx = _ctx(shape)
    ctx.segments[semantic_region] = 8

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
            skeleton=skeleton,
        ),
    )

    assert np.array_equal(partition.core_prior_mask, support)
    assert np.array_equal(partition.core_workspace_mask, tube)
    assert not np.any(partition.skeleton_label_edge_mask)


def test_skeleton_label_support_uses_labels_crossed_in_semantic_region():
    shape = (12, 12)
    semantic_region = np.zeros(shape, dtype=bool)
    semantic_region[2:10, 2:10] = True
    tube = np.zeros(shape, dtype=bool)
    tube[4:8, 4:8] = True
    support = semantic_region & tube
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[4:8, 4] = True
    skeleton[4:8, 8] = True

    ctx = _ctx(shape)
    ctx.segments[support] = 8
    crossed_extra_label = semantic_region & ~support
    ctx.segments[crossed_extra_label] = 9

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
            skeleton=skeleton,
        ),
    )

    assert len(partition.skeleton_label_support_masks) == 2
    assert any(
        np.array_equal(label_mask, crossed_extra_label)
        for label_mask in partition.skeleton_label_support_masks
    )


def test_skeleton_label_support_groups_upper_and_lower_limb_labels():
    shape = (14, 14)
    upper_arm = np.zeros(shape, dtype=bool)
    upper_arm[2:7, 3:10] = True
    lower_arm = np.zeros(shape, dtype=bool)
    lower_arm[7:12, 3:10] = True
    semantic_region = upper_arm | lower_arm

    tube = np.zeros(shape, dtype=bool)
    tube[3:6, 4:9] = True
    support = semantic_region & tube
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[3:6, 6] = True

    ctx = _ctx(shape)
    ctx.segments[upper_arm] = SAPIENS2_CLASSES['left-upper-arm']
    ctx.segments[lower_arm] = SAPIENS2_CLASSES['left-lower-arm']

    partition = _build_limb_partition_context(
        ctx,
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            semantic_region=semantic_region,
            tube=tube,
            skeleton=skeleton,
            side='anatomical-left',
        ),
    )

    assert len(partition.skeleton_label_support_masks) == 1
    assert np.array_equal(
        partition.skeleton_label_support_masks[0],
        semantic_region,
    )
    assert not partition.skeleton_label_edge_wide_mask[6, 6]
    assert not partition.skeleton_label_edge_wide_mask[7, 6]


def test_arm_tube_radius_uses_pseudo_3d_shoulder_width_for_bounds():
    shape = (120, 120)
    semantic = np.ones(shape, dtype=bool)
    landmarks = {
        'left_shoulder': np.asarray([50.0, 30.0], dtype=np.float32),
        'right_shoulder': np.asarray([70.0, 30.0], dtype=np.float32),
        'left_elbow': np.asarray([50.0, 80.0], dtype=np.float32),
        'left_wrist': np.asarray([50.0, 110.0], dtype=np.float32),
        'right_elbow': np.asarray([70.0, 80.0], dtype=np.float32),
        'right_wrist': np.asarray([70.0, 110.0], dtype=np.float32),
        'nose': np.asarray([60.0, 10.0], dtype=np.float32),
    }
    landmark_depths = {
        name: np.asarray([point[0], point[1], 0.0], dtype=np.float32)
        for name, point in landmarks.items()
    }
    landmark_depths['right_shoulder'] = np.asarray(
        [70.0, 30.0, 100.0],
        dtype=np.float32,
    )

    def resolve(name):
        point = landmarks.get(name)
        if point is None:
            return None
        return ResolvedLandmark(
            xy=point,
            xyz_px=landmark_depths.get(name),
        )

    geometries = build_arm_crop_geometries(
        resolve_landmark=resolve,
        image_shape=shape,
        semantic_region=semantic,
        which='anatomical-left',
    )

    assert len(geometries) == 1
    # The projected shoulder width is only 20 px, which would clamp the radius
    # near 14 px. The pseudo-3D width includes the depth separation and therefore
    # lets the first-segment guardrail become the limiting factor instead.
    assert geometries[0].tube_radius > 30.0


def test_arm_tube_radius_probes_outward_along_shoulder_elbow_band():
    shape = (120, 120)
    semantic = np.zeros(shape, dtype=bool)
    semantic[25:40, 45:52] = True
    semantic[45:75, 25:52] = True
    semantic[75:115, 45:52] = True

    landmarks = {
        'left_shoulder': np.asarray([50.0, 30.0], dtype=np.float32),
        'right_shoulder': np.asarray([115.0, 30.0], dtype=np.float32),
        'left_elbow': np.asarray([50.0, 80.0], dtype=np.float32),
        'left_wrist': np.asarray([50.0, 110.0], dtype=np.float32),
        'nose': np.asarray([65.0, 10.0], dtype=np.float32),
    }

    def resolve(name):
        point = landmarks.get(name)
        if point is None:
            return None
        return ResolvedLandmark(xy=point)

    geometries = build_arm_crop_geometries(
        resolve_landmark=resolve,
        image_shape=shape,
        semantic_region=semantic,
        which='anatomical-left',
    )

    assert len(geometries) == 1
    # At the shoulder the semantic support extends only a few pixels outward.
    # Farther down the shoulder-elbow band it widens substantially, and the
    # single-radius tube should be large enough to contain that widest support.
    assert geometries[0].tube_radius > 18.0


def test_limb_core_edge_extension_uses_workspace_domain(monkeypatch):
    shape = (10, 10)
    support = np.zeros(shape, dtype=bool)
    support[3:7, 3:7] = True
    tube = np.zeros(shape, dtype=bool)
    tube[1:9, 1:9] = True
    cap = np.zeros(shape, dtype=bool)
    cap[2:5, 2:5] = True
    synthetic = np.zeros(shape, dtype=bool)
    synthetic[2, 3] = True
    synthetic[7, 6] = True

    expected_workspace = tube
    expected_prior = support & tube
    expected_synthetic_stop = synthetic & (tube | cap)
    captured: dict[str, np.ndarray] = {}

    def fake_build_canny_edge_map(*, local_rgb, local_mask, limb_chain_length):
        edges = np.zeros_like(local_mask, dtype=bool)
        edges[4, 3] = True
        edges[4, 6] = True
        captured['canny_domain'] = local_mask.copy()
        return CannyEdgeMap(
            raw_mask=edges.copy(),
            cleaned_mask=edges.copy(),
            processing_domain=local_mask.copy(),
            scale=100.0,
        )

    def fake_bridge_consistent_edge_endpoints(edges, *, allowed_domain, **kwargs):
        captured['bridge_domain'] = allowed_domain.copy()
        return edges

    def fake_connect_limb_edge_endpoints(edges, *, allowed_domain, **kwargs):
        captured['extension_domain'] = allowed_domain.copy()
        captured['synthetic_stop_mask'] = kwargs['synthetic_stop_mask'].copy()
        return edges, CannyPairExtensionDebug(
            endpoint_marks=[],
            accepted_segments=[],
            accepted_extension_mask=np.zeros_like(edges, dtype=bool),
            accepted_endpoint_mask=np.zeros_like(edges, dtype=bool),
            rejection_marks=[],
        )

    monkeypatch.setattr(
        'morphalo.nodes.preprocess.subject_crop2.build_canny_edge_map',
        fake_build_canny_edge_map,
    )
    monkeypatch.setattr(
        'morphalo.nodes.preprocess.subject_crop2.'
        'bridge_consistent_edge_endpoints',
        fake_bridge_consistent_edge_endpoints,
    )
    monkeypatch.setattr(
        'morphalo.nodes.preprocess.subject_crop2.connect_limb_edge_endpoints',
        fake_connect_limb_edge_endpoints,
    )

    core = _extract_internal_limb_core_regions(
        _ctx(shape),
        limb_geometry=_geometry(
            shape=shape,
            support=support,
            tube=tube,
            synthetic=synthetic,
            cap=cap,
        ),
    )

    assert np.array_equal(captured['canny_domain'], expected_prior)
    assert np.array_equal(captured['bridge_domain'], expected_workspace)
    assert np.array_equal(captured['extension_domain'], expected_workspace)
    assert np.array_equal(
        captured['synthetic_stop_mask'],
        expected_synthetic_stop,
    )
    assert np.all(core.region_mask <= expected_prior)
    assert np.all(core.expanded_mask <= expected_prior)


def test_expanded_mask_does_not_create_proximal_continuation():
    ctx = _ctx()
    support = np.zeros((12, 12), dtype=bool)
    support[1:10, 1:10] = True
    cap = np.zeros_like(support)
    cap[4:6, 4:6] = True
    quadrant = np.zeros_like(support)
    quadrant[5:9, 5:9] = True
    region = np.zeros_like(support)
    expanded = np.zeros_like(support)
    expanded[5, 5] = True

    extension = _extend_limb_regions_proximally(
        ctx,
        limb_geometry=_geometry(
            support=support,
            tube=support,
            cap=cap,
            quadrant=quadrant,
        ),
        core=_core(
            ctx=ctx,
            region_mask=region,
            expanded_mask=expanded,
            support=support,
            cap=cap,
            quadrant=quadrant,
        ),
    )

    assert not np.any(extension.region_mask)
    assert not np.any(extension.expanded_mask)


def test_proximal_extension_uses_region_continuity_and_excludes_core():
    ctx = _ctx()
    support = np.zeros((12, 12), dtype=bool)
    support[1:10, 1:10] = True
    cap = np.zeros_like(support)
    cap[4:6, 4:6] = True
    quadrant = np.zeros_like(support)
    quadrant[5:9, 5:9] = True
    region = np.zeros_like(support)
    region[5, 5] = True

    extension = _extend_limb_regions_proximally(
        ctx,
        limb_geometry=_geometry(
            support=support,
            tube=support,
            cap=cap,
            quadrant=quadrant,
        ),
        core=_core(
            ctx=ctx,
            region_mask=region,
            expanded_mask=region,
            support=support,
            cap=cap,
            quadrant=quadrant,
        ),
    )

    assert not extension.region_mask[5, 5]
    assert extension.region_mask[6, 6]
    assert np.any(extension.expanded_mask)


def test_core_overlap_subtraction_updates_region_and_expanded_masks(monkeypatch):
    ctx = _ctx()
    left_region = np.zeros((12, 12), dtype=bool)
    left_region[3:6, 3:6] = True
    left_expanded = left_region.copy()
    left_expanded[6, 6] = True
    right_region = np.zeros((12, 12), dtype=bool)
    right_region[4:7, 4:7] = True
    subtract = np.zeros((12, 12), dtype=bool)
    subtract[4, 4] = True
    subtract[6, 6] = True

    def fake_subtraction(*args, **kwargs):
        return LimbOverlapSubtraction(
            subtraction_mask=subtract,
            changed=True,
            compared_segment_pairs=1,
            accepted_segment_pairs=1,
        )

    monkeypatch.setattr(
        'morphalo.nodes.preprocess.subject_crop2.'
        '_build_front_other_limb_subtraction',
        fake_subtraction,
    )

    cores = {
        'anatomical-left': _core(
            ctx=ctx,
            region_mask=left_region,
            expanded_mask=left_expanded,
        ),
        'anatomical-right': _core(ctx=ctx, region_mask=right_region),
    }
    geometries = {
        'anatomical-left': _geometry(side='anatomical-left'),
        'anatomical-right': _geometry(side='anatomical-right'),
    }

    cleaned = _subtract_limb_core_overlaps(
        ctx,
        geometries_by_side=geometries,
        cores_by_side=cores,
    )

    assert not cleaned['anatomical-left'].region_mask[4, 4]
    assert not cleaned['anatomical-left'].expanded_mask[4, 4]
    assert not cleaned['anatomical-left'].expanded_mask[6, 6]
    assert cleaned['anatomical-left'].region_mosaic is cores[
        'anatomical-left'
    ].region_mosaic


def test_apply_limb_overlap_subtraction_does_not_mutate_input():
    mask = np.ones((4, 4), dtype=bool)
    subtraction_mask = np.zeros((4, 4), dtype=bool)
    subtraction_mask[1, 1] = True
    original = mask.copy()

    out = _apply_limb_overlap_subtraction(
        mask=mask,
        subtraction=LimbOverlapSubtraction(
            subtraction_mask=subtraction_mask,
            changed=True,
            compared_segment_pairs=1,
            accepted_segment_pairs=1,
        ),
    )

    assert np.array_equal(mask, original)
    assert not out[1, 1]
    assert out[0, 0]


def test_merge_limb_core_and_extension_is_plain_union():
    ctx = _ctx()
    core_mask = np.zeros((12, 12), dtype=bool)
    core_mask[2, 2] = True
    extension_mask = np.zeros((12, 12), dtype=bool)
    extension_mask[8, 8] = True
    core = _core(ctx=ctx, region_mask=core_mask)
    extension = replace(
        core,
        region_mask=extension_mask,
        expanded_mask=extension_mask,
    )

    out = _merge_limb_core_and_extension(
        core=core,
        extension=extension,
    )

    assert out[2, 2]
    assert out[8, 8]
    assert int(np.count_nonzero(out)) == 2


def test_limb_topology_debug_directory_is_self_contained(tmp_path):
    img_rgb = np.full((8, 8, 3), 180, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    edge = np.zeros((8, 8), dtype=bool)
    edge[4, 1:7] = True

    out_path = tmp_path / 'crop.png'
    overview_path = tmp_path / 'crop_debug_bbox.png'
    Image.fromarray(img_rgb).save(overview_path)

    debug_dir = write_limb_topology_debug_directory(
        img_rgb=img_rgb,
        out_path=out_path,
        overview_path=overview_path,
        images=[
            LimbTopologyDebugImage(
                filename='arm_left_core_edges.png',
                title='arm anatomical-left core edges',
                mask_overlays=[
                    CropDebugMaskOverlay('mask', mask, (255, 0, 0), 0.4),
                ],
                edge_overlays=[
                    CropDebugEdgeOverlay('edge', edge, (0, 255, 0)),
                ],
            ),
        ],
    )

    assert debug_dir == tmp_path / 'crop_debug'
    assert (debug_dir / '00_overview.png').is_file()
    assert (debug_dir / '01_arm_left_core_edges.png').is_file()
    assert not overview_path.exists()

    with (debug_dir / 'debug.json').open(encoding='utf-8') as handle:
        manifest = json.load(handle)

    assert manifest['overview'] == '00_overview.png'
    assert manifest['images'] == [
        {'filename': '00_overview.png', 'title': 'overview'},
        {
            'filename': '01_arm_left_core_edges.png',
            'title': 'arm anatomical-left core edges',
        },
    ]


def test_connect_limb_edge_endpoints_rejects_cross_skeleton_bridge():
    edges = np.zeros((24, 24), dtype=bool)
    edges[3:15, 5] = True
    edges[3:15, 18] = True

    skeleton = np.zeros_like(edges)
    skeleton[1:22, 12] = True
    allowed = np.ones_like(edges)

    connected, debug = connect_limb_edge_endpoints(
        edges,
        limb_skeleton_mask=skeleton,
        allowed_domain=allowed,
        anatomical_segments=[
            (
                np.asarray([12.0, 1.0], dtype=np.float32),
                np.asarray([12.0, 22.0], dtype=np.float32),
            ),
        ],
        min_contour_parallelism=0.45,
    )

    assert not np.any(connected[:, 12])
    assert np.all(connected[:, 5])
    assert np.all(connected[:, 18])
    assert all(
        segment.color == (255, 128, 0)
        for segment in debug.accepted_segments
    )


def test_connect_limb_edge_endpoints_rejects_misaligned_terminal_tangents():
    edges = np.zeros((14, 14), dtype=bool)
    edges[1:4, 3] = True
    edges[1:4, 10] = True

    skeleton = np.zeros_like(edges)
    skeleton[7, :] = True
    allowed = np.ones_like(edges)

    connected, debug = connect_limb_edge_endpoints(
        edges,
        limb_skeleton_mask=skeleton,
        allowed_domain=allowed,
        anatomical_segments=[
            (
                np.asarray([0.0, 7.0], dtype=np.float32),
                np.asarray([13.0, 7.0], dtype=np.float32),
            ),
        ],
        min_contour_parallelism=0.80,
        ignore_intersection_radius=1,
        intersection_blocker_dilation_radius=0,
    )

    assert not any(
        segment.color == (0, 128, 255)
        for segment in debug.accepted_segments
    )
    assert any(
        color == (0, 215, 255)
        for _, _, color in debug.rejection_marks
    )
    assert np.all(connected[1:4, 3])
    assert np.all(connected[1:4, 10])


def test_prolong_endpoint_extends_to_workspace_boundary():
    domain = np.ones((7, 7), dtype=bool)
    stop_mask = np.zeros_like(domain)

    extension, target_xy = prolong_endpoint(
        branch_pixels=[(3, 3), (3, 4)],
        allowed_domain=domain,
        stop_mask=stop_mask,
    )

    assert target_xy == (0, 3)
    assert np.array_equal(
        extension[3],
        np.asarray([True, True, True, True, False, False, False]),
    )


def test_prolong_endpoint_stops_at_existing_barrier():
    domain = np.ones((7, 7), dtype=bool)
    stop_mask = np.zeros_like(domain)
    stop_mask[3, 5] = True

    extension, target_xy = prolong_endpoint(
        branch_pixels=[(3, 2), (3, 1)],
        allowed_domain=domain,
        stop_mask=stop_mask,
    )

    assert target_xy == (5, 3)
    assert np.array_equal(
        extension[3],
        np.asarray([False, False, True, True, True, True, False]),
    )


def test_connect_limb_edge_endpoints_rejects_projection_through_skeleton():
    edges = np.zeros((11, 15), dtype=bool)
    edges[5, 5:9] = True
    edges[4:7, 8] = True

    skeleton = np.zeros_like(edges)
    skeleton[5, 2] = True
    allowed = np.ones_like(edges)

    connected, debug = connect_limb_edge_endpoints(
        edges,
        limb_skeleton_mask=skeleton,
        allowed_domain=allowed,
        anatomical_segments=[
            (
                np.asarray([0.0, 8.0], dtype=np.float32),
                np.asarray([14.0, 8.0], dtype=np.float32),
            ),
        ],
        min_contour_parallelism=0.80,
        ignore_intersection_radius=1,
        intersection_blocker_dilation_radius=0,
    )

    assert np.array_equal(connected, edges)
    assert not debug.accepted_segments
    assert any(
        color == (180, 0, 255)
        for _, _, color in debug.rejection_marks
    )


def test_connect_limb_edge_endpoints_keeps_nontrivial_short_branches():
    edges = np.zeros((13, 13), dtype=bool)
    edges[6, 4:9] = True

    skeleton = np.zeros_like(edges)
    skeleton[6, :] = True
    allowed = np.ones_like(edges)

    connected, debug = connect_limb_edge_endpoints(
        edges,
        limb_skeleton_mask=skeleton,
        allowed_domain=allowed,
        anatomical_segments=[
            (
                np.asarray([0.0, 6.0], dtype=np.float32),
                np.asarray([12.0, 6.0], dtype=np.float32),
            ),
        ],
        min_contour_parallelism=0.80,
    )

    candidate_marks = [
        mark
        for mark in debug.endpoint_marks
        if mark.color == (255, 0, 255)
    ]
    short_marks = [
        mark
        for mark in debug.endpoint_marks
        if mark.color == (0, 0, 255)
    ]

    assert len(candidate_marks) == 2
    assert short_marks == []
    assert np.array_equal(connected, edges)


def test_connect_limb_edge_endpoints_rejects_one_step_spurs():
    edges = np.zeros((7, 7), dtype=bool)
    edges[3, 3:5] = True

    skeleton = np.zeros_like(edges)
    skeleton[3, :] = True
    allowed = np.ones_like(edges)

    connected, debug = connect_limb_edge_endpoints(
        edges,
        limb_skeleton_mask=skeleton,
        allowed_domain=allowed,
        anatomical_segments=[
            (
                np.asarray([0.0, 3.0], dtype=np.float32),
                np.asarray([6.0, 3.0], dtype=np.float32),
            ),
        ],
        min_contour_parallelism=0.80,
    )

    candidate_marks = [
        mark
        for mark in debug.endpoint_marks
        if mark.color == (255, 0, 255)
    ]
    short_marks = [
        mark
        for mark in debug.endpoint_marks
        if mark.color == (0, 0, 255)
    ]

    assert candidate_marks == []
    assert len(short_marks) == 2
    assert np.array_equal(connected, edges)
