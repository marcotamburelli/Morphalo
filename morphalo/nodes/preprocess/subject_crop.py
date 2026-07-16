import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import (get_mediapipe_face_landmarker,
                                   get_mediapipe_hand_landmarker,
                                   get_mediapipe_pose_landmarker, get_sam,
                                   get_yolo)
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.crop_debug import write_crop_debug_overlay
from morphalo.nodes.preprocess.utils import (CropModeSpec, SizeExpr,
                                             expand_bbox_toward_ratio,
                                             parse_crop_mode,
                                             positive_points_for_sam,
                                             read_shape_cleanup_config,
                                             validate_size_expr)
from morphalo.nodes.preprocess.utils.geometry import (
    clip_mask_to_bbox, expand_clip_bbox, expand_clip_bbox_by_size_expr,
    tight_mask_bbox, union_bboxes_xyxy)
from morphalo.nodes.preprocess.utils.mask_ops import (
    bridge_consistent_edge_endpoints, cleanup_shape_mask, cleanup_shape_mask_by_parts, extend_consistent_edge_endpoints, invert_mask_inside_box,
    labeled_points_inside_mask, prepare_output_mask,
    reachable_components_from_labeled_points,
    remove_small_components_unless_touching)
from morphalo.nodes.preprocess.utils.mask_selection import \
    reference_mask_coverage
from morphalo.nodes.preprocess.utils.sam import (
    SamMaskCandidate, build_sam_candidates_from_raw_masks, predict_sam_mask,
    select_best_guided_sam_mask)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.chromatic_segmentation import \
    split_segment_by_chromatic_runs
from morphalo.nodes.vision.face_region import (face_bbox_xyxy_from_landmarks,
                                               mp_face_landmarks)
from morphalo.nodes.vision.human import (ArmRegionGeometry, SamRegion,
                                         arm_regions_from_landmarks,
                                         arm_segments_from_landmarks,
                                         crop_head_area_from_pose,
                                         foot_sam_region_with_prompt_bbox,
                                         foot_sam_regions_from_landmarks,
                                         hands_bbox_xyxy_from_landmarks,
                                         hands_mask_from_landmarks,
                                         mp_hand_landmarks_full,
                                         mp_pose_landmarks_xy,
                                         resolve_person_bbox_xyxy,
                                         square_head_bbox_from_face_bbox)

# Internal geometry constants.
#
# FACE_SEARCH_AREA_EXPANSION defines how generously the pose-derived head area is cropped
# before running the Face Landmarker.
FACE_SEARCH_AREA_EXPANSION = 1.6
FOOT_LEG_PROBE_COVERAGE_THRESHOLD = 0.55
FOOT_CHROMATIC_LAB_DISTANCE_THRESHOLD = 14.0
FOOT_CHROMATIC_MIN_SEGMENT_LEN_PX = 3.0
ARM_CHROMATIC_LAB_DISTANCE_THRESHOLD = 14.0
ARM_CHROMATIC_MIN_SEGMENT_LEN_PX = 4.0

# ``ARM_CANNY_L_*`` controls edge detection on the LAB luminance channel.
# Lower values increase sensitivity to weak brightness boundaries such as
# low-contrast skin, fabric, shadows, and soft silhouette transitions.
ARM_CANNY_L_LOW_THRESHOLD = 30
ARM_CANNY_L_HIGH_THRESHOLD = 100
#
# ``ARM_CANNY_AB_*`` controls edge detection on the LAB chromatic channels.
# These channels recover boundaries between similarly bright but differently
# colored regions. They are usually noisier than luminance, so their thresholds
# should generally remain stricter than extremely permissive values.
ARM_CANNY_AB_LOW_THRESHOLD = 35
ARM_CANNY_AB_HIGH_THRESHOLD = 100
#
# Local contrast enhancement
# --------------------------
# CLAHE is applied only to the LAB luminance channel before Canny.
#
# ``ARM_CANNY_CLAHE_CLIP_LIMIT`` controls how aggressively local contrast is
# amplified. Higher values reveal weaker boundaries but also emphasize fabric
# texture, hair strands, wrinkles, and image noise.
ARM_CANNY_CLAHE_CLIP_LIMIT = 1.5
#
# ``ARM_CANNY_CLAHE_TILE_SIZE`` is the width and height of each CLAHE tile.
# Smaller tiles make enhancement more local and aggressive; larger tiles make
# it smoother and closer to global contrast enhancement.
ARM_CANNY_CLAHE_TILE_SIZE = 8
#
# Spatial edge filtering
# ----------------------
# Luminance edges are accepted up to the complete arm tube radius.
# Chromatic edges are more easily contaminated by texture, so they are accepted
# only within this fraction of the maximum skeleton distance.
#
# Lower values reduce chromatic noise but may miss the external boundary of
# wide sleeves. Higher values preserve more clothing boundaries but may retain
# unrelated color transitions.
ARM_CANNY_AB_DISTANCE_RATIO = 0.75
#
# Barrier morphology
# ------------------
# ``ARM_CANNY_EDGE_CLOSE_RADIUS`` applies blind morphological closing to the
# complete edge mask. It is currently disabled because endpoint-aware bridging
# is more selective and less likely to connect unrelated texture fragments.
ARM_CANNY_EDGE_CLOSE_RADIUS = 0
#
# ``ARM_CANNY_EDGE_DILATE_RADIUS`` thickens accepted visual barriers before
# flood-fill. Larger values close tiny leaks more effectively, but may consume
# too much valid arm area or merge nearby contours.
ARM_CANNY_EDGE_DILATE_RADIUS = 1
#
# ``ARM_CANNY_MIN_EDGE_COMPONENT_AREA`` removes isolated visual-edge components
# smaller than this number of pixels, unless they touch an explicit synthetic
# shoulder or wrist barrier. Increasing it suppresses more texture noise but
# may discard legitimate short contour fragments.
ARM_CANNY_MIN_EDGE_COMPONENT_AREA = 12
#
# Endpoint-aware contour bridging
# -------------------------------
# Interrupted contours are reconnected only when two skeletonized edge
# endpoints are spatially close and their local outgoing directions are
# geometrically compatible.
#
# ``ARM_CANNY_MAX_BRIDGE_GAP`` is the maximum endpoint distance, in pixels,
# eligible for reconnection. Larger values close longer missing contour
# sections but increase the risk of connecting unrelated edges.
ARM_CANNY_MAX_BRIDGE_GAP = 15.0
#
# ``ARM_CANNY_BRIDGE_TANGENT_RADIUS`` is the local skeleton graph distance used
# to estimate the outgoing tangent at each endpoint. Larger values give a more
# stable direction on smooth contours, while smaller values follow local curves
# more closely but are more sensitive to pixel noise.
ARM_CANNY_BRIDGE_TANGENT_RADIUS = 6
#
# ``ARM_CANNY_BRIDGE_MIN_FACING_ALIGNMENT`` is the minimum cosine alignment
# between each endpoint's outgoing tangent and the direction toward the other
# endpoint. Values closer to 1.0 require the endpoints to face each other more
# directly.
ARM_CANNY_BRIDGE_MIN_FACING_ALIGNMENT = 0.70
#
# ``ARM_CANNY_BRIDGE_MIN_PARALLELISM`` is the minimum absolute cosine
# similarity between the two endpoint tangents. Values closer to 1.0 require
# the interrupted contour fragments to be more nearly collinear.
ARM_CANNY_BRIDGE_MIN_PARALLELISM = 0.65
#
# ``ARM_CANNY_BRIDGE_MIN_ALLOWED_FRACTION`` is the minimum fraction of bridge
# pixels that must lie inside the permitted bridge domain, defined by the arm
# skeleton distance and the local SAM neighborhood.
ARM_CANNY_BRIDGE_MIN_ALLOWED_FRACTION = 0.90
#
# Flood-fill edge restoration
# ---------------------------
# Thick barriers temporarily consume pixels that may belong to the arm.
# ``ARM_CANNY_RESTORE_EDGE_RADIUS`` controls how far from the reachable
# flood-filled region barrier pixels may be restored, provided they were also
# selected by SAM.
ARM_CANNY_RESTORE_EDGE_RADIUS = 2
#
# Refined-mask validation
# -----------------------
# The refined result is compared with the original SAM mask before acceptance.
#
# ``ARM_CANNY_ACCEPT_MIN_AREA_RATIO`` rejects refinements that retain too little
# of the SAM candidate, which usually indicates a leak, an over-aggressive
# barrier, or poor flood-fill connectivity.
ARM_CANNY_ACCEPT_MIN_AREA_RATIO = 0.15
#
# ``ARM_CANNY_ACCEPT_MAX_AREA_RATIO`` rejects refinements that grow excessively
# relative to SAM. Values above 1.0 allow limited growth caused by controlled
# restoration logic, although the current pipeline normally keeps the final
# candidate inside the SAM mask.
ARM_CANNY_ACCEPT_MAX_AREA_RATIO = 1.25

# Endpoint extension
# ------------------
# Endpoint bridging requires compatible contour fragments on both sides of a
# gap. Endpoint extension handles the complementary case where one reliable
# contour terminates and no matching fragment is visible beyond the gap.
#
# An endpoint is eligible only when it belongs to a sufficiently long
# skeletonized edge component and its outgoing tangent is approximately
# parallel to the local arm skeleton. The extension follows the edge tangent,
# not the skeleton itself, and remains constrained to the permitted bridge
# domain.
#
# ``ARM_CANNY_EXTENSION_MIN_COMPONENT_LENGTH`` is the minimum skeletonized
# component length, in pixels, required before one of its endpoints may be
# extended. Larger values reduce the chance of extending short texture fragments
# or noise, but may reject legitimate weak arm contours.
ARM_CANNY_EXTENSION_MIN_COMPONENT_LENGTH = 20
#
# ``ARM_CANNY_EXTENSION_MAX_COMPONENT_LENGTH`` optionally excludes components
# longer than the configured value. ``None`` disables the upper bound. Long
# contours are usually the most reliable candidates, so an upper limit is
# normally unnecessary.
ARM_CANNY_EXTENSION_MAX_COMPONENT_LENGTH = None
#
# ``ARM_CANNY_EXTENSION_MAX_LENGTH`` is the maximum number of pixels that may be
# synthesized beyond an eligible endpoint. Increasing it can bridge longer
# low-contrast regions, but also increases the risk of inventing a barrier where
# no real arm boundary exists.
ARM_CANNY_EXTENSION_MAX_LENGTH = 12.0
#
# ``ARM_CANNY_EXTENSION_TANGENT_RADIUS`` is the local edge-skeleton graph
# distance used to estimate the endpoint's outgoing tangent. Larger values give
# more stable directions on smooth contours; smaller values follow sharp local
# curvature more closely but are more sensitive to pixel noise.
ARM_CANNY_EXTENSION_TANGENT_RADIUS = 6
#
# ``ARM_CANNY_EXTENSION_SKELETON_TANGENT_RADIUS`` is the radius, in pixels,
# around the nearest arm-skeleton point used to derive candidate local skeleton
# directions. Near the elbow this neighborhood may expose both upper-arm and
# forearm directions, allowing the endpoint to match the locally relevant one.
ARM_CANNY_EXTENSION_SKELETON_TANGENT_RADIUS = 10
#
# ``ARM_CANNY_EXTENSION_MIN_SKELETON_PARALLELISM`` is the minimum absolute
# cosine similarity between the outgoing edge tangent and at least one nearby
# arm-skeleton direction. Values closer to 1.0 require the contour to be more
# nearly longitudinal with respect to the arm.
ARM_CANNY_EXTENSION_MIN_SKELETON_PARALLELISM = 0.75
#
# ``ARM_CANNY_EXTENSION_SNAP_RADIUS`` is the local search radius used while
# extending an endpoint. When another edge is encountered inside this radius,
# the extension snaps to the best forward-aligned pixel and stops.
ARM_CANNY_EXTENSION_SNAP_RADIUS = 2

# Flood-fill margin recovery
# --------------------------
# Flood-fill always stops on the inner side of the visual barriers. Even after
# restoring the reachable-side barrier pixels, the accepted region remains
# slightly contracted because the detected contour itself occupies a finite
# thickness.
#
# Expand the accepted flood-filled region by the estimated barrier thickness
# plus one additional safety pixel. The result is clipped to the original SAM
# mask, so this expansion can only recover pixels that already belonged to the
# selected SAM candidate.
#
# The expansion radius is intentionally derived from the configured barrier
# dilation to keep both stages geometrically consistent.
ARM_CANNY_FLOOD_EXPANSION_RADIUS = (
    ARM_CANNY_EDGE_DILATE_RADIUS + 1
)


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    yolo_model: Optional[str]
    sam_model: Optional[str]
    mode: str
    crop_mode: Optional[CropModeSpec]
    conf: float
    box_margin: SizeExpr
    prompt_expansion: float
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: Optional[str]
    hand_landmarker_task: Optional[str]
    pose_landmarker_task: str
    smoothing_radius: int
    min_landmark_fraction: Optional[float]
    shape_cleanup: dict[str, Any]


@dataclass(frozen=True)
class SegmentationContext:
    """
    Runtime dependencies shared by target-specific segmentation pipelines.

    The context keeps target pipelines focused on segmentation decisions instead
    of carrying long parameter lists through every helper. It intentionally
    contains only immutable shared inputs and model handles; target-specific
    products such as face landmarks, hand results, or foot regions belong in
    ``TargetSegmentationResult``.

    Parameters
    ----------
    node_id : str
        Current node identifier, used for contextual errors.
    cfg : Config
        Validated node configuration.
    img_rgb : np.ndarray
        Full-frame RGB source image.
    pose_xy : np.ndarray
        Full-frame MediaPipe pose landmarks in pixel coordinates.
    width : int
        Source image width in pixels.
    height : int
        Source image height in pixels.
    sam_processor : Any
        SAM-compatible processor returned by ``get_sam``.
    sam_model : Any
        SAM-compatible model returned by ``get_sam``.
    """
    node_id: str
    cfg: Config
    img_rgb: np.ndarray
    pose_xy: np.ndarray
    width: int
    height: int
    sam_processor: Any
    sam_model: Any


@dataclass(frozen=True)
class TargetSegmentationResult:
    """
    Result produced by one target-specific segmentation pipeline.

    The final output stage should not need to know how a target was segmented.
    Each pipeline therefore returns the selected mask, the geometry used for
    component selection/debug, and any optional target-specific debug data in a
    single object.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame boolean mask for the selected target.
    target_bbox : tuple[int, int, int, int]
        End-exclusive bbox describing the target-local geometry. This bbox is
        used by connected-component selection and debug rendering.
    prompt_bbox : tuple[int, int, int, int]
        End-exclusive bbox used as the main SAM prompt domain. For multi-region
        targets, this is the union of all region prompt boxes.
    shape_part_masks : np.ndarray | list[np.ndarray] | None, optional
        Logical part masks used to run shape cleanup independently before
        unioning multi-part targets such as hands or feet.
    face_xy : np.ndarray | None, optional
        Full-frame face landmarks used only by debug rendering.
    hands_result : Any, optional
        MediaPipe hand-landmarker result used only by debug rendering.
    sam_regions : list[SamRegion] | None, optional
        Final SAM prompt regions, including expanded prompt boxes and any
        target-specific prompt enrichment, used by debug rendering.
    prompt_bbox_label : str, default='person'
        Human-readable label for the prompt bbox in debug overlays.
    preserve_all_components : bool, default=False
        If true, the shared component-selection stage keeps every foreground
        component before cleanup. This is separate from ``shape_part_masks``:
        component preservation decides what survives before cleanup, while part
        masks decide how cleanup is applied.
    debug_region_masks : list[np.ndarray] | None, optional
        Full-frame target-local prior masks used only by debug rendering.
    debug_edge_masks : list[np.ndarray] | None, optional
        Full-frame edge/barrier masks used only by debug rendering.
    """
    mask: np.ndarray
    target_bbox: tuple[int, int, int, int]
    prompt_bbox: tuple[int, int, int, int]
    shape_part_masks: Optional[np.ndarray | list[np.ndarray]] = None
    face_xy: Optional[np.ndarray] = None
    hands_result: Any = None
    sam_regions: Optional[list[SamRegion]] = None
    prompt_bbox_label: str = 'person'
    preserve_all_components: bool = False
    debug_region_masks: Optional[list[np.ndarray]] = None
    debug_edge_masks: Optional[list[np.ndarray]] = None


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'bf16')))

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': Invalid mode={mode!r}")

    # It should apply only when mode='default'
    if mode == 'default':
        crop_mode = parse_crop_mode(
            params.get('crop_mode', 'trim'),
            node_id=node_id,
        )
    else:
        crop_mode = None

    conf = float(params.get('conf', 0.35))
    box_margin = params.get('box_margin', '12%')
    validate_size_expr(box_margin)

    prompt_expansion = float(params.get('prompt_expansion', 0.0))
    if prompt_expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid prompt_expansion={prompt_expansion!r} "
            '(expected >= 0)'
        )

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))
    min_landmark_fraction = params.get('min_landmark_fraction', 0.8)
    if min_landmark_fraction is not None:
        min_landmark_fraction = float(min_landmark_fraction)
        if not (0.0 <= min_landmark_fraction <= 1.0):
            raise ValueError(
                f"'{node_id}': invalid min_landmark_fraction={min_landmark_fraction!r} "
                '(expected a float in [0, 1] or None)'
            )

    shape_cleanup = read_shape_cleanup_config(
        params.get('postprocess', None),
        node_id=node_id,
    )

    target = str(params.get('target', 'person'))
    if target not in (
        'person',
        'head',
        'hands',
        'left-hand',
        'right-hand',
        'arms',
        'left-arm',
        'right-arm',
        'feet',
        'left-foot',
        'right-foot',
    ):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'person', "
            "'head', 'hands', 'left-hand', 'right-hand', 'arms', "
            "'left-arm', 'right-arm', 'feet', 'left-foot' or "
            "'right-foot'; use FaceCrop for face, eye, and eyebrow targets)"
        )

    sam_model = str(model.get('sam_model', 'facebook/sam-vit-large'))

    expansion = float(params.get('expansion', 1.0))
    if expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} (expected >= 0)"
        )

    save_debug = bool(debug.get('save_debug', False))

    face_landmarker_task = None
    hand_landmarker_task = None
    yolo_model = None

    pose_landmarker_task = model.get('pose_landmarker_task')
    if pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': Missing 'model.pose_landmarker_task' (MediaPipe .task path)"
        )
    pose_landmarker_task = str(pose_landmarker_task)

    if target in (
        'person',
        'head',
        'hands',
        'left-hand',
        'right-hand',
        'arms',
        'left-arm',
        'right-arm',
        'feet',
        'left-foot',
        'right-foot',
    ):
        yolo_model = str(model.get('yolo_model', 'yolov8n.pt'))

    if target == 'head':
        # Face landmarks are required for head targets.
        face_landmarker_task = model.get('face_landmarker_task')
        if not face_landmarker_task:
            raise ValueError(
                f"'{node_id}': target={target!r} requires model.face_landmarker_task (MediaPipe .task path)"
            )
        face_landmarker_task = str(face_landmarker_task)

    if target in ('hands', 'left-hand', 'right-hand'):
        hand_landmarker_task = model.get('hand_landmarker_task')
        if not hand_landmarker_task:
            raise ValueError(
                f"'{node_id}': target={target!r} requires model.hand_landmarker_task (MediaPipe .task path)"
            )
        hand_landmarker_task = str(hand_landmarker_task)

    return Config(
        device=device,
        dtype=dtype,
        yolo_model=yolo_model,
        sam_model=sam_model,
        mode=mode,
        crop_mode=crop_mode,
        conf=conf,
        box_margin=box_margin,
        prompt_expansion=prompt_expansion,
        save_debug=save_debug,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        target=target,
        expansion=expansion,
        face_landmarker_task=face_landmarker_task,
        hand_landmarker_task=hand_landmarker_task,
        pose_landmarker_task=pose_landmarker_task,
        smoothing_radius=smoothing_radius,
        min_landmark_fraction=min_landmark_fraction,
        shape_cleanup=shape_cleanup,
    )


def _build_person_sam_candidates(
    *,
    img_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pose_xy: np.ndarray,
    processor: Any,
    model: Any,
    device: str,
    node_id: str,
    use_landmark_prompts: bool = True,
) -> list[SamMaskCandidate]:
    """
    Build person SAM candidates using strict and complete prompting.

    This helper is used only for ``target='person'`` and intentionally combines
    two prompt strategies:

    strict
        Uses the resolved person bbox together with positive pose landmarks.
        This tends to produce cleaner, more subject-specific masks, but may miss
        weakly supported silhouette regions.

    complete
        Uses the resolved person bbox only. This may preserve a fuller
        silhouette, but is more likely to attach nearby background fragments or
        ambiguous objects.

    For each raw SAM mask, the local inverse inside the prompt bbox is also
    added as a candidate. The inverse candidate keeps the same SAM score because
    it is derived from the same raw prediction, but it is marked with
    ``inverted=True`` so the selector can prefer non-inverted candidates when
    all stronger criteria are tied.

    Returns
    -------
    list[SamMaskCandidate]
        Full-frame boolean candidate masks with selection metadata.
    """
    candidates: list[SamMaskCandidate] = []

    prompt_runs: list[tuple[str, Optional[list[list[float]]],
                            Optional[list[int]]]] = []

    if use_landmark_prompts:
        point_coords, point_labels = positive_points_for_sam(
            xy=pose_xy,
            bbox=bbox,
        )

        if point_coords is not None and point_labels is not None:
            prompt_runs.append(('strict', point_coords, point_labels))

    prompt_runs.append(('complete', None, None))

    for source, run_point_coords, run_point_labels in prompt_runs:
        masks, scores = predict_sam_mask(
            img_rgb=img_rgb,
            bbox=bbox,
            processor=processor,
            model=model,
            device=device,
            point_coords=run_point_coords,
            point_labels=run_point_labels,
        )

        candidates.extend(build_sam_candidates_from_raw_masks(
            masks,
            scores,
            bbox=bbox,
            source=source,
            node_id=node_id,
            include_inverted=True,
        ))

    return candidates


def _enrich_foot_sam_regions_with_chromatic_points(
    *,
    img_rgb: np.ndarray,
    pose_xy: np.ndarray,
    foot_regions: list[SamRegion],
) -> list[SamRegion]:
    """
    Rebuild foot SAM regions with image-aware positive prompt points.

    MediaPipe gives us sparse foot anchors: ankle and foot_index. A fixed
    geometric midpoint between them is fragile because it may land on skin,
    sandal, shadow, a gap between straps, or background depending on pose and
    footwear. Instead, this helper samples the actual image along the
    ankle -> foot_index line, splits that line into chromatically coherent
    sub-segments, and adds each retained sub-segment center as a positive SAM
    point.

    The added points are intended to improve recall across visually
    discontinuous parts of the same foot or footwear, such as exposed skin,
    straps, soles, shadows, or separated shoe regions.

    The centers are intentionally used instead of segment boundaries: boundaries
    are exactly where material/color transitions happen and are therefore
    ambiguous prompts.

    This helper returns rebuilt ``SamRegion`` objects. Existing bbox/probe
    geometry is preserved, while ``point_coords`` / ``point_labels`` are
    regenerated through ``foot_sam_region_with_prompt_bbox`` so debug overlays
    and SAM receive the same final prompt.

    Parameters
    ----------
    img_rgb : np.ndarray
        Source RGB image.
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    foot_regions : list[SamRegion]
        Foot prompt regions with base ankle/foot_index prompts.

    Returns
    -------
    list[SamRegion]
        Regions with chromatic positive points appended when any valid
        chromatic segment centers are found. If no enrichment is possible, the
        original region list is returned unchanged.
    """
    additional_positive_points_by_side: dict[str, list[list[float]]] = {}

    for region in foot_regions:
        positive_points = [
            point
            for point, label in zip(
                region.point_coords or [],
                region.point_labels or [],
            )
            if int(label) == 1
        ]

        if len(positive_points) < 2:
            continue

        ankle_point = positive_points[0]
        foot_index_point = positive_points[1]
        foot_axis_len = float(np.linalg.norm(
            np.asarray(foot_index_point, dtype=np.float32)
            - np.asarray(ankle_point, dtype=np.float32)
        ))

        if foot_axis_len < 1.0:
            continue

        chromatic_segments = split_segment_by_chromatic_runs(
            img_rgb,
            (float(ankle_point[0]), float(ankle_point[1])),
            (float(foot_index_point[0]), float(foot_index_point[1])),
            lab_distance_threshold=FOOT_CHROMATIC_LAB_DISTANCE_THRESHOLD,
            min_segment_len_px=FOOT_CHROMATIC_MIN_SEGMENT_LEN_PX,
        )

        additional_positive_points_by_side[region.side] = [
            [float(segment.center_xy[0]), float(segment.center_xy[1])]
            for segment in chromatic_segments
        ]

    if not additional_positive_points_by_side:
        return foot_regions

    return [
        foot_sam_region_with_prompt_bbox(
            region,
            pose_xy,
            prompt_bbox=region.prompt_bbox,
            additional_positive_points_by_side=additional_positive_points_by_side,
        )
        for region in foot_regions
    ]


def _enrich_arm_sam_regions_with_chromatic_points(
    *,
    img_rgb: np.ndarray,
    pose_xy: np.ndarray,
    arm_regions: list[SamRegion],
    prior_masks: list[np.ndarray],
) -> list[SamRegion]:
    """
    Add image-aware chromatic positive prompt points to arm SAM regions.

    The base arm regions are built from pose geometry in ``human.py``. This
    helper samples the RGB image along each arm landmark segment, splits the
    segment into chromatically coherent runs, and appends each retained run
    center as an extra positive SAM point when it lies inside the corresponding
    arm prior. This helps SAM receive hints on visually distinct parts of the
    same arm, such as sleeve, skin, shadow, or fabric transitions.

    Parameters
    ----------
    img_rgb : np.ndarray
        Full-frame RGB source image.
    pose_xy : np.ndarray
        MediaPipe Pose landmarks in full-frame image coordinates.
    arm_regions : list[SamRegion]
        Base arm regions whose prompt points should be enriched.
    prior_masks : list[np.ndarray]
        Full-frame boolean arm prior masks aligned with ``arm_regions``.

    Returns
    -------
    list[SamRegion]
        Rebuilt arm regions with additional positive prompt points appended
        when chromatic samples are available. If no samples are retained, the
        original ``arm_regions`` list is returned unchanged.
    """
    additional_positive_points_by_side: dict[str, list[list[float]]] = {}

    for region, prior_mask in zip(arm_regions, prior_masks):
        points: list[list[float]] = []
        h, w = prior_mask.shape[:2]

        for start, end in arm_segments_from_landmarks(
            pose_xy,
            side=region.side,
        ):
            if float(np.linalg.norm(end - start)) < 4.0:
                continue

            try:
                chromatic_segments = split_segment_by_chromatic_runs(
                    img_rgb,
                    (float(start[0]), float(start[1])),
                    (float(end[0]), float(end[1])),
                    lab_distance_threshold=ARM_CHROMATIC_LAB_DISTANCE_THRESHOLD,
                    min_segment_len_px=ARM_CHROMATIC_MIN_SEGMENT_LEN_PX,
                )
            except ValueError:
                continue

            for segment in chromatic_segments:
                px = int(round(float(segment.center_xy[0])))
                py = int(round(float(segment.center_xy[1])))
                if 0 <= px < w and 0 <= py < h and prior_mask[py, px]:
                    points.append([
                        float(segment.center_xy[0]),
                        float(segment.center_xy[1]),
                    ])

        if points:
            additional_positive_points_by_side[region.side] = points

    if not additional_positive_points_by_side:
        return arm_regions

    enriched_regions: list[SamRegion] = []
    for region in arm_regions:
        point_coords = list(region.point_coords or [])
        point_labels = list(region.point_labels or [])

        for point in additional_positive_points_by_side.get(region.side, []):
            point_coords.append(point)
            point_labels.append(1)

        enriched_regions.append(SamRegion(
            side=region.side,
            base_bbox=region.base_bbox,
            prompt_bbox=region.prompt_bbox,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            probe_point=region.probe_point,
        ))

    return enriched_regions


def _select_best_person_sam_mask(
    candidates: list[SamMaskCandidate],
    *,
    bbox: tuple[int, int, int, int],
    pose_xy: np.ndarray,
    target_norm_area: float = 0.25,
    min_landmark_fraction: Optional[float] = 0.8,
) -> np.ndarray:
    """
    Select the best person mask among SAM candidates.

    The selector assumes that candidates may come from different prompt
    strategies, typically:

    - ``strict``: bbox + pose landmarks;
    - ``complete``: bbox only.

    Before ranking candidates, the selector tries to discard masks that do not
    contain enough stable pose landmarks. This helps reject SAM masks that cover
    nearby objects, props, furniture, or the local inverse/background instead of the
    actual subject.

    This landmark filtering is intentionally conservative and non-fatal: if no
    candidate passes the landmark-consistency check, the selector falls back to the
    full candidate set.

    Selection criteria, in order
    ----------------------------
    1. Landmark-consistency filtering, when possible.
    2. Quantized SAM predicted-IoU score.
    3. Distance from the preferred normalized candidate area.
    4. Non-inverted candidates.
    5. Strict prompt candidates.

    Landmark-consistency filtering
    ------------------------------
    Only a small set of relatively stable body anchors is used, such as nose,
    shoulders, elbows, wrists, hips, and knees. Foot-related landmarks are
    intentionally ignored because they are often occluded, truncated, or confused
    with supports, props, seats, or background objects.

    A candidate is kept if it contains at least a minimum fraction of the valid
    stable landmarks. If no valid landmarks are available, or if all candidates fail
    the check, the original candidate set is used.

    Candidate area normalization
    ----------------------------
    Candidate areas are measured inside the SAM prompt bbox and normalized
    relative to the available candidate set after optional landmark filtering:

    - smallest candidate area -> 0.0
    - largest candidate area -> 1.0

    The default ``target_norm_area=0.25`` intentionally favors conservative
    person masks while still allowing candidates larger than the smallest one.
    This reduces the risk of attaching external background fragments while
    keeping a chance to recover more complete silhouettes when SAM provides a
    good intermediate candidate.

    Parameters
    ----------
    candidates : list[SamMaskCandidate]
        Candidate masks and metadata.

    bbox : tuple[int, int, int, int]
        SAM prompt bbox ``(x1, y1, x2, y2)``.

    pose_xy : np.ndarray
        MediaPipe pose landmarks in full-image coordinates, with invalid points
        encoded as ``(-1, -1)``. A stable subset of these landmarks is used to
        reject masks that do not plausibly cover the selected subject.

    target_norm_area : float, default=0.25
        Preferred normalized candidate area. ``0.0`` favors the smallest
        candidate, ``1.0`` favors the largest candidate, and intermediate values
        favor masks between the two extremes.

    min_landmark_fraction : float | None, default=0.8
        Minimum fraction of usable stable pose landmarks that must be contained
        within a candidate SAM mask for it to be considered landmark-consistent.
        A higher value makes candidate selection stricter and favors masks that
        better align with pose predictions. A lower value makes selection more
        permissive and can improve results for noisy or partially occluded poses.
        If set to 0.0, landmark guidance remains enabled, but a candidate only
        needs to contain at least one usable stable landmark when such landmarks
        are available.

        If ``None``, landmark-consistency filtering is skipped. In the standard
        ``SubjectCrop`` person path, this value is also used upstream to generate
        only bbox-only SAM candidates.

    Returns
    -------
    np.ndarray
        Selected full-frame boolean mask.

    Raises
    ------
    RuntimeError
        If no candidates are available or selection fails.

    ValueError
        If ``target_norm_area`` is outside ``[0, 1]``.
    """
    # Stable body anchors:
    # 0  = nose
    # 11 = left shoulder
    # 12 = right shoulder
    # 13 = left elbow
    # 14 = right elbow
    # 15 = left wrist
    # 16 = right wrist
    # 23 = left hip
    # 24 = right hip
    # 25 = left knee
    # 26 = right knee
    #
    # Wrists and knees are included because they help preserve visible arms and legs
    # without relying on more fragile extremity landmarks.
    #
    # Ankles, heels, and foot tips are intentionally excluded because they are often
    # occluded, outside the actual visible subject, or confused with supports / props.
    safe_pose_idxs = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    # Require a strong majority of usable stable pose landmarks to fall inside the
    # candidate mask.
    #
    # "Usable" means: present in pose_xy, valid, inside image bounds, and included in
    # safe_pose_idxs. If only one stable landmark is usable, that landmark is allowed
    # to be decisive: it is still better than ignoring landmark consistency entirely.
    # The configurable ``min_landmark_fraction`` controls how strict the landmark
    # consistency check is.
    required_points: list[list[float]] = []
    for idx in safe_pose_idxs:
        if idx < 0 or idx >= pose_xy.shape[0]:
            continue

        px, py = pose_xy[idx, :2]
        if px < 0 or py < 0:
            continue

        required_points.append([float(px), float(py)])

    return select_best_guided_sam_mask(
        candidates,
        bbox=bbox,
        required_points=required_points,
        forbidden_points=None,
        min_required_fraction=min_landmark_fraction,
        target_norm_area=target_norm_area,
        preferred_source='strict',
        context='person SAM mask',
    )


def _select_best_foot_sam_mask(
    candidates: list[SamMaskCandidate],
    *,
    bbox: tuple[int, int, int, int],
    point_coords: Optional[list[list[float]]],
    point_labels: Optional[list[int]],
    target_norm_area: float = 0.25,
    preferred_source: Optional[str] = None,
) -> np.ndarray:
    """
    Select a foot SAM mask using the foot prompt points as strong guidance.

    Foot prompts are local and sparse, so SAM's own predicted-IoU score often
    prefers masks that look coherent to the model but are wrong for our crop
    target. This wrapper reuses the generic guided selector with foot-specific
    semantics:

    - candidates containing all valid positive prompt points are preferred as
      a strict filtered pool when at least one such candidate exists;
    - if no candidate satisfies all positive points, selection falls back to
      the full candidate set;
    - negative prompt points are preferred outside the selected candidate;
    - area preference remains conservative inside the local foot prompt bbox.

    Parameters
    ----------
    candidates : list[SamMaskCandidate]
        Raw and optionally inverted SAM candidates for one foot prompt.
    bbox : tuple[int, int, int, int]
        End-exclusive foot prompt bbox passed to SAM.
    point_coords : list[list[float]] or None
        SAM point coordinates used for the foot prompt.
    point_labels : list[int] or None
        SAM labels aligned with ``point_coords``. ``1`` means positive and
        ``0`` means negative.
    target_norm_area : float, default=0.25
        Preferred normalized candidate area inside the foot prompt bbox.
    preferred_source : str or None, optional
        Prompt source to prefer as a final tie-breaker.

    Returns
    -------
    np.ndarray
        Selected full-frame boolean mask.

    Raises
    ------
    RuntimeError
        If no candidates are available or selection fails.
    ValueError
        If ``target_norm_area`` is outside ``[0, 1]``.
    """
    positive_points: list[list[float]] = []
    negative_points: list[list[float]] = []

    # Keep this split local to the foot wrapper: only the foot selector consumes
    # SAM prompt labels directly. The generic selector only needs target-level
    # required/forbidden points and does not need to know SAM's label encoding.
    if point_coords and point_labels:
        for point, label in zip(point_coords, point_labels):
            if int(label) == 1:
                positive_points.append(point)
            else:
                negative_points.append(point)

    return select_best_guided_sam_mask(
        candidates,
        bbox=bbox,
        required_points=positive_points,
        forbidden_points=negative_points,
        min_required_fraction=1.0,
        target_norm_area=target_norm_area,
        preferred_source=preferred_source,
        context='foot SAM mask',
    )


def _select_best_arm_sam_mask(
    candidates: list[SamMaskCandidate],
    *,
    bbox: tuple[int, int, int, int],
    point_coords: Optional[list[list[float]]],
    point_labels: Optional[list[int]],
    preferred_source: Optional[str] = None,
) -> np.ndarray:
    """
    Select an arm SAM candidate using positive/negative prompt consistency.

    Arm prompts can include many positives: anatomical landmarks plus optional
    chromatic samples along the limb. Requiring every positive can be too strict
    when SAM slightly misses a color-run sample, so this selector requires all
    positives for sparse prompts and a strong majority for richer prompts.
    Negative points, when present, are treated as forbidden anchors.

    Parameters
    ----------
    candidates : list[SamMaskCandidate]
        Candidate masks returned by SAM, optionally including local inverses.
    bbox : tuple[int, int, int, int]
        Local bbox used to normalize candidate area during ranking.
    point_coords : list[list[float]] | None
        SAM point coordinates in the same coordinate system as the candidates.
    point_labels : list[int] | None
        Labels aligned with ``point_coords``. ``1`` means positive and ``0``
        means negative.
    preferred_source : str | None, optional
        Candidate source label used as a final tie-breaker.

    Returns
    -------
    np.ndarray
        Selected boolean SAM mask.
    """
    positive_points: list[list[float]] = []
    negative_points: list[list[float]] = []

    if point_coords and point_labels:
        for point, label in zip(point_coords, point_labels):
            if int(label) == 1:
                positive_points.append(point)
            else:
                negative_points.append(point)

    min_fraction = 0.65 if len(positive_points) > 2 else 1.0

    return select_best_guided_sam_mask(
        candidates,
        bbox=bbox,
        required_points=positive_points,
        forbidden_points=negative_points,
        min_required_fraction=min_fraction,
        target_norm_area=0.35,
        preferred_source=preferred_source,
        context='arm SAM mask',
    )


def _refine_arm_mask_with_canny_barriers(
    *,
    local_rgb: np.ndarray,
    local_sam_mask: np.ndarray,
    local_skeleton_mask: np.ndarray,
    local_synthetic_barrier_mask: np.ndarray,
    local_shoulder_circle_mask: np.ndarray,
    local_shoulder_quadrant_mask: np.ndarray,
    tube_radius: float,
    point_coords: list[list[float]],
    point_labels: list[int],
    flood_seed_coords: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a local SAM arm mask using multi-channel Canny barriers and
    seeded flood-fill.

    The SAM mask remains the primary segmentation domain. Edge detection does
    not create a replacement mask from the complete crop. Instead, visual and
    synthetic barriers subdivide the SAM mask, and only regions reachable from
    dedicated arm flood-fill seeds are retained.

    The visual barrier pipeline operates in LAB color space:

    1. compute luminance and chromatic Canny edges from the original local RGB
       crop;
    2. retain luminance and chromatic edges only inside their configured
       distance from the arm skeleton;
    3. reconnect pairs of interrupted edge endpoints when their local outgoing
       tangents are mutually compatible;
    4. extend remaining endpoints belonging to sufficiently long contours when
       their local tangent is approximately parallel to the nearby arm skeleton;
    5. remove small disconnected edge components;
    6. optionally close and dilate the remaining visual edges to create stronger
       flood-fill barriers;
    7. combine the visual barriers with explicit synthetic shoulder and wrist
       barriers;
    8. flood-fill inside the SAM mask from positive arm seed points outside the
       trusted shoulder circle;
    9. restore the reachable-side portion of the visual barriers;
    10. expand the accepted flood-filled region by the estimated barrier thickness,
        while remaining inside the original SAM mask;
    11. restore SAM-selected shoulder support inside:
        - the geometric shoulder circle;
        - the external shoulder quadrant;
    12. validate positive-point coverage and the refined-to-SAM area ratio.

    Endpoint bridging and extension solve different failure modes. Bridging
    reconnects two compatible contour fragments separated by a short gap.
    Extension continues a single reliable contour when Canny loses the opposite
    fragment entirely, for example across a weakly contrasted fabric fold.

    The coarse arm prior used to neutralize the image before SAM is deliberately
    not accepted by this function. It must never be reintroduced after the
    flood-fill stage.

    Parameters
    ----------
    local_rgb : np.ndarray
        Original local RGB crop used for edge detection. This image must not be
        neutralized with the arm prior, because the refinement stage needs the
        real visual boundaries of the arm, clothing, torso, hair, and
        surrounding image content.

    local_sam_mask : np.ndarray
        Boolean local SAM mask selected for the arm. Flood-fill and final mask
        restoration remain constrained to this segmentation domain.

    local_skeleton_mask : np.ndarray
        Boolean local shoulder-to-elbow-to-wrist centerline mask. It is used to
        filter visual edges by distance and to validate whether an endpoint
        extension follows a plausible longitudinal arm direction.

    local_synthetic_barrier_mask : np.ndarray
        Boolean local mask containing only explicit synthetic shoulder and wrist
        barrier lines. These pixels restrict flood-fill and are never restored
        directly as target geometry.

    local_shoulder_circle_mask : np.ndarray
        Boolean local filled circle centered on the shoulder. The circle defines
        a shoulder restoration domain, but only pixels already selected by SAM
        inside it are restored after flood-fill.

    local_shoulder_quadrant_mask : np.ndarray
        Boolean local mask representing the external shoulder quadrant. Only
        pixels already selected by SAM inside this domain are restored.

    tube_radius : float
        Radius used to build the coarse arm tube. It defines the maximum useful
        distance between visual edges and the arm skeleton. Chromatic edges may
        use a stricter fraction of this distance.

    point_coords : list[list[float]]
        Local SAM point coordinates used both for candidate selection and final
        positive-point coverage validation.

    point_labels : list[int]
        Labels aligned with ``point_coords``. Positive prompts use label ``1``
        and negative prompts use label ``0``.

    flood_seed_coords : list[list[float]]
        Positive local points used exclusively as flood-fill seeds. Unlike the
        complete SAM prompt set, these coordinates exclude points inside the
        trusted shoulder circle because shoulder support is restored explicitly
        after flood-fill. They normally include elbow, wrist, and chromatic
        samples along the arm that lie outside the shoulder circle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(refined_mask, barrier_mask)``.

        ``refined_mask`` is the accepted refined local arm mask. The original
        SAM mask is returned when refinement cannot be applied safely or fails
        validation.

        ``barrier_mask`` is the final local union of visual Canny-derived
        barriers and synthetic shoulder/wrist barriers. It is returned for debug
        rendering even when the original SAM mask is retained.

    Notes
    -----
    The refinement deliberately favors arm completeness over pixel-perfect edge
    placement. Flood-fill is used primarily to identify the correct SAM-connected
    arm component rather than to determine the final contour with pixel accuracy.
    A small expansion then recovers the arm margin lost to the visual barriers
    while remaining strictly constrained by the original SAM mask.

    Endpoint extension follows the same philosophy. It is intentionally
    conservative and is limited to sufficiently long observed contours, bounded by
    a maximum synthetic length, constrained to the permitted domain, and accepted
    only when its local tangent is approximately parallel to the nearby arm
    skeleton.
    """
    import cv2

    sam_mask = local_sam_mask.astype(bool)
    skeleton_mask = local_skeleton_mask.astype(bool)
    synthetic_barriers = local_synthetic_barrier_mask.astype(bool)
    shoulder_circle = local_shoulder_circle_mask.astype(bool)
    shoulder_quadrant = local_shoulder_quadrant_mask.astype(bool)

    empty_barriers = np.zeros_like(sam_mask, dtype=bool)

    sam_area = int(np.count_nonzero(sam_mask))
    if sam_area <= 0:
        return sam_mask, empty_barriers

    positive_inside_sam, positive_valid = labeled_points_inside_mask(
        sam_mask,
        point_coords,
        point_labels,
    )

    required_positive_count = (
        positive_valid
        if positive_valid <= 2
        else int(math.ceil(0.65 * positive_valid))
    )

    # Keep the same positive-point tolerance used by arm SAM candidate
    # selection. Sparse prompts remain strict, while richer prompts may miss
    # a minority of chromatic samples without disabling Canny refinement.
    if positive_inside_sam < required_positive_count:
        return sam_mask, empty_barriers

    if not np.any(skeleton_mask):
        return sam_mask, empty_barriers

    # -------------------------------------------------------------
    # Multi-channel Canny edge detection
    # -------------------------------------------------------------
    distance_to_skeleton = cv2.distanceTransform(
        (~skeleton_mask).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    max_edge_distance = max(
        2.0,
        float(tube_radius),
    )

    lab = cv2.cvtColor(
        local_rgb,
        cv2.COLOR_RGB2LAB,
    )

    clahe = cv2.createCLAHE(
        clipLimit=ARM_CANNY_CLAHE_CLIP_LIMIT,
        tileGridSize=(
            ARM_CANNY_CLAHE_TILE_SIZE,
            ARM_CANNY_CLAHE_TILE_SIZE,
        ),
    )

    l_channel = clahe.apply(
        lab[:, :, 0],
    )
    l_channel = cv2.GaussianBlur(
        l_channel,
        (5, 5),
        0,
    )

    a_channel = cv2.GaussianBlur(
        lab[:, :, 1],
        (5, 5),
        0,
    )
    b_channel = cv2.GaussianBlur(
        lab[:, :, 2],
        (5, 5),
        0,
    )

    edges_l = cv2.Canny(
        l_channel,
        ARM_CANNY_L_LOW_THRESHOLD,
        ARM_CANNY_L_HIGH_THRESHOLD,
    ) > 0

    edges_a = cv2.Canny(
        a_channel,
        ARM_CANNY_AB_LOW_THRESHOLD,
        ARM_CANNY_AB_HIGH_THRESHOLD,
    ) > 0

    edges_b = cv2.Canny(
        b_channel,
        ARM_CANNY_AB_LOW_THRESHOLD,
        ARM_CANNY_AB_HIGH_THRESHOLD,
    ) > 0

    l_domain = (
        distance_to_skeleton
        <= max_edge_distance
    )

    ab_domain = (
        distance_to_skeleton
        <= ARM_CANNY_AB_DISTANCE_RATIO * max_edge_distance
    )

    edges_l &= l_domain
    edges_a &= ab_domain
    edges_b &= ab_domain

    canny = (
        edges_l
        | edges_a
        | edges_b
    )

    # Creating barriers far outside SAM cannot help the flood-fill. A small
    # dilation still permits reconnecting borders immediately adjacent to SAM.
    sam_bridge_domain = cv2.dilate(
        sam_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
        iterations=1,
    ) > 0

    bridge_domain = (
        l_domain
        & sam_bridge_domain
    )

    canny = bridge_consistent_edge_endpoints(
        canny,
        allowed_domain=bridge_domain,
        max_gap=ARM_CANNY_MAX_BRIDGE_GAP,
        tangent_radius=ARM_CANNY_BRIDGE_TANGENT_RADIUS,
        min_facing_alignment=ARM_CANNY_BRIDGE_MIN_FACING_ALIGNMENT,
        min_parallelism=ARM_CANNY_BRIDGE_MIN_PARALLELISM,
        min_allowed_fraction=ARM_CANNY_BRIDGE_MIN_ALLOWED_FRACTION,
        bridge_thickness=1,
    )

    canny = extend_consistent_edge_endpoints(
        canny,
        arm_skeleton_mask=skeleton_mask,
        allowed_domain=bridge_domain,
        min_component_length=(
            ARM_CANNY_EXTENSION_MIN_COMPONENT_LENGTH
        ),
        max_component_length=(
            ARM_CANNY_EXTENSION_MAX_COMPONENT_LENGTH
        ),
        max_extension_length=(
            ARM_CANNY_EXTENSION_MAX_LENGTH
        ),
        tangent_radius=(
            ARM_CANNY_EXTENSION_TANGENT_RADIUS
        ),
        skeleton_tangent_radius=(
            ARM_CANNY_EXTENSION_SKELETON_TANGENT_RADIUS
        ),
        min_skeleton_parallelism=(
            ARM_CANNY_EXTENSION_MIN_SKELETON_PARALLELISM
        ),
        snap_radius=(
            ARM_CANNY_EXTENSION_SNAP_RADIUS
        ),
        bridge_thickness=1,
    )
    # -------------------------------------------------------------
    # Strengthen visual edges.
    # -------------------------------------------------------------
    visual_barriers = canny

    if ARM_CANNY_EDGE_CLOSE_RADIUS > 0:
        kernel_size = 2 * ARM_CANNY_EDGE_CLOSE_RADIUS + 1
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        visual_barriers = cv2.morphologyEx(
            visual_barriers.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            close_kernel,
        ) > 0

    if ARM_CANNY_EDGE_DILATE_RADIUS > 0:
        kernel_size = 2 * ARM_CANNY_EDGE_DILATE_RADIUS + 1
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        visual_barriers = cv2.dilate(
            visual_barriers.astype(np.uint8) * 255,
            dilate_kernel,
            iterations=1,
        ) > 0

    # Remove small isolated Canny fragments. Synthetic barriers are not passed
    # through this cleanup because they are explicit trusted closures.
    visual_barriers = remove_small_components_unless_touching(
        visual_barriers,
        reference_mask=synthetic_barriers,
        min_area=ARM_CANNY_MIN_EDGE_COMPONENT_AREA,
    )

    barriers = visual_barriers | synthetic_barriers

    # -------------------------------------------------------------
    # Flood-fill only inside the SAM mask.
    #
    # This is the central distinction from the previous implementation:
    # Canny cuts SAM into reachable regions instead of creating a completely
    # new mask from the entire local crop.
    # -------------------------------------------------------------
    walkable = sam_mask & ~barriers

    reachable = reachable_components_from_labeled_points(
        walkable,
        flood_seed_coords,
        [1] * len(flood_seed_coords),
    )

    if not np.any(reachable):
        return sam_mask, barriers

    # -------------------------------------------------------------
    # Restore the reachable-side half of thickened visual barriers.
    #
    # Flood-fill excludes the complete barrier thickness, so the reachable region
    # ends on the inner edge of each barrier and may become noticeably contracted.
    #
    # Each visual-barrier pixel is assigned to the closest side:
    #
    # - pixels closer to the reachable arm region are restored;
    # - pixels closer to the rejected SAM region remain excluded.
    #
    # This approximately restores the mask up to the estimated contour centerline
    # without dilating through the barrier into another SAM component.
    #
    # Synthetic shoulder and wrist barriers are deliberately excluded from this
    # restoration. They are logical flood-fill closures rather than detected image
    # boundaries.
    # -------------------------------------------------------------
    restored_edge_pixels = np.zeros_like(
        reachable,
        dtype=bool,
    )

    if (
        ARM_CANNY_RESTORE_EDGE_RADIUS > 0
        and np.any(visual_barriers)
    ):
        distance_to_reachable = cv2.distanceTransform(
            (~reachable).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        rejected_region = (
            sam_mask
            & ~reachable
            & ~barriers
        )

        if np.any(rejected_region):
            distance_to_rejected = cv2.distanceTransform(
                (~rejected_region).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
        else:
            distance_to_rejected = np.full(
                reachable.shape,
                np.inf,
                dtype=np.float32,
            )

        restored_edge_pixels = (
            visual_barriers
            & sam_mask
            & (
                distance_to_reachable
                <= float(ARM_CANNY_RESTORE_EDGE_RADIUS)
            )
            & (
                distance_to_reachable
                <= distance_to_rejected
            )
        )

    flood_part = (
        reachable
        | restored_edge_pixels
    )

    # -------------------------------------------------------------
    # Recover the arm margin consumed by flood-fill barriers.
    #
    # Even after restoring the reachable-side barrier pixels, the accepted region
    # still tends to terminate inside the true arm contour because flood-fill
    # cannot cross the visual barriers.
    #
    # Expand the accepted region by a small configurable radius while remaining
    # strictly inside the original SAM mask. This produces a more realistic arm
    # outline without allowing leakage into neighboring regions.
    # -------------------------------------------------------------
    if ARM_CANNY_FLOOD_EXPANSION_RADIUS > 0:
        kernel_size = (
            2 * ARM_CANNY_FLOOD_EXPANSION_RADIUS + 1
        )

        expansion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        flood_part = cv2.dilate(
            flood_part.astype(np.uint8) * 255,
            expansion_kernel,
            iterations=1,
        ) > 0

        flood_part &= sam_mask

    # -------------------------------------------------------------
    # Restore shoulder support selected by SAM.
    #
    # The shoulder circle and external quadrant define the geometric domains
    # where SAM-selected shoulder pixels may be restored after flood-fill.
    # Neither mask may introduce pixels that were rejected by SAM.
    # -------------------------------------------------------------
    shoulder_restore = (
        sam_mask
        & (
            shoulder_circle
            | shoulder_quadrant
        )
    )

    candidate = (
        flood_part
        | shoulder_restore
    )

    candidate &= sam_mask

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------
    positive_inside_candidate, _ = labeled_points_inside_mask(
        candidate,
        point_coords,
        point_labels,
    )

    if positive_inside_candidate < required_positive_count:
        return sam_mask, barriers

    candidate_area = int(np.count_nonzero(candidate))
    if candidate_area <= 0:
        return sam_mask, barriers

    area_ratio = float(candidate_area) / float(max(1, sam_area))

    if area_ratio < ARM_CANNY_ACCEPT_MIN_AREA_RATIO:
        return sam_mask, barriers

    if area_ratio > ARM_CANNY_ACCEPT_MAX_AREA_RATIO:
        return sam_mask, barriers

    return candidate, barriers


def _predict_arm_mask_on_prior_crop(
    ctx: SegmentationContext,
    *,
    arm_geometry: ArmRegionGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SAM on one local arm-prior crop and refine the selected mask with
    Canny and synthetic flood-fill barriers.

    The arm geometry object keeps the different masks conceptually separate:

    - ``prior_mask`` determines which source-image pixels remain visible before
      SAM; pixels outside it are replaced with a neutral color;
    - ``skeleton_mask`` limits relevant Canny edges to the expected arm axis;
    - ``synthetic_barrier_mask`` contains only explicit shoulder and wrist
      flood-fill closures;
    - ``shoulder_circle_mask`` is trusted geometric support restored after
      flood-fill;
    - ``shoulder_quadrant_mask`` restricts which SAM-selected shoulder pixels
      may be restored.

    SAM is run on ``sam_region.prompt_bbox`` using point prompts only. All
    full-frame masks are cropped to that same coordinate frame before
    refinement, then the accepted local result and barrier mask are pasted back
    into full-frame coordinates.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing the source image, SAM handles,
        device, and image dimensions.
    arm_geometry : ArmRegionGeometry
        Geometry, prompts, and masks for one anatomical arm.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mask, edge_mask)`` where both arrays are full-frame boolean masks.
        ``mask`` is the selected and optionally refined arm mask.
        ``edge_mask`` contains the final Canny and synthetic barriers for debug
        rendering.
    """
    region = arm_geometry.sam_region

    # Use the prompt bbox as the actual image domain shown to SAM. The base bbox
    # describes target geometry, while the prompt bbox may include additional
    # context around it.
    x1, y1, x2, y2 = region.prompt_bbox

    local_height = int(y2 - y1)
    local_width = int(x2 - x1)

    if local_width <= 0 or local_height <= 0:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': invalid arm prompt bbox "
            f'{region.prompt_bbox!r}.'
        )

    local_prior = arm_geometry.prior_mask[
        y1:y2,
        x1:x2,
    ].astype(bool)

    local_skeleton_mask = arm_geometry.skeleton_mask[
        y1:y2,
        x1:x2,
    ].astype(bool)

    local_synthetic_barrier_mask = arm_geometry.synthetic_barrier_mask[
        y1:y2,
        x1:x2,
    ].astype(bool)

    local_shoulder_circle_mask = arm_geometry.shoulder_circle_mask[
        y1:y2,
        x1:x2,
    ].astype(bool)

    local_shoulder_quadrant_mask = arm_geometry.shoulder_quadrant_mask[
        y1:y2,
        x1:x2,
    ].astype(bool)

    if not np.any(local_prior):
        return (
            arm_geometry.prior_mask.copy(),
            np.zeros_like(arm_geometry.prior_mask, dtype=bool),
        )

    local_source_rgb = np.ascontiguousarray(
        ctx.img_rgb[y1:y2, x1:x2, :].copy()
    )

    # Show SAM only pixels inside the coarse arm prior. The prior is deliberately
    # generous and is not used as the final mask.
    prior_pixels = local_source_rgb[local_prior]

    if prior_pixels.size == 0:
        return (
            arm_geometry.prior_mask.copy(),
            np.zeros_like(arm_geometry.prior_mask, dtype=bool),
        )

    neutral_color = np.median(
        prior_pixels,
        axis=0,
    ).astype(np.uint8)

    local_sam_rgb = local_source_rgb.copy()
    local_sam_rgb[~local_prior] = neutral_color

    # Convert full-frame prompt points into coordinates relative to the local
    # SAM crop.
    local_points: list[list[float]] = []
    local_labels: list[int] = []

    for point, label in zip(
        region.point_coords or [],
        region.point_labels or [],
    ):
        if len(point) < 2:
            continue

        local_x = float(point[0]) - float(x1)
        local_y = float(point[1]) - float(y1)

        if not (
            0.0 <= local_x < float(local_width)
            and 0.0 <= local_y < float(local_height)
        ):
            continue

        local_points.append([local_x, local_y])
        local_labels.append(int(label))

    if not any(label == 1 for label in local_labels):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': arm region "
            f'{region.side!r} has no valid local positive SAM prompts.'
        )

    masks, scores = predict_sam_mask(
        img_rgb=local_sam_rgb,
        bbox=None,
        processor=ctx.sam_processor,
        model=ctx.sam_model,
        device=ctx.cfg.device,
        point_coords=local_points,
        point_labels=local_labels,
    )

    local_bbox = (
        0,
        0,
        local_width,
        local_height,
    )

    candidates = build_sam_candidates_from_raw_masks(
        masks,
        scores,
        bbox=local_bbox,
        source='arm-prior-crop',
        node_id=ctx.node_id,
        include_inverted=True,
    )

    local_sam_mask = _select_best_arm_sam_mask(
        candidates,
        bbox=local_bbox,
        point_coords=local_points,
        point_labels=local_labels,
        preferred_source='arm-prior-crop',
    ).astype(bool)

    flood_seed_coords: list[list[float]] = []

    for point, label in zip(local_points, local_labels):
        if int(label) != 1:
            continue

        px = int(round(float(point[0])))
        py = int(round(float(point[1])))

        if not (
            0 <= px < local_width
            and 0 <= py < local_height
        ):
            continue

        # Positive points inside the trusted shoulder circle are useful for SAM,
        # but must not seed the arm flood-fill. The shoulder region is restored
        # explicitly after flood-fill.
        if local_shoulder_circle_mask[py, px]:
            continue

        flood_seed_coords.append([
            float(point[0]),
            float(point[1]),
        ])

    if not flood_seed_coords:
        full_mask = np.zeros(
            (ctx.height, ctx.width),
            dtype=bool,
        )
        full_mask[y1:y2, x1:x2] = local_sam_mask

        full_barriers = np.zeros(
            (ctx.height, ctx.width),
            dtype=bool,
        )
        full_barriers[y1:y2, x1:x2] = (
            local_synthetic_barrier_mask
        )

        return full_mask, full_barriers

    local_refined_mask, local_barriers = (
        _refine_arm_mask_with_canny_barriers(
            local_rgb=local_source_rgb,
            local_sam_mask=local_sam_mask,
            local_skeleton_mask=local_skeleton_mask,
            local_synthetic_barrier_mask=local_synthetic_barrier_mask,
            local_shoulder_circle_mask=local_shoulder_circle_mask,
            local_shoulder_quadrant_mask=local_shoulder_quadrant_mask,
            tube_radius=arm_geometry.tube_radius,
            point_coords=local_points,
            point_labels=local_labels,
            flood_seed_coords=flood_seed_coords,
        )
    )

    full_mask = np.zeros(
        (ctx.height, ctx.width),
        dtype=bool,
    )
    full_mask[y1:y2, x1:x2] = local_refined_mask[
        :local_height,
        :local_width,
    ]

    full_barriers = np.zeros(
        (ctx.height, ctx.width),
        dtype=bool,
    )
    full_barriers[y1:y2, x1:x2] = local_barriers[
        :local_height,
        :local_width,
    ]

    return full_mask, full_barriers


def _resolve_person_bbox(
    ctx: SegmentationContext,
) -> tuple[int, int, int, int]:
    """
    Resolve the person bbox using YOLO proposals constrained by pose landmarks.

    This is a shared primitive rather than a target pipeline: head, hands and
    feet still use the resolved person bbox as SAM context even when their final
    target geometry is smaller and target-local.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, pose landmarks, YOLO
        configuration and node metadata.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive full-frame person bbox ``(x1, y1, x2, y2)`` selected from
        YOLO person proposals, or a pose-derived fallback bbox when proposals do
        not align with the detected pose.
    """
    yolo = get_yolo(model_name=ctx.cfg.yolo_model, device=ctx.cfg.device)
    res = yolo.predict(
        ctx.img_rgb,
        conf=float(ctx.cfg.conf),
        verbose=False,
        device=ctx.cfg.device,
    )[0]

    return resolve_person_bbox_xyxy(
        res,
        ctx.node_id,
        pose_xy=ctx.pose_xy,
        image_shape=ctx.img_rgb.shape,
    )


def _expand_prompt_bbox(
    ctx: SegmentationContext,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Apply the node-level prompt expansion to a SAM prompt bbox.

    ``prompt_expansion`` is a segmentation-only control: it changes the area
    shown to SAM while leaving final crop padding to ``box_margin`` and target
    geometry. Keeping this in one helper makes the target pipelines explicit
    about where the prompt domain changes.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image dimensions and configuration.
    bbox : tuple[int, int, int, int]
        End-exclusive bbox to expand.

    Returns
    -------
    tuple[int, int, int, int]
        Expanded and image-clipped bbox, or ``bbox`` unchanged when
        ``prompt_expansion`` is disabled.
    """
    if ctx.cfg.prompt_expansion <= 0:
        return bbox

    return expand_clip_bbox(
        *bbox,
        ctx.width,
        ctx.height,
        ctx.cfg.prompt_expansion,
    )


def _segment_person_silhouette(
    ctx: SegmentationContext,
    *,
    person_bbox: tuple[int, int, int, int],
    use_landmark_prompts: Optional[bool] = None,
) -> np.ndarray:
    """
    Segment a person silhouette inside a resolved person prompt bbox.

    This helper captures the reusable "segment the subject" operation used by
    the person pipeline and by target pipelines that need person-level guidance.
    It builds strict and complete SAM candidates, then selects a conservative
    mask using pose-landmark consistency.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, pose landmarks, SAM
        handles and configuration.
    person_bbox : tuple[int, int, int, int]
        End-exclusive bbox passed to SAM as the person prompt domain.
    use_landmark_prompts : bool | None, optional
        Whether strict SAM prompting should include positive pose points. If
        ``None``, the value follows ``cfg.min_landmark_fraction``. This does
        not disable landmark-based candidate validation; prompt construction and
        candidate validation are intentionally separate concerns.

    Returns
    -------
    np.ndarray
        Full-frame boolean mask for the selected person silhouette.
    """
    if use_landmark_prompts is None:
        use_landmark_prompts = ctx.cfg.min_landmark_fraction is not None

    candidates = _build_person_sam_candidates(
        img_rgb=ctx.img_rgb,
        bbox=person_bbox,
        pose_xy=ctx.pose_xy,
        processor=ctx.sam_processor,
        model=ctx.sam_model,
        device=ctx.cfg.device,
        node_id=ctx.node_id,
        use_landmark_prompts=use_landmark_prompts,
    )

    return _select_best_person_sam_mask(
        candidates,
        bbox=person_bbox,
        pose_xy=ctx.pose_xy,
        min_landmark_fraction=ctx.cfg.min_landmark_fraction,
    ).astype(bool)


def _segment_subject_with_single_sam_prompt(
    ctx: SegmentationContext,
    *,
    prompt_bbox: tuple[int, int, int, int],
    point_coords: Optional[list[list[float]]] = None,
    point_labels: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Segment a prompted subject by selecting the highest-scored SAM mask.

    This preserves the previous non-person behavior for head and hands: ask SAM
    once in the person prompt domain, take SAM's best-scored mask, then let the
    target pipeline apply polarity checks and target-local geometry.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, SAM handles and
        configuration.
    prompt_bbox : tuple[int, int, int, int]
        End-exclusive bbox passed to SAM.
    point_coords : list[list[float]] | None, optional
        Optional full-frame SAM point prompts.
    point_labels : list[int] | None, optional
        Optional SAM point labels aligned with ``point_coords``.

    Returns
    -------
    np.ndarray
        Full-frame boolean mask selected from SAM's raw candidates.

    Raises
    ------
    RuntimeError
        If SAM returns no masks or no candidate can be selected.
    """
    masks, scores = predict_sam_mask(
        img_rgb=ctx.img_rgb,
        bbox=prompt_bbox,
        processor=ctx.sam_processor,
        model=ctx.sam_model,
        device=ctx.cfg.device,
        point_coords=point_coords,
        point_labels=point_labels,
    )

    if masks is None or len(masks) == 0:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': SAM returned no masks."
        )

    best_mask = None
    best_key = None

    for i in range(len(masks)):
        mi = masks[i].astype(bool)
        score_i = (
            float(scores[i])
            if scores is not None and i < len(scores)
            else 0.0
        )

        if best_key is None or score_i > best_key:
            best_key = score_i
            best_mask = mi

    if best_mask is None:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': failed to select a SAM mask."
        )

    return best_mask.astype(bool)


def _ensure_subject_mask_polarity(
    ctx: SegmentationContext,
    *,
    mask: np.ndarray,
    prompt_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Invert likely-background SAM masks inside the prompt bbox.

    SAM may occasionally return the local background instead of the prompted
    subject. For target pipelines that first segment the whole person and then
    apply target-local geometry, pose landmark coverage is a cheap sanity check:
    if too few valid pose points fall inside the mask, the local inverse is more
    likely to represent the subject.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing pose landmarks.
    mask : np.ndarray
        Full-frame boolean candidate mask to validate.
    prompt_bbox : tuple[int, int, int, int]
        End-exclusive prompt bbox inside which inversion is allowed.

    Returns
    -------
    np.ndarray
        ``mask`` unchanged when polarity appears correct, otherwise the mask
        inverted only inside ``prompt_bbox``.
    """
    valid = (
        (ctx.pose_xy[:, 0] >= 0) &
        (ctx.pose_xy[:, 1] >= 0)
    )
    pts = ctx.pose_xy[valid]

    if len(pts) <= 0:
        return mask

    inside = 0
    mask_h, mask_w = mask.shape

    for px, py in pts:
        px = int(px)
        py = int(py)

        if 0 <= px < mask_w and 0 <= py < mask_h and mask[py, px]:
            inside += 1

    min_inside = max(1, int(math.ceil(len(pts) * 0.5)))

    if inside >= min_inside:
        return mask

    return invert_mask_inside_box(mask=mask, box=prompt_bbox)


# Target segmentation pipelines.
#
# Each function in this section owns the full segmentation flow for one logical
# target family. The pipeline resolves any target-specific geometry, performs the
# SAM calls it needs, applies target-local restrictions, and returns a uniform
# ``TargetSegmentationResult`` for the shared cleanup/output stage.
def _segment_person(ctx: SegmentationContext) -> TargetSegmentationResult:
    """
    Segment the full visible person target.

    The person pipeline is the simplest target-specific path: resolve the person
    bbox, optionally expand the SAM prompt domain, and select the best full-body
    silhouette. For this target the prompt bbox and target bbox are identical.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, pose landmarks and SAM
        dependencies.

    Returns
    -------
    TargetSegmentationResult
        Full-frame person mask with matching ``target_bbox`` and
        ``prompt_bbox``.
    """
    person_bbox = _resolve_person_bbox(ctx)
    prompt_bbox = _expand_prompt_bbox(ctx, person_bbox)
    mask = _segment_person_silhouette(
        ctx,
        person_bbox=prompt_bbox,
    )

    return TargetSegmentationResult(
        mask=mask,
        target_bbox=prompt_bbox,
        prompt_bbox=prompt_bbox,
    )


def _segment_head(ctx: SegmentationContext) -> TargetSegmentationResult:
    """
    Segment the head target using person-level SAM guidance.

    The head pipeline deliberately separates segmentation guidance from output
    geometry. SAM sees the person prompt bbox so it can find the correct subject;
    the final mask is then clipped to a face-derived square head bbox so crop
    geometry stays head-local and hair-friendly.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, pose landmarks, SAM
        dependencies and face-landmarker configuration.

    Returns
    -------
    TargetSegmentationResult
        Head-local mask, target head bbox, person prompt bbox, and full-frame
        face landmarks for debug rendering.
    """
    person_bbox = _resolve_person_bbox(ctx)
    person_prompt_bbox = _expand_prompt_bbox(ctx, person_bbox)

    landmarker = get_mediapipe_face_landmarker(
        model_asset_path=ctx.cfg.face_landmarker_task,
        device=ctx.cfg.device,
    )

    head_area_rgb, a_x, a_y = crop_head_area_from_pose(
        img_rgb=ctx.img_rgb,
        pose_xy=ctx.pose_xy,
        expansion=FACE_SEARCH_AREA_EXPANSION,
    )

    face_xy = mp_face_landmarks(
        img_rgb=head_area_rgb,
        face_landmarker=landmarker,
    )
    debug_face_xy = face_xy.copy()
    debug_face_xy[:, 0] += a_x
    debug_face_xy[:, 1] += a_y

    r_x1, r_y1, r_x2, r_y2 = face_bbox_xyxy_from_landmarks(
        face_xy,
        image_shape=head_area_rgb.shape,
    )

    fx1 = r_x1 + a_x
    fy1 = r_y1 + a_y
    fx2 = r_x2 + a_x
    fy2 = r_y2 + a_y

    head_bbox = square_head_bbox_from_face_bbox(
        fx1,
        fy1,
        fx2,
        fy2,
        ctx.width,
        ctx.height,
        expansion=ctx.cfg.expansion,
    )

    mask = _segment_subject_with_single_sam_prompt(
        ctx,
        prompt_bbox=person_prompt_bbox,
    )
    mask = _ensure_subject_mask_polarity(
        ctx,
        mask=mask,
        prompt_bbox=person_prompt_bbox,
    )

    hx1, hy1, hx2, hy2 = head_bbox
    head_region_mask = np.zeros((ctx.height, ctx.width), dtype=bool)
    head_region_mask[hy1:hy2, hx1:hx2] = True

    return TargetSegmentationResult(
        mask=mask & head_region_mask,
        target_bbox=head_bbox,
        prompt_bbox=person_prompt_bbox,
        face_xy=debug_face_xy,
    )


def _segment_hands(ctx: SegmentationContext) -> TargetSegmentationResult:
    """
    Segment hand targets using hand geometry and person-level SAM guidance.

    The hand pipeline keeps the existing semantics: SAM segments the selected
    person in the person prompt bbox, while MediaPipe hand landmarks define the
    target-local hand bbox and mask. Intersecting both masks preserves
    subject-vs-background separation without letting the crop grow beyond the
    requested hand target.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, pose landmarks, SAM
        dependencies and hand-landmarker configuration.

    Returns
    -------
    TargetSegmentationResult
        Hand-local mask, hand bbox, person prompt bbox, hand part mask for
        independent cleanup, and hand-landmarker result for debug rendering.
    """
    person_bbox = _resolve_person_bbox(ctx)
    person_prompt_bbox = _expand_prompt_bbox(ctx, person_bbox)

    hand_landmarker = get_mediapipe_hand_landmarker(
        model_asset_path=ctx.cfg.hand_landmarker_task,
        device=ctx.cfg.device,
    )

    hands_res = mp_hand_landmarks_full(
        img_rgb=ctx.img_rgb,
        hand_landmarker=hand_landmarker,
    )

    hand_which = {
        'hands': 'both',
        'left-hand': 'left',
        'right-hand': 'right',
    }[ctx.cfg.target]

    hand_bbox = hands_bbox_xyxy_from_landmarks(
        hands_res,
        ctx.img_rgb.shape,
        which=hand_which,
        expansion=max(1.0, float(ctx.cfg.expansion)),
    )

    hand_mask = hands_mask_from_landmarks(
        hands_res,
        ctx.img_rgb.shape,
        which=hand_which,
        expansion=max(1.0, float(ctx.cfg.expansion)),
    )

    point_coords, point_labels = positive_points_for_sam(
        xy=ctx.pose_xy,
        bbox=person_prompt_bbox,
    )
    subject_mask = _segment_subject_with_single_sam_prompt(
        ctx,
        prompt_bbox=person_prompt_bbox,
        point_coords=point_coords,
        point_labels=point_labels,
    )
    subject_mask = _ensure_subject_mask_polarity(
        ctx,
        mask=subject_mask,
        prompt_bbox=person_prompt_bbox,
    )

    return TargetSegmentationResult(
        mask=subject_mask & hand_mask,
        target_bbox=hand_bbox,
        prompt_bbox=person_prompt_bbox,
        shape_part_masks=hand_mask,
        hands_result=hands_res,
        preserve_all_components=True,
    )


def _segment_arms(ctx: SegmentationContext) -> TargetSegmentationResult:
    """
    Segment one or both arms using pose-guided geometry, local SAM inference,
    Canny barriers, and seeded flood-fill refinement.

    The arm pipeline treats each selected arm as an independent segmentation
    region. It first segments the complete person silhouette, then derives a
    coarse arm prior from shoulder, elbow, and wrist landmarks. The prior is
    used to neutralize unrelated image content before running SAM on a local
    crop.

    Each selected SAM mask is subsequently refined against the original,
    non-neutralized image. Canny edges close to the expected arm skeleton are
    strengthened and combined with synthetic barriers at the shoulder and
    wrist. A flood-fill seeded by positive arm points outside the trusted
    shoulder circle retains only arm-connected regions.

    The shoulder is handled separately from the main flood-fill result:

    - the geometric shoulder circle is restored explicitly;
    - SAM-selected pixels inside the external shoulder quadrant are preserved;
    - positive prompts inside the shoulder circle are excluded from flood-fill
      seeds so they cannot bypass the synthetic shoulder barrier.

    Chromatic prompt enrichment adds positive points along visually coherent
    runs between the arm landmarks. This helps SAM follow transitions between
    skin, sleeves, cuffs, shadows, and patterned fabric.

    When multiple arms are requested, each arm is segmented and refined
    independently. Their masks are then combined, while the per-arm masks are
    preserved for independent shape cleanup and debug rendering.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing the source image, MediaPipe pose
        landmarks, validated node configuration, image dimensions, and loaded
        SAM processor and model.

    Returns
    -------
    TargetSegmentationResult
        Arm segmentation result containing:

        - the union of all selected full-frame arm masks;
        - the union of the arm base bounding boxes as ``target_bbox``;
        - the union of the local SAM crop boxes as ``prompt_bbox``;
        - one full-frame mask per arm in ``shape_part_masks``;
        - the enriched SAM regions used for inference and debug rendering;
        - the coarse arm priors and final edge/barrier masks as debug data.

        ``preserve_all_components`` is enabled because valid arm masks may
        contain multiple disconnected regions, especially around loose
        clothing, occlusions, or separate visible fabric and skin areas.

    Raises
    ------
    RuntimeError
        If the person silhouette cannot be segmented, no usable arm geometry
        can be derived from the pose landmarks, or a selected arm has no valid
        local positive SAM prompts.

    ValueError
        If the configured arm target cannot be mapped to a supported arm
        selection mode.

    Notes
    -----
    The coarse arm prior is used only to control which pixels are visible to
    SAM. It is not used directly as the final arm mask.

    Canny operates on the original local RGB crop rather than the neutralized
    SAM input so that real boundaries between arm, clothing, torso, and
    surrounding content remain available during refinement.

    Side-specific targets use image/viewer perspective. Anatomical side
    resolution is handled internally by the arm geometry helpers.
    """
    person_bbox = _resolve_person_bbox(ctx)
    person_prompt_bbox = _expand_prompt_bbox(ctx, person_bbox)
    silhouette = _segment_person_silhouette(
        ctx,
        person_bbox=person_prompt_bbox,
    )

    arm_which = {
        'arms': 'both',
        'left-arm': 'left',
        'right-arm': 'right',
    }[ctx.cfg.target]

    arm_geometries = arm_regions_from_landmarks(
        pose_xy=ctx.pose_xy,
        image_shape=ctx.img_rgb.shape,
        silhouette=silhouette,
        which=arm_which,
        expansion=max(1.0, float(ctx.cfg.expansion)),
    )
    arm_regions = [
        geometry.sam_region
        for geometry in arm_geometries
    ]

    prior_masks = [
        geometry.prior_mask
        for geometry in arm_geometries
    ]

    enriched_regions = _enrich_arm_sam_regions_with_chromatic_points(
        img_rgb=ctx.img_rgb,
        pose_xy=ctx.pose_xy,
        arm_regions=arm_regions,
        prior_masks=prior_masks,
    )

    arm_geometries = [
        replace(
            geometry,
            sam_region=enriched_region,
        )
        for geometry, enriched_region in zip(
            arm_geometries,
            enriched_regions,
        )
    ]

    mask = np.zeros((ctx.height, ctx.width), dtype=bool)
    arm_region_masks: list[np.ndarray] = []
    arm_edge_masks: list[np.ndarray] = []

    for arm_geometry in arm_geometries:
        region_mask, edge_mask = _predict_arm_mask_on_prior_crop(
            ctx,
            arm_geometry=arm_geometry,
        )

        if not np.any(region_mask):
            region_mask = arm_geometry.prior_mask.copy()

        arm_region_masks.append(region_mask)
        arm_edge_masks.append(edge_mask)
        mask |= region_mask

    return TargetSegmentationResult(
        mask=mask,
        target_bbox=union_bboxes_xyxy([
            geometry.sam_region.base_bbox
            for geometry in arm_geometries
        ]),
        prompt_bbox=union_bboxes_xyxy([
            geometry.sam_region.prompt_bbox
            for geometry in arm_geometries
        ]),
        shape_part_masks=arm_region_masks,
        sam_regions=[
            geometry.sam_region
            for geometry in arm_geometries
        ],
        prompt_bbox_label='arm-prompt',
        preserve_all_components=True,
        debug_region_masks=[
            geometry.prior_mask
            for geometry in arm_geometries
        ],
        debug_edge_masks=arm_edge_masks,
    )


def _segment_feet(ctx: SegmentationContext) -> TargetSegmentationResult:
    """
    Segment foot targets independently by foot prompt region.

    Feet are handled per region because each foot may need different prompt
    geometry, chromatic prompt enrichment, leg-probe refinement and cleanup.
    Each region is segmented independently and clipped to its prompt bbox before
    the regional masks are unioned.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context containing image data, pose landmarks, SAM
        dependencies and foot-related configuration.

    Returns
    -------
    TargetSegmentationResult
        Unioned foot mask, unioned target bbox from base foot regions, unioned
        SAM prompt bbox, per-foot masks for independent cleanup, and final foot
        regions for debug rendering.
    """
    person_bbox = _resolve_person_bbox(ctx)

    foot_which = {
        'feet': 'both',
        'left-foot': 'left',
        'right-foot': 'right',
    }[ctx.cfg.target]

    foot_regions = foot_sam_regions_from_landmarks(
        ctx.pose_xy,
        ctx.img_rgb.shape,
        which=foot_which,
        expansion=max(1.0, float(ctx.cfg.expansion)),
        person_bbox=person_bbox,
    )

    if ctx.cfg.prompt_expansion > 0:
        foot_regions = [
            foot_sam_region_with_prompt_bbox(
                region,
                ctx.pose_xy,
                prompt_bbox=expand_clip_bbox(
                    *region.prompt_bbox,
                    ctx.width,
                    ctx.height,
                    ctx.cfg.prompt_expansion,
                ),
            )
            for region in foot_regions
        ]

    foot_regions = _enrich_foot_sam_regions_with_chromatic_points(
        img_rgb=ctx.img_rgb,
        pose_xy=ctx.pose_xy,
        foot_regions=foot_regions,
    )

    mask = np.zeros((ctx.height, ctx.width), dtype=bool)
    foot_region_masks: list[np.ndarray] = []

    for region in foot_regions:
        masks, scores = predict_sam_mask(
            img_rgb=ctx.img_rgb,
            bbox=region.prompt_bbox,
            processor=ctx.sam_processor,
            model=ctx.sam_model,
            device=ctx.cfg.device,
            point_coords=region.point_coords,
            point_labels=region.point_labels,
        )

        foot_candidates = build_sam_candidates_from_raw_masks(
            masks,
            scores,
            bbox=region.prompt_bbox,
            source='foot',
            node_id=ctx.node_id,
            include_inverted=True,
        )
        region_mask = _select_best_foot_sam_mask(
            foot_candidates,
            bbox=region.prompt_bbox,
            point_coords=region.point_coords,
            point_labels=region.point_labels,
            preferred_source='foot',
        )

        region_mask = clip_mask_to_bbox(region_mask, region.prompt_bbox)

        if region.probe_point is not None:
            probe_masks, probe_scores = predict_sam_mask(
                img_rgb=ctx.img_rgb,
                bbox=region.prompt_bbox,
                processor=ctx.sam_processor,
                model=ctx.sam_model,
                device=ctx.cfg.device,
                point_coords=[region.probe_point],
                point_labels=[1],
            )
            probe_candidates = build_sam_candidates_from_raw_masks(
                probe_masks,
                probe_scores,
                bbox=region.prompt_bbox,
                source='leg-probe',
                node_id=ctx.node_id,
                include_inverted=True,
            )
            probe_mask = select_best_guided_sam_mask(
                probe_candidates,
                bbox=region.prompt_bbox,
                required_points=[region.probe_point],
                forbidden_points=None,
                min_required_fraction=1.0,
                target_norm_area=0.5,
                preferred_source='leg-probe',
                context='foot leg-probe SAM mask',
            )
            probe_mask = clip_mask_to_bbox(
                probe_mask,
                region.prompt_bbox,
            )
            probe_coverage = reference_mask_coverage(region_mask, probe_mask)

            if probe_coverage < FOOT_LEG_PROBE_COVERAGE_THRESHOLD:
                refined_points = list(region.point_coords or [])
                refined_labels = list(region.point_labels or [])
                refined_points.append(region.probe_point)
                refined_labels.append(0)

                refined_masks, refined_scores = predict_sam_mask(
                    img_rgb=ctx.img_rgb,
                    bbox=region.prompt_bbox,
                    processor=ctx.sam_processor,
                    model=ctx.sam_model,
                    device=ctx.cfg.device,
                    point_coords=refined_points,
                    point_labels=refined_labels,
                )
                refined_candidates = build_sam_candidates_from_raw_masks(
                    refined_masks,
                    refined_scores,
                    bbox=region.prompt_bbox,
                    source='refined-foot',
                    node_id=ctx.node_id,
                    include_inverted=True,
                )
                region_mask = _select_best_foot_sam_mask(
                    refined_candidates,
                    bbox=region.prompt_bbox,
                    point_coords=refined_points,
                    point_labels=refined_labels,
                    preferred_source='refined-foot',
                )
                region_mask = clip_mask_to_bbox(
                    region_mask,
                    region.prompt_bbox,
                )

        foot_region_masks.append(region_mask)
        mask |= region_mask

    return TargetSegmentationResult(
        mask=mask,
        target_bbox=union_bboxes_xyxy([
            region.base_bbox for region in foot_regions
        ]),
        prompt_bbox=union_bboxes_xyxy([
            region.prompt_bbox for region in foot_regions
        ]),
        shape_part_masks=foot_region_masks,
        sam_regions=foot_regions,
        prompt_bbox_label='foot-prompt',
        preserve_all_components=True,
    )


def _segment_target(ctx: SegmentationContext) -> TargetSegmentationResult:
    """
    Dispatch to the target-specific segmentation pipeline.

    Parameters
    ----------
    ctx : SegmentationContext
        Shared runtime context. ``ctx.cfg.target`` selects the target pipeline.

    Returns
    -------
    TargetSegmentationResult
        Uniform segmentation result produced by the selected target pipeline.

    Raises
    ------
    ValueError
        If ``ctx.cfg.target`` is not one of the validated SubjectCrop targets.
    """
    if ctx.cfg.target == 'person':
        return _segment_person(ctx)
    if ctx.cfg.target == 'head':
        return _segment_head(ctx)
    if ctx.cfg.target in ('hands', 'left-hand', 'right-hand'):
        return _segment_hands(ctx)
    if ctx.cfg.target in ('arms', 'left-arm', 'right-arm'):
        return _segment_arms(ctx)
    if ctx.cfg.target in ('feet', 'left-foot', 'right-foot'):
        return _segment_feet(ctx)

    raise ValueError(f"'{ctx.node_id}': invalid target={ctx.cfg.target!r}")


@dataclass
class SubjectCrop(CudaPostRunMixin, NodeRef):
    """
    Subject-aware crop and inpaint-mask generator using MediaPipe, YOLO, and
    SAM-compatible segmentation.

    ``SubjectCrop`` detects subject/body-level regions of interest and produces
    either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image
    (``mode='mask'`` or ``mode='negative-mask'``).

    Supported targets are:

    - ``person``: full visible subject crop/mask;
    - ``head``: head-oriented crop/mask with hair-friendly framing;
    - ``hands``: one or more visible hand crops/masks;
    - ``left-hand``: hand appearing on the left side of the image;
    - ``right-hand``: hand appearing on the right side of the image;
    - ``arms``: one or more visible arm crops/masks, from shoulder to wrist;
    - ``left-arm``: arm appearing on the left side of the image;
    - ``right-arm``: arm appearing on the right side of the image;
    - ``feet``: one or more visible foot crops/masks;
    - ``left-foot``: foot appearing on the left side of the image;
    - ``right-foot``: foot appearing on the right side of the image.

    Face-detail targets such as ``face``, ``eyes``, ``left-eye``,
    ``right-eye``, ``eyebrows``, ``left-eyebrow`` and ``right-eyebrow`` are
    handled by ``FaceCrop``. Keeping these concerns separate makes
    ``SubjectCrop`` responsible only for person selection, pose-guided geometry,
    hand/foot localization, and SAM-based subject segmentation.

    The node is intended to support workflows such as:

    - extracting subjects for compositing (e.g. with ``ImageStack``);
    - producing full-frame inpaint masks for SDXL pipelines;
    - extracting head regions for FaceID / IP-Adapter refinement;
    - extracting hand regions for localized hand repair/refinement;
    - extracting arm regions for localized pose, sleeve or skin refinement;
    - extracting foot regions for localized foot repair/refinement;
    - refining a small region by cropping, processing it separately, and
      reinserting it at the original coordinates.

    Side-specific hand and foot targets use image/viewer perspective:

    - ``left-hand`` refers to the hand on the left side of the image;
    - ``right-hand`` refers to the hand on the right side of the image.
    - ``left-arm`` refers to the arm on the left side of the image;
    - ``right-arm`` refers to the arm on the right side of the image.
    - ``left-foot`` refers to the foot on the left side of the image;
    - ``right-foot`` refers to the foot on the right side of the image.

    This is intentionally different from MediaPipe handedness labels, which follow
    anatomical subject perspective and are mapped internally when needed.

    Pipeline
    --------

    The node combines several models to robustly localize a subject region:

    - MediaPipe Pose is always executed first to obtain body landmarks.
      These landmarks provide the primary anatomical consistency signal for
      locating the selected subject.

    - YOLO (COCO class 0) proposes candidate person bounding boxes for all
      supported targets. YOLO boxes are accepted only when they are consistent
      with the valid MediaPipe pose landmarks. If YOLO fails or returns an
      inconsistent person box, a pose-derived fallback person bbox is used.

    - For ``target='head'``, a coarse head / upper-body search area is derived
      from pose landmarks. MediaPipe Face Landmarker runs inside this search area
      to obtain accurate face landmarks. A square head crop box is then derived
      from the face bbox, with an upward bias to preserve hair.

    - For hand targets, MediaPipe Hand Landmarker runs on the full image and is
      used to derive hand-local bounding boxes and landmark-based hand masks.

    - For foot targets, MediaPipe Pose foot landmarks (ankle, heel and
      foot_index) are used to derive foot-local search regions.

    - Hands and feet use target-local segmentation, where the requested extremity
      defines the primary segmentation geometry while the resolved person is used
      only as contextual guidance.
    - Arms use a target-local shoulder/arm ROI clipped to the person silhouette
      before and after SAM segmentation.

    ``target='person'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The best YOLO bbox is selected using pose-landmark alignment.
    - The selected YOLO bbox is accepted only if it contains a sufficient
      fraction of valid pose landmarks; otherwise, a fallback bbox is inferred
      directly from pose landmarks.
    - The prompt bbox may optionally be expanded using ``prompt_expansion``.
    - A SAM-compatible segmentation model generates multiple candidate masks
      using complementary prompt strategies.
    - Candidate masks are evaluated using semantic and geometric consistency
      criteria to select the most plausible subject segmentation.
    - The final crop region is derived from the selected mask and then
      optionally expanded using ``box_margin``.

    ``target='head'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - A coarse head / upper-body search area is derived from pose landmarks.
    - A landmark-derived face bbox is computed in local search-area coordinates
      and remapped to full-image coordinates.
    - A square head crop box is computed from the face bbox using
      ``expansion`` and an upward bias to preserve hair.
    - SAM segmentation is guided by the resolved person bbox for robustness,
      while the final crop region corresponds to the derived head box.
    - This intentionally separates segmentation guidance from crop geometry:
      the person bbox helps SAM find the correct subject, while the head box
      defines the output crop.

    ``target='hands'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - MediaPipe Hand Landmarker detects reliable hands in the full image.
    - A landmark-derived hand mask is constructed for the selected hand(s).
    - Hand-local crop boxes are derived from landmarks and optionally expanded
      via ``expansion``.
    - SAM segmentation is guided by the resolved person bbox and positive
      pose landmarks.
    - The final mask is the intersection of the SAM subject mask and the
      landmark-derived hand mask.
    - The final crop region isolates the requested hand(s).

    ``target='left-hand'`` and ``target='right-hand'``
    - A single hand is selected using image/viewer perspective.
    - MediaPipe handedness labels are mapped internally because they follow
      anatomical subject perspective.

    ``target='feet'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - Foot-local search regions are derived from pose landmarks and optionally
      expanded via ``expansion``.
    - Each visible foot is segmented independently using target-local
      segmentation guided by the resolved person bbox.
    - The resulting foot masks are combined into the final output.
    - The final crop region isolates one or more visible feet.
    - The implementation favors preserving the complete visible foot or
      footwear over producing perfectly clean masks. Small artifacts, holes or
      detached fragments may therefore remain and can be cleaned by downstream
      mask post-processing.

    ``target='left-foot'`` / ``target='right-foot'``
    - Follow the same pipeline while selecting only the foot appearing on the
      left or right side of the image, respectively.
    - Side selection always follows image/viewer perspective.

    Local extremity targets
    -------------------

    Hands and feet are segmented using target-local geometry together with
    subject-aware segmentation.

    Unlike the person target, the crop geometry is determined by the selected
    extremity rather than the subject bbox.

    These targets prioritize preserving the requested extremity while limiting
    background inclusion.

    Small artifacts or detached fragments may remain in difficult cases and are
    intended to be handled by downstream mask cleanup.

    Parameters
    ----------

    name : str, optional
        Node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default input
        (``input['default']['image']`` or ``input['default']['path']``).

    spec : dict or str or Path, optional
        Node specification (inline dict or path to a config file), resolved via
        ``resolve_spec``.

    Expected structure:

    ``model`` : dict
        ``device`` : str, optional
            Inference device (e.g. ``'cuda'``, ``'cuda:0'``, ``'cpu'``).
            Default: ``'cuda'``.

        ``dtype`` : str, optional
            Torch dtype used to load the SAM-compatible segmentation model.
            Supported values follow ``resolve_dtype`` conventions, e.g.
            ``'bf16'``, ``'float16'`` or ``'float32'``. Default: ``'bf16'``.

        ``sam_model`` : str, optional
            Hugging Face SAM-compatible model identifier used for mask
            generation. Default: ``'facebook/sam-vit-large'``.

            Typical values include:

            - ``'facebook/sam-vit-base'``;
            - ``'facebook/sam-vit-large'``;
            - ``'facebook/sam-vit-huge'``;
            - ``'syscv-community/sam-hq-vit-base'``;
            - ``'syscv-community/sam-hq-vit-large'``;
            - ``'syscv-community/sam-hq-vit-huge'``.

            Required for all supported targets because ``SubjectCrop`` uses SAM
            for ``person``, ``head``, hand and foot targets.

        ``yolo_model`` : str, optional
            YOLO weights used to propose person boxes. Default:
            ``'yolov8n.pt'``.

            YOLO boxes are accepted only when they are consistent with
            MediaPipe pose landmarks; otherwise the node falls back to a
            pose-derived person bbox.

        ``pose_landmarker_task`` : str
            MediaPipe PoseLandmarker ``.task`` path. Required for all targets.
            Pose landmarks are used to select the correct person bbox and to
            guide subject-level geometry.

        ``face_landmarker_task`` : str, optional
            MediaPipe FaceLandmarker ``.task`` path. Required only for
            ``target='head'``.

        ``hand_landmarker_task`` : str, optional
            MediaPipe HandLandmarker ``.task`` path. Required for
            ``target='hands'``, ``target='left-hand'`` and
            ``target='right-hand'``.

    ``params`` : dict
        ``target`` : {'person', 'head', 'hands', 'left-hand', 'right-hand', 'arms', 'left-arm', 'right-arm', 'feet', 'left-foot', 'right-foot'}, optional
            Region to extract. Default: ``'person'``.

        ``mode`` : {'default', 'mask', 'negative-mask'}, optional
            Output type:

            - ``'default'``:
            RGBA cutout.
            - ``'mask'``:
            full-frame 8-bit mask (white = selected region).
            - ``'negative-mask'``:
            inverted full-frame mask (white = background).

            Default: ``'default'``.

        ``crop_mode`` : {'bbox', 'bbox[w:h]', 'trim', 'full_frame'}, optional
            Applies only when ``mode='default'`` and controls the spatial
            layout of the RGBA cutout.

            Supported forms are:

            - ``'bbox'``:
              Output is the rectangular crop inside the selected crop box,
              including the original background; alpha is fully opaque
              (255 everywhere).

            - ``'bbox[w:h]'``:
              Same as ``'bbox'``, but the selected crop box is expanded toward
              the requested aspect ratio ``w:h`` while keeping the target fully
              inside the crop and staying within source-image bounds.

              The requested ratio is treated as a target, not a hard constraint.
              If the source image does not provide enough room near the borders,
              the final crop may deviate from the requested ratio.

            - ``'trim'``:
              Output is an RGBA cutout cropped to the selected bounding box,
              with alpha derived from the mask. The result is then tightly
              trimmed to the minimal box containing non-transparent pixels.

            - ``'full_frame'``:
              Same cutout as ``'trim'`` but placed back into a full-size RGBA
              canvas of the original image dimensions, preserving the original
              coordinates.

              Default: ``'trim'``.

        ``conf`` : float, optional
            YOLO confidence threshold. Typical range: 0.2-0.6.
            Default: 0.35.

        ``box_margin`` : int or str, optional
            Symmetric margin applied to the final crop bbox derived from the
            post-processed target mask. Supported forms follow the standard
            size-expression convention: integer pixels, ``'<n>px'`` or
            ``'<n>%'``. Percentages are resolved against the mask bbox width
            for left/right and mask bbox height for top/bottom.

            This margin is applied only to cropped default outputs
            (``crop_mode='trim'``, ``'bbox'`` or ``'bbox[w:h]'``). It is ignored
            for ``mode='mask'``, ``mode='negative-mask'`` and
            ``crop_mode='full_frame'`` because those outputs preserve the full
            source frame.

            Default: ``'12%'``.

        ``prompt_expansion`` : float, optional
            Advanced SAM prompt padding ratio. This expands the bbox passed to
            SAM before segmentation while leaving the output crop geometry
            controlled by the post-processed mask and ``box_margin``.

            Default: 0.0.

        ``postprocess`` : dict, optional
            Structural cleanup applied to the selected target silhouette before
            deriving crop geometry, RGBA alpha, or full-frame mask output. This
            block defines the canonical shape used by the node, so it is applied
            in every mode. Output-only mask refinements such as
            ``close_radius``, ``dilate_radius`` and ``smoothing_radius`` are
            applied later and only for ``mode='mask'`` or
            ``mode='negative-mask'``.

            Processing order is fixed: ``fill_holes`` ->
            ``morph_open_radius`` -> ``min_component_area``.
            For logical multi-part targets such as ``hands`` and ``feet``,
            cleanup is applied independently to each target part before the
            parts are unioned; therefore ``min_component_area='biggest'`` keeps
            the largest component per hand/foot, not one hand/foot globally.

            ``fill_holes`` : int, float, str, 'all' or None, optional
                Fill enclosed background holes inside the selected shape before
                removing thin details. ``0`` or ``None`` disables hole filling.
                ``'all'`` fills every enclosed hole. Numeric values are pixel
                areas. Percentage strings such as ``'1%'`` follow the shared
                component-area convention: the percentage is measured on the
                image long side and squared into an area threshold. Only holes
                with area less than or equal to the resolved threshold are
                filled.

            ``morph_open_radius`` : int, optional
                Radius in pixels for morphological opening, applied after hole
                filling. Opening removes thin lines, speckles, and small bridges
                while preserving surviving larger regions. ``0`` disables this
                step.

            ``min_component_area`` : int, float, str, 'biggest' or None, optional
                Remove disconnected foreground components after hole filling and
                opening. ``0`` or ``None`` disables component filtering. Numeric
                values are pixel areas. Percentage strings use the same
                long-side area convention as ``fill_holes``. ``'biggest'`` keeps
                only the largest connected component, useful for single-subject
                crops but potentially destructive for legitimate multi-part
                targets such as hands, feet, or eyes.

            Default: all disabled.

        ``expansion`` : float, optional
            Expansion factor applied to target-local crop geometry.

            - for ``target='head'``:
             controls the derived square head crop size;
            - for hand targets:
              expands the landmark-derived hand bbox / mask;
            - for arm targets:
              expands the pose-derived shoulder cap and arm tube radius;
            - for foot targets:
              expands the derived foot-local search regions;
            - for ``target='person'``:
              the output crop padding is controlled by ``box_margin``.

              Default: ``1.0``.

        ``dilate_radius`` : int, optional
            Mask dilation radius in pixels.

            In ``mode='mask'``, dilation expands the repaintable selected
            region. In ``mode='negative-mask'``, dilation is applied before
            inversion, so it expands the protected subject region and creates a
            safety margin between the subject and the repaintable background.

            Default: 0.

        ``close_radius`` : int, optional
            Morphological closing radius in pixels.

            Closing is applied before optional inversion. It fills small holes
            and gaps in the selected subject mask. This is often useful for
            stable inpainting, but in ``mode='negative-mask'`` it also means
            that small background holes inside the subject silhouette become
            protected after inversion. Set this to 0 when those internal holes
            should remain repaintable background.

            Default: 0.

        ``smoothing_radius`` : int, optional
            Gaussian smoothing radius in pixels.

            Smoothing is applied before optional inversion. In ``mode='mask'``,
            it softens the repaintable selected region. In
            ``mode='negative-mask'``, it softens the protected subject boundary
            before inversion, producing a feathered transition between protected
            subject and repaintable background.

            Default: 0.

        ``min_landmark_fraction`` : float | None, optional
            Minimum fraction of usable stable pose landmarks that must be
            contained within a candidate SAM mask to consider it
            landmark-consistent.

            Default: 0.8.

            Typical values around 0.7 are often effective. Increasing the value
            makes pose consistency stricter, while lowering it makes candidate
            selection more permissive.

            If set to ``None``, strict landmark prompting is disabled for
            ``target='person'`` and only bbox-only SAM mask candidates are
            generated and evaluated.

            This parameter can help tune performance for difficult poses,
            occluded limbs, or noisy landmark detections.

    ``debug`` : dict
        ``save_debug`` : bool, optional
            If True, saves a debug image with the selected crop/prompt bbox
            overlay. Default: False.

    Mask post-processing
    --------------------

    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the detected subject mask is
    post-processed before it is written to disk.

    Post-processing is always applied to the positive subject mask first, before any
    optional polarity inversion:

    1. the selected subject region is assembled as a full-frame positive mask;
    2. ``close_radius``, ``dilate_radius``, and ``smoothing_radius`` are applied;
    3. if ``mode='negative-mask'``, the post-processed subject mask is inverted.

    This ordering is intentional.

    For ``mode='mask'``, the output mask directly marks the selected subject region
    as repaintable. Dilation and smoothing therefore expand and soften the subject
    region itself, which is useful when repainting or refining the selected target.

    For ``mode='negative-mask'``, the output mask marks the background as
    repaintable and protects the selected subject. Applying dilation and smoothing
    before inversion creates a protected safety band around the subject. This
    prevents the background inpaint area from bleeding into the subject boundary.

    In other words, with ``mode='negative-mask'``:

    * ``dilate_radius`` expands the protected subject area before inversion;
    * ``smoothing_radius`` feathers the transition around the protected subject;
    * ``close_radius`` closes small holes inside the protected subject area before
      inversion.

    If preserving holes inside the subject mask is important, for example gaps
    between arms, fingers, hair strands, or other background-visible openings,
    prefer setting ``close_radius=0``.

    Returns
    -------

    dict
    Output metadata dictionary (also written as a JSON sidecar) with:

    ``ok`` : bool
        Success flag.

    ``node`` : str
        Operator name.

    ``id`` : str
        Node identifier.

    ``input_image`` : str
        Source image path.

    ``mode`` : str
        Output mode.

    ``image`` : str
        Output file path (RGBA cutout or mask).

    ``params`` : dict
        Configuration parameters.

    ``model`` : dict
        Resolved model/runtime metadata, including the selected ``sam_model``,
        MediaPipe task paths, optional YOLO model, device, and dtype.

    ``crop`` : dict
        Crop metadata useful for reinsertion/compositing:

        ``anchor_xy`` : list[int]
            Local anchor inside the output crop image.

        ``position`` : list[int]
            Source-image position where ``anchor_xy`` should be placed when
            reconstructing the original geometry.

        ``bbox_size`` : list[int]
            Width and height of the effective crop box in pixels.

        ``bbox_xyxy`` : list[int]
            Effective end-exclusive crop box ``[x1, y1, x2, y2]`` in
            source-image coordinates.

    ``debug_bbox`` : str, optional
        Debug image path, present only when ``debug.save_debug`` is true.

    ``metadata`` : str
        JSON sidecar path.

    Notes
    -----

    - Mask outputs are always full-frame and aligned to the original image size.
    - In ``mode='default'``, ``crop_mode`` controls whether the output is a
      target-local RGBA cutout, a bbox crop, or a full-frame RGBA canvas.
    - In ``mode='mask'`` and ``mode='negative-mask'``, ``crop_mode`` is ignored
      because mask outputs are always full-frame.
    - ``dilate_radius``, ``close_radius`` and ``smoothing_radius`` are applied
      to the positive target mask before output. In ``mode='default'``, the
      crop bbox is derived from this post-processed mask.
    - ``target='hands'``, ``target='arms'`` and ``target='feet'`` may preserve
      multiple disconnected target components inside the same crop.
    - Side-specific hand, arm and foot targets always follow image/viewer
      perspective.
    - MediaPipe handedness labels follow anatomical subject perspective and are
      mapped internally to preserve the image/viewer convention.
    - This node does not require a local ``sam_checkpoint`` /
    ``sam_model_type`` pair. Use ``model.sam_model`` to select the Hugging Face
      model id.
    - For ``crop_mode='bbox[w:h]'``, the requested aspect ratio is treated as a
      target rather than a hard guarantee. Near image boundaries the final crop
      may deviate from the requested ratio.
    - If you change the implementation or specification and need fresh outputs,
      delete the existing sidecar JSON to avoid reusing cached results.
  """

    # Either pass a path explicitly, or wire an upstream image into default input.
    path: Optional[Union[str, Path]] = None

    # Optional node spec (device, etc.)
    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        return is_cuda_device(spec.get('model', {}).get('device', 'cuda'))

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        # Local imports to avoid hard deps if node unused
        import cv2

        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        # -----------------------------
        # Resolve input image path
        # -----------------------------
        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input
        )

        out_dir = Path(output_dir)

        # -----------------------------
        # Load image
        # -----------------------------
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(
                f"SubjectCrop node '{node_id}': cannot read image: {img_path}")

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        debug_mask: Optional[np.ndarray] = None

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        pose_xy = mp_pose_landmarks_xy(
            img_rgb=img_rgb,
            pose_landmarker=pose_landmarker,
        )

        sam_model_id = cfg.sam_model
        processor, sam_model = get_sam(
            model_id=sam_model_id,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        ctx = SegmentationContext(
            node_id=node_id,
            cfg=cfg,
            img_rgb=img_rgb,
            pose_xy=pose_xy,
            width=w,
            height=h,
            sam_processor=processor,
            sam_model=sam_model,
        )
        segmentation = _segment_target(ctx)

        mask = segmentation.mask.astype(bool)
        shape_part_masks = segmentation.shape_part_masks
        hint_x1, hint_y1, hint_x2, hint_y2 = segmentation.target_bbox

        # --------------------------------------------------
        # Build the final positive mask and derive output geometry from it.
        # --------------------------------------------------

        cm = mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm, connectivity=8
        )

        if num > 1:
            if segmentation.preserve_all_components:
                # Preserve target-local components selected by the target
                # pipeline, such as paired hands/feet or small disconnected
                # extremity fragments.
                mask = (labels != 0)

            else:
                cx = int((hint_x1 + hint_x2) // 2)
                cy = int((hint_y1 + hint_y2) // 2)
                target = 0
                if 0 <= cx < w and 0 <= cy < h:
                    target = int(labels[cy, cx])

                if target == 0:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    target = 1 + int(np.argmax(areas))

                mask = (labels == target)

        if shape_part_masks is not None:
            shape_mask = cleanup_shape_mask_by_parts(
                mask,
                shape_part_masks,
                **cfg.shape_cleanup,
            )
        else:
            shape_mask = cleanup_shape_mask(mask, **cfg.shape_cleanup)
        shape_mask_u8 = shape_mask.astype(np.uint8) * 255
        debug_mask = shape_mask.copy()

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        crop_x1 = 0
        crop_y1 = 0
        crop_x2 = w
        crop_y2 = h

        if cfg.mode == 'default' and cfg.crop_mode is not None:
            if cfg.crop_mode.mode != 'full_frame':
                crop_x1, crop_y1, crop_x2, crop_y2 = tight_mask_bbox(
                    shape_mask.astype(np.uint8)
                )
                crop_x1, crop_y1, crop_x2, crop_y2 = expand_clip_bbox_by_size_expr(
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2,
                    w,
                    h,
                    cfg.box_margin,
                )

                if (
                    cfg.crop_mode.mode == 'bbox'
                    and cfg.crop_mode.ratio is not None
                ):
                    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox_toward_ratio(
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                        full_w=w,
                        full_h=h,
                        ratio=cfg.crop_mode.ratio,
                    )

        out_x1 = int(crop_x1)
        out_y1 = int(crop_y1)
        out_x2 = int(crop_x2)
        out_y2 = int(crop_y2)

        if cfg.mode == 'default':
            crop_rgb = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2, :]
            crop_h = int(crop_y2 - crop_y1)
            crop_w = int(crop_x2 - crop_x1)

            if cfg.crop_mode is None:
                raise ValueError(
                    f"{self.id}: crop_mode must be defined when mode='default'"
                )

            if cfg.crop_mode.mode == 'bbox':
                # Include the original background inside the crop; alpha is fully opaque.
                alpha = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                crop_rgba = np.dstack([crop_rgb, alpha])
                Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

            else:
                # 'trim' and 'full_frame' -> alpha of mask
                alpha = shape_mask_u8[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_rgba = np.dstack([crop_rgb, alpha])

                if cfg.crop_mode.mode == 'trim':
                    Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

                elif cfg.crop_mode.mode == 'full_frame':
                    full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    full_rgba[crop_y1:crop_y2, crop_x1:crop_x2, :] = crop_rgba

                    # For full-frame outputs, crop metadata must describe the output image
                    # itself in source-image coordinates.
                    out_x1 = 0
                    out_y1 = 0
                    out_x2 = w
                    out_y2 = h

                    Image.fromarray(full_rgba, mode='RGBA').save(out_path)

                else:
                    raise ValueError(
                        f'{self.id}: invalid crop_mode={cfg.crop_mode!r}'
                    )

        else:
            full_mask = prepare_output_mask(
                shape_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                full_mask = 255 - full_mask

            Image.fromarray(full_mask, mode='L').save(out_path)

        # Optional debug bbox overlay
        dbg_path = None
        if cfg.save_debug:
            dbg_path = write_crop_debug_overlay(
                img_rgb=img_rgb,
                out_path=out_path,
                target=cfg.target,
                pose_xy=pose_xy,
                sam_prompt_bbox=segmentation.prompt_bbox,
                target_bbox=segmentation.target_bbox,
                hands_res=segmentation.hands_result,
                face_xy=segmentation.face_xy,
                mask=debug_mask,
                sam_regions=segmentation.sam_regions,
                prompt_bbox_label=segmentation.prompt_bbox_label,
                region_masks=segmentation.debug_region_masks,
                edge_masks=segmentation.debug_edge_masks,
            )

        b_width = int(out_x2 - out_x1)
        b_height = int(out_y2 - out_y1)
        anchor_x = int((out_x1 + out_x2) // 2)
        anchor_y = int((out_y1 + out_y2) // 2)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'model': {
                **({} if cfg.yolo_model is None else {'yolo': cfg.yolo_model}),
                **({} if sam_model_id is None else {
                    'sam_model': sam_model_id,
                }),
                **({} if cfg.face_landmarker_task is None else {
                    'face_landmarker_task': cfg.face_landmarker_task,
                }),
                **({} if cfg.pose_landmarker_task is None else {
                    'pose_landmarker_task': cfg.pose_landmarker_task,
                }),
                **({} if cfg.hand_landmarker_task is None else {
                    'hand_landmarker_task': cfg.hand_landmarker_task,
                }),
                'device': cfg.device,
                'dtype': str(cfg.dtype).replace('torch.', ''),
            },
            'params': {
                'target': cfg.target,
                'mode': cfg.mode,
                'crop_mode': None if cfg.crop_mode is None else cfg.crop_mode.raw,
                'conf': cfg.conf,
                'box_margin': cfg.box_margin,
                'prompt_expansion': cfg.prompt_expansion,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
                'min_landmark_fraction': cfg.min_landmark_fraction,
                'postprocess': cfg.shape_cleanup,
            },
            'crop': {
                'anchor_xy': [
                    int(anchor_x - out_x1),
                    int(anchor_y - out_y1),
                ],
                'position': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
                'bbox_xyxy': [int(out_x1), int(out_y1), int(out_x2), int(out_y2)],
            }
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
