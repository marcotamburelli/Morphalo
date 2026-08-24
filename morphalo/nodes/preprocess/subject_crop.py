import math
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import get_mediapipe_pose_landmarker, get_yolo
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.crop_debug import (
    CropDebugEdgeOverlay, CropDebugMaskOverlay, CropDebugRegion,
    LimbTopologyDebugImage, write_crop_debug_overlay,
    write_limb_topology_debug_directory, write_mask_debug_overlay)
from morphalo.nodes.preprocess.utils import (CropModeSpec, SizeExpr,
                                             expand_bbox_toward_ratio,
                                             parse_crop_mode,
                                             read_shape_cleanup_config,
                                             validate_size_expr)
from morphalo.nodes.preprocess.utils.canny_edge_ops import (
    CannyBarrierSet, CannyEdgeMap, CannyPairExtensionDebug, RegionMosaic,
    bridge_consistent_edge_endpoints, build_canny_edge_map,
    build_canny_partition_barriers, build_region_mosaic,
    connect_limb_edge_endpoints, internal_region_labels, neighbors8,
    region_labels_intersecting_mask, region_mask_from_labels,
    thin_binary_edges)
from morphalo.nodes.preprocess.utils.geometry import (
    expand_clip_bbox, expand_clip_bbox_by_size_expr, segment_capsule_mask,
    tight_mask_bbox, union_bboxes_xyxy)
from morphalo.nodes.preprocess.utils.mask_geometry import (
    LandmarkResolver, LimbCropGeometry, ResolvedLandmark,
    build_arm_crop_geometries, build_leg_crop_geometries)
from morphalo.nodes.preprocess.utils.mask_ops import (
    cleanup_shape_mask_by_parts, prepare_output_mask)
from morphalo.nodes.preprocess.utils.mask_selection import (
    filter_mask_components_by_constraint, label_values_intersecting_mask)
from morphalo.nodes.preprocess.utils.sapiens2_seg import (
    SAPIENS2_CLASSES, ParsedSegmentTarget, ResolvedSegmentTarget,
    parse_target_specs, predict_segments, resolve_parsed_target,
    segment_part_masks)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.human import (mp_pose_landmarks_full,
                                         person_bboxes_xyxy,
                                         select_person_bbox_xyxy)

SUBJECT_CROP_TARGETS = {
    'person',
    'head',
    'hands',
    'left-hand',
    'right-hand',
    'feet',
    'left-foot',
    'right-foot',
    'arms',
    'left-arm',
    'right-arm',
    'legs',
    'left-leg',
    'right-leg',
}

SUBJECT_CROP_POSE_TARGETS: frozenset[str] = frozenset({
    'arms',
    'left-arm',
    'right-arm',
    'legs',
    'left-leg',
    'right-leg',
})

_ARM_SEMANTIC_SUPPORT_LABELS = (
    'left-lower-arm',
    'left-upper-arm',
    'right-lower-arm',
    'right-upper-arm',
    'upper-clothing',
    'torso',
)

_LEG_SEMANTIC_SUPPORT_LABELS = (
    'left-lower-leg',
    'left-upper-leg',
    'right-lower-leg',
    'right-upper-leg',
    'apparel',
    'upper-clothing',
    'lower-clothing',
)

_LIMB_LABELS_BY_SIDE = {
    ('arm', 'anatomical-left'): (
        'left-lower-arm',
        'left-upper-arm',
    ),
    ('arm', 'anatomical-right'): (
        'right-lower-arm',
        'right-upper-arm',
    ),
    ('leg', 'anatomical-left'): (
        'left-lower-leg',
        'left-upper-leg',
    ),
    ('leg', 'anatomical-right'): (
        'right-lower-leg',
        'right-upper-leg',
    ),
}


def _skeleton_touched_limb_label_groups(
    touched_label_values: frozenset[int],
    *,
    limb_side: str,
) -> tuple[tuple[int, ...], ...]:
    """
    Group upper/lower Sapiens2 limb labels for the same anatomical limb.

    The raw Sapiens2 map keeps upper and lower limb classes separate. For limb
    topology, those classes describe one anatomical target: if the skeleton
    touches either half, both halves should participate in support filtering and
    semantic boundary extraction.
    """
    touched_values = {int(value) for value in touched_label_values}
    groups: list[tuple[int, ...]] = []
    grouped_values: set[int] = set()

    for (_target_kind, side), label_names in sorted(_LIMB_LABELS_BY_SIDE.items()):
        if side != limb_side:
            continue

        label_values = tuple(
            int(SAPIENS2_CLASSES[label_name])
            for label_name in label_names
        )
        if not any(value in touched_values for value in label_values):
            continue

        groups.append(tuple(sorted(label_values)))
        grouped_values.update(label_values)

    for value in sorted(touched_values - grouped_values):
        groups.append((int(value),))

    return tuple(groups)


MEDIAPIPE_POSE_LANDMARKS: dict[str, int] = {
    'nose': 0,
    'left_eye': 2,
    'right_eye': 5,
    'left_ear': 7,
    'right_ear': 8,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_big_toe': 31,
    'right_big_toe': 32,
}


# Minimum reliable difference between interpolated MediaPipe image-space depths.
# Smaller values make front/back classification more sensitive but also more
# vulnerable to pose-depth noise.
LIMB_DEPTH_MIN_DELTA = 0.02

# Local tube relief is deliberately bounded: skeleton-label boundaries may
# justify widening the geometric tube, but only while they remain close to its
# boundary.
_LIMB_WORKSPACE_RELIEF_MAX_DISTANCE_RADIUS_RATIO = 0.35
_LIMB_WORKSPACE_RELIEF_HARD_DISTANCE_RADIUS_RATIO = 0.45
_LIMB_WORKSPACE_RELIEF_DILATION_PX = 4
_LIMB_WORKSPACE_RELIEF_OPPOSITE_SKELETON_EXCLUSION_RADIUS_RATIO = 0.10
_LIMB_WORKSPACE_RELIEF_OPPOSITE_SKELETON_EXCLUSION_MIN_RADIUS_PX = 1
_LIMB_WORKSPACE_RELIEF_BRANCH_LOOKAHEAD_PX = 8
_LIMB_WORKSPACE_RELIEF_MICRO_SPUR_PX = 3


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    yolo_model: str
    segment_model: str
    pose_landmarker_task: Optional[str]
    mode: str
    crop_mode: Optional[CropModeSpec]
    target: str
    parsed_target: ParsedSegmentTarget
    conf: float
    expansion: float
    box_margin: SizeExpr
    search_expansion: float
    dilate_radius: int
    close_radius: int
    smoothing_radius: int
    save_debug: bool
    shape_cleanup: dict[str, Any]


@dataclass(frozen=True)
class PoseContext:
    """
    MediaPipe pose information used by SubjectCrop.

    Parameters
    ----------
    xy : np.ndarray
        Full-frame pose coordinates with shape ``(33, 2)``. Invalid landmarks
        are encoded as ``(-1, -1)``.
    z : np.ndarray
        MediaPipe image-space relative depths with shape ``(33,)``. Smaller
        values are closer to the camera. Invalid depths are encoded as
        ``np.nan``.
    xyz_px : np.ndarray
        Full-frame pseudo-3D image-space coordinates with shape ``(33, 3)``.
        ``x`` and ``y`` are pixel coordinates in the full source image. ``z`` is
        MediaPipe's image-space relative depth scaled to pixels in the pose crop
        width. This is intended for perspective-aware length estimates; the raw
        ``z`` field remains available for relative front/back comparisons.
    """

    xy: np.ndarray
    z: np.ndarray
    xyz_px: np.ndarray


@dataclass(frozen=True)
class HumanParseContext:
    """
    Runtime products shared by the Sapiens2-based subject crop pipeline.

    The context keeps target-specific segmentation helpers focused on mask and
    geometry decisions instead of passing the same model outputs through long
    parameter lists.
    """
    node_id: str
    cfg: Config
    img_rgb: np.ndarray
    width: int
    height: int
    person_bbox: tuple[int, int, int, int]
    pose: Optional[PoseContext]
    segments: np.ndarray
    runtime_dtype: torch.dtype


@dataclass(frozen=True)
class TargetSegmentationResult:
    """
    Segmentation result consumed by the shared output/crop finalization.
    """
    mask: np.ndarray
    target_bbox: tuple[int, int, int, int]
    labels: list[str]
    label_ids: list[int]
    selected_candidates: list[str]
    side_resolutions: list[dict[str, Any]]
    prompt_bbox: Optional[tuple[int, int, int, int]] = None
    shape_part_masks: Optional[list[np.ndarray]] = None
    crop_regions: Optional[list[CropDebugRegion]] = None
    prompt_bbox_label: str = 'person'
    debug_region_masks: Optional[list[np.ndarray]] = None
    debug_edge_masks: Optional[list[np.ndarray]] = None
    limb_debug: Optional['LimbSegmentationDebug'] = None


@dataclass(frozen=True)
class TubeEdgeIntersections:
    """
    Logical contacts between a one-pixel edge graph and a tube mask.

    Parameters
    ----------
    contact_labels : np.ndarray
        Integer label map aligned with the input masks. Zero denotes pixels that
        do not belong to a contact. Positive values identify distinct logical
        contacts between the edge graph and the tube boundary.

    representative_points : dict[int, tuple[int, int]]
        Mapping from each positive contact ID to one deterministic outside-edge
        traversal seed in image-array ``(y, x)`` order.
    """

    contact_labels: np.ndarray
    representative_points: dict[int, tuple[int, int]]


def _mask_boundary_inside_workspace(
    mask: np.ndarray,
    *,
    workspace_mask: np.ndarray,
) -> np.ndarray:
    """
    Return the one-pixel inner boundary of ``mask`` inside ``workspace_mask``.

    This is the common boundary primitive used for semantic barriers and for
    skeleton-label visual edges. Keeping it shared prevents the two boundary
    categories from drifting apart morphologically.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional boolean or binary mask whose inner boundary should be
        extracted.

    workspace_mask : np.ndarray
        Two-dimensional boolean or binary workspace. Boundary pixels outside
        this workspace are discarded.

    Returns
    -------
    np.ndarray
        Boolean boundary mask aligned with the inputs.
    """
    source = np.asarray(mask).astype(bool)
    workspace = np.asarray(workspace_mask).astype(bool)
    if source.ndim != 2 or workspace.ndim != 2:
        raise ValueError('mask and workspace_mask must be HxW')
    if source.shape != workspace.shape:
        raise ValueError('mask and workspace_mask must align')

    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    eroded = cv2.erode(
        source.astype(np.uint8),
        boundary_kernel,
        iterations=1,
    ) > 0
    return source & ~eroded & workspace


def _ordered_contour_arc(
    contour_xy: np.ndarray,
    *,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return both ordered arcs of one closed contour between two points.

    Parameters
    ----------
    contour_xy : np.ndarray
        Closed contour points with shape ``(N, 2)`` in ``(x, y)`` order.
    start_xy : tuple[int, int]
        First endpoint in ``(x, y)`` order.
    end_xy : tuple[int, int]
        Second endpoint in ``(x, y)`` order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The two possible contour arcs, both including endpoints. Empty arrays are
        returned when the contour is too small to describe a closed boundary.
    """
    points = np.asarray(contour_xy, dtype=np.int32).reshape(-1, 2)
    if points.shape[0] < 2:
        empty = np.zeros((0, 2), dtype=np.int32)
        return empty, empty

    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    start_index = int(np.argmin(np.sum(
        (points.astype(np.float32) - start) ** 2,
        axis=1,
    )))
    end_index = int(np.argmin(np.sum(
        (points.astype(np.float32) - end) ** 2,
        axis=1,
    )))

    if start_index <= end_index:
        forward = points[start_index:end_index + 1]
        backward = np.concatenate((
            points[end_index:],
            points[:start_index + 1],
        ), axis=0)[::-1]
    else:
        forward = np.concatenate((
            points[start_index:],
            points[:end_index + 1],
        ), axis=0)
        backward = points[end_index:start_index + 1][::-1]

    return forward.astype(np.int32), backward.astype(np.int32)


def _fill_relief_between_path_and_tube(
    *,
    path_yx: list[tuple[int, int]],
    tube_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill the smaller area enclosed by an external edge path and the tube border.

    Parameters
    ----------
    path_yx : list[tuple[int, int]]
        External edge path connecting two tube-boundary intersections, expressed
        in image-array ``(y, x)`` order.
    tube_mask : np.ndarray
        Boolean tube mask in the same local coordinate system.

    Returns
    -------
    np.ndarray
        Boolean relief patch outside ``tube_mask``. Empty when the path cannot be
        closed against one ordered tube contour.
    """
    tube = np.asarray(tube_mask).astype(bool)
    if tube.ndim != 2:
        raise ValueError('tube_mask must be HxW')
    if len(path_yx) < 2 or not np.any(tube):
        return np.zeros_like(tube, dtype=bool)

    contours, _ = cv2.findContours(
        tube.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return np.zeros_like(tube, dtype=bool)

    start_y, start_x = path_yx[0]
    end_y, end_x = path_yx[-1]
    path_xy = np.asarray(
        [(int(x), int(y)) for y, x in path_yx],
        dtype=np.int32,
    )

    best_patch: Optional[np.ndarray] = None
    best_area: Optional[int] = None

    for contour in contours:
        contour_xy = contour.reshape(-1, 2)
        arc_a, arc_b = _ordered_contour_arc(
            contour_xy,
            start_xy=(int(start_x), int(start_y)),
            end_xy=(int(end_x), int(end_y)),
        )

        for arc in (arc_a, arc_b):
            if arc.shape[0] < 2:
                continue
            polygon = np.concatenate((path_xy, arc[::-1]), axis=0)
            if polygon.shape[0] < 3:
                continue

            filled = np.zeros_like(tube, dtype=np.uint8)
            cv2.fillPoly(
                filled,
                [polygon.reshape(-1, 1, 2)],
                255,
                lineType=cv2.LINE_8,
            )
            patch = (filled > 0) & ~tube
            area = int(np.count_nonzero(patch))
            if area <= 0:
                continue
            if best_area is None or area < best_area:
                best_area = area
                best_patch = patch

    if best_patch is None:
        return np.zeros_like(tube, dtype=bool)

    return best_patch


def _edge_branch_tube_distance_score(
    *,
    edge_mask: np.ndarray,
    tube_mask: np.ndarray,
    tube_distance: np.ndarray,
    start_yx: tuple[int, int],
    previous_yx: tuple[int, int],
    max_steps: int,
) -> tuple[float, float]:
    """
    Score a candidate edge branch by how close it stays to the tube.

    Parameters
    ----------
    edge_mask : np.ndarray
        Boolean traversable edge graph.
    tube_mask : np.ndarray
        Boolean tube mask. Branch lookahead only follows pixels outside it.
    tube_distance : np.ndarray
        Distance transform whose values measure distance from the tube.
    start_yx : tuple[int, int]
        First candidate branch pixel in ``(y, x)`` order.
    previous_yx : tuple[int, int]
        Pixel immediately preceding ``start_yx``.
    max_steps : int
        Maximum number of pixels sampled along the branch.

    Returns
    -------
    tuple[float, float]
        ``(best_distance, final_distance)``. Lower values mean the branch remains
        closer to the tube and is preferred at bifurcations.
    """
    current = start_yx
    previous = previous_yx
    visited = {previous, current}
    distances = [float(tube_distance[current])]

    for _ in range(max(0, int(max_steps))):
        candidates = [
            neighbor
            for neighbor in neighbors8(
                current[0],
                current[1],
                shape=edge_mask.shape,
            )
            if (
                neighbor != previous
                and neighbor not in visited
                and edge_mask[neighbor]
                and not tube_mask[neighbor]
            )
        ]
        if not candidates:
            break
        candidates.sort(key=lambda point: float(tube_distance[point]))
        previous = current
        current = candidates[0]
        visited.add(current)
        distances.append(float(tube_distance[current]))

    return min(distances), distances[-1]


def _edge_branch_has_min_unvisited_length(
    *,
    edge_mask: np.ndarray,
    tube_mask: np.ndarray,
    start_yx: tuple[int, int],
    previous_yx: tuple[int, int],
    blocked_yx: set[tuple[int, int]],
    min_steps: int,
) -> bool:
    """
    Return whether a branch continues beyond a tiny rasterization spur.
    """
    min_steps = max(0, int(min_steps))
    if min_steps <= 0:
        return True

    edge = np.asarray(edge_mask).astype(bool)
    tube = np.asarray(tube_mask).astype(bool)

    if not edge[start_yx] or tube[start_yx] or start_yx in blocked_yx:
        return False

    queue: deque[tuple[tuple[int, int], int]] = deque([(start_yx, 1)])
    visited = {previous_yx, start_yx}

    while queue:
        current, depth = queue.popleft()
        if depth >= min_steps:
            return True

        for neighbour in neighbors8(
            current[0],
            current[1],
            shape=edge.shape,
        ):
            if (
                neighbour in visited
                or neighbour in blocked_yx
                or not edge[neighbour]
                or tube[neighbour]
            ):
                continue
            visited.add(neighbour)
            queue.append((neighbour, depth + 1))

    return False


def _find_edge_tube_intersections(
    *,
    edge_mask: np.ndarray,
    tube_mask: np.ndarray,
) -> TubeEdgeIntersections:
    """
    Find outside edge endpoints that touch the tube boundary.

    The tube area acts as a cutter for the semantic edge graph. Contacts are
    true 8-connected endpoints of the outside edge graph whose local
    neighbourhood touches the one-pixel tube perimeter.

    Parameters
    ----------
    edge_mask : np.ndarray
        Two-dimensional boolean or binary mask containing the one-pixel edge
        graph.

    tube_mask : np.ndarray
        Two-dimensional boolean or binary anatomical tube mask aligned with
        ``edge_mask``.

    Returns
    -------
    TubeEdgeIntersections
        Contact label map and one outside traversal seed per logical contact.

    Raises
    ------
    ValueError
        If the masks are not two-dimensional or are not shape-aligned.
    """
    edge = np.asarray(edge_mask).astype(bool)
    tube = np.asarray(tube_mask).astype(bool)

    if edge.ndim != 2 or tube.ndim != 2:
        raise ValueError('edge_mask and tube_mask must be HxW')
    if edge.shape != tube.shape:
        raise ValueError('edge_mask and tube_mask must align')

    empty_labels = np.zeros(edge.shape, dtype=np.int32)

    tube_edges = _mask_boundary_inside_workspace(
        tube,
        workspace_mask=np.ones_like(tube, dtype=bool),
    )

    outside_edges = edge & ~tube
    if not np.any(tube_edges) or not np.any(outside_edges):
        return TubeEdgeIntersections(
            contact_labels=empty_labels,
            representative_points={},
        )

    contact_mask = np.zeros(edge.shape, dtype=bool)
    ys, xs = np.where(outside_edges)
    for y, x in zip(ys.tolist(), xs.tolist()):
        neighbours = neighbors8(y, x, shape=edge.shape)
        outside_degree = sum(
            bool(outside_edges[point]) for point in neighbours)
        if outside_degree > 1:
            continue
        if any(bool(tube_edges[point]) for point in neighbours):
            contact_mask[int(y), int(x)] = True

    if not np.any(contact_mask):
        return TubeEdgeIntersections(
            contact_labels=empty_labels,
            representative_points={},
        )

    component_count, contact_labels = cv2.connectedComponents(
        contact_mask.astype(np.uint8),
        connectivity=8,
    )

    representative_points: dict[int, tuple[int, int]] = {}

    for contact_id in range(1, component_count):
        ys, xs = np.where(contact_labels == contact_id)
        candidates = [
            (int(y), int(x))
            for y, x in zip(ys, xs)
            if contact_mask[int(y), int(x)]
        ]
        if not candidates:
            continue

        representative_points[contact_id] = min(candidates)

    return TubeEdgeIntersections(
        contact_labels=contact_labels.astype(np.int32),
        representative_points=representative_points,
    )


def _trace_external_edge_path_from_tube_intersection(
    *,
    edge_mask: np.ndarray,
    tube_mask: np.ndarray,
    contact_labels: np.ndarray,
    open_contact_ids: set[int],
    start_yx: tuple[int, int],
    start_contact_id: int,
    branch_lookahead_px: int,
) -> tuple[list[tuple[int, int]], Optional[int]]:
    """
    Follow an external edge path until it reaches another logical tube contact.

    Parameters
    ----------
    edge_mask : np.ndarray
        Boolean traversable edge graph.

    tube_mask : np.ndarray
        Boolean geometric tube. Traversal follows edge pixels outside it.

    contact_labels : np.ndarray
        Integer contact-label map. Zero denotes ordinary edge pixels; positive
        values identify logical edge/tube contacts.

    open_contact_ids : set[int]
        Contact IDs that have not yet been consumed by an accepted path.

    start_yx : tuple[int, int]
        Outside-edge traversal seed for the starting contact, expressed in
        image-array ``(y, x)`` order.

    start_contact_id : int
        Positive ID of the starting logical contact.

    branch_lookahead_px : int
        Number of pixels used to choose the branch that tends to remain closest
        to or return toward the tube.

    Returns
    -------
    tuple[list[tuple[int, int]], int | None]
        Traversed external path and reached logical contact ID. ``None``
        indicates that no valid reconnection was found.
    """
    edge = np.asarray(edge_mask).astype(bool)
    tube = np.asarray(tube_mask).astype(bool)
    contacts = np.asarray(contact_labels, dtype=np.int32)

    if not (
        edge.shape
        == tube.shape
        == contacts.shape
    ):
        raise ValueError(
            'edge_mask, tube_mask, and contact_labels must align'
        )

    if start_contact_id <= 0:
        raise ValueError('start_contact_id must be positive')
    if not edge[start_yx] or tube[start_yx]:
        return [], None
    if int(contacts[start_yx]) != start_contact_id:
        return [], None

    tube_distance = cv2.distanceTransform(
        (~tube).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )

    current = start_yx
    previous: Optional[tuple[int, int]] = None
    path = [start_yx]
    visited = {start_yx}

    while True:
        current_contact_id = int(contacts[current])
        if (
            current_contact_id > 0
            and current_contact_id != start_contact_id
            and current_contact_id in open_contact_ids
        ):
            return path, current_contact_id

        candidates = [
            neighbour
            for neighbour in neighbors8(
                current[0],
                current[1],
                shape=edge.shape,
            )
            if (
                neighbour != previous
                and neighbour not in visited
                and edge[neighbour]
                and not tube[neighbour]
            )
        ]

        if not candidates:
            return path, None

        if len(candidates) > 1:
            continuing_candidates = [
                candidate
                for candidate in candidates
                if (
                    int(contacts[candidate]) in open_contact_ids
                    or _edge_branch_has_min_unvisited_length(
                        edge_mask=edge,
                        tube_mask=tube,
                        start_yx=candidate,
                        previous_yx=current,
                        blocked_yx=visited,
                        min_steps=_LIMB_WORKSPACE_RELIEF_MICRO_SPUR_PX,
                    )
                )
            ]
            if continuing_candidates:
                candidates = continuing_candidates

        candidates.sort(key=lambda point: (
            _edge_branch_tube_distance_score(
                edge_mask=edge,
                tube_mask=tube,
                tube_distance=tube_distance,
                start_yx=point,
                previous_yx=current,
                max_steps=branch_lookahead_px,
            ),
            int(point[0]),
            int(point[1]),
        ))

        previous = current
        current = candidates[0]
        path.append(current)
        visited.add(current)


def _build_limb_workspace_relief_from_semantic_edges(
    *,
    semantic_region_mask: np.ndarray,
    tube_mask: np.ndarray,
    base_workspace_mask: np.ndarray,
    skeleton_label_edge_mask: np.ndarray,
    tube_radius: float,
    opposite_skeleton_exclusion_mask: Optional[np.ndarray] = None,
    rejected_component_accumulator: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Expand a geometric tube where skeleton-label borders leave and re-enter it.

    Parameters
    ----------
    semantic_region_mask : np.ndarray
        Semantic support region aligned with the tube and edge masks.
    tube_mask : np.ndarray
        Pure geometric tube mask.
    base_workspace_mask : np.ndarray
        Initial workspace ``tube | proximal_cap``. Relief is added to this later
        but is not treated as part of the tube itself.
    skeleton_label_edge_mask : np.ndarray
        Thinned one-pixel boundaries of the semantic label values touched by the
        anatomical skeleton.
    tube_radius : float
        Geometric tube radius used to bound how far relief paths may leave the
        tube.

    Returns
    -------
    np.ndarray
        Boolean workspace-relief mask aligned with the inputs. It is empty when no
        external skeleton-label border path reconnects two tube-boundary contacts
        within the allowed distance.
    """
    semantic_region = np.asarray(semantic_region_mask).astype(bool)
    tube = np.asarray(tube_mask).astype(bool)
    base_workspace = np.asarray(base_workspace_mask).astype(bool)
    label_edges = np.asarray(skeleton_label_edge_mask).astype(bool)
    opposite_exclusion = (
        None
        if opposite_skeleton_exclusion_mask is None
        else np.asarray(opposite_skeleton_exclusion_mask).astype(bool)
    )
    rejected_accumulator = rejected_component_accumulator

    if not (
        semantic_region.shape
        == tube.shape
        == base_workspace.shape
        == label_edges.shape
    ):
        raise ValueError(
            'semantic_region_mask, tube_mask, base_workspace_mask, and '
            'skeleton_label_edge_mask must align'
        )
    if opposite_exclusion is not None and opposite_exclusion.shape != tube.shape:
        raise ValueError(
            'opposite_skeleton_exclusion_mask must align with relief masks'
        )
    if (
        rejected_accumulator is not None
        and rejected_accumulator.shape != tube.shape
    ):
        raise ValueError(
            'rejected_component_accumulator must align with relief masks'
        )
    opposite_exclusion_mask = (
        np.zeros_like(tube, dtype=bool)
        if opposite_exclusion is None
        else opposite_exclusion
    )
    if not np.any(semantic_region) or not np.any(tube):
        return np.zeros_like(tube, dtype=bool)

    if not np.any(label_edges):
        return np.zeros_like(tube, dtype=bool)

    semantic_edges = label_edges & semantic_region

    intersections = _find_edge_tube_intersections(
        edge_mask=semantic_edges,
        tube_mask=tube,
    )

    if len(intersections.representative_points) < 2:
        return np.zeros_like(tube, dtype=bool)

    max_distance_px = max(
        1.0,
        float(tube_radius) * _LIMB_WORKSPACE_RELIEF_MAX_DISTANCE_RADIUS_RATIO,
    )
    hard_max_distance_px = max(
        max_distance_px,
        float(tube_radius) * _LIMB_WORKSPACE_RELIEF_HARD_DISTANCE_RADIUS_RATIO,
    )

    open_contact_ids = set(intersections.representative_points)
    relief = np.zeros_like(tube, dtype=bool)
    tube_distance = cv2.distanceTransform(
        (~tube).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    relief_dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * _LIMB_WORKSPACE_RELIEF_DILATION_PX + 1,
            2 * _LIMB_WORKSPACE_RELIEF_DILATION_PX + 1,
        ),
    )

    while open_contact_ids:
        start_contact_id = min(open_contact_ids)
        open_contact_ids.discard(start_contact_id)

        start_yx = intersections.representative_points[start_contact_id]

        path, end_contact_id = (
            _trace_external_edge_path_from_tube_intersection(
                edge_mask=semantic_edges,
                tube_mask=tube,
                contact_labels=intersections.contact_labels,
                open_contact_ids=open_contact_ids,
                start_yx=start_yx,
                start_contact_id=start_contact_id,
                branch_lookahead_px=(
                    _LIMB_WORKSPACE_RELIEF_BRANCH_LOOKAHEAD_PX
                ),
            )
        )

        if end_contact_id is None:
            continue

        path_distances = np.asarray(
            [float(tube_distance[point]) for point in path],
            dtype=np.float32,
        )
        path_p95_distance = float(np.percentile(path_distances, 95))
        path_max_distance = float(np.max(path_distances))

        open_contact_ids.discard(end_contact_id)

        patch = _fill_relief_between_path_and_tube(
            path_yx=path,
            tube_mask=tube,
        )
        patch_allowed_mask = np.ones_like(tube, dtype=bool)
        if path_max_distance > hard_max_distance_px:
            patch_allowed_mask &= tube_distance <= hard_max_distance_px
        if path_p95_distance > max_distance_px:
            patch_allowed_mask &= tube_distance <= max_distance_px
        patch &= patch_allowed_mask & semantic_region
        if not np.any(patch):
            continue

        if np.any(patch & opposite_exclusion_mask):
            _, patch_component_labels = cv2.connectedComponents(
                patch.astype(np.uint8),
                connectivity=8,
            )
            filtered_patch = np.zeros_like(patch, dtype=bool)
            for component_id in np.unique(patch_component_labels):
                if int(component_id) == 0:
                    continue
                component = patch_component_labels == int(component_id)
                if np.any(component & opposite_exclusion_mask):
                    if rejected_accumulator is not None:
                        rejected_accumulator |= component
                    continue
                filtered_patch |= component
            patch = filtered_patch
            if not np.any(patch):
                continue

        patch = cv2.dilate(
            patch.astype(np.uint8),
            relief_dilation_kernel,
            iterations=1,
        ) > 0
        relief |= patch & patch_allowed_mask

    if not np.any(relief):
        return relief

    # Relief widens the topological workspace only where it remains supported by
    # semantic target labels; reconstruction is clipped by core_prior later.
    return relief & semantic_region & ~base_workspace


@dataclass(frozen=True)
class LimbCannyDebug:
    """
    Optional Canny diagnostics captured during one limb partition pass.

    ``endpoint_debug`` contains local endpoint classifications and accepted
    synthetic edge segments returned by ``connect_limb_edge_endpoints``. It is
    intentionally kept out of core logic and consumed only by debug renderers.
    """
    endpoint_debug: Optional[CannyPairExtensionDebug] = None


@dataclass(frozen=True)
class LimbSegmentationDebug:
    """
    Debug data collected by the limb orchestrator for directory rendering.
    """
    target_kind: Literal['arm', 'leg']
    selected_sides: tuple[str, ...]
    geometries_by_side: dict[str, LimbCropGeometry]
    cores_by_side: dict[str, 'LimbCoreExtraction']
    extensions_by_side: dict[str, 'LimbProximalExtension']
    core_subtractions_by_side: dict[str, 'LimbOverlapSubtraction']
    post_subtractions_by_side: dict[str, 'LimbOverlapSubtraction']
    final_masks_by_side: dict[str, np.ndarray]


@dataclass(frozen=True)
class LimbPartitionContext:
    """
    Local masks and geometry required to partition one anatomical limb.

    The context normalizes full-frame limb geometry into the local crop
    coordinate system used by Canny extraction and region topology.

    Core topology uses three distinct masks:

    ``core_workspace = tube_mask | proximal_cap_mask | workspace_relief_mask``
        Geometric free-space domain. Its boundary defines external regions.

    ``core_prior = semantic_support_mask & core_workspace``
        Reconstruction prior. Canny extraction and the final selected core are
        clipped to this mask.

    ``semantic_barrier = boundary(semantic_region_mask) & core_workspace``
        Internal barrier from the broad, unclipped semantic region. It closes
        the semantic support without turning the geometry-clipped core-prior
        boundary into the external workspace boundary.

    ``skeleton_label_support_masks = semantic labels touched by skeleton``
        One semantic-support mask per Sapiens2 label value touched by the
        anatomical skeleton. Their separate boundaries are injected into the
        visual edge map so the mosaic preserves boundaries between
        skeleton-anchored semantic classes such as skin and clothing without
        duplicating them as synthetic barriers.

    Parameters
    ----------
    crop_bbox : tuple[int, int, int, int]
        End-exclusive full-frame crop box ``(x1, y1, x2, y2)``.

    source_rgb : np.ndarray
        Local RGB source image aligned with every local mask.

    semantic_support_mask : np.ndarray
        Local Sapiens2 semantic support associated with this anatomical limb.
        It is clipped by geometry before this context is built, but landmark-
        chain component filtering is applied later to selected topological
        regions so Canny can still see useful clothing/skin boundaries.

    tube_mask : np.ndarray
        Local anatomical tube limiting the limb core.

    workspace_relief_mask : np.ndarray
        Local semantic-edge-driven workspace expansion. The tube remains pure
        geometry; relief only widens the topological workspace where a refined
        edge path leaves and re-enters the tube close to its border.

    core_prior_mask : np.ndarray
        Local limb core prior defined as the semantic support intersected with
        the union of the anatomical tube, proximal cap, and workspace relief.
    core_workspace_mask : np.ndarray
        Local topological workspace used for core region classification.

        The workspace is the geometric tube-plus-proximal-cap envelope plus
        local semantic-edge relief. Its boundary defines which free-space regions
        are external. Semantic support is added as an internal barrier and as
        final reconstruction prior, but it does not define the external boundary.

    proximal_cap_mask : np.ndarray
        Local proximal cap included in the core geometric envelope. Its synthetic
        boundary contributes the proximal closure, while selected undilated regions
        inside the cap later provide the continuity reference for proximal
        extension.

    proximal_half_plane_mask : np.ndarray
        Local proximal half-plane or quadrant available for proximal extension.

    distal_barrier_mask : np.ndarray
        Local synthetic distal closure.

    proximal_barrier_mask : np.ndarray
        Local synthetic proximal closure lying on the cap boundary.

        It closes the core mosaic but is deliberately excluded from the proximal
        extension mosaic so cap regions can continue into the outward quadrant.

    semantic_barrier_mask : np.ndarray
        Local boundary extracted from the broad semantic region inside the
        geometric workspace.

    skeleton_label_support_masks : tuple[np.ndarray, ...]
        Local semantic-support masks for each Sapiens2 label value intersected
        by the anatomical skeleton. Each entry may be disconnected, but label
        values are kept separate so their mutual boundaries remain available.

    skeleton_label_edge_mask : np.ndarray
        Thinned union of the separate boundaries of
        ``skeleton_label_support_masks`` inside ``core_workspace_mask``. It is
        injected into the visual edge map before Canny bridge/prolongation and
        is not added to synthetic barriers.

    tube_edge_mask : np.ndarray
        One-pixel local boundary of ``tube_mask`` used by workspace-relief
        contact detection.

    skeleton_label_edge_wide_mask : np.ndarray
        Union of the skeleton-touched label boundaries before clipping to the
        final workspace. This is the raw semantic edge input to workspace relief.

    skeleton_label_edge_thin_mask : np.ndarray
        Thinned semantic label edges used by workspace relief.

    outside_skeleton_label_edge_mask : np.ndarray
        Thinned semantic edge pixels outside the tube and inside the broad
        semantic region. Relief tracing walks on this mask.

    relief_contact_mask : np.ndarray
        Semantic edge pixels adjacent to the one-pixel tube boundary.

    relief_contact_endpoint_mask : np.ndarray
        Representative outside endpoint pixels selected from ``relief_contact``.

    relief_opposite_skeleton_exclusion_mask : np.ndarray
        Local dilated opposite-limb skeleton mask used to reject relief patch
        components.

    relief_rejected_component_mask : np.ndarray
        Local relief patch components rejected because they intersected
        ``relief_opposite_skeleton_exclusion_mask``.

    skeleton_mask : np.ndarray
        Local anatomical skeleton used only for Canny refinement.

    anatomical_segments : tuple[tuple[np.ndarray, np.ndarray], ...]
        Valid local proximal-to-distal anatomical segments.

    proximal_point : np.ndarray | None
        Local proximal landmark.

    middle_point : np.ndarray | None
        Local middle landmark.

    distal_point : np.ndarray | None
        Local distal landmark.

    limb_chain_length : float
        Full proximal-to-distal landmark-chain length in pixels.

    """

    crop_bbox: tuple[int, int, int, int]
    source_rgb: np.ndarray

    semantic_support_mask: np.ndarray
    tube_mask: np.ndarray
    workspace_relief_mask: np.ndarray
    core_prior_mask: np.ndarray
    core_workspace_mask: np.ndarray

    proximal_cap_mask: np.ndarray
    proximal_half_plane_mask: np.ndarray

    distal_barrier_mask: np.ndarray
    proximal_barrier_mask: np.ndarray
    semantic_barrier_mask: np.ndarray
    skeleton_label_support_masks: tuple[np.ndarray, ...]
    skeleton_label_edge_mask: np.ndarray
    tube_edge_mask: np.ndarray
    skeleton_label_edge_wide_mask: np.ndarray
    skeleton_label_edge_thin_mask: np.ndarray
    outside_skeleton_label_edge_mask: np.ndarray
    relief_contact_mask: np.ndarray
    relief_contact_endpoint_mask: np.ndarray
    relief_opposite_skeleton_exclusion_mask: np.ndarray
    relief_rejected_component_mask: np.ndarray

    skeleton_mask: np.ndarray
    anatomical_segments: tuple[
        tuple[np.ndarray, np.ndarray],
        ...,
    ]

    proximal_point: Optional[np.ndarray]
    middle_point: Optional[np.ndarray]
    distal_point: Optional[np.ndarray]

    limb_chain_length: float


@dataclass(frozen=True)
class LimbCoreExtraction:
    """
    Topological extraction of one anatomical limb core.

    The selected regions are preserved both before and after restoration of the
    thickness occupied by strengthened partition barriers.

    Parameters
    ----------
    region_mosaic : RegionMosaic
        Local free-space topology built inside the closed core workspace.

    selected_region_labels : frozenset[int]
        Labels classified as internal because their free-space regions do not
        touch the boundary of the closed core workspace.

    region_mask : np.ndarray
        Full-frame mask of the selected topological regions before restoration of
        partition-barrier thickness.

        This is the canonical core representation used as the proximal
        continuation source. It may later be reduced by opposite-limb subtraction,
        but it is never replaced by the expanded silhouette.

    expanded_mask : np.ndarray
        Full-frame structural silhouette after restoring partition-barrier
        thickness and applying conservative morphology inside ``prior_mask``.

    barrier_set : CannyBarrierSet
        Local strengthened visual, semantic, and synthetic barriers used to
        construct ``region_mosaic``.

    edge_map : CannyEdgeMap
        Local image-derived Canny products before barrier strengthening.

    refined_edge_mask : np.ndarray
        Full-frame one-pixel visual edge mask after short-gap bridging and
        endpoint extension, before barrier strengthening/dilation.

        Canny pixels originate inside ``prior_mask``. Synthetic bridge and
        extension pixels are allowed inside ``workspace_mask`` so they can create
        topological separators across semantic holes; final reconstruction is
        clipped back to ``prior_mask``.

    prior_mask : np.ndarray
        Full-frame effective core prior, defined as the filtered semantic support
        intersected with the union of the anatomical tube and proximal cap.

    workspace_mask : np.ndarray
        Full-frame workspace used for closed-core topological classification.

        This is the geometric tube-plus-proximal-cap envelope plus any local
        semantic-edge relief. Its boundary defines external free-space regions
        and it may be wider than ``prior_mask``.

    workspace_relief_mask : np.ndarray
        Full-frame semantic-edge-driven workspace expansion added to the pure
        geometric tube/cap envelope before core region topology is built.

    tube_edge_mask : np.ndarray | None
        Full-frame one-pixel tube boundary used by workspace-relief contact
        detection. Present only in debug-producing extraction paths.

    skeleton_label_edge_wide_mask : np.ndarray | None
        Full-frame raw semantic label edges selected from labels touched by the
        limb skeleton before final workspace clipping.

    skeleton_label_edge_thin_mask : np.ndarray | None
        Full-frame thinned semantic label edges used by workspace relief.

    outside_skeleton_label_edge_mask : np.ndarray | None
        Full-frame thinned semantic label edges outside the anatomical tube.

    relief_contact_mask : np.ndarray | None
        Full-frame semantic label edge pixels adjacent to the tube boundary.

    relief_contact_endpoint_mask : np.ndarray | None
        Full-frame representative contact endpoints used to start relief traces.

    relief_opposite_skeleton_exclusion_mask : np.ndarray | None
        Full-frame dilated opposite-limb skeleton mask used to reject relief
        patch components.

    relief_rejected_component_mask : np.ndarray | None
        Full-frame relief patch components rejected because they intersected the
        opposite-limb skeleton exclusion mask.

    proximal_cap_mask : np.ndarray
        Full-frame proximal cap retained for proximal continuity selection.

    proximal_half_plane_mask : np.ndarray
        Full-frame proximal extension domain.

    semantic_support_mask : np.ndarray
        Full-frame filtered semantic support used to constrain both core
        reconstruction and proximal extension.

    canny_debug : LimbCannyDebug | None
        Optional endpoint-level diagnostics captured during visual Canny
        refinement.

    overlap_subtraction : LimbOverlapSubtraction | None
        Optional opposite-limb subtraction applied to this core.
    """

    region_mosaic: RegionMosaic
    selected_region_labels: frozenset[int]

    region_mask: np.ndarray
    expanded_mask: np.ndarray

    barrier_set: CannyBarrierSet
    edge_map: CannyEdgeMap
    refined_edge_mask: np.ndarray

    prior_mask: np.ndarray
    workspace_mask: np.ndarray
    workspace_relief_mask: np.ndarray
    proximal_cap_mask: np.ndarray
    proximal_half_plane_mask: np.ndarray
    semantic_support_mask: np.ndarray
    tube_edge_mask: Optional[np.ndarray] = None
    skeleton_label_edge_wide_mask: Optional[np.ndarray] = None
    skeleton_label_edge_thin_mask: Optional[np.ndarray] = None
    outside_skeleton_label_edge_mask: Optional[np.ndarray] = None
    relief_contact_mask: Optional[np.ndarray] = None
    relief_contact_endpoint_mask: Optional[np.ndarray] = None
    relief_opposite_skeleton_exclusion_mask: Optional[np.ndarray] = None
    relief_rejected_component_mask: Optional[np.ndarray] = None
    canny_debug: Optional[LimbCannyDebug] = None
    overlap_subtraction: Optional['LimbOverlapSubtraction'] = None


@dataclass(frozen=True)
class LimbOverlapSubtraction:
    """
    Subtraction decision produced for one target limb.

    Parameters
    ----------
    subtraction_mask : np.ndarray
        Full-frame target-mask pixels also covered by an opposite-limb mask and by
        an opposite anatomical segment classified as reliably closer to the
        camera.

    changed : bool
        Whether ``subtraction_mask`` contains at least one pixel.

    compared_segment_pairs : int
        Number of capsule-overlapping segment pairs for which depth comparison
        was attempted.

    accepted_segment_pairs : int
        Number of opposite-limb segment pairs classified as reliably in front.
    """

    subtraction_mask: np.ndarray
    changed: bool
    compared_segment_pairs: int
    accepted_segment_pairs: int


@dataclass(frozen=True)
class LimbPostExtensionOverlapResult:
    """
    Result of the post-extension opposite-limb overlap cleanup pass.

    ``masks_by_side`` contains the cleaned selected limb masks used by the
    segmentation pipeline. ``subtractions_by_side`` contains the side-specific
    subtraction decisions that produced those masks and is preserved for debug
    rendering.
    """
    masks_by_side: dict[str, np.ndarray]
    subtractions_by_side: dict[str, LimbOverlapSubtraction]


@dataclass(frozen=True)
class LimbProximalExtension:
    """
    Topological continuation of a limb through its proximal workspace.

    Parameters
    ----------
    region_mosaic : RegionMosaic
        Local free-space topology built inside the proximal workspace.

    selected_region_labels : frozenset[int]
        Proximal-workspace labels intersecting the cleaned, undilated core regions
        inside the proximal cap.

    region_mask : np.ndarray
        Full-frame proximal continuation before barrier-thickness restoration.

    expanded_mask : np.ndarray
        Full-frame proximal continuation after restoring barrier thickness and
        applying conservative morphology inside the proximal workspace.

    barrier_set : CannyBarrierSet
        Local barriers used to partition the proximal workspace.

    edge_map : CannyEdgeMap
        Local Canny products extracted from the proximal workspace.

    workspace_mask : np.ndarray
        Full-frame proximal workspace.

    continuation_mask : np.ndarray
        Full-frame intersection between the cleaned undilated core and the
        proximal cap.

        This mask provides the topological continuity reference used to select
        regions in the proximal workspace.

    canny_debug : LimbCannyDebug | None
        Optional endpoint-level diagnostics captured during proximal Canny
        refinement.
    """

    region_mosaic: RegionMosaic
    selected_region_labels: frozenset[int]

    region_mask: np.ndarray
    expanded_mask: np.ndarray

    barrier_set: CannyBarrierSet
    edge_map: CannyEdgeMap

    workspace_mask: np.ndarray
    continuation_mask: np.ndarray
    canny_debug: Optional[LimbCannyDebug] = None


def _build_limb_partition_context(
    ctx: HumanParseContext,
    *,
    limb_geometry: LimbCropGeometry,
    opposite_limb_skeleton_mask: Optional[np.ndarray] = None,
) -> LimbPartitionContext:
    """
    Build the local partition context for one anatomical limb.

    The geometry builder remains the authority for the anatomical tube,
    skeleton, synthetic closures, proximal cap, and proximal quadrant. This helper
    validates those full-frame products, crops them to ``crop_bbox``, normalizes
    them to boolean masks, derives the local limb core domain, and converts the
    available landmarks to local coordinates.

    The base geometric core workspace is the union of the anatomical tube and
    proximal cap:

    ``tube_mask | proximal_cap_mask``

    A skeleton-label boundary pass can add local ``workspace_relief`` where a
    semantic label border leaves and re-enters the tube close to its border. The
    effective core prior is the portion of the final workspace supported by the
    filtered Sapiens2 semantic mask:

    ``semantic_support_mask & (tube_mask | proximal_cap_mask | workspace_relief)``

    The semantic-support boundary is added as an internal barrier instead of
    becoming the workspace boundary. Final reconstruction is still clipped to the
    effective prior.

    In short:

    ``core_workspace = tube_mask | proximal_cap_mask | workspace_relief``

    ``core_prior = semantic_support_mask & core_workspace``

    ``semantic_barrier = boundary(semantic_region_mask) & core_workspace``

    Parameters
    ----------
    ctx : HumanParseContext
        Full-frame runtime context containing image dimensions and RGB pixels.

    limb_geometry : LimbCropGeometry
        Existing full-frame anatomical limb geometry to localize.

    Returns
    -------
    LimbPartitionContext
        Immutable local context used by Canny extraction, barrier construction,
        and region topology stages.

    Raises
    ------
    RuntimeError
        If ``limb_geometry.crop_bbox`` is outside the frame or empty.

    ValueError
        If any required geometry mask is not aligned with the full frame.

    Notes
    -----
    This helper does not extract Canny edges, refine visual barriers, classify
    topological regions, subtract the opposite limb, or perform proximal
    extension.
    """
    x1, y1, x2, y2 = limb_geometry.crop_bbox
    if x1 < 0 or y1 < 0 or x2 > ctx.width or y2 > ctx.height or x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': invalid limb crop bbox "
            f'{limb_geometry.crop_bbox!r}.'
        )

    shape = (ctx.height, ctx.width)
    for name, mask in (
        ('semantic_support_mask', limb_geometry.semantic_support_mask),
        ('semantic_region_mask', limb_geometry.semantic_region_mask),
        ('tube_mask', limb_geometry.tube_mask),
        ('skeleton_mask', limb_geometry.skeleton_mask),
        ('synthetic_barrier_mask', limb_geometry.synthetic_barrier_mask),
        ('proximal_circle_mask', limb_geometry.proximal_circle_mask),
        ('proximal_quadrant_mask', limb_geometry.proximal_quadrant_mask),
    ):
        if np.asarray(mask).shape != shape:
            raise ValueError(
                f'{name} must match full-frame shape {shape!r}, '
                f'got {np.asarray(mask).shape!r}.'
            )

    opposite_skeleton = (
        None
        if opposite_limb_skeleton_mask is None
        else np.asarray(opposite_limb_skeleton_mask).astype(bool)
    )
    if opposite_skeleton is not None and opposite_skeleton.shape != shape:
        raise ValueError(
            'opposite_limb_skeleton_mask must match full-frame shape '
            f'{shape!r}, got {opposite_skeleton.shape!r}.'
        )

    local_source_rgb = np.ascontiguousarray(ctx.img_rgb[y1:y2, x1:x2, :])
    local_segment_labels = np.asarray(ctx.segments[y1:y2, x1:x2])
    local_semantic_support = (
        limb_geometry.semantic_support_mask[y1:y2, x1:x2]
        .astype(bool)
    )
    local_semantic_region = (
        limb_geometry.semantic_region_mask[y1:y2, x1:x2]
        .astype(bool)
    )
    local_tube = limb_geometry.tube_mask[y1:y2, x1:x2].astype(bool)
    local_skeleton = limb_geometry.skeleton_mask[y1:y2, x1:x2].astype(bool)
    local_synthetic = limb_geometry.synthetic_barrier_mask[y1:y2, x1:x2].astype(
        bool)
    local_proximal_cap = limb_geometry.proximal_circle_mask[y1:y2, x1:x2].astype(
        bool)
    local_proximal_half_plane = (
        limb_geometry.proximal_quadrant_mask[y1:y2, x1:x2]
        .astype(bool)
    )

    # Pipeline stage: derive skeleton-anchored semantic labels. Their boundaries
    # can widen the workspace before becoming final visual edges.
    local_base_workspace = local_tube | local_proximal_cap
    # The skeleton-derived support is selected by all Sapiens2 label values
    # touched by the skeleton inside the broad semantic region, not by connected
    # components of the filtered support mask.
    touched_skeleton_label_values = label_values_intersecting_mask(
        local_semantic_region,
        label_map=local_segment_labels,
        selector_mask=local_skeleton,
    )
    local_skeleton_label_groups = _skeleton_touched_limb_label_groups(
        touched_skeleton_label_values,
        limb_side=limb_geometry.side,
    )
    local_skeleton_label_edges_wide = np.zeros_like(
        local_base_workspace,
        dtype=bool,
    )
    local_skeleton_label_support_masks: list[np.ndarray] = []
    full_skeleton_label_relief_inputs: list[
        tuple[np.ndarray, np.ndarray]
    ] = []
    # Keep label boundaries separate here and derive them from the unclipped
    # semantic region. Using the geometry-clipped prior would add artificial
    # tube/cap edges as if they were semantic label boundaries.
    full_segment_labels = np.asarray(ctx.segments)
    for label_values in local_skeleton_label_groups:
        label_support = (
            local_semantic_region
            & np.isin(local_segment_labels, np.asarray(label_values))
        )
        if not np.any(label_support):
            continue
        local_skeleton_label_support_masks.append(label_support.astype(bool))

        full_label_support = (
            limb_geometry.semantic_region_mask
            & np.isin(full_segment_labels, np.asarray(label_values))
        )
        full_label_edge = _mask_boundary_inside_workspace(
            full_label_support,
            workspace_mask=limb_geometry.semantic_region_mask,
        )

        # Keep the local crop for final visual barriers and diagnostics.
        local_skeleton_label_edges_wide |= (
            full_label_edge[y1:y2, x1:x2]
        )

        # Relief traversal must use the complete label contour. Cropping the edge
        # before tracing can interrupt an external arc before it reconnects to the
        # tube.
        full_label_relief_edge = thin_binary_edges(
            full_label_edge & full_label_support,
        )
        if np.any(full_label_relief_edge):
            full_skeleton_label_relief_inputs.append((
                full_label_support.astype(bool),
                full_label_relief_edge,
            ))

    local_tube_edges = _mask_boundary_inside_workspace(
        local_tube,
        workspace_mask=np.ones_like(local_tube, dtype=bool),
    )

    local_skeleton_label_relief_edges = np.zeros_like(
        local_base_workspace,
        dtype=bool,
    )
    for _, full_label_relief_edge in full_skeleton_label_relief_inputs:
        local_skeleton_label_relief_edges |= (
            full_label_relief_edge[y1:y2, x1:x2]
        )
    local_outside_skeleton_label_edges = (
        local_skeleton_label_relief_edges
        & ~local_tube
    )
    local_relief_contact_mask = np.zeros_like(
        local_base_workspace,
        dtype=bool,
    )
    local_relief_contact_endpoint_mask = np.zeros_like(
        local_base_workspace,
        dtype=bool,
    )

    for _, full_label_relief_edge in full_skeleton_label_relief_inputs:
        relief_intersections = _find_edge_tube_intersections(
            edge_mask=full_label_relief_edge,
            tube_mask=limb_geometry.tube_mask,
        )

        local_relief_contact_mask |= (
            relief_intersections.contact_labels[y1:y2, x1:x2] > 0
        )

        for point_y, point_x in (
            relief_intersections.representative_points.values()
        ):
            if (
                y1 <= point_y < y2
                and x1 <= point_x < x2
            ):
                local_relief_contact_endpoint_mask[
                    point_y - y1,
                    point_x - x1,
                ] = True

    # The geometry builder stores both longitudinal closures in one synthetic
    # mask. Split them so the proximal closure can be excluded from the later
    # proximal-extension mosaic.
    local_proximal_barrier = local_synthetic & local_proximal_cap
    local_distal_barrier = local_synthetic & ~local_proximal_cap

    anatomical_segments: list[tuple[np.ndarray, np.ndarray]] = []

    def to_local_point(point: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if point is None:
            return None
        point_array = np.asarray(point, dtype=np.float32).reshape(-1)
        if point_array.size < 2 or not np.all(np.isfinite(point_array[:2])):
            return None
        return np.asarray(
            [
                float(point_array[0]) - float(x1),
                float(point_array[1]) - float(y1),
            ],
            dtype=np.float32,
        )

    local_proximal_point = to_local_point(limb_geometry.proximal_point)
    local_middle_point = to_local_point(limb_geometry.middle_point)
    local_distal_point = to_local_point(limb_geometry.distal_point)

    if local_proximal_point is not None and local_middle_point is not None:
        anatomical_segments.append((local_proximal_point, local_middle_point))
    if local_middle_point is not None and local_distal_point is not None:
        anatomical_segments.append((local_middle_point, local_distal_point))
    if (
        not anatomical_segments
        and local_proximal_point is not None
        and local_distal_point is not None
    ):
        anatomical_segments.append((local_proximal_point, local_distal_point))

    # Pipeline stage: widen the workspace only where a per-label semantic
    # boundary leaves and re-enters the geometric tube. Running relief per label
    # avoids ambiguous traversal when adjacent label boundaries form parallel
    # contours or shared junctions.
    full_base_workspace = (
        limb_geometry.tube_mask.astype(bool)
        | limb_geometry.proximal_circle_mask.astype(bool)
    )

    full_workspace_relief = np.zeros(
        (ctx.height, ctx.width),
        dtype=bool,
    )
    full_relief_rejected_components = np.zeros_like(
        full_workspace_relief,
        dtype=bool,
    )
    full_opposite_skeleton_exclusion = np.zeros_like(
        full_workspace_relief,
        dtype=bool,
    )
    if opposite_skeleton is not None and np.any(opposite_skeleton):
        exclusion_radius = max(
            int(_LIMB_WORKSPACE_RELIEF_OPPOSITE_SKELETON_EXCLUSION_MIN_RADIUS_PX),
            int(round(
                float(limb_geometry.tube_radius)
                * _LIMB_WORKSPACE_RELIEF_OPPOSITE_SKELETON_EXCLUSION_RADIUS_RATIO
            )),
        )
        if exclusion_radius == 0:
            full_opposite_skeleton_exclusion = opposite_skeleton.copy()
        else:
            exclusion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * exclusion_radius + 1,
                    2 * exclusion_radius + 1,
                ),
            )
            full_opposite_skeleton_exclusion = cv2.dilate(
                opposite_skeleton.astype(np.uint8),
                exclusion_kernel,
                iterations=1,
            ) > 0

    for full_label_support, full_label_relief_edge in (
        full_skeleton_label_relief_inputs
    ):
        full_workspace_relief |= (
            _build_limb_workspace_relief_from_semantic_edges(
                semantic_region_mask=full_label_support,
                tube_mask=limb_geometry.tube_mask,
                base_workspace_mask=full_base_workspace,
                skeleton_label_edge_mask=full_label_relief_edge,
                tube_radius=float(limb_geometry.tube_radius),
                opposite_skeleton_exclusion_mask=(
                    full_opposite_skeleton_exclusion
                ),
                rejected_component_accumulator=(
                    full_relief_rejected_components
                ),
            )
        )

    # Only the relief result is localized. The edge traversal itself has already
    # been completed on the full semantic contour.
    local_workspace_relief = (
        full_workspace_relief[y1:y2, x1:x2]
    )
    local_relief_opposite_skeleton_exclusion = (
        full_opposite_skeleton_exclusion[y1:y2, x1:x2]
    )
    local_relief_rejected_components = (
        full_relief_rejected_components[y1:y2, x1:x2]
    )

    # Build the final closed-core envelope. This workspace, not the semantic
    # prior, defines which regions are external.
    local_core_workspace = (
        local_base_workspace
        | local_workspace_relief
    )

    # The geometry builder's semantic support is intentionally clipped to the
    # original tube/cap envelope. Re-admit semantic pixels inside relief so the
    # widened workspace can actually participate in Canny extraction and final
    # reconstruction.
    local_effective_semantic_support = (
        local_semantic_support
        | (local_semantic_region & local_workspace_relief)
    )

    # Restrict reconstruction to pixels supported by the effective Sapiens2
    # semantic prior.
    local_core_prior = (
        local_effective_semantic_support
        & local_core_workspace
    )

    # Pipeline stage: build semantic partition barriers in the final workspace.
    local_semantic_barrier = _mask_boundary_inside_workspace(
        local_semantic_region,
        workspace_mask=local_core_workspace,
    )
    local_skeleton_label_edges = (
        local_skeleton_label_relief_edges
        & local_core_workspace
    )

    return LimbPartitionContext(
        crop_bbox=limb_geometry.crop_bbox,
        source_rgb=local_source_rgb,
        semantic_support_mask=local_effective_semantic_support,
        tube_mask=local_tube,
        workspace_relief_mask=local_workspace_relief,
        core_prior_mask=local_core_prior,
        core_workspace_mask=local_core_workspace,
        proximal_cap_mask=local_proximal_cap,
        proximal_half_plane_mask=local_proximal_half_plane,
        distal_barrier_mask=local_distal_barrier,
        proximal_barrier_mask=local_proximal_barrier,
        semantic_barrier_mask=local_semantic_barrier,
        skeleton_label_support_masks=tuple(local_skeleton_label_support_masks),
        skeleton_label_edge_mask=local_skeleton_label_edges,
        tube_edge_mask=local_tube_edges,
        skeleton_label_edge_wide_mask=local_skeleton_label_edges_wide,
        skeleton_label_edge_thin_mask=local_skeleton_label_relief_edges,
        outside_skeleton_label_edge_mask=local_outside_skeleton_label_edges,
        relief_contact_mask=local_relief_contact_mask,
        relief_contact_endpoint_mask=local_relief_contact_endpoint_mask,
        relief_opposite_skeleton_exclusion_mask=(
            local_relief_opposite_skeleton_exclusion
        ),
        relief_rejected_component_mask=local_relief_rejected_components,
        skeleton_mask=local_skeleton,
        anatomical_segments=tuple(anatomical_segments),
        proximal_point=local_proximal_point,
        middle_point=local_middle_point,
        distal_point=local_distal_point,
        limb_chain_length=_limb_geometry_chain_length(limb_geometry),
    )


def _expand_regions_over_partition_barriers(
    *,
    region_mask: np.ndarray,
    barrier_mask: np.ndarray,
    allowed_mask: np.ndarray,
    dilation_radius: int,
    output_dilation_radius: int = 0,
) -> np.ndarray:
    """
    Build an expanded silhouette from selected topological regions.

    The input regions remain the authoritative undilated topological
    representation. Partition-barrier pixels are restored by propagating from
    the selected regions through the barrier mask for a bounded number of
    one-pixel steps.

    An optional final dilation can then smooth or slightly enlarge the derived
    output silhouette without affecting the original region topology.

    Parameters
    ----------
    region_mask : np.ndarray
        Selected free-space regions before partition-barrier restoration.

    barrier_mask : np.ndarray
        Strengthened partition barriers whose thickness was removed from the
        free-space topology.

    allowed_mask : np.ndarray
        Domain inside which barrier restoration and optional output dilation are
        allowed.

    dilation_radius : int
        Maximum number of one-pixel propagation steps through connected
        partition-barrier pixels.

        This should normally match the effective barrier-strengthening radius.
        A value of zero disables barrier restoration.

    output_dilation_radius : int, default=0
        Optional final morphological dilation radius applied after partition
        barriers have been restored.

        Unlike ``dilation_radius``, this operation expands the complete derived
        silhouette and is therefore clipped back to ``allowed_mask``.

    Returns
    -------
    np.ndarray
        Boolean expanded silhouette clipped by ``allowed_mask``.

    Raises
    ------
    ValueError
        If the masks are not two-dimensional, are not shape-aligned, or if
        either radius is negative.

    Notes
    -----
    Barrier restoration is constrained to ``barrier_mask``. It cannot propagate
    through ordinary unselected free-space regions.

    ``dilation_radius`` controls how deeply restoration may travel into a
    connected strengthened barrier. ``output_dilation_radius`` controls a
    separate final enlargement of the reconstructed silhouette.
    """
    region = np.asarray(region_mask)
    barriers = np.asarray(barrier_mask)
    allowed = np.asarray(allowed_mask)

    if region.ndim != 2 or barriers.ndim != 2 or allowed.ndim != 2:
        raise ValueError(
            'region_mask, barrier_mask, and allowed_mask must be HxW'
        )
    if region.shape != barriers.shape or region.shape != allowed.shape:
        raise ValueError(
            'region_mask, barrier_mask, and allowed_mask must align'
        )

    dilation_radius = int(dilation_radius)
    output_dilation_radius = int(output_dilation_radius)

    if dilation_radius < 0 or output_dilation_radius < 0:
        raise ValueError(
            'dilation_radius and output_dilation_radius must be non-negative'
        )

    selected = region.astype(bool)
    barrier = barriers.astype(bool)
    domain = allowed.astype(bool)

    selected &= domain

    if not np.any(selected):
        return np.zeros_like(selected, dtype=bool)

    expanded = selected.copy()

    if dilation_radius > 0 and np.any(barrier):
        propagation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        for _ in range(dilation_radius):
            reached = cv2.dilate(
                expanded.astype(np.uint8),
                propagation_kernel,
                iterations=1,
            ) > 0

            restored_barriers = (
                reached
                & barrier
                & domain
                & ~expanded
            )

            if not np.any(restored_barriers):
                break

            expanded |= restored_barriers

    if output_dilation_radius == 0:
        return expanded

    output_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * output_dilation_radius + 1,
            2 * output_dilation_radius + 1,
        ),
    )
    expanded = cv2.dilate(
        expanded.astype(np.uint8),
        output_kernel,
        iterations=1,
    ) > 0

    return expanded & domain


def _extract_internal_limb_core_regions(
    ctx: HumanParseContext,
    *,
    limb_geometry: LimbCropGeometry,
    opposite_limb_skeleton_mask: Optional[np.ndarray] = None,
) -> LimbCoreExtraction:
    """
    Extract the closed topological regions forming one anatomical limb core.

    Canny edges, skeleton-label visual boundaries, the semantic-support boundary,
    and synthetic proximal/distal closures partition the final core workspace
    into free-space regions. Regions that do not touch that workspace boundary
    are classified as belonging to the closed limb core. The final reconstructed
    core remains clipped to the semantic support.

    A skeleton-label boundary pass may widen the geometric tube before this final
    partition is built. Final image-derived Canny extraction is clipped to the
    effective core prior, while short Canny bridges and anatomical endpoint
    extensions are allowed inside the wider core workspace because they act as
    separators for topology, not as reconstructed limb pixels.

    Region selection is first topological. The selected free-space components
    are then constrained to intersect semantic-support components touched by the
    anatomical skeleton. Chromatic prompts, flood seeds, midpoint prompts, and
    other point-based selectors are not used.

    Parameters
    ----------
    ctx : HumanParseContext
        Full-frame runtime context containing source pixels, image dimensions,
        semantic labels, pose, and node configuration.

    limb_geometry : LimbCropGeometry
        Full-frame anatomical geometry for the limb side being extracted.

    Returns
    -------
    LimbCoreExtraction
        Structured full-frame core extraction plus the local Canny edge map,
        barrier set, and region mosaic used to derive it.
        An empty prior or an empty internal-region selection produces a structured
        empty extraction. This function does not fall back to semantic-only masks or
        point-based region selection.

    Notes
    -----
    This function does not subtract the opposite limb, choose public
    image-relative sides or extend into the proximal half-plane. It does produce
    a cleaned structural ``expanded_mask`` for overlap subtraction and output.
    """
    partition_context = _build_limb_partition_context(
        ctx,
        limb_geometry=limb_geometry,
        opposite_limb_skeleton_mask=opposite_limb_skeleton_mask,
    )
    x1, y1, x2, y2 = partition_context.crop_bbox
    local_height, local_width = partition_context.core_prior_mask.shape

    edge_map = build_canny_edge_map(
        local_rgb=partition_context.source_rgb,
        local_mask=partition_context.core_prior_mask,
        limb_chain_length=partition_context.limb_chain_length,
    )

    def paste(local_mask: np.ndarray) -> np.ndarray:
        full = np.zeros((ctx.height, ctx.width), dtype=bool)
        full[y1:y2, x1:x2] = local_mask[:local_height, :local_width]
        return full

    if not np.any(partition_context.core_prior_mask):
        barrier_set = build_canny_partition_barriers(
            visual_barrier_mask=np.zeros_like(
                partition_context.core_prior_mask,
                dtype=bool,
            ),
            semantic_barrier_mask=partition_context.semantic_barrier_mask,
            synthetic_barrier_mask=(
                partition_context.proximal_barrier_mask
                | partition_context.distal_barrier_mask
            ),
        )
        region_mosaic = build_region_mosaic(
            workspace_mask=partition_context.core_workspace_mask,
            barrier_mask=barrier_set.partition_mask,
        )
        empty_local = np.zeros_like(
            partition_context.core_prior_mask,
            dtype=bool,
        )
        return LimbCoreExtraction(
            region_mosaic=region_mosaic,
            selected_region_labels=frozenset(),
            region_mask=paste(empty_local),
            expanded_mask=paste(empty_local),
            barrier_set=barrier_set,
            edge_map=edge_map,
            refined_edge_mask=paste(empty_local),
            prior_mask=paste(partition_context.core_prior_mask),
            workspace_mask=paste(partition_context.core_workspace_mask),
            workspace_relief_mask=paste(
                partition_context.workspace_relief_mask),
            proximal_cap_mask=paste(partition_context.proximal_cap_mask),
            proximal_half_plane_mask=paste(
                partition_context.proximal_half_plane_mask,
            ),
            semantic_support_mask=paste(
                partition_context.semantic_support_mask),
            tube_edge_mask=paste(partition_context.tube_edge_mask),
            skeleton_label_edge_wide_mask=paste(
                partition_context.skeleton_label_edge_wide_mask,
            ),
            skeleton_label_edge_thin_mask=paste(
                partition_context.skeleton_label_edge_thin_mask,
            ),
            outside_skeleton_label_edge_mask=paste(
                partition_context.outside_skeleton_label_edge_mask,
            ),
            relief_contact_mask=paste(partition_context.relief_contact_mask),
            relief_contact_endpoint_mask=paste(
                partition_context.relief_contact_endpoint_mask,
            ),
            relief_opposite_skeleton_exclusion_mask=paste(
                partition_context.relief_opposite_skeleton_exclusion_mask,
            ),
            relief_rejected_component_mask=paste(
                partition_context.relief_rejected_component_mask,
            ),
            canny_debug=LimbCannyDebug(endpoint_debug=None),
        )

    # Canny extraction is clipped to the semantic prior, but synthetic visual
    # bridge/extension pixels may cross the geometric workspace. Region
    # reconstruction is clipped back to the prior after topology selection.
    visual_barriers = (
        edge_map.cleaned_mask
        | partition_context.skeleton_label_edge_mask
    )

    if np.any(visual_barriers):
        visual_barriers = bridge_consistent_edge_endpoints(
            visual_barriers,
            allowed_domain=partition_context.core_workspace_mask,
            max_gap=max(2.0, 0.025 * float(edge_map.scale)),
            tangent_radius=max(3, int(round(0.015 * float(edge_map.scale)))),
        )

    endpoint_debug = None
    if partition_context.anatomical_segments:
        visual_barriers, endpoint_debug = connect_limb_edge_endpoints(
            visual_barriers,
            limb_skeleton_mask=partition_context.skeleton_mask,
            allowed_domain=partition_context.core_workspace_mask,
            anatomical_segments=list(partition_context.anatomical_segments),
            min_contour_parallelism=0.90,
            synthetic_stop_mask=(
                partition_context.proximal_barrier_mask
                | partition_context.distal_barrier_mask
            ),
        )

    # Pipeline stage: assemble all trusted core barriers. Skeleton-label
    # boundaries have already been injected into visual_barriers, so synthetic
    # barriers contain only explicit geometric closures.
    barrier_set = build_canny_partition_barriers(
        visual_barrier_mask=visual_barriers,
        semantic_barrier_mask=partition_context.semantic_barrier_mask,
        synthetic_barrier_mask=(
            partition_context.proximal_barrier_mask
            | partition_context.distal_barrier_mask
        ),
    )

    # Partition the geometric core workspace. Semantic support acts as an
    # internal barrier and as the final reconstruction prior, not as the external
    # free-space boundary.
    region_mosaic = build_region_mosaic(
        workspace_mask=partition_context.core_workspace_mask,
        barrier_mask=barrier_set.partition_mask,
    )

    # Closed core regions are exactly the free-space regions that do not touch the
    # workspace boundary.
    selected_labels = internal_region_labels(region_mosaic)

    local_region_mask = region_mask_from_labels(
        region_mosaic,
        selected_labels,
    )

    local_reconstruction_domain = (
        partition_context.core_prior_mask
    )

    local_region_mask &= local_reconstruction_domain
    # Pipeline stage: after topological selection, discard selected components
    # that are disconnected from the skeleton-anchored semantic labels.
    local_region_mask = filter_mask_components_by_constraint(
        local_region_mask,
        constraint_masks=partition_context.skeleton_label_support_masks,
    )

    # Preserve the undilated topological regions for proximal continuation and
    # restore only the adjacent pixels occupied by partition barriers.
    local_expanded_mask = _expand_regions_over_partition_barriers(
        region_mask=local_region_mask,
        barrier_mask=barrier_set.partition_mask,
        allowed_mask=local_reconstruction_domain,
        dilation_radius=2,
        output_dilation_radius=1,
    )

    return LimbCoreExtraction(
        region_mosaic=region_mosaic,
        selected_region_labels=selected_labels,
        region_mask=paste(local_region_mask),
        expanded_mask=paste(local_expanded_mask),
        barrier_set=barrier_set,
        edge_map=edge_map,
        refined_edge_mask=paste(visual_barriers),
        prior_mask=paste(partition_context.core_prior_mask),
        workspace_mask=paste(partition_context.core_workspace_mask),
        workspace_relief_mask=paste(partition_context.workspace_relief_mask),
        proximal_cap_mask=paste(partition_context.proximal_cap_mask),
        proximal_half_plane_mask=paste(
            partition_context.proximal_half_plane_mask),
        semantic_support_mask=paste(partition_context.semantic_support_mask),
        tube_edge_mask=paste(partition_context.tube_edge_mask),
        skeleton_label_edge_wide_mask=paste(
            partition_context.skeleton_label_edge_wide_mask,
        ),
        skeleton_label_edge_thin_mask=paste(
            partition_context.skeleton_label_edge_thin_mask,
        ),
        outside_skeleton_label_edge_mask=paste(
            partition_context.outside_skeleton_label_edge_mask,
        ),
        relief_contact_mask=paste(partition_context.relief_contact_mask),
        relief_contact_endpoint_mask=paste(
            partition_context.relief_contact_endpoint_mask,
        ),
        relief_opposite_skeleton_exclusion_mask=paste(
            partition_context.relief_opposite_skeleton_exclusion_mask,
        ),
        relief_rejected_component_mask=paste(
            partition_context.relief_rejected_component_mask,
        ),
        canny_debug=LimbCannyDebug(endpoint_debug=endpoint_debug),
    )


def _build_front_other_limb_subtraction(
    ctx: HumanParseContext,
    *,
    target_geometry: LimbCropGeometry,
    other_geometry: LimbCropGeometry,
    target_mask: np.ndarray,
    other_mask: np.ndarray,
    min_depth_delta: float = LIMB_DEPTH_MIN_DELTA,
) -> LimbOverlapSubtraction:
    """
    Build the pixels to remove from one target limb where the opposite limb is
    reliably in front.

    Anatomical segment capsules determine whether corresponding target and
    opposite-limb portions compete spatially. MediaPipe depth is compared only over
    actual mask overlap inside intersecting capsules. When the opposite segment is
    closer by at least ``min_depth_delta``, its competing mask portion is added to
    the subtraction result.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context containing frame shape and MediaPipe pose depth.

    target_geometry : LimbCropGeometry
        Anatomical geometry for the limb being cleaned.

    other_geometry : LimbCropGeometry
        Anatomical geometry for the opposite limb being tested as occluder.

    target_mask : np.ndarray
        Full-frame target limb mask snapshot.

    other_mask : np.ndarray
        Full-frame opposite limb mask snapshot.

    min_depth_delta : float, default=LIMB_DEPTH_MIN_DELTA
        Minimum MediaPipe depth advantage required for the opposite segment to
        be considered reliably in front.

    Returns
    -------
    LimbOverlapSubtraction
        Full-frame subtraction mask plus comparison counters.

    Raises
    ------
    ValueError
        If ``min_depth_delta`` is negative or the masks do not match the frame.

    Notes
    -----
    Ambiguous, missing, contradictory, or insufficient depth evidence leaves the
    target unchanged. Inputs are copied to boolean snapshots and are not mutated.
    """
    if min_depth_delta < 0.0:
        raise ValueError(
            f'min_depth_delta must be >= 0, got {min_depth_delta!r}.'
        )

    shape = (ctx.height, ctx.width)
    target_bool = np.asarray(target_mask).astype(bool).copy()
    other_bool = np.asarray(other_mask).astype(bool).copy()
    if target_bool.shape != shape or other_bool.shape != shape:
        raise ValueError(
            'target_mask and other_mask must match full-frame shape')

    target_segments = _limb_geometry_depth_segments(ctx, target_geometry)
    other_segments = _limb_geometry_depth_segments(ctx, other_geometry)
    subtraction_mask = np.zeros(shape, dtype=bool)
    added_overlap = np.zeros(shape, dtype=bool)
    compared_segment_pairs = 0
    accepted_segment_pairs = 0

    if not target_segments or not other_segments:
        return LimbOverlapSubtraction(
            subtraction_mask=subtraction_mask,
            changed=False,
            compared_segment_pairs=0,
            accepted_segment_pairs=0,
        )

    for target_segment in target_segments:
        target_capsule = segment_capsule_mask(
            shape,
            np.asarray(target_segment['start_xy'], dtype=np.float32),
            np.asarray(target_segment['end_xy'], dtype=np.float32),
            float(target_geometry.tube_radius),
        )

        for other_segment in other_segments:
            other_capsule = segment_capsule_mask(
                shape,
                np.asarray(other_segment['start_xy'], dtype=np.float32),
                np.asarray(other_segment['end_xy'], dtype=np.float32),
                float(other_geometry.tube_radius),
            )
            # Evaluate depth only where both masks and both anatomical
            # capsules overlap.
            overlap = (
                target_capsule
                & other_capsule
                & target_bool
                & other_bool
                & ~added_overlap
            )
            if not np.any(overlap):
                continue

            target_interval = _project_mask_on_segment_interval(
                overlap,
                np.asarray(target_segment['start_xy'], dtype=np.float32),
                np.asarray(target_segment['end_xy'], dtype=np.float32),
            )
            other_interval = _project_mask_on_segment_interval(
                overlap,
                np.asarray(other_segment['start_xy'], dtype=np.float32),
                np.asarray(other_segment['end_xy'], dtype=np.float32),
            )
            if target_interval is None or other_interval is None:
                continue

            compared_segment_pairs += 1
            target_depth = _segment_interval_mean_depth(
                target_segment,
                target_interval,
            )
            other_depth = _segment_interval_mean_depth(
                other_segment,
                other_interval,
            )

            if not (other_depth + float(min_depth_delta) < target_depth):
                continue

            # Once the opposite segment is classified as being in front,
            # remove its complete competing portion inside the current target mask.
            subtraction = (
                target_bool
                & other_bool
                & other_capsule
                & ~subtraction_mask
            )
            if not np.any(subtraction):
                continue

            subtraction_mask |= subtraction
            added_overlap |= overlap
            accepted_segment_pairs += 1

    return LimbOverlapSubtraction(
        subtraction_mask=subtraction_mask,
        changed=bool(np.any(subtraction_mask)),
        compared_segment_pairs=int(compared_segment_pairs),
        accepted_segment_pairs=int(accepted_segment_pairs),
    )


def _apply_limb_overlap_subtraction(
    *,
    mask: np.ndarray,
    subtraction: LimbOverlapSubtraction,
) -> np.ndarray:
    """
    Apply one opposite-limb subtraction to a mask.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame mask to clean.

    subtraction : LimbOverlapSubtraction
        Subtraction decision whose mask is aligned with ``mask``.

    Returns
    -------
    np.ndarray
        Boolean ``mask & ~subtraction.subtraction_mask``.

    Raises
    ------
    ValueError
        If ``mask`` and ``subtraction.subtraction_mask`` are not shape-aligned.

    Notes
    -----
    No cleanup, dilation, topology reconstruction, or relabeling is performed.
    """
    source = np.asarray(mask).astype(bool)
    subtraction_mask = np.asarray(subtraction.subtraction_mask).astype(bool)
    if source.shape != subtraction_mask.shape:
        raise ValueError('mask and subtraction.subtraction_mask must align')
    return source & ~subtraction_mask


def _subtract_limb_core_overlaps(
    ctx: HumanParseContext,
    *,
    geometries_by_side: dict[str, LimbCropGeometry],
    cores_by_side: dict[str, LimbCoreExtraction],
) -> dict[str, LimbCoreExtraction]:
    """
    Apply bilateral overlap subtraction consistently to limb core products.

    Subtraction decisions are computed from ``expanded_mask`` snapshots because
    those masks represent the effective competing image pixels. The resulting
    side-specific subtraction mask is then applied to both ``region_mask`` and
    ``expanded_mask`` so removed contamination cannot later become a proximal
    continuation anchor.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context containing frame shape and pose depth.

    geometries_by_side : dict[str, LimbCropGeometry]
        Full-frame anatomical geometries keyed by anatomical side.

    cores_by_side : dict[str, LimbCoreExtraction]
        Core extraction products keyed by anatomical side.

    Returns
    -------
    dict[str, LimbCoreExtraction]
        New core extraction objects with cleaned masks. Mosaics, selected
        labels, barrier sets, edge maps, and geometry metadata are preserved.

    Raises
    ------
    ValueError
        If any core mask does not match the full-frame shape.

    Notes
    -----
    All subtraction decisions use the original bilateral snapshot.
    The same subtraction mask is applied to both the undilated region mask and the
    expanded silhouette. This prevents pixels removed as opposite-limb
    contamination from later becoming proximal-continuation anchors.
    """
    region_snapshots = {
        side: np.asarray(core.region_mask).astype(bool).copy()
        for side, core in cores_by_side.items()
    }
    expanded_snapshots = {
        side: np.asarray(core.expanded_mask).astype(bool).copy()
        for side, core in cores_by_side.items()
    }

    shape = (ctx.height, ctx.width)
    for mask_kind, snapshots in (
        ('region', region_snapshots),
        ('expanded', expanded_snapshots),
    ):
        for side, mask in snapshots.items():
            if mask.shape != shape:
                raise ValueError(
                    f'{mask_kind} core mask for side {side!r} must match '
                    f'full-frame shape {shape!r}'
                )

    cleaned: dict[str, LimbCoreExtraction] = {}
    for side, core in cores_by_side.items():
        target_geometry = geometries_by_side.get(side)
        other_side = (
            'anatomical-right'
            if side == 'anatomical-left'
            else 'anatomical-left'
        )
        other_geometry = geometries_by_side.get(other_side)
        other_expanded = expanded_snapshots.get(other_side)

        if (
            target_geometry is None
            or other_geometry is None
            or other_expanded is None
        ):
            cleaned[side] = core
            continue

        subtraction = _build_front_other_limb_subtraction(
            ctx,
            target_geometry=target_geometry,
            other_geometry=other_geometry,
            target_mask=expanded_snapshots[side],
            other_mask=other_expanded,
        )
        cleaned[side] = replace(
            core,
            region_mask=_apply_limb_overlap_subtraction(
                mask=region_snapshots[side],
                subtraction=subtraction,
            ),
            expanded_mask=_apply_limb_overlap_subtraction(
                mask=expanded_snapshots[side],
                subtraction=subtraction,
            ),
            overlap_subtraction=subtraction,
        )

    return cleaned


def _extend_limb_regions_proximally(
    ctx: HumanParseContext,
    *,
    limb_geometry: LimbCropGeometry,
    core: LimbCoreExtraction,
) -> LimbProximalExtension:
    """
    Extend a cleaned limb core through its proximal semantic workspace.

    The proximal workspace is the filtered semantic support inside the union of
    the proximal cap and outward proximal quadrant. A new Canny partition is built
    inside that workspace without reintroducing the synthetic proximal closure
    used to close the core.

    Proximal regions are selected by topological continuity with the cleaned,
    undilated core pixels lying inside the proximal cap. Landmark points,
    chromatic points, flood seeds, and internal-region classification are not used
    for proximal selection.

    The continuation reference is:

    ``core.region_mask & core.proximal_cap_mask``

    The undilated core is deliberately used instead of ``core.expanded_mask`` so
    restored barrier thickness cannot create artificial connections.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context containing frame shape and RGB pixels.

    limb_geometry : LimbCropGeometry
        Full-frame anatomical geometry for the limb side being extended.

    core : LimbCoreExtraction
        Cleaned core extraction whose undilated region mask supplies the
        proximal continuation anchor.

    Returns
    -------
    LimbProximalExtension
        Structured extension masks and the local proximal mosaic, selected
        labels, barrier set, edge map, workspace, and continuation mask.

    Raises
    ------
    ValueError
        If any full-frame mask stored in ``core`` is not frame-aligned.

    Notes
    -----
    The returned masks contain only the new proximal extension. This helper does
    not merge with the core, extend unselected limbs, classify merely internal
    proximal regions, subtract opposite limbs, or apply final cleanup.
    """
    shape = (ctx.height, ctx.width)
    for name, mask in (
        ('core.region_mask', core.region_mask),
        ('core.expanded_mask', core.expanded_mask),
        ('core.proximal_cap_mask', core.proximal_cap_mask),
        ('core.proximal_half_plane_mask', core.proximal_half_plane_mask),
        ('core.semantic_support_mask', core.semantic_support_mask),
    ):
        if np.asarray(mask).shape != shape:
            raise ValueError(f'{name} must match full-frame shape {shape!r}')

    support = np.asarray(core.semantic_support_mask).astype(bool)
    proximal_cap = np.asarray(core.proximal_cap_mask).astype(bool)
    proximal_half_plane = np.asarray(
        core.proximal_half_plane_mask).astype(bool)
    core_region = np.asarray(core.region_mask).astype(bool)
    core_expanded = np.asarray(core.expanded_mask).astype(bool)

    # Limit proximal reconstruction to semantic support inside the cap and outward
    # proximal quadrant.
    proximal_workspace = support & (proximal_cap | proximal_half_plane)

    # Use only cleaned undilated core pixels inside the cap as the continuity
    # reference.
    continuation_mask = core_region & proximal_cap

    x1, y1, x2, y2 = limb_geometry.crop_bbox
    local_shape = (int(y2 - y1), int(x2 - x1))

    empty_local_edge = build_canny_edge_map(
        local_rgb=np.ascontiguousarray(ctx.img_rgb[y1:y2, x1:x2, :]),
        local_mask=np.zeros(local_shape, dtype=bool),
        limb_chain_length=_limb_geometry_chain_length(limb_geometry),
    )
    empty_barrier_set = build_canny_partition_barriers(
        visual_barrier_mask=np.zeros(local_shape, dtype=bool),
    )
    empty_mosaic = build_region_mosaic(
        workspace_mask=np.zeros(local_shape, dtype=bool),
        barrier_mask=np.zeros(local_shape, dtype=bool),
    )
    empty_full = np.zeros(shape, dtype=bool)

    if not np.any(proximal_workspace) or not np.any(continuation_mask):
        return LimbProximalExtension(
            region_mosaic=empty_mosaic,
            selected_region_labels=frozenset(),
            region_mask=empty_full.copy(),
            expanded_mask=empty_full.copy(),
            barrier_set=empty_barrier_set,
            edge_map=empty_local_edge,
            workspace_mask=proximal_workspace.copy(),
            continuation_mask=continuation_mask.copy(),
            canny_debug=LimbCannyDebug(endpoint_debug=None),
        )

    partition_context = _build_limb_partition_context(
        ctx,
        limb_geometry=limb_geometry,
    )

    local_workspace = proximal_workspace[y1:y2, x1:x2]
    local_continuation = continuation_mask[y1:y2, x1:x2]
    local_source_rgb = np.ascontiguousarray(ctx.img_rgb[y1:y2, x1:x2, :])

    local_semantic_region = (
        limb_geometry.semantic_region_mask[y1:y2, x1:x2]
        .astype(bool)
    )
    # Do not reintroduce the proximal closure used by the core mosaic. Keeping that
    # barrier would disconnect the cap regions from their natural continuation in
    # the proximal quadrant.
    local_synthetic = (
        partition_context.distal_barrier_mask
        & local_workspace
    )

    semantic_boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    local_semantic_barrier = (
        local_semantic_region
        & ~(
            cv2.erode(
                local_semantic_region.astype(np.uint8),
                semantic_boundary_kernel,
                iterations=1,
            ) > 0
        )
        & local_workspace
    )

    edge_map = build_canny_edge_map(
        local_rgb=local_source_rgb,
        local_mask=local_workspace,
        limb_chain_length=_limb_geometry_chain_length(limb_geometry),
    )
    visual_barriers = bridge_consistent_edge_endpoints(
        edge_map.cleaned_mask,
        allowed_domain=local_workspace,
        max_gap=max(2.0, 0.02 * float(edge_map.scale)),
        tangent_radius=max(3, int(round(0.015 * float(edge_map.scale)))),
    )
    barrier_set = build_canny_partition_barriers(
        visual_barrier_mask=visual_barriers,
        semantic_barrier_mask=local_semantic_barrier,
        synthetic_barrier_mask=local_synthetic,
    )
    region_mosaic = build_region_mosaic(
        workspace_mask=local_workspace,
        barrier_mask=barrier_set.partition_mask,
    )

    # Select proximal mosaic regions containing cleaned undilated core pixels
    # inside the cap. Each selected label then represents the complete free-space
    # component topologically continuous with those pixels.
    selected_labels = region_labels_intersecting_mask(
        region_mosaic,
        local_continuation,
    )
    local_region_mask = region_mask_from_labels(
        region_mosaic,
        selected_labels,
    )
    local_region_mask &= local_workspace

    # Return only newly reconstructed proximal pixels; the merge stage will restore
    # the already existing core.
    local_region_mask &= ~core_region[y1:y2, x1:x2]

    local_expanded_mask = _expand_regions_over_partition_barriers(
        region_mask=local_region_mask,
        barrier_mask=barrier_set.partition_mask,
        allowed_mask=local_workspace,
        dilation_radius=2,
        output_dilation_radius=1,
    )
    # Keep the expanded extension disjoint from the already expanded core.
    local_expanded_mask &= ~core_expanded[y1:y2, x1:x2]

    full_region = np.zeros(shape, dtype=bool)
    full_expanded = np.zeros(shape, dtype=bool)
    full_region[y1:y2, x1:x2] = local_region_mask
    full_expanded[y1:y2, x1:x2] = local_expanded_mask

    return LimbProximalExtension(
        region_mosaic=region_mosaic,
        selected_region_labels=selected_labels,
        region_mask=full_region,
        expanded_mask=full_expanded,
        barrier_set=barrier_set,
        edge_map=edge_map,
        workspace_mask=proximal_workspace.copy(),
        continuation_mask=continuation_mask.copy(),
        canny_debug=LimbCannyDebug(endpoint_debug=None),
    )


def _merge_limb_core_and_extension(
    *,
    core: LimbCoreExtraction,
    extension: LimbProximalExtension,
) -> np.ndarray:
    """
    Merge an expanded limb core with its expanded proximal continuation.

    Parameters
    ----------
    core : LimbCoreExtraction
        Core extraction whose ``expanded_mask`` represents the visible core
        silhouette.

    extension : LimbProximalExtension
        Proximal continuation whose ``expanded_mask`` represents only the new
        visible extension pixels.

    Returns
    -------
    np.ndarray
        Boolean union of ``core.expanded_mask`` and ``extension.expanded_mask``.

    Raises
    ------
    ValueError
        If the expanded masks are not shape-aligned.

    Notes
    -----
    This function performs no subtraction, topology reconstruction,
    morphology, or structural cleanup.
    """
    core_mask = np.asarray(core.expanded_mask).astype(bool)
    extension_mask = np.asarray(extension.expanded_mask).astype(bool)
    if core_mask.shape != extension_mask.shape:
        raise ValueError(
            'core.expanded_mask and extension.expanded_mask must align')
    return core_mask | extension_mask


def _subtract_post_extension_limb_overlaps(
    ctx: HumanParseContext,
    *,
    geometries_by_side: dict[str, LimbCropGeometry],
    target_masks_by_side: dict[str, np.ndarray],
    reference_masks_by_side: dict[str, np.ndarray],
) -> LimbPostExtensionOverlapResult:
    """
    Remove opposite-limb contamination introduced by proximal extension.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime pose and image context.

    geometries_by_side : dict[str, LimbCropGeometry]
        Anatomical geometries for all available sides.

    target_masks_by_side : dict[str, np.ndarray]
        Final core-plus-extension masks for the limbs selected for output.

    reference_masks_by_side : dict[str, np.ndarray]
        Current masks representing each opposite limb.

        For bilateral targets these should normally be the extended masks of
        both sides. For a single-side target, the unselected opposite limb may
        be represented by its cleaned core mask.

    Returns
    -------
    LimbPostExtensionOverlapResult
        Cleaned selected masks and the side-specific subtraction decisions used
        to produce them.

    Raises
    ------
    ValueError
        If any target or reference mask does not match the full-frame shape.

    Notes
    -----
    This second overlap pass removes opposite-limb contamination that may have
    been introduced when a selected core was extended into its proximal quadrant.
    It uses immutable snapshots for every decision, so
    cleaning one selected side cannot influence another. It does not rebuild any
    region mosaic.
    """
    shape = (ctx.height, ctx.width)
    target_snapshots: dict[str, np.ndarray] = {}
    reference_snapshots: dict[str, np.ndarray] = {}

    for side, mask in target_masks_by_side.items():
        snapshot = np.asarray(mask).astype(bool).copy()
        if snapshot.shape != shape:
            raise ValueError(
                f'target mask for side {side!r} must match {shape!r}'
            )
        target_snapshots[side] = snapshot

    for side, mask in reference_masks_by_side.items():
        snapshot = np.asarray(mask).astype(bool).copy()
        if snapshot.shape != shape:
            raise ValueError(
                f'reference mask for side {side!r} must match {shape!r}'
            )
        reference_snapshots[side] = snapshot

    cleaned: dict[str, np.ndarray] = {}
    subtractions: dict[str, LimbOverlapSubtraction] = {}
    for side, target_snapshot in target_snapshots.items():
        target_geometry = geometries_by_side.get(side)
        other_side = (
            'anatomical-right'
            if side == 'anatomical-left'
            else 'anatomical-left'
        )
        other_geometry = geometries_by_side.get(other_side)
        other_reference = reference_snapshots.get(other_side)

        if (
            target_geometry is None
            or other_geometry is None
            or other_reference is None
        ):
            cleaned[side] = target_snapshot.copy()
            continue

        subtraction = _build_front_other_limb_subtraction(
            ctx,
            target_geometry=target_geometry,
            other_geometry=other_geometry,
            target_mask=target_snapshot,
            other_mask=other_reference,
        )
        subtractions[side] = subtraction
        cleaned[side] = _apply_limb_overlap_subtraction(
            mask=target_snapshot,
            subtraction=subtraction,
        )

    return LimbPostExtensionOverlapResult(
        masks_by_side=cleaned,
        subtractions_by_side=subtractions,
    )


def _paste_local_bool_mask(
    *,
    local_mask: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """
    Paste a local crop mask into a full-frame boolean canvas.

    Parameters
    ----------
    local_mask : np.ndarray
        Two-dimensional local mask aligned with ``crop_bbox``.

    crop_bbox : tuple[int, int, int, int]
        End-exclusive full-frame destination box ``(x1, y1, x2, y2)``.

    full_shape : tuple[int, int]
        Full-frame output shape as ``(height, width)``.

    Returns
    -------
    np.ndarray
        Boolean full-frame mask containing ``local_mask`` inside ``crop_bbox``.
    """
    x1, y1, x2, y2 = crop_bbox
    full = np.zeros(full_shape, dtype=bool)
    full[y1:y2, x1:x2] = (
        np.asarray(local_mask).astype(bool)[:y2 - y1, :x2 - x1]
    )
    return full


def _limb_topology_debug_images(
    *,
    ctx: HumanParseContext,
    limb_debug: LimbSegmentationDebug,
    final_mask: np.ndarray,
) -> list[LimbTopologyDebugImage]:
    """
    Build render specifications for the limb topology debug directory.
    """
    images: list[LimbTopologyDebugImage] = []
    full_shape = (ctx.height, ctx.width)

    def paste(local_mask: np.ndarray, geometry: LimbCropGeometry) -> np.ndarray:
        return _paste_local_bool_mask(
            local_mask=local_mask,
            crop_bbox=geometry.crop_bbox,
            full_shape=full_shape,
        )

    for side in sorted(limb_debug.geometries_by_side):
        geometry = limb_debug.geometries_by_side[side]
        core = limb_debug.cores_by_side.get(side)
        if core is None:
            continue

        side_slug = side.replace('anatomical-', '')
        endpoint_debug = (
            None
            if core.canny_debug is None
            else core.canny_debug.endpoint_debug
        )

        def debug_mask(mask: Optional[np.ndarray]) -> np.ndarray:
            if mask is None:
                return np.zeros(full_shape, dtype=bool)
            return mask.astype(bool)

        images.append(LimbTopologyDebugImage(
            filename=f'{limb_debug.target_kind}_{side_slug}_relief_edges.png',
            title=f'{limb_debug.target_kind} {side} relief edges',
            crop_bbox=geometry.crop_bbox,
            mask_overlays=[
                CropDebugMaskOverlay(
                    'base workspace',
                    core.workspace_mask & ~core.workspace_relief_mask,
                    (120, 120, 120),
                    0.16,
                ),
                CropDebugMaskOverlay(
                    'workspace relief',
                    core.workspace_relief_mask,
                    (255, 190, 70),
                    0.48,
                ),
                CropDebugMaskOverlay(
                    'opposite skeleton exclusion',
                    debug_mask(core.relief_opposite_skeleton_exclusion_mask),
                    (80, 120, 255),
                    0.28,
                ),
                CropDebugMaskOverlay(
                    'rejected relief components',
                    debug_mask(core.relief_rejected_component_mask),
                    (255, 40, 40),
                    0.62,
                ),
            ],
            edge_overlays=[
                CropDebugEdgeOverlay(
                    'tube boundary 1px',
                    debug_mask(core.tube_edge_mask),
                    (160, 220, 255),
                ),
                CropDebugEdgeOverlay(
                    'skeleton label edge thinned',
                    debug_mask(core.skeleton_label_edge_thin_mask),
                    (255, 255, 80),
                ),
                CropDebugEdgeOverlay(
                    'skeleton label edge thinned outside tube',
                    debug_mask(core.outside_skeleton_label_edge_mask),
                    (255, 120, 255),
                ),
                CropDebugEdgeOverlay(
                    'skeleton label edge raw wide',
                    debug_mask(core.skeleton_label_edge_wide_mask),
                    (150, 80, 150),
                ),
                CropDebugEdgeOverlay(
                    'tube contact edge pixels',
                    debug_mask(core.relief_contact_mask),
                    (0, 255, 120),
                ),
                CropDebugEdgeOverlay(
                    'relief contact endpoints',
                    debug_mask(core.relief_contact_endpoint_mask),
                    (255, 80, 80),
                ),
                CropDebugEdgeOverlay(
                    'anatomical skeleton',
                    geometry.skeleton_mask,
                    (0, 255, 255),
                ),
            ],
        ))

        images.append(LimbTopologyDebugImage(
            filename=f'{limb_debug.target_kind}_{side_slug}_core_edges.png',
            title=f'{limb_debug.target_kind} {side} core edges',
            crop_bbox=geometry.crop_bbox,
            mask_overlays=[
                CropDebugMaskOverlay(
                    'core prior',
                    core.prior_mask,
                    (55, 150, 255),
                    0.28,
                ),
                CropDebugMaskOverlay(
                    'workspace',
                    core.workspace_mask,
                    (120, 120, 120),
                    0.14,
                ),
                CropDebugMaskOverlay(
                    'workspace relief',
                    core.workspace_relief_mask,
                    (255, 190, 70),
                    0.46,
                ),
            ],
            edge_overlays=[
                CropDebugEdgeOverlay(
                    'raw canny skeleton 1px',
                    paste(core.edge_map.raw_mask, geometry),
                    (80, 160, 255),
                ),
                CropDebugEdgeOverlay(
                    'cleaned canny skeleton 1px',
                    paste(core.edge_map.cleaned_mask, geometry),
                    (255, 220, 80),
                ),
                CropDebugEdgeOverlay(
                    'refined visual edge 1px',
                    core.refined_edge_mask,
                    (255, 80, 80),
                ),
                CropDebugEdgeOverlay(
                    'anatomical skeleton',
                    geometry.skeleton_mask,
                    (0, 255, 255),
                ),
            ],
            endpoint_debug=endpoint_debug,
        ))

        images.append(LimbTopologyDebugImage(
            filename=f'{limb_debug.target_kind}_{side_slug}_core_regions.png',
            title=f'{limb_debug.target_kind} {side} core regions',
            crop_bbox=geometry.crop_bbox,
            mask_overlays=[
                CropDebugMaskOverlay(
                    'expanded core',
                    core.expanded_mask,
                    (70, 190, 255),
                    0.34,
                ),
                CropDebugMaskOverlay(
                    'proximal cap',
                    core.proximal_cap_mask,
                    (255, 220, 60),
                    0.22,
                ),
                CropDebugMaskOverlay(
                    'workspace relief',
                    core.workspace_relief_mask,
                    (255, 190, 70),
                    0.46,
                ),
            ],
            edge_overlays=[
                CropDebugEdgeOverlay(
                    'visual barrier dilated',
                    paste(core.barrier_set.visual_mask, geometry),
                    (255, 255, 255),
                ),
                CropDebugEdgeOverlay(
                    'semantic barrier dilated',
                    paste(core.barrier_set.semantic_mask, geometry),
                    (255, 120, 255),
                ),
                CropDebugEdgeOverlay(
                    'synthetic barrier dilated',
                    paste(core.barrier_set.synthetic_mask, geometry),
                    (255, 170, 60),
                ),
            ],
            region_mosaic=core.region_mosaic,
            selected_region_labels=core.selected_region_labels,
        ))

        subtraction = limb_debug.core_subtractions_by_side.get(side)
        if subtraction is not None and subtraction.changed:
            images.append(LimbTopologyDebugImage(
                filename=(
                    f'{limb_debug.target_kind}_{side_slug}_'
                    'core_subtraction.png'
                ),
                title=f'{limb_debug.target_kind} {side} core subtraction',
                crop_bbox=geometry.crop_bbox,
                mask_overlays=[
                    CropDebugMaskOverlay(
                        'cleaned core',
                        core.expanded_mask,
                        (60, 220, 120),
                        0.32,
                    ),
                    CropDebugMaskOverlay(
                        'subtraction',
                        subtraction.subtraction_mask,
                        (255, 70, 45),
                        0.60,
                    ),
                ],
            ))

    for side in limb_debug.selected_sides:
        geometry = limb_debug.geometries_by_side[side]
        extension = limb_debug.extensions_by_side.get(side)
        if extension is None:
            continue

        side_slug = side.replace('anatomical-', '')
        images.append(LimbTopologyDebugImage(
            filename=f'{limb_debug.target_kind}_{side_slug}_proximal.png',
            title=f'{limb_debug.target_kind} {side} proximal continuation',
            crop_bbox=geometry.crop_bbox,
            mask_overlays=[
                CropDebugMaskOverlay(
                    'proximal workspace',
                    extension.workspace_mask,
                    (80, 140, 255),
                    0.20,
                ),
                CropDebugMaskOverlay(
                    'cap continuation',
                    extension.continuation_mask,
                    (255, 235, 80),
                    0.50,
                ),
                CropDebugMaskOverlay(
                    'expanded extension',
                    extension.expanded_mask,
                    (75, 230, 150),
                    0.42,
                ),
            ],
            edge_overlays=[
                CropDebugEdgeOverlay(
                    'partition barrier dilated',
                    paste(extension.barrier_set.partition_mask, geometry),
                    (255, 255, 255),
                ),
            ],
            region_mosaic=extension.region_mosaic,
            selected_region_labels=extension.selected_region_labels,
            endpoint_debug=(
                None
                if extension.canny_debug is None
                else extension.canny_debug.endpoint_debug
            ),
        ))

        subtraction = limb_debug.post_subtractions_by_side.get(side)
        if subtraction is not None and subtraction.changed:
            images.append(LimbTopologyDebugImage(
                filename=(
                    f'{limb_debug.target_kind}_{side_slug}_'
                    'post_extension_subtraction.png'
                ),
                title=(
                    f'{limb_debug.target_kind} {side} '
                    'post-extension subtraction'
                ),
                crop_bbox=geometry.crop_bbox,
                mask_overlays=[
                    CropDebugMaskOverlay(
                        'final side mask',
                        limb_debug.final_masks_by_side[side],
                        (70, 220, 120),
                        0.38,
                    ),
                    CropDebugMaskOverlay(
                        'subtraction',
                        subtraction.subtraction_mask,
                        (255, 70, 45),
                        0.60,
                    ),
                ],
            ))

    final_overlays = [
        CropDebugMaskOverlay(
            'final mask',
            final_mask,
            (255, 0, 255),
            0.30,
        ),
    ]
    for side in limb_debug.selected_sides:
        final_overlays.append(CropDebugMaskOverlay(
            side.replace('anatomical-', ''),
            limb_debug.final_masks_by_side[side],
            (60, 220, 120) if side.endswith('left') else (70, 180, 255),
            0.34,
        ))

    images.append(LimbTopologyDebugImage(
        filename=f'{limb_debug.target_kind}_final.png',
        title=f'{limb_debug.target_kind} final selected mask',
        mask_overlays=final_overlays,
    ))

    return images


def _segment_limb_target(
    ctx: HumanParseContext,
    *,
    target_kind: Literal['arm', 'leg'],
    semantic_labels: tuple[str, ...],
    requested_side: str,
    geometry_builder: Any,
    prompt_bbox_label: str,
) -> TargetSegmentationResult:
    """
    Segment one public arm or leg target through the shared topological pipeline.

    The pipeline builds both anatomical limb geometries, extracts their closed
    topological cores, removes opposite-limb overlap using capsule and MediaPipe
    depth evidence, resolves the requested image-relative side, extends only the
    selected limbs through their proximal semantic workspaces, repeats overlap
    cleanup after extension, and finally applies structural mask cleanup.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context containing image, segmentation, pose, and node config.

    target_kind : {'arm', 'leg'}
        Limb family name used for diagnostics and label metadata.

    semantic_labels : tuple[str, ...]
        Sapiens2 labels that form the bilateral semantic support mask passed to
        the limb geometry builder.

    requested_side : str
        Public side request resolved by the caller: ``'both'``, ``'left'``, or
        ``'right'`` in image-relative coordinates.

    geometry_builder : Any
        Function compatible with ``build_arm_crop_geometries`` and
        ``build_leg_crop_geometries``.

    prompt_bbox_label : str
        Debug label used for the union of selected local processing crops.

    Returns
    -------
    TargetSegmentationResult
        Final cleaned target mask, target metadata, prompt bbox, per-side part
        masks, and debug masks.
    """
    semantic_support_region = np.isin(
        ctx.segments,
        np.asarray(
            [
                SAPIENS2_CLASSES[label]
                for label in semantic_labels
            ],
            dtype=np.int64,
        ),
    )

    resolve_landmark = _make_pose_landmark_resolver(ctx)

    all_geometries = geometry_builder(
        resolve_landmark=resolve_landmark,
        image_shape=ctx.img_rgb.shape,
        semantic_region=semantic_support_region,
        which='both',
        expansion=max(1.0, float(ctx.cfg.expansion)),
    )

    if not all_geometries:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': could not derive "
            f'{target_kind} geometry.'
        )

    geometries_by_side = {
        geometry.side: geometry
        for geometry in all_geometries
    }

    # Extract both anatomical cores before resolving the public image-relative
    # target so overlap reasoning always has bilateral information.
    cores_by_side: dict[str, LimbCoreExtraction] = {}
    for geometry in all_geometries:
        opposite_geometry = next(
            (
                other_geometry
                for other_geometry in all_geometries
                if other_geometry.side != geometry.side
            ),
            None,
        )
        cores_by_side[geometry.side] = _extract_internal_limb_core_regions(
            ctx,
            limb_geometry=geometry,
            opposite_limb_skeleton_mask=(
                None
                if opposite_geometry is None
                else opposite_geometry.skeleton_mask
            ),
        )
    # Clean both cores from the same immutable bilateral snapshot.
    cores_by_side = _subtract_limb_core_overlaps(
        ctx,
        geometries_by_side=geometries_by_side,
        cores_by_side=cores_by_side,
    )

    selected_side, side_resolution = _select_limb_side_by_proximal_landmark(
        ctx=ctx,
        requested=requested_side,
        target_name=ctx.cfg.target,
        geometries=all_geometries,
    )
    resolved = _resolved_limb_target(
        ctx=ctx,
        target_kind=target_kind,
        public_target=ctx.cfg.target,
        selected_side=selected_side,
        side_resolution=side_resolution,
    )

    if selected_side == 'both':
        selected_geometries = all_geometries
    else:
        target_geometry = geometries_by_side.get(selected_side)
        if target_geometry is None:
            raise RuntimeError(
                f"SubjectCrop node '{ctx.node_id}': could not derive "
                f'geometry for requested image-side {requested_side} '
                f'{target_kind}.'
            )
        selected_geometries = [target_geometry]

    selected_sides = {
        geometry.side
        for geometry in selected_geometries
    }
    extensions_by_side = {
        side: _extend_limb_regions_proximally(
            ctx,
            limb_geometry=geometries_by_side[side],
            core=cores_by_side[side],
        )
        for side in selected_sides
    }
    target_masks_by_side = {
        side: _merge_limb_core_and_extension(
            core=cores_by_side[side],
            extension=extensions_by_side[side],
        )
        for side in selected_sides
    }

    if selected_side == 'both':
        reference_masks_by_side = target_masks_by_side
    else:
        reference_masks_by_side = {
            side: core.expanded_mask
            for side, core in cores_by_side.items()
        }
        reference_masks_by_side.update(target_masks_by_side)

    # Repeat overlap cleanup after proximal extension because newly added proximal
    # regions may contain pixels belonging to the opposite limb.
    post_extension_overlap = _subtract_post_extension_limb_overlaps(
        ctx,
        geometries_by_side=geometries_by_side,
        target_masks_by_side=target_masks_by_side,
        reference_masks_by_side=reference_masks_by_side,
    )
    target_masks_by_side = post_extension_overlap.masks_by_side
    post_subtractions_by_side = (
        post_extension_overlap.subtractions_by_side
    )
    core_subtractions_by_side = {
        side: core.overlap_subtraction
        for side, core in cores_by_side.items()
        if core.overlap_subtraction is not None
    }

    mask = np.zeros((ctx.height, ctx.width), dtype=bool)
    part_masks: list[np.ndarray] = []
    debug_region_masks: list[np.ndarray] = []
    debug_edge_masks: list[np.ndarray] = []

    for geometry in selected_geometries:
        side = geometry.side
        side_mask = target_masks_by_side[side]
        part_masks.append(side_mask)
        mask |= side_mask
        debug_region_masks.append(geometry.semantic_support_mask)

        core_barriers = _paste_local_bool_mask(
            local_mask=cores_by_side[side].barrier_set.partition_mask,
            crop_bbox=geometry.crop_bbox,
            full_shape=(ctx.height, ctx.width),
        )
        extension_barriers = _paste_local_bool_mask(
            local_mask=extensions_by_side[side].barrier_set.partition_mask,
            crop_bbox=geometry.crop_bbox,
            full_shape=(ctx.height, ctx.width),
        )
        debug_edge_masks.append(core_barriers | extension_barriers)

    mask = cleanup_shape_mask_by_parts(
        mask,
        part_masks,
        **ctx.cfg.shape_cleanup,
    )

    if not np.any(mask):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': {target_kind} mask is empty."
        )

    # Derive the output bbox from the final cleaned mask rather than from the
    # broader semantic processing geometry.
    target_bbox = tight_mask_bbox(
        mask.astype(np.uint8) * 255
    )

    return TargetSegmentationResult(
        mask=mask,
        target_bbox=target_bbox,
        labels=resolved.labels,
        label_ids=resolved.label_ids,
        selected_candidates=resolved.selected_candidates,
        side_resolutions=resolved.side_resolutions,
        prompt_bbox=union_bboxes_xyxy([
            geometry.crop_bbox
            for geometry in selected_geometries
        ]),
        shape_part_masks=part_masks,
        crop_regions=None,
        prompt_bbox_label=prompt_bbox_label,
        debug_region_masks=debug_region_masks,
        debug_edge_masks=debug_edge_masks,
        limb_debug=LimbSegmentationDebug(
            target_kind=target_kind,
            selected_sides=tuple(sorted(selected_sides)),
            geometries_by_side=geometries_by_side,
            cores_by_side=cores_by_side,
            extensions_by_side=extensions_by_side,
            core_subtractions_by_side=core_subtractions_by_side,
            post_subtractions_by_side=post_subtractions_by_side,
            final_masks_by_side=target_masks_by_side,
        ),
    )


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'bfloat16')))
    yolo_model = str(model.get('yolo_model', 'yolov8n.pt'))
    segment_model = str(
        model.get('segment_model', 'facebook/sapiens2-seg-0.4b')
    )
    pose_landmarker_task_raw = model.get('pose_landmarker_task')
    pose_landmarker_task = (
        None
        if pose_landmarker_task_raw is None
        else str(pose_landmarker_task_raw)
    )

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': invalid mode={mode!r}")

    crop_mode = (
        parse_crop_mode(params.get('crop_mode', 'trim'), node_id=node_id)
        if mode == 'default'
        else None
    )

    target = str(params.get('target', 'person'))
    if target not in SUBJECT_CROP_TARGETS:
        supported = ', '.join(repr(t) for t in sorted(SUBJECT_CROP_TARGETS))
        raise ValueError(
            f"'{node_id}': target={target!r} is not implemented by "
            f"SubjectCrop (supported: {supported})"
        )
    parsed_target = parse_target_specs(target, node_id=node_id)
    if target in SUBJECT_CROP_POSE_TARGETS and pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': target={target!r} requires "
            "model.pose_landmarker_task (MediaPipe .task path)"
        )

    conf = float(params.get('conf', 0.35))
    expansion = float(params.get('expansion', 1.0))
    if expansion <= 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} "
            '(expected > 0)'
        )

    box_margin = params.get('box_margin', '12%')
    validate_size_expr(box_margin)

    search_expansion = float(params.get('search_expansion', 0.35))
    if search_expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid search_expansion={search_expansion!r} "
            '(expected >= 0)'
        )

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))
    for name, value in (
        ('dilate_radius', dilate_radius),
        ('close_radius', close_radius),
        ('smoothing_radius', smoothing_radius),
    ):
        if value < 0:
            raise ValueError(
                f"'{node_id}': {name} must be >= 0, got {value}"
            )

    return Config(
        device=device,
        dtype=dtype,
        yolo_model=yolo_model,
        segment_model=segment_model,
        pose_landmarker_task=pose_landmarker_task,
        mode=mode,
        crop_mode=crop_mode,
        target=target,
        parsed_target=parsed_target,
        conf=conf,
        expansion=expansion,
        box_margin=box_margin,
        search_expansion=search_expansion,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        smoothing_radius=smoothing_radius,
        save_debug=bool(debug.get('save_debug', False)),
        shape_cleanup=read_shape_cleanup_config(
            params.get('postprocess', None),
            node_id=node_id,
        ),
    )


def _resolve_person_bbox(
    *,
    img_rgb: np.ndarray,
    cfg: Config,
    node_id: str,
) -> tuple[int, int, int, int]:
    """
    Resolve the Sapiens2 search bbox from YOLO person detections.

    The bbox is used as the Sapiens2 segmentation search region. Limb landmark
    geometry is resolved independently through MediaPipe pose when needed.
    """
    h, w = img_rgb.shape[:2]
    yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
    res = yolo.predict(
        img_rgb,
        conf=float(cfg.conf),
        verbose=False,
        device=cfg.device,
    )[0]

    person_boxes = person_bboxes_xyxy(res, node_id)
    x1, y1, x2, y2 = select_person_bbox_xyxy(
        person_boxes,
        pose_xy=None,
    )

    return expand_clip_bbox(
        x1,
        y1,
        x2,
        y2,
        w,
        h,
        cfg.search_expansion,
    )


def _predict_search_segments(
    image_rgb: Image.Image,
    *,
    bbox: tuple[int, int, int, int],
    cfg: Config,
    node_id: str,
) -> tuple[np.ndarray, torch.dtype]:
    """
    Run Sapiens2 segmentation inside ``bbox`` and paste ids into full frame.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"SubjectCrop node '{node_id}': invalid search bbox={bbox!r}."
        )

    crop = image_rgb.crop((x1, y1, x2, y2))
    local_segments, runtime_dtype = predict_segments(
        crop,
        model_id=cfg.segment_model,
        device=cfg.device,
        dtype=cfg.dtype,
        error_prefix='SubjectCrop',
    )

    full_h = int(image_rgb.height)
    full_w = int(image_rgb.width)
    segments = np.full(
        (full_h, full_w),
        SAPIENS2_CLASSES['background'],
        dtype=np.int64,
    )
    segments[y1:y2, x1:x2] = local_segments[:y2 - y1, :x2 - x1]
    return segments, runtime_dtype


def _predict_pose(
    img_rgb: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    cfg: Config,
    node_id: str,
) -> Optional[PoseContext]:
    """
    Run MediaPipe pose on the YOLO-selected person crop and return landmarks
    in full-frame image coordinates.
    """
    if cfg.pose_landmarker_task is None:
        return None

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"SubjectCrop node '{node_id}': invalid pose bbox={bbox!r}."
        )

    crop_rgb = np.ascontiguousarray(
        img_rgb[y1:y2, x1:x2, :]
    )

    pose_landmarker = get_mediapipe_pose_landmarker(
        model_asset_path=cfg.pose_landmarker_task,
        device=cfg.device,
    )
    local_pose = mp_pose_landmarks_full(
        crop_rgb,
        pose_landmarker,
        min_visibility=0.35,
        min_presence=0.35,
    )

    pose_xy = np.asarray(
        local_pose.xy,
        dtype=np.float32,
    ).copy()

    pose_z = np.asarray(
        local_pose.z,
        dtype=np.float32,
    ).copy()
    pose_xyz_px = np.asarray(
        local_pose.xyz_px,
        dtype=np.float32,
    ).copy()

    valid = (
        (pose_xy[:, 0] >= 0.0)
        & (pose_xy[:, 1] >= 0.0)
    )

    if not np.any(valid):
        raise RuntimeError(
            f"SubjectCrop node '{node_id}': MediaPipe returned no usable "
            'pose landmarks.'
        )

    pose_xy[valid, 0] += float(x1)
    pose_xy[valid, 1] += float(y1)

    # ``xyz_px`` is an image-space pseudo-3D representation, not MediaPipe's
    # metric ``world_xyz``. The pose detector ran on the person crop, so x/y need
    # the crop offset to become full-frame pixels. The z coordinate is a relative
    # depth scaled by the crop width and must not receive any spatial offset.
    pose_xyz_px[valid, 0] += float(x1)
    pose_xyz_px[valid, 1] += float(y1)

    return PoseContext(xy=pose_xy, z=pose_z, xyz_px=pose_xyz_px)


def _target_requires_pose(target: str) -> bool:
    return target in SUBJECT_CROP_POSE_TARGETS


def _normalize_pose_landmark_name(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace('-', '_')
        .replace(' ', '_')
    )


def _resolve_pose_landmark(
    pose: PoseContext,
    name: str,
) -> Optional[ResolvedLandmark]:
    """
    Resolve one MediaPipe pose landmark with optional depth information.

    Parameters
    ----------
    pose : PoseContext
        Full-frame MediaPipe pose coordinates.

    name : str
        Semantic MediaPipe landmark name.

    Returns
    -------
    ResolvedLandmark or None
        Landmark containing valid full-frame two-dimensional image coordinates
        and optional pseudo-3D image-space coordinates. ``None`` is returned
        when the two-dimensional landmark is unavailable or invalid. Invalid
        depth information affects only ``xyz_px`` and does not invalidate the
        two-dimensional landmark.

    Raises
    ------
    ValueError
        If ``name`` is not a known MediaPipe pose landmark.
    """
    normalized_name = _normalize_pose_landmark_name(name)

    idx = MEDIAPIPE_POSE_LANDMARKS.get(normalized_name)
    if idx is None:
        raise ValueError(
            f'Unknown MediaPipe pose landmark {name!r}.'
        )

    if idx >= pose.xy.shape[0]:
        return None

    xy = np.asarray(
        pose.xy[idx, :2],
        dtype=np.float32,
    )

    if (
        xy.shape != (2,)
        or not np.all(np.isfinite(xy))
        or float(xy[0]) < 0.0
        or float(xy[1]) < 0.0
    ):
        return None

    xyz_px = None

    if idx < pose.xyz_px.shape[0]:
        candidate_xyz = np.asarray(
            pose.xyz_px[idx, :3],
            dtype=np.float32,
        )

        if (
            candidate_xyz.shape == (3,)
            and np.all(np.isfinite(candidate_xyz))
        ):
            xyz_px = candidate_xyz.copy()

    return ResolvedLandmark(
        xy=xy.copy(),
        xyz_px=xyz_px,
    )


def _require_pose(
    ctx: HumanParseContext,
) -> PoseContext:
    if ctx.pose is None:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': "
            f"target={ctx.cfg.target!r} requires MediaPipe pose."
        )
    return ctx.pose


def _make_pose_landmark_resolver(
    ctx: HumanParseContext,
) -> LandmarkResolver:
    pose = _require_pose(ctx)

    def resolve(name: str) -> Optional[ResolvedLandmark]:
        return _resolve_pose_landmark(
            pose,
            name,
        )

    return resolve


def _limb_geometry_depth_segments(
    ctx: HumanParseContext,
    geometry: LimbCropGeometry,
) -> list[dict[str, Any]]:
    """
    Return consecutive limb segments with endpoint depth metadata.
    """
    pose = _require_pose(ctx)
    pose_z = np.asarray(pose.z, dtype=np.float32).reshape(-1)

    side_prefix = (
        'left'
        if str(geometry.side) == 'anatomical-left'
        else 'right'
    )

    def _point_with_depth(
        point: Optional[np.ndarray],
        point_name: str,
    ) -> Optional[tuple[np.ndarray, float]]:
        if point is None:
            return None

        point_xy = np.asarray(point, dtype=np.float32).reshape(-1)
        if point_xy.size < 2 or not np.all(np.isfinite(point_xy[:2])):
            return None

        landmark_index = MEDIAPIPE_POSE_LANDMARKS.get(
            _normalize_pose_landmark_name(f'{side_prefix}_{point_name}')
        )
        if landmark_index is None or landmark_index >= pose_z.size:
            return None

        depth = float(pose_z[landmark_index])
        if not math.isfinite(depth):
            return None

        return point_xy[:2].copy(), depth

    chain = [
        _point_with_depth(geometry.proximal_point, geometry.proximal_name),
        _point_with_depth(geometry.middle_point, geometry.middle_name),
        _point_with_depth(geometry.distal_point, geometry.distal_name),
    ]

    segments: list[dict[str, Any]] = []
    for index, pair in enumerate(((chain[0], chain[1]), (chain[1], chain[2]))):
        if pair[0] is None or pair[1] is None:
            continue

        (start_xy, start_z), (end_xy, end_z) = pair
        if float(np.linalg.norm(end_xy - start_xy)) < 2.0:
            continue

        segments.append({
            'index': int(index),
            'start_xy': start_xy,
            'end_xy': end_xy,
            'start_z': float(start_z),
            'end_z': float(end_z),
        })

    return segments


def _project_mask_on_segment_interval(
    mask: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Return the closed segment interval covered by mask pixels.
    """
    coords_yx = np.argwhere(np.asarray(mask).astype(bool))
    if coords_yx.size == 0:
        return None

    start = np.asarray(start_xy, dtype=np.float32).reshape(-1)[:2]
    end = np.asarray(end_xy, dtype=np.float32).reshape(-1)[:2]
    direction = end - start
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 1e-6:
        return None

    points_xy = np.stack(
        (
            coords_yx[:, 1].astype(np.float32),
            coords_yx[:, 0].astype(np.float32),
        ),
        axis=1,
    )
    t_values = np.clip(
        ((points_xy - start) @ direction) / length_sq,
        0.0,
        1.0,
    )
    return float(np.min(t_values)), float(np.max(t_values))


def _segment_interval_mean_depth(
    segment: dict[str, Any],
    interval: tuple[float, float],
) -> float:
    """
    Interpolate the mean depth of a segment interval.
    """
    start_z = float(segment['start_z'])
    end_z = float(segment['end_z'])
    t0, t1 = interval
    mean_t = 0.5 * (float(t0) + float(t1))
    return (
        (1.0 - mean_t) * start_z
        + mean_t * end_z
    )


def _pose_context_to_mediapipe_xy(
    pose: Optional[PoseContext],
) -> np.ndarray:
    if pose is None:
        return np.full((33, 2), -1, dtype=np.float32)
    return np.asarray(pose.xy, dtype=np.float32)


def _resolve_segment_target(
    ctx: HumanParseContext,
    *,
    target: str,
) -> ResolvedSegmentTarget:
    """
    Resolve target labels and image-relative side metadata without building a mask.

    Limb targets use custom semantic support masks, so they only need the public
    target metadata produced by ``resolve_parsed_target``. Callers that also
    need the boolean support mask should use ``_resolve_semantic_region``.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context containing the node id and the full-frame Sapiens2
        segmentation map used for image-relative side selection.

    target : str
        Public target name to resolve, for example ``'left-leg'`` or
        ``'hands'``.

    Returns
    -------
    ResolvedSegmentTarget
        Resolved public target metadata: selected label names, label ids,
        candidate names and optional image-relative side-resolution details.
    """
    parsed = parse_target_specs(
        target,
        node_id=ctx.node_id,
    )

    return resolve_parsed_target(
        parsed,
        node_id=ctx.node_id,
        segments=ctx.segments,
        error_prefix='SubjectCrop',
    )


def _resolve_semantic_region(
    ctx: HumanParseContext,
    *,
    target: str,
    extra_labels: tuple[str, ...] = (),
) -> tuple[np.ndarray, ResolvedSegmentTarget]:
    """
    Resolve a public Sapiens2 target and build its semantic support mask.

    ``resolved`` describes only the public target requested by the user.
    ``extra_labels`` enlarge the internal semantic support mask without changing
    the target metadata returned to downstream code.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context containing the Sapiens2 segmentation map.

    target : str
        Public Sapiens2 target to resolve.

    extra_labels : tuple[str, ...], optional
        Additional Sapiens2 labels included only in the internal support mask.
        They are not added to ``resolved.labels`` or
        ``resolved.label_ids``.

    Returns
    -------
    tuple[np.ndarray, ResolvedSegmentTarget]
        Boolean semantic support mask and resolved public target metadata.

    Raises
    ------
    RuntimeError
        If the resulting semantic support mask is empty.
    """
    resolved = _resolve_segment_target(
        ctx,
        target=target,
    )

    support_label_ids = list(dict.fromkeys([
        *resolved.label_ids,
        *(
            SAPIENS2_CLASSES[label]
            for label in extra_labels
        ),
    ]))

    mask = np.isin(
        ctx.segments,
        np.asarray(
            support_label_ids,
            dtype=np.int64,
        ),
    )

    if not np.any(mask):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': semantic region is empty "
            f'for target={target!r}.'
        )

    return mask, resolved


def _limb_geometry_landmark_chain(
    geometry: LimbCropGeometry,
) -> list[np.ndarray]:
    """
    Return valid landmarks from one limb geometry in proximal-to-distal order.

    Parameters
    ----------
    geometry : LimbCropGeometry
        Limb geometry containing proximal, middle, and distal landmark points.

    Returns
    -------
    list[np.ndarray]
        Available landmark points ordered from proximal joint to distal joint.
    """
    return [
        point
        for point in (
            geometry.proximal_point,
            geometry.middle_point,
            geometry.distal_point,
        )
        if point is not None
    ]


def _limb_geometry_chain_length(
    geometry: LimbCropGeometry,
) -> float:
    """
    Compute the proximal-to-distal landmark-chain length for a limb geometry.

    Parameters
    ----------
    geometry : LimbCropGeometry
        Limb geometry containing proximal, middle, and distal landmark points.

    Returns
    -------
    float
        Sum of available consecutive segment lengths in pixels.
    """
    points = (
        geometry.proximal_point,
        geometry.middle_point,
        geometry.distal_point,
    )

    total = 0.0
    for start, end in zip(
        points,
        points[1:],
    ):
        if start is None or end is None:
            continue

        total += float(
            np.linalg.norm(
                np.asarray(end, dtype=np.float32)
                - np.asarray(start, dtype=np.float32)
            )
        )

    return total


def _select_limb_side_by_proximal_landmark(
    *,
    ctx: HumanParseContext,
    requested: str,
    target_name: str,
    geometries: list[LimbCropGeometry],
) -> tuple[str, dict[str, Any]]:
    """
    Select a public left/right limb target by the proximal landmark position.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context used for contextual errors.

    requested : str
        Public image-relative side request: ``'left'``, ``'right'`` or
        ``'both'``.

    target_name : str
        Public target name being resolved, for example ``'left-leg'``.

    geometries : list[LimbCropGeometry]
        Candidate anatomical limb geometries.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Selected anatomical side name, or ``'both'``, and side-resolution
        metadata for the output sidecar.

    Raises
    ------
    RuntimeError
        If a side-specific target is requested but no usable proximal landmark
        exists.
    """
    if requested == 'both':
        return 'both', {}

    candidates: list[tuple[str, float]] = []
    for geometry in geometries:
        proximal = geometry.proximal_point
        if proximal is None:
            continue

        x = float(proximal[0])
        y = float(proximal[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < 0.0 or y < 0.0:
            continue

        candidates.append((geometry.side, x))

    if not candidates:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': no valid proximal "
            f'landmarks for image-relative target={target_name!r}.'
        )

    candidates = sorted(candidates, key=lambda item: item[1])
    selected_side, selected_x = (
        candidates[0] if requested == 'left' else candidates[-1]
    )

    return selected_side, {
        'target': target_name,
        'mode': 'image-relative-landmark-chain',
        'image_side': requested,
        'selected_candidate': selected_side,
        'selected_proximal_x': float(selected_x),
        'candidate_proximal_x': {
            side: float(x)
            for side, x in candidates
        },
    }


def _resolved_limb_target(
    *,
    ctx: HumanParseContext,
    target_kind: Literal['arm', 'leg'],
    public_target: str,
    selected_side: str,
    side_resolution: dict[str, Any],
) -> ResolvedSegmentTarget:
    """
    Build target metadata for a limb selected by image-relative landmarks.

    Parameters
    ----------
    ctx : HumanParseContext
        Runtime context used to resolve the public target when both sides are
        selected.

    target_kind : {'arm', 'leg'}
        Limb family used to map anatomical sides to Sapiens2 labels.

    public_target : str
        Public target string requested by the node.

    selected_side : str
        Selected anatomical side, or ``'both'``.

    side_resolution : dict[str, Any]
        Side-selection metadata to include in the result.

    Returns
    -------
    ResolvedSegmentTarget
        Target metadata containing labels, label ids, selected candidates and
        side-resolution records.
    """
    if selected_side == 'both':
        return _resolve_segment_target(
            ctx,
            target=public_target,
        )

    labels = list(_LIMB_LABELS_BY_SIDE[
        (target_kind, selected_side)
    ])
    label_ids = [
        int(SAPIENS2_CLASSES[label])
        for label in labels
    ]
    selected_candidate = f'{selected_side}-{target_kind}'
    side_meta = {
        **side_resolution,
        'selected_candidate': selected_candidate,
        'selected_label_ids': label_ids,
    }

    return ResolvedSegmentTarget(
        labels=labels,
        label_ids=label_ids,
        selected_candidates=[selected_candidate],
        side_resolutions=[side_meta],
    )


def _segment_arms(
    ctx: HumanParseContext,
) -> TargetSegmentationResult:
    """
    Segment one or both arms using topological Canny limb reconstruction.
    """
    requested_side = {
        'arms': 'both',
        'left-arm': 'left',
        'right-arm': 'right',
    }[ctx.cfg.target]
    return _segment_limb_target(
        ctx,
        target_kind='arm',
        semantic_labels=_ARM_SEMANTIC_SUPPORT_LABELS,
        requested_side=requested_side,
        geometry_builder=build_arm_crop_geometries,
        prompt_bbox_label='arm-crop',
    )


def _segment_legs(
    ctx: HumanParseContext,
) -> TargetSegmentationResult:
    """
    Segment one or both legs using topological Canny limb reconstruction.
    """
    requested_side = {
        'legs': 'both',
        'left-leg': 'left',
        'right-leg': 'right',
    }[ctx.cfg.target]
    return _segment_limb_target(
        ctx,
        target_kind='leg',
        semantic_labels=_LEG_SEMANTIC_SUPPORT_LABELS,
        requested_side=requested_side,
        geometry_builder=build_leg_crop_geometries,
        prompt_bbox_label='leg-crop',
    )


def _segment_target(
    ctx: HumanParseContext,
) -> TargetSegmentationResult:
    if ctx.cfg.target == 'head':
        return _segment_head(ctx)
    if ctx.cfg.target in ('arms', 'left-arm', 'right-arm'):
        return _segment_arms(ctx)
    if ctx.cfg.target in ('legs', 'left-leg', 'right-leg'):
        return _segment_legs(ctx)
    return _segment_simple_target(ctx)


def _segment_head(
    ctx: HumanParseContext,
) -> TargetSegmentationResult:
    """
    Resolve the semantic head mask but keep crop geometry face-size driven.

    Sapiens2's ``head`` target includes hair, which is useful for alpha but can
    over-expand the crop when long hair is visible. The ``face-neck`` label gives
    a stable center and lower bound, so the final mask is clipped to a
    geometry-driven head bbox and the reported ``target_bbox`` remains detached
    from long lateral hair.
    """
    head_support_mask, resolved = _resolve_semantic_region(
        ctx,
        target='head',
    )
    label_ids = resolved.label_ids

    part_masks = segment_part_masks(ctx.segments, label_ids)
    selected_mask = cleanup_shape_mask_by_parts(
        head_support_mask,
        part_masks,
        **ctx.cfg.shape_cleanup,
    )
    if not np.any(selected_mask):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': structural cleanup removed "
            "the entire selected mask for target='head'."
        )

    face_neck_label_id = int(SAPIENS2_CLASSES['face-neck'])
    face_neck_mask = (ctx.segments == face_neck_label_id) & head_support_mask
    if not np.any(face_neck_mask):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': semantic region is empty "
            "for target='face-neck', required to derive head crop geometry."
        )

    fx1, fy1, fx2, fy2 = tight_mask_bbox(face_neck_mask.astype(np.uint8))

    hair_label_id = int(SAPIENS2_CLASSES['hair'])
    hair_mask = (ctx.segments == hair_label_id) & head_support_mask
    top_mask = hair_mask if np.any(hair_mask) else face_neck_mask
    _, top_y, _, _ = tight_mask_bbox(top_mask.astype(np.uint8))

    face_neck_y, face_neck_x = np.nonzero(face_neck_mask)
    if face_neck_x.size == 0:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': cannot derive head crop "
            "center from empty face-neck mask."
        )

    bottom_y = int(fy2)
    height = max(1, bottom_y - int(top_y))
    center_x = float(np.median(face_neck_x))
    half_width = max(1.0, 0.4 * float(height))

    head_bbox = (
        max(0, int(math.floor(center_x - half_width))),
        max(0, int(top_y)),
        min(ctx.width, int(math.ceil(center_x + half_width))),
        min(ctx.height, bottom_y),
    )
    if head_bbox[2] <= head_bbox[0] or head_bbox[3] <= head_bbox[1]:
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': invalid derived head bbox "
            f'{head_bbox!r}.'
        )

    hx1, hy1, hx2, hy2 = head_bbox
    head_region_mask = np.zeros((ctx.height, ctx.width), dtype=bool)
    head_region_mask[hy1:hy2, hx1:hx2] = True
    selected_mask = selected_mask & head_region_mask
    if not np.any(selected_mask):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': semantic head mask does not "
            "overlap the face-derived head crop geometry."
        )

    return TargetSegmentationResult(
        mask=selected_mask,
        target_bbox=head_bbox,
        labels=resolved.labels,
        label_ids=label_ids,
        selected_candidates=resolved.selected_candidates,
        side_resolutions=[
            *resolved.side_resolutions,
            {
                'mode': 'hair-face-neck-derived-head-bbox',
                'face_neck_bbox_xyxy': [int(fx1), int(fy1), int(fx2), int(fy2)],
                'head_bbox_xyxy': [int(v) for v in head_bbox],
            },
        ],
        shape_part_masks=part_masks,
        debug_region_masks=[head_region_mask],
    )


def _segment_simple_target(
    ctx: HumanParseContext,
) -> TargetSegmentationResult:
    """
    Resolve direct semantic targets from Sapiens2 labels.
    """
    selected_mask, resolved = _resolve_semantic_region(
        ctx,
        target=ctx.cfg.target,
    )
    label_ids = resolved.label_ids

    part_masks = segment_part_masks(ctx.segments, label_ids)
    selected_mask = cleanup_shape_mask_by_parts(
        selected_mask,
        part_masks,
        **ctx.cfg.shape_cleanup,
    )
    if not np.any(selected_mask):
        raise RuntimeError(
            f"SubjectCrop node '{ctx.node_id}': structural cleanup removed "
            f"the entire selected mask for target={ctx.cfg.target!r}."
        )

    target_bbox = tight_mask_bbox(selected_mask.astype(np.uint8) * 255)

    return TargetSegmentationResult(
        mask=selected_mask,
        target_bbox=target_bbox,
        labels=resolved.labels,
        label_ids=label_ids,
        selected_candidates=resolved.selected_candidates,
        side_resolutions=resolved.side_resolutions,
        shape_part_masks=part_masks,
    )


def _pose_landmark_records(
    pose_xy: np.ndarray,
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for name, idx in MEDIAPIPE_POSE_LANDMARKS.items():
        if idx >= pose_xy.shape[0]:
            continue
        x = float(pose_xy[idx, 0])
        y = float(pose_xy[idx, 1])
        if x < 0.0 or y < 0.0:
            continue
        records.append({
            'index': int(idx),
            'name': name,
            'x': x,
            'y': y,
        })
    return records


@dataclass
class SubjectCrop(CudaPostRunMixin, NodeRef):
    """
    Subject-aware crop and inpaint-mask generator using YOLO, Sapiens2 semantic
    segmentation, and MediaPipe pose landmarks.

    ``SubjectCrop`` detects human body regions of interest and produces either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image
      (``mode='mask'`` or ``mode='negative-mask'``).

    Supported targets are:

    - ``person``: full visible subject crop/mask;
    - ``head``: semantic head crop/mask;
    - ``hands``: one or more visible hand crops/masks;
    - ``left-hand``: hand appearing on the left side of the image;
    - ``right-hand``: hand appearing on the right side of the image;
    - ``arms``: one or more visible arm crops/masks, from shoulder to wrist;
    - ``left-arm``: arm appearing on the left side of the image;
    - ``right-arm``: arm appearing on the right side of the image;
    - ``legs``: one or more visible leg crops/masks, from hip to ankle;
    - ``left-leg``: leg appearing on the left side of the image;
    - ``right-leg``: leg appearing on the right side of the image;
    - ``feet``: one or more visible foot crops/masks;
    - ``left-foot``: foot appearing on the left side of the image;
    - ``right-foot``: foot appearing on the right side of the image.

    Face-detail targets such as ``face``, ``eyes``, ``left-eye``,
    ``right-eye``, ``eyebrows``, ``left-eyebrow`` and ``right-eyebrow`` are
    handled by ``FaceCrop``. ``SubjectCrop`` is responsible for person
    selection, semantic body parsing, pose-guided limb geometry, and output crop
    or mask generation.

    The node is intended to support workflows such as:

    - extracting subjects for compositing (e.g. with ``ImageStack``);
    - producing full-frame inpaint masks for SDXL pipelines;
    - extracting semantic head, hand, foot, arm, or leg regions for localized
      repair/refinement;
    - refining a small region by cropping, processing it separately, and
      reinserting it at the original coordinates.

    Side-specific targets use image/viewer perspective.

    Hands and feet are selected from their visible semantic regions. Arms and legs
    resolve image-relative laterality from the horizontal position of their proximal
    pose landmarks:

    - ``left-arm`` selects the arm whose shoulder appears farther left;
    - ``right-arm`` selects the arm whose shoulder appears farther right;
    - ``left-leg`` selects the leg whose hip appears farther left;
    - ``right-leg`` selects the leg whose hip appears farther right.

    This proximal-landmark rule remains deterministic when limbs cross and should
    not be interpreted as selecting whichever complete limb silhouette occupies the
    leftmost or rightmost image area.

    This is intentionally different from MediaPipe anatomical landmark labels.
    Limb helpers resolve anatomical sides internally while exposing
    image/viewer-side target semantics.

    Pipeline
    --------

    The node combines object detection, semantic parsing, and pose geometry:

    - YOLO (COCO class 0) proposes a person search box. The selected box is
      expanded with ``search_expansion`` before Sapiens2 inference so semantic
      parsing has enough context around hands, feet, and loose silhouettes.

    - Sapiens2 segmentation runs inside the expanded search box and is pasted
      back into full-frame coordinates as a dense semantic class map.

    - Direct semantic targets such as ``person``, ``head``, ``hands`` and
      ``feet`` are resolved from the Sapiens2 class map, with optional
      image-side selection for side-specific hands and feet.

    - MediaPipe Pose runs only for topology-dependent limb targets:
      ``arms``, ``left-arm``, ``right-arm``, ``legs``, ``left-leg`` and
      ``right-leg``.

    - Limb targets build a broad bilateral semantic support region and select
      connected components touched by each anatomical landmark chain. Component
      selection is evaluated independently for each Sapiens2 semantic class so that
      touching skin and clothing labels do not collapse prematurely into one binary
      component.

    ``target='person'``
    - YOLO proposes a candidate person search bbox.
    - Sapiens2 segmentation is evaluated inside the expanded search bbox.
    - Person-related semantic classes are selected and cleaned.
    - The final crop region is derived from the selected mask and optionally
      expanded using ``box_margin``.

    ``target='head'``
    - YOLO proposes a candidate person search bbox.
    - Sapiens2 segmentation is evaluated inside the expanded search bbox.
    - Head-related semantic classes are selected directly from the semantic map.
    - The final crop region is derived from the selected head mask.

    ``target='hands'``
    - YOLO proposes a candidate person search bbox.
    - Sapiens2 segmentation is evaluated inside the expanded search bbox.
    - Hand semantic classes are selected and cleaned.
    - The final crop region isolates one or more visible hands.

    ``target='left-hand'`` and ``target='right-hand'``
    - Follow the same semantic pipeline while selecting only the hand appearing on
      the left or right side of the image, respectively.
    - Side selection follows the image/viewer perspective resolved by the shared
      semantic target-selection logic.

    ``target='arms'``
    - MediaPipe Pose landmarks are computed for the expanded person search
      region.
    - A broad bilateral arm support mask is built from Sapiens2 arm skin,
      torso, and upper-clothing classes so sleeves and shoulder attachments can
      be kept without relying exclusively on Sapiens2 anatomical laterality.
    - Each visible arm is associated with a shoulder-elbow-wrist landmark chain.
    - Connected support components are selected independently for each Sapiens2
      semantic class using that landmark chain. This prevents touching skin,
      clothing, torso, or opposite-arm regions from being merged prematurely
      into one binary component before side attribution.
    - The filtered semantic support is constrained by the arm tube, shoulder cap,
      and outward proximal quadrant before topological reconstruction.
    - A skeleton-label boundary pass can locally widen the arm core workspace
      where semantic-label borders leave and re-enter the geometric tube.
    - The arm tube, shoulder cap, and any semantic-edge relief form the core
      workspace; its boundary defines which free-space regions are external.
    - The filtered semantic support inside that workspace forms the closed core
      prior and contributes a semantic-support boundary barrier.
    - Canny edges, skeleton-label visual boundaries, the semantic-support
      boundary, and synthetic proximal and distal closures partition the core
      workspace into free-space regions.
    - Regions not touching the geometric core-workspace boundary are selected as
      the undilated arm core.
    - Partition-barrier thickness is restored without performing generic mask
      dilation.
    - When the two arm masks overlap, segment-capsule intersection and interpolated
      MediaPipe depth are used to remove from each target mask pixels attributed to
      the opposite arm when that opposite arm is reliably in front.
    - Selected arms are extended through the shoulder cap into the outward
      proximal quadrant by selecting proximal regions topologically continuous
      with the undilated core.
    - Opposite-arm overlap cleanup is repeated after proximal extension.

    ``target='left-arm'`` / ``target='right-arm'``
    - Follow the same pipeline while selecting only the arm whose shoulder
      appears on the left or right side of the image, respectively.
    - Side selection uses image/viewer perspective and is resolved from the
      horizontal position of the proximal shoulder landmark.

    ``target='legs'``
    - MediaPipe Pose landmarks are computed for the expanded person search
      region.
    - A broad bilateral leg support mask is built from Sapiens2 leg skin,
      generic apparel, and lower-clothing classes. Foot-specific classes are
      intentionally excluded.
    - Each visible leg is associated with a hip-knee-ankle landmark chain.
    - Connected support components are selected independently for each Sapiens2
      semantic class using that landmark chain. This prevents touching skin,
      clothing, or opposite-leg regions from being merged prematurely into one
      binary component before side attribution.
    - A skeleton-label boundary pass can locally widen the leg core workspace
      where semantic-label borders leave and re-enter the geometric tube.
    - The leg tube, hip cap, and any semantic-edge relief form the core
      workspace; its boundary defines which free-space regions are external.
    - The filtered semantic support inside that workspace forms the closed core
      prior and contributes a semantic-support boundary barrier.
    - Canny edges, skeleton-label visual boundaries, the semantic-support
      boundary, and synthetic proximal and distal closures partition the core
      workspace into free-space regions.
    - Regions not touching the geometric core-workspace boundary are selected as
      the undilated leg core.
    - Partition-barrier thickness is restored separately from topological region
      selection.
    - Segment-capsule overlap and interpolated MediaPipe depth are used to remove
      from each target mask pixels attributed to the opposite leg when reliable
      front/back evidence identifies that leg as being in front.
    - Selected legs are extended through the hip cap into the outward proximal
      quadrant by selecting proximal regions continuous with the undilated core.
    - Opposite-leg overlap cleanup is repeated after proximal extension.

    ``target='left-leg'`` / ``target='right-leg'``
    - Follow the same pipeline while selecting only the leg whose hip appears
      on the left or right side of the image, respectively.
    - Side selection uses image/viewer perspective and is resolved from the
      horizontal position of the proximal hip landmark.

    ``target='feet'``
    - YOLO proposes a candidate person search bbox.
    - Sapiens2 segmentation is evaluated inside the expanded search bbox.
    - Foot, shoe, and sock semantic classes are selected and cleaned.
    - The final crop region isolates one or more visible feet or footwear.

    ``target='left-foot'`` / ``target='right-foot'``
    - Follow the same semantic pipeline while selecting only the foot appearing
      on the left or right side of the image, respectively.
    - Side selection always follows image/viewer perspective.

    Local limb and extremity targets
    --------------------------------

    Hands, arms, legs, and feet are segmented using target-local semantics.
    Arms and legs additionally use pose-derived geometry to preserve the
    requested limb while limiting unrelated body parts and surrounding
    background.

    Unlike the person target, the crop geometry for local targets is determined
    by the selected anatomical region rather than by the overall subject bbox.

    Small artifacts or detached fragments may remain in difficult cases. They can
    be reduced through the node's structural ``postprocess`` configuration or by
    additional downstream mask processing.

    Parameters
    ----------
    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default
        input (``input['default']['image']`` or ``input['default']['path']``).

    spec : dict or str or Path, optional
        Node specification (inline dict or path to a config file), resolved via
        ``resolve_spec``.

    Expected structure:

    ``model`` : dict
        ``device`` : str, optional
            Inference device (e.g. ``'cuda'``, ``'cuda:0'``, ``'cpu'``).
            Default: ``'cuda'``.

        ``dtype`` : str, optional
            Torch dtype used to load the Sapiens2 segmentation model.
            Supported values follow ``resolve_dtype`` conventions, e.g.
            ``'bfloat16'``, ``'bf16'``, ``'float16'`` or ``'float32'``.
            Default: ``'bfloat16'``.

        ``segment_model`` : str, optional
            Hugging Face Sapiens2 segmentation model identifier used for dense
            body-part parsing. Default: ``'facebook/sapiens2-seg-0.4b'``.

        ``yolo_model`` : str, optional
            YOLO weights used to propose the person search box. Default:
            ``'yolov8n.pt'``.

        ``pose_landmarker_task`` : str, optional
            MediaPipe PoseLandmarker ``.task`` path. Required only for
            ``target='arms'``, ``target='left-arm'``,
            ``target='right-arm'``, ``target='legs'``,
            ``target='left-leg'`` and ``target='right-leg'``.

    ``params`` : dict
        ``target`` : {'person', 'head', 'hands', 'left-hand', 'right-hand', 'arms', 'left-arm', 'right-arm', 'legs', 'left-leg', 'right-leg', 'feet', 'left-foot', 'right-foot'}, optional
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

              The requested ratio is treated as a target, not a hard
              constraint. If the source image does not provide enough room near
              the borders, the final crop may deviate from the requested ratio.

            - ``'trim'``:
              Output is an RGBA cutout cropped to the selected bounding box,
              with alpha derived from the mask.

            - ``'full_frame'``:
              Same cutout as ``'trim'`` but placed back into a full-size RGBA
              canvas of the original image dimensions, preserving the original
              coordinates.

              Default: ``'trim'``.

        ``conf`` : float, optional
            YOLO confidence threshold. Typical range: 0.2-0.6.
            Default: 0.35.

        ``search_expansion`` : float, optional
            Proportional expansion applied to the selected YOLO person bbox
            before Sapiens2 segmentation and optional pose inference.
            Default: 0.35.

        ``box_margin`` : int or str, optional
            Symmetric margin applied to the resolved target crop bbox.

            For every target, the base bbox is derived from the final cleaned target mask.
            For arm and leg targets, this therefore reflects topological reconstruction,
            opposite-limb subtraction, proximal extension, and structural cleanup rather
            than the broader semantic processing geometry.

            Supported forms follow the standard size-expression convention: integer pixels,
            ``'<n>px'`` or ``'<n>%'``. Percentages are resolved against the target bbox
            width for left/right margins and its height for top/bottom margins.

            This margin is applied only to cropped default outputs
            (``crop_mode='trim'``, ``'bbox'`` or ``'bbox[w:h]'``). It is
            ignored for ``mode='mask'``, ``mode='negative-mask'`` and
            ``crop_mode='full_frame'`` because those outputs preserve the full
            source frame.

            Default: ``'12%'``.

        ``postprocess`` : dict, optional
            Structural cleanup applied to the selected target silhouette before
            deriving crop geometry, RGBA alpha, or full-frame mask output. This
            block defines the canonical shape used by the node, so it is
            applied in every mode. Output-only mask refinements such as
            ``close_radius``, ``dilate_radius`` and ``smoothing_radius`` are
            applied later and only for ``mode='mask'`` or
            ``mode='negative-mask'``.

            Processing order is fixed: ``fill_holes`` ->
            ``morph_open_radius`` -> ``min_component_area``.
            For logical multi-part targets such as ``hands``, ``arms``,
            ``legs`` and ``feet``, cleanup is applied independently to each
            target part before the parts are unioned. Therefore
            ``min_component_area='biggest'`` keeps the largest component per
            target part, rather than one component globally.

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
                filling. Opening removes thin lines, speckles, and small
                bridges while preserving surviving larger regions. ``0``
                disables this step.

            ``min_component_area`` : int, float, str, 'biggest' or None, optional
                Remove disconnected foreground components after hole filling
                and opening. ``0`` or ``None`` disables component filtering.
                Numeric values are pixel areas. Percentage strings use the same
                long-side area convention as ``fill_holes``.
                ``'biggest'`` keeps the largest connected component for each
                logical part. This may remove valid split limb fragments when
                semantic segmentation itself separates one limb into multiple
                foreground islands; use ``0`` or an explicit area threshold when
                preserving such splits matters more than aggressive cleanup.

            Default: all disabled.

        ``expansion`` : float, optional
            Expansion factor applied to target-local crop geometry.

            - for arm targets:
              expands the pose-derived shoulder cap and arm tube radius;
            - for leg targets:
              expands the pose-derived hip cap and leg tube radius;
            - for direct semantic targets:
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
            before inversion, producing a feathered transition between
            protected subject and repaintable background.

            Default: 0.

    ``debug`` : dict
        ``save_debug`` : bool, optional
            If True, saves the overview debug image and, for arm and leg targets,
            a structured topology-debug directory containing semantic priors,
            workspaces, Canny products, partition barriers, region mosaics,
            selected labels, proximal continuations, overlap subtractions, and
            final per-side masks.

    Mask post-processing
    --------------------

    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the detected target mask
    is post-processed before it is written to disk.

    Post-processing is always applied to the positive target mask first, before
    any optional polarity inversion:

    1. the selected target region is assembled as a full-frame positive mask;
    2. ``close_radius``, ``dilate_radius``, and ``smoothing_radius`` are
       applied;
    3. if ``mode='negative-mask'``, the post-processed target mask is inverted.

    This ordering is intentional.

    For ``mode='mask'``, the output mask directly marks the selected target
    region as repaintable. Dilation and smoothing therefore expand and soften
    the target region itself, which is useful when repainting or refining the
    selected target.

    For ``mode='negative-mask'``, the output mask marks the background as
    repaintable and protects the selected target. Applying dilation and
    smoothing before inversion creates a protected safety band around the
    target. This prevents the background inpaint area from bleeding into the
    target boundary.

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
            Resolved model/runtime metadata, including the selected YOLO model,
            Sapiens2 segmentation model, optional MediaPipe task path, device,
            requested dtype and runtime dtype.

        ``subject_crop`` : dict
            Target-specific metadata including person search bbox, selected
            Sapiens2 labels, side-resolution metadata, optional MediaPipe pose
            landmark count, and optional pose landmark coordinates.

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
            Single debug image path, present only when ``debug.save_debug`` is
            true and no structured debug directory is produced.

        ``debug_dir`` : str, optional
            Structured debug directory path, present for limb targets when
            ``debug.save_debug`` is true.

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
      to the positive target mask before output.
    - Side-specific hand, arm, leg and foot targets always follow image/viewer
      perspective.
    - Sapiens2 semantic left/right labels follow anatomical subject
      perspective and are mapped internally where necessary.
    - MediaPipe pose landmarks are required only for arm and leg targets.
    - For ``crop_mode='bbox[w:h]'``, the requested aspect ratio is treated as a
      target rather than a hard guarantee. Near image boundaries the final crop
      may deviate from the requested ratio.
    - If you change the implementation or specification and need fresh outputs,
      delete the existing sidecar JSON to avoid reusing cached results.
    """
    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        return is_cuda_device(spec.get('model', {}).get('device', 'cuda'))

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)
        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)
        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )

        with Image.open(img_path) as image:
            image_rgb = image.convert('RGB')

        w, h = image_rgb.size
        rgb = np.asarray(image_rgb, dtype=np.uint8)

        person_bbox = _resolve_person_bbox(
            img_rgb=rgb,
            cfg=cfg,
            node_id=node_id,
        )

        pose = None
        if _target_requires_pose(cfg.target):
            pose = _predict_pose(
                rgb,
                bbox=person_bbox,
                cfg=cfg,
                node_id=node_id,
            )

        segments, runtime_dtype = _predict_search_segments(
            image_rgb,
            bbox=person_bbox,
            cfg=cfg,
            node_id=node_id,
        )

        ctx = HumanParseContext(
            node_id=node_id,
            cfg=cfg,
            img_rgb=rgb,
            width=w,
            height=h,
            person_bbox=person_bbox,
            pose=pose,
            segments=segments,
            runtime_dtype=runtime_dtype,
        )
        segmentation = _segment_target(ctx)

        alpha_full = segmentation.mask.astype(np.uint8) * 255
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = segmentation.target_bbox

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=node_id,
            ext='png',
        )

        out_x1 = 0
        out_y1 = 0
        out_x2 = w
        out_y2 = h

        if cfg.mode == 'default':
            if cfg.crop_mode is None:
                raise RuntimeError(
                    f"{self.id}: crop_mode must be defined when mode='default'"
                )

            if cfg.crop_mode.mode == 'full_frame':
                output_image = Image.fromarray(
                    np.dstack([rgb, alpha_full]),
                    mode='RGBA',
                )

            else:
                x1, y1, x2, y2 = expand_clip_bbox_by_size_expr(
                    bbox_x1,
                    bbox_y1,
                    bbox_x2,
                    bbox_y2,
                    w,
                    h,
                    cfg.box_margin,
                )

                if cfg.crop_mode.mode == 'bbox':
                    if cfg.crop_mode.ratio is not None:
                        x1, y1, x2, y2 = expand_bbox_toward_ratio(
                            x1,
                            y1,
                            x2,
                            y2,
                            full_w=w,
                            full_h=h,
                            ratio=cfg.crop_mode.ratio,
                        )

                    out_x1 = int(x1)
                    out_y1 = int(y1)
                    out_x2 = int(x2)
                    out_y2 = int(y2)
                    crop_rgb = rgb[out_y1:out_y2, out_x1:out_x2, :]
                    alpha = np.full(
                        (out_y2 - out_y1, out_x2 - out_x1),
                        255,
                        dtype=np.uint8,
                    )
                    output_image = Image.fromarray(
                        np.dstack([crop_rgb, alpha]),
                        mode='RGBA',
                    )

                elif cfg.crop_mode.mode == 'trim':
                    out_x1 = int(x1)
                    out_y1 = int(y1)
                    out_x2 = int(x2)
                    out_y2 = int(y2)
                    crop_rgb = rgb[out_y1:out_y2, out_x1:out_x2, :]
                    crop_alpha = alpha_full[out_y1:out_y2, out_x1:out_x2]
                    output_image = Image.fromarray(
                        np.dstack([crop_rgb, crop_alpha]),
                        mode='RGBA',
                    )

                else:
                    raise ValueError(
                        f'{self.id}: invalid crop_mode={cfg.crop_mode.mode!r}'
                    )

        else:
            out_mask_u8 = prepare_output_mask(
                segmentation.mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                out_mask_u8 = 255 - out_mask_u8

            output_image = Image.fromarray(out_mask_u8, mode='L')

        output_image.save(out_path)

        dbg_path = None
        debug_dir_path = None
        if cfg.save_debug:
            if (
                segmentation.crop_regions
                or segmentation.debug_region_masks
                or segmentation.debug_edge_masks
            ):
                pose_xy = _pose_context_to_mediapipe_xy(pose)
                dbg_path = write_crop_debug_overlay(
                    img_rgb=rgb,
                    out_path=out_path,
                    target=cfg.target,
                    pose_xy=pose_xy,
                    crop_prompt_bbox=(
                        segmentation.prompt_bbox
                        or segmentation.target_bbox
                    ),
                    target_bbox=segmentation.target_bbox,
                    mask=segmentation.mask,
                    crop_regions=segmentation.crop_regions,
                    prompt_bbox_label=segmentation.prompt_bbox_label,
                    region_masks=segmentation.debug_region_masks,
                    edge_masks=segmentation.debug_edge_masks,
                )
            else:
                dbg_path = write_mask_debug_overlay(
                    img_rgb=rgb,
                    mask=segmentation.mask,
                    target_bbox=segmentation.target_bbox,
                    out_path=out_path,
                    target=cfg.target,
                )

            if segmentation.limb_debug is not None:
                debug_dir_path = write_limb_topology_debug_directory(
                    img_rgb=rgb,
                    out_path=out_path,
                    images=_limb_topology_debug_images(
                        ctx=ctx,
                        limb_debug=segmentation.limb_debug,
                        final_mask=segmentation.mask,
                    ),
                    overview_path=dbg_path,
                )

        bbox_w = int(out_x2 - out_x1)
        bbox_h = int(out_y2 - out_y1)
        anchor_x = int((out_x1 + out_x2) // 2)
        anchor_y = int((out_y1 + out_y2) // 2)

        params = {
            'target': cfg.target,
            'mode': cfg.mode,
            'crop_mode': None if cfg.crop_mode is None else cfg.crop_mode.raw,
            'expansion': cfg.expansion,
            'box_margin': cfg.box_margin,
            'search_expansion': cfg.search_expansion,
            'dilate_radius': cfg.dilate_radius,
            'close_radius': cfg.close_radius,
            'smoothing_radius': cfg.smoothing_radius,
            'postprocess': cfg.shape_cleanup,
        }

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'input_size': [int(w), int(h)],
            'output_size': [int(output_image.width), int(output_image.height)],
            'model': {
                'yolo_model': cfg.yolo_model,
                'segment_model': cfg.segment_model,
                'pose_landmarker_task': cfg.pose_landmarker_task,
                'device': cfg.device,
                'requested_dtype': str(cfg.dtype).replace('torch.', ''),
                'runtime_dtype': str(runtime_dtype).replace('torch.', ''),
            },
            'subject_crop': {
                **params,
                'person_bbox': [int(v) for v in person_bbox],
                'pose_landmarks_count': (
                    0
                    if pose is None
                    else int(np.count_nonzero(
                        (pose.xy[:, 0] >= 0) & (pose.xy[:, 1] >= 0)
                    ))
                ),
                'pose_landmarks': (
                    []
                    if pose is None
                    else _pose_landmark_records(pose.xy)
                ),
                'labels': segmentation.labels,
                'label_ids': [
                    int(label_id) for label_id in segmentation.label_ids
                ],
                'selected_candidates': segmentation.selected_candidates,
                'side_resolutions': segmentation.side_resolutions,
            },
            'params': params,
            'crop': {
                'anchor_xy': [
                    int(anchor_x - out_x1),
                    int(anchor_y - out_y1),
                ],
                'position': [anchor_x, anchor_y],
                'bbox_size': [bbox_w, bbox_h],
                'bbox_xyxy': [
                    int(out_x1),
                    int(out_y1),
                    int(out_x2),
                    int(out_y2),
                ],
            },
        }
        if dbg_path is not None and debug_dir_path is None:
            out['debug_bbox'] = str(dbg_path)
        if debug_dir_path is not None:
            out['debug_dir'] = str(debug_dir_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
