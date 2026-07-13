import math
from dataclasses import dataclass, field
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
from morphalo.nodes.preprocess.segmentation import predict_sam_mask
from morphalo.nodes.preprocess.utils import (CropModeSpec, SizeExpr,
                                             cleanup_shape_mask,
                                             cleanup_shape_mask_by_parts,
                                             expand_bbox_toward_ratio,
                                             expand_clip_bbox_by_size_expr,
                                             expand_clip_bbox,
                                             invert_mask_inside_box,
                                             parse_crop_mode,
                                             positive_points_for_sam,
                                             prepare_output_mask,
                                             read_shape_cleanup_config,
                                             tight_alpha_bbox,
                                             validate_size_expr)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.chromatic_segmentation import (
    split_segment_by_chromatic_runs,
)
from morphalo.nodes.vision.face_region import (face_bbox_xyxy_from_landmarks,
                                               mp_face_landmarks)
from morphalo.nodes.vision.human import (FootSamRegion,
                                         crop_head_area_from_pose,
                                         foot_sam_region_with_prompt_bbox,
                                         feet_sam_regions_from_landmarks,
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
        'feet',
        'left-foot',
        'right-foot',
    ):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'person', "
            "'head', 'hands', 'left-hand', 'right-hand', 'feet', "
            "'left-foot' or 'right-foot'; use FaceCrop for face, eye, and "
            "eyebrow targets)"
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


@dataclass(frozen=True)
class SamMaskCandidate:
    """
    SAM mask candidate with metadata used by guided mask selection.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame boolean candidate mask.

    sam_score : float
        SAM predicted-IoU score associated with the raw mask.

    source : str
        Prompt strategy that produced the raw mask, such as ``'strict'``,
        ``'complete'``, ``'foot'`` or ``'refined-foot'``.

    inverted : bool, default=False
        Whether this candidate is the local inverse of the raw SAM mask.
    """
    mask: np.ndarray
    sam_score: float
    source: str
    inverted: bool = False


def _build_sam_candidates_from_raw_masks(
    masks: np.ndarray,
    scores: Optional[np.ndarray],
    *,
    bbox: tuple[int, int, int, int],
    source: str,
    node_id: str,
    include_inverted: bool = True,
) -> list[SamMaskCandidate]:
    """
    Convert raw SAM outputs into scored mask candidates.

    Parameters
    ----------
    masks : np.ndarray
        SAM candidate masks, expected as ``(N, H, W)`` boolean-like array.
    scores : np.ndarray or None
        Optional predicted-IoU scores aligned with ``masks``. Missing scores are
        treated as ``0.0``.
    bbox : tuple[int, int, int, int]
        End-exclusive prompt bbox used to compute local inverses.
    source : str
        Label describing the prompt strategy that produced these masks.
    node_id : str
        Node id used to build actionable error messages.
    include_inverted : bool, default=True
        If true, add the local inverse of each raw mask inside ``bbox`` as an
        additional candidate.

    Returns
    -------
    list[SamMaskCandidate]
        Full-frame boolean candidate masks with selection metadata.

    Raises
    ------
    RuntimeError
        If SAM returned no masks.
    """
    if masks is None or len(masks) == 0:
        raise RuntimeError(
            f"SubjectCrop node '{node_id}': SAM returned no masks."
        )

    candidates: list[SamMaskCandidate] = []

    for i, mask in enumerate(masks):
        mi = mask.astype(bool)
        score_i = (
            float(scores[i])
            if scores is not None and i < len(scores)
            else 0.0
        )

        candidates.append(SamMaskCandidate(
            mask=mi,
            sam_score=score_i,
            source=source,
            inverted=False,
        ))

        if include_inverted:
            candidates.append(SamMaskCandidate(
                mask=invert_mask_inside_box(mi, bbox),
                sam_score=score_i,
                source=source,
                inverted=True,
            ))

    return candidates


def _build_person_sam_candidates(
    *,
    img_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pose_xy: np.ndarray,
    processor: Any,
    model: Any,
    device: str,
    node_id: str,
    use_landmarks: bool = True,
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

    if use_landmarks:
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

        candidates.extend(_build_sam_candidates_from_raw_masks(
            masks,
            scores,
            bbox=bbox,
            source=source,
            node_id=node_id,
            include_inverted=True,
        ))

    return candidates


def _quantize_score(value: float, *, bins: int = 10) -> int:
    """
    Quantize a normalized score into an integer bucket.

    Quantization prevents tiny score differences from dominating later
    tie-breakers such as area preference or prompt source.

    Parameters
    ----------
    value : float
        Score expected in ``[0, 1]``. Values outside the range are clipped.

    bins : int, default=10
        Number of score intervals. The returned bucket is in ``[0, bins]``.

    Returns
    -------
    int
        Quantized score bucket.
    """
    clipped = min(1.0, max(0.0, float(value)))
    return int(round(clipped * float(bins)))


def _union_bboxes(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """
    Return the smallest bbox containing all input boxes.

    Parameters
    ----------
    boxes : list[tuple[int, int, int, int]]
        Non-empty list of end-exclusive ``(x1, y1, x2, y2)`` boxes.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive union bbox.

    Raises
    ------
    RuntimeError
        If ``boxes`` is empty or the union is invalid.
    """
    if not boxes:
        raise RuntimeError('Cannot union an empty bbox list.')

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid bbox union.')

    return x1, y1, x2, y2


def _rebuild_foot_regions_with_chromatic_prompt_points(
    *,
    img_rgb: np.ndarray,
    pose_xy: np.ndarray,
    foot_sam_regions: list[FootSamRegion],
) -> list[FootSamRegion]:
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

    This helper returns rebuilt ``FootSamRegion`` objects. Existing bbox/probe
    geometry is preserved, while ``point_coords`` / ``point_labels`` are
    regenerated through ``foot_sam_region_with_prompt_bbox`` so debug overlays
    and SAM receive the same final prompt.

    Parameters
    ----------
    img_rgb : np.ndarray
        Source RGB image.
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    foot_sam_regions : list[FootSamRegion]
        Foot prompt regions with base ankle/foot_index prompts.

    Returns
    -------
    list[FootSamRegion]
        Regions with chromatic positive points appended when any valid
        chromatic segment centers are found. If no enrichment is possible, the
        original region list is returned unchanged.
    """
    additional_positive_points_by_side: dict[str, list[list[float]]] = {}

    for region in foot_sam_regions:
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
        return foot_sam_regions

    return [
        foot_sam_region_with_prompt_bbox(
            region,
            pose_xy,
            prompt_bbox=region.prompt_bbox,
            additional_positive_points_by_side=additional_positive_points_by_side,
        )
        for region in foot_sam_regions
    ]


def _clip_foot_mask_to_bbox(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Keep foot mask pixels only inside an end-exclusive bbox.

    Foot probing intentionally compares masks only inside the local foot prompt
    bbox (the cyan debug box), never inside the full person box. This prevents a
    probe on the lower leg from being judged by unrelated body pixels.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame boolean mask returned by SAM.
    bbox : tuple[int, int, int, int]
        End-exclusive foot prompt bbox that defines the local foot domain.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as ``mask`` and all pixels outside
        ``bbox`` set to ``False``.
    """
    x1, y1, x2, y2 = bbox
    out = np.zeros_like(mask, dtype=bool)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def _foot_mask_coverage(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """
    Return how much of ``reference`` is covered by ``candidate``.

    This is intentionally not IoU. For the leg-probe heuristic, the question is:
    "does the mask produced by a lower-leg point cover the foot mask?" If the
    probe mask is very large, IoU could be low even when it fully contains the
    foot, while coverage still reports the continuity we care about.

    A high coverage value indicates containment/continuity, but does not imply
    that the candidate mask is spatially precise or compact.

    Parameters
    ----------
    reference : np.ndarray
        Boolean mask treated as the denominator, usually the foot mask.
    candidate : np.ndarray
        Boolean mask whose overlap with ``reference`` is measured, usually the
        leg-probe mask.

    Returns
    -------
    float
        ``area(reference & candidate) / area(reference)``. Returns ``0.0`` when
        ``reference`` is empty.
    """
    ref_area = int(np.count_nonzero(reference))
    if ref_area <= 0:
        return 0.0

    inter = int(np.count_nonzero(reference & candidate))
    return float(inter) / float(ref_area)


def _points_inside_count(
    mask: np.ndarray,
    points: Optional[list[list[float]]],
) -> tuple[int, int]:
    """
    Count how many point prompts fall inside a candidate mask.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame boolean mask.
    points : list[list[float]] or None
        Full-image point coordinates to evaluate.

    Returns
    -------
    tuple[int, int]
        ``(inside, valid)`` where ``inside`` is the number of valid points
        covered by the mask and ``valid`` is the number of usable points.
    """
    if not points:
        return 0, 0

    mask_h, mask_w = mask.shape
    inside = 0
    valid = 0

    for point in points:
        if len(point) < 2:
            continue

        px, py = point[:2]

        if px < 0 or py < 0:
            continue

        px_i = int(px)
        py_i = int(py)

        if not (0 <= px_i < mask_w and 0 <= py_i < mask_h):
            continue

        valid += 1

        if mask[py_i, px_i]:
            inside += 1

    return inside, valid


def _select_best_guided_sam_mask(
    candidates: list[SamMaskCandidate],
    *,
    bbox: tuple[int, int, int, int],
    required_points: Optional[list[list[float]]] = None,
    forbidden_points: Optional[list[list[float]]] = None,
    min_required_fraction: Optional[float] = 0.8,
    target_norm_area: float = 0.25,
    preferred_source: Optional[str] = None,
    context: str = 'SAM mask',
) -> np.ndarray:
    """
    Select the most suitable SAM mask using target-specific point constraints.

    SAM predicted-IoU estimates mask quality according to the model, but it does
    not necessarily reflect whether a candidate corresponds to the target
    intended by the caller. This selector therefore combines SAM scores with
    target-specific positive and negative anchors:

    - ``required_points`` identify locations that should belong to the selected
      target;
    - ``forbidden_points`` identify locations that should remain outside the
      selected target.

    Required-point filtering
    ------------------------
    When ``min_required_fraction`` is not ``None`` and valid required points are
    available, each candidate is evaluated by the fraction of required points
    it contains.

    If at least one candidate reaches the requested fraction, candidates that
    do not reach it are discarded.

    The filter is intentionally non-fatal. If no candidate reaches the requested
    fraction, selection falls back to the complete candidate set rather than
    failing or ranking candidates by partial required-point coverage. This
    preserves useful fallback behavior when landmarks or generated masks are
    noisy, incomplete, or slightly misaligned.

    When ``min_required_fraction`` is ``None``, required-point filtering is
    disabled.

    Forbidden-point filtering
    -------------------------
    Forbidden points are treated as exclusion constraints.

    If at least one remaining candidate excludes every valid forbidden point,
    candidates containing one or more forbidden points are discarded.

    If every remaining candidate contains at least one forbidden point, the
    filter is also non-fatal: all candidates remain eligible, but candidates
    containing fewer forbidden points are preferred during ranking.

    Candidate ranking
    -----------------
    After optional point filtering, candidates are ranked using the following
    criteria, in order:

    1. quantized SAM predicted-IoU score;
    2. proximity to the preferred normalized candidate area;
    3. fewer contained forbidden points when no candidate satisfies all
       forbidden-point constraints;
    4. preference for non-inverted candidates;
    5. preference for ``preferred_source`` when provided.

    Candidate area is measured inside ``bbox`` and normalized relative to the
    current candidate pool. A ``target_norm_area`` of ``0.0`` favors the
    smallest candidate, ``1.0`` favors the largest candidate, and intermediate
    values favor candidates between those extremes.

    Parameters
    ----------
    candidates : list[SamMaskCandidate]
        Candidate masks and associated SAM metadata.

    bbox : tuple[int, int, int, int]
        End-exclusive SAM prompt bbox ``(x1, y1, x2, y2)`` used to measure and
        normalize candidate areas.

    required_points : list[list[float]] or None, optional
        Positive target anchors in full-image coordinates. These points are
        evaluated only during candidate selection; they do not participate in
        candidate generation or SAM prompt construction.

    forbidden_points : list[list[float]] or None, optional
        Negative target anchors in full-image coordinates. Candidates excluding
        all valid forbidden points are preferred whenever such candidates exist.

    min_required_fraction : float or None, default=0.8
        Minimum fraction of valid ``required_points`` that a candidate must
        contain to pass required-point filtering.

        If at least one candidate reaches this fraction, only passing candidates
        remain eligible. If no candidate reaches it, the complete candidate set
        is retained.

        If ``None``, required-point filtering is skipped.

    target_norm_area : float, default=0.25
        Preferred normalized candidate area within the current candidate pool.
        Must be in ``[0, 1]``.

    preferred_source : str or None, optional
        Prompt-source label to prefer as the final ranking criterion.

    context : str, default='SAM mask'
        Human-readable target description used in error messages.

    Returns
    -------
    np.ndarray
        Selected full-frame boolean mask.

    Raises
    ------
    RuntimeError
        If ``candidates`` is empty or no candidate can be selected.

    ValueError
        If ``target_norm_area`` or a non-``None``
        ``min_required_fraction`` lies outside ``[0, 1]``.
    """
    if not candidates:
        raise RuntimeError(f'Cannot select best {context}: no candidates.')

    if not (0.0 <= float(target_norm_area) <= 1.0):
        raise ValueError(
            f'target_norm_area must be in [0, 1], got {target_norm_area!r}.'
        )

    if (
        min_required_fraction is not None
        and not (0.0 <= float(min_required_fraction) <= 1.0)
    ):
        raise ValueError(
            'min_required_fraction must be in [0, 1] or None, '
            f'got {min_required_fraction!r}.'
        )

    x1, y1, x2, y2 = bbox

    measured: list[tuple[SamMaskCandidate, int, int]] = []
    required_filtered: list[tuple[SamMaskCandidate, int, int]] = []

    for candidate in candidates:
        area = int(candidate.mask[y1:y2, x1:x2].sum())
        req_inside, req_valid = _points_inside_count(
            candidate.mask,
            required_points,
        )

        # Negative/forbidden points are supposed to remain outside the mask.
        # This count is therefore a violation count, not a positive score:
        #   0  -> candidate respects all forbidden points
        #   >0 -> candidate includes at least one point we wanted excluded
        # We keep the count on each row so it can first filter the pool, then
        # act as a softer penalty if every candidate violates a forbidden point.
        forbidden_inside, _ = _points_inside_count(
            candidate.mask,
            forbidden_points,
        )

        row = (candidate, area, forbidden_inside)
        measured.append(row)

        if min_required_fraction is not None and req_valid > 0:
            min_inside = max(
                1,
                int(math.ceil(float(req_valid) * min_required_fraction)),
            )

            if req_inside >= min_inside:
                required_filtered.append(row)

    pool = (
        required_filtered
        if min_required_fraction is not None and required_filtered
        else measured
    )

    if forbidden_points:
        # Prefer masks that exclude every forbidden point. This is the strong
        # negative-point behavior: when SAM gives us at least one candidate that
        # respects all negatives, candidates that include a negative point are
        # treated as wrong target hypotheses and removed from consideration.
        without_forbidden = [
            row for row in pool
            if row[2] == 0
        ]
        if without_forbidden:
            pool = without_forbidden

    areas = np.asarray(
        [area for _, area, _ in pool],
        dtype=np.float32,
    )

    min_area = float(np.min(areas))
    max_area = float(np.max(areas))
    area_span = max(1.0, max_area - min_area)

    best_candidate = None
    best_key = None

    for candidate, area, forbidden_inside in pool:
        norm_area = (float(area) - min_area) / area_span
        area_distance = abs(norm_area - float(target_norm_area))

        key = (
            _quantize_score(candidate.sam_score, bins=10),
            -float(area_distance),
            # If all remaining candidates include at least one forbidden point,
            # still prefer the least-bad mask by penalizing higher violation
            # counts. A smaller forbidden_inside value produces a larger key.
            -int(forbidden_inside),
            not candidate.inverted,
            candidate.source == preferred_source
            if preferred_source is not None
            else False,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(f'Failed to select best {context}.')

    return best_candidate.mask


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

    return _select_best_guided_sam_mask(
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

    return _select_best_guided_sam_mask(
        candidates,
        bbox=bbox,
        required_points=positive_points,
        forbidden_points=negative_points,
        min_required_fraction=1.0,
        target_norm_area=target_norm_area,
        preferred_source=preferred_source,
        context='foot SAM mask',
    )


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
    - extracting foot regions for localized foot repair/refinement;
    - refining a small region by cropping, processing it separately, and
      reinserting it at the original coordinates.

    Side-specific hand and foot targets use image/viewer perspective:

    - ``left-hand`` refers to the hand on the left side of the image;
    - ``right-hand`` refers to the hand on the right side of the image.
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
        ``target`` : {'person', 'head', 'hands', 'left-hand', 'right-hand', 'feet', 'left-foot', 'right-foot'}, optional
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
    - ``target='hands'`` and ``target='feet'`` may preserve multiple disconnected
      target components inside the same crop.
    - Side-specific hand and foot targets always follow image/viewer perspective.
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

        hand_mask: Optional[np.ndarray] = None
        foot_mask: Optional[np.ndarray] = None
        shape_part_masks: Optional[np.ndarray | list[np.ndarray]] = None
        foot_sam_regions: Optional[list[FootSamRegion]] = None
        debug_face_xy: Optional[np.ndarray] = None
        debug_hands_res = None
        debug_mask: Optional[np.ndarray] = None

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        pose_xy = mp_pose_landmarks_xy(
            img_rgb=img_rgb,
            pose_landmarker=pose_landmarker,
        )

        if cfg.target == 'person':
            # YOLO proposes person bbox candidates; resolve_person_bbox_xyxy()
            # accepts a YOLO box only when it is consistent with MediaPipe pose,
            # otherwise it falls back to a pose-derived bbox.
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

        elif cfg.target == 'head':
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            head_area_rgb, a_x, a_y = crop_head_area_from_pose(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
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
                image_shape=head_area_rgb.shape
            )

            fx1 = r_x1 + a_x
            fy1 = r_y1 + a_y
            fx2 = r_x2 + a_x
            fy2 = r_y2 + a_y

            fx1, fy1, fx2, fy2 = square_head_bbox_from_face_bbox(
                fx1, fy1, fx2, fy2, w, h,
                expansion=cfg.expansion,
            )

        elif cfg.target in ('hands', 'left-hand', 'right-hand'):
            # --------------------------------------------------
            # Resolve the subject bbox first.
            # This bbox is used only to guide SAM toward the correct person.
            # --------------------------------------------------

            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

            # --------------------------------------------------
            # Resolve the hand-local geometry from hand landmarks.
            #
            # - h* bbox defines the final crop region
            # - hand_mask is later intersected with the SAM subject mask
            #   so the final alpha stays hand-focused and background-free
            # --------------------------------------------------
            hand_landmarker = get_mediapipe_hand_landmarker(
                model_asset_path=cfg.hand_landmarker_task,
                device=cfg.device,
            )

            hands_res = mp_hand_landmarks_full(
                img_rgb=img_rgb,
                hand_landmarker=hand_landmarker,
            )
            debug_hands_res = hands_res

            hand_which = {
                'hands': 'both',
                'left-hand': 'left',
                'right-hand': 'right',
            }[cfg.target]

            hx1, hy1, hx2, hy2 = hands_bbox_xyxy_from_landmarks(
                hands_res,
                img_rgb.shape,
                which=hand_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            hand_mask = hands_mask_from_landmarks(
                hands_res,
                img_rgb.shape,
                which=hand_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

        elif cfg.target in ('feet', 'left-foot', 'right-foot'):
            # --------------------------------------------------
            # Resolve the subject bbox first.
            # This bbox is used only to guide SAM toward the correct person.
            # --------------------------------------------------

            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

            foot_which = {
                'feet': 'both',
                'left-foot': 'left',
                'right-foot': 'right',
            }[cfg.target]

            person_bbox_for_feet = (bx1, by1, bx2, by2)

            foot_sam_regions = feet_sam_regions_from_landmarks(
                pose_xy,
                img_rgb.shape,
                which=foot_which,
                expansion=max(1.0, float(cfg.expansion)),
                person_bbox=person_bbox_for_feet,
            )

            fx1, fy1, fx2, fy2 = _union_bboxes([
                region.base_bbox for region in foot_sam_regions
            ])
            bx1, by1, bx2, by2 = _union_bboxes([
                region.prompt_bbox for region in foot_sam_regions
            ])

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        # Optionally expand the SAM prompt bbox. This is an advanced segmentation
        # knob and is intentionally separate from output crop margin.
        if cfg.prompt_expansion > 0 and cfg.target in (
            'feet',
            'left-foot',
            'right-foot',
        ):
            if foot_sam_regions is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': foot regions not computed."
                )

            foot_sam_regions = [
                foot_sam_region_with_prompt_bbox(
                    region,
                    pose_xy,
                    prompt_bbox=expand_clip_bbox(
                        *region.prompt_bbox, w, h, cfg.prompt_expansion
                    ),
                )
                for region in foot_sam_regions
            ]
            bx1, by1, bx2, by2 = _union_bboxes([
                region.prompt_bbox for region in foot_sam_regions
            ])

        elif cfg.prompt_expansion > 0:
            bx1, by1, bx2, by2 = expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.prompt_expansion
            )

        if cfg.target in ('feet', 'left-foot', 'right-foot'):
            if foot_sam_regions is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': foot regions not computed."
                )

            # Rebuild the foot regions after the final prompt bboxes are known:
            # keep the bbox/probe geometry unchanged, but regenerate the SAM
            # point prompts with extra positives derived from chromatic/texture
            # consistency along the ankle->foot_index axis.
            foot_sam_regions = _rebuild_foot_regions_with_chromatic_prompt_points(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                foot_sam_regions=foot_sam_regions,
            )

            foot_mask = np.zeros((h, w), dtype=bool)
            for region in foot_sam_regions:
                rx1, ry1, rx2, ry2 = region.prompt_bbox
                foot_mask[ry1:ry2, rx1:rx2] = True

        # -----------------------------
        # Mask generation
        # -----------------------------
        sam_model_id = cfg.sam_model
        processor, sam_model = get_sam(
            model_id=sam_model_id,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        if cfg.target == 'person':
            # For full-body crops, neither strict nor complete prompting is universally
            # better:
            #
            # - strict candidates are cleaner but can miss weak silhouette regions
            # - complete candidates can preserve more silhouette but may attach artifacts
            #
            # We evaluate both families and select a conservative candidate using SAM
            # score, normalized area, and weak tie-breakers.
            candidates = _build_person_sam_candidates(
                img_rgb=img_rgb,
                bbox=(bx1, by1, bx2, by2),
                pose_xy=pose_xy,
                processor=processor,
                model=sam_model,
                device=cfg.device,
                node_id=node_id,
                use_landmarks=(cfg.min_landmark_fraction is not None),
            )

            mask = _select_best_person_sam_mask(
                candidates,
                bbox=(bx1, by1, bx2, by2),
                pose_xy=pose_xy,
                min_landmark_fraction=cfg.min_landmark_fraction,
            ).astype(bool)

        elif cfg.target in ('feet', 'left-foot', 'right-foot'):
            if foot_sam_regions is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': foot regions not computed."
                )

            mask = np.zeros((h, w), dtype=bool)
            foot_region_masks: list[np.ndarray] = []

            for region in foot_sam_regions:
                masks, scores = predict_sam_mask(
                    img_rgb=img_rgb,
                    bbox=region.prompt_bbox,
                    processor=processor,
                    model=sam_model,
                    device=cfg.device,
                    point_coords=region.point_coords,
                    point_labels=region.point_labels,
                )

                foot_candidates = _build_sam_candidates_from_raw_masks(
                    masks,
                    scores,
                    bbox=region.prompt_bbox,
                    source='foot',
                    node_id=node_id,
                    include_inverted=True,
                )
                region_mask = _select_best_foot_sam_mask(
                    foot_candidates,
                    bbox=region.prompt_bbox,
                    point_coords=region.point_coords,
                    point_labels=region.point_labels,
                    preferred_source='foot',
                )

                region_mask = _clip_foot_mask_to_bbox(
                    region_mask,
                    region.prompt_bbox,
                )

                if region.leg_probe_point is not None:
                    # Probe pass:
                    # Ask SAM what it segments when prompted only by the point above
                    # the ankle, inside the same local foot bbox. If this probe mask
                    # substantially covers the foot mask, lower leg and foot are
                    # visually continuous (bare skin, sandals, flip-flops), so the
                    # probe would be a harmful negative. If coverage is low, there is
                    # likely a real discontinuity (pants/socks/shoes), so we rerun the
                    # foot prompt with the probe point as a negative.
                    probe_masks, probe_scores = predict_sam_mask(
                        img_rgb=img_rgb,
                        bbox=region.prompt_bbox,
                        processor=processor,
                        model=sam_model,
                        device=cfg.device,
                        point_coords=[region.leg_probe_point],
                        point_labels=[1],
                    )
                    probe_candidates = _build_sam_candidates_from_raw_masks(
                        probe_masks,
                        probe_scores,
                        bbox=region.prompt_bbox,
                        source='leg-probe',
                        node_id=node_id,
                        include_inverted=True,
                    )
                    probe_mask = _select_best_guided_sam_mask(
                        probe_candidates,
                        bbox=region.prompt_bbox,
                        required_points=[region.leg_probe_point],
                        forbidden_points=None,
                        min_required_fraction=1.0,
                        target_norm_area=0.5,
                        preferred_source='leg-probe',
                        context='foot leg-probe SAM mask',
                    )
                    probe_mask = _clip_foot_mask_to_bbox(
                        probe_mask,
                        region.prompt_bbox,
                    )
                    probe_coverage = _foot_mask_coverage(region_mask, probe_mask)

                    if probe_coverage < FOOT_LEG_PROBE_COVERAGE_THRESHOLD:
                        refined_points = list(region.point_coords or [])
                        refined_labels = list(region.point_labels or [])
                        refined_points.append(region.leg_probe_point)
                        refined_labels.append(0)

                        refined_masks, refined_scores = predict_sam_mask(
                            img_rgb=img_rgb,
                            bbox=region.prompt_bbox,
                            processor=processor,
                            model=sam_model,
                            device=cfg.device,
                            point_coords=refined_points,
                            point_labels=refined_labels,
                        )
                        refined_candidates = _build_sam_candidates_from_raw_masks(
                            refined_masks,
                            refined_scores,
                            bbox=region.prompt_bbox,
                            source='refined-foot',
                            node_id=node_id,
                            include_inverted=True,
                        )
                        region_mask = _select_best_foot_sam_mask(
                            refined_candidates,
                            bbox=region.prompt_bbox,
                            point_coords=refined_points,
                            point_labels=refined_labels,
                            preferred_source='refined-foot',
                        )
                        region_mask = _clip_foot_mask_to_bbox(
                            region_mask,
                            region.prompt_bbox,
                        )

                foot_region_masks.append(region_mask)
                mask |= region_mask

            shape_part_masks = foot_region_masks

        else:
            point_coords = None
            point_labels = None

            if cfg.target in ('hands', 'left-hand', 'right-hand'):
                # Use body pose points intentionally: SAM is asked to segment the selected
                # person inside the person bbox. The target-local landmark mask is applied
                # later to restrict the result to the requested hand region.
                point_coords, point_labels = positive_points_for_sam(
                    xy=pose_xy,
                    bbox=(bx1, by1, bx2, by2),
                )

            masks, scores = predict_sam_mask(
                img_rgb=img_rgb,
                bbox=(bx1, by1, bx2, by2),
                processor=processor,
                model=sam_model,
                device=cfg.device,
                point_coords=point_coords,
                point_labels=point_labels,
            )

            if masks is None or len(masks) == 0:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': SAM returned no masks."
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
                    f"SubjectCrop node '{node_id}': failed to select a SAM mask."
                )

            mask = best_mask.astype(bool)

        if cfg.target != 'person' and cfg.target not in (
            'feet',
            'left-foot',
            'right-foot',
        ):
            valid = (
                (pose_xy[:, 0] >= 0) &
                (pose_xy[:, 1] >= 0)
            )

            pts = pose_xy[valid]

            if len(pts) > 0:
                inside = 0
                mask_h, mask_w = mask.shape

                for px, py in pts:
                    px = int(px)
                    py = int(py)

                    if 0 <= px < mask_w and 0 <= py < mask_h and mask[py, px]:
                        inside += 1

                min_inside = max(1, int(math.ceil(len(pts) * 0.5)))

                if inside < min_inside:
                    # SAM may occasionally return the local background instead of the prompted
                    # subject region. For non-person targets, we use pose landmark coverage as a
                    # cheap polarity sanity check: if too few pose points fall inside the mask, we
                    # invert the mask inside the prompt bbox.
                    mask = invert_mask_inside_box(
                        mask=mask,
                        box=(bx1, by1, bx2, by2)
                    )

        if cfg.target in ('hands', 'left-hand', 'right-hand'):
            if hand_mask is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': hand_mask not computed."
                )
            # Keep only the hand-local part of the SAM subject mask.
            # SAM separates subject vs background; the landmark mask constrains the
            # result to the selected hand region(s).
            mask = mask & hand_mask
            shape_part_masks = hand_mask

        if cfg.target == 'head':
            head_region_mask = np.zeros((h, w), dtype=bool)
            head_region_mask[fy1:fy2, fx1:fx2] = True

            # SAM is guided by the resolved person bbox for robustness, so its
            # raw mask can cover the whole subject. Restrict it back to the
            # target-local head box before deriving crop geometry from the mask.
            mask = mask & head_region_mask

        if cfg.target in ('feet', 'left-foot', 'right-foot'):
            if foot_mask is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': foot_mask not computed."
                )
            # Keep only the foot-local part of the SAM subject mask.
            # The expanded foot bbox is intentionally broad; SAM supplies the
            # subject-vs-background boundary inside it.
            mask = mask & foot_mask

        # --------------------------------------------------
        # Build the final positive mask and derive output geometry from it.
        # --------------------------------------------------

        if cfg.target == 'head':
            hint_x1, hint_y1, hint_x2, hint_y2 = fx1, fy1, fx2, fy2
        elif cfg.target in ('hands', 'left-hand', 'right-hand'):
            hint_x1, hint_y1, hint_x2, hint_y2 = hx1, hy1, hx2, hy2
        elif cfg.target in ('feet', 'left-foot', 'right-foot'):
            hint_x1, hint_y1, hint_x2, hint_y2 = bx1, by1, bx2, by2
        else:
            hint_x1, hint_y1, hint_x2, hint_y2 = bx1, by1, bx2, by2

        cm = mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm, connectivity=8
        )

        if num > 1:
            if cfg.target in (
                'hands',
                'left-hand',
                'right-hand',
                'feet',
                'left-foot',
                'right-foot',
            ):
                # Keep all target-local components. This preserves paired hands,
                # paired feet, and small disconnected extremity fragments.
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
                crop_x1, crop_y1, crop_x2, crop_y2 = tight_alpha_bbox(
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
            dbg_x1, dbg_y1, dbg_x2, dbg_y2 = bx1, by1, bx2, by2
            prompt_bbox_label = 'person'
            if cfg.target == 'head':
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = fx1, fy1, fx2, fy2
            elif cfg.target in ('hands', 'left-hand', 'right-hand'):
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = hx1, hy1, hx2, hy2
            elif cfg.target in ('feet', 'left-foot', 'right-foot'):
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = fx1, fy1, fx2, fy2
                prompt_bbox_label = 'foot-prompt'

            dbg_path = write_crop_debug_overlay(
                img_rgb=img_rgb,
                out_path=out_path,
                target=cfg.target,
                pose_xy=pose_xy,
                sam_prompt_bbox=(bx1, by1, bx2, by2),
                target_bbox=(dbg_x1, dbg_y1, dbg_x2, dbg_y2),
                hands_res=debug_hands_res,
                face_xy=debug_face_xy,
                mask=debug_mask,
                foot_sam_regions=foot_sam_regions,
                prompt_bbox_label=prompt_bbox_label,
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
