import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from morphalo.nodes.preprocess.utils.mask_ops import invert_mask_inside_box


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


def predict_sam_mask(
    *,
    img_rgb: np.ndarray,
    bbox: Optional[tuple[int, int, int, int]],
    processor: Any,
    model: Any,
    device: str,
    point_coords: Optional[list[list[float]]] = None,
    point_labels: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict SAM mask candidates from optional bbox and/or point prompts.

    This helper targets the Hugging Face SAM / SAM-HQ processor API, where mask
    post-processing requires both ``original_sizes`` and ``reshaped_input_sizes``.
    SAM2-style processors are intentionally not handled here.
    """
    image = Image.fromarray(np.ascontiguousarray(img_rgb))

    kwargs: dict[str, Any] = {
        'images': image,
        'return_tensors': 'pt',
    }

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        kwargs['input_boxes'] = [[[float(x1), float(y1), float(x2), float(y2)]]]

    if point_coords is not None and point_labels is not None:
        kwargs['input_points'] = [[point_coords]]
        kwargs['input_labels'] = [[point_labels]]

    inputs = processor(**kwargs)

    model_dtype = next(model.parameters()).dtype

    inputs = {
        key: (
            value.to(device=device, dtype=model_dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value.to(device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    masks_t = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().float().cpu(),
        inputs['original_sizes'].detach().cpu(),
        inputs['reshaped_input_sizes'].detach().cpu(),
    )[0]

    masks_np = masks_t.numpy()

    if masks_np.ndim == 4:
        masks_np = masks_np[0]
    elif masks_np.ndim != 3:
        raise RuntimeError(f'Unexpected SAM mask shape: {masks_np.shape!r}')

    scores_np = (
        outputs.iou_scores
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    return masks_np.astype(bool), scores_np.astype(np.float32)


def build_sam_candidates_from_raw_masks(
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


def quantize_score(value: float, *, bins: int = 10) -> int:
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


def points_inside_count(
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


def select_best_guided_sam_mask(
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
        req_inside, req_valid = points_inside_count(
            candidate.mask,
            required_points,
        )

        # Negative/forbidden points are supposed to remain outside the mask.
        # This count is therefore a violation count, not a positive score:
        #   0  -> candidate respects all forbidden points
        #   >0 -> candidate includes at least one point we wanted excluded
        # We keep the count on each row so it can first filter the pool, then
        # act as a softer penalty if every candidate violates a forbidden point.
        forbidden_inside, _ = points_inside_count(
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
            quantize_score(candidate.sam_score, bins=10),
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
