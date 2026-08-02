from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import cv2
import numpy as np

ImageSide = Literal['left', 'right']


@dataclass(frozen=True)
class ImageSideSelection:
    """
    Result of image-relative candidate-mask selection.

    Parameters
    ----------
    selected_name : str
        Name of the selected candidate.

    selected_index : int
        Position of the selected candidate in the input candidate sequence.

    selected_center_x : float
        Horizontal center of the selected candidate mask bbox in image
        coordinates.

    candidate_centers_x : dict[str, float]
        Horizontal bbox centers for all visible candidates considered during
        selection.

    image_side : {'left', 'right'}
        Requested image-relative side.
    """
    selected_name: str
    selected_index: int
    selected_center_x: float
    candidate_centers_x: dict[str, float]
    image_side: ImageSide


def mask_bbox_center_x(mask: np.ndarray) -> Optional[float]:
    """
    Compute the horizontal center of a boolean mask's bounding box.

    The returned value is measured in pixel coordinates and uses an
    end-exclusive bbox convention, matching the crop helpers in this package.
    Empty masks return ``None`` so callers can ignore invisible candidates.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional candidate mask.

    Returns
    -------
    float | None
        Horizontal bbox center, or ``None`` when the mask has no foreground.

    Raises
    ------
    ValueError
        If ``mask`` is not a two-dimensional array.
    """
    if mask.ndim != 2:
        raise ValueError(f'Candidate mask must be HxW, got {mask.shape!r}.')

    _, xs = np.where(mask)
    if xs.size == 0:
        return None

    return 0.5 * (float(xs.min()) + float(xs.max()) + 1.0)


def select_image_side_mask_candidate(
    candidates: Sequence[tuple[str, np.ndarray]],
    *,
    image_side: ImageSide,
    node_id: str,
    target_name: str,
    error_prefix: str,
) -> ImageSideSelection:
    """
    Select the visible candidate mask on the requested side of the image.

    This helper implements the project-wide convention for public
    ``left-*``/``right-*`` crop targets: side names are image/viewer-relative,
    even when the underlying detector exposes anatomical left/right labels.
    Callers provide the candidate pair masks already produced by their own
    model or landmark pipeline; this function only compares the horizontal
    bbox centers of the visible masks.

    Parameters
    ----------
    candidates : sequence[tuple[str, np.ndarray]]
        Ordered candidate names and masks. For side-specific anatomical sources
        this should normally contain the anatomical-left and anatomical-right
        candidate pair.

    image_side : {'left', 'right'}
        Image-relative side requested by the public target.

    node_id : str
        Node id used in error messages.

    target_name : str
        Public target name being resolved.

    error_prefix : str
        Human-readable node/error prefix, for example ``'Sapiens2SegmentCrop'``
        or ``'FaceCrop'``.

    Returns
    -------
    ImageSideSelection
        Selected candidate name, input index, selected center and all visible
        candidate centers.

    Raises
    ------
    ValueError
        If ``image_side`` is invalid or fewer than two candidates are provided.

    RuntimeError
        If no candidate is visible, or if only one candidate is visible but it
        lies on the opposite half of the image.
    """
    if image_side not in ('left', 'right'):
        raise ValueError(f"Invalid image_side={image_side!r}.")
    if len(candidates) < 2:
        raise ValueError(
            'Image-relative side selection requires at least two candidates.'
        )

    visible: list[tuple[int, str, float]] = []
    image_shape: Optional[tuple[int, int]] = None

    for index, (name, mask) in enumerate(candidates):
        if mask.ndim != 2:
            raise ValueError(
                f'Candidate mask must be HxW, got {mask.shape!r}.'
            )
        if image_shape is None:
            image_shape = (int(mask.shape[0]), int(mask.shape[1]))
        elif tuple(mask.shape) != image_shape:
            raise ValueError('Candidate masks must share the same shape.')

        center_x = mask_bbox_center_x(mask)
        if center_x is not None:
            visible.append((index, name, center_x))

    if not visible:
        raise RuntimeError(
            f"{error_prefix} node '{node_id}': no visible candidate for "
            f"image-relative target={target_name!r}."
        )

    visible = sorted(visible, key=lambda item: item[2])
    selected_index, selected_name, selected_center_x = (
        visible[0] if image_side == 'left' else visible[-1]
    )

    if len(visible) == 1:
        if image_shape is None:
            raise RuntimeError(
                f"{error_prefix} node '{node_id}': cannot infer image width "
                f"for target={target_name!r}."
            )
        image_mid_x = 0.5 * float(image_shape[1])
        if image_side == 'left' and selected_center_x > image_mid_x:
            raise RuntimeError(
                f"{error_prefix} node '{node_id}': could not reliably select "
                f"left-side target={target_name!r}; only visible candidate is "
                'on the right half of the image.'
            )
        if image_side == 'right' and selected_center_x < image_mid_x:
            raise RuntimeError(
                f"{error_prefix} node '{node_id}': could not reliably select "
                f"right-side target={target_name!r}; only visible candidate is "
                'on the left half of the image.'
            )

    return ImageSideSelection(
        selected_name=selected_name,
        selected_index=selected_index,
        selected_center_x=float(selected_center_x),
        candidate_centers_x={
            name: float(center_x)
            for _, name, center_x in visible
        },
        image_side=image_side,
    )


def _valid_xy(point: object) -> Optional[tuple[int, int]]:
    """
    Convert one point-like object to rounded non-negative pixel coordinates.

    Parameters
    ----------
    point : object
        Point-like value convertible to an array with at least two finite
        coordinates.

    Returns
    -------
    tuple[int, int] | None
        Rounded ``(x, y)`` coordinates, or ``None`` when the input is invalid,
        non-finite, too short, or has a negative coordinate.
    """
    try:
        arr = np.asarray(point, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None

    if arr.size < 2:
        return None

    x = float(arr[0])
    y = float(arr[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if x < 0.0 or y < 0.0:
        return None

    return int(round(x)), int(round(y))


def select_mask_components_intersecting_mask(
    mask: np.ndarray,
    *,
    selector_mask: np.ndarray,
) -> np.ndarray:
    """
    Keep mask components intersecting a selector mask.

    Components are computed on ``mask`` as one binary spatial domain. No semantic
    labels are considered, so adjacent pixels from different upstream classes
    remain part of the same component when they touch in ``mask``.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary mask whose connected components should be
        selected.

    selector_mask : np.ndarray
        Two-dimensional binary mask aligned with ``mask``. Components of
        ``mask`` intersecting this selector are kept.

    Returns
    -------
    np.ndarray
        Boolean mask containing the union of selected components. Returns an
        empty mask when no component intersects ``selector_mask``.

    Raises
    ------
    ValueError
        If the masks are not two-dimensional or are not shape-aligned.
    """
    if mask.ndim != 2:
        raise ValueError(f'mask must be HxW, got {mask.shape!r}.')
    if selector_mask.ndim != 2:
        raise ValueError(
            f'selector_mask must be HxW, got {selector_mask.shape!r}.'
        )
    if selector_mask.shape != mask.shape:
        raise ValueError(
            'selector_mask must have the same shape as mask, got '
            f'{selector_mask.shape!r} and {mask.shape!r}.'
        )

    mask_bool = mask.astype(bool, copy=False)
    selector_bool = selector_mask.astype(bool, copy=False)
    selected = np.zeros_like(mask_bool, dtype=bool)
    if not np.any(mask_bool) or not np.any(selector_bool):
        return selected

    component_count, component_labels = cv2.connectedComponents(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return selected

    touched_labels = np.unique(component_labels[selector_bool & mask_bool])
    touched_labels = touched_labels[touched_labels > 0]
    if touched_labels.size == 0:
        return selected

    return np.isin(component_labels, touched_labels)


def label_values_intersecting_mask(
    mask: np.ndarray,
    *,
    label_map: np.ndarray,
    selector_mask: np.ndarray,
) -> frozenset[int]:
    """
    Return label values touched by a selector inside a support mask.

    Unlike component-based selection, this helper returns semantic label ids
    rather than a fused mask. Callers can then build separate per-label masks and
    preserve boundaries between labels such as skin and clothing.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary support mask to filter.

    label_map : np.ndarray
        Two-dimensional integer-like label map aligned with ``mask``.

    selector_mask : np.ndarray
        Two-dimensional binary selector aligned with ``mask``. Label values
        intersecting this selector inside ``mask`` are returned.

    Returns
    -------
    frozenset[int]
        Label values touched by ``selector_mask`` inside ``mask``.

    Raises
    ------
    ValueError
        If inputs are not two-dimensional or are not shape-aligned.
    """
    support = np.asarray(mask).astype(bool)
    labels = np.asarray(label_map)
    selector = np.asarray(selector_mask).astype(bool)

    if support.ndim != 2 or labels.ndim != 2 or selector.ndim != 2:
        raise ValueError('mask, label_map, and selector_mask must be HxW')
    if support.shape != labels.shape or support.shape != selector.shape:
        raise ValueError('mask, label_map, and selector_mask must align')

    if not np.any(support) or not np.any(selector):
        return frozenset()

    touched_values = np.unique(labels[support & selector])
    if touched_values.size == 0:
        return frozenset()

    return frozenset(int(value) for value in touched_values)


def masks_for_label_values(
    mask: np.ndarray,
    *,
    label_map: np.ndarray,
    label_values: Iterable[int],
) -> tuple[np.ndarray, ...]:
    """
    Build one support mask per requested label value.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary support mask.

    label_map : np.ndarray
        Two-dimensional integer-like label map aligned with ``mask``.

    label_values : iterable[int]
        Label values to materialize as separate masks.

    Returns
    -------
    tuple[np.ndarray, ...]
        Boolean masks ``mask & (label_map == value)`` for every requested value
        that has at least one support pixel. The order follows sorted label
        values for deterministic downstream rendering and tests.
    """
    support = np.asarray(mask).astype(bool)
    labels = np.asarray(label_map)
    if support.ndim != 2 or labels.ndim != 2:
        raise ValueError('mask and label_map must be HxW')
    if support.shape != labels.shape:
        raise ValueError('mask and label_map must align')

    result: list[np.ndarray] = []
    for value in sorted({int(value) for value in label_values}):
        label_mask = support & (labels == int(value))
        if np.any(label_mask):
            result.append(label_mask)

    return tuple(result)


def filter_mask_components_by_constraint(
    mask: np.ndarray,
    *,
    constraint_mask: Optional[np.ndarray] = None,
    constraint_masks: Optional[Sequence[np.ndarray]] = None,
    preserve_on_empty: bool = True,
) -> np.ndarray:
    """
    Filter mask components by intersection with a constraint mask.

    This helper is useful after a broader selection step has produced a mask and
    a later stage needs to discard disconnected selected components that never
    touch a trusted reference. Components are computed on ``mask`` itself.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary mask whose components should be filtered.

    constraint_mask : np.ndarray | None, optional
        Single two-dimensional binary constraint mask aligned with ``mask``.

    constraint_masks : sequence[np.ndarray] | None, optional
        Multiple two-dimensional binary constraint masks aligned with ``mask``.
        They are unioned only for component filtering; callers that need
        per-mask boundaries should compute those boundaries before unioning.

    preserve_on_empty : bool, default=True
        When true, return ``mask`` unchanged if the constraint is empty or if no
        mask component intersects it. This is a conservative fallback for
        pipelines where an empty selector may indicate failed diagnostics rather
        than a real negative signal.

    Returns
    -------
    np.ndarray
        Boolean mask containing the kept components.
    """
    source = np.asarray(mask)
    if source.ndim != 2:
        raise ValueError(f'mask must be HxW, got {source.shape!r}.')

    constraints: list[np.ndarray] = []
    if constraint_mask is not None:
        constraints.append(np.asarray(constraint_mask).astype(bool))
    if constraint_masks is not None:
        constraints.extend(
            np.asarray(candidate).astype(bool)
            for candidate in constraint_masks
        )

    if not constraints:
        if preserve_on_empty:
            return source.astype(bool)
        return np.zeros_like(source, dtype=bool)

    constraint_union = np.zeros_like(source, dtype=bool)
    for constraint in constraints:
        if constraint.ndim != 2:
            raise ValueError(
                f'constraint masks must be HxW, got {constraint.shape!r}.'
            )
        if constraint.shape != source.shape:
            raise ValueError(
                'constraint masks must have the same shape as mask, got '
                f'{constraint.shape!r} and {source.shape!r}.'
            )
        constraint_union |= constraint

    selected = select_mask_components_intersecting_mask(
        mask,
        selector_mask=constraint_union,
    )
    if np.any(selected):
        return selected
    if preserve_on_empty:
        return np.asarray(mask).astype(bool)
    return selected


def reference_mask_coverage(
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
