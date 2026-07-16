from dataclasses import dataclass
from typing import Literal, Optional, Sequence

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
