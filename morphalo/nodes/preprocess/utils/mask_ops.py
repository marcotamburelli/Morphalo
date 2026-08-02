from typing import Any, Iterable, Optional

import cv2
import numpy as np

from morphalo.nodes.preprocess.utils import resolve_min_component_area


def invert_mask_inside_box(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Invert a mask only inside a prompt bounding box.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional boolean or binary mask.

    box : tuple[int, int, int, int]
        End-exclusive ``(x1, y1, x2, y2)`` box defining the only region where
        inversion is copied to the output.

    Returns
    -------
    np.ndarray
        Boolean mask with inverted values inside ``box`` and background
        everywhere outside it.
    """
    x1, y1, x2, y2 = box

    inv = ~mask
    out = np.zeros_like(mask, dtype=bool)
    out[y1:y2, x1:x2] = inv[y1:y2, x1:x2]

    return out


def remove_small_components(
    mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """
    Remove foreground connected components whose area is below a threshold.

    The input is treated as a binary foreground mask: non-zero values are
    foreground, zero values are background. Components are computed with
    8-connectivity. Components with area less than or equal to ``min_area`` are
    removed; larger components are preserved.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional mask. Boolean masks are returned as boolean masks;
        integer masks are returned as boolean masks suitable for indexing or
        conversion back to ``uint8`` by the caller.

    min_area : int
        Maximum area, in pixels, to remove. Values less than or equal to zero
        disable cleanup and return ``mask`` unchanged.

    Returns
    -------
    np.ndarray
        Boolean keep mask when cleanup is applied, or the original ``mask`` when
        ``min_area <= 0``.
    """
    if min_area <= 0:
        return mask

    if mask.ndim != 2:
        raise ValueError('Component mask must be HxW')

    cm = (mask != 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return mask

    keep = np.zeros(num, dtype=bool)
    keep[0] = False
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > int(min_area)
    return keep[labels]


def labeled_points_inside_mask(
    mask: np.ndarray,
    point_coords: list[list[float]],
    point_labels: list[int],
    *,
    target_label: int = 1,
) -> tuple[int, int]:
    """
    Count labeled point prompts that fall inside a mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or binary mask in the same coordinate system as the points.
    point_coords : list[list[float]]
        Point coordinates as ``[x, y]`` pairs.
    point_labels : list[int]
        Labels aligned with ``point_coords``.
    target_label : int, default=1
        Only points with this label are counted.

    Returns
    -------
    tuple[int, int]
        ``(inside, valid)`` where ``valid`` is the number of in-bounds points
        with ``target_label`` and ``inside`` is how many of them are foreground.
    """
    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')

    inside = 0
    valid = 0
    h, w = mask.shape[:2]
    m = mask.astype(bool)

    for point, label in zip(point_coords, point_labels):
        if int(label) != int(target_label) or len(point) < 2:
            continue

        px = int(round(float(point[0])))
        py = int(round(float(point[1])))
        if not (0 <= px < w and 0 <= py < h):
            continue

        valid += 1
        if m[py, px]:
            inside += 1

    return inside, valid


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest foreground connected component.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary foreground mask.

    Returns
    -------
    np.ndarray
        Boolean mask containing the largest foreground component. Empty masks
        and masks with a single foreground component are returned as boolean
        masks without further filtering.

    Raises
    ------
    ValueError
        If ``mask`` is not two-dimensional.
    """
    if mask.ndim != 2:
        raise ValueError('Component mask must be HxW')

    cm = (mask != 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask.astype(bool)

    target = 1 + int(np.argmax(areas))
    return labels == target


def keep_largest_components(
    mask: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    """
    Keep the ``count`` largest foreground connected components.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary mask.
    count : int
        Maximum number of foreground components to preserve.

    Returns
    -------
    np.ndarray
        Boolean mask containing up to ``count`` largest components.

    Raises
    ------
    ValueError
        If ``mask`` is not two-dimensional or ``count`` is less than one.
    """
    if mask.ndim != 2:
        raise ValueError('Component mask must be HxW')
    if count < 1:
        raise ValueError(f'count must be >= 1, got {count}.')

    cm = (mask != 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return mask.astype(bool)

    component_ids = np.arange(1, num)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask.astype(bool)

    order = np.argsort(areas)[::-1]
    keep_ids = component_ids[order[:int(count)]]
    return np.isin(labels, keep_ids)


def fill_mask_holes(
    mask: np.ndarray,
    *,
    max_area: int | None,
) -> np.ndarray:
    """
    Fill background holes that do not touch the image border.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional foreground mask. Non-zero pixels are treated as
        foreground.

    max_area : int | None
        Maximum enclosed background-component area to fill. ``None`` fills all
        enclosed holes regardless of area.

    Returns
    -------
    np.ndarray
        Boolean foreground mask with eligible enclosed holes filled.

    Raises
    ------
    ValueError
        If ``mask`` is not two-dimensional.
    """
    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')

    fg = mask.astype(bool)
    bg = (~fg).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        bg,
        connectivity=8,
    )
    if num <= 1:
        return fg

    h, w = fg.shape[:2]
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[h - 1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, w - 1]).tolist())

    out = fg.copy()
    for label in range(1, num):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if max_area is None or area <= int(max_area):
            out[labels == label] = True

    return out


def cleanup_shape_mask(
    mask: np.ndarray,
    *,
    fill_holes: Any = 0,
    morph_open_radius: int = 0,
    min_component_area: Any = 0,
) -> np.ndarray:
    """
    Clean a crop target shape mask before bbox/alpha/output derivation.

    Processing order:
      1. fill internal holes;
      2. apply morphological opening;
      3. remove small components or keep only the biggest component.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary mask to clean.

    fill_holes : Any, default=0
        Hole filling threshold. ``'all'`` fills every enclosed hole. Numeric
        values and percentage strings are resolved through
        ``resolve_min_component_area`` and fill holes up to the resulting area.
        ``0`` or ``None`` disables hole filling.

    morph_open_radius : int, default=0
        Radius in pixels for morphological opening. ``0`` disables opening.

    min_component_area : Any, default=0
        Component cleanup threshold. ``'biggest'`` keeps only the largest
        foreground component. Numeric values and percentage strings remove
        components up to the resolved area. ``0`` or ``None`` disables
        component filtering.

    Returns
    -------
    np.ndarray
        Boolean cleaned mask.

    Raises
    ------
    ValueError
        If ``mask`` is not two-dimensional or ``morph_open_radius`` is
        negative.
    """
    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')
    if morph_open_radius < 0:
        raise ValueError('morph_open_radius must be >= 0')

    out = mask.astype(bool)
    h, w = out.shape[:2]

    if fill_holes is not None:
        fill_all = isinstance(fill_holes, str) and fill_holes.strip() == 'all'
        fill_area = 0
        if fill_all:
            fill_area = None
        else:
            fill_area = resolve_min_component_area(
                fill_holes,
                width=w,
                height=h,
            )
        if fill_all or int(fill_area) > 0:
            out = fill_mask_holes(out, max_area=fill_area)

    if morph_open_radius > 0:
        k = 2 * int(morph_open_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            kernel,
        ) > 0

    if isinstance(min_component_area, str) and min_component_area.strip() == 'biggest':
        out = keep_largest_component(out)
    else:
        resolved_min_area = resolve_min_component_area(
            min_component_area,
            width=w,
            height=h,
        )
        out = remove_small_components(out, min_area=resolved_min_area)

    return out.astype(bool)


def cleanup_shape_mask_by_parts(
    mask: np.ndarray,
    parts: np.ndarray | Iterable[np.ndarray] | None,
    *,
    fill_holes: Any = 0,
    morph_open_radius: int = 0,
    min_component_area: Any = 0,
    largest_component_counts: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    Clean a shape mask independently inside logical target parts.

    Composite targets such as both feet, both hands, eyes, or eyebrows may be
    represented by multiple intentionally disconnected shapes. Running
    ``min_component_area='biggest'`` after their union would keep only one
    target. This helper applies the same cleanup to each logical part first,
    then unions the cleaned parts.

    ``largest_component_counts`` refines the meaning of
    ``min_component_area='biggest'`` per part. By default each part keeps only
    its largest component. Passing counts such as ``[2, 1]`` keeps the two
    largest components for the first logical part and the largest component for
    the second. Other cleanup steps, such as hole filling and morphological
    opening, still run before this largest-component filtering. This is useful
    when a valid target part may be split by occlusion while small residual
    fragments should still be discarded.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary shape mask to clean.

    parts : np.ndarray | iterable[np.ndarray] | None
        Logical part masks aligned with ``mask``. When an array is provided,
        each connected foreground component of that array is treated as one
        logical part. When an iterable is provided, each item is treated as one
        logical part. ``None`` falls back to ``cleanup_shape_mask``.

    fill_holes : Any, default=0
        Hole filling threshold passed to ``cleanup_shape_mask``. ``'all'``
        fills every enclosed hole; numeric values or percentage strings fill
        holes up to the resolved area.

    morph_open_radius : int, default=0
        Morphological opening radius in pixels.

    min_component_area : Any, default=0
        Component filtering threshold passed to ``cleanup_shape_mask``.
        ``'biggest'`` keeps the largest component per logical part unless
        ``largest_component_counts`` requests more components for that part.

    largest_component_counts : iterable[int] | None, default=None
        Optional per-part maximum number of largest components to keep when
        ``min_component_area='biggest'``. The count sequence must have the same
        length as the resolved logical part list. Counts are clamped to at
        least one. Ignored unless ``min_component_area`` is ``'biggest'``.

    Returns
    -------
    np.ndarray
        Boolean mask containing the union of cleaned logical parts.

    Raises
    ------
    ValueError
        If ``mask`` or any part mask is not two-dimensional/aligned, or if
        ``largest_component_counts`` does not match the number of parts.
    """
    if parts is None:
        return cleanup_shape_mask(
            mask,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    if mask.ndim != 2:
        raise ValueError('Shape mask must be HxW')

    shape = mask.shape
    source_masks: list[np.ndarray] = []

    if isinstance(parts, np.ndarray):
        if parts.shape != shape:
            raise ValueError('Shape part mask must match mask shape')
        pm = (parts != 0).astype(np.uint8)
        num, labels, _, _ = cv2.connectedComponentsWithStats(
            pm, connectivity=8)
        for label in range(1, num):
            source_masks.append(labels == label)
    else:
        for part in parts:
            if part.shape != shape:
                raise ValueError('Shape part mask must match mask shape')
            source_masks.append(part != 0)

    if not source_masks:
        return cleanup_shape_mask(
            mask,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    keep_biggest = (
        isinstance(min_component_area, str)
        and min_component_area.strip() == 'biggest'
    )

    if not keep_biggest or largest_component_counts is None:
        component_counts = [1 for _ in source_masks]
    else:
        component_counts = [
            max(1, int(count))
            for count in largest_component_counts
        ]
        if len(component_counts) != len(source_masks):
            raise ValueError(
                'largest_component_counts must match the number of shape '
                f'parts: got {len(component_counts)} counts for '
                f'{len(source_masks)} parts.'
            )

    base = (mask != 0)
    out = np.zeros(shape, dtype=bool)
    hit = False

    use_largest_counts = (
        keep_biggest
        and any(count > 1 for count in component_counts)
    )

    if use_largest_counts:
        cleanup_min_component_area: Any = 0
    else:
        cleanup_min_component_area = min_component_area

    for part_mask, component_count in zip(source_masks, component_counts):
        piece = base & part_mask
        if not np.any(piece):
            continue
        hit = True
        cleaned_piece = cleanup_shape_mask(
            piece,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=cleanup_min_component_area,
        )
        if use_largest_counts:
            cleaned_piece = keep_largest_components(
                cleaned_piece,
                count=component_count,
            )
        out |= cleaned_piece

    if not hit:
        return cleanup_shape_mask(
            mask,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    return out.astype(bool)


def prepare_output_mask(
    mask: np.ndarray,
    *,
    dilate_radius: int = 0,
    close_radius: int = 0,
    smoothing_radius: int = 0,
) -> np.ndarray:
    """
    Prepare a binary/soft shape mask for mask or negative-mask output.

    Processing order:

    1. morphological closing;
    2. dilation;
    3. Gaussian smoothing.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional input mask. Boolean masks are mapped to ``0``/``255``;
        floating masks are treated as ``0..1`` and scaled to ``0..255``;
        integer masks are cast to ``uint8``.

    dilate_radius : int, default=0
        Radius in pixels used to expand the mask outward. ``0`` disables
        dilation.

    close_radius : int, default=0
        Radius in pixels used for morphological closing. ``0`` disables
        closing.

    smoothing_radius : int, default=0
        Radius in pixels used for Gaussian blur. ``0`` disables smoothing.

    Returns
    -------
    np.ndarray
        ``uint8`` mask in range ``0..255``.

    Raises
    ------
    ValueError
        If ``mask`` is not two-dimensional.
    """

    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')

    # normalize to uint8 0..255
    if mask.dtype == bool:
        m = mask.astype(np.uint8) * 255
    elif np.issubdtype(mask.dtype, np.floating):
        m = np.clip(mask * 255, 0, 255).astype(np.uint8)
    else:
        m = mask.astype(np.uint8)

    # --- closing (fills holes) ---
    if close_radius > 0:
        k = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # --- dilation (expand mask) ---
    if dilate_radius > 0:
        k = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)

    # --- smoothing / feather ---
    if smoothing_radius > 0:
        k = 2 * smoothing_radius + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=0, sigmaY=0)

    return np.clip(m, 0, 255).astype(np.uint8)
