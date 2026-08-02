import numpy as np

from morphalo.nodes.preprocess.utils import SizeExpr, resolve_size_expr

Point = tuple[float, float]


def tight_mask_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return the tight end-exclusive bbox of non-zero mask pixels.

    Parameters
    ----------
    alpha : np.ndarray
        Two-dimensional mask or alpha channel. Pixels greater than zero are
        treated as foreground.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive ``(x1, y1, x2, y2)`` foreground bbox.

    Raises
    ------
    RuntimeError
        If ``alpha`` contains no foreground pixels.
    """
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError('Trimmed crop has no non-transparent pixels.')

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def union_bboxes_xyxy(
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


def clip_mask_to_bbox(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Keep mask pixels only inside an end-exclusive bbox.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame boolean mask returned by SAM.
    bbox : tuple[int, int, int, int]
        End-exclusive prompt bbox that defines the local foot domain.

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


def segment_capsule_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Rasterize a radius-expanded 2D segment as a boolean capsule mask.

    A segment capsule is the set of all pixels whose centers are within
    ``radius`` of the finite segment ``start -> end``. Geometrically, it is a
    rectangle around the segment plus circular caps at both endpoints.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Image shape. Only the first two dimensions are used as ``(H, W)``.
    start, end : np.ndarray
        Segment endpoints as ``(x, y)`` image coordinates.
    radius : float
        Radius in pixels around the segment.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(H, W)``.
    """
    h, w = image_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    axis = end - start
    axis_len_sq = float(np.dot(axis, axis))

    if axis_len_sq < 1e-6:
        dist = np.hypot(xx - float(start[0]), yy - float(start[1]))
        return dist <= float(radius)

    t = (
        ((xx - float(start[0])) * float(axis[0]))
        + ((yy - float(start[1])) * float(axis[1]))
    ) / axis_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    proj_x = float(start[0]) + t_clamped * float(axis[0])
    proj_y = float(start[1]) + t_clamped * float(axis[1])
    dist = np.hypot(xx - proj_x, yy - proj_y)
    return dist <= float(radius)


def expand_clip_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    w: int,
    h: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """
    Expand an end-exclusive bbox by a proportional margin and clip to image.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        End-exclusive bbox coordinates.
    w : int
        Image width in pixels.
    h : int
        Image height in pixels.
    margin : float
        Fraction of bbox width/height added to each side.

    Returns
    -------
    tuple[int, int, int, int]
        Expanded and clipped end-exclusive bbox.

    Raises
    ------
    RuntimeError
        If the input or expanded bbox is invalid.
    """
    # assume x2,y2 are end-exclusive after this normalization
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bbox: {(x1, y1, x2, y2)}")

    bw, bh = (x2 - x1), (y2 - y1)
    dx = int(round(bw * margin))
    dy = int(round(bh * margin))

    bx1 = max(0, x1 - dx)
    by1 = max(0, y1 - dy)
    bx2 = min(w, x2 + dx)
    by2 = min(h, y2 + dy)

    if bx2 <= bx1 or by2 <= by1:
        raise RuntimeError(f"Invalid expanded bbox: {(bx1, by1, bx2, by2)}")

    return bx1, by1, bx2, by2


def expand_clip_bbox_by_size_expr(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    w: int,
    h: int,
    margin: SizeExpr,
) -> tuple[int, int, int, int]:
    """
    Expand an end-exclusive bbox by a size expression and clip to image.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        End-exclusive bbox coordinates.
    w : int
        Image width in pixels.
    h : int
        Image height in pixels.
    margin : SizeExpr
        Pixel, percentage, or numeric size expression resolved independently
        against bbox width and height.

    Returns
    -------
    tuple[int, int, int, int]
        Expanded and clipped end-exclusive bbox.

    Raises
    ------
    RuntimeError
        If the input or expanded bbox is invalid.
    """
    # assume x2,y2 are end-exclusive after this normalization
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bbox: {(x1, y1, x2, y2)}")

    bw, bh = (x2 - x1), (y2 - y1)
    dx = resolve_size_expr(margin, reference=bw, min_size=0)
    dy = resolve_size_expr(margin, reference=bh, min_size=0)

    bx1 = max(0, x1 - dx)
    by1 = max(0, y1 - dy)
    bx2 = min(w, x2 + dx)
    by2 = min(h, y2 + dy)

    if bx2 <= bx1 or by2 <= by1:
        raise RuntimeError(f"Invalid expanded bbox: {(bx1, by1, bx2, by2)}")

    return bx1, by1, bx2, by2


def offset_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    dx: int,
    dy: int,
) -> tuple[int, int, int, int]:
    """
    Translate an end-exclusive bounding box by a constant offset.

    Parameters
    ----------
    bbox_xyxy : tuple[int, int, int, int]
        End-exclusive ``(x1, y1, x2, y2)`` bbox.
    dx : int
        Horizontal offset in pixels.
    dy : int
        Vertical offset in pixels.

    Returns
    -------
    tuple[int, int, int, int]
        Translated end-exclusive bbox.

    Raises
    ------
    RuntimeError
        If ``bbox_xyxy`` is invalid.
    """
    x1, y1, x2, y2 = bbox_xyxy

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f'Invalid bbox: {bbox_xyxy!r}')

    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def offset_landmarks_xy(
    xy: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> np.ndarray:
    """
    Translate landmark coordinates by a constant offset.

    Parameters
    ----------
    xy : np.ndarray
        Landmark coordinates with shape ``(N, 2)``.
    dx : int
        Horizontal offset in pixels.
    dy : int
        Vertical offset in pixels.

    Returns
    -------
    np.ndarray
        Translated landmark coordinates as ``int32`` with shape ``(N, 2)``.

    Raises
    ------
    ValueError
        If ``xy`` does not have shape ``(N, 2)``.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f'Invalid landmark array shape {xy.shape!r}; expected (N, 2).'
        )

    out = xy.astype(np.int32, copy=True)
    out[:, 0] += int(dx)
    out[:, 1] += int(dy)

    return out
