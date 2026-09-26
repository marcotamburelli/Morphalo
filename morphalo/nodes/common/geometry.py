import numpy as np


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
