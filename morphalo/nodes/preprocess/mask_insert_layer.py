import time
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.mask_geometry import (
    estimate_mask_centerline_alignment_angle,
    estimate_mask_quad_alignment_angle)
from morphalo.nodes.preprocess.utils import (resolve_size_expr,
                                             validate_size_expr,
                                             validate_percentage_size_expr)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

Anchor = Literal[
    'center',
    'top',
    'bottom',
    'left',
    'right',
    'top-left',
    'top-right',
    'bottom-left',
    'bottom-right',
]
SizeExpr = Union[int, str]
ResizeMode = Literal['fit', 'cover', 'stretch']
NormalizedBoxBlend = Tuple[float, float, float, float]

ALLOWED_ANCHORS = {
    'center',
    'top',
    'bottom',
    'left',
    'right',
    'top-left',
    'top-right',
    'bottom-left',
    'bottom-right',
}


@dataclass(frozen=True)
class LayerSpec:
    """
    Runtime specification for the overlay image attached to MaskInsertLayer.

    Parameters
    ----------
    position : Anchor
        Placement of the fitted overlay inside the local rectangle content box.

        Accepted values are:

        - ``'center'``
        - ``'top'``
        - ``'bottom'``
        - ``'left'``
        - ``'right'``
        - ``'top-left'``
        - ``'top-right'``
        - ``'bottom-left'``
        - ``'bottom-right'``

    resize : {'fit', 'cover', 'stretch'}
        Resize mode used to adapt the overlay image to the local rectangle.
    rotation_deg : float
        Manual clockwise rotation applied to the overlay before resizing and
        placement, in local rectangle coordinates.
    """

    position: Anchor = 'center'
    resize: ResizeMode = 'fit'
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class RotatedRect:
    """
    Rotated rectangle fitted inside a binary mask.

    Parameters
    ----------
    cx : float
        Rectangle center x-coordinate in its current coordinate system.
    cy : float
        Rectangle center y-coordinate in its current coordinate system.
    width : float
        Rectangle width in pixels.
    height : float
        Rectangle height in pixels.
    angle_deg : float
        Rectangle angle in degrees, measured in image coordinates.
    corners : list[tuple[float, float]]
        Rectangle corners in the same coordinate system, ordered clockwise.
    """

    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float
    corners: List[Tuple[float, float]]


@dataclass(frozen=True)
class RectFitResult:
    """
    Result of fitting a placement rectangle inside a binary mask.

    Parameters
    ----------
    rect : RotatedRect
        Final rectangle selected for placement.
    inner_rect : RotatedRect
        Largest fully-inscribed rectangle.
    outer_rect : RotatedRect
        Rotated foreground bounding box.
    alignment_angle : float
        Alignment angle used for fitting, in degrees.
    alignment_method : str
        Name of the alignment estimator used to compute ``alignment_angle``.
    """

    rect: RotatedRect
    inner_rect: RotatedRect
    outer_rect: RotatedRect
    alignment_angle: float
    alignment_method: str


@dataclass
class Config:
    """
    Runtime configuration for MaskInsertLayer.

    Parameters
    ----------
    threshold : int
        Grayscale threshold used to binarize the mask.

    inset_left : int or str
        Left content inset applied to the fitted rectangle before placing the
        overlay. Positive values move the content edge inward; negative values
        expand it outward.

    inset_right : int or str
        Right content inset applied to the fitted rectangle before placing the
        overlay. Positive values move the content edge inward; negative values
        expand it outward.

    inset_top : int or str
        Top content inset applied to the fitted rectangle before placing the
        overlay. Positive values move the content edge inward; negative values
        expand it outward.

    inset_bottom : int or str
        Bottom content inset applied to the fitted rectangle before placing the
        overlay. Positive values move the content edge inward; negative values
        expand it outward.

    box_blend : tuple[float, float, float, float]
        Normalized per-side blend factors between the largest fully-inscribed
        rectangle and the rotated foreground bounding box.

        Values are stored in ``(left, top, right, bottom)`` order.

        ``0.0`` keeps the corresponding side from the strict inner rectangle.
        ``1.0`` moves the corresponding side to the rotated foreground bounding
        box.
        Intermediate values expand that side toward the outer box.

    max_bridge_distance : str or None
        Maximum percentage distance used to connect nearby mask components before
        fitting.

    fill_holes : bool
        If true, fill enclosed holes in the fitting mask.

    save_debug : bool
        If true, save a debug image showing the fitted rectangle.
    """

    threshold: int
    inset_left: SizeExpr
    inset_right: SizeExpr
    inset_top: SizeExpr
    inset_bottom: SizeExpr
    box_blend: NormalizedBoxBlend
    max_bridge_distance: Optional[str]
    fill_holes: bool
    save_debug: bool


def _normalize_box_blend(
    value: Any,
    *,
    node_id: str,
) -> NormalizedBoxBlend:
    """
    Normalize a box blend specification to per-side factors.

    Parameters
    ----------
    value : Any
        Blend specification. Accepted forms are:

        - ``float``:
          Same blend factor for all sides.

        - ``(horizontal, vertical)``:
          Horizontal factor is applied to left/right sides; vertical factor is
          applied to top/bottom sides.

        - ``(left, top, right, bottom)``:
          Per-side blend factors.

    node_id : str
        Node id used in error messages.

    Returns
    -------
    tuple[float, float, float, float]
        Normalized blend factors in ``(left, top, right, bottom)`` order.

    Raises
    ------
    ValueError
        If the value cannot be interpreted or if any factor is outside ``[0, 1]``.
    """

    if isinstance(value, (int, float)):
        vals = (float(value),) * 4

    elif isinstance(value, (list, tuple)):
        if len(value) == 2:
            horizontal = float(value[0])
            vertical = float(value[1])
            vals = (
                horizontal,
                vertical,
                horizontal,
                vertical,
            )
        elif len(value) == 4:
            vals = tuple(float(x) for x in value)
        else:
            raise ValueError(
                f"'{node_id}': invalid box_blend={value!r}; "
                'expected a number, a 2-item sequence, or a 4-item sequence.'
            )

    else:
        raise ValueError(
            f"'{node_id}': invalid box_blend={value!r}; "
            'expected a number, a 2-item sequence, or a 4-item sequence.'
        )

    if any(not isfinite(x) or not 0.0 <= x <= 1.0 for x in vals):
        raise ValueError(
            f"'{node_id}': invalid box_blend={value!r}; "
            'all blend factors must be finite values in [0, 1].'
        )

    return vals


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Read and validate node configuration.

    Parameters
    ----------
    spec : dict
        Resolved node specification.
    node_id : str
        Node id used in error messages.

    Returns
    -------
    Config
        Validated runtime configuration.
    """

    params = spec.get('params', {})
    debug = spec.get('debug', {})

    threshold = int(params.get('threshold', 128))
    if not 0 <= threshold <= 255:
        raise ValueError(
            f"'{node_id}': invalid threshold={threshold!r}; expected 0..255."
        )

    box_blend = _normalize_box_blend(
        params.get('box_blend', 0.0),
        node_id=node_id,
    )

    return Config(
        threshold=threshold,
        inset_left=params.get('inset_left', '0%'),
        inset_right=params.get('inset_right', '0%'),
        inset_top=params.get('inset_top', '0%'),
        inset_bottom=params.get('inset_bottom', '0%'),
        box_blend=box_blend,
        max_bridge_distance=validate_percentage_size_expr(
            params.get('max_bridge_distance', None),
            node_id=node_id,
            name='params.max_bridge_distance',
        ),
        fill_holes=bool(params.get('fill_holes', False)),
        save_debug=bool(debug.get('save_debug', False)),
    )


def _load_mask(mask_path: str, *, threshold: int) -> np.ndarray:
    """
    Load and binarize a mask image.

    Parameters
    ----------
    mask_path : str
        Filesystem path to the mask image.
    threshold : int
        Threshold used to binarize the grayscale mask.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(H, W)``.
    """

    img = Image.open(mask_path).convert('L')
    arr = np.asarray(img, dtype=np.uint8)
    mask = arr >= int(threshold)

    if not np.any(mask):
        raise RuntimeError(
            f'mask {mask_path!r} contains no foreground pixels.')

    return mask


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a binary mask.

    A hole is defined as a background region that is fully enclosed by
    foreground pixels and is not connected to the image border.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Boolean mask with enclosed holes filled.
    """

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    if not np.any(mask):
        return mask

    h, w = mask.shape[:2]

    foreground = mask.astype(np.uint8)
    background = (1 - foreground).astype(np.uint8)

    # Flood-fill background connected to the image border.
    flood = background.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Fill all border-connected background with value 2.
    for x in range(w):
        if flood[0, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, 0), 2)
        if flood[h - 1, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, h - 1), 2)

    for y in range(h):
        if flood[y, 0] == 1:
            cv2.floodFill(flood, flood_mask, (0, y), 2)
        if flood[y, w - 1] == 1:
            cv2.floodFill(flood, flood_mask, (w - 1, y), 2)

    # Background still equal to 1 was not reachable from the border, therefore
    # it is an internal hole.
    holes = flood == 1

    return mask | holes


def _component_boundary_points(
    labels: np.ndarray,
    *,
    label: int,
) -> np.ndarray:
    """
    Extract boundary points for a connected component.

    Parameters
    ----------
    labels : np.ndarray
        Connected-component label image with shape ``(H, W)``.
    label : int
        Component label to extract.

    Returns
    -------
    np.ndarray
        Boundary points with shape ``(N, 2)`` in ``(y, x)`` order.
    """

    component = labels == int(label)

    if not np.any(component):
        return np.empty((0, 2), dtype=np.int32)

    kernel = np.ones((3, 3), dtype=np.uint8)

    eroded = cv2.erode(
        component.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0

    boundary = component & ~eroded
    pts = np.column_stack(np.where(boundary))

    if pts.size == 0:
        pts = np.column_stack(np.where(component))

    return pts.astype(np.int32)


def _draw_bridges_between_point_sets(
    out: np.ndarray,
    *,
    points_a: np.ndarray,
    points_b: np.ndarray,
    max_dist2: float,
    thickness: int,
    chunk_size: int = 512,
) -> None:
    """
    Draw bridge segments between all point pairs closer than a threshold.

    Parameters
    ----------
    out : np.ndarray
        Mutable uint8 output mask. Bridges are drawn in-place.
    points_a : np.ndarray
        First point set with shape ``(N, 2)`` in ``(y, x)`` order.
    points_b : np.ndarray
        Second point set with shape ``(M, 2)`` in ``(y, x)`` order.
    max_dist2 : float
        Squared maximum distance for drawing bridge segments.
    thickness : int
        Fixed bridge line thickness in pixels.
    chunk_size : int, default=512
        Number of points from ``points_a`` processed per chunk.
    """

    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError('points_a must have shape (N, 2).')

    if points_b.ndim != 2 or points_b.shape[1] != 2:
        raise ValueError('points_b must have shape (M, 2).')

    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return

    b = points_b.astype(np.float32)

    for start in range(0, points_a.shape[0], int(chunk_size)):
        a = points_a[start:start + int(chunk_size)].astype(np.float32)

        dy = a[:, None, 0] - b[None, :, 0]
        dx = a[:, None, 1] - b[None, :, 1]
        dist2 = dx * dx + dy * dy

        close_pairs = np.argwhere(dist2 <= float(max_dist2))

        for local_i, j in close_pairs:
            p_a = points_a[start + int(local_i)]
            p_b = points_b[int(j)]

            # cv2.line expects points in (x, y) order.
            cv2.line(
                out,
                (int(p_a[1]), int(p_a[0])),
                (int(p_b[1]), int(p_b[0])),
                color=1,
                thickness=int(thickness),
                lineType=cv2.LINE_8,
            )


def _connect_nearby_components(
    mask: np.ndarray,
    *,
    max_bridge_distance: Optional[str],
) -> np.ndarray:
    """
    Connect nearby foreground components by drawing fixed-width bridges.

    This function connects distinct foreground connected components when their
    boundary points are close enough. For each pair of components, all boundary
    point pairs whose distance is less than or equal to ``max_bridge_distance``
    are connected with fixed-width line segments.

    The operation is intended only to build a helper mask for geometric fitting.
    It does not fill holes inside a single component and does not apply
    morphological closing.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    max_bridge_distance : str or None
        Maximum distance between boundary points of two distinct connected
        components for a bridge segment to be drawn. If None, the input mask is
        returned unchanged.

        Only percentage strings are accepted, for example ``'5%'``. Percentages are
        resolved against ``max(H, W)`` of the mask passed to this function.

    Returns
    -------
    np.ndarray
        Boolean mask with nearby components connected.
    """

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    if max_bridge_distance is None:
        return mask

    h, w = mask.shape[:2]
    ref = max(h, w)

    max_dist_px = resolve_size_expr(
        max_bridge_distance,
        reference=ref,
        min_size=0,
        # The value is already validated as percentage-only in _read_cfg().
        # allow_unitless=True is harmless here and keeps the helper permissive
        # if this function is ever called directly with a numeric string.
        allow_unitless=True,
    )

    if max_dist_px <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    # Label 0 is background. If there is 0 or 1 foreground component, there is
    # nothing to connect.
    if num_labels <= 2:
        return mask

    component_labels = list(range(1, num_labels))
    boundaries = {
        label: _component_boundary_points(labels, label=label)
        for label in component_labels
    }

    out = mask.astype(np.uint8).copy()

    max_dist2 = float(max_dist_px * max_dist_px)
    bridge_thickness = 3

    for i, label_a in enumerate(component_labels):
        pts_a = boundaries[label_a]
        if pts_a.shape[0] == 0:
            continue

        for label_b in component_labels[i + 1:]:
            pts_b = boundaries[label_b]
            if pts_b.shape[0] == 0:
                continue

            _draw_bridges_between_point_sets(
                out,
                points_a=pts_a,
                points_b=pts_b,
                max_dist2=max_dist2,
                thickness=bridge_thickness,
            )

    return out > 0


def _rotate_mask(
    mask: np.ndarray,
    *,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a binary mask around its center.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Rotated boolean mask and affine transform matrix mapping original
        coordinates into rotated coordinates.
    """

    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))

    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    src = mask.astype(np.uint8) * 255
    rotated = cv2.warpAffine(
        src,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated > 0, matrix


def _largest_axis_aligned_rect(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the largest axis-aligned all-true rectangle inside a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.

    Returns
    -------
    tuple[int, int, int, int]
        Rectangle ``(x, y, width, height)`` in mask coordinates.
    """

    h, w = mask.shape[:2]
    heights = np.zeros(w, dtype=np.int32)

    best_area = 0
    best_rect = (0, 0, 0, 0)

    for y in range(h):
        row = mask[y]
        heights[row] += 1
        heights[~row] = 0

        stack: List[int] = []
        for x in range(w + 1):
            cur_h = int(heights[x]) if x < w else 0

            while stack and cur_h < int(heights[stack[-1]]):
                top = stack.pop()
                rect_h = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                rect_w = x - left
                area = rect_w * rect_h

                if area > best_area:
                    best_area = area
                    best_rect = (
                        left,
                        y - rect_h + 1,
                        rect_w,
                        rect_h,
                    )

            stack.append(x)

    if best_area <= 0:
        raise RuntimeError('Cannot find an inner rectangle in an empty mask.')

    return best_rect


def _foreground_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the minimal axis-aligned rectangle containing all foreground pixels.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.

    Returns
    -------
    tuple[int, int, int, int]
        Rectangle ``(x, y, width, height)`` in mask coordinates.

    Raises
    ------
    RuntimeError
        If the mask contains no foreground pixels.
    """

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    ys, xs = np.where(mask)

    if xs.size == 0 or ys.size == 0:
        raise RuntimeError('Cannot find foreground bbox in an empty mask.')

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2 - x1, y2 - y1


def _blend_axis_aligned_rects(
    inner: Tuple[int, int, int, int],
    outer: Tuple[int, int, int, int],
    *,
    box_blend: NormalizedBoxBlend,
) -> Tuple[float, float, float, float]:
    """
    Blend an inner rectangle toward an outer rectangle, independently per side.

    Parameters
    ----------
    inner : tuple[int, int, int, int]
        Inner rectangle ``(x, y, width, height)``.
    outer : tuple[int, int, int, int]
        Outer rectangle ``(x, y, width, height)``.
    box_blend : tuple[float, float, float, float]
        Per-side blend factors in ``(left, top, right, bottom)`` order.

        Each value must be in ``[0, 1]``:

        - ``0.0`` keeps the corresponding side from the inner rectangle.
        - ``1.0`` moves the corresponding side to the outer rectangle.

    Returns
    -------
    tuple[float, float, float, float]
        Blended rectangle ``(x, y, width, height)``.
    """

    left_blend, top_blend, right_blend, bottom_blend = box_blend

    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer

    ix1 = float(ix)
    iy1 = float(iy)
    ix2 = float(ix + iw)
    iy2 = float(iy + ih)

    ox1 = float(ox)
    oy1 = float(oy)
    ox2 = float(ox + ow)
    oy2 = float(oy + oh)

    x1 = ix1 + left_blend * (ox1 - ix1)
    y1 = iy1 + top_blend * (oy1 - iy1)
    x2 = ix2 + right_blend * (ox2 - ix2)
    y2 = iy2 + bottom_blend * (oy2 - iy2)

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    return x1, y1, width, height


def _rect_to_corners(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> np.ndarray:
    """
    Convert an axis-aligned rectangle to corner coordinates.

    Parameters
    ----------
    x : float
        Left coordinate.
    y : float
        Top coordinate.
    width : float
        Rectangle width.
    height : float
        Rectangle height.

    Returns
    -------
    np.ndarray
        Array with shape ``(4, 2)`` containing corners in clockwise order.
    """

    x1 = float(x)
    y1 = float(y)
    x2 = x1 + float(width)
    y2 = y1 + float(height)

    return np.asarray(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def _invert_affine_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Map points through the inverse of an OpenCV affine transform.

    Parameters
    ----------
    points : np.ndarray
        Array with shape ``(N, 2)``.
    matrix : np.ndarray
        Affine transform matrix with shape ``(2, 3)``.

    Returns
    -------
    np.ndarray
        Transformed points with shape ``(N, 2)``.
    """

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('points must have shape (N, 2).')

    if matrix.shape != (2, 3):
        raise ValueError('matrix must have shape (2, 3).')

    inv = cv2.invertAffineTransform(matrix.astype(np.float64))

    pts = points.astype(np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)

    out = pts_h @ inv.T
    return out.astype(np.float32)


def _rect_from_corners(
    corners: np.ndarray,
) -> RotatedRect:
    """
    Build a canonical rotated rectangle from four corner points.

    The returned corner order is normalized as:

    - top-left
    - top-right
    - bottom-right
    - bottom-left

    This canonicalization makes the local x axis follow the visually top edge of
    the rectangle, rather than the arbitrary axis produced by the rotated-mask scan.
    This keeps overlay orientation stable in image coordinates.

    Parameters
    ----------
    corners : np.ndarray
        Array with shape ``(4, 2)`` containing rectangle corners.

    Returns
    -------
    RotatedRect
        Canonical rotated rectangle.
    """

    if corners.shape != (4, 2):
        raise ValueError('corners must have shape (4, 2).')

    pts = corners.astype(np.float32)

    # Keep the cyclic order, but choose the edge whose midpoint is visually
    # highest as the local top edge.
    edge_mid_y = []
    for i in range(4):
        j = (i + 1) % 4
        edge_mid_y.append(0.5 * (float(pts[i, 1]) + float(pts[j, 1])))

    i0 = int(np.argmin(edge_mid_y))
    i1 = (i0 + 1) % 4

    # Ensure the top edge is ordered from left to right in image coordinates.
    if float(pts[i0, 0]) <= float(pts[i1, 0]):
        ordered = np.asarray(
            [
                pts[i0],
                pts[i1],
                pts[(i1 + 1) % 4],
                pts[(i1 + 2) % 4],
            ],
            dtype=np.float32,
        )
    else:
        ordered = np.asarray(
            [
                pts[i1],
                pts[i0],
                pts[(i0 - 1) % 4],
                pts[(i0 - 2) % 4],
            ],
            dtype=np.float32,
        )

    cx = float(np.mean(ordered[:, 0]))
    cy = float(np.mean(ordered[:, 1]))

    u = ordered[1] - ordered[0]
    v = ordered[3] - ordered[0]

    width = float(np.linalg.norm(u))
    height = float(np.linalg.norm(v))

    if width <= 0.0 or height <= 0.0:
        raise RuntimeError('Cannot build rectangle from degenerate corners.')

    angle_deg = float(np.degrees(np.arctan2(float(u[1]), float(u[0]))))

    return RotatedRect(
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        angle_deg=angle_deg,
        corners=[(float(x), float(y)) for x, y in ordered],
    )


def _fit_rect_at_angle(
    mask: np.ndarray,
    *,
    angle_deg: float,
    box_blend: NormalizedBoxBlend = (0.0, 0.0, 0.0, 0.0),
) -> Tuple[RotatedRect, RotatedRect, RotatedRect]:
    """
    Fit an axis-aligned placement rectangle after rotating the mask by one angle.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    angle_deg : float
        Candidate rectangle angle in degrees.
    box_blend : tuple[float, float, float, float], default=(0.0, 0.0, 0.0, 0.0)
        Per-side blend factors between the largest fully-inscribed rectangle
        and the rotated foreground bounding box, in ``(left, top, right, bottom)``
        order.

        Each value controls one side independently:

        - ``0.0`` keeps the corresponding side from the strict inner rectangle.
        - ``1.0`` moves the corresponding side to the rotated foreground bounding
          box.

    Returns
    -------
    tuple[RotatedRect, RotatedRect, RotatedRect]
        Final blended rectangle, strict inner rectangle, and rotated foreground
        bounding box, all mapped back into the coordinate system of the input mask.
    """

    rotated_mask, matrix = _rotate_mask(mask, angle_deg=-float(angle_deg))

    inner_axis = _largest_axis_aligned_rect(rotated_mask)
    outer_axis = _foreground_bbox(rotated_mask)

    if any(x > 0.0 for x in box_blend):
        x, y, width, height = _blend_axis_aligned_rects(
            inner_axis,
            outer_axis,
            box_blend=box_blend,
        )
    else:
        x, y, width, height = inner_axis

    final_corners_rot = _rect_to_corners(
        x=float(x),
        y=float(y),
        width=float(width),
        height=float(height),
    )
    inner_corners_rot = _rect_to_corners(
        x=float(inner_axis[0]),
        y=float(inner_axis[1]),
        width=float(inner_axis[2]),
        height=float(inner_axis[3]),
    )
    outer_corners_rot = _rect_to_corners(
        x=float(outer_axis[0]),
        y=float(outer_axis[1]),
        width=float(outer_axis[2]),
        height=float(outer_axis[3]),
    )

    final_rect = _rect_from_corners(
        _invert_affine_points(final_corners_rot, matrix)
    )
    inner_rect = _rect_from_corners(
        _invert_affine_points(inner_corners_rot, matrix)
    )
    outer_rect = _rect_from_corners(
        _invert_affine_points(outer_corners_rot, matrix)
    )

    return final_rect, inner_rect, outer_rect


def _fit_rect_on_mask_axis(
    mask: np.ndarray,
    *,
    box_blend: NormalizedBoxBlend = (0.0, 0.0, 0.0, 0.0),
) -> RectFitResult:
    """
    Fit placement rectangles aligned with the estimated mask axis.

    The function first tries the quadrilateral-frame estimator and falls back to the
    centerline estimator if the quadrilateral frame cannot be estimated.
    """

    try:
        alignment_angle = estimate_mask_quad_alignment_angle(mask)
        alignment_method = 'quad'
    except RuntimeError:
        alignment_angle = estimate_mask_centerline_alignment_angle(mask)
        alignment_method = 'centerline'

    rect, inner_rect, outer_rect = _fit_rect_at_angle(
        mask,
        angle_deg=alignment_angle,
        box_blend=box_blend,
    )

    return RectFitResult(
        rect=rect,
        inner_rect=inner_rect,
        outer_rect=outer_rect,
        alignment_angle=alignment_angle,
        alignment_method=alignment_method,
    )


def _warp_local_to_rect(
    local_rgba: Image.Image,
    *,
    rect: RotatedRect,
    canvas_size: Tuple[int, int],
    local_origin: Tuple[float, float] = (0.0, 0.0),
    local_rect_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Warp a local RGBA layer onto a rotated rectangle in full image space.

    Parameters
    ----------
    local_rgba : PIL.Image.Image
        Local RGBA layer expressed in the rectangle local coordinate system. The
        canvas may be larger than the fitted rectangle when overflow is enabled.
    rect : RotatedRect
        Destination rotated rectangle.
    canvas_size : tuple[int, int]
        Output canvas size ``(width, height)``.
    local_origin : tuple[float, float], default=(0.0, 0.0)
        Local coordinates of the input canvas top-left corner in the fitted
        rectangle coordinate system.
    local_rect_size : tuple[int, int], optional
        Size of the fitted rectangle in local coordinates. If omitted, the input
        canvas size is used, preserving the previous non-overflow behavior.

    Returns
    -------
    PIL.Image.Image
        Full-size transparent RGBA layer.
    """

    local_rgba = local_rgba.convert('RGBA')
    src_w, src_h = local_rgba.size
    canvas_w, canvas_h = canvas_size
    local_rect_w, local_rect_h = local_rect_size or (src_w, src_h)

    if src_w <= 0 or src_h <= 0:
        raise ValueError(f'invalid local layer size {src_w}x{src_h}')

    if local_rect_w <= 0 or local_rect_h <= 0:
        raise ValueError(
            f'invalid local rect size {local_rect_w}x{local_rect_h}')

    src = np.asarray(
        [
            [0.0, 0.0],
            [float(src_w), 0.0],
            [float(src_w), float(src_h)],
            [0.0, float(src_h)],
        ],
        dtype=np.float32,
    )

    rect_corners = np.asarray(rect.corners, dtype=np.float32)

    if rect_corners.shape != (4, 2):
        raise ValueError('rect.corners must have shape (4, 2).')

    p0 = rect_corners[0]
    u = rect_corners[1] - rect_corners[0]
    v = rect_corners[3] - rect_corners[0]

    def _local_to_world(px: float, py: float) -> np.ndarray:
        return (
            p0
            + u * (float(px) / float(local_rect_w))
            + v * (float(py) / float(local_rect_h))
        )

    ox, oy = local_origin
    local_corners = np.asarray(
        [
            [float(ox), float(oy)],
            [float(ox) + float(src_w), float(oy)],
            [float(ox) + float(src_w), float(oy) + float(src_h)],
            [float(ox), float(oy) + float(src_h)],
        ],
        dtype=np.float32,
    )

    dst = np.asarray(
        [_local_to_world(float(x), float(y)) for x, y in local_corners],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)

    local_arr = np.asarray(local_rgba, dtype=np.uint8)
    warped = cv2.warpPerspective(
        local_arr,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return Image.fromarray(warped, mode='RGBA')


def _clip_layer_to_mask(
    layer: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """
    Clip an RGBA layer alpha channel to a binary mask.

    Parameters
    ----------
    layer : PIL.Image.Image
        Full-size RGBA layer.
    mask : np.ndarray
        Boolean mask with shape matching the layer.

    Returns
    -------
    PIL.Image.Image
        Clipped RGBA layer.
    """

    layer = layer.convert('RGBA')
    w, h = layer.size

    if mask.shape[:2] != (h, w):
        raise ValueError(
            f'mask shape {mask.shape[:2]} does not match layer size {(h, w)}.'
        )

    arr = np.asarray(layer, dtype=np.uint8).copy()
    alpha = arr[:, :, 3]

    clip = mask.astype(np.uint8)
    arr[:, :, 3] = (alpha * clip).astype(np.uint8)

    return Image.fromarray(arr, mode='RGBA')


def _make_debug_image(
    mask: np.ndarray,
    rect: RotatedRect,
    *,
    inner_rect: Optional[RotatedRect] = None,
    outer_rect: Optional[RotatedRect] = None,
    content_box: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    """
    Create a debug visualization of the fitted insertion rectangle.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    rect : RotatedRect
        Fitted rectangle in full-image coordinates.
    inner_rect : RotatedRect, optional
        Strict largest fully-inscribed rectangle, drawn as diagnostic overlay.
    outer_rect : RotatedRect, optional
        Rotated foreground bounding box, drawn as diagnostic overlay.
    content_box : tuple[int, int, int, int], optional
        Optional local content box ``(x, y, width, height)`` after applying
        content insets. When provided, it is projected into full-image
        coordinates and drawn relative to the fitted rectangle.

    Returns
    -------
    PIL.Image.Image
        RGB debug image showing the mask, fitted rectangle, and optional content
        box.
    """

    def _draw_rect(
        rct: RotatedRect,
        *,
        color: Tuple[int, int, int],
        width: int,
    ) -> None:
        pts = [(float(x), float(y)) for x, y in rct.corners]
        if len(pts) != 4:
            raise ValueError('rect.corners must contain exactly 4 points.')
        draw.line(
            pts + [pts[0]],
            fill=color,
            width=width,
        )

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    mask_u8 = mask.astype(np.uint8) * 255
    rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)

    img = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)

    corners = [(float(x), float(y)) for x, y in rect.corners]
    if len(corners) != 4:
        raise ValueError('rect.corners must contain exactly 4 points.')

    # Fitted rectangle.
    if outer_rect is not None:
        _draw_rect(
            outer_rect,
            color=(180, 80, 255),
            width=2,
        )

    if inner_rect is not None:
        _draw_rect(
            inner_rect,
            color=(0, 128, 255),
            width=2,
        )

    # Final fitted / blended rectangle.
    _draw_rect(
        rect,
        color=(255, 0, 0),
        width=3,
    )

    # Rectangle center.
    r = 4
    cx = float(rect.cx)
    cy = float(rect.cy)
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=(0, 255, 0),
        outline=(0, 0, 0),
    )

    # Local axes, useful to see rectangle orientation.
    draw.line(
        [corners[0], corners[1]],
        fill=(255, 128, 0),
        width=2,
    )
    draw.line(
        [corners[0], corners[3]],
        fill=(0, 255, 255),
        width=2,
    )

    # Optional content box after applying content insets.
    if content_box is not None:
        bx, by, bw, bh = content_box

        if bw <= 0 or bh <= 0:
            raise ValueError(f'invalid content_box size: {content_box!r}')

        local_w = float(rect.width)
        local_h = float(rect.height)

        if local_w <= 0 or local_h <= 0:
            raise ValueError('rect width and height must be positive.')

        rect_corners = np.asarray(rect.corners, dtype=np.float32)
        u = rect_corners[1] - rect_corners[0]
        v = rect_corners[3] - rect_corners[0]

        def _local_to_world(px: float, py: float) -> Tuple[float, float]:
            q = (
                rect_corners[0]
                + u * (float(px) / local_w)
                + v * (float(py) / local_h)
            )
            return float(q[0]), float(q[1])

        content_corners = [
            _local_to_world(bx, by),
            _local_to_world(bx + bw, by),
            _local_to_world(bx + bw, by + bh),
            _local_to_world(bx, by + bh),
        ]

        draw.line(
            content_corners + [content_corners[0]],
            fill=(0, 255, 128),
            width=2,
        )

    return img


def _resolve_overlay_path(
    input: Optional[Dict[str, Dict[str, Any]]],
    *,
    input_id: str = 'overlay',
) -> str:
    """
    Resolve the overlay image path from a wired input.

    Parameters
    ----------
    input : dict[str, dict] or None
        Runtime input mapping received by the node.
    input_id : str, default='overlay'
        Input channel used for the overlay image.

    Returns
    -------
    str
        Filesystem path to the overlay image.

    Raises
    ------
    RuntimeError
        If the overlay input is missing or does not contain an image path.
    """

    input = input or {}
    upstream = input.get(input_id)

    if upstream is None:
        raise RuntimeError(
            f"MaskInsertLayer requires an overlay image wired to '{input_id}'."
        )

    path = upstream.get('image') or upstream.get(
        'path') or upstream.get('images')

    if not path:
        raise RuntimeError(
            f"Overlay input '{input_id}' must contain 'image' or 'path'."
        )

    if isinstance(path, list):
        if len(path) != 1:
            raise RuntimeError(
                f"Overlay input '{input_id}' must contain exactly one image path."
            )
        path = path[0]

    return str(path)


def _resize_overlay_to_box(
    overlay: Image.Image,
    *,
    box_width: int,
    box_height: int,
    inset_left: SizeExpr,
    inset_right: SizeExpr,
    inset_top: SizeExpr,
    inset_bottom: SizeExpr,
    resize: ResizeMode,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Resize an overlay image against a local rectangle content box.

    Parameters
    ----------
    overlay : PIL.Image.Image
        Source overlay image.
    box_width : int
        Local rectangle canvas width.
    box_height : int
        Local rectangle canvas height.
    inset_left : int or str
        Left content inset. Positive values move the content edge inward;
        negative values expand it outward.
    inset_right : int or str
        Right content inset. Positive values move the content edge inward;
        negative values expand it outward.
    inset_top : int or str
        Top content inset. Positive values move the content edge inward;
        negative values expand it outward.
    inset_bottom : int or str
        Bottom content inset. Positive values move the content edge inward;
        negative values expand it outward.
    resize : {'fit', 'cover', 'stretch'}
        Resize mode.

    Returns
    -------
    tuple[PIL.Image.Image, tuple[int, int, int, int]]
        Resized overlay and available content box
        ``(x, y, width, height)`` in local coordinates.
    """

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f'invalid box size {box_width}x{box_height}')

    def _resolve_inset(value: SizeExpr, *, reference: int) -> int:
        validate_size_expr(
            value,
            allow_unitless=True,
            allow_negative=True,
        )

        if isinstance(value, int):
            return value

        s = value.strip().lower()
        if s.endswith('px'):
            return int(round(float(s[:-2])))
        if s.endswith('%'):
            return int(round(reference * float(s[:-1]) / 100.0))

        return int(round(float(s)))

    left = _resolve_inset(inset_left, reference=box_width)
    right = _resolve_inset(inset_right, reference=box_width)
    top = _resolve_inset(inset_top, reference=box_height)
    bottom = _resolve_inset(inset_bottom, reference=box_height)

    avail_w = box_width - left - right
    avail_h = box_height - top - bottom

    if avail_w <= 0 or avail_h <= 0:
        raise ValueError(
            'content box collapsed after applying insets: '
            f'box={box_width}x{box_height}, '
            f'insets=(left={left}, right={right}, top={top}, bottom={bottom})'
        )

    overlay = overlay.convert('RGBA')
    src_w, src_h = overlay.size

    if src_w <= 0 or src_h <= 0:
        raise ValueError(f'invalid overlay size {src_w}x{src_h}')

    if resize == 'stretch':
        out_w, out_h = avail_w, avail_h
    else:
        sx = avail_w / float(src_w)
        sy = avail_h / float(src_h)
        scale = min(sx, sy) if resize == 'fit' else max(sx, sy)

        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))

    resized = overlay.resize(
        (out_w, out_h),
        resample=Image.LANCZOS,
    )

    return resized, (left, top, avail_w, avail_h)


def _rotate_overlay(
    overlay: Image.Image,
    *,
    rotation_deg: float,
) -> Image.Image:
    """
    Rotate an overlay before local resize and placement.

    Positive angles are clockwise in image coordinates.
    """

    overlay = overlay.convert('RGBA')
    angle = float(rotation_deg)

    if angle % 360.0 == 0.0:
        return overlay

    return overlay.rotate(
        -angle,
        resample=Image.BICUBIC,
        expand=True,
    )


def _resolve_local_center_xy(
    position: Anchor,
    *,
    content_box: Tuple[int, int, int, int],
    layer_width: int,
    layer_height: int,
) -> Tuple[int, int]:
    """
    Resolve overlay center inside a local content box.

    Parameters
    ----------
    position : Anchor
        Anchor inside the local content box.
    content_box : tuple[int, int, int, int]
        Available local box ``(x, y, width, height)``.
    layer_width : int
        Overlay width after resizing.
    layer_height : int
        Overlay height after resizing.

    Returns
    -------
    tuple[int, int]
        Local center coordinates.
    """

    if position not in ALLOWED_ANCHORS:
        raise ValueError(
            f'Unknown local position anchor: {position!r}. '
            f'Allowed: {sorted(ALLOWED_ANCHORS)}'
        )

    x0, y0, w, h = content_box

    half_w = layer_width / 2.0
    half_h = layer_height / 2.0

    left_cx = int(round(x0 + half_w))
    right_cx = int(round(x0 + w - half_w))
    top_cy = int(round(y0 + half_h))
    bottom_cy = int(round(y0 + h - half_h))

    mid_cx = int(round(x0 + w / 2.0))
    mid_cy = int(round(y0 + h / 2.0))

    if position == 'center':
        return mid_cx, mid_cy
    if position == 'top':
        return mid_cx, top_cy
    if position == 'bottom':
        return mid_cx, bottom_cy
    if position == 'left':
        return left_cx, mid_cy
    if position == 'right':
        return right_cx, mid_cy
    if position == 'top-left':
        return left_cx, top_cy
    if position == 'top-right':
        return right_cx, top_cy
    if position == 'bottom-left':
        return left_cx, bottom_cy
    if position == 'bottom-right':
        return right_cx, bottom_cy

    raise AssertionError(f'unhandled position anchor: {position!r}')


def _place_overlay_on_local_canvas(
    overlay: Image.Image,
    *,
    box_width: int,
    box_height: int,
    content_box: Tuple[int, int, int, int],
    position: Anchor,
    allow_overflow: bool = True,
) -> Tuple[Image.Image, Tuple[float, float]]:
    """
    Place a resized overlay on a local transparent rectangle canvas.

    Parameters
    ----------
    overlay : PIL.Image.Image
        Resized RGBA overlay.
    box_width : int
        Local canvas width.
    box_height : int
        Local canvas height.
    content_box : tuple[int, int, int, int]
        Content box ``(x, y, width, height)`` in local rectangle coordinates.
        Its origin may be outside the fitted rectangle when negative insets are
        used.
    position : Anchor
        Anchor inside ``content_box``.
    allow_overflow : bool, default=True
        If true, expand the local canvas to include placed overlay pixels that
        fall outside the fitted rectangle. The fitted rectangle remains the
        coordinate frame; final visibility is controlled by downstream mask
        clipping.

    Returns
    -------
    tuple[PIL.Image.Image, tuple[float, float]]
        Local RGBA canvas with the overlay placed on it, and the local
        coordinates of the canvas top-left corner in the rectangle frame.
    """

    overlay = overlay.convert('RGBA')
    ow, oh = overlay.size

    cx, cy = _resolve_local_center_xy(
        position,
        content_box=content_box,
        layer_width=ow,
        layer_height=oh,
    )

    x0 = int(round(cx - ow / 2.0))
    y0 = int(round(cy - oh / 2.0))
    x1 = x0 + ow
    y1 = y0 + oh

    if allow_overflow:
        min_x = min(0, x0)
        min_y = min(0, y0)
        max_x = max(box_width, x1)
        max_y = max(box_height, y1)
    else:
        min_x = 0
        min_y = 0
        max_x = box_width
        max_y = box_height

    canvas_w = max(1, int(max_x - min_x))
    canvas_h = max(1, int(max_y - min_y))

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.alpha_composite(overlay, (x0 - min_x, y0 - min_y))

    return canvas, (float(min_x), float(min_y))


@dataclass
class MaskInsertLayer(NodeRef):
    """
    Create a transparent overlay layer fitted inside a mask.

    ``MaskInsertLayer`` reads a binary or grayscale mask, finds a large rotated
    rectangle inside the foreground region, fits an attached overlay image into
    that rectangle, and emits a full-frame RGBA layer aligned with the original
    mask coordinates.

    The node is intended for deterministic artifact placement workflows where a
    prepared graphical asset must be inserted into a semantic or manually defined
    region before optional downstream harmonization. Typical examples include:

    - placing a logo on a t-shirt, top, bag, or other segmented garment,
    - placing a badge, decal, label, or symbol on a detected object,
    - inserting a prepared text/logo composition into a sign, poster, monitor,
      panel, or masked planar surface,
    - creating a transparent layer that can be composited with ``ImageStack`` and
      optionally refined with low-strength ``Img2Img``.

    Unlike generative inpainting, this node does not synthesize the inserted
    content. The overlay is expected to be prepared upstream as an image, for
    example by a ``FileImage`` node or by a deterministic graphics/text rendering
    node. ``MaskInsertLayer`` is responsible only for spatial fitting, placement,
    warping, optional clipping, and output materialization.

    Processing pipeline
    -------------------
    1. Load and binarize the input mask.
    2. Optionally connect nearby foreground components in the fitting mask.
    3. Optionally fill enclosed holes in the fitting mask.
    4. Estimate the visual vertical axis of the fitting mask using a quadrilateral
       frame heuristic, with centerline fallback.
    5. Rotate the fitting mask once, using the estimated visual axis, so the
       desired rectangle becomes axis-aligned in the temporary rotated mask.
    6. Compute the largest fully-inscribed rectangle and the rotated foreground
       bounding box.
    7. Blend the inner rectangle toward the foreground bounding box according to
       ``params.box_blend``.
    8. Map the selected rectangle back into the original image coordinates.
    9. Load the attached overlay image from the ``overlay`` input sink.
    10. Apply the optional manual overlay rotation.
    11. Resize the overlay against the rectangle content area using the selected
        resize mode.
    12. Place the resized overlay according to the selected local anchor.
    13. Warp the local overlay canvas into the rotated rectangle in full-image
        coordinates.
    14. Clip the final alpha channel to the original mask.
    15. Save the generated full-frame RGBA layer and JSON sidecar.

    Rectangle fitting
    -----------------
    The fitted rectangle is computed from the mask using a pragmatic geometric
    heuristic:

    - four approximate border lines are fitted from the mask foreground:
      left/right borders are fitted from row extrema, while top/bottom borders
      are fitted from column extrema,
    - the intersections of those fitted border lines define an approximate
      quadrilateral frame,
    - the alignment axis is estimated from the line connecting the midpoint of
      the top side to the midpoint of the bottom side,
    - if the quadrilateral frame cannot be estimated, the node falls back to a
      centerline estimate based on row midpoints,
    - the mask is rotated once so that the placement rectangle can be computed in
      axis-aligned coordinates,
    - the largest fully-inscribed rectangle is computed,
    - the rotated foreground bounding box is also computed,
    - ``params.box_blend`` optionally expands the inner rectangle toward the
      foreground bounding box,
    - the selected rectangle is mapped back into the original image coordinates.

    This is not intended to solve the exact maximum-inscribed-rectangle problem.
    The goal is to find a large, stable, and useful placement area for downstream
    image editing workflows.

    Overlay placement
    -----------------
    The overlay image is inserted into the local rectangle coordinate system.
    Content insets define the local content area used for resizing and anchoring:
    positive values move the corresponding edge inward, while negative values
    expand it outward beyond the fitted rectangle. Overlay pixels outside the
    fitted rectangle are preserved during local placement and final clipping is
    performed only against the source mask.

    ``resize='fit'``
        Preserve aspect ratio and scale the overlay so that it is fully contained
        inside the available content area.

    ``resize='cover'``
        Preserve aspect ratio and scale the overlay so that it covers the whole
        available content area. The resized overlay may exceed both the content box
        and the fitted rectangle.

    ``resize='stretch'``
        Resize the overlay exactly to the available content area, without
        preserving aspect ratio.

    The local placement anchor controls where the resized overlay is positioned
    inside the content area. Supported anchors are:

    - ``'center'``
    - ``'top'``, ``'bottom'``, ``'left'``, ``'right'``
    - ``'top-left'``, ``'top-right'``
    - ``'bottom-left'``, ``'bottom-right'``

    ``rotation_deg`` can be set on :meth:`overlay` to manually rotate the overlay
    clockwise in local rectangle coordinates before resize and placement.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.

    path : str or pathlib.Path, optional
        Optional filesystem path to the input mask. If omitted, the mask is
        resolved from the upstream ``default`` input.

    spec : dict or str or pathlib.Path or sequence
        Node configuration specification.

        The specification may be provided as an in-memory dictionary, as a path
        to a HOCON configuration file, or as a sequence of such elements. When a
        sequence is given, each element is resolved independently and merged from
        left to right, with later elements overriding earlier ones.

    Inputs
    ------
    default : dict
        Required upstream mask payload, unless ``path`` is provided. The node
        expects either:

        - ``image`` : str
            Filesystem path to the mask image.
        - ``path`` : str
            Alternative key for the filesystem path.

        The mask may be binary or grayscale. It is binarized using
        ``params.threshold``. White/foreground pixels define the usable insertion
        region.

    overlay : dict
        Required overlay image payload, wired through :meth:`overlay`. The node
        expects either:

        - ``image`` : str
            Filesystem path to the overlay image.
        - ``path`` : str
            Alternative key for the filesystem path.

        The overlay is converted to ``RGBA`` before resizing and placement.
        Transparent pixels in the overlay are preserved.

    Configuration
    -------------
    ``params.threshold`` : int, default=128
        Threshold used to binarize the grayscale mask. Pixels greater than or
        equal to this value are treated as foreground.

    ``params.inset_left`` : int or str, default='0%'
        Left content inset. Percentages are resolved against the rectangle width.
        Positive values move the content edge inward; negative values expand it
        outward.

    ``params.inset_right`` : int or str, default='0%'
        Right content inset. Percentages are resolved against the rectangle width.
        Positive values move the content edge inward; negative values expand it
        outward.

    ``params.inset_top`` : int or str, default='0%'
        Top content inset. Percentages are resolved against the rectangle height.
        Positive values move the content edge inward; negative values expand it
        outward.

    ``params.inset_bottom`` : int or str, default='0%'
        Bottom content inset. Percentages are resolved against the rectangle height.
        Positive values move the content edge inward; negative values expand it
        outward.

    ``params.box_blend`` : float or sequence, default=0.0
        Blend factor between the largest fully-inscribed rectangle and the rotated
        foreground bounding box.

        Accepted forms are:

        - ``0.25``:
            Apply the same blend factor to all sides.

        - ``[0.30, 0.10]``:
            Apply horizontal and vertical factors. The first value is used for
            left/right sides; the second value is used for top/bottom sides.

        - ``[0.50, 0.10, 0.20, 0.30]``:
            Apply per-side factors in ``left, top, right, bottom`` order.

        Each factor must be in ``[0, 1]``:

        - ``0.0`` keeps the corresponding side from the fully-inscribed rectangle.
        - ``1.0`` moves the corresponding side to the rotated foreground bounding box.

        This is useful when the strict inner rectangle is too small, poorly centered,
        or must be expanded more on one side than on another.

    ``params.max_bridge_distance`` : str or None, default=None
        Maximum distance used to connect nearby foreground components before fitting
        the rectangle. Only percentage strings are accepted, for example ``'5%'``.
        Percentages are resolved against the larger side of the fitting mask. If
        None, components are not bridged.

    ``params.fill_holes`` : bool, default=False
        If true, fill enclosed holes in the helper mask used for rectangle
        fitting.

    ``debug.save_debug`` : bool, default=False
        If true, save a debug image showing the mask, final fitted rectangle,
        diagnostic inner/outer rectangles, rectangle center, local axes, and optional
        content box after applying content insets.

    Outputs
    -------
    dict
        JSON-serializable output containing:

        - ``ok`` : bool
            Success flag.
        - ``node`` : str
            Operator name.
        - ``id`` : str
            Node identifier.
        - ``image`` : str
            Path to the generated full-frame RGBA overlay layer.
        - ``mask`` : str
            Path to the source mask image.
        - ``overlay`` : str
            Path to the source overlay image.
        - ``rect`` : dict
            Fitted rotated rectangle metadata, including center, width, height,
            angle, and corners.
        - ``fit_rects`` : dict
            Diagnostic fitting metadata, including the strict inner rectangle, the
            outer rotated foreground bounding box, the estimated alignment angle, and the
            alignment estimator used.
        - ``placement`` : dict
            Overlay placement metadata, including anchor, resize mode, manual
            rotation, content box, original overlay size, rotated overlay size,
            fitted overlay size, local canvas size, and local origin.
        - ``params`` : dict
            Resolved node parameters.
        - ``debug_image`` : str or None
            Path to the optional debug image.
        - ``metadata`` : str
            Path to the JSON sidecar.
        - ``timing`` : dict
            Runtime timing information.

    Notes
    -----
    - The output image is always a full-frame ``RGBA`` layer aligned with the
      input mask size.
    - This node does not composite the overlay onto the original image. Use
      ``ImageStack`` or another compositing node for that step.
    - This node does not perform perspective estimation from the source image.
      The insertion geometry is derived only from the mask shape.
    - For realistic final results, the generated overlay layer can be composited
      onto the base image and then passed through a low-strength ``Img2Img`` or
      inpainting refinement step.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    _layer: LayerSpec = field(
        default_factory=LayerSpec,
        init=False,
        repr=False
    )

    def overlay(
        self,
        *,
        position: Anchor = 'center',
        resize: ResizeMode = 'fit',
        rotation_deg: float = 0.0,
    ) -> AttachmentSink:
        """
        Declare the overlay image input sink.

        This method exposes the ``overlay`` input channel used by
        ``MaskInsertLayer`` to receive the image that will be inserted into the
        mask-fitted rectangle.

        The method also stores the placement configuration associated with the
        overlay input. At runtime, the attached overlay image is rotated according
        to ``rotation_deg``, resized according to ``resize``, placed inside the
        fitted rectangle according to ``position``, warped into full-image
        coordinates, and clipped to the source mask.

        Parameters
        ----------
        position : {'center', 'top', 'bottom', 'left', 'right', 'top-left',
            'top-right', 'bottom-left', 'bottom-right'}, default='center'
            Anchor used to place the resized overlay inside the local content box
            of the fitted rectangle.

            The anchor is evaluated in the rectangle local coordinate system, after
            ``inset_left``, ``inset_right``, ``inset_top``, and
            ``inset_bottom`` have been applied.

            Supported values are:

            - ``'center'``:
              Center the overlay inside the content box.
            - ``'top'``:
              Center the overlay horizontally and align it to the top edge.
            - ``'bottom'``:
              Center the overlay horizontally and align it to the bottom edge.
            - ``'left'``:
              Center the overlay vertically and align it to the left edge.
            - ``'right'``:
              Center the overlay vertically and align it to the right edge.
            - ``'top-left'``:
              Align the overlay to the top-left corner.
            - ``'top-right'``:
              Align the overlay to the top-right corner.
            - ``'bottom-left'``:
              Align the overlay to the bottom-left corner.
            - ``'bottom-right'``:
              Align the overlay to the bottom-right corner.

        resize : {'fit', 'cover', 'stretch'}, default='fit'
            Resize mode used to adapt the overlay image to the rectangle content
            box.

            Supported values are:

            - ``'fit'``:
              Preserve aspect ratio and scale the overlay so that it is fully
              contained inside the content box.
            - ``'cover'``:
              Preserve aspect ratio and scale the overlay so that it covers the
              whole content box. Excess pixels may overflow the fitted rectangle
              and are clipped only by the final mask.
            - ``'stretch'``:
              Resize the overlay exactly to the content box, without preserving
              aspect ratio.

        rotation_deg : float, default=0.0
            Manual clockwise rotation applied to the overlay before resizing and
            placement, in the fitted rectangle local coordinate system.

        Returns
        -------
        AttachmentSink
            Sink receiving the overlay image.

            The upstream node must provide one of:

            - ``image`` : str
                Filesystem path to the overlay image.
            - ``path`` : str
                Alternative filesystem path key.

            The overlay image is converted to ``RGBA`` before placement. Existing
            transparency is preserved.

        Raises
        ------
        ValueError
            If ``position`` is not one of the supported anchors.
        ValueError
            If ``resize`` is not one of ``'fit'``, ``'cover'``, or ``'stretch'``.
        ValueError
            If ``rotation_deg`` is not a finite number.

        Examples
        --------
        Place a logo at the center of the fitted mask rectangle while preserving its
        aspect ratio::

            mask_insert = MaskInsertLayer(
                name='shirt_logo',
                spec={
                    'params': {
                        'inset_left': '6%',
                        'inset_right': '6%',
                        'inset_top': '6%',
                        'inset_bottom': '6%',
                    },
                },
            )

            shirt_mask >> mask_insert
            logo_image >> mask_insert.overlay(position='center', resize='fit')

        Place a badge near the top-right corner of the fitted rectangle::

            shirt_mask >> mask_insert
            badge_image >> mask_insert.overlay(
                position='top-right', resize='fit')
        """

        if position not in ALLOWED_ANCHORS:
            raise ValueError(
                f'{self.id}: invalid position {position!r}, '
                f'expected one of {sorted(ALLOWED_ANCHORS)}.'
            )

        if resize not in ('fit', 'cover', 'stretch'):
            raise ValueError(
                f'{self.id}: invalid resize mode {resize!r}, '
                "expected 'fit', 'cover', or 'stretch'."
            )

        rotation_deg = float(rotation_deg)
        if not isfinite(rotation_deg):
            raise ValueError(
                f'{self.id}: invalid rotation_deg {rotation_deg!r}; '
                'expected a finite number.'
            )

        self._layer = LayerSpec(
            position=position,
            resize=resize,
            rotation_deg=rotation_deg,
        )

        return AttachmentSink(
            name=f'mask_insert_overlay:{self.id}',
            target=self,
            input_id='overlay',
        )

    def run(
        self,
        output_dir: str,
        input: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()

        cfg_dict = resolve_spec(self.spec)
        cfg = _read_cfg(cfg_dict, self.id)

        mask_path = resolve_single_image_path(
            node_id=self.id,
            path=self.path,
            input=input,
        )
        mask = _load_mask(mask_path, threshold=cfg.threshold)
        fit_mask = mask

        fit_mask = _connect_nearby_components(
            fit_mask,
            max_bridge_distance=cfg.max_bridge_distance,
        )

        if cfg.fill_holes:
            fit_mask = _fill_mask_holes(fit_mask)

        overlay_path = _resolve_overlay_path(input, input_id='overlay')
        overlay = Image.open(overlay_path).convert('RGBA')
        placed_overlay = _rotate_overlay(
            overlay,
            rotation_deg=self._layer.rotation_deg,
        )

        fit = _fit_rect_on_mask_axis(
            fit_mask,
            box_blend=cfg.box_blend,
        )

        rect = fit.rect

        box_width = max(1, int(round(rect.width)))
        box_height = max(1, int(round(rect.height)))

        fitted_overlay, content_box = _resize_overlay_to_box(
            placed_overlay,
            box_width=box_width,
            box_height=box_height,
            inset_left=cfg.inset_left,
            inset_right=cfg.inset_right,
            inset_top=cfg.inset_top,
            inset_bottom=cfg.inset_bottom,
            resize=self._layer.resize,
        )

        local_layer, local_origin = _place_overlay_on_local_canvas(
            fitted_overlay,
            box_width=box_width,
            box_height=box_height,
            content_box=content_box,
            position=self._layer.position,
        )

        h, w = mask.shape[:2]
        layer = _warp_local_to_rect(
            local_layer,
            rect=rect,
            canvas_size=(w, h),
            local_origin=local_origin,
            local_rect_size=(box_width, box_height),
        )

        layer = _clip_layer_to_mask(layer, mask)

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='png',
        )
        layer.save(out_path)

        debug_path = None
        if cfg.save_debug:
            debug_img = _make_debug_image(
                mask,
                rect,
                inner_rect=fit.inner_rect,
                outer_rect=fit.outer_rect,
                content_box=content_box,
            )

            debug_path_obj = make_node_output_path(
                out_dir=Path(output_dir),
                node_id=f'{self.id}_debug',
                ext='png',
            )

            debug_img.save(debug_path_obj)
            debug_path = str(debug_path_obj)

        def _rect_to_metadata(rect: RotatedRect) -> Dict[str, Any]:
            return {
                'cx': rect.cx,
                'cy': rect.cy,
                'width': rect.width,
                'height': rect.height,
                'angle_deg': rect.angle_deg,
                'corners': rect.corners,
            }

        out: Dict[str, Any] = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'image': str(out_path),
            'mask': str(mask_path),
            'overlay': str(overlay_path),
            'rect': _rect_to_metadata(rect),
            'fit_rects': {
                'inner': _rect_to_metadata(fit.inner_rect),
                'outer': _rect_to_metadata(fit.outer_rect),
                'alignment_angle_deg': fit.alignment_angle,
                'alignment_method': fit.alignment_method,
            },
            'placement': {
                'position': self._layer.position,
                'resize': self._layer.resize,
                'rotation_deg': self._layer.rotation_deg,
                'content_box': list(content_box),
                'overlay_size': list(overlay.size),
                'rotated_overlay_size': list(placed_overlay.size),
                'fitted_overlay_size': list(fitted_overlay.size),
                'local_canvas_size': list(local_layer.size),
                'local_origin': list(local_origin),
            },
            'params': {
                'threshold': cfg.threshold,
                'inset_left': cfg.inset_left,
                'inset_right': cfg.inset_right,
                'inset_top': cfg.inset_top,
                'inset_bottom': cfg.inset_bottom,
                'box_blend': list(cfg.box_blend),
                'max_bridge_distance': cfg.max_bridge_distance,
                'fill_holes': cfg.fill_holes,
            },
            'debug_image': debug_path,
            'timing': {
                'elapsed_sec': time.time() - t0,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
