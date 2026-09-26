import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (PositionSpec, ResizeMode,
                                             SizeExpr, SpatialTransform,
                                             merge_spatial_transform,
                                             read_spatial_transform,
                                             resolve_size_expr,
                                             validate_size_expr)
from morphalo.nodes.preprocess.utils.color_ops import \
    match_lab_color_statistics
from morphalo.nodes.preprocess.utils.geometry import Point

CornerDelta = tuple[SizeExpr, SizeExpr] | None


@dataclass(frozen=True)
class CornerOffsets:
    """
    Optional per-corner displacement used to warp a layer in local image space.

    Each corner offset is expressed as ``(dx, dy)`` and is applied before layer
    resizing and placement.

    Offset components support:

    - ``int``:
        Absolute displacement in pixels.
    - ``'<number>px'``:
        Absolute displacement in pixels.
    - ``'<number>%'``:
        Percentage displacement relative to the current layer size.
        Horizontal components are resolved against layer width, vertical
        components against layer height.

    A ``None`` value means no displacement for that corner.
    """
    top_left: CornerDelta = None
    top_right: CornerDelta = None
    bottom_right: CornerDelta = None
    bottom_left: CornerDelta = None

    def resolved(
        self,
        *,
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Resolve all corner offsets to pixel deltas.

        Parameters
        ----------
        width : int
            Current layer width.
        height : int
            Current layer height.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int],
            tuple[int, int], tuple[int, int]]
            Resolved offsets in clockwise order:
            ``top_left``, ``top_right``, ``bottom_right``, ``bottom_left``.
        """
        return (
            resolve_corner_delta(self.top_left, width=width, height=height),
            resolve_corner_delta(self.top_right, width=width, height=height),
            resolve_corner_delta(
                self.bottom_right, width=width, height=height),
            resolve_corner_delta(self.bottom_left, width=width, height=height),
        )


@dataclass(frozen=True)
class LayerSpec:
    idx: int
    position: PositionSpec = 'center'
    resize: ResizeMode = None
    rotation: float = 0.0
    corner_offsets: Optional[CornerOffsets] = None
    brightness: float = 0.0
    color_transfer: float = 0.0
    feather: int | str = 0
    corner_radius: Optional[int | str] = None
    alpha: float = 1.0


@dataclass
class Config:
    width: Optional[int]
    height: Optional[int]
    background: Optional[Union[str, Tuple[int, int, int, int]]]
    out_mode: Literal['RGBA', 'RGB']


def resolve_feather_xy(
    feather: int | str,
    *,
    width: int,
    height: int,
) -> Tuple[int, int]:
    validate_size_expr(feather)

    if isinstance(feather, int):
        px = max(0, feather)
        return px, px

    s = feather.strip().lower()

    if s.endswith('px'):
        px = max(0, int(round(float(s[:-2]))))
        return px, px

    pct = max(0.0, float(s[:-1])) / 100.0
    feather_x = max(0, int(round(width * pct)))
    feather_y = max(0, int(round(height * pct)))

    return feather_x, feather_y


def resolve_feather_radius(
    feather: int | str,
    *,
    width: int,
    height: int,
) -> int:
    validate_size_expr(feather)

    if isinstance(feather, int):
        return max(0, feather)

    s = feather.strip().lower()

    if s.endswith('px'):
        return max(0, int(round(float(s[:-2]))))

    pct = max(0.0, float(s[:-1])) / 100.0
    mean_dim = (width + height) / 2.0

    # Non-zero percentage feathering should resolve to at least one pixel.
    return max(1, int(round(mean_dim * pct)))


def resolve_delta_expr(value: int | str, *, max_size: int) -> int:
    """
    Resolve a displacement expression to pixels.

    Parameters
    ----------
    value : int or str
        Displacement expression.

        Supported formats are:

        - ``int``:
            Absolute displacement in pixels.
        - ``'<number>px'``:
            Absolute displacement in pixels.
        - ``'<number>%'``:
            Percentage of ``max_size``.

        Negative values are allowed.

    max_size : int
        Reference size used to resolve percentage expressions.

    Returns
    -------
    int
        Resolved displacement in pixels.
    """
    if isinstance(value, int):
        return value

    if not isinstance(value, str):
        raise TypeError(
            f'Invalid delta expression type {type(value).__name__}; '
            'expected int or str.'
        )

    s = value.strip().lower()

    if s.endswith('px'):
        return int(round(float(s[:-2])))

    if s.endswith('%'):
        return int(round(max_size * float(s[:-1]) / 100.0))

    raise ValueError(
        f'Invalid delta expression {value!r}. '
        'Expected int, "<number>px", or "<number>%".'
    )


def resolve_corner_delta(
    delta: CornerDelta,
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    """
    Resolve an optional corner delta to pixel offsets.

    Parameters
    ----------
    delta : tuple[int | str, int | str] or None
        Optional ``(dx, dy)`` displacement.
    width : int
        Current layer width.
    height : int
        Current layer height.

    Returns
    -------
    tuple[int, int]
        Resolved ``(dx, dy)`` in pixels.
    """
    if delta is None:
        return 0, 0

    dx, dy = delta

    return (
        resolve_delta_expr(dx, max_size=width),
        resolve_delta_expr(dy, max_size=height),
    )


def feather_is_nonzero(feather: int | str) -> bool:
    if isinstance(feather, int):
        return feather > 0

    s = feather.strip().lower()

    return float(s[:-2] if s.endswith('px') else s[:-1]) > 0


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    width_raw = params.get('width')
    height_raw = params.get('height')
    width = None if width_raw is None else int(width_raw)
    height = None if height_raw is None else int(height_raw)
    if (width is not None and width <= 0) or (height is not None and height <= 0):
        raise ValueError(f"'{node_id}': invalid canvas size {width}x{height}")

    # Background configuration:
    # - None -> transparent
    # - str -> any color accepted by PIL
    # - [r, g, b] or [r, g, b, a] -> explicit color tuple
    bg = params.get('background', None)
    background: Optional[Union[str, Tuple[int, int, int, int]]]
    if bg is None:
        background = None
    elif isinstance(bg, str):
        background = bg
    elif isinstance(bg, (list, tuple)):
        if len(bg) == 3:
            r, g, b = map(int, bg)
            background = (r, g, b, 255)
        elif len(bg) == 4:
            r, g, b, a = map(int, bg)
            background = (r, g, b, a)
        else:
            raise ValueError(
                f"'{node_id}': background must be str, [r,g,b], [r,g,b,a], or null")
    else:
        raise ValueError(
            f"'{node_id}': background must be str/list/tuple/null")

    out_mode = str(params.get('out_mode', 'RGBA')).upper()
    if out_mode not in ('RGBA', 'RGB'):
        raise ValueError(
            f"'{node_id}': invalid out_mode={out_mode!r} (expected 'RGBA' or 'RGB')")

    # type: ignore[arg-type]
    return Config(width=width, height=height, background=background, out_mode=out_mode)


def _input_image_path(
    upstream: Any,
    *,
    node_id: str,
    input_name: str,
) -> Optional[str | Path]:
    if upstream is None:
        return None

    if isinstance(upstream, (str, Path)):
        return upstream

    if isinstance(upstream, dict):
        path = upstream.get('image') or upstream.get('path')
        if path:
            return path
        raise ValueError(
            f"{node_id}: upstream for {input_name!r} must contain 'image' or 'path'"
        )

    raise TypeError(
        f'{node_id}: upstream for {input_name!r} must be a dict, str, or Path, '
        f'got {type(upstream).__name__}.'
    )


def _resolve_position_xy(
    position: PositionSpec,
    W: int,
    H: int,
    lw: int,
    lh: int,
    anchor_xy: Point,
) -> Point:
    """
    Resolve a placement specification to canvas coordinates.

    The returned point is the canvas position where the layer local anchor must
    be placed. It is not necessarily the geometric center of the layer.

    Parameters
    ----------
    position : PositionSpec
        Placement specification.

        If a tuple is provided, it is interpreted as the target canvas
        coordinates for the layer anchor. Tuple components may be:

        - ``int``:
            Absolute canvas coordinate in pixels.
        - ``'<number>px'``:
            Explicit pixel coordinate.
        - ``'<number>%'``:
            Percentage of the corresponding canvas dimension.
        - ``None``:
            Use the center of the corresponding canvas axis.

        If a string is provided, it is interpreted as an edge or corner
        placement. The layer is positioned so that its transformed bounding box
        touches the requested canvas edge or corner while preserving the current
        local anchor.

    W : int
        Canvas width.
    H : int
        Canvas height.
    lw : int
        Current transformed layer width.
    lh : int
        Current transformed layer height.
    anchor_xy : tuple[float, float]
        Current layer anchor in transformed local coordinates.

    Returns
    -------
    tuple[float, float]
        Canvas coordinates where ``anchor_xy`` must be placed.
    """
    if isinstance(position, tuple):
        if len(position) != 2:
            raise ValueError(
                f'Position tuple must have length 2, got {position!r}'
            )

        x_expr, y_expr = position

        if x_expr is None:
            x = W / 2.0
        else:
            x = float(resolve_size_expr(x_expr, max_size=W, min_size=0))

        if y_expr is None:
            y = H / 2.0
        else:
            y = float(resolve_size_expr(y_expr, max_size=H, min_size=0))

        return x, y

    a = position.lower().strip()

    anchor_x, anchor_y = anchor_xy

    left_x = float(anchor_x)
    right_x = float(W) - (float(lw) - float(anchor_x))
    top_y = float(anchor_y)
    bottom_y = float(H) - (float(lh) - float(anchor_y))

    mid_x = float(W) / 2.0
    mid_y = float(H) / 2.0

    if a == 'center':
        return mid_x, mid_y

    if a in ('top', 'center-top', 'top-center'):
        return mid_x, top_y
    if a in ('bottom', 'center-bottom', 'bottom-center'):
        return mid_x, bottom_y
    if a in ('left', 'center-left', 'left-center'):
        return left_x, mid_y
    if a in ('right', 'center-right', 'right-center'):
        return right_x, mid_y

    if a in ('top-left', 'left-top'):
        return left_x, top_y
    if a in ('top-right', 'right-top'):
        return right_x, top_y
    if a in ('bottom-left', 'left-bottom'):
        return left_x, bottom_y
    if a in ('bottom-right', 'right-bottom'):
        return right_x, bottom_y

    raise ValueError(f'Unknown position anchor: {position!r}')


def _place_on_canvas_by_anchor(
    canvas: Image.Image,
    layer_rgba: Image.Image,
    *,
    placement_xy: Point,
    anchor_xy: Point,
) -> None:
    """
    Alpha-composite a layer by matching its local anchor to a canvas point.

    Parameters
    ----------
    canvas : PIL.Image.Image
        Destination canvas in ``RGBA`` mode.
    layer_rgba : PIL.Image.Image
        Layer image in ``RGBA`` mode.
    placement_xy : tuple[float, float]
        Canvas coordinates where the layer anchor must be placed.
    anchor_xy : tuple[float, float]
        Anchor position in the transformed layer local coordinates.

    Notes
    -----
    The layer top-left corner is computed as ``placement_xy - anchor_xy``.
    Out-of-bounds placement is handled by cropping the visible intersection.
    """
    W, H = canvas.size
    lw, lh = layer_rgba.size

    placement_x, placement_y = placement_xy
    anchor_x, anchor_y = anchor_xy

    x0 = int(round(placement_x - anchor_x))
    y0 = int(round(placement_y - anchor_y))
    x1 = x0 + lw
    y1 = y0 + lh

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)

    if ix1 <= ix0 or iy1 <= iy0:
        return

    lx0 = ix0 - x0
    ly0 = iy0 - y0
    lx1 = lx0 + (ix1 - ix0)
    ly1 = ly0 + (iy1 - iy0)

    layer_crop = layer_rgba.crop((lx0, ly0, lx1, ly1))

    patch = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    patch.alpha_composite(layer_crop, (ix0, iy0))
    canvas.alpha_composite(patch)


def _visible_layer_and_canvas_crops(
    canvas: Image.Image,
    layer_rgba: Image.Image,
    *,
    placement_xy: Point,
    anchor_xy: Point,
) -> tuple[tuple[int, int, int, int], Image.Image, Image.Image] | None:
    """
    Return matching visible layer and canvas crops for anchor-based placement.

    Parameters
    ----------
    canvas : PIL.Image.Image
        Destination canvas in ``RGBA`` mode.
    layer_rgba : PIL.Image.Image
        Layer image in ``RGBA`` mode.
    placement_xy : tuple[float, float]
        Canvas coordinates where the layer anchor must be placed.
    anchor_xy : tuple[float, float]
        Anchor position in the transformed layer local coordinates.

    Returns
    -------
    tuple[tuple[int, int, int, int], PIL.Image.Image, PIL.Image.Image] or None
        Matching ``(layer_box, layer_crop, canvas_crop)`` for the visible
        intersection, or ``None`` when the layer lies completely outside the
        canvas. ``layer_box`` is expressed in layer-local coordinates.
    """
    W, H = canvas.size
    lw, lh = layer_rgba.size

    placement_x, placement_y = placement_xy
    anchor_x, anchor_y = anchor_xy

    x0 = int(round(placement_x - anchor_x))
    y0 = int(round(placement_y - anchor_y))
    x1 = x0 + lw
    y1 = y0 + lh

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)

    if ix1 <= ix0 or iy1 <= iy0:
        return None

    lx0 = ix0 - x0
    ly0 = iy0 - y0
    lx1 = lx0 + (ix1 - ix0)
    ly1 = ly0 + (iy1 - iy0)

    layer_box = (lx0, ly0, lx1, ly1)

    return (
        layer_box,
        layer_rgba.crop(layer_box),
        canvas.crop((ix0, iy0, ix1, iy1)),
    )


def _apply_corner_offsets(
    img: Image.Image,
    corner_offsets: Optional[CornerOffsets],
    anchor_xy: Point,
) -> tuple[Image.Image, Point]:
    """
    Apply a local perspective warp and propagate the layer anchor.

    The transformation is applied in the layer local coordinate space, before
    rotation, resizing, feathering, and placement. The same perspective matrix is
    applied to ``anchor_xy`` so that anchor-based placement remains consistent
    after the warp.

    Parameters
    ----------
    img : PIL.Image.Image
        Input layer image. It is expected to be in ``RGBA`` mode.
    corner_offsets : CornerOffsets or None
        Optional per-corner displacement. ``None`` leaves the image and anchor
        unchanged.
    anchor_xy : tuple[float, float]
        Anchor position in the input layer local coordinates.

    Returns
    -------
    tuple[PIL.Image.Image, tuple[float, float]]
        Warped layer image and transformed anchor coordinates.
    """
    if corner_offsets is None:
        return img, anchor_xy

    import cv2
    import numpy as np

    w, h = img.size

    offsets = corner_offsets.resolved(width=w, height=h)
    if all(dx == 0 and dy == 0 for dx, dy in offsets):
        return img, anchor_xy

    # Source rectangle in local layer coordinates.
    src_pts = np.asarray(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )

    base_dst = np.asarray(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )

    delta = np.asarray(offsets, dtype=np.float32)
    dst_pts = base_dst + delta

    # Expand local output bounds so moved corners are not clipped.
    min_x = float(np.floor(dst_pts[:, 0].min()))
    min_y = float(np.floor(dst_pts[:, 1].min()))
    max_x = float(np.ceil(dst_pts[:, 0].max()))
    max_y = float(np.ceil(dst_pts[:, 1].max()))

    out_w = max(1, int(max_x - min_x + 1))
    out_h = max(1, int(max_y - min_y + 1))

    # Shift destination quad into positive local coordinates.
    dst_pts[:, 0] -= min_x
    dst_pts[:, 1] -= min_y

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    src_rgba = np.asarray(img, dtype=np.uint8)

    warped = cv2.warpPerspective(
        src_rgba,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    out = Image.fromarray(warped, mode='RGBA')

    anchor = np.asarray(
        [[float(anchor_xy[0]), float(anchor_xy[1])]],
        dtype=np.float32,
    )
    transformed = cv2.perspectiveTransform(anchor[None, :, :], matrix)[0, 0]

    return out, (float(transformed[0]), float(transformed[1]))


def _apply_rotation_around_anchor(
    img: Image.Image,
    rotation: float,
    anchor_xy: Point,
) -> tuple[Image.Image, Point]:
    """
    Rotate an RGBA layer around an arbitrary local anchor point.

    Parameters
    ----------
    img : PIL.Image.Image
        Input layer image in RGBA mode.
    rotation : float
        Counter-clockwise rotation angle in degrees.
    anchor_xy : tuple[float, float]
        Anchor position in local image coordinates.

    Returns
    -------
    tuple[PIL.Image.Image, tuple[float, float]]
        Rotated image and transformed anchor position in the rotated image.
    """
    angle = float(rotation) % 360.0
    if abs(angle) < 1e-9 or abs(angle - 360.0) < 1e-9:
        return img, anchor_xy

    import numpy as np

    w, h = img.size
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])

    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(w), 0.0],
            [float(w), float(h)],
            [0.0, float(h)],
        ],
        dtype=np.float64,
    )

    rel = corners - np.asarray([ax, ay], dtype=np.float64)

    rotated = np.empty_like(rel)
    rotated[:, 0] = rel[:, 0] * cos_t - rel[:, 1] * sin_t + ax
    rotated[:, 1] = rel[:, 0] * sin_t + rel[:, 1] * cos_t + ay

    min_x = float(math.floor(rotated[:, 0].min()))
    min_y = float(math.floor(rotated[:, 1].min()))
    max_x = float(math.ceil(rotated[:, 0].max()))
    max_y = float(math.ceil(rotated[:, 1].max()))

    out_w = max(1, int(max_x - min_x))
    out_h = max(1, int(max_y - min_y))

    new_anchor = (
        ax - min_x,
        ay - min_y,
    )

    inv_cos = cos_t
    inv_sin = -sin_t

    def source_from_output(x: float, y: float) -> tuple[float, float]:
        qx = x + min_x - ax
        qy = y + min_y - ay
        sx = qx * inv_cos - qy * inv_sin + ax
        sy = qx * inv_sin + qy * inv_cos + ay
        return sx, sy

    c = source_from_output(0.0, 0.0)
    x_unit = source_from_output(1.0, 0.0)
    y_unit = source_from_output(0.0, 1.0)

    matrix = (
        x_unit[0] - c[0],
        y_unit[0] - c[0],
        c[0],
        x_unit[1] - c[1],
        y_unit[1] - c[1],
        c[1],
    )

    rotated_img = img.transform(
        (out_w, out_h),
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    return rotated_img, new_anchor


def _apply_resize(
    img: Image.Image,
    resize: ResizeMode,
    canvas_w: int,
    canvas_h: int,
    anchor_xy: Point,
) -> tuple[Image.Image, Point]:
    """
    Resize a layer and propagate its local anchor consistently.

    Parameters
    ----------
    img : PIL.Image.Image
        Input layer image.
    resize : ResizeMode
        Resize specification. If ``None``, the image is returned unchanged.
    canvas_w : int
        Canvas width used to resolve percentage-based width expressions.
    canvas_h : int
        Canvas height used to resolve percentage-based height expressions.
    anchor_xy : tuple[float, float]
        Anchor position in local image coordinates before resizing.

    Returns
    -------
    tuple[PIL.Image.Image, tuple[float, float]]
        Resized image and anchor position in the resized image local
        coordinates.

    Notes
    -----
    The anchor is scaled by the same horizontal and vertical resize factors
    applied to the image. This keeps anchor-based placement stable after
    resizing.
    """
    if resize is None:
        return img, anchor_xy

    old_w, old_h = img.size

    if isinstance(resize, str):
        if resize == 'fit':
            scale = min(canvas_w / old_w, canvas_h / old_h)
        elif resize == 'cover':
            scale = max(canvas_w / old_w, canvas_h / old_h)
        else:
            raise ValueError(f'invalid resize mode: {resize!r}')

        new_w = max(1, int(round(old_w * scale)))
        new_h = max(1, int(round(old_h * scale)))

    else:
        target_w, target_h = resize
        if target_w is None and target_h is None:
            raise ValueError('resize cannot be (None, None)')

        if target_w is not None and target_h is not None:
            new_w = resolve_size_expr(target_w, max_size=canvas_w)
            new_h = resolve_size_expr(target_h, max_size=canvas_h)

        elif target_w is not None:
            new_w = resolve_size_expr(target_w, max_size=canvas_w)
            scale = float(new_w) / float(old_w)
            new_h = max(1, int(round(old_h * scale)))

        else:
            new_h = resolve_size_expr(target_h, max_size=canvas_h)
            scale = float(new_h) / float(old_h)
            new_w = max(1, int(round(old_w * scale)))

    new_w = max(1, int(new_w))
    new_h = max(1, int(new_h))

    if (new_w, new_h) == (old_w, old_h):
        return img, anchor_xy

    resized = img.resize((new_w, new_h), resample=Image.LANCZOS)

    scale_x = float(new_w) / float(old_w)
    scale_y = float(new_h) / float(old_h)

    return resized, (
        float(anchor_xy[0]) * scale_x,
        float(anchor_xy[1]) * scale_y,
    )


def _perturbed_alpha_ramp(
    w: int,
    h: int,
    *,
    feather_x: int,
    feather_y: int,
    corner_radius: int,
) -> Image.Image:
    """
    Create a perturbed alpha ramp for compositing rectangular crops.

    The generated alpha is based on an inset support region whose borders are
    softened before compositing. Horizontal and vertical feather widths are
    handled independently:

    - ``feather_x`` controls the fade width for left and right edges
    - ``feather_y`` controls the fade width for top and bottom edges

    To reduce visible rectangular seams, each side of the support region is
    perturbed independently with smooth random jitter. If ``corner_radius`` is
    greater than zero, the four corners are additionally constrained by radial
    ramps centered on the rounded-corner arc centers.

    Parameters
    ----------
    w : int
        Layer width in pixels.
    h : int
        Layer height in pixels.
    feather_x : int
        Feather width in pixels for left and right edges.
    feather_y : int
        Feather width in pixels for top and bottom edges.
    corner_radius : int
        Rounded-corner radius in pixels.

    Returns
    -------
    PIL.Image.Image
        Alpha mask in mode ``'L'`` with shape ``(w, h)``.
    """
    import numpy as np

    fx = max(0, int(feather_x))
    fy = max(0, int(feather_y))
    corner = max(0, int(corner_radius))

    if w <= 0 or h <= 0:
        return Image.new('L', (max(1, w), max(1, h)), 0)

    if fx <= 0 and fy <= 0:
        return Image.new('L', (w, h), 255)

    def _smooth_noise_1d(n: int, amp: float, sigma: float) -> np.ndarray:
        """
        Create smooth zero-mean 1D noise.

        Parameters
        ----------
        n : int
            Number of samples.
        amp : float
            Target peak amplitude in pixels.
        sigma : float
            Gaussian smoothing sigma in samples.

        Returns
        -------
        np.ndarray
            Smooth noise of shape ``(n,)``.
        """
        if n <= 1 or amp <= 0.0:
            return np.zeros((n,), dtype=np.float32)

        noise = np.random.normal(0.0, 1.0, size=n).astype(np.float32)

        sigma = max(1.0, float(sigma))
        radius = max(1, int(round(3.0 * sigma)))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(xs ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        kernel /= kernel.sum()

        smooth = np.convolve(noise, kernel, mode='same')

        max_abs = float(np.max(np.abs(smooth)))
        if max_abs > 1e-6:
            smooth = smooth / max_abs

        return smooth * float(amp)

    def _smoothstep(x: np.ndarray) -> np.ndarray:
        """
        Apply smoothstep on ``[0, 1]``.

        Parameters
        ----------
        x : np.ndarray
            Input values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Smoothstep output.
        """
        return x * x * (3.0 - 2.0 * x)

    # A small inward bias helps hide mismatched crop borders.
    bias_x = max(1, int(round(fx * 0.15))) if fx > 0 else 0
    bias_y = max(1, int(round(fy * 0.15))) if fy > 0 else 0

    x0 = bias_x
    y0 = bias_y
    x1 = (w - 1) - bias_x
    y1 = (h - 1) - bias_y

    if x1 <= x0 or y1 <= y0:
        return Image.new('L', (w, h), 255)

    support_w = x1 - x0 + 1
    support_h = y1 - y0 + 1

    # Clamp corner radius to valid support geometry.
    max_corner = max(0, (min(support_w, support_h) // 2) - 1)
    corner = min(corner, max_corner)

    # Perturbation amplitude is proportional to feather width.
    amp_x = 0.35 * float(fx) if fx > 0 else 0.0
    amp_y = 0.35 * float(fy) if fy > 0 else 0.0

    # Smoothing scale for side jitter. The exact factor is heuristic.
    sigma_x = max(1.0, max(fx, fy) * 0.35)
    sigma_y = max(1.0, max(fx, fy) * 0.35)

    # Vertical borders vary along y; horizontal borders vary along x.
    left_jitter = _smooth_noise_1d(h, amp=amp_x, sigma=sigma_y)
    right_jitter = _smooth_noise_1d(h, amp=amp_x, sigma=sigma_y)
    top_jitter = _smooth_noise_1d(w, amp=amp_y, sigma=sigma_x)
    bottom_jitter = _smooth_noise_1d(w, amp=amp_y, sigma=sigma_x)

    left_edge = x0 + left_jitter
    right_edge = x1 + right_jitter
    top_edge = y0 + top_jitter
    bottom_edge = y1 + bottom_jitter

    left_edge = np.clip(left_edge, 0.0, float(w - 1))
    right_edge = np.clip(right_edge, 0.0, float(w - 1))
    top_edge = np.clip(top_edge, 0.0, float(h - 1))
    bottom_edge = np.clip(bottom_edge, 0.0, float(h - 1))

    # Keep borders from crossing.
    min_gap_x = max(1.0, float(fx if fx > 0 else 1))
    min_gap_y = max(1.0, float(fy if fy > 0 else 1))
    right_edge = np.maximum(right_edge, left_edge + min_gap_x)
    bottom_edge = np.maximum(bottom_edge, top_edge + min_gap_y)

    right_edge = np.clip(right_edge, 0.0, float(w - 1))
    bottom_edge = np.clip(bottom_edge, 0.0, float(h - 1))

    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]

    # Distances to the four perturbed sides.
    dist_left = xx - left_edge[:, None]
    dist_right = right_edge[:, None] - xx
    dist_top = yy - top_edge[None, :]
    dist_bottom = bottom_edge[None, :] - yy

    # Distance to the nearest vertical or horizontal side.
    dist_x = np.minimum(dist_left, dist_right)
    dist_y = np.minimum(dist_top, dist_bottom)

    inside = (dist_x >= 0.0) & (dist_y >= 0.0)

    if fx > 0:
        tx = np.clip(dist_x / float(fx), 0.0, 1.0)
    else:
        tx = np.ones((h, w), dtype=np.float32)

    if fy > 0:
        ty = np.clip(dist_y / float(fy), 0.0, 1.0)
    else:
        ty = np.ones((h, w), dtype=np.float32)

    # Base anisotropic ramp: the nearest side dominates.
    t = np.minimum(tx, ty)

    # Apply rounded-corner constraints with radial ramps.
    #
    # The support region behaves as a rounded rectangle:
    # - side ramps control straight border segments
    # - radial ramps control the four corner arcs
    if corner > 0:
        fr = max(1, min(fx if fx > 0 else corner, fy if fy > 0 else corner))
        fr = min(fr, corner)

        if fr > 0:
            # Corner boxes.
            tl_mask = (xx < (x0 + corner)) & (yy < (y0 + corner))
            tr_mask = (xx > (x1 - corner)) & (yy < (y0 + corner))
            bl_mask = (xx < (x0 + corner)) & (yy > (y1 - corner))
            br_mask = (xx > (x1 - corner)) & (yy > (y1 - corner))

            # Arc centers.
            cx_tl, cy_tl = float(x0 + corner), float(y0 + corner)
            cx_tr, cy_tr = float(x1 - corner), float(y0 + corner)
            cx_bl, cy_bl = float(x0 + corner), float(y1 - corner)
            cx_br, cy_br = float(x1 - corner), float(y1 - corner)

            # Radial distance fields.
            d_tl = np.sqrt((xx - cx_tl) ** 2 + (yy - cy_tl) ** 2)
            d_tr = np.sqrt((xx - cx_tr) ** 2 + (yy - cy_tr) ** 2)
            d_bl = np.sqrt((xx - cx_bl) ** 2 + (yy - cy_bl) ** 2)
            d_br = np.sqrt((xx - cx_br) ** 2 + (yy - cy_br) ** 2)

            # Pixels are fully opaque sufficiently inside the arc,
            # fade over ``fr`` pixels, and become transparent outside.
            t_tl = np.clip((float(corner) - d_tl) / float(fr), 0.0, 1.0)
            t_tr = np.clip((float(corner) - d_tr) / float(fr), 0.0, 1.0)
            t_bl = np.clip((float(corner) - d_bl) / float(fr), 0.0, 1.0)
            t_br = np.clip((float(corner) - d_br) / float(fr), 0.0, 1.0)

            t[tl_mask] = np.minimum(t[tl_mask], t_tl[tl_mask])
            t[tr_mask] = np.minimum(t[tr_mask], t_tr[tr_mask])
            t[bl_mask] = np.minimum(t[bl_mask], t_bl[bl_mask])
            t[br_mask] = np.minimum(t[br_mask], t_br[br_mask])

            # Update inside-mask so pixels outside the rounded corners are removed.
            inside_tl = (~tl_mask) | (d_tl <= float(corner))
            inside_tr = (~tr_mask) | (d_tr <= float(corner))
            inside_bl = (~bl_mask) | (d_bl <= float(corner))
            inside_br = (~br_mask) | (d_br <= float(corner))

            inside &= inside_tl & inside_tr & inside_bl & inside_br

    t[~inside] = 0.0

    # Smooth the perceptual falloff.
    t = _smoothstep(t)

    alpha = np.clip(t * 255.0, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(alpha, mode='L')


@dataclass
class ImageLayerAttachmentSink(AttachmentSink):
    idx: int

    def transform(self) -> AttachmentSink:
        """
        Declare a spatial transform input for this layer.

        This method creates an additional wiring endpoint associated with the
        current layer, identified by ``idx``. The endpoint is intended for
        upstream nodes that provide spatial metadata describing how the layer
        should be anchored, placed, and optionally resized on the stack canvas.

        The transform input accepts either ``crop`` or ``placement`` metadata.
        Both use the same contract:

        - ``crop`` is intended for nodes that physically extract or refine a
          region from an image.
        - ``placement`` is intended for nodes that only describe how an image
          should be positioned.

        Expected upstream metadata
        --------------------------
        The upstream node must provide exactly one of the following keys in its
        output dictionary:

        ``crop`` : dict
            Crop-derived spatial metadata.

        ``placement`` : dict
            Generic spatial placement metadata.

        Both dictionaries support the same fields:

        ``anchor_xy`` : list[float] or tuple[float, float]
            Required. Anchor coordinates in the local coordinate frame of the
            image attached to this layer.

            The anchor is the logical pivot used to place the layer. For crop
            nodes, it is usually relative to the emitted crop image. For
            placement nodes, it is relative to the full attached image.

        ``position`` : str or list/tuple of two int | str | None values, optional
            Target canvas position where ``anchor_xy`` must be placed.

            If omitted, the layer ``position`` declared via :meth:`image` is
            used. If provided, it overrides that layer position.

            String values use the same edge/corner placement names accepted by
            :meth:`image`, such as ``'center'``, ``'top-left'``, or
            ``'bottom-right'``.

            Tuple/list values define explicit canvas coordinates for the
            anchor. Components may be integers, pixel strings, percentage
            strings, or ``None``. A ``None`` component resolves to the center of
            the corresponding canvas axis.

        ``bbox_size`` : list or tuple of two int | str | None values, optional
            Target size for the layer.

            If omitted, the layer ``resize`` declared via :meth:`image` is used.
            If provided, it overrides that layer resize setting.

            The value follows the same tuple resize conventions as
            :meth:`image`: ``(W, H)``, ``(W, None)``, or ``(None, H)``.

        Runtime behavior
        ----------------
        If no transform input is connected, the layer uses the geometry declared
        via :meth:`image`:

        - the default anchor is the geometric center of the input layer;
        - ``position`` comes from :meth:`image`;
        - ``resize`` comes from :meth:`image`.

        If a transform input is connected:

        - ``anchor_xy`` is read from ``crop`` or ``placement`` and becomes the
          layer local anchor;
        - ``position`` is overridden only if present in the transform metadata;
        - ``resize`` is overridden only if ``bbox_size`` is present.

        Geometry compatibility
        ----------------------
        Transform metadata is interpreted in the local coordinate system of the
        image attached to this layer. ``ImageStack`` does not know whether that
        image is still geometrically equivalent to the upstream crop or placement
        source.

        This matters when a crop is sent through a refinement or generation step
        that changes its pixel size, for example producing a 1024x1024 image
        from a much smaller face crop. In that case, ``crop.anchor_xy`` still
        refers to the original crop coordinate frame, while the layer image now
        has a different coordinate frame. The resulting placement may therefore
        look wrong even if ``position`` and ``bbox_size`` are correct.

        For crop reconstruction workflows, the recommended pattern is to resize
        the refined/generated layer back to the transform ``bbox_size`` before
        feeding it to ``ImageStack``. ``ResizeImage.transform()`` can consume the
        same ``crop`` or ``placement`` metadata and use its ``bbox_size`` for
        this alignment step.

        The layer is then transformed while preserving anchor consistency:

        1. ``corner_offsets`` are applied and the anchor is transformed with the
           warped layer;
        2. ``rotation`` is applied around the current local anchor;
        3. ``resize`` is applied and the anchor is scaled accordingly;
        4. ``position`` is resolved into a canvas placement point;
        5. the layer is composited so that its local anchor coincides with that
           placement point.

        Returns
        -------
        AttachmentSink
            A sink bound to this node with ``input_id=f"transform:{idx}"``.
            It must be wired from a node producing compatible ``crop`` or
            ``placement`` metadata.

        Notes
        -----
        - The transform input affects only spatial placement metadata:
          ``anchor_xy``, ``position``, and ``bbox_size``.
        - Visual layer operations such as ``corner_offsets``, ``rotation``,
          ``brightness``, ``color_transfer``, ``feather``, ``corner_radius``,
          and ``alpha`` remain controlled by the layer declaration.
        - ``crop`` and ``placement`` are mutually exclusive. Supplying both is
          considered ambiguous.
        """

        return AttachmentSink(
            name=f'imagestack_transform:{self.target.id}:{self.idx}',
            target=self.target,
            input_id=f'transform:{self.idx}',
        )


@dataclass
class ImageStack(NodeRef):
    """
    Layer-based image compositor node.

    ``ImageStack`` is a geometry-driven compositing node that collects multiple
    upstream images via typed sinks created by :meth:`image` and composites
    them in ascending ``idx`` order onto a configurable canvas.

    The node is designed for DAG-native compositing workflows such as:

    - placing subject cutouts onto new backgrounds,
    - reconstructing refined regions back into their original frame,
    - assembling multiple extracted elements into a single image,
    - layering logos, UI elements, masks, or overlays procedurally.

    Layer model
    -----------
    Each upstream connection declared via :meth:`image` defines a *layer*.

    Layers are rendered deterministically with respect to ordering:

    - lowest ``idx`` → bottom layer
    - highest ``idx`` → top layer

    For each layer, the following pipeline is executed:

    1. Load the upstream image and convert it to ``RGBA``.
    2. Resolve optional ``crop`` / ``placement`` transform metadata.
    3. Initialize the layer local anchor.
    4. Apply optional local perspective corner offsets and propagate the anchor.
    5. Apply optional rotation around the current anchor.
    6. Apply optional resizing and scale the anchor accordingly.
    7. Adjust layer luminosity.
    8. Resolve layer placement on the canvas.
    9. Optionally match layer Lab color statistics to the underlying canvas.
    10. Optionally soften the layer edges.
    11. Apply global layer opacity.
    12. Alpha-composite the transformed layer by matching anchor to placement.

    This execution order is important:

    - ``anchor_xy`` is initialized before local geometric transforms;
    - ``corner_offsets`` is applied first, in the current local layer coordinate
      space, and propagates the anchor through the perspective warp;
    - ``rotation`` is applied after the local corner warp, around the current
      propagated anchor;
    - ``resize`` is applied after both local corner warp and rotation, and
      scales the propagated anchor accordingly;
    - therefore ``resize`` refers to the transformed layer bounding box, not to
      the original unwarped image size;
    - final placement is computed by matching the transformed local anchor to a
      resolved canvas placement point.

    Canvas
    ------
    Canvas geometry can be supplied by a wired ``default`` input or by
    ``params.width`` and ``params.height``. When ``input['default']`` contains an
    ``image`` or ``path`` entry, that image becomes the background canvas and
    its dimensions are used; ``params.width``, ``params.height`` and
    ``params.background`` are ignored for that run.

    Without a ``default`` background input, the canvas is configured from params.
    The canvas is always constructed in ``RGBA`` mode and may be initialized as:

    - fully transparent (``background = null``),
    - a named PIL color (for example ``'white'``),
    - an explicit RGB or RGBA tuple.

    Final output color mode is controlled by ``params.out_mode``:

    - ``'RGBA'`` → preserve the alpha channel,
    - ``'RGB'`` → drop the alpha channel before saving.

    Corner offsets
    --------------
    Each layer may optionally define ``corner_offsets`` to apply a local perspective
    warp before rotation, resizing, feathering, and placement.

    ``corner_offsets`` is a :class:`CornerOffsets` instance. It contains optional
    per-corner displacements:

    - ``top_left``
    - ``top_right``
    - ``bottom_right``
    - ``bottom_left``

    Each corner value may be either:

    - ``None``:
        Do not move that corner.
    - ``(dx, dy)``:
        Move that corner by ``dx`` and ``dy`` in local layer coordinates.

    The corner order is clockwise:

    1. top-left
    2. top-right
    3. bottom-right
    4. bottom-left

    Delta components support:

    - ``int``:
        Absolute displacement in pixels.
    - ``'<number>px'``:
        Absolute displacement in pixels.
    - ``'<number>%'``:
        Percentage displacement relative to the current layer size.

    For percentage deltas, horizontal components are resolved against the current
    layer width, while vertical components are resolved against the current layer
    height.

    Examples:

    - ``CornerOffsets(top_left=(-10, 5))``
        Move only the top-left corner by ``-10`` pixels horizontally and ``5`` pixels
        vertically.
    - ``CornerOffsets(top_right=('5%', '-3%'))``
        Move only the top-right corner by ``5%`` of layer width and ``-3%`` of layer
        height.
    - ``CornerOffsets(top_left=('-4%', '2%'), bottom_right=('3%', '5%'))``
        Apply a light perspective-like deformation to two opposite corners.

    The local warped canvas is expanded so moved corners are preserved instead of
    being clipped.

    Rotation
    --------
    Each layer may optionally be rotated after local corner-offset warping and
    before resizing and placement using ``rotation``.

    - ``0.0``:
        No rotation.
    - ``float``:
        Counter-clockwise rotation angle in degrees.

    Rotation is applied around the current layer anchor.

    Without transform metadata, the anchor is initialized to the geometric
    center of the input layer, so rotation behaves like ordinary center-based
    rotation.

    With ``crop`` or ``placement`` transform metadata, the anchor may be any
    local point declared by the upstream node. In that case, rotation preserves
    that logical pivot instead of rotating around the transformed bounding-box
    center.

    The rotated canvas is expanded so that the full rotated content is
    preserved, and newly exposed pixels are filled with transparency.

    Since rotation is applied before resizing, any subsequent ``resize``
    operation acts on the rotated layer as a whole.

    Resizing
    --------
    Each layer may be resized prior to placement using ``resize``:

    - ``None``:
        Preserve the current layer size.
    - ``(W, H)``:
        Force exact dimensions.
    - ``(W, None)``:
        Set width to ``W`` and preserve aspect ratio.
    - ``(None, H)``:
        Set height to ``H`` and preserve aspect ratio.
    - ``'fit'``:
        Isotropic resize to the largest size that fits entirely inside the
        canvas without distortion.
    - ``'cover'``:
        Isotropic resize to the smallest size that fully covers the canvas
        without distortion. The layer may extend beyond canvas bounds and is
        clipped during compositing.

    Each component in tuple mode may be:

    - ``int``:
        Explicit size in pixels.
    - ``'<number>px'``:
        Explicit size in pixels.
    - ``'<number>%'``:
        Percentage of the canvas dimension
        (width for ``W``, height for ``H``).
    - ``None``:
        Preserve aspect ratio on that axis.

    Examples:

    - ``(512, 512)`` → force exact size,
    - ``('50%', None)`` → width = 50% of canvas, height scaled proportionally,
    - ``(None, '30%')`` → height = 30% of canvas, width scaled proportionally.

    Positioning
    -----------
    The ``position`` parameter controls layer placement on the canvas.

    - If a tuple ``(x, y)`` is provided, it represents the canvas position where the
      current layer anchor must be placed. Without transform metadata, the default
      anchor is the geometric center of the input layer, so this behaves like
      center-based placement. With transform metadata, the anchor may be any local
      point declared by the upstream node.
    - If a string anchor is provided, the layer is attached to the
      corresponding edge or corner such that its bounding box touches the
      canvas border(s), not such that its center lies on the border.

    Supported anchors include:

    - ``'center'``
    - ``'top'``, ``'bottom'``, ``'left'``, ``'right'``
    - ``'top-left'``, ``'top-right'``, ``'bottom-left'``, ``'bottom-right'``
    - ``'center-top'``, ``'center-bottom'``, ``'center-left'``, ``'center-right'``

    Placement resolution depends on the canvas dimensions, the transformed layer
    dimensions, and the current propagated local anchor. This logic is
    implemented by ``_resolve_position_xy``.

    Transform input (crop / placement driven geometry)
    --------------------------------------------------
    A layer may optionally declare an additional transform input via
    :meth:`ImageLayerAttachmentSink.transform`.

    If a compatible upstream node is wired into ``transform:{idx}``, ``ImageStack``
    looks for either ``crop`` or ``placement`` metadata.

    The metadata must provide:

    - ``anchor_xy``:
      local layer anchor coordinates.

    It may also provide:

    - ``position``:
      canvas placement specification overriding ``image(position=...)``;
    - ``bbox_size``:
      target layer size overriding ``image(resize=...)``.

    This enables both crop reconstruction workflows and generic semantic placement
    workflows.

    Geometry compatibility
    ----------------------
    ``crop`` and ``placement`` metadata describe coordinates in the local geometry
    of the image that produced that metadata. The image wired as the actual layer
    should therefore be geometrically compatible with that metadata.

    If a crop is refined or regenerated at a different resolution before being
    reinserted, resize it back to the transform ``bbox_size`` first. Otherwise
    ``anchor_xy`` may refer to the old crop coordinates while the attached layer
    uses a new coordinate frame. ``ResizeImage.transform()`` is designed for this
    case: wire the same crop/placement metadata into the resize node so it can
    use ``bbox_size`` as the target size before the layer reaches ``ImageStack``.

    Feathering
    ----------
    ``feather`` softens layer edges before compositing.

    Accepted formats are:

    - ``int`` → feather width in pixels
    - ``'<number>px'`` → explicit pixel units
    - ``'<number>%'`` → percentage of the layer size

    The behavior depends on the alpha channel of the layer.

    **Fully opaque alpha (typical for ``crop_mode='bbox'``):**

    - The layer contains no transparency.
    - A synthetic alpha mask is generated.
    - The mask edge is softened using a feather ramp derived from
      the specified feather width.
    - When a percentage is used, the feather width is resolved
      independently for the horizontal and vertical axes.

    **Non-uniform alpha (for example segmentation masks or subject cutouts):**

    - The existing alpha channel is preserved.
    - The existing alpha channel is refined and softened using a small
      erosion followed by Gaussian blur.
    - Percentage values are resolved relative to the average of the
      layer width and height.

    A value of ``0`` disables feathering.

    Brightness
    ----------
    ``brightness`` adjusts the luminosity of each layer independently before
    compositing, without changing its alpha channel:

    - ``0.0`` preserves the original layer colors;
    - negative values darken the layer;
    - ``-1.0`` makes the layer RGB channels black;
    - positive values brighten the layer;
    - ``1.0`` applies the maximum supported brightening.

    This is useful when an overlay has insufficient contrast against its target,
    for example darkening white lettering before placing it on a white T-shirt.
    The adjustment is explicit and does not inspect or modify the canvas below
    the layer.

    Color Transfer
    --------------
    ``color_transfer`` optionally reduces chromatic drift when reinserting a
    refined crop into its original canvas.

    When enabled, ``ImageStack`` compares the visible layer pixels with the
    canvas pixels currently underneath the same area. For each Lab channel, it
    shifts and scales the layer distribution to match the target region mean
    and standard deviation, then blends the corrected Lab colors back over the
    original layer colors according to ``color_transfer``.

    Pixels with alpha ``0`` in either the layer or the underlying canvas are
    ignored while computing and applying the correction. The layer alpha channel
    is preserved.

    A value of ``0.0`` disables color transfer. A value of ``1.0`` applies the
    full Lab mean/std match.

    Configuration
    -------------
    ``spec`` may be a dictionary or a path-like configuration file.

    Expected structure:

    ``params`` : dict
        ``width`` : int, optional
            Output canvas width in pixels.
            Ignored when a ``default`` background input is provided.

        ``height`` : int, optional
            Output canvas height in pixels.
            Ignored when a ``default`` background input is provided.

        ``background`` : str or RGB/RGBA sequence or null, optional
            Initial canvas background.
            Ignored when a ``default`` background input is provided.

            Accepted forms are:

            - ``null`` / ``None``:
              Create a fully transparent canvas.

            - ``str``:
              Named color or CSS-style color string accepted by PIL, such as
              ``'white'``, ``'black'``, ``'#ffffff'``, ``'#000000'``,
              or ``'#ffcc00'``.

            - RGB sequence:
              A list or tuple of three integers ``[r, g, b]`` or
              ``(r, g, b)``. The alpha channel is assumed to be fully opaque.

            - RGBA sequence:
              A list or tuple of four integers ``[r, g, b, a]`` or
              ``(r, g, b, a)``.

            Channel values must be in the ``0..255`` range.
        ``out_mode`` : {'RGBA', 'RGB'}
            Output color mode.

    Methods
    -------
    image(
        idx: int,
        position: tuple[int | str | None, int | str | None] | str = 'center',
        resize=None,
        rotation=0.0,
        corner_offsets=None,
        brightness=0.0,
        color_transfer=0.0,
        feather=0,
        corner_radius=None,
        alpha=1.0,
    )
        Declare a compositing layer and return an :class:`AttachmentSink`
        for wiring.

        ``idx`` must be unique. Layers are rendered in increasing order.

    Returns
    -------
    dict
        Output metadata dictionary containing:

        ``ok`` : bool
            Success flag.
        ``node`` : str
            Operator name.
        ``id`` : str
            Node identifier.
        ``image`` : str
            Path to the composited output image.
        ``params.layers`` : list[dict]
            Per-layer configuration, including ``idx``, ``position``, ``resize``,
            ``rotation``, ``corner_offsets``, ``brightness``,
            ``color_transfer``, ``feather``, ``corner_radius``, and ``alpha``.
        ``metadata`` : str
            Path to the JSON sidecar.

    Notes
    -----
    - Upstream nodes must provide an image path under ``'image'`` or ``'path'``.
    - Layers are composited deterministically with respect to ordering.
    - Layers extending beyond canvas bounds are safely clipped.
    - Synthetic alpha masks used for fully opaque layers include a small
      stochastic edge perturbation to reduce visible rectangular seams.
      Consequently, the exact output pixels may vary slightly between runs.
    """

    spec: SpecInput = field(default_factory=dict)
    _layers: Dict[int, LayerSpec] = field(
        default_factory=dict,
        init=False,
        repr=False
    )

    def image(
        self,
        idx: int,
        *,
        position: PositionSpec = 'center',
        resize: ResizeMode = None,
        rotation: float = 0.0,
        corner_offsets: Optional[CornerOffsets] = None,
        brightness: float = 0.0,
        color_transfer: float = 0.0,
        feather: int | str = 0,
        corner_radius: Optional[int | str] = None,
        alpha: float = 1.0,
    ) -> ImageLayerAttachmentSink:
        """
        Declare an input image layer.

        This method registers a new compositing layer and returns an
        :class:`AttachmentSink` that can be wired to an upstream node
        producing an image.

        Layers are rendered in ascending ``idx`` order:
        lower indices form the background, higher indices are drawn on top.

        Parameters
        ----------
        idx : int
            Unique layer index. Must not be reused.
            Layers are composited in increasing order (bottom → top).

        position : tuple[int | str | None, int | str | None] or str, optional
            Placement of the layer on the canvas.

            If a tuple ``(x, y)`` is provided, it represents the canvas position
            where the current layer anchor must be placed.

            Tuple components may be:

            - ``int``:
              Absolute canvas coordinate in pixels.
            - ``"<number>px"``:
              Explicit pixel coordinate.
            - ``"<number>%"``:
              Percentage of the corresponding canvas dimension.
            - ``None``:
              Use the center of the corresponding canvas axis.

            If a string is provided, it is interpreted as an edge/corner
            placement. The layer is positioned so that its transformed bounding
            box touches the requested canvas edge or corner.

            Without transform metadata, the layer anchor defaults to the
            geometric center of the input layer. With transform metadata, the
            upstream ``crop`` or ``placement`` metadata may define a different
            local anchor.

            Supported anchors include:

            ``"center"``,
            ``"top"``, ``"bottom"``, ``"left"``, ``"right"``,
            ``"top-left"``, ``"top-right"``, ``"bottom-left"``,
            ``"bottom-right"``, ``"center-top"``, ``"center-bottom"``,
            ``"center-left"``, ``"center-right"``.

            Default: ``"center"``.

        resize : tuple[int | str | None, int | str | None] or {"fit", "cover"} or
                 None, optional
            Optional resizing applied before placement.

            Accepted forms:

            - ``None``:
            No resizing. The original image size is preserved.

            - ``"fit"``:
            Isotropic resize to the largest size that fits entirely inside
            the canvas without distortion.

            - ``"cover"``:
            Isotropic resize to the smallest size that fully covers the canvas
            without distortion. The layer may extend beyond canvas bounds and
            will be cropped during compositing.

            - ``(W, H)``:
            Explicit target size.

            Each component may be:

            - ``int`` → size in pixels
            - ``"<number>px"`` → explicit pixel units
            - ``"<number>%"`` → percentage of the canvas dimension
                (width for ``W``, height for ``H``)
            - ``None`` → keep aspect ratio along that axis

            Examples:

            - ``(512, 512)`` → force exact size
            - ``("50%", None)`` → width = 50% of canvas, height scaled to preserve aspect ratio
            - ``(None, "30%")`` → height = 30% of canvas, width scaled proportionally

        rotation : float, optional
            Counter-clockwise rotation angle in degrees applied to the layer
            before resizing, feathering, and placement.

            Rotation is applied around the current layer anchor.

            Without transform metadata, the anchor defaults to the geometric
            center of the input layer. With transform metadata, the upstream
            ``crop`` or ``placement`` metadata may define a different local
            anchor.

            The rotated canvas is expanded to preserve the full rotated
            content, and newly exposed pixels are transparent.

            Since resizing is applied after rotation, ``resize`` refers to the
            final rotated layer bounding box, not to the original unrotated
            image size.

            Since placement is resolved after rotation and resizing, tuple
            positions and anchor strings refer to the propagated anchor and the
            transformed layer bounding box.

            Default: ``0.0``.

        corner_offsets : CornerOffsets or None, optional
            Optional local perspective corner warp applied before rotation, resizing,
            feathering, and placement.

            The value is a :class:`CornerOffsets` instance with optional per-corner
            deltas:

            - ``top_left``
            - ``top_right``
            - ``bottom_right``
            - ``bottom_left``

            Each corner delta may be:

            - ``None`` → no displacement for that corner
            - ``(dx, dy)`` → move that corner in local layer coordinates

            Delta components may be:

            - ``int`` → absolute displacement in pixels
            - ``"<number>px"`` → absolute displacement in pixels
            - ``"<number>%"`` → percentage of the current layer size

            Percentage deltas are resolved before resizing:

            - horizontal deltas use the current layer width
            - vertical deltas use the current layer height

            The local warped canvas is expanded to preserve moved corners.

            Example:

            ``CornerOffsets(top_left=("-4%", "2%"), top_right=("6%", "-3%"))``

            Default: ``None``.

        brightness : float, optional
            Layer luminosity adjustment applied after geometric transforms and
            resizing, but before feathering and compositing.

            - ``-1.0`` produces black RGB while preserving alpha.
            - Values in ``(-1.0, 0.0)`` darken the layer.
            - ``0.0`` keeps the original RGB values.
            - Values in ``(0.0, 1.0]`` brighten the layer.

            Internally, the adjustment is converted to the PIL brightness
            factor ``1 + brightness``. The value must be finite and in
            ``[-1.0, 1.0]``. The alpha channel is preserved exactly.

            Default: ``0.0``.

        color_transfer : float, optional
            Lab-space statistical color transfer applied after brightness and
            before feathering and compositing.

            The value must be finite and in ``[0.0, 1.0]``:

            - ``0.0`` disables color transfer.
            - intermediate values blend between the current layer colors and
              the statistically matched colors.
            - ``1.0`` fully applies the Lab mean/std match.

            Statistics are computed from the visible layer area and the canvas
            pixels currently underneath that area. Pixels with alpha ``0`` in
            either image are ignored. The layer alpha channel is preserved.

            Default: ``0.0``.

        feather : int or str, optional
            Feathering applied to the layer edges before compositing.

            Supported formats:

            - ``int`` → feather width in pixels
            - ``"<number>px"`` → explicit pixel units
            - ``"<number>%"`` → percentage of the layer size

            The interpretation depends on the alpha channel of the layer:

            **Opaque alpha (typical for ``crop_mode='bbox'``):**

            - The alpha channel contains no transparency.
            - A synthetic alpha mask is generated.
            - The mask edge is softened using a feather ramp whose width is
              derived from ``feather``.
            - When a percentage is used, the feather width is resolved
              independently for the horizontal and vertical axes.

            **Non-uniform alpha (e.g. segmentation masks, subject cutouts):**

            - The existing alpha channel is preserved.
            - A Gaussian blur is applied to the alpha channel to soften edges.
            - Percentage values are resolved relative to the average of the
            layer width and height.

            A value of ``0`` disables feathering.

            Default: ``0``.

        corner_radius : int or str or None, optional
            Corner rounding radius used when generating the synthetic alpha mask
            for fully opaque layers.

            Supported formats are:

            - ``None`` → Automatic mode. The corner radius is derived
              heuristically from the feather width.
            - ``int`` → Explicit radius in pixels.
            - ``"<number>px"`` → Explicit radius in pixels.
            - ``"<number>%"`` → Percentage of the maximum valid corner radius,
              defined as half of the shorter layer side.

            A value of ``0`` or ``"0%"`` disables corner rounding.

            This parameter has no effect when the layer already contains
            transparency (e.g. segmentation masks).

        alpha : float, optional
            Global opacity multiplier applied to the layer before compositing.

            The value must be in the ``[0.0, 1.0]`` range:

            - ``1.0``:
              Keep the layer opacity unchanged.
            - ``0.0``:
              Make the layer fully transparent.
            - intermediate values:
              Multiply the current alpha channel by the given factor.

            This operation is applied after local transformations, resizing,
            and feathering, but before final alpha compositing onto the canvas.

            Since ``alpha`` multiplies the existing alpha channel, it works both
            for fully opaque layers and for layers that already contain
            transparency, such as segmentation cutouts.

            Default: ``1.0``.

        Returns
        -------
        AttachmentSink
            A sink bound to this node with ``input_id=f"image:{idx}"``.
            The upstream node must provide an output image under the standard
            keys (``"image"`` or ``"path"``).

        Notes
        -----
        - The layer transformation is purely geometric and includes anchor-aware
          local corner-offset warping, anchor-centered rotation, resizing, and
          final placement.
        - ``color_transfer`` performs only statistical color matching. It does
          not synthesize lighting, shadows, or geometry-aware harmonization.
        - ``idx`` must be unique; attempting to reuse an index raises an error.
        """

        idx = int(idx)
        if idx < 0:
            raise ValueError(f'{self.id}: idx must be >= 0, got {idx}')

        # --- validate resize ---
        if resize is not None:

            if isinstance(resize, str):
                if resize not in ('fit', 'cover'):
                    raise ValueError(
                        f'{self.id}: invalid resize mode {resize!r}, '
                        "expected 'fit' or 'cover'"
                    )

            elif isinstance(resize, tuple):
                if len(resize) != 2:
                    raise ValueError(
                        f'{self.id}: resize tuple must have length 2, got {resize!r}'
                    )

                w_target, h_target = resize

                if w_target is None and h_target is None:
                    raise ValueError(
                        f'{self.id}: resize cannot be (None, None)'
                    )

                if w_target is not None:
                    validate_size_expr(w_target)

                if h_target is not None:
                    validate_size_expr(h_target)

            else:
                raise TypeError(
                    f'{self.id}: resize must be None, tuple or str, got {type(resize)}'
                )

        rotation = float(rotation)
        if not math.isfinite(rotation):
            raise ValueError(
                f'{self.id}: rotation must be a finite number, got {rotation!r}'
            )

        if corner_offsets is not None:
            for name, delta in (
                ('top_left', corner_offsets.top_left),
                ('top_right', corner_offsets.top_right),
                ('bottom_right', corner_offsets.bottom_right),
                ('bottom_left', corner_offsets.bottom_left),
            ):
                if delta is None:
                    continue
                if not isinstance(delta, tuple) or len(delta) != 2:
                    raise ValueError(
                        f'{self.id}: corner_offsets.{name} must be None or a '
                        f'(dx, dy) tuple, got {delta!r}'
                    )

                dx, dy = delta

                # Validate syntax using dummy reference sizes.
                resolve_delta_expr(dx, max_size=100)
                resolve_delta_expr(dy, max_size=100)

        idx = int(idx)

        if idx in self._layers:
            raise ValueError(
                f'{self.id}: layer idx={idx} already declared. '
                'Each layer index must be unique.'
            )

        if corner_radius is not None:
            validate_size_expr(corner_radius)

        validate_size_expr(feather)

        alpha = float(alpha)
        if not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f'{self.id}: alpha must be a finite float in [0, 1], got {alpha!r}'
            )

        brightness = float(brightness)
        if not math.isfinite(brightness) or not (-1.0 <= brightness <= 1.0):
            raise ValueError(
                f'{self.id}: brightness must be a finite float in [-1, 1], '
                f'got {brightness!r}'
            )

        color_transfer = float(color_transfer)
        if not math.isfinite(color_transfer) or not (0.0 <= color_transfer <= 1.0):
            raise ValueError(
                f'{self.id}: color_transfer must be a finite float in [0, 1], '
                f'got {color_transfer!r}'
            )

        self._layers[idx] = LayerSpec(
            idx=idx,
            position=position,
            resize=resize,
            rotation=rotation,
            corner_offsets=corner_offsets,
            brightness=brightness,
            color_transfer=color_transfer,
            feather=feather,
            corner_radius=corner_radius,
            alpha=alpha,
        )

        return ImageLayerAttachmentSink(
            idx=idx,
            name=f'imagestack_image:{self.id}:{idx}',
            target=self,
            input_id=f'image:{idx}',
        )

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)
        cfg = _read_cfg(spec, node_id=self.id)

        input = input or {}

        # Build canvas.
        background_input_path = _input_image_path(
            input.get('default'),
            node_id=self.id,
            input_name='default',
        )
        if background_input_path is not None:
            with Image.open(background_input_path) as im:
                canvas = im.convert('RGBA')
            canvas_w, canvas_h = canvas.size
            background_source = 'input:default'
            background_value: Any = str(background_input_path)
        else:
            canvas_w = cfg.width if cfg.width is not None else 1024
            canvas_h = cfg.height if cfg.height is not None else 1024
            if cfg.background is None:
                canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
            elif isinstance(cfg.background, str):
                canvas = Image.new(
                    'RGBA', (canvas_w, canvas_h), cfg.background
                )
            else:
                canvas = Image.new(
                    'RGBA', (canvas_w, canvas_h), cfg.background
                )
            background_source = 'params'
            background_value = cfg.background

        if not self._layers:
            raise ValueError(
                f'{self.id}: no layers declared. Use src >> node.image(idx=...)')

        # Composite layers in ascending idx order.
        for idx in sorted(self._layers.keys()):
            layer_spec = self._layers[idx]
            up = input.get(f'image:{idx}')
            if up is None:
                raise ValueError(
                    f'{self.id}: missing upstream for layer idx={idx} '
                    f"(expected wiring into input_id='image:{idx}')"
                )

            path = _input_image_path(
                up,
                node_id=self.id,
                input_name=f'image:{idx}',
            )

            with Image.open(path) as im:
                layer = im.convert('RGBA')

            # --- optional crop / placement transform metadata ---
            tr = input.get(f'transform:{idx}')

            position = layer_spec.position
            resize = layer_spec.resize

            anchor_xy: Point = (
                float(layer.width) / 2.0,
                float(layer.height) / 2.0,
            )

            if tr is not None:
                default_spatial = SpatialTransform(
                    anchor_xy=anchor_xy,
                    position=position,
                )
                transform_spatial = read_spatial_transform(
                    tr,
                    node_id=self.id,
                    input_name=f'transform:{idx}',
                )
                if transform_spatial is None:
                    raise ValueError(
                        f'{self.id}: missing transform metadata on transform:{idx}; '
                        "expected key 'crop' or 'placement'."
                    )

                spatial = merge_spatial_transform(
                    default_spatial,
                    transform_spatial,
                )

                if spatial is None:
                    raise ValueError(
                        f'{self.id}: failed to resolve spatial transform for layer idx={idx}.'
                    )

                if spatial.anchor_xy is None:
                    raise ValueError(
                        f'{self.id}: missing resolved anchor_xy for transform:{idx}. '
                        'The layer default anchor or transform metadata must provide it.'
                    )

                if spatial.position is None:
                    raise ValueError(
                        f'{self.id}: missing resolved position for transform:{idx}. '
                        'The layer default position or transform metadata must provide it.'
                    )

                anchor_xy = spatial.anchor_xy
                position = spatial.position

                if spatial.bbox_size is not None:
                    resize = spatial.bbox_size

            layer, anchor_xy = _apply_corner_offsets(
                layer,
                layer_spec.corner_offsets,
                anchor_xy,
            )

            layer, anchor_xy = _apply_rotation_around_anchor(
                layer,
                layer_spec.rotation,
                anchor_xy,
            )

            layer, anchor_xy = _apply_resize(
                layer,
                resize,
                canvas_w,
                canvas_h,
                anchor_xy,
            )

            if layer_spec.brightness != 0.0:
                r, g, b, a = layer.split()
                rgb = Image.merge('RGB', (r, g, b))
                rgb = ImageEnhance.Brightness(rgb).enhance(
                    1.0 + layer_spec.brightness
                )
                r, g, b = rgb.split()
                layer = Image.merge('RGBA', (r, g, b, a))

            lw, lh = layer.size

            placement_xy = _resolve_position_xy(
                position,
                canvas_w,
                canvas_h,
                layer.width,
                layer.height,
                anchor_xy,
            )

            if layer_spec.color_transfer > 0.0:
                visible = _visible_layer_and_canvas_crops(
                    canvas,
                    layer,
                    placement_xy=placement_xy,
                    anchor_xy=anchor_xy,
                )
                if visible is not None:
                    layer_box, layer_crop, canvas_crop = visible
                    matched_crop = match_lab_color_statistics(
                        layer_crop,
                        canvas_crop,
                        strength=layer_spec.color_transfer,
                    )
                    layer.paste(matched_crop, layer_box)

            # Apply feathering through alpha-channel handling.
            if feather_is_nonzero(layer_spec.feather):
                r, g, b, a = layer.split()

                # If alpha is fully opaque (typical for crop_mode='bbox'),
                # blurring it does nothing. Build a soft-edge alpha mask instead.
                a_min, a_max = a.getextrema()

                if a_min == 255 and a_max == 255:
                    feather_x, feather_y = resolve_feather_xy(
                        layer_spec.feather,
                        width=lw,
                        height=lh,
                    )
                    if feather_x > 0 or feather_y > 0:
                        max_corner = max(0, (min(lw, lh) // 2) - 1)

                        # Corner rounding radius.
                        # - None: backward compatible heuristic based on feather radius
                        # - 0: no rounding
                        # - >0: explicit
                        if layer_spec.corner_radius is None:
                            corner = int((feather_x + feather_y) * 3.0 / 2.0)
                        else:
                            corner = resolve_size_expr(
                                layer_spec.corner_radius,
                                max_size=max_corner,
                                min_size=0,
                            )

                        # Clamp to avoid impossible / over-rounding geometries
                        corner = max(0, min(corner, max_corner))

                        a = _perturbed_alpha_ramp(
                            lw,
                            lh,
                            feather_x=feather_x,
                            feather_y=feather_y,
                            corner_radius=corner,
                        )
                        layer = Image.merge('RGBA', (r, g, b, a))

                else:
                    # Normal case: soften existing cutout edges
                    rad = resolve_feather_radius(
                        layer_spec.feather,
                        width=lw,
                        height=lh,
                    )
                    if rad > 0:
                        trim = min(2, max(1, rad // 10)) if rad > 0 else 0
                        a_core = a.filter(
                            ImageFilter.MinFilter(trim * 2 + 1)
                        )
                        a_soft = a_core.filter(
                            ImageFilter.GaussianBlur(radius=rad)
                        )
                        a = ImageChops.darker(a_soft, a)
                        layer = Image.merge('RGBA', (r, g, b, a))

            if layer_spec.alpha < 1.0:
                r, g, b, a = layer.split()
                a = a.point(lambda v: int(round(v * layer_spec.alpha)))
                layer = Image.merge('RGBA', (r, g, b, a))

            _place_on_canvas_by_anchor(
                canvas,
                layer,
                placement_xy=placement_xy,
                anchor_xy=anchor_xy,
            )

        out_dir = Path(output_dir)

        out_path = make_node_output_path(out_dir=out_dir, node_id=self.id)

        if cfg.out_mode == 'RGB':
            canvas.convert('RGB').save(out_path)
        else:
            canvas.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'image': str(out_path),
            'params': {
                'width': canvas_w,
                'height': canvas_h,
                'background': background_value,
                'background_source': background_source,
                'out_mode': cfg.out_mode,
                'layers': [
                    {
                        'idx': ls.idx,
                        'position': ls.position,
                        'resize': ls.resize,
                        'rotation': ls.rotation,
                        'corner_offsets': None if ls.corner_offsets is None else {
                            'top_left': ls.corner_offsets.top_left,
                            'top_right': ls.corner_offsets.top_right,
                            'bottom_right': ls.corner_offsets.bottom_right,
                            'bottom_left': ls.corner_offsets.bottom_left,
                        },
                        'brightness': ls.brightness,
                        'color_transfer': ls.color_transfer,
                        'feather': ls.feather,
                        'corner_radius': ls.corner_radius,
                        'alpha': ls.alpha,
                    }
                    for ls in (self._layers[i] for i in sorted(self._layers.keys()))
                ],
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)
        return out
