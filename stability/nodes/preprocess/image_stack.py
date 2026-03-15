import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from PIL import Image, ImageChops, ImageFilter

from stability.core.paths import make_node_output_path
from stability.dag import AttachmentSink, NodeRef
from stability.nodes.common.config_resolve import SpecInput, resolve_spec
from stability.nodes.common.io import write_json_sidecar

Pos = Union[Tuple[int, int], str]
ResizeMode = Tuple[Optional[int | str], Optional[int | str]] \
    | Literal['fit', 'cover'] \
    | None


@dataclass(frozen=True)
class LayerSpec:
    idx: int
    position: Pos = 'center'
    resize: ResizeMode = None
    feather: int | str = 0
    corner_radius: Optional[int | str] = None


@dataclass
class Config:
    width: int
    height: int
    background: Optional[Union[str, Tuple[int, int, int, int]]]
    out_mode: Literal['RGBA', 'RGB']


def validate_size_expr(size_expr: int | str) -> None:
    # Absolute pixel value
    if isinstance(size_expr, int):
        if size_expr < 0:
            raise ValueError('size expression must be >= 0')
        return

    # Must be a string from here
    if not isinstance(size_expr, str):
        raise ValueError(
            f'invalid size expression type {type(size_expr).__name__}; '
            'expected int or str'
        )

    s = size_expr.strip().lower()

    pattern = r'^\d+(\.\d+)?(px|%)$'

    if not re.match(pattern, s):
        raise ValueError(
            f'invalid size expression value "{size_expr}". '
            'Expected formats: int, "<number>px", "<number>%".'
        )


def resolve_size_expr(
    size_expr: int | str,
    *,
    max_size: int,
    min_size: int = 1,
) -> int:
    """
    Resolve a size expression to pixels.

    Parameters
    ----------
    size_expr : int or str
        Size specification.

        Supported formats are:

        - ``int``:
          Explicit size in pixels.
        - ``'<number>px'``:
          Explicit size in pixels.
        - ``'<number>%'``:
          Percentage of ``max_size``.

    max_size : int
        Reference size used to resolve percentage expressions.
    min_size : int, default=1
        Minimum resolved value returned by the function.

        This is useful because some geometric quantities, such as output
        image size, should never collapse to zero, while others, such as
        corner radius, may validly resolve to zero.

    Returns
    -------
    int
        Resolved size in pixels, clamped to be at least ``min_size``.
    """
    validate_size_expr(size_expr)

    if isinstance(size_expr, int):
        return max(min_size, size_expr)

    s = size_expr.strip().lower()

    if s.endswith('px'):
        return max(min_size, int(round(float(s[:-2]))))

    pct = max(0.0, float(s[:-1])) / 100.0

    return max(min_size, int(round(max_size * pct)))


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

    # I think that feather, if defined non zero, should always be at least 1
    return max(1, int(round(mean_dim * pct)))


def feather_is_nonzero(feather: int | str) -> bool:
    if isinstance(feather, int):
        return feather > 0

    s = feather.strip().lower()

    return float(s[:-2] if s.endswith('px') else s[:-1]) > 0


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    width = int(params.get('width', 1024))
    height = int(params.get('height', 1024))
    if width <= 0 or height <= 0:
        raise ValueError(f"'{node_id}': invalid canvas size {width}x{height}")

    # background:
    # - None -> transparent
    # - 'white'/'black'/etc accepted by PIL
    # - [r,g,b] or [r,g,b,a]
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


def _resolve_center_xy(position: Pos, W: int, H: int, lw: int, lh: int) -> Tuple[int, int]:
    """
    Resolve layer center (cx, cy) on a WxH canvas.

    - If position is a tuple (x,y), it's interpreted as the layer center in pixels.
    - If position is an anchor string, it is interpreted as "attach the layer
      to the corresponding canvas edge/corner", i.e. the layer's bounding box
      touches the canvas border (not its center on the border).
    """
    if isinstance(position, tuple):
        return int(position[0]), int(position[1])

    a = position.lower().strip()

    # helper: centers that make the layer touch the canvas borders (robust for odd sizes)
    half_w = lw / 2.0
    half_h = lh / 2.0

    left_cx = int(round(half_w))
    right_cx = int(round(W - half_w))

    top_cy = int(round(half_h))
    bottom_cy = int(round(H - half_h))

    mid_cx = W // 2
    mid_cy = H // 2

    if a == 'center':
        return mid_cx, mid_cy

    if a in ('top', 'center-top', 'top-center'):
        return mid_cx, top_cy
    if a in ('bottom', 'center-bottom', 'bottom-center'):
        return mid_cx, bottom_cy
    if a in ('left', 'center-left', 'left-center'):
        return left_cx, mid_cy
    if a in ('right', 'center-right', 'right-center'):
        return right_cx, mid_cy

    if a in ('top-left', 'left-top'):
        return left_cx, top_cy
    if a in ('top-right', 'right-top'):
        return right_cx, top_cy
    if a in ('bottom-left', 'left-bottom'):
        return left_cx, bottom_cy
    if a in ('bottom-right', 'right-bottom'):
        return right_cx, bottom_cy

    raise ValueError(f'Unknown position anchor: {position!r}')


def _place_on_canvas(canvas: Image.Image, layer_rgba: Image.Image, cx: int, cy: int) -> None:
    """
    Alpha-composite layer_rgba onto canvas, placing its *center* at (cx, cy).
    Handles out-of-bounds by cropping.
    """
    W, H = canvas.size
    lw, lh = layer_rgba.size

    # top-left placement so that center aligns
    x0 = int(round(cx - lw / 2))
    y0 = int(round(cy - lh / 2))
    x1 = x0 + lw
    y1 = y0 + lh

    # intersection with canvas
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)

    if ix1 <= ix0 or iy1 <= iy0:
        return  # fully outside

    # crop corresponding region from layer
    lx0 = ix0 - x0
    ly0 = iy0 - y0
    lx1 = lx0 + (ix1 - ix0)
    ly1 = ly0 + (iy1 - iy0)

    layer_crop = layer_rgba.crop((lx0, ly0, lx1, ly1))

    # alpha composite requires same size, so make a temp patch
    patch = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    patch.alpha_composite(layer_crop, (ix0, iy0))
    canvas.alpha_composite(patch)


def _apply_resize(img: Image.Image, resize: ResizeMode, canvas_w: int, canvas_h: int) -> Image.Image:
    if resize is None:
        return img

    w, h = img.size

    if isinstance(resize, str):
        if resize == 'fit':
            s = min(canvas_w / w, canvas_h / h)
        elif resize == 'cover':
            s = max(canvas_w / w, canvas_h / h)
        else:
            raise ValueError(f'invalid resize mode: {resize!r}')

        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        return img.resize((nw, nh), resample=Image.LANCZOS)

    # tuple mode
    tw, th = resize
    if tw is None and th is None:
        raise ValueError('resize cannot be (None, None)')

    if tw is not None and th is not None:
        nw = resolve_size_expr(tw, max_size=canvas_w)
        nh = resolve_size_expr(th, max_size=canvas_h)
    elif tw is not None:
        nw = resolve_size_expr(tw, max_size=canvas_w)
        s = float(nw) / float(w)
        nh = max(1, int(round(h * s)))
    else:
        nh = resolve_size_expr(th, max_size=canvas_h)
        s = float(nh) / float(h)
        nw = max(1, int(round(w * s)))

    nw = max(1, nw)
    nh = max(1, nh)
    return img.resize((nw, nh), resample=Image.LANCZOS)


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
        Declare a transform input for this layer driven by a crop-producing node.

        This method creates an additional wiring endpoint associated with the
        current layer (identified by ``idx``). When a compatible upstream node
        (e.g. ``SubjectCrop``) is connected to this sink, the layer geometry
        (position and size) is automatically derived from the crop metadata.

        Expected upstream metadata
        --------------------------
        The upstream node must provide a ``'crop'`` entry in its output dictionary
        with the following structure:

        ``crop`` : dict
            ``anchor_xy`` : (int, int)
                Absolute center coordinates of the crop bounding box in the
                original image reference frame.
            ``bbox_size`` : (int, int)
                Width and height of the crop bounding box in pixels.

        Runtime behavior
        ----------------
        If a transform input is present for this layer:

        - ``position`` is overridden using ``crop.anchor_xy``.
        The layer center on the canvas is set to these coordinates.
        - ``resize`` is overridden using ``crop.bbox_size``.
        The layer is resized to match the crop bounding box dimensions.

        If no transform input is connected, the layer uses the geometry
        specified explicitly via :meth:`image`.

        Returns
        -------
        AttachmentSink
            A sink bound to this node with ``input_id=f"transform:{idx}"``.
            It must be wired from a node producing compatible ``crop`` metadata.

        Notes
        -----
        - This mechanism enables declarative geometric reconstruction workflows,
        such as:
            1. Crop a region from an image.
            2. Refine it independently (e.g. via ``Img2Img``).
            3. Reinsert it into a stack using the original position and size.
        - The transform input overrides only spatial parameters (position and
          resize).
        - The transform mechanism assumes that the stack canvas shares the same
        coordinate reference as the image from which the crop was generated.
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
    - layering logos, UI elements, masks or overlays procedurally.

    Layer Model
    -----------
    Each upstream connection declared via :meth:`image` defines a *layer*.

    Layers are rendered deterministically with respect to ordering:

    - lowest ``idx`` → bottom layer
    - highest ``idx`` → top layer

    For each layer, the following pipeline is executed:

    1. Load upstream image and convert to ``RGBA``.
    2. Apply optional resizing (``resize``).
    3. Optionally soften the layer edges (``feather``).
    4. Resolve placement center (``position``) on the canvas.
    5. Alpha-composite the layer onto the canvas.

    Canvas
    ------
    Canvas geometry is configured via ``params.width`` and ``params.height``.
    The canvas is always constructed in ``RGBA`` mode and may be initialized as:

    - fully transparent (``background = null``),
    - a named PIL color (e.g. ``'white'``),
    - an explicit RGB or RGBA tuple.

    Final output color mode is controlled by ``params.out_mode``:

    - ``'RGBA'`` → preserve alpha channel,
    - ``'RGB'`` → drop alpha before saving.

    Resizing
    --------
    Each layer may be resized prior to placement using ``resize``:

    - ``None``:
        Preserve original size.
    - ``(W, H)``:
        Force exact dimensions in pixels.
    - ``(W, None)``:
        Set width to ``W`` and preserve aspect ratio.
    - ``(None, H)``:
        Set height to ``H`` and preserve aspect ratio.
    - ``'fit'``:
        Isotropic resize to the largest size that fits entirely inside
        the canvas without distortion.
    - ``'cover'``:
        Isotropic resize to the smallest size that fully covers the canvas
        without distortion. The layer may extend beyond canvas bounds and
        is clipped during compositing.

    Positioning
    -----------
    The ``position`` parameter controls layer placement on the canvas.

    - If a tuple ``(x, y)`` is provided, it represents the *center* of the
      layer in canvas pixel coordinates.
    - If a string anchor is provided, the layer is attached to the
      corresponding edge or corner such that its bounding box touches the
      canvas border(s), not such that its center lies on the border.

    Supported anchors include:

    - ``'center'``
    - ``'top'``, ``'bottom'``, ``'left'``, ``'right'``
    - ``'top-left'``, ``'top-right'``, ``'bottom-left'``, ``'bottom-right'``
    - ``'center-top'``, ``'center-bottom'``, ``'center-left'``, ``'center-right'``

    Anchor resolution depends on both canvas dimensions and the current
    layer dimensions (after resizing). This logic is implemented by
    ``_resolve_center_xy``.

    Transform Input (Crop-driven reconstruction)
    --------------------------------------------
    A layer may optionally declare an additional transform input via
    :meth:`ImageLayerAttachmentSink.transform`.

    If a compatible upstream node (e.g. ``SubjectCrop``) is wired into
    ``transform:{idx}``, the layer geometry is automatically overridden:

    - ``position`` is derived from ``crop.anchor_xy``.
    - ``resize`` is derived from ``crop.bbox_size``.

    This enables declarative reconstruction workflows:

    1. Crop a region from an image.
    2. Refine or upsample the cropped region independently.
    3. Reinsert it into a stack at the original spatial location.

    If no transform input is provided, the layer uses the geometry
    declared explicitly via :meth:`image`.

    Feathering
    ----------
    ``feather`` softens layer edges before compositing.

    Accepted formats:

    - ``int`` → feather width in pixels
    - ``"<number>px"`` → explicit pixel units
    - ``"<number>%"`` → percentage of the layer size

    The behavior depends on the alpha channel of the layer.

    **Fully opaque alpha (typical for ``crop_mode='bbox'``):**

    - The layer contains no transparency.
    - A synthetic alpha mask is generated.
    - The mask edge is softened using a feather ramp derived from
      the specified feather width.
    - When a percentage is used, the feather width is resolved
      independently for the horizontal and vertical axes.

    **Non-uniform alpha (e.g. segmentation masks or subject cutouts):**

    - The existing alpha channel is preserved.
    - The existing alpha channel is refined and softened using a small
      erosion followed by Gaussian blur.
    - Percentage values are resolved relative to the average of the
      layer width and height.

    A value of ``0`` disables feathering.

    Configuration
    -------------
    ``spec`` may be a dictionary or a path-like configuration file.

    Expected structure:

    ``params`` : dict
        ``width`` : int
            Output canvas width in pixels.
        ``height`` : int
            Output canvas height in pixels.
        ``background`` : str or (r, g, b) or (r, g, b, a) or null
            Initial canvas background.
        ``out_mode`` : {'RGBA', 'RGB'}
            Output color mode.

    Methods
    -------
    image(idx: int, position='center', resize=None, feather=0)
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
        ``metadata`` : str
            Path to JSON sidecar.

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
        position: Pos = 'center',
        resize: ResizeMode = None,
        feather: int | str = 0,
        corner_radius: Optional[int | str] = None,
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

        position : tuple[int, int] or str, optional
            Placement of the layer on the canvas.

            - If a tuple ``(x, y)`` is provided, it represents the **center**
            of the layer in canvas pixel coordinates.
            - If a string is provided, it is interpreted as an *edge/corner anchor*.
            The layer is positioned so that its bounding box touches the
            corresponding canvas border(s), not so that its center lies on the border.

            Supported anchors include:

            ``"center"``,
            ``"top"``, ``"bottom"``, ``"left"``, ``"right"``,
            ``"top-left"``, ``"top-right"``, ``"bottom-left"``, ``"bottom-right"``,
            ``"center-top"``, ``"center-bottom"``,
            ``"center-left"``, ``"center-right"``.

            Anchor resolution depends on both canvas dimensions and the
            current layer size (after resizing).

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

        Returns
        -------
        AttachmentSink
            A sink bound to this node with ``input_id=f"image:{idx}"``.
            The upstream node must provide an output image under the standard
            keys (``"image"`` or ``"path"``).

        Notes
        -----
        - The layer transformation is purely geometric (resize + placement).
        - No automatic color matching, lighting harmonization, or shadow
        synthesis is performed.
        - ``idx`` must be unique; attempting to reuse an index raises an error.
        """
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

        idx = int(idx)

        if idx in self._layers:
            raise ValueError(
                f'{self.id}: layer idx={idx} already declared. '
                'Each layer index must be unique.'
            )

        if corner_radius is not None:
            validate_size_expr(corner_radius)

        validate_size_expr(feather)

        self._layers[idx] = LayerSpec(
            idx=idx,
            position=position,
            resize=resize,
            feather=feather,
            corner_radius=corner_radius,
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

        # build canvas
        if cfg.background is None:
            canvas = Image.new('RGBA', (cfg.width, cfg.height), (0, 0, 0, 0))
        elif isinstance(cfg.background, str):
            canvas = Image.new('RGBA', (cfg.width, cfg.height), cfg.background)
        else:
            canvas = Image.new('RGBA', (cfg.width, cfg.height), cfg.background)

        if not self._layers:
            raise ValueError(
                f'{self.id}: no layers declared. Use src >> node.image(idx=...)')

        # composite in idx order
        for idx in sorted(self._layers.keys()):
            layer_spec = self._layers[idx]
            up = input.get(f'image:{idx}')
            if up is None:
                raise ValueError(
                    f'{self.id}: missing upstream for layer idx={idx} '
                    f"(expected wiring into input_id='image:{idx}')"
                )

            path = up.get('image') or up.get('path')
            if not path:
                raise ValueError(
                    f"{self.id}: upstream for idx={idx} must contain 'image' or 'path'"
                )

            with Image.open(path) as im:
                layer = im.convert('RGBA')

            # --- optional transform input ---
            tr = input.get(f'transform:{idx}')
            if tr is not None:
                # crop node output overrides anchor position and size information
                crop_meta = tr.get('crop')
                if crop_meta is None:
                    raise ValueError(
                        f'{self.id}: missing crop metadata on transform:{idx} '
                        "(expected upstream SubjectCrop to output key 'crop')"
                    )

                anchor_xy = crop_meta.get('anchor_xy')
                if anchor_xy is None:
                    raise ValueError(
                        f'{self.id}: missing crop.anchor_xy metadata on transform:{idx} '
                        "(expected upstream SubjectCrop to output key 'crop')"
                    )
                ax, ay = anchor_xy
                position = (int(ax), int(ay))

                bbox_size = crop_meta.get('bbox_size')
                if bbox_size is None:
                    raise ValueError(
                        f'{self.id}: missing crop.bbox_size metadata on transform:{idx} '
                        "(expected upstream SubjectCrop to output key 'crop')"
                    )
                bw, bh = bbox_size
                resize = (int(bw), int(bh))
            else:
                position = layer_spec.position
                resize = layer_spec.resize

            layer = _apply_resize(
                layer, resize, cfg.width, cfg.height
            )

            lw, lh = layer.size

            # feather (alpha handling)
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

            cx, cy = _resolve_center_xy(
                position,
                cfg.width,
                cfg.height,
                layer.width,
                layer.height,
            )

            _place_on_canvas(canvas, layer, cx=cx, cy=cy)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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
                'width': cfg.width,
                'height': cfg.height,
                'background': cfg.background,
                'out_mode': cfg.out_mode,
                'layers': [
                    {
                        'idx': ls.idx,
                        'position': ls.position,
                        'resize': ls.resize,
                        'feather': ls.feather,
                        'corner_radius': ls.corner_radius,
                    }
                    for ls in (self._layers[i] for i in sorted(self._layers.keys()))
                ],
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)
        return out
