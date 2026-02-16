from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFilter

from stability.core.paths import make_node_output_path
from stability.dag import AttachmentSink, NodeRef
from stability.nodes.common.config_resolve import resolve_spec
from stability.nodes.common.io import write_json_sidecar

Pos = Union[Tuple[int, int], str]
BlendMode = Literal['alpha']
ResizeMode = Tuple[Optional[int], Optional[int]] \
    | Literal['fit', 'cover'] \
    | None


@dataclass(frozen=True)
class LayerSpec:
    idx: int
    position: Pos = 'center'   # tuple -> center coords; str -> anchor
    resize: ResizeMode = None
    blend: BlendMode = 'alpha'
    feather: int = 0           # px gaussian blur on alpha


@dataclass
class Config:
    width: int
    height: int
    background: Optional[Union[str, Tuple[int, int, int, int]]]
    out_mode: Literal['RGBA', 'RGB']


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
    patch.paste(layer_crop, (ix0, iy0), layer_crop)
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
        nw = int(tw)
        nh = int(th)
    elif tw is not None:
        s = float(tw) / float(w)
        nw = int(tw)
        nh = max(1, int(round(h * s)))
    else:
        s = float(th) / float(h)
        nh = int(th)
        nw = max(1, int(round(w * s)))

    nw = max(1, nw)
    nh = max(1, nh)
    return img.resize((nw, nh), resample=Image.LANCZOS)


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
        - The transform input overrides only spatial parameters
        (position and resize). Blending mode and feathering remain unchanged.
        - The transform mechanism assumes that the stack canvas shares the same
        coordinate reference as the image from which the crop was generated.
        """

        return AttachmentSink(
            id=f'imagestack_transform:{self.target.id}:{self.idx}',
            target=self.target,
            input_id=f'transform:{self.idx}',
        )


@dataclass
class ImageStack(NodeRef):
    """
    Layer-based image compositor node.

    ``ImageStack`` is a deterministic, geometry-driven compositing node that
    collects multiple upstream images via typed sinks created by :meth:`image`
    and composites them in ascending ``idx`` order onto a configurable canvas.

    The node is designed for DAG-native compositing workflows such as:

    - placing subject cutouts onto new backgrounds,
    - reconstructing refined regions back into their original frame,
    - assembling multiple extracted elements into a single image,
    - layering logos, UI elements, masks or overlays procedurally.

    Layer Model
    -----------
    Each upstream connection declared via :meth:`image` defines a *layer*.

    Layers are rendered deterministically:

    - lowest ``idx`` → bottom layer
    - highest ``idx`` → top layer

    For each layer, the following pipeline is executed:

    1. Load upstream image and convert to ``RGBA``.
    2. Apply optional resizing (``resize``).
    3. Optionally feather the alpha channel (``feather``).
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

    - If a tuple ``(x, y)`` is provided, it is interpreted as the *center*
    of the layer in canvas pixel coordinates.
    - If a string anchor is provided, the layer is attached to the corresponding
    edge or corner such that its bounding box touches the canvas border(s),
    not such that its center lies on the border.

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

    Blending
    --------
    Currently supported blend modes:

    - ``'alpha'``:
        Standard alpha compositing using the layer's alpha channel.

    The compositor is purely geometric and does not perform:

    - color matching,
    - lighting harmonization,
    - shadow synthesis,
    - edge relighting.

    Feathering
    ----------
    ``feather`` applies a Gaussian blur to the layer’s alpha channel
    before compositing. This is particularly useful for:

    - segmentation cutouts,
    - hair or soft contours,
    - reintegration of refined subregions.

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
    image(idx: int, position='center', resize=None, blend='alpha', feather=0)
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
    - Layers are composited deterministically.
    - Layers extending beyond canvas bounds are safely clipped.
    - The node is deterministic and contains no stochastic components.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)
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
        blend: BlendMode = 'alpha',
        feather: int = 0,
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

        resize : tuple[int | None, int | None] or {"fit", "cover"} or None, optional
            Optional resizing applied before placement.

            - ``None``:
                No resizing; original image size is preserved.
            - ``(W, H)``:
                Force exact size in pixels.
            - ``(W, None)``:
                Set width to ``W`` and preserve aspect ratio.
            - ``(None, H)``:
                Set height to ``H`` and preserve aspect ratio.
            - ``"fit"``:
                Resize isotropically to the largest size that fits entirely
                inside the canvas without distortion.
            - ``"cover"``:
                Resize isotropically to the smallest size that fully covers
                the canvas without distortion. The layer may extend beyond
                canvas bounds and will be cropped during compositing.

        blend : {"alpha"}, optional
            Blending mode used during compositing.

            Currently supported:
            - ``"alpha"`` → standard alpha compositing using the image's
            alpha channel (or opaque if absent).

            Default: ``"alpha"``.

        feather : int, optional
            Gaussian blur radius (in pixels) applied to the layer's alpha
            channel before compositing.

            Useful for:
            - softening segmentation edges,
            - blending cutouts into backgrounds,
            - reducing visible halos.

            A value of ``0`` disables feathering.

            Default: ``0``.

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
                    if not isinstance(w_target, int) or w_target <= 0:
                        raise ValueError(
                            f'{self.id}: resize width must be positive int or None, '
                            f'got {w_target!r}'
                        )

                if h_target is not None:
                    if not isinstance(h_target, int) or h_target <= 0:
                        raise ValueError(
                            f'{self.id}: resize height must be positive int or None, '
                            f'got {h_target!r}'
                        )

            else:
                raise TypeError(
                    f'{self.id}: resize must be None, tuple or str, got {type(resize)}'
                )

        if blend != 'alpha':
            raise ValueError(f'{self.id}: unsupported blend={blend!r}')

        idx = int(idx)

        if idx in self._layers:
            raise ValueError(
                f'{self.id}: layer idx={idx} already declared. '
                'Each layer index must be unique.'
            )

        self._layers[idx] = LayerSpec(
            idx=idx,
            position=position,
            resize=resize,
            blend=blend,
            feather=int(feather),
        )

        return ImageLayerAttachmentSink(
            idx=idx,
            id=f'imagestack_image:{self.id}:{idx}',
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
                    f"{self.id}: upstream for idx={idx} must contain 'image' or 'path'")

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

            # feather (alpha handling)
            if layer_spec.feather and layer_spec.feather > 0:
                rad = int(layer_spec.feather)
                r, g, b, a = layer.split()

                # If alpha is fully opaque (typical for crop_mode='bbox'),
                # blurring it does nothing. Build a soft-edge alpha mask instead.
                a_min, a_max = a.getextrema()
                if a_min == 255 and a_max == 255:
                    lw, lh = layer.size

                    # Start from transparent and paint an inset fully-opaque rect,
                    # then blur -> produces a soft falloff near the layer borders.
                    m = Image.new('L', (lw, lh), 0)
                    draw = ImageDraw.Draw(m)

                    inset = rad
                    x0, y0 = inset, inset
                    x1, y1 = lw - inset - 1, lh - inset - 1

                    # Corner rounding radius (in pixels).
                    # Heuristic: tie it to feather radius, but clamp to avoid over-rounding.
                    corner = max(1, int(round(rad * 3)))
                    corner = min(corner, max(1, (min(lw, lh) // 2) - 1))

                    if (x1 - x0) > 1 and (y1 - y0) > 1:
                        # Prefer rounded corners so the gradient doesn't look "boxy".
                        if hasattr(draw, 'rounded_rectangle'):
                            draw.rounded_rectangle(
                                [x0, y0, x1, y1], radius=corner, fill=255)
                        else:
                            # Pillow too old: fallback to normal rectangle.
                            draw.rectangle([x0, y0, x1, y1], fill=255)
                    else:
                        # fallback: very small layer, just fill
                        draw.rectangle([0, 0, lw - 1, lh - 1], fill=255)

                    a = m.filter(ImageFilter.GaussianBlur(radius=rad))
                else:
                    # Normal case: soften existing cutout edges
                    a = a.filter(ImageFilter.GaussianBlur(radius=rad))

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

        out_path = make_node_output_path(
            out_dir=out_dir, node_id=self.id, ext='png')

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
                        'blend': ls.blend,
                        'feather': ls.feather,
                    }
                    for ls in (self._layers[i] for i in sorted(self._layers.keys()))
                ],
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)
        return out
