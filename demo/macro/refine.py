from __future__ import annotations

from typing import Literal, Optional

from morphalo.dag import NodeGroup
from morphalo.nodes import FaceIdEmbedImage, Img2Img, Inpaint, Tap
from morphalo.nodes.common.config_resolve import SpecInput
from morphalo.nodes.preprocess import (BoxCrop, FaceCrop, ImageStack,
                                       ImgAuxMap, ResizeImage, SubjectCrop)
from morphalo.nodes.preprocess.utils import SizeExpr, resolve_size_expr
from morphalo.nodes.preprocess.utils.geometry import \
    expand_clip_bbox_by_size_expr

FineRegion = Literal[
    'face',
    'eyes',
    'left-eye',
    'right-eye',
    'eyebrows',
    'left-eyebrow',
    'right-eyebrow',
    'head',
]

SegmentAxis = Literal['x', 'y']
SegmentOrder = Literal['forward', 'reverse']


def _validate_controlnet_lists(
    *,
    name: str,
    controlnet_models: list[str] | None = None,
    controlnet_conditioning_scales: list[float] | None = None,
    aux_map_specs: list[SpecInput] | None = None,
) -> None:
    """
    Validate parallel ControlNet argument lists.
    """
    used = any(
        value is not None
        for value in (
            controlnet_models,
            controlnet_conditioning_scales,
            aux_map_specs,
        )
    )

    if not used:
        return

    if (
        controlnet_models is None
        or controlnet_conditioning_scales is None
        or aux_map_specs is None
    ):
        raise ValueError(
            f'{name}: controlnet_models, controlnet_conditioning_scales '
            'and aux_map_specs must all be set or all be None'
        )
    if not (
        len(controlnet_models)
        == len(controlnet_conditioning_scales)
        == len(aux_map_specs)
    ):
        raise ValueError(
            f'{name}: controlnet_models, controlnet_conditioning_scales '
            'and aux_map_specs must have the same length'
        )


SUBJECT_MODEL_SPEC = {
    'yolo_model': 'yolov8n.pt',
    'segment_model': 'facebook/sapiens2-seg-0.4b',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
    'device': 'cuda',
    'dtype': 'bfloat16',
}

FACE_MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}


def refine_face_group(
    name: str,
    *,
    refine_spec: SpecInput,
    stack_spec: SpecInput = {},
    face_id_scale: float,
    face_id_clip_strength: float = 1.0,
    layer_feather: int | str = 30,
    layer_corner_radius: int | str = 50,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines a head/face region and overlays it back.

    This group exposes two ports (implemented as entry nodes):

    - 'in_image' (Tap): base image that will be used both for:
        1) background (stack layer 0)
        2) head crop (SubjectCrop)

    - 'in_face_image' (FaceIdEmbedImage): reference face image used to build
      the FaceID embedding. The node supports both:
        - path-based input (if wired/produced upstream)
        - direct path configuration (depending on FaceIdEmbedImage behavior)

    - 'in_prompt' (Tap)
        Prompt payload to feed into the internal Img2Img prompt sink.

    The group's single output node is the ImageStack ('merge'), so the group can
    be wired as a unit via `group >> downstream`.

    Parameters
    ----------
    name : str
        NodeGroup name (scope prefix).
    refine_spec : SpecInput
        Spec for Img2Img.
    stack_spec : SpecInput
        Spec for ImageStack (canvas size, background, out_mode, etc.).
    face_id_scale : float
        IP-Adapter FaceID scale.
    face_id_clip_strength : float, optional
        FaceID CLIP conditioning strength.
    layer_feather : int | str, optional
        Feather applied when compositing the refined layer.
    layer_corner_radius : int | str, optional
        Corner radius applied to the refined layer mask.

    Returns
    -------
    NodeGroup
        The constructed group.

    Notes
    -----
    This group assumes square head crop (target='head').

    External wiring example:

        img_node  >> group('in_image')
        face_node >> group('in_face_image')
        prompt    >> g('in_prompt')
        group >> downstream
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports (entry nodes)
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        face_embed = FaceIdEmbedImage(name='in_face_image')

        # -------------------
        # Internal nodes
        # -------------------
        crop_face = SubjectCrop(
            name='crop_face',
            spec={
                'model': SUBJECT_MODEL_SPEC,
                'params': {
                    'target': 'head',
                    'mode': 'default',
                    'crop_mode': 'bbox',
                    'box_margin': 0.12,
                    'expansion': 1.2,
                },
            },
        )

        refine_face = Img2Img(
            name='refine_face',
            spec=refine_spec,
        )

        face_id_sink = refine_face.face_id.add(
            model_id='h94/IP-Adapter-FaceID',
            weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
            # weight_name='ip-adapter-faceid_sdxl.bin',
            scale=face_id_scale,
            clip_strength=face_id_clip_strength,
            key='face_id',
        )

        stack = ImageStack(
            name='out',
            spec=stack_spec,
        )

        # -------------------
        # Wiring
        # -------------------

        # 1) Base image as background
        tap_image >> stack.image(0)

        # 2) Crop head from base image
        tap_image >> crop_face

        # 3) Refine cropped head
        crop_face >> refine_face

        # 3b) FaceID embedding into adapter sink
        face_embed >> face_id_sink

        # 4) Overlay refined head using crop metadata
        layer1 = stack.image(
            1,
            position='center',
            feather=layer_feather,
            corner_radius=layer_corner_radius,
        )

        tap_prompt >> refine_face.prompt()
        refine_face >> layer1

        # Use crop metadata (anchor + bbox size) to place and scale the layer
        crop_face >> layer1.transform()

        # Expose ports by registering the entry nodes
        g.register_ports(tap_image, face_embed, tap_prompt)

    return g


def apply_face_id_head_group(
    name: str,
    *,
    img2img_spec: SpecInput,
    face_id_scale: float,
    face_id_clip_strength: float = 0.5,
    head_expansion: float = 2.0,
    mask_dilate_radius: int = 10,
    mask_close_radius: int = 10,
    mask_smoothing_radius: int = 30,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that applies FaceID to an image, localized to the head.

    This macro implements the pattern:

    - image -> head mask (SubjectCrop)
    - face image -> FaceIdEmbedImage -> FaceID adapter sink
    - image -> Img2Img (base)
    - head mask -> face_id.mask() (localize FaceID effect to head region)
    - prompt -> Img2Img.prompt()

    Ports
    -----
    The group exposes three ports via registered entry nodes:

    - 'in_image' (Tap)
        Base image to refine.

    - 'in_face_image' (FaceIdEmbedImage)
        Reference face image used to compute FaceID embedding. This node is kept
        as a first-class node because it naturally materializes its output and
        can also accept face image upstream (or path-based input, depending on
        its implementation).

    - 'in_prompt' (Tap)
        Prompt payload to feed into the internal Img2Img prompt sink.

    Output
    ------
    The group output node is the internal Img2Img node ('out'), so:

        group >> downstream

    is supported.

    Parameters
    ----------
    name : str
        NodeGroup name (scope prefix).
    img2img_spec : SpecInput
        Spec for the internal Img2Img node.
    face_id_scale : float
        FaceID adapter scale (higher anchors identity more strongly).
    face_id_clip_strength : float, optional
        FaceID CLIP conditioning strength.
    head_expansion : float, optional
        Expansion factor for the head mask crop. Higher values include more hair/helmet.
    mask_dilate_radius : int, optional
        Morphological dilation applied to the head mask.
    mask_close_radius : int, optional
        Morphological closing applied to the head mask.
    mask_smoothing_radius : int, optional
        Mask smoothing radius.

    Returns
    -------
    NodeGroup
        The constructed group.

    Notes
    -----
    External wiring example:

        img_node  >> g('in_image')
        face_node >> g('in_face_image')
        prompt    >> g('in_prompt')
        g >> downstream
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports (entry nodes)
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt')

        # FaceIdEmbedImage is a real node (it materializes embedding by itself).
        face_embed = FaceIdEmbedImage(name='in_face_image')

        # -------------------
        # Internal nodes
        # -------------------
        head_mask = SubjectCrop(
            name='head_mask',
            spec={
                'model': SUBJECT_MODEL_SPEC,
                'params': {
                    'target': 'head',
                    'mode': 'mask',
                    'dilate_radius': mask_dilate_radius,
                    'close_radius': mask_close_radius,
                    'smoothing_radius': mask_smoothing_radius,
                    'expansion': head_expansion,
                },
            },
        )

        out = Img2Img(
            name='out',
            spec=img2img_spec,
        )

        face_id_sink = out.face_id.add(
            model_id='h94/IP-Adapter-FaceID',
            weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
            scale=face_id_scale,
            clip_strength=face_id_clip_strength,
            key='face_id',
        )

        # -------------------
        # Wiring
        # -------------------

        # Base image to Img2Img
        tap_image >> out

        # Head mask from the same base image
        tap_image >> head_mask

        # Prompt into Img2Img prompt sink
        tap_prompt >> out.prompt()

        # Face embedding into FaceID sink
        face_embed >> face_id_sink

        # Localize FaceID effect to head region
        head_mask >> face_id_sink.mask()

        # -------------------
        # Register selectable ports
        # -------------------
        g.register_ports(tap_image, face_embed, tap_prompt)

    return g


def fine_face_details_group(
    name: str,
    *,
    inpaint_spec: SpecInput,
    face_id_scale: Optional[float] = None,
    face_id_clip_strength: float = 1.0,
    region: FineRegion = 'face',
    mask_dilate_radius: int = -5,
    mask_close_radius: int = 10,
    mask_smoothing_radius: int = 10,
    mask_expansion: float | None = None,
) -> NodeGroup:
    """
    Create a reusable NodeGroup for subtle local detail consolidation.

    This macro implements the pattern:

    - image -> region mask (SubjectCrop target=<region>, mode='mask')
    - image -> Inpaint (low strength)
    - prompt (details) -> Inpaint.prompt()
    - (optional) face image -> FaceIdEmbedImage -> FaceID adapter sink

    The intent is to perform small local edits (eye color, eyebrows, micro skin
    details). When ``region='face'``, identity is stabilized via FaceID anchoring.
    For eye-only regions, FaceID is typically unnecessary and may over-constrain
    edits.

    IMPORTANT
    ---------
    When ``region='face'``, this macro assumes the face already present in the
    input image is consistent with the FaceID reference. If the input face
    differs significantly, the result may become unstable (it is not a face swap).

    Ports
    -----
    - 'in_image' (Tap)
        The base image to be edited.
    - 'in_prompt' (Tap)
        Prompt payload used for local micro-details.
    - 'in_face_image' (FaceIdEmbedImage), only when ``region='face'``
        Reference image used to compute FaceID embedding.

    Parameters
    ----------
    name : str
        NodeGroup name (scope prefix). This is used to namespace internal node ids.

    inpaint_spec : SpecInput
        Spec for the internal :class:`Inpaint` node. You typically set low
        strength / conservative denoise here for subtle edits.

    face_id_scale : float or None, optional
        FaceID adapter scale used to stabilize identity. This is only used when
        ``region='face'`` (i.e. when FaceID wiring is enabled).

        - If ``region='face'``, this parameter is **required** (must not be None).
        - If ``region!='face'``, this parameter is ignored.

        Higher values enforce stronger identity preservation, but may also reduce
        the model's freedom to apply local edits.

    face_id_clip_strength : float, optional
        CLIP conditioning strength for the FaceID adapter. Only used when
        ``region='face'``. Typical useful range for subtle consolidation is
        0.0–0.2; higher values may over-constrain the edit. Default: 1.0.

    region : {'face', 'eyes', 'left-eye', 'right-eye', 'head'}, optional
        Region mask used to constrain inpainting via :class:`SubjectCrop`.

        - ``'face'``: full face region. Enables FaceID anchoring (requires
          ``face_id_scale``) and exposes the ``in_face_image`` port.
        - ``'eyes'``: both eyes combined. FaceID anchoring is disabled.
        - ``'left-eye'`` / ``'right-eye'``: single-eye masking. Recommended for
          heterochromia (run two passes to avoid color harmonization). FaceID
          anchoring is disabled.
        - ``'head'``: head region (hair-friendly) if supported by your
          :class:`SubjectCrop` configuration. FaceID anchoring is disabled.

        Default: ``'face'``.

    mask_dilate_radius : int, optional
        Dilation radius (pixels) applied to the generated mask before it is used
        for inpainting. Positive values expand the masked region; negative values
        shrink it. Default: -5.

    mask_close_radius : int, optional
        Morphological closing radius (pixels) applied to the mask. This helps
        fill small holes and connect thin gaps. Default: 10.

    mask_smoothing_radius : int, optional
        Gaussian smoothing radius (pixels) applied to the mask edges. This
        produces softer transitions and reduces inpaint seams. Default: 10.

    mask_expansion : float or None, optional
        Optional expansion factor forwarded to :class:`SubjectCrop` for targets
        that benefit from a slightly looser crop/mask (commonly ``'eyes'`` and
        ``'head'``). When None, the parameter is omitted. Default: None.

    Returns
    -------
    NodeGroup
        The constructed group.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports (entry nodes)
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt')

        # Create FaceID port only when stabilizing the whole face.
        use_face_id = (region == 'face')
        if use_face_id and face_id_scale is None:
            raise ValueError(
                f'{name}: face_id_scale is required when region="face"'
            )

        # -------------------
        # Internal nodes
        # -------------------
        mask_params = {
            'target': region,
            'mode': 'mask',
            'dilate_radius': mask_dilate_radius,
            'close_radius': mask_close_radius,
            'smoothing_radius': mask_smoothing_radius,
        }
        if mask_expansion is not None:
            mask_params['expansion'] = mask_expansion

        crop_node_cls = SubjectCrop if region == 'head' else FaceCrop
        model_spec = SUBJECT_MODEL_SPEC if region == 'head' else FACE_MODEL_SPEC

        region_mask = crop_node_cls(
            name='region_mask',
            spec={
                'model': model_spec,
                'params': mask_params,
            },
        )

        out = Inpaint(
            name='out',
            spec=inpaint_spec,
        )

        # -------------------
        # Wiring
        # -------------------

        # Base image to both mask generation and inpaint input
        tap_image >> region_mask
        tap_image >> out

        # Mask constrains inpaint to the face region
        region_mask >> out.mask()

        # Prompt drives the micro-detail edit
        tap_prompt >> out.prompt()

        # -------------------
        # Register selectable ports
        # -------------------
        if use_face_id:
            face_embed = FaceIdEmbedImage(name='in_face_image')
            face_id_sink = out.face_id.add(
                model_id='h94/IP-Adapter-FaceID',
                weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
                scale=float(face_id_scale),
                clip_strength=face_id_clip_strength,
                key='face_id',
            )
            # Face embedding anchors identity during inpaint
            face_embed >> face_id_sink

            g.register_ports(tap_image, face_embed, tap_prompt)
        else:
            g.register_ports(tap_image, tap_prompt)

    return g


HandSide = Literal['left', 'right']


def refine_hand_group(
    name: str,
    *,
    refine_spec: SpecInput,
    stack_spec: SpecInput = {},
    hand: HandSide = 'right',
    default_prompt: str = 'well formed hand, anatomically correct fingers, natural hand pose',
    hand_crop_mode: str = 'bbox[1:1]',
    hand_expansion: float = 1.8,
    hand_box_margin: float = 0.20,
    layer_feather: int | str = 25,
    layer_corner_radius: int | str = 40,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines one hand and overlays it back.

    The group crops a single subject hand, refines the crop with ``Img2Img``,
    and composites the refined crop back over the original image.

    The selected hand follows ``SubjectCrop`` semantics: ``left`` and ``right``
    refer to the subject perspective, not the viewer perspective.

    Ports
    -----
    in_image
        Base image used both as background and as source for the hand crop.

    in_prompt
        Optional prompt payload for the internal ``Img2Img`` node. If omitted,
        the macro falls back to ``default_prompt``.

    Output
    ------
    The group output corresponds to the internal ``ImageStack`` node ``out``.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for the internal ``Img2Img`` node.

    stack_spec : SpecInput, optional
        Spec for ``ImageStack``. Usually defines canvas size, background, and
        output mode.

    hand : {'left', 'right'}, optional
        Hand to refine, using subject-perspective semantics.

    default_prompt : str, optional
        Fallback prompt used when no external ``in_prompt`` is wired.

    hand_crop_mode : str, optional
        Crop mode forwarded to ``SubjectCrop``. A ratio crop such as
        ``'bbox[1:1]'`` is useful for giving the model enough local context.

    hand_expansion : float, optional
        Expansion factor for the hand crop.

    hand_box_margin : float, optional
        Additional margin around the detected hand region.

    layer_feather : int or str, optional
        Feather applied when compositing the refined hand crop.

    layer_corner_radius : int or str, optional
        Corner radius applied to the refined layer mask.

    Returns
    -------
    NodeGroup
        The constructed group.
    """

    if hand not in ('left', 'right'):
        raise ValueError(
            f'{name}: invalid hand={hand!r}. Expected "left" or "right".'
        )

    target = f'{hand}-hand'

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)

        # -------------------
        # Crop one hand
        # -------------------
        crop_hand = SubjectCrop(
            name='crop_hand',
            spec={
                'model': SUBJECT_MODEL_SPEC,
                'params': {
                    'target': target,
                    'mode': 'default',
                    'crop_mode': hand_crop_mode,
                    'box_margin': hand_box_margin,
                    'expansion': hand_expansion,
                },
            },
        )

        # -------------------
        # Refine hand crop
        # -------------------
        refine_hand = Img2Img(
            name='refine_hand',
            spec=[
                refine_spec,
                {
                    'prompt': default_prompt,
                },
            ],
        )

        # -------------------
        # Composite back
        # -------------------
        stack = ImageStack(
            name='out',
            spec=stack_spec,
        )

        # Base image as background.
        tap_image >> stack.image(0)

        # Crop selected hand.
        tap_image >> crop_hand

        # Refine cropped hand.
        crop_hand >> refine_hand
        tap_prompt >> refine_hand.prompt()

        # Overlay refined crop using crop metadata.
        layer = stack.image(
            1,
            position='center',
            feather=layer_feather,
            corner_radius=layer_corner_radius,
        )

        refine_hand >> layer
        crop_hand >> layer.transform()

        g.register_ports(
            tap_image,
            tap_prompt,
        )

    return g


def _sliding_window_bboxes_xyxy(
    *,
    image_size: tuple[int, int],
    grid: tuple[int, int],
    window_fraction: tuple[float, float] = (0.5, 0.5),
) -> list[tuple[int, int, int, int]]:
    """
    Build overlapping sliding-window crop boxes.

    Parameters
    ----------
    image_size : tuple[int, int]
        Source image size as ``(width, height)``.

    grid : tuple[int, int]
        Number of sliding-window positions as ``(cols, rows)``.

    window_fraction : tuple[float, float], optional
        Window size expressed as a fraction of the image size, using
        ``(width_fraction, height_fraction)``.

        For example, with ``grid=(3, 3)`` and
        ``window_fraction=(0.5, 0.5)``, the generated windows use starts:

        - x: ``0``, ``0.25 * width``, ``0.5 * width``
        - y: ``0``, ``0.25 * height``, ``0.5 * height``

        This covers the whole image with overlapping half-image windows.

    Returns
    -------
    list of tuple[int, int, int, int]
        Crop boxes in ``xyxy`` format.
    """
    width, height = image_size
    cols, rows = grid
    win_fx, win_fy = window_fraction

    if cols < 1 or rows < 1:
        raise ValueError('grid values must be >= 1')

    if not 0 < win_fx <= 1:
        raise ValueError('window_fraction[0] must be in the range (0, 1]')

    if not 0 < win_fy <= 1:
        raise ValueError('window_fraction[1] must be in the range (0, 1]')

    win_w = max(1, round(width * win_fx))
    win_h = max(1, round(height * win_fy))

    max_x = width - win_w
    max_y = height - win_h

    if max_x < 0 or max_y < 0:
        raise ValueError('window size cannot exceed image size')

    x_positions = [
        round(max_x * i / max(cols - 1, 1))
        for i in range(cols)
    ]

    y_positions = [
        round(max_y * i / max(rows - 1, 1))
        for i in range(rows)
    ]

    return [
        (x, y, x + win_w, y + win_h)
        for y in y_positions
        for x in x_positions
    ]


def _band_segment_bboxes_xyxy(
    *,
    image_size: tuple[int, int],
    axis: SegmentAxis = 'y',
    start: SizeExpr = 0,
    end: SizeExpr | None = None,
    segments: int = 4,
    overlap: SizeExpr = 0,
    order: SegmentOrder = 'forward',
) -> list[tuple[int, int, int, int]]:
    """
    Build full-width or full-height segment boxes along one image axis.

    ``axis='y'`` creates horizontal bands. ``axis='x'`` creates vertical bands.
    ``start``, ``end`` and ``overlap`` accept the shared size-expression syntax:
    integer pixels, ``"120px"``, or percentages such as ``"25%"`` resolved
    against the selected axis length.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError('image_size values must be > 0')

    if axis not in ('x', 'y'):
        raise ValueError("axis must be 'x' or 'y'")

    if order not in ('forward', 'reverse'):
        raise ValueError("order must be 'forward' or 'reverse'")

    if segments < 1:
        raise ValueError('segments must be >= 1')

    axis_size = height if axis == 'y' else width
    start_px = resolve_size_expr(
        start,
        reference=axis_size,
        min_size=0,
        allow_unitless=True,
    )
    end_px = (
        axis_size
        if end is None
        else resolve_size_expr(
            end,
            reference=axis_size,
            min_size=0,
            allow_unitless=True,
        )
    )
    overlap_px = resolve_size_expr(
        overlap,
        reference=axis_size,
        min_size=0,
        allow_unitless=True,
    )

    start_px = max(0, min(axis_size, start_px))
    end_px = max(0, min(axis_size, end_px))
    if end_px <= start_px:
        raise ValueError('segment end must be greater than segment start')

    span = end_px - start_px
    boxes: list[tuple[int, int, int, int]] = []

    for idx in range(segments):
        band_start = start_px + round(span * idx / segments)
        band_end = start_px + round(span * (idx + 1) / segments)

        band_start = max(start_px, band_start - overlap_px)
        band_end = min(end_px, band_end + overlap_px)

        if band_end <= band_start:
            continue

        if axis == 'y':
            boxes.append((0, band_start, width, band_end))
        else:
            boxes.append((band_start, 0, band_end, height))

    if order == 'reverse':
        boxes.reverse()

    return boxes


def refine_sliding_tiles_group(
    name: str,
    *,
    refine_spec: SpecInput,
    image_size: tuple[int, int],
    grid: tuple[int, int] = (3, 3),
    window_fraction: tuple[float, float] = (0.5, 0.5),
    adapter_weight_names: list[str] | None = None,
    adapter_scales: list[float | dict] | None = None,
    layer_feather: int | str = 40,
    layer_corner_radius: int | str = 60,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines an image through overlapping tiles.

    The upstream image is first resized to ``image_size``. The resized image is
    then divided into overlapping rectangular regions, each region is refined
    with ``Img2Img`` using two IP-Adapter references, and the refined crop is
    composited back onto the progressively updated image.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for every internal ``Img2Img`` refinement node.

    image_size : tuple[int, int]
        Working image size as ``(width, height)``.

        The macro resizes the upstream image to this exact size before generating
        tile boxes. All crop coordinates, crop metadata, and internal stack
        canvases use this same coordinate space.

    grid : tuple[int, int], optional
        Number of tile positions as ``(cols, rows)``.

    window_fraction : tuple[float, float], optional
        Tile size as a fraction of ``image_size``, expressed as
        ``(width_fraction, height_fraction)``.

    adapter_weight_names : list[str] or None, optional
        IP-Adapter weight names. When ``None``, no style adapters are added.

    adapter_scales : list[float | dict] or None, optional
        IP-Adapter scales aligned with ``adapter_weight_names``.

    layer_feather : int or str, optional
        Feather applied when compositing every refined tile.

    layer_corner_radius : int or str, optional
        Corner radius applied to every refined tile mask.

    Returns
    -------
    NodeGroup
        The constructed tiled-refinement group.

    Notes
    -----
    This macro intentionally owns its working canvas size. The upstream image may
    have any size, but it is resized to ``image_size`` before the first tile is
    cropped.

    When ``image_size`` has a different aspect ratio from the upstream image,
    the initial resize may stretch the image. This is intentional: ``image_size``
    defines the exact coordinate space used by the tiled refinement pass.

    Tiles are processed sequentially. Each tile is cropped from the progressively
    updated image produced by the previous tile.
    """
    stack_spec = {
        'params': {
            'width': int(image_size[0]),
            'height': int(image_size[1]),
        },
    }

    bboxes = _sliding_window_bboxes_xyxy(
        image_size=image_size,
        grid=grid,
        window_fraction=window_fraction,
    )

    if not bboxes:
        raise ValueError(f'{name}: no tile boxes were generated')

    if (adapter_weight_names is None) != (adapter_scales is None):
        raise ValueError(
            f'{name}: adapter_weight_names and adapter_scales must both be set '
            'or both be None'
        )
    if adapter_weight_names is not None and (
        len(adapter_weight_names) != len(adapter_scales)
    ):
        raise ValueError(
            f'{name}: adapter_weight_names and adapter_scales must have the '
            'same length'
        )

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_styles = [
            Tap(name=f'style_{idx + 1}')
            for idx in range(len(adapter_weight_names or []))
        ]

        resized_image = ResizeImage(
            name='resize_input',
            spec={
                'params': {
                    'size': [
                        int(image_size[0]),
                        int(image_size[1]),
                    ],
                },
            },
        )

        tap_image >> resized_image
        previous_image = resized_image

        for idx, bbox in enumerate(bboxes):
            is_last = idx == len(bboxes) - 1

            crop_tile = BoxCrop(
                name=f'crop_{idx:02d}',
                spec={
                    'params': {
                        'bbox_format': 'xyxy',
                        'bbox': list(bbox),
                    },
                },
            )

            refine_tile = Img2Img(
                name=f'refine_{idx:02d}',
                spec=refine_spec,
            )

            resize_tile = ResizeImage(
                name=f'resize_{idx:02d}',
                spec={},
            )

            adapter_sinks = []
            for style_idx, (weight_name, scale) in enumerate(zip(
                adapter_weight_names or [],
                adapter_scales or [],
            )):
                adapter_sinks.append(refine_tile.ip_adapter.add(
                    'h94/IP-Adapter',
                    subfolder='sdxl_models',
                    weight_name=weight_name,
                    scale=scale,
                    key=f'style_{style_idx + 1}',
                ))

            stack_name = 'out' if is_last else f'stack_{idx:02d}'

            stack = ImageStack(
                name=stack_name,
                spec=stack_spec,
            )

            # 1) Use the progressively refined image as the current background.
            previous_image >> stack.image(0)

            # 2) Crop the current tile from the progressively refined image.
            previous_image >> crop_tile

            # 3) Refine the tile.
            crop_tile >> refine_tile
            tap_prompt >> refine_tile.prompt()
            for tap_style, adapter_sink in zip(tap_styles, adapter_sinks):
                tap_style >> adapter_sink

            # 4) Restore the refined tile to the original crop size before
            #    compositing. Img2Img may emit a different resolution, while
            #    BoxCrop anchor metadata is expressed in crop-local pixels.
            refine_tile >> resize_tile
            crop_tile >> resize_tile.transform()

            # 5) Composite the resized refined tile back using crop metadata.
            layer = stack.image(
                1,
                position='center',
                feather=layer_feather,
                corner_radius=layer_corner_radius,
            )

            resize_tile >> layer
            crop_tile >> layer.transform()

            previous_image = stack

        g.register_ports(
            tap_image,
            tap_prompt,
            *tap_styles,
        )

    return g


def refine_sliding_tiles_with_controlnet_group(
    name: str,
    *,
    refine_spec: SpecInput,
    image_size: tuple[int, int],
    grid: tuple[int, int] = (3, 3),
    window_fraction: tuple[float, float] = (0.5, 0.5),
    controlnet_models: list[str] | None = None,
    controlnet_conditioning_scales: list[float] | None = None,
    aux_map_specs: list[SpecInput] | None = None,
    adapter_weight_names: list[str] | None = None,
    adapter_scales: list[float | dict] | None = None,
    adapter_tile_indices: dict[int, list[int]] | None = None,
    lora_models: list[str | dict] | None = None,
    lora_weights: list[float] | None = None,
    lora_tile_indices: dict[int, list[int]] | None = None,
    layer_feather: int | str = 40,
    layer_corner_radius: int | str = 60,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines an image through overlapping tiles
    with ControlNet.

    The upstream image is first resized to ``image_size`` and converted once
    into a full-image auxiliary conditioning map. Each tile is then cropped
    from the progressively refined image, while the matching ControlNet crop is
    taken from the original full auxiliary map. The tile is refined with
    ``Img2Img`` using IP-Adapter and ControlNet constraints, and composited back
    into the same coordinate space.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for every internal ``Img2Img`` refinement node.

    image_size : tuple[int, int]
        Working image size as ``(width, height)``.

        The macro resizes the upstream image to this exact size before generating
        tile boxes. All crop coordinates, crop metadata, auxiliary maps, and
        internal stack canvases use this same coordinate space.

    grid : tuple[int, int], optional
        Number of tile positions as ``(cols, rows)``. Default is ``(3, 3)``.

    window_fraction : tuple[float, float], optional
        Tile size as a fraction of ``image_size``, expressed as
        ``(width_fraction, height_fraction)``. Default is ``(0.5, 0.5)``.

    controlnet_models : list[str] or None, optional
        ControlNet model identifiers. When ``None`` and no legacy single
        ControlNet arguments are provided, no ControlNet is added.

    controlnet_conditioning_scales : list[float] or None, optional
        Conditioning scales aligned with ``controlnet_models``.

    aux_map_specs : list[SpecInput] or None, optional
        ``ImgAuxMap`` specs aligned with ``controlnet_models``.

    adapter_weight_names : list[str] or None, optional
        IP-Adapter weight names. When ``None``, no style adapters are added.

    adapter_scales : list[float | dict] or None, optional
        IP-Adapter scales aligned with ``adapter_weight_names``.

    adapter_tile_indices : dict[int, list[int]] or None, optional
        Map each adapter index to the tile indices it influences. Both indices
        are zero-based: adapter ``0`` refers to ``adapter_weight_names[0]``
        and port ``style_1``. Tiles are numbered left to right, top to bottom.

        An adapter absent from the mapping influences every tile. A mapped
        adapter influences only its listed tiles; an empty list disables that
        adapter on every tile. ``None`` and ``{}`` therefore apply all adapters
        to all tiles, preserving the default behavior.

        For example, ``{0: [0, 1], 1: []}`` limits adapter 0 to tiles 0 and 1,
        disables adapter 1, and leaves any other adapters active on every tile.
        A tile receives no adapters when all adapters are explicitly mapped
        and none lists that tile. Selected adapters retain their original
        order and scales. Invalid indices and duplicate tile indices are
        rejected.

    lora_models : list[str | dict] or None, optional
        LoRA models applied to every tile, independently of
        ``adapter_tile_indices``. Each entry is a repository ID or local path,
        or a dictionary with required ``id`` and optional ``weight_name`` to
        select a particular weight file. Other dictionary keys are rejected.
        No additional input ports are needed.

        Example: ``['org/detail', {'id': 'org/repo',
        'weight_name': 'style.safetensors'}]``.

    lora_weights : list[float] or None, optional
        Adapter weights aligned with ``lora_models``, passed to
        ``LoraRegistry.add`` as ``adapter_weight``. Both lists must be supplied
        together and have the same length. ``None`` for both, or two empty
        lists, adds no LoRAs. Declaration order is preserved on every tile.

    lora_tile_indices : dict[int, list[int]] or None, optional
        Map each zero-based LoRA index in ``lora_models`` to the zero-based tile
        indices it influences. Tiles are numbered left to right, top to bottom.

        A LoRA absent from the mapping influences every tile. A mapped LoRA
        influences only its listed tiles; an empty list disables that LoRA on
        every tile. ``None`` and ``{}`` therefore apply all LoRAs to all tiles.

        For example, ``{0: [0, 1], 1: []}`` limits LoRA 0 to tiles 0 and 1,
        disables LoRA 1, and leaves any other LoRAs active on every tile. A tile
        receives no LoRAs when all LoRAs are explicitly mapped and none lists
        that tile. Selected LoRAs retain their original order and weights.
        Invalid indices and duplicate tile indices are rejected.

    layer_feather : int or str, optional
        Feather applied when compositing every refined tile.

    layer_corner_radius : int or str, optional
        Corner radius applied to every refined tile mask.

    Returns
    -------
    NodeGroup
        The constructed tiled-refinement group with ControlNet constraints.

    Notes
    -----
    This macro intentionally owns its working canvas size. The upstream image may
    have any size, but it is resized to ``image_size`` before the first tile is
    cropped.

    When ``image_size`` has a different aspect ratio from the upstream image,
    the initial resize may stretch the image. This is intentional: ``image_size``
    defines the exact coordinate space used by the tiled refinement pass.

    Tiles are processed sequentially. Each tile is cropped from the progressively
    updated image produced by the previous tile. This makes overlapping regions
    accumulate refinements instead of having every tile compete directly against
    the original image.

    Auxiliary maps are generated once from the resized input image, then cropped
    per tile. This keeps ControlNet conditioning spatially aligned with every
    local refinement while anchoring all tiles to the same original geometry.

    Recommended settings:
        - strength <= 0.30
        - cfg/guidance_scale between 2.0 and 3.5
        - short prompts only

    High denoising strength or high CFG values may cause tiled drift, because each
    overlapping crop is regenerated independently and then propagated to subsequent
    tiles.
    """
    stack_spec = {
        'params': {
            'width': int(image_size[0]),
            'height': int(image_size[1]),
        },
    }

    bboxes = _sliding_window_bboxes_xyxy(
        image_size=image_size,
        grid=grid,
        window_fraction=window_fraction,
    )

    if not bboxes:
        raise ValueError(f'{name}: no tile boxes were generated')

    _validate_controlnet_lists(
        name=name,
        controlnet_models=controlnet_models,
        controlnet_conditioning_scales=controlnet_conditioning_scales,
        aux_map_specs=aux_map_specs,
    )

    if (adapter_weight_names is None) != (adapter_scales is None):
        raise ValueError(
            f'{name}: adapter_weight_names and adapter_scales must both be set '
            'or both be None'
        )
    if adapter_weight_names is not None and (
        len(adapter_weight_names) != len(adapter_scales)
    ):
        raise ValueError(
            f'{name}: adapter_weight_names and adapter_scales must have the '
            'same length'
        )

    if (lora_models is None) != (lora_weights is None):
        raise ValueError(
            f'{name}: lora_models and lora_weights must both be set or both be None'
        )
    if lora_models is not None and len(lora_models) != len(lora_weights):
        raise ValueError(f'{name}: lora_models and lora_weights must have the same length')
    for model in lora_models or []:
        if isinstance(model, str):
            valid = bool(model.strip())
        elif isinstance(model, dict):
            valid = (
                not (model.keys() - {'id', 'weight_name'})
                and isinstance(model.get('id'), str)
                and bool(model['id'].strip())
                and (
                    model.get('weight_name') is None
                    or isinstance(model['weight_name'], str)
                    and bool(model['weight_name'].strip())
                )
            )
        else:
            valid = False
        if not valid:
            raise ValueError(
                f'{name}: each lora_models entry must be a nonempty ID '
                'or a dictionary with id and optional weight_name'
            )

    lora_tile_indices = lora_tile_indices or {}
    lora_count = len(lora_models or [])
    for lora_idx, tile_indices in lora_tile_indices.items():
        if type(lora_idx) is not int or not 0 <= lora_idx < lora_count:
            raise ValueError(f'{name}: invalid LoRA index {lora_idx!r}')
        if not isinstance(tile_indices, list):
            raise ValueError(
                f'{name}: tile indices for LoRA {lora_idx} must be a list'
            )
        for tile_idx in tile_indices:
            if type(tile_idx) is not int or not 0 <= tile_idx < len(bboxes):
                raise ValueError(
                    f'{name}: invalid tile index {tile_idx!r} '
                    f'for LoRA {lora_idx}'
                )
        if len(set(tile_indices)) != len(tile_indices):
            raise ValueError(
                f'{name}: duplicate tile indices for LoRA {lora_idx}'
            )

    adapter_tile_indices = adapter_tile_indices or {}
    adapter_count = len(adapter_weight_names or [])
    for style_idx, tile_indices in adapter_tile_indices.items():
        if type(style_idx) is not int or not 0 <= style_idx < adapter_count:
            raise ValueError(f'{name}: invalid adapter index {style_idx!r}')
        if not isinstance(tile_indices, list):
            raise ValueError(
                f'{name}: tile indices for adapter {style_idx} must be a list'
            )
        for tile_idx in tile_indices:
            if type(tile_idx) is not int or not 0 <= tile_idx < len(bboxes):
                raise ValueError(
                    f'{name}: invalid tile index {tile_idx!r} '
                    f'for adapter {style_idx}'
                )
        if len(set(tile_indices)) != len(tile_indices):
            raise ValueError(
                f'{name}: duplicate tile indices for adapter {style_idx}'
            )

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_styles = [
            Tap(name=f'style_{idx + 1}')
            for idx in range(len(adapter_weight_names or []))
        ]

        resized_image = ResizeImage(
            name='resize_input',
            spec={
                'params': {
                    'size': [
                        int(image_size[0]),
                        int(image_size[1]),
                    ],
                },
            },
        )

        aux_maps = [
            ImgAuxMap(
                name=f'aux_map_{idx + 1:02d}',
                spec=spec,
            )
            for idx, spec in enumerate(aux_map_specs or [])
        ]

        tap_image >> resized_image
        for aux_map in aux_maps:
            resized_image >> aux_map
        previous_image = resized_image

        for idx, bbox in enumerate(bboxes):
            is_last = idx == len(bboxes) - 1

            # -------------------
            # Crop tile from progressively refined image
            # -------------------
            crop_tile = BoxCrop(
                name=f'crop_{idx:02d}',
                spec={
                    'params': {
                        'bbox_format': 'xyxy',
                        'bbox': list(bbox),
                    },
                },
            )

            crop_aux_maps = [
                BoxCrop(
                    name=f'crop_aux_{idx:02d}_{control_idx + 1:02d}',
                    spec={
                        'params': {
                            'bbox_format': 'xyxy',
                            'bbox': list(bbox),
                        },
                    },
                )
                for control_idx in range(len(controlnet_models or []))
            ]

            # -------------------
            # Refine tile with style adapters and geometric constraint
            # -------------------
            refine_tile = Img2Img(
                name=f'refine_{idx:02d}',
                spec=refine_spec,
            )

            lora_indices = [
                lora_idx for lora_idx in range(lora_count)
                if lora_idx not in lora_tile_indices
                or idx in lora_tile_indices[lora_idx]
            ]
            for lora_idx in lora_indices:
                model = lora_models[lora_idx]
                refine_tile.lora.add(
                    model if isinstance(model, str) else model['id'],
                    weight_name=None if isinstance(model, str) else model.get('weight_name'),
                    adapter_weight=lora_weights[lora_idx],
                )

            resize_tile = ResizeImage(
                name=f'resize_{idx:02d}',
                spec={},
            )

            style_indices = [
                style_idx for style_idx in range(adapter_count)
                if style_idx not in adapter_tile_indices
                or idx in adapter_tile_indices[style_idx]
            ]
            adapter_sinks = []
            for style_idx in style_indices:
                adapter_sinks.append(refine_tile.ip_adapter.add(
                    'h94/IP-Adapter',
                    subfolder='sdxl_models',
                    weight_name=adapter_weight_names[style_idx],
                    scale=adapter_scales[style_idx],
                    key=f'style_{style_idx + 1}',
                ))

            controlnet_sinks = [
                refine_tile.controlnet.add(
                    model,
                    conditioning_scale=float(scale),
                    key=f'geometry_{control_idx + 1}',
                )
                for control_idx, (model, scale) in enumerate(zip(
                    controlnet_models or [],
                    controlnet_conditioning_scales or [],
                ))
            ]

            # -------------------
            # Composite refined tile back
            # -------------------
            stack_name = 'out' if is_last else f'stack_{idx:02d}'

            stack = ImageStack(
                name=stack_name,
                spec=stack_spec,
            )

            # 1) Use the progressively refined image as the current background.
            previous_image >> stack.image(0)

            # 2) Crop the current tile from the progressively refined image.
            previous_image >> crop_tile

            # 3) Crop every stable full-image auxiliary map for this tile.
            for aux_map, crop_aux_map in zip(aux_maps, crop_aux_maps):
                aux_map >> crop_aux_map

            # 4) Refine the tile with all constraints.
            crop_tile >> refine_tile
            tap_prompt >> refine_tile.prompt()
            for style_idx, adapter_sink in zip(style_indices, adapter_sinks):
                tap_styles[style_idx] >> adapter_sink
            for crop_aux_map, controlnet_sink in zip(
                crop_aux_maps,
                controlnet_sinks,
            ):
                crop_aux_map >> controlnet_sink

            # 5) Restore the refined tile to the original crop size before
            #    compositing. Img2Img may emit a different resolution, while
            #    BoxCrop anchor metadata is expressed in crop-local pixels.
            refine_tile >> resize_tile
            crop_tile >> resize_tile.transform()

            # 6) Composite the resized refined tile back using crop metadata.
            layer = stack.image(
                1,
                position='center',
                feather=layer_feather,
                corner_radius=layer_corner_radius,
            )

            resize_tile >> layer
            crop_tile >> layer.transform()

            previous_image = stack

        g.register_ports(
            tap_image,
            tap_prompt,
            *tap_styles,
        )

    return g


def refine_mask_segments_with_controlnet_group(
    name: str,
    *,
    refine_spec: SpecInput,
    image_size: tuple[int, int],
    axis: SegmentAxis = 'y',
    start: SizeExpr = 0,
    end: SizeExpr | None = None,
    segments: int = 4,
    overlap: SizeExpr = 0,
    order: SegmentOrder = 'forward',
    bbox_margin: SizeExpr = 0,
    mask_feather: int | str = 0,
    layer_feather: int | str = 40,
    layer_corner_radius: int | str = 0,
    controlnet_models: list[str] | None = None,
    controlnet_conditioning_scales: list[float] | None = None,
    aux_map_specs: list[SpecInput] | None = None,
    adapter_weight_names: list[str] | None = None,
    adapter_scales: list[float | dict] | None = None,
) -> NodeGroup:
    """
    Refine a full-frame masked region through sequential axis-aligned segments.

    This macro is meant for cases where an upstream semantic crop has already
    produced a full-frame mask. The mask is split into horizontal or vertical
    bands, and each band is used as the inpaint mask for one sequential
    full-frame ``Inpaint`` pass. Every segment bbox may be expanded by
    ``bbox_margin`` before cropping, so nearby context can be included without
    changing the segment-generation range.

    Ports
    -----
    ``in_image``
        Image to refine. It must already use the same coordinate space as
        ``image_size``.

    ``in_mask``
        Full-frame mask to segment. It must already use the same coordinate
        space as ``image_size``. It is cropped per band, then placed back onto
        a black full-frame canvas with
        ``mask_feather`` using ``ImageStack`` and the crop transform metadata.

    ``in_prompt``
        Prompt forwarded to every internal ``Inpaint`` node.

    ``style_N``
        Optional IP-Adapter style ports, one for each item in
        ``adapter_weight_names``.

    Notes
    -----
    ``axis='y'`` creates horizontal bands between ``start`` and ``end``.
    ``axis='x'`` creates vertical bands. ``start``, ``end`` and ``overlap`` use
    the shared size-expression syntax: pixels, ``"120px"``, or percentages.

    The refined segment is cropped from the full-frame inpaint result and
    composited back onto the progressively updated image with ``layer_feather``.
    This keeps ``BoxCrop`` purely geometric while ``ImageStack`` owns the
    blending behavior.

    ControlNet auxiliary maps are generated once from the input image and
    reused for every segment, matching the stable-geometry behavior of
    ``refine_sliding_tiles_with_controlnet_group``.
    """
    base_bboxes = _band_segment_bboxes_xyxy(
        image_size=image_size,
        axis=axis,
        start=start,
        end=end,
        segments=segments,
        overlap=overlap,
        order=order,
    )
    bboxes = [
        expand_clip_bbox_by_size_expr(
            x1,
            y1,
            x2,
            y2,
            int(image_size[0]),
            int(image_size[1]),
            bbox_margin,
        )
        for x1, y1, x2, y2 in base_bboxes
    ]

    if not bboxes:
        raise ValueError(f'{name}: no segment boxes were generated')

    _validate_controlnet_lists(
        name=name,
        controlnet_models=controlnet_models,
        controlnet_conditioning_scales=controlnet_conditioning_scales,
        aux_map_specs=aux_map_specs,
    )

    if (adapter_weight_names is None) != (adapter_scales is None):
        raise ValueError(
            f'{name}: adapter_weight_names and adapter_scales must both be set '
            'or both be None'
        )
    if adapter_weight_names is not None and (
        len(adapter_weight_names) != len(adapter_scales)
    ):
        raise ValueError(
            f'{name}: adapter_weight_names and adapter_scales must have the '
            'same length'
        )

    with NodeGroup(name) as g:
        tap_image = Tap(name='in_image')
        tap_mask = Tap(name='in_mask')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_styles = [
            Tap(name=f'style_{idx + 1}')
            for idx in range(len(adapter_weight_names or []))
        ]

        stack_spec = {
            'params': {
                'width': int(image_size[0]),
                'height': int(image_size[1]),
            },
        }
        mask_stack_spec = {
            'params': {
                'width': int(image_size[0]),
                'height': int(image_size[1]),
                'background': [0, 0, 0, 255],
                'out_mode': 'RGB',
            },
        }

        aux_maps = [
            ImgAuxMap(
                name=f'aux_map_{idx + 1:02d}',
                spec=spec,
            )
            for idx, spec in enumerate(aux_map_specs or [])
        ]
        for aux_map in aux_maps:
            tap_image >> aux_map

        previous_image = tap_image

        for idx, bbox in enumerate(bboxes):
            is_last = idx == len(bboxes) - 1

            crop_image = BoxCrop(
                name=f'crop_image_{idx:02d}',
                spec={
                    'params': {
                        'bbox_format': 'xyxy',
                        'bbox': list(bbox),
                    },
                },
            )

            crop_mask = BoxCrop(
                name=f'crop_mask_{idx:02d}',
                spec={
                    'params': {
                        'bbox_format': 'xyxy',
                        'bbox': list(bbox),
                    },
                },
            )

            segment_mask = ImageStack(
                name=f'segment_mask_{idx:02d}',
                spec=mask_stack_spec,
            )

            refine_segment = Inpaint(
                name=f'inpaint_{idx:02d}',
                spec=refine_spec,
            )

            crop_refined = BoxCrop(
                name=f'crop_refined_{idx:02d}',
                spec={
                    'params': {
                        'bbox_format': 'xyxy',
                        'bbox': list(bbox),
                    },
                },
            )

            stack = ImageStack(
                name='out' if is_last else f'stack_{idx:02d}',
                spec=stack_spec,
            )

            adapter_sinks = []
            for style_idx, (weight_name, scale) in enumerate(zip(
                adapter_weight_names or [],
                adapter_scales or [],
            )):
                adapter_sinks.append(refine_segment.ip_adapter.add(
                    'h94/IP-Adapter',
                    subfolder='sdxl_models',
                    weight_name=weight_name,
                    scale=scale,
                    key=f'style_{style_idx + 1}',
                ))

            previous_image >> crop_image
            tap_mask >> crop_mask

            mask_layer = segment_mask.image(
                1,
                position='center',
                feather=mask_feather,
            )
            crop_mask >> mask_layer
            crop_mask >> mask_layer.transform()

            previous_image >> refine_segment
            segment_mask >> refine_segment.mask()
            tap_prompt >> refine_segment.prompt()

            for tap_style, adapter_sink in zip(tap_styles, adapter_sinks):
                tap_style >> adapter_sink

            for control_idx, (aux_map, model, scale) in enumerate(zip(
                aux_maps,
                controlnet_models or [],
                controlnet_conditioning_scales or [],
            )):
                aux_map >> refine_segment.controlnet.add(
                    model,
                    conditioning_scale=float(scale),
                    key=f'geometry_{control_idx + 1}',
                )

            previous_image >> stack.image(0)
            refine_segment >> crop_refined

            layer = stack.image(
                1,
                position='center',
                feather=layer_feather,
                corner_radius=layer_corner_radius,
            )
            crop_refined >> layer
            crop_image >> layer.transform()

            previous_image = stack

        g.register_ports(
            tap_image,
            tap_mask,
            tap_prompt,
            *tap_styles,
        )

    return g
