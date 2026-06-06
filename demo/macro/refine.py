from __future__ import annotations

from typing import Literal, Optional

from morphalo.dag import NodeGroup
from morphalo.nodes import FaceIdEmbedImage, Img2Img, Inpaint, Tap
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.preprocess import (BoxCrop, FaceCrop, ImageStack,
                                       ImgAuxMap, SubjectCrop)

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
                'model': {
                    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
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
                'model': {
                    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
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
        model_spec = {
            'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
        }
        if region == 'head':
            model_spec['pose_landmarker_task'] = '~/models/mediapipe/pose_landmarker_heavy.task'

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
                'model': {
                    'hand_landmarker_task': '~/models/mediapipe/hand_landmarker.task',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
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


def refine_sliding_tiles_group(
    name: str,
    *,
    refine_spec: SpecInput,
    image_size: tuple[int, int],
    grid: tuple[int, int] = (3, 3),
    window_fraction: tuple[float, float] = (0.5, 0.5),
    stack_spec: SpecInput = {},
    texture_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
    struct_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
    texture_weight: float = 0.75,
    struct_weight: float = 0.75,
    texture_compl: float = 0.1,
    struct_compl: float = 0.1,
    layer_feather: int | str = 40,
    layer_corner_radius: int | str = 60,
    base_resize: Literal['fit', 'cover'] = 'cover',
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines an image through overlapping tiles.

    This group applies a sliding-window refinement strategy. The input image is
    divided into overlapping rectangular regions, each region is refined with
    ``Img2Img`` using two IP-Adapter references, and the refined crop is
    composited back onto the progressively updated image.

    The macro is intended to improve small or mid-sized details that global
    generation often approximates poorly, such as malformed hands, feet, small
    objects, distorted background details, accessories, fabric folds, or local
    artifacts.

    Unlike semantic refinement macros, this group does not detect a specific
    target. It scans the image using a fixed overlapping grid.

    Ports
    -----
    in_image
        Base image to refine.

    in_prompt
        Optional prompt payload wired into every internal ``Img2Img`` node.

    in_texture
        Style reference used mostly for texture/detail conditioning.

    in_struct
        Style reference used mostly for structure/composition conditioning.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for every internal ``Img2Img`` refinement node.

    image_size : tuple[int, int]
        Input image size fallback as ``(width, height)``. If ``stack_spec``
        declares ``params.width`` / ``params.height``, that canvas size is used
        for the initial resize stack and for static tile generation.

    grid : tuple[int, int], optional
        Number of tile positions as ``(cols, rows)``. Default is ``(3, 3)``.

    window_fraction : tuple[float, float], optional
        Tile size as a fraction of the image size, expressed as
        ``(width_fraction, height_fraction)``.

        With ``grid=(3, 3)`` and ``window_fraction=(0.5, 0.5)``, the image is
        refined using nine half-image crops whose origins slide over the image
        with a stride of one quarter of the image size.

    stack_spec : SpecInput, optional
        Spec for every internal ``ImageStack`` node.

    texture_weight_name : str, optional
        IP-Adapter weight name used for the texture reference.

    struct_weight_name : str, optional
        IP-Adapter weight name used for the structure reference.

    texture_weight : float, optional
        Main texture strength applied mostly to the upper blocks.

    struct_weight : float, optional
        Main structure strength applied mostly to the lower blocks.

    texture_compl : float, optional
        Complementary low-block texture strength.

    struct_compl : float, optional
        Complementary up-block structure strength.

    layer_feather : int or str, optional
        Feather applied when compositing every refined tile.

    layer_corner_radius : int or str, optional
        Corner radius applied to every refined tile mask.

    base_resize : {'fit', 'cover'}, optional
        Resize mode used only when ``stack_spec.params.width`` / ``height``
        differ from ``image_size``. In that case the input image is first placed
        on an ``ImageStack`` canvas with this resize mode, and all subsequent
        crops operate on that resized canvas.

    Returns
    -------
    NodeGroup
        The constructed tiled-refinement group.

    Notes
    -----
    Tiles are processed sequentially. Each tile is cropped from the progressively
    updated image produced by the previous tile. This makes overlapping regions
    accumulate refinements instead of having every tile compete directly against
    the original image.

    This is usually safer than refining all tiles in parallel and stacking them
    at the end, because parallel tiles may disagree in overlapping regions.

    This macro is intended for conservative refinement.

    Recommended settings:
        - strength <= 0.30
        - cfg/guidance_scale between 2.0 and 3.5
        - short prompts only

    High denoising strength or high CFG values may cause tiled drift, because each
    overlapping crop is regenerated independently and then propagated to subsequent
    tiles.
    """
    stack_cfg = resolve_spec(stack_spec)
    stack_params = stack_cfg.get('params', {})
    canvas_size = (
        int(stack_params.get('width', image_size[0])),
        int(stack_params.get('height', image_size[1])),
    )

    bboxes = _sliding_window_bboxes_xyxy(
        image_size=canvas_size,
        grid=grid,
        window_fraction=window_fraction,
    )

    if not bboxes:
        raise ValueError(f'{name}: no tile boxes were generated')

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_texture = Tap(name='in_texture')
        tap_style_struct = Tap(name='in_struct')

        previous_image = tap_image

        if canvas_size != image_size:
            base_stack = ImageStack(
                name='base_canvas',
                spec=stack_spec,
            )
            tap_image >> base_stack.image(0, resize=base_resize)
            previous_image = base_stack

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

            texture_sink = refine_tile.ip_adapter.add(
                'h94/IP-Adapter',
                subfolder='sdxl_models',
                weight_name=texture_weight_name,
                scale={
                    'down': {'block_2': [0, texture_compl]},
                    'up': {'block_0': [0.0, texture_weight, 0.0]},
                },
                key='texture',
            )

            struct_sink = refine_tile.ip_adapter.add(
                'h94/IP-Adapter',
                subfolder='sdxl_models',
                weight_name=struct_weight_name,
                scale={
                    'down': {'block_2': [0, struct_weight]},
                    'up': {'block_0': [0.0, struct_compl, 0.0]},
                },
                key='structure',
            )

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
            tap_style_texture >> texture_sink
            tap_style_struct >> struct_sink

            # 4) Composite the refined tile back using crop metadata.
            layer = stack.image(
                1,
                position='center',
                feather=layer_feather,
                corner_radius=layer_corner_radius,
            )

            refine_tile >> layer
            crop_tile >> layer.transform()

            previous_image = stack

        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style_texture,
            tap_style_struct,
        )

    return g


def refine_sliding_tiles_with_controlnet_group(
    name: str,
    *,
    refine_spec: SpecInput,
    image_size: tuple[int, int],
    grid: tuple[int, int] = (3, 3),
    window_fraction: tuple[float, float] = (0.5, 0.5),
    stack_spec: SpecInput = {},
    controlnet_model: str = 'diffusers/controlnet-depth-sdxl-1.0',
    controlnet_conditioning_scale: float = 0.7,
    aux_map_spec: SpecInput = {
        'processor': 'depth_midas',
    },
    texture_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
    struct_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
    texture_weight: float = 0.75,
    struct_weight: float = 0.75,
    texture_compl: float = 0.1,
    struct_compl: float = 0.1,
    layer_feather: int | str = 40,
    layer_corner_radius: int | str = 60,
    base_resize: Literal['fit', 'cover'] = 'cover',
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines an image through overlapping tiles with ControlNet.

    Similar to `refine_sliding_tiles_group`, but each tile refinement is additionally
    constrained by a geometric conditioning signal (e.g., depth, canny edges) via ControlNet.

    The macro:
    1. Divides the input image into overlapping rectangular tiles.
    2. For each tile:
       - Crops the current (progressively refined) image.
       - Generates an auxiliary map (depth, canny, etc.) from the tile using ``ImgAuxMap``.
       - Refines the tile with ``Img2Img`` using:
         * Two IP-Adapter references (texture and structure)
         * The auxiliary map as ControlNet conditioning
       - Composites the refined tile back onto the progressively updated image.

    This approach combines both semantic (style via IP-Adapters) and geometric (structure
    via ControlNet) constraints to improve detail coherence.

    Ports
    -----
    in_image
        Base image to refine.

    in_prompt
        Optional prompt payload wired into every internal ``Img2Img`` node.

    in_texture
        Style reference used mostly for texture/detail conditioning.

    in_struct
        Style reference used mostly for structure/composition conditioning.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for every internal ``Img2Img`` refinement node.

    image_size : tuple[int, int]
        Input image size fallback as ``(width, height)``. If ``stack_spec``
        declares ``params.width`` / ``params.height``, that canvas size is used
        for the initial resize stack and for static tile generation.

    grid : tuple[int, int], optional
        Number of tile positions as ``(cols, rows)``. Default is ``(3, 3)``.

    window_fraction : tuple[float, float], optional
        Tile size as a fraction of the image size, expressed as
        ``(width_fraction, height_fraction)``. Default is ``(0.5, 0.5)``.

    stack_spec : SpecInput, optional
        Spec for every internal ``ImageStack`` node.

    controlnet_model : str, optional
        ControlNet model identifier (e.g., ``'diffusers/controlnet-depth-sdxl-1.0'``).
        Default is ``'diffusers/controlnet-depth-sdxl-1.0'``.

    controlnet_conditioning_scale : float, optional
        Conditioning scale for the ControlNet adapter. Default is ``0.7``.

    aux_map_spec : SpecInput, optional
        Spec for every internal ``ImgAuxMap`` node used to generate the geometric
        constraint. By default this uses MiDaS depth, matching ``controlnet_model``:
        ``{'processor': 'depth_midas'}``.
        Pass a matching processor/model pair when using a different ControlNet.

    texture_weight_name : str, optional
        IP-Adapter weight name used for the texture reference.

    struct_weight_name : str, optional
        IP-Adapter weight name used for the structure reference.

    texture_weight : float, optional
        Main texture strength applied mostly to the upper blocks.

    struct_weight : float, optional
        Main structure strength applied mostly to the lower blocks.

    texture_compl : float, optional
        Complementary low-block texture strength.

    struct_compl : float, optional
        Complementary up-block structure strength.

    layer_feather : int or str, optional
        Feather applied when compositing every refined tile.

    layer_corner_radius : int or str, optional
        Corner radius applied to every refined tile mask.

    base_resize : {'fit', 'cover'}, optional
        Resize mode used only when ``stack_spec.params.width`` / ``height``
        differ from ``image_size``. In that case the input image is first placed
        on an ``ImageStack`` canvas with this resize mode, and all subsequent
        crops operate on that resized canvas.

    Returns
    -------
    NodeGroup
        The constructed tiled-refinement group with ControlNet constraints.

    Notes
    -----
    Tiles are processed sequentially. Each tile is cropped from the progressively
    updated image produced by the previous tile.

    The auxiliary map is generated only for the current tile (not the full image),
    which keeps ControlNet conditioning spatially aligned with the local refinement.

    Recommended settings:
        - refine_spec strength <= 0.30
        - cfg/guidance_scale between 2.0 and 3.5
        - short prompts only
    """
    stack_cfg = resolve_spec(stack_spec)
    stack_params = stack_cfg.get('params', {})
    canvas_size = (
        int(stack_params.get('width', image_size[0])),
        int(stack_params.get('height', image_size[1])),
    )

    bboxes = _sliding_window_bboxes_xyxy(
        image_size=canvas_size,
        grid=grid,
        window_fraction=window_fraction,
    )

    if not bboxes:
        raise ValueError(f'{name}: no tile boxes were generated')

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_texture = Tap(name='in_texture')
        tap_style_struct = Tap(name='in_struct')

        previous_image = tap_image

        if canvas_size != image_size:
            base_stack = ImageStack(
                name='base_canvas',
                spec=stack_spec,
            )
            tap_image >> base_stack.image(0, resize=base_resize)
            previous_image = base_stack

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

            # -------------------
            # Generate auxiliary map (depth/canny/etc) for the cropped tile
            # -------------------
            aux_map = ImgAuxMap(
                name=f'aux_map_{idx:02d}',
                spec=aux_map_spec,
            )

            # -------------------
            # Refine tile with style adapters and geometric constraint
            # -------------------
            refine_tile = Img2Img(
                name=f'refine_{idx:02d}',
                spec=refine_spec,
            )

            # Texture IP-Adapter
            texture_sink = refine_tile.ip_adapter.add(
                'h94/IP-Adapter',
                subfolder='sdxl_models',
                weight_name=texture_weight_name,
                scale={
                    'down': {'block_2': [0, texture_compl]},
                    'up': {'block_0': [0.0, texture_weight, 0.0]},
                },
                key='texture',
            )

            # Structure IP-Adapter
            struct_sink = refine_tile.ip_adapter.add(
                'h94/IP-Adapter',
                subfolder='sdxl_models',
                weight_name=struct_weight_name,
                scale={
                    'down': {'block_2': [0, struct_weight]},
                    'up': {'block_0': [0.0, struct_compl, 0.0]},
                },
                key='structure',
            )

            # ControlNet from auxiliary map
            controlnet_sink = refine_tile.controlnet.add(
                controlnet_model,
                conditioning_scale=controlnet_conditioning_scale,
                key='geometry',
            )

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

            # 3) Generate auxiliary map from the cropped tile.
            crop_tile >> aux_map

            # 4) Refine the tile with all constraints.
            crop_tile >> refine_tile
            tap_prompt >> refine_tile.prompt()
            tap_style_texture >> texture_sink
            tap_style_struct >> struct_sink
            aux_map >> controlnet_sink

            # 5) Composite the refined tile back using crop metadata.
            layer = stack.image(
                1,
                position='center',
                feather=layer_feather,
                corner_radius=layer_corner_radius,
            )

            refine_tile >> layer
            crop_tile >> layer.transform()

            previous_image = stack

        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style_texture,
            tap_style_struct,
        )

    return g
