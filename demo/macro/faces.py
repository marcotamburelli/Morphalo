from __future__ import annotations

from stability.dag import NodeGroup
from stability.nodes import FaceIdEmbedImage, Img2Img, Inpaint, Tap
from stability.nodes.common.config_resolve import SpecInput
from stability.nodes.preprocess import ImageStack, SubjectCrop


def refine_face_group(
    name: str,
    *,
    refine_spec: SpecInput,
    stack_spec: SpecInput,
    face_id_scale: float,
    face_id_clip_strength: float = 1.0,
    layer_position: str = 'center',
    layer_feather: int = 30,
    layer_corner_radius: int = 50,
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
    layer_position : str, optional
        Overlay position for the refined head layer.
    layer_feather : int, optional
        Feather applied when compositing the refined layer.
    layer_corner_radius : int, optional
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
        group >> downstream
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports (entry nodes)
        # -------------------
        tap_image = Tap(name='in_image')
        face_embed = FaceIdEmbedImage(name='in_face_image')

        # -------------------
        # Internal nodes
        # -------------------
        crop_face = SubjectCrop(
            name='crop_face',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
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
            model_name='h94/IP-Adapter-FaceID',
            weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
            scale=face_id_scale,
            clip_strength=face_id_clip_strength,
            key='face_id',
        )

        stack = ImageStack(
            name='merge',
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
            position=layer_position,
            feather=layer_feather,
            corner_radius=layer_corner_radius,
        )

        refine_face >> layer1

        # Use crop metadata (anchor + bbox size) to place and scale the layer
        crop_face >> layer1.transform()

        # Expose ports by registering the entry nodes
        g.register_ports(tap_image, face_embed)

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
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
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
    face_id_scale: float,
    face_id_clip_strength: float = 1.0,
    mask_dilate_radius: int = -5,
    mask_close_radius: int = 10,
    mask_smoothing_radius: int = 10,
    mask_expansion: float | None = None,
) -> NodeGroup:
    """
    Create a reusable NodeGroup for subtle facial detail consolidation.

    This macro implements the pattern:

    - image -> face mask (SubjectCrop target='face', mode='mask')
    - image -> Inpaint (low strength)
    - prompt (details) -> Inpaint.prompt()
    - face image -> FaceIdEmbedImage -> FaceID adapter sink
    - (optional) face mask -> face_id.mask()

    The intent is to perform small local edits (eye color, eyebrows, micro skin
    details) while keeping identity stable via FaceID anchoring.

    IMPORTANT
    ---------
    This macro assumes the face already present in the input image is consistent
    with the FaceID reference. If the input face differs significantly, the
    result may become unstable or inconsistent (it is not a "face swap" macro).

    Ports
    -----
    - 'in_image' (Tap)
        The base image to be edited.
    - 'in_face_image' (FaceIdEmbedImage)
        The reference image used to compute FaceID embedding.
    - 'in_prompt' (Tap)
        Prompt payload used for facial micro-details.

    Output
    ------
    The group output node is the internal Inpaint node ('out').

    Parameters
    ----------
    name : str
        NodeGroup name (scope prefix).
    inpaint_spec : SpecInput
        Spec for the internal Inpaint node (you typically set low strength here).
    face_id_scale : float
        FaceID adapter scale. For this macro you usually want it relatively high
        to stabilize identity while inpaint does local edits.
    face_id_clip_strength : float, optional
        CLIP conditioning strength for FaceID adapter. Often 0..0.2 works well
        for detail consolidation.
    mask_dilate_radius : int, optional
        Dilation radius for the mask. Negative values shrink the region.
    mask_close_radius : int, optional
        Closing radius for the mask.
    mask_smoothing_radius : int, optional
        Smoothing radius for the mask.
    mask_expansion : float or None, optional
        Optional expansion factor passed to SubjectCrop for targets that support it
        (e.g. 'head'). When None, the parameter is omitted.

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
        face_embed = FaceIdEmbedImage(name='in_face_image')

        # -------------------
        # Internal nodes
        # -------------------
        mask_params = {
            'target': 'face',
            'mode': 'mask',
            'dilate_radius': mask_dilate_radius,
            'close_radius': mask_close_radius,
            'smoothing_radius': mask_smoothing_radius,
        }
        if mask_expansion is not None:
            mask_params['expansion'] = mask_expansion

        face_mask = SubjectCrop(
            name='face_mask',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
                },
                'params': mask_params,
            },
        )

        out = Inpaint(
            name='out',
            spec=inpaint_spec,
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

        # Base image to both mask generation and inpaint input
        tap_image >> face_mask
        tap_image >> out

        # Mask constrains inpaint to the face region
        face_mask >> out.mask()

        # Prompt drives the micro-detail edit
        tap_prompt >> out.prompt()

        # Face embedding anchors identity during inpaint
        face_embed >> face_id_sink

        # -------------------
        # Register selectable ports
        # -------------------
        g.register_ports(tap_image, face_embed, tap_prompt)

    return g
