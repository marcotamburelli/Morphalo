from morphalo.dag import NodeGroup
from morphalo.nodes import Img2Img, Tap, Txt2Img
from morphalo.nodes.common.config_resolve import SpecInput
from morphalo.nodes.evaluate import PersonScorer, PromptScorer
from morphalo.nodes.preprocess import (BoxCrop, ImgAuxMap,
                                       SubjectCrop)
from morphalo.nodes.process import ImageStack
from morphalo.nodes.wiring.ip_adapter import IpAdapterScale


SUBJECT_MODEL_SPEC = {
    'yolo_model': 'yolov8n.pt',
    'segment_model': 'facebook/sapiens2-seg-0.4b',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
    'device': 'cuda',
    'dtype': 'bfloat16',
}


def stylize_subject_background_singlepass_group(
    name: str,
    *,
    img2img_spec: SpecInput,
    bg_adapter_scale: IpAdapterScale,
    subject_adapter_scale: IpAdapterScale,
    subject_mask_dilate_radius: int = 10,
    subject_mask_close_radius: int = 10,
    subject_mask_smoothing_radius: int = 20,
    bg_mask_dilate_radius: int = 10,
    bg_mask_close_radius: int = 10,
    bg_mask_smoothing_radius: int = 20,
) -> NodeGroup:
    """
    Single-pass subject/background stylization using two masked IP-Adapters.

    The group computes:
    - subject mask via SAM (mode='mask')
    - background mask via SAM (mode='negative-mask')

    Then runs a single Img2Img where:
    - background style adapter is masked by background mask
    - subject style adapter is masked by subject mask

    Ports
    -----
    - 'in_image' (Tap)
    - 'in_prompt' (Tap)
    - 'bg_style' (Tap)
    - 'subject_style' (Tap)

    Output
    ------
    The group output is the internal Img2Img node named 'out'.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt')
        tap_bg_style = Tap(name='bg_style')
        tap_subject_style = Tap(name='subject_style')

        # -------------------
        # Masks (from input image)
        # -------------------
        subject_mask = SubjectCrop(
            name='subject_mask',
            spec={
                'model': SUBJECT_MODEL_SPEC,
                'params': {
                    'mode': 'mask',
                    'dilate_radius': subject_mask_dilate_radius,
                    'close_radius': subject_mask_close_radius,
                    'smoothing_radius': subject_mask_smoothing_radius,
                },
            },
        )

        bg_mask = SubjectCrop(
            name='bg_mask',
            spec={
                'model': SUBJECT_MODEL_SPEC,
                'params': {
                    'mode': 'negative-mask',
                    'dilate_radius': bg_mask_dilate_radius,
                    'close_radius': bg_mask_close_radius,
                    'smoothing_radius': bg_mask_smoothing_radius,
                },
            },
        )

        # -------------------
        # Single Img2Img pass
        # -------------------
        out = Img2Img(
            name='out',
            spec=img2img_spec,
        )

        bg_adapter = out.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=bg_adapter_scale,
            key='bg_style',
        )

        subject_adapter = out.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=subject_adapter_scale,
            key='subject_style',
        )

        # -------------------
        # Wiring
        # -------------------
        tap_image >> subject_mask
        tap_image >> bg_mask

        tap_image >> out
        tap_prompt >> out.prompt()

        tap_bg_style >> bg_adapter
        tap_subject_style >> subject_adapter

        bg_mask >> bg_adapter.mask()
        subject_mask >> subject_adapter.mask()

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(tap_image, tap_prompt,
                         tap_bg_style, tap_subject_style)

    return g


def two_stage_style_group(
    name: str,
    *,
    img1_spec: SpecInput,
    img2_spec: SpecInput,
    style1_scale: IpAdapterScale,
    style2_scale: IpAdapterScale,
) -> NodeGroup:
    """
    Two-stage Img2Img refinement group with two sequential style applications.

    This macro builds a ``NodeGroup`` composed of two ``Img2Img`` nodes in
    sequence. Each stage applies the same main prompt but uses a different
    IP-Adapter style reference.

    Pipeline
    --------
    Stage 1:
        - ``img_1`` takes ``in_image`` as init image
        - uses ``in_prompt`` as text conditioning
        - applies ``style_1`` through IP-Adapter

    Stage 2:
        - ``img_2`` takes the output of ``img_1`` as init image
        - reuses the same ``in_prompt``
        - applies ``style_2`` through IP-Adapter

    Input ports
    -----------
    in_image
        Base image for the first Img2Img stage.

    in_prompt
        Main prompt used by both Img2Img stages.

    style_1
        Style reference image(s) for the first Img2Img stage.

    style_2
        Style reference image(s) for the second Img2Img stage.

    Output
    ------
    The group output corresponds to the internal node ``img_2``.

    Parameters
    ----------
    name : str
        Name of the NodeGroup.

    img1_spec : SpecInput
        Configuration for the first Img2Img stage.

    img2_spec : SpecInput
        Configuration for the second Img2Img stage.

    style1_scale : IpAdapterScale
        IP-Adapter scale configuration for the first stage.

    style2_scale : IpAdapterScale
        IP-Adapter scale configuration for the second stage.

    Notes
    -----
    - This macro is a simplified counterpart of a two-stage style-and-scoring
      pipeline: it keeps the staged style transfer structure but removes all
      intermediate selection logic.
    - Both stages reuse the same generation prompt.
    - IP-Adapter weights are intentionally hardcoded in this version.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_1 = Tap(name='style_1')
        tap_style_2 = Tap(name='style_2')

        # -------------------
        # Stage 1
        # -------------------
        img_1 = Img2Img(
            name='img_1',
            spec=img1_spec,
        )

        tap_image >> img_1
        tap_prompt >> img_1.prompt()
        tap_style_1 >> img_1.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style1_scale,
            key='style_1',
        )

        # -------------------
        # Stage 2
        # -------------------
        img_2 = Img2Img(
            name='img_2',
            spec=img2_spec,
        )

        img_1 >> img_2
        tap_prompt >> img_2.prompt()
        tap_style_2 >> img_2.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style2_scale,
            key='style_2',
        )

        # -------------------
        # Ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style_1,
            tap_style_2,
        )

    return g


def two_stage_style_scoring_group(
    name: str,
    *,
    img1_spec: SpecInput,
    img2_spec: SpecInput,
    person_scorer_spec: SpecInput,
    prompt_scorer_spec: SpecInput,
    batch: int,
    style1_scale: IpAdapterScale,
    style2_scale: IpAdapterScale,
) -> NodeGroup:
    """
    Two-stage Img2Img refinement with intermediate person and prompt scoring.

    This macro builds a ``NodeGroup`` that performs two sequential Img2Img passes,
    each followed by a selection step based on:

    - structural plausibility (``PersonScorer``)
    - prompt alignment (``PromptScorer``)

    The same scoring configuration is applied at both stages to ensure
    consistency and comparability of the ranking signal.

    Pipeline
    --------
    Stage 1:
        - ``img_1`` generates ``batch`` candidate images from ``in_image``,
          conditioned by ``in_prompt`` and ``style_1``.
        - Candidates are evaluated by ``person_1`` and then ``prompt_1``.
        - ``prompt_1`` selects the best image.

    Stage 2:
        - ``img_2`` takes the selected image from stage 1 and generates
          a new batch of candidates using ``in_prompt`` and ``style_2``.
        - Candidates are evaluated again by ``person_2`` and ``prompt_2``.
        - ``prompt_2`` produces the final selection.

    Input ports
    -----------
    in_image
        Base image for the first Img2Img stage.

    in_prompt
        Main generation prompt used by both Img2Img stages.

    score_prompt
        Prompt used by both PromptScorer nodes.
        This is typically a secondary prompt focused on evaluation
        (e.g., structure, style consistency, or specific traits).

    style_1
        Style reference(s) for the first Img2Img stage (IP-Adapter).

    style_2
        Style reference(s) for the second Img2Img stage (IP-Adapter).

    Output
    ------
    The group output corresponds to the internal node ``prompt_2``, which
    provides:

    - selected best image (``image``)
    - final aggregated score (``best_score``)
    - full ranking structure (``rankings``)

    Parameters
    ----------
    name : str
        Name of the NodeGroup.

    img1_spec : SpecInput
        Configuration for the first Img2Img stage.

    img2_spec : SpecInput
        Configuration for the second Img2Img stage.

    person_scorer_spec : SpecInput
        Configuration for both PersonScorer nodes.

    prompt_scorer_spec : SpecInput
        Configuration for both PromptScorer nodes.

    batch : int
        Number of candidate images generated at each stage.

    style1_scale : IpAdapterScale
        IP-Adapter scale configuration for the first stage.

    style2_scale : IpAdapterScale
        IP-Adapter scale configuration for the second stage.

    Notes
    -----
    - The same scorer specs are reused across both stages to avoid
      configuration drift and ensure consistent ranking behavior.
    - The ``score_prompt`` is decoupled from ``in_prompt`` to allow
      evaluation using a different semantic or stylistic target.
    - The node operates as a refinement loop: stage 1 explores,
      stage 2 refines around the best candidate.
    """

    if batch <= 0:
        raise ValueError(f'Invalid batch={batch!r}. Expected batch > 0.')

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_score_prompt = Tap(name='score_prompt', strict=False)
        tap_style_1 = Tap(name='style_1')
        tap_style_2 = Tap(name='style_2')

        # -------------------
        # Stage 1 generation
        # -------------------
        img_1 = Img2Img(
            name='img_1',
            spec=[
                img1_spec,
                {'batch': batch},
            ],
        )

        tap_image >> img_1
        tap_prompt >> img_1.prompt()
        tap_style_1 >> img_1.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style1_scale,
            key='style_1',
        )

        # -------------------
        # Stage 1 scoring
        # -------------------
        person_1 = PersonScorer(
            name='person_1',
            spec=person_scorer_spec,
        )

        prompt_1 = PromptScorer(
            name='prompt_1',
            spec=prompt_scorer_spec,
        )

        img_1 >> person_1 >> prompt_1
        tap_score_prompt >> prompt_1.prompt()

        # -------------------
        # Stage 2 generation
        # -------------------
        img_2 = Img2Img(
            name='img_2',
            spec=[
                img2_spec,
                {'batch': batch},
            ],
        )

        prompt_1 >> img_2
        tap_prompt >> img_2.prompt()
        tap_style_2 >> img_2.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style2_scale,
            key='style_2',
        )

        # -------------------
        # Stage 2 scoring
        # -------------------
        person_2 = PersonScorer(
            name='person_2',
            spec=person_scorer_spec,
        )

        prompt_2 = PromptScorer(
            name='prompt_2',
            spec=prompt_scorer_spec,
        )

        img_2 >> person_2 >> prompt_2
        tap_score_prompt >> prompt_2.prompt()

        # -------------------
        # Ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            tap_score_prompt,
            tap_style_1,
            tap_style_2,
        )

    return g


def two_stage_style_canny_group(
    name: str,
    *,
    img1_spec: SpecInput,
    img2_spec: SpecInput,
    style1_scale: IpAdapterScale,
    style2_scale: IpAdapterScale,
    canny_detect_resolution: int = 1024,
    canny_conditioning_scale_1: float = 0.7,
    canny_conditioning_scale_2: float = 0.7,
) -> NodeGroup:
    """
    Two-stage Img2Img refinement with two sequential style applications and a
    shared Canny ControlNet derived from the input image.

    This macro builds a ``NodeGroup`` composed of two ``Img2Img`` nodes in
    sequence. Each stage reuses the same main prompt but applies a different
    IP-Adapter style reference.

    In addition, a Canny edge map is computed once from ``in_image`` and wired
    as ControlNet conditioning into both stages.

    Pipeline
    --------
    Shared preprocessing:
        - ``canny`` is computed from ``in_image`` via ``ImgAuxMap``

    Stage 1:
        - ``img_1`` takes ``in_image`` as init image
        - uses ``in_prompt`` as text conditioning
        - applies ``style_1`` through IP-Adapter
        - applies shared Canny ControlNet

    Stage 2:
        - ``img_2`` takes the output of ``img_1`` as init image
        - reuses the same ``in_prompt``
        - applies ``style_2`` through IP-Adapter
        - reuses the same shared Canny ControlNet

    Input ports
    -----------
    in_image
        Base image for the first Img2Img stage and source image for the Canny map.

    in_prompt
        Main prompt used by both Img2Img stages.

    style_1
        Style reference image(s) for the first Img2Img stage.

    style_2
        Style reference image(s) for the second Img2Img stage.

    Output
    ------
    The group output corresponds to the internal node ``img_2``.

    Parameters
    ----------
    name : str
        Name of the NodeGroup.

    img1_spec : SpecInput
        Configuration for the first Img2Img stage.

    img2_spec : SpecInput
        Configuration for the second Img2Img stage.

    style1_scale : IpAdapterScale
        IP-Adapter scale configuration for the first stage.

    style2_scale : IpAdapterScale
        IP-Adapter scale configuration for the second stage.

    canny_detect_resolution : int, optional
        Native short-side resolution used when computing the Canny map.

    canny_conditioning_scale_1 : float, optional
        ControlNet conditioning scale for stage 1.

    canny_conditioning_scale_2 : float, optional
        ControlNet conditioning scale for stage 2.

    Notes
    -----
    - The Canny map is computed once from ``in_image`` and reused in both stages.
    - This keeps structural guidance stable across the full two-stage refinement.
    - Both stages reuse the same generation prompt.
    - IP-Adapter and ControlNet are attached declaratively through the node
      registries exposed by ``Img2Img``.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_1 = Tap(name='style_1')
        tap_style_2 = Tap(name='style_2')

        # -------------------
        # Shared structural conditioning
        # -------------------
        canny = ImgAuxMap(
            name='canny',
            spec={
                'processor': 'canny',
                'detect_resolution': canny_detect_resolution,
            },
        )

        tap_image >> canny

        # -------------------
        # Stage 1
        # -------------------
        img_1 = Img2Img(
            name='img_1',
            spec=img1_spec,
        )

        tap_image >> img_1
        tap_prompt >> img_1.prompt()

        tap_style_1 >> img_1.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style1_scale,
            key='style_1',
        )

        canny >> img_1.controlnet.add(
            'diffusers/controlnet-canny-sdxl-1.0',
            conditioning_scale=canny_conditioning_scale_1,
            key='canny',
        )

        # -------------------
        # Stage 2
        # -------------------
        img_2 = Img2Img(
            name='img_2',
            spec=img2_spec,
        )

        img_1 >> img_2
        tap_prompt >> img_2.prompt()

        tap_style_2 >> img_2.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style2_scale,
            key='style_2',
        )

        canny >> img_2.controlnet.add(
            'diffusers/controlnet-canny-sdxl-1.0',
            conditioning_scale=canny_conditioning_scale_2,
            key='canny',
        )

        # -------------------
        # Ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style_1,
            tap_style_2,
        )

    return g


def style_depth_txt2img_group(
    name: str,
    *,
    txt2img_spec: SpecInput,
    style_scale: IpAdapterScale,
    depth_detect_resolution: int = 1024,
    depth_conditioning_scale: float = 0.7,
    ip_adapter_model_id: str = 'h94/IP-Adapter',
    ip_adapter_subfolder: str = 'sdxl_models',
    ip_adapter_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
) -> NodeGroup:
    """
    Single-stage Txt2Img generation with one style reference and one shared
    depth ControlNet derived from an external image.

    This macro builds a ``NodeGroup`` composed of:

    - one ``Txt2Img`` node;
    - one ``ImgAuxMap`` node computing a depth map from ``in_image``.

    The generated image is driven by:

    - ``in_prompt`` for semantic and stylistic text conditioning;
    - ``in_style`` through IP-Adapter for appearance/style transfer;
    - a depth map extracted from ``in_image`` through ControlNet for
      structural guidance.

    Unlike an Img2Img-based pattern, the input image is **not** used as an
    init image for generation. It is used only as the source from which the
    depth conditioning map is computed.

    Pipeline
    --------
    Shared preprocessing:
        - ``depth`` is computed from ``in_image`` via ``ImgAuxMap``

    Generation:
        - ``out`` is a ``Txt2Img`` node
        - ``out`` uses ``in_prompt`` as text conditioning
        - ``out`` applies ``in_style`` through IP-Adapter
        - ``out`` applies the shared depth ControlNet derived from ``in_image``

    Input ports
    -----------
    in_image
        Source image used only to compute the depth map for ControlNet guidance.

    in_prompt
        Main prompt used by the ``Txt2Img`` node.

    in_style
        Style reference image or images for the IP-Adapter branch.

    Output
    ------
    The group output corresponds to the internal node ``out``.

    Parameters
    ----------
    name : str
        Name of the NodeGroup.

    txt2img_spec : SpecInput
        Configuration for the internal ``Txt2Img`` node.

    style_scale : IpAdapterScale
        IP-Adapter scale configuration for the style branch.

    depth_detect_resolution : int, optional
        Native short-side resolution used when computing the depth map.

    depth_conditioning_scale : float, optional
        ControlNet conditioning scale for the depth branch.

    ip_adapter_model_id : str, optional
        Hugging Face repository identifier for the IP-Adapter model.

    ip_adapter_subfolder : str, optional
        Repository subfolder containing the IP-Adapter weights.

    ip_adapter_weight_name : str, optional
        IP-Adapter weight file name.

    Returns
    -------
    NodeGroup
        The constructed group.

    Notes
    -----
    - The depth map is computed once from ``in_image`` and used only as
      ControlNet conditioning.
    - ``in_image`` is not passed directly to the ``Txt2Img`` node.
    - Structural guidance is provided by ControlNet depth, while visual style
      guidance is provided by IP-Adapter.
    - The IP-Adapter is configurable through model id, subfolder, and weight
      name.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style = Tap(name='in_style')

        # -------------------
        # Shared structural conditioning
        # -------------------
        depth = ImgAuxMap(
            name='depth',
            spec={
                'processor': 'depth_midas',
                'detect_resolution': depth_detect_resolution,
            },
        )

        tap_image >> depth

        # -------------------
        # Single generation stage
        # -------------------
        out = Txt2Img(
            name='out',
            spec=txt2img_spec,
        )

        tap_prompt >> out.prompt()

        tap_style >> out.ip_adapter.add(
            ip_adapter_model_id,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
            scale=style_scale,
            key='style',
        )

        depth >> out.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale,
            key='depth',
        )

        # -------------------
        # Ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style,
        )

    return g


def two_stage_style_depth_group(
    name: str,
    *,
    img1_spec: SpecInput,
    img2_spec: SpecInput,
    style1_scale: IpAdapterScale,
    style2_scale: IpAdapterScale,
    depth_detect_resolution: int = 1024,
    depth_conditioning_scale_1: float = 0.7,
    depth_conditioning_scale_2: float = 0.7,
) -> NodeGroup:
    """
    Two-stage Img2Img refinement with two sequential style applications and a
    shared depth ControlNet derived from the input image.

    This macro builds a ``NodeGroup`` composed of two ``Img2Img`` nodes in
    sequence. Each stage reuses the same main prompt but applies a different
    IP-Adapter style reference.

    In addition, a depth map is computed once from ``in_image`` via MiDaS and
    wired as ControlNet conditioning into both stages.

    Pipeline
    --------
    Shared preprocessing:
        - ``depth`` is computed from ``in_image`` via ``ImgAuxMap``

    Stage 1:
        - ``img_1`` takes ``in_image`` as init image
        - uses ``in_prompt`` as text conditioning
        - applies ``style_1`` through IP-Adapter
        - applies shared depth ControlNet

    Stage 2:
        - ``img_2`` takes the output of ``img_1`` as init image
        - reuses the same ``in_prompt``
        - applies ``style_2`` through IP-Adapter
        - reuses the same shared depth ControlNet

    Input ports
    -----------
    in_image
        Base image for the first Img2Img stage and source image for the depth map.

    in_prompt
        Main prompt used by both Img2Img stages.

    style_1
        Style reference image(s) for the first Img2Img stage.

    style_2
        Style reference image(s) for the second Img2Img stage.

    Output
    ------
    The group output corresponds to the internal node ``img_2``.

    Parameters
    ----------
    name : str
        Name of the NodeGroup.

    img1_spec : SpecInput
        Configuration for the first Img2Img stage.

    img2_spec : SpecInput
        Configuration for the second Img2Img stage.

    style1_scale : IpAdapterScale
        IP-Adapter scale configuration for the first stage.

    style2_scale : IpAdapterScale
        IP-Adapter scale configuration for the second stage.

    depth_detect_resolution : int, optional
        Native short-side resolution used when computing the depth map.

    depth_conditioning_scale_1 : float, optional
        ControlNet conditioning scale for stage 1.

    depth_conditioning_scale_2 : float, optional
        ControlNet conditioning scale for stage 2.

    Notes
    -----
    - The depth map is computed once from ``in_image`` and reused in both stages.
    - This keeps structural guidance stable across the full two-stage refinement.
    - Both stages reuse the same generation prompt.
    - IP-Adapter and ControlNet are attached declaratively through the node
      registries exposed by ``Img2Img``.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_1 = Tap(name='style_1')
        tap_style_2 = Tap(name='style_2')

        # -------------------
        # Shared structural conditioning
        # -------------------
        depth = ImgAuxMap(
            name='depth',
            spec={
                'processor': 'depth_midas',
                'detect_resolution': depth_detect_resolution,
            },
        )

        tap_image >> depth

        # -------------------
        # Stage 1
        # -------------------
        img_1 = Img2Img(
            name='img_1',
            spec=img1_spec,
        )

        tap_image >> img_1
        tap_prompt >> img_1.prompt()

        tap_style_1 >> img_1.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style1_scale,
            key='style_1',
        )

        depth >> img_1.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale_1,
            key='depth',
        )

        # -------------------
        # Stage 2
        # -------------------
        img_2 = Img2Img(
            name='img_2',
            spec=img2_spec,
        )

        img_1 >> img_2
        tap_prompt >> img_2.prompt()

        tap_style_2 >> img_2.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style2_scale,
            key='style_2',
        )

        depth >> img_2.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale_2,
            key='depth',
        )

        # -------------------
        # Ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style_1,
            tap_style_2,
        )

    return g


def refine_region_group(
    name: str,
    *,
    refine_spec: SpecInput,
    stack_spec: SpecInput = {},
    bbox: tuple[int, ...],
    bbox_format: str = 'xyxy',
    style_scale: IpAdapterScale,
    layer_feather: int | str = 30,
    layer_corner_radius: int | str = 50,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines a manually selected rectangular region.

    The group crops a fixed rectangular region from the input image, refines that
    crop with Img2Img using an IP-Adapter style reference, and overlays the refined
    crop back onto the original image using the crop metadata emitted by BoxCrop.

    Ports
    -----
    in_image
        Base image used both as stack background and crop source.

    in_style
        Style reference image wired into the IP-Adapter slot of the internal
        Img2Img node.

    in_prompt
        Optional prompt payload wired into the internal Img2Img prompt sink.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for the internal Img2Img refinement node.

    stack_spec : SpecInput, optional
        Spec for ImageStack.

    bbox : tuple[int, ...]
        Manual crop box values interpreted according to ``bbox_format``.

    bbox_format : {'xyxy', 'xywh', 'xyl'}, optional
        Format of ``bbox``. Default is ``'xyxy'``.

    style_scale : IpAdapterScale
        IP-Adapter scale for the style reference.

    layer_feather : int or str, optional
        Feather applied when compositing the refined crop.

    layer_corner_radius : int or str, optional
        Corner radius applied to the refined crop mask.

    Input Ports
    -----
    in_image : Tap
        Base image used both as:
        1) background layer for ``ImageStack``;
        2) source image for ``BoxCrop``.

    in_style : Tap
        Style reference image, or reference image list, wired into the internal
        IP-Adapter slot of ``refine_region``.

    in_prompt : Tap
        Optional prompt payload wired into the prompt sink of ``refine_region``.
        The port uses ``strict=False``, so the group can run without an external
        prompt input when ``refine_spec`` already provides the prompt.

    Returns
    -------
    NodeGroup
        The constructed region-refinement group.

    Notes
    -----
    This group is the manual-box counterpart of ``refine_face_group``: it does
    not run semantic detection and relies entirely on the provided coordinates.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style = Tap(name='in_style')

        # -------------------
        # Internal nodes
        # -------------------
        crop_region = BoxCrop(
            name='crop_region',
            spec={
                'params': {
                    'bbox_format': bbox_format,
                    'bbox': list(bbox),
                },
            },
        )

        refine_region = Img2Img(
            name='refine_region',
            spec=refine_spec,
        )

        style_sink = refine_region.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=style_scale,
            key='style',
        )

        stack = ImageStack(
            name='out',
            spec=stack_spec,
        )

        # -------------------
        # Wiring
        # -------------------

        # 1) Base image as background.
        tap_image >> stack.image(0)

        # 2) Crop manual region from base image.
        tap_image >> crop_region

        # 3) Refine cropped region.
        crop_region >> refine_region
        tap_prompt >> refine_region.prompt()
        tap_style >> style_sink

        # 4) Overlay refined crop using BoxCrop transform metadata.
        layer1 = stack.image(
            1,
            position='center',
            feather=layer_feather,
            corner_radius=layer_corner_radius,
        )

        refine_region >> layer1
        crop_region >> layer1.transform()

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            tap_style,
        )

    return g


def refine_region_multi_style_group(
    name: str,
    *,
    refine_spec: SpecInput,
    stack_spec: SpecInput = {},
    bbox: tuple[int, ...],
    bbox_format: str = 'xyxy',
    adapter_weight_names: list[str] | None = None,
    adapter_scales: list[float | dict] | None = None,
    layer_feather: int | str = 30,
    layer_corner_radius: int | str = 50,
) -> NodeGroup:
    """
    Create a reusable NodeGroup that refines a manually selected rectangular
    region using zero or more IP-Adapter references.

    The group crops a fixed rectangular region from the input image, refines that
    crop with ``Img2Img`` using zero or more IP-Adapter slots, and overlays the
    refined crop back onto the original image using the crop metadata emitted by
    ``BoxCrop``.

    Ports
    -----
    in_image
        Base image used both as stack background and crop source.

    style_1, style_2, ...
        Style reference ports created from ``adapter_weight_names``.

    in_prompt
        Optional prompt payload wired into the internal ``Img2Img`` prompt sink.

    Parameters
    ----------
    name : str
        NodeGroup name.

    refine_spec : SpecInput
        Spec for the internal ``Img2Img`` refinement node.

    stack_spec : SpecInput, optional
        Spec for ``ImageStack``.

    bbox : tuple[int, ...]
        Manual crop box values interpreted according to ``bbox_format``.

    bbox_format : {'xyxy', 'xywh', 'xyl'}, optional
        Format of ``bbox``. Default is ``'xyxy'``.

    adapter_weight_names : list[str] or None, optional
        IP-Adapter weight names. When ``None``, no style adapters are added.

    adapter_scales : list[float | dict] or None, optional
        IP-Adapter scales aligned with ``adapter_weight_names``.

    layer_feather : int or str, optional
        Feather applied when compositing the refined crop.

    layer_corner_radius : int or str, optional
        Corner radius applied to the refined crop mask.

    Input Ports
    -----------
    in_image : Tap
        Base image used both as:
        1) background layer for ``ImageStack``;
        2) source image for ``BoxCrop``.

    style_1, style_2, ... : Tap
        Style reference images, or reference image lists, wired into dynamic
        IP-Adapter slots of ``refine_region``.

    in_prompt : Tap
        Optional prompt payload wired into the prompt sink of ``refine_region``.
        The port uses ``strict=False``, so the group can run without an external
        prompt input when ``refine_spec`` already provides the prompt.

    Returns
    -------
    NodeGroup
        The constructed region-refinement group.

    Notes
    -----
    This group does not run semantic detection and relies entirely on the
    provided coordinates.
    """
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

        # -------------------
        # Internal nodes
        # -------------------
        crop_region = BoxCrop(
            name='crop_region',
            spec={
                'params': {
                    'bbox_format': bbox_format,
                    'bbox': list(bbox),
                },
            },
        )

        refine_region = Img2Img(
            name='refine_region',
            spec=refine_spec,
        )

        adapter_sinks = []
        for style_idx, (weight_name, scale) in enumerate(zip(
            adapter_weight_names or [],
            adapter_scales or [],
        )):
            adapter_sinks.append(refine_region.ip_adapter.add(
                'h94/IP-Adapter',
                subfolder='sdxl_models',
                weight_name=weight_name,
                scale=scale,
                key=f'style_{style_idx + 1}',
            ))

        stack = ImageStack(
            name='out',
            spec=stack_spec,
        )

        # -------------------
        # Wiring
        # -------------------

        # 1) Base image as background.
        tap_image >> stack.image(0)

        # 2) Crop manual region from base image.
        tap_image >> crop_region

        # 3) Refine cropped region.
        crop_region >> refine_region
        tap_prompt >> refine_region.prompt()
        for tap_style, adapter_sink in zip(tap_styles, adapter_sinks):
            tap_style >> adapter_sink

        # 4) Overlay refined crop using BoxCrop transform metadata.
        layer1 = stack.image(
            1,
            position='center',
            feather=layer_feather,
            corner_radius=layer_corner_radius,
        )

        refine_region >> layer1
        crop_region >> layer1.transform()

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_prompt,
            *tap_styles,
        )

    return g
