from morphalo.dag import NodeGroup
from morphalo.nodes import Tap
from morphalo.nodes.common.config_resolve import SpecInput
from morphalo.nodes.evaluate import PersonScorer, PromptScorer
from morphalo.nodes.img2img import Img2Img
from morphalo.nodes.preprocess import ImgAuxMap, SubjectCrop
from morphalo.nodes.wiring.ip_adapter import IpAdapterScale


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
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
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
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                },
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
    canny_detect_long_side: int = 1024,
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

    canny_detect_long_side : int, optional
        Long-side resolution used when computing the Canny map.

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
                'detect_long_side': canny_detect_long_side,
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
