from morphalo.dag import NodeGroup
from morphalo.nodes import Img2Img, Inpaint, Tap, Txt2Img
from morphalo.nodes.common.config_resolve import SpecInput
from morphalo.nodes.preprocess import ImgAuxMap


def reconstruct_from_geometry_group(
    name: str,
    *,
    txt2img_spec: SpecInput,
    canny_conditioning_scale: float,
    depth_conditioning_scale: float,
    canny_detect_long_side: int = 1024,
    depth_detect_long_side: int = 1024,
) -> NodeGroup:
    """
    Reconstruct a photorealistic image from geometric cues extracted from a source image.

    This NodeGroup implements a geometry-driven generation pattern:

    - A source image (often stylized / non-photorealistic) is used only to extract
    geometric guidance signals:
    - Canny edges (fine 2D contours)
    - Depth map (coarse 3D structure)
    - These signals are fed into a Txt2Img node through two ControlNet branches.
    - A Prompt node (wired from outside the group) controls the semantic content,
    style and realism, while ControlNet constrains geometry.

    The goal is to "rebuild" an image that respects the pose/shape/contours of the
    source but is rendered in a more realistic (or otherwise target) domain.

    Parameters
    ----------
    name : str
        Group name (scope prefix) used for hierarchical node ids.

    txt2img_spec : SpecInput
        Txt2Img specification passed to the internal Txt2Img node. This should
        include your model/pipeline configuration (e.g. SDXL), scheduler, and
        generation parameters (cfg, steps, resolution, etc.).

    canny_conditioning_scale : float
        ControlNet conditioning strength for the Canny branch.

        This parameter governs how strongly the generated image adheres to the
        extracted 2D edges. It is highly dependent on the input image:

        - Higher values: stronger preservation of fine contours and silhouettes,
        but increased risk of "over-constrained" generations (rigid, less natural,
        artifacts around edges, reduced ability to improve details).
        - Lower values: more freedom for the model to reinterpret shapes, which can
        improve realism but may drift from the original facial/body proportions.

    depth_conditioning_scale : float
        ControlNet conditioning strength for the Depth branch.

        This parameter governs how strongly the generated image adheres to the
        extracted coarse 3D structure (volumes, perspective, pose). It is also
        input-dependent:

        - Higher values: stronger adherence to pose and depth geometry, but can
        lock in errors from the depth estimator (especially for stylized images).
        - Lower values: more freedom to reshape 3D structure according to the prompt,
        which may improve aesthetics but risks changing the intended pose.

    canny_detect_long_side : int, optional
        Target long-side resolution used by the Canny preprocessor. Higher values
        preserve finer edge detail but may also capture unwanted micro-edges/noise.

    depth_detect_long_side : int, optional
        Target long-side resolution used by the depth preprocessor (e.g. MiDaS).
        Higher values can improve structural fidelity but may amplify estimator noise
        or produce unstable depth boundaries on stylized images.

    Returns
    -------
    NodeGroup
        A NodeGroup exposing at least two ports:

        - 'in_image': the source image providing geometric cues (canny + depth)
        - 'prompt'  : the prompt payload fed to Txt2Img.prompt()

        The group output is the internal Txt2Img node (the generated image).

    Notes
    -----
    - This macro intentionally provides no defaults for conditioning scales:
    there is no universally "optimal" value. The correct setting depends on the
    source image domain (photo vs stylized), its noise level, and how strictly
    you want to preserve geometry vs allow reinterpretation.
    - This macro does not include FaceID or head/face refinement; those are expected
    to be composed downstream using dedicated macros.
    - If the input image is already photorealistic, lowering both scales often
    avoids overfitting to the preprocessing signals and yields cleaner results.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='prompt')

        # -------------------
        # Geometry extraction
        # -------------------
        canny = ImgAuxMap(
            name='canny',
            spec={
                'processor': 'canny',
                'detect_long_side': canny_detect_long_side,
            },
        )
        depth = ImgAuxMap(
            name='depth_midas',
            spec={
                'processor': 'depth_midas',
                'detect_long_side': depth_detect_long_side,
            },
        )

        # -------------------
        # Txt2Img with ControlNets
        # -------------------
        out = Txt2Img(
            name='out',
            spec=txt2img_spec,
        )

        canny_sink = out.controlnet.add(
            'diffusers/controlnet-canny-sdxl-1.0',
            conditioning_scale=canny_conditioning_scale,
            key='canny',
        )
        depth_sink = out.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale,
            key='depth',
        )

        # -------------------
        # Wiring
        # -------------------
        tap_image >> canny >> canny_sink
        tap_image >> depth >> depth_sink
        tap_prompt >> out.prompt()

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(tap_image, tap_prompt)

    return g


def reconstruct_from_geometry_img2img_group(
    name: str,
    *,
    img2img_spec: SpecInput,
    canny_conditioning_scale: float,
    depth_conditioning_scale: float,
    adapter_scale: float,
    canny_detect_long_side: int = 1024,
    depth_detect_long_side: int = 1024,
) -> NodeGroup:
    """
    Reconstruct a more realistic image from geometric cues while preserving the source via Img2Img.

    This NodeGroup implements a geometry-driven Img2Img pattern:

    - A source image is used both as:
      - the Img2Img starting point (content/color/overall composition prior), and
      - the extractor source for geometric guidance signals:
        - Canny edges (fine 2D contours)
        - Depth map (coarse 3D structure)
    - These signals are fed into an Img2Img node through two ControlNet branches.
    - A style image is fed into Img2Img through an IP-Adapter branch.
    - A prompt (wired from outside the group) controls semantic content,
      style and realism, while ControlNet constrains geometry.

    Ports
    -----
    - 'in_image' (Tap)
        Source image used as Img2Img init and as the source for canny/depth maps.
    - 'in_prompt' (Tap)
        Prompt payload wired to Img2Img.prompt().
    - 'in_style' (Tap)
        Style reference image wired to the IP-Adapter sink.

    Parameters
    ----------
    name : str
        Group name (scope prefix) used for hierarchical node ids.

    img2img_spec : SpecInput
        Img2Img specification passed to the internal Img2Img node. This should
        include your model/pipeline configuration (e.g. SDXL), scheduler, and
        generation parameters (strength, cfg, steps, resolution, etc.).

    canny_conditioning_scale : float
        ControlNet conditioning strength for the Canny branch.

    depth_conditioning_scale : float
        ControlNet conditioning strength for the Depth branch.

    adapter_scale : float
        IP-Adapter scale controlling how strongly the output borrows style cues
        from the provided style image. Higher values yield stronger style transfer
        but may harm identity/details.

    canny_detect_long_side : int, optional
        Target long-side resolution used by the Canny preprocessor.

    depth_detect_long_side : int, optional
        Target long-side resolution used by the depth preprocessor.

    Returns
    -------
    NodeGroup
        The constructed group. The group output is the internal Img2Img node
        ('out'), producing the generated image.

    Notes
    -----
    - This macro uses the same input image for Img2Img init and for extracting
      ControlNet conditioning maps, ensuring maximal geometric consistency.
    - FaceID or head/face refinement is expected to be composed downstream using
      dedicated macros.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_image = Tap(name='in_style')

        # -------------------
        # Geometry extraction
        # -------------------
        canny = ImgAuxMap(
            name='canny',
            spec={
                'processor': 'canny',
                'detect_long_side': canny_detect_long_side,
            },
        )
        depth = ImgAuxMap(
            name='depth_midas',
            spec={
                'processor': 'depth_midas',
                'detect_long_side': depth_detect_long_side,
            },
        )

        # -------------------
        # Img2Img with ControlNets + IP-Adapter
        # -------------------
        out = Img2Img(
            name='out',
            spec=img2img_spec,
        )

        canny_sink = out.controlnet.add(
            'diffusers/controlnet-canny-sdxl-1.0',
            conditioning_scale=canny_conditioning_scale,
            key='canny',
        )
        depth_sink = out.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale,
            key='depth',
        )

        style_sink = out.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=adapter_scale,
            key='style',
        )

        # -------------------
        # Wiring
        # -------------------
        # Same image is used as Img2Img init and as source for aux maps.
        tap_image >> out
        tap_image >> canny >> canny_sink
        tap_image >> depth >> depth_sink

        # Prompt drives the semantic target and realism.
        tap_prompt >> out.prompt()

        # Style reference guides appearance through IP-Adapter.
        tap_style_image >> style_sink

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(tap_image, tap_prompt, tap_style_image)

    return g


def reconstruct_masked_region_from_geometry_inpaint_group(
    name: str,
    *,
    inpaint_spec: SpecInput,
    canny_conditioning_scale: float,
    depth_conditioning_scale: float,
    adapter_scale: float,
    canny_detect_long_side: int = 1024,
    depth_detect_long_side: int = 1024,
    ip_adapter_model_id: str = 'h94/IP-Adapter',
    ip_adapter_subfolder: str = 'sdxl_models',
    ip_adapter_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
) -> NodeGroup:
    """
    Reconstruct a masked region in a base image using external geometric and
    stylistic references.

    This NodeGroup implements a reference-guided inpainting pattern:

    - ``in_image`` provides the base image to be edited.
    - ``in_mask`` defines the editable region.
    - ``in_geometry_image`` provides geometric guidance:
      - Canny edges for fine 2D contours.
      - Depth map for coarse 3D structure.
    - ``in_style`` provides the IP-Adapter reference image.
    - ``in_prompt`` provides semantic and stylistic text conditioning.

    The intent is to replace or reinterpret only the masked region of the base
    image while borrowing shape/pose/layout cues from a separate geometry image
    and visual appearance cues from a separate style image.

    Typical use case
    ----------------
    Given two partially aligned images:

    - image A: target/base image;
    - image B: reference image containing the desired shape or structure;

    this macro can inpaint a masked area of image A using Canny/depth extracted
    from image B, while optionally using a third image as style reference.

    Pipeline
    --------
    Geometry preprocessing:
        - ``canny`` is computed from ``in_geometry_image``.
        - ``depth_midas`` is computed from ``in_geometry_image``.

    Inpainting:
        - ``out`` takes ``in_image`` as base image.
        - ``out`` takes ``in_mask`` as inpainting mask.
        - ``out`` uses ``in_prompt`` as prompt conditioning.
        - ``out`` uses Canny ControlNet from ``in_geometry_image``.
        - ``out`` uses Depth ControlNet from ``in_geometry_image``.
        - ``out`` uses IP-Adapter reference from ``in_style``.

    Ports
    -----
    in_image
        Base image to be edited by inpainting.

    in_geometry_image
        Reference image used only to compute geometric conditioning maps
        through Canny and depth preprocessing.

    in_mask
        Inpainting mask. White areas are repainted; black areas are preserved.

    in_prompt
        Prompt payload wired to ``Inpaint.prompt()``.

    in_style
        Style reference image or images wired to the IP-Adapter sink.

    Output
    ------
    The group output corresponds to the internal node ``out``.

    Parameters
    ----------
    name : str
        NodeGroup name used as scope prefix for internal node ids.

    inpaint_spec : SpecInput
        Configuration for the internal :class:`Inpaint` node. This should include
        model/runtime settings and generation parameters such as strength, CFG,
        steps, scheduler, and output resolution.

    canny_conditioning_scale : float
        ControlNet conditioning strength for the Canny branch.

    depth_conditioning_scale : float
        ControlNet conditioning strength for the Depth branch.

    adapter_scale : float
        IP-Adapter scale controlling how strongly ``in_style`` influences the
        generated region.

    canny_detect_long_side : int, optional
        Long-side resolution used by the Canny preprocessor.

    depth_detect_long_side : int, optional
        Long-side resolution used by the depth preprocessor.

    ip_adapter_model_id : str, optional
        Hugging Face repository identifier or local path for the IP-Adapter model.

    ip_adapter_subfolder : str, optional
        Repository subfolder containing the IP-Adapter weights.

    ip_adapter_weight_name : str, optional
        IP-Adapter weight file name.

    Returns
    -------
    NodeGroup
        Constructed group. The output node is the internal ``Inpaint`` node
        named ``out``.

    Notes
    -----
    - Geometry and style are intentionally separated:
      ``in_geometry_image`` controls Canny/depth, while ``in_style`` controls
      IP-Adapter appearance guidance.
    - The base image is not used to compute ControlNet maps in this macro.
    - The mask is external, so this macro can be composed with ``SubjectCrop``,
      ``FileImage``, or custom mask-producing nodes.
    - The IP-Adapter weight is configurable; by default this uses
      ``ip-adapter_sdxl_vit-h.bin``.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_geometry_image = Tap(name='in_geometry_image')
        tap_mask = Tap(name='in_mask')
        tap_prompt = Tap(name='in_prompt', strict=False)
        tap_style_image = Tap(name='in_style')

        # -------------------
        # Geometry extraction
        # -------------------
        canny = ImgAuxMap(
            name='canny',
            spec={
                'processor': 'canny',
                'detect_long_side': canny_detect_long_side,
            },
        )

        depth = ImgAuxMap(
            name='depth_midas',
            spec={
                'processor': 'depth_midas',
                'detect_long_side': depth_detect_long_side,
            },
        )

        tap_geometry_image >> canny
        tap_geometry_image >> depth

        # -------------------
        # Inpaint with ControlNets + IP-Adapter
        # -------------------
        out = Inpaint(
            name='out',
            spec=inpaint_spec,
        )

        tap_image >> out
        tap_mask >> out.mask()
        tap_prompt >> out.prompt()

        canny >> out.controlnet.add(
            'diffusers/controlnet-canny-sdxl-1.0',
            conditioning_scale=canny_conditioning_scale,
            key='canny',
        )

        depth >> out.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale,
            key='depth',
        )

        tap_style_image >> out.ip_adapter.add(
            ip_adapter_model_id,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
            scale=adapter_scale,
            key='style',
        )

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(
            tap_image,
            tap_geometry_image,
            tap_mask,
            tap_prompt,
            tap_style_image,
        )

    return g
