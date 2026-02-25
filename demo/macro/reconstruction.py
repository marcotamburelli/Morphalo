from stability.dag import NodeGroup
from stability.nodes import Tap
from stability.nodes.common.config_resolve import SpecInput
from stability.nodes.preprocess import ImgAuxMap
from stability.nodes.txt2img import Txt2Img


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
