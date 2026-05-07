from __future__ import annotations

from typing import Optional

from morphalo.dag import NodeGroup
from morphalo.nodes import Img2Img, Tap, Txt2Img
from morphalo.nodes.common.config_resolve import SpecInput
from morphalo.nodes.preprocess import ImageStack, ImgAuxMap, SubjectCrop
from morphalo.nodes.preprocess.image_stack import ResizeMode
from morphalo.nodes.wiring.ip_adapter import IpAdapterScale


def cutout_stack_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: Optional[ResizeMode] = None,
) -> NodeGroup:
    """
    Compose a cut-out subject over a background, then harmonize with Img2Img.

    Pattern
    -------
    foreground -> SubjectCrop(full_frame=True) -> ImageStack.layer(1)
    background -------------------------------> ImageStack.layer(0)
    ImageStack -> Img2Img(out)  (+ prompt)

    Ports
    -----
    - 'foreground' (Tap)
        Image containing the subject to extract.
    - 'background' (Tap)
        Background image to place the subject onto.
    - 'prompt' (Tap)
        Prompt payload wired to the final Img2Img pass.

    Output
    ------
    The group output is the internal Img2Img node named 'out'.

    Parameters
    ----------
    name : str
        Group name (scope prefix).
    out_spec : SpecInput
        Spec for the final Img2Img harmonization pass.
    fg_layer_feather : int | str, optional
        Feather applied to the subject layer in the stack.
    fg_layer_position : str, optional
        Subject placement anchor/position for stack.image(...).
    fg_layer_resize : ResizeMode, optional
        Optional resize parameter. See `stack.image()` for more details.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        fg = Tap(name='foreground')
        bg = Tap(name='background')
        prompt = Tap(name='prompt', strict=False)

        # -------------------
        # Subject cut-out
        # -------------------
        crop = SubjectCrop(
            name='cutout',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
                'params': {
                },
            },
        )

        # -------------------
        # Stack (composite)
        # -------------------
        stack = ImageStack(
            name='stack',
            spec={
                'params': {},
            },
        )

        # Background at layer 0
        bg >> stack.image(0)

        # Foreground cut-out at layer 1
        fg >> crop
        fg_layer = stack.image(
            1,
            position=fg_layer_position,
            resize=fg_layer_resize,
            feather=fg_layer_feather,
        )
        crop >> fg_layer

        # -------------------
        # Harmonize pass
        # -------------------
        out = Img2Img(
            name='out',
            spec=out_spec,
        )

        prompt >> out.prompt()
        stack >> out

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(fg, bg, prompt)

    return g


def cutout_stack_canny_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: Optional[ResizeMode] = None,
    canny_conditioning_scale: float = 0.7,
) -> NodeGroup:
    """
    Compose a trimmed cut-out subject over a background, derive a Canny map
    from the composite, then harmonize with Img2Img using Canny ControlNet.

    Pattern
    -------
    foreground -> SubjectCrop(trim) -> ImageStack.layer(1)
    background -------------------> ImageStack.layer(0)
    ImageStack -> ImgAuxMap(canny) -> Img2Img.controlnet
    ImageStack -------------------> Img2Img(default)
    prompt -----------------------> Img2Img(prompt)

    Ports
    -----
    - 'foreground' (Tap)
        Image containing the subject to extract.
    - 'background' (Tap)
        Background image used for the composition.
    - 'prompt' (Tap)
        Prompt payload wired to the final Img2Img pass.

    Output
    ------
    The group output is the internal Img2Img node named 'out'.

    Parameters
    ----------
    name : str
        Group name (scope prefix).
    out_spec : SpecInput
        Spec for the final Img2Img harmonization pass.
    fg_layer_feather : int | str, optional
        Feather applied to the subject layer in the stack.
    fg_layer_position : str, optional
        Subject placement anchor/position for ``stack.image(...)``.
    fg_layer_resize : ResizeMode, optional
        Optional resize parameter for the subject layer.
    canny_conditioning_scale : float, optional
        Conditioning scale used for the Canny ControlNet.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        fg = Tap(name='foreground')
        bg = Tap(name='background')
        prompt = Tap(name='prompt', strict=False)

        # -------------------
        # Subject cut-out
        # -------------------
        crop = SubjectCrop(
            name='cutout',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
                'params': {
                    'target': 'person',
                    'mode': 'default',
                    'crop_mode': 'trim',
                },
            },
        )

        # -------------------
        # Stack (composite)
        # -------------------
        stack = ImageStack(
            name='stack',
            spec={
                'params': {},
            },
        )

        bg >> stack.image(0)

        fg >> crop
        fg_layer = stack.image(
            1,
            position=fg_layer_position,
            resize=fg_layer_resize,
            feather=fg_layer_feather,
        )
        crop >> fg_layer

        # -------------------
        # Canny control image
        # -------------------
        canny = ImgAuxMap(
            name='canny',
            spec={
                'processor': 'canny',
                'detect_long_side': 1024,
            },
        )

        stack >> canny

        # -------------------
        # Harmonize pass
        # -------------------
        out = Txt2Img(
            name='out',
            spec=out_spec,
        )

        prompt >> out.prompt()

        canny >> out.controlnet.add(
            'diffusers/controlnet-canny-sdxl-1.0',
            conditioning_scale=canny_conditioning_scale,
            key='canny',
        )

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(fg, bg, prompt)

    return g


def cutout_stack_depth_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: Optional[ResizeMode] = None,
    depth_conditioning_scale: float = 0.7,
) -> NodeGroup:
    """
    Compose a trimmed cut-out subject over a background, derive a depth map
    from the composite, then harmonize with Img2Img using Depth ControlNet.

    Pattern
    -------
    foreground -> SubjectCrop(trim) -> ImageStack.layer(1)
    background -------------------> ImageStack.layer(0)
    ImageStack -> ImgAuxMap(depth) -> Img2Img.controlnet
    ImageStack -------------------> Img2Img(default)
    prompt -----------------------> Img2Img(prompt)

    Ports
    -----
    - 'foreground' (Tap)
        Image containing the subject to extract.
    - 'background' (Tap)
        Background image used for the composition.
    - 'prompt' (Tap)
        Prompt payload wired to the final Img2Img pass.

    Output
    ------
    The group output is the internal Img2Img node named 'out'.

    Parameters
    ----------
    name : str
        Group name (scope prefix).
    out_spec : SpecInput
        Spec for the final Img2Img harmonization pass.
    fg_layer_feather : int | str, optional
        Feather applied to the subject layer in the stack.
    fg_layer_position : str, optional
        Subject placement anchor/position for ``stack.image(...)``.
    fg_layer_resize : ResizeMode, optional
        Optional resize parameter for the subject layer.
    depth_conditioning_scale : float, optional
        Conditioning scale used for the Depth ControlNet.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        fg = Tap(name='foreground')
        bg = Tap(name='background')
        prompt = Tap(name='prompt', strict=False)

        # -------------------
        # Subject cut-out
        # -------------------
        crop = SubjectCrop(
            name='cutout',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
                'params': {
                    'target': 'person',
                    'mode': 'default',
                    'crop_mode': 'trim',
                },
            },
        )

        # -------------------
        # Stack (composite)
        # -------------------
        stack = ImageStack(
            name='stack',
            spec={
                'params': {},
            },
        )

        bg >> stack.image(0)

        fg >> crop
        fg_layer = stack.image(
            1,
            position=fg_layer_position,
            resize=fg_layer_resize,
            feather=fg_layer_feather,
        )
        crop >> fg_layer

        # -------------------
        # Depth control image
        # -------------------
        depth = ImgAuxMap(
            name='depth',
            spec={
                'processor': 'depth_midas',
                'detect_long_side': 1024,
            },
        )

        stack >> depth

        # -------------------
        # Harmonize pass
        # -------------------
        out = Txt2Img(
            name='out',
            spec=out_spec,
        )

        prompt >> out.prompt()

        depth >> out.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale,
            key='depth',
        )

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(fg, bg, prompt)

    return g


def cutout_stack_pose_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: Optional[ResizeMode] = None,
    pose_conditioning_scale: float = 0.9,
) -> NodeGroup:
    """
    Compose a trimmed cut-out subject over a background, derive a depth map
    from the composite, then harmonize with Img2Img using Depth ControlNet.

    Pattern
    -------
    foreground -> SubjectCrop(trim) -> ImageStack.layer(1)
    ImageStack -> ImgAuxMap(depth) -> Img2Img.controlnet
    ImageStack -------------------> Img2Img(default)
    prompt -----------------------> Img2Img(prompt)

    Ports
    -----
    - 'foreground' (Tap)
        Image containing the subject to extract.
    - 'prompt' (Tap)
        Prompt payload wired to the final Img2Img pass.

    Output
    ------
    The group output is the internal Img2Img node named 'out'.

    Parameters
    ----------
    name : str
        Group name (scope prefix).
    out_spec : SpecInput
        Spec for the final Img2Img harmonization pass.
    fg_layer_feather : int | str, optional
        Feather applied to the subject layer in the stack.
    fg_layer_position : str, optional
        Subject placement anchor/position for ``stack.image(...)``.
    fg_layer_resize : ResizeMode, optional
        Optional resize parameter for the subject layer.
    depth_conditioning_scale : float, optional
        Conditioning scale used for the Depth ControlNet.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        fg = Tap(name='foreground')
        prompt = Tap(name='prompt', strict=False)

        # -------------------
        # Subject cut-out
        # -------------------
        crop = SubjectCrop(
            name='cutout',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
                'params': {
                    'target': 'person',
                    'mode': 'default',
                    'crop_mode': 'trim',
                },
            },
        )

        # -------------------
        # Stack (composite)
        # -------------------
        stack = ImageStack(
            name='stack',
            spec={
                'params': {},
            },
        )

        fg >> crop
        fg_layer = stack.image(
            1,
            position=fg_layer_position,
            resize=fg_layer_resize,
            feather=fg_layer_feather,
        )
        crop >> fg_layer

        # -------------------
        # Depth control image
        # -------------------
        pose = ImgAuxMap(
            name='pose',
            spec={
                'processor': 'openpose',
                'detect_long_side': 1024,
            },
        )

        stack >> pose

        # -------------------
        # Harmonize pass
        # -------------------
        out = Txt2Img(
            name='out',
            spec=out_spec,
        )

        prompt >> out.prompt()

        pose >> out.t2i_adapter.add(
            'TencentARC/t2i-adapter-openpose-sdxl-1.0',
            conditioning_scale=pose_conditioning_scale,
            key='pose',
        )

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(fg, prompt)

    return g


def cutout_stack_ip_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    style_scale: IpAdapterScale,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: Optional[ResizeMode] = None,
    ip_adapter_model_id: str = 'h94/IP-Adapter',
    ip_adapter_subfolder: str = 'sdxl_models',
    ip_adapter_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
) -> NodeGroup:
    """
    Compose a trimmed cut-out subject over a background, then harmonize the
    composite with Img2Img using only IP-Adapter style conditioning.

    This NodeGroup implements a simple compositing + harmonization pattern:

    - a foreground image is cut out with ``SubjectCrop``;
    - the cut-out is placed over a background using ``ImageStack``;
    - the stacked composite is passed as the init image to ``Img2Img``;
    - an external style image is attached through IP-Adapter;
    - an optional prompt controls semantic/style refinement during the final pass.

    Pattern
    -------
    foreground -> SubjectCrop(trim) -> ImageStack.layer(1)
    background -------------------> ImageStack.layer(0)
    ImageStack -------------------> Img2Img(default)
    prompt -----------------------> Img2Img(prompt)
    style ------------------------> Img2Img.ip_adapter

    Ports
    -----
    foreground
        Image containing the subject to extract.

    background
        Background image used for the composition.

    prompt
        Prompt payload wired to the final Img2Img pass.

    style
        IP-Adapter reference image or images used to guide the final Img2Img pass.

    Output
    ------
    The group output corresponds to the internal node ``out``.

    Parameters
    ----------
    name : str
        Group name.

    out_spec : SpecInput
        Configuration for the final Img2Img harmonization pass.

    style_scale : IpAdapterScale
        IP-Adapter scale configuration for the style reference.

    fg_layer_feather : int | str, optional
        Feather applied to the subject layer in the stack.

    fg_layer_position : str, optional
        Subject placement anchor/position for ``stack.image(...)``.

    fg_layer_resize : ResizeMode, optional
        Optional resize parameter for the subject layer.

    ip_adapter_model_id : str, optional
        Hugging Face repository identifier for the IP-Adapter model.

    ip_adapter_subfolder : str, optional
        Repository subfolder containing the IP-Adapter weights.

    ip_adapter_weight_name : str, optional
        IP-Adapter weight file name.

    Notes
    -----
    - This macro does not use ControlNet.
    - The final pass is a true Img2Img pass: the stacked composite is used as
      the init image.
    - Structural fidelity comes only from the init image itself, while style
      guidance is provided by IP-Adapter.
    - The IP-Adapter is configurable through model id, subfolder, and weight name.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        fg = Tap(name='foreground')
        bg = Tap(name='background')
        prompt = Tap(name='prompt', strict=False)
        style = Tap(name='style')

        # -------------------
        # Subject cut-out
        # -------------------
        crop = SubjectCrop(
            name='cutout',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
                'params': {
                    'target': 'person',
                    'mode': 'default',
                    'crop_mode': 'trim',
                },
            },
        )

        # -------------------
        # Stack (composite)
        # -------------------
        stack = ImageStack(
            name='stack',
            spec={
                'params': {},
            },
        )

        bg >> stack.image(0)

        fg >> crop
        fg_layer = stack.image(
            1,
            position=fg_layer_position,
            resize=fg_layer_resize,
            feather=fg_layer_feather,
        )
        crop >> fg_layer

        # -------------------
        # Harmonize pass
        # -------------------
        out = Img2Img(
            name='out',
            spec=out_spec,
        )

        stack >> out
        prompt >> out.prompt()

        style >> out.ip_adapter.add(
            ip_adapter_model_id,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
            scale=style_scale,
            key='style',
        )

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(fg, bg, prompt, style)

    return g


def cutout_stack_depth_ip_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    style_scale: IpAdapterScale,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: Optional[ResizeMode] = None,
    depth_detect_long_side: int = 1024,
    depth_conditioning_scale: float = 0.7,
    ip_adapter_model_id: str = 'h94/IP-Adapter',
    ip_adapter_subfolder: str = 'sdxl_models',
    ip_adapter_weight_name: str = 'ip-adapter_sdxl_vit-h.bin',
) -> NodeGroup:
    """
    Compose a trimmed cut-out subject over a background, derive a depth map
    from the composite, then harmonize the composite with Img2Img using both
    Depth ControlNet and IP-Adapter style conditioning.

    Pattern
    -------
    foreground -> SubjectCrop(trim) -> ImageStack.layer(1)
    background -------------------> ImageStack.layer(0)
    ImageStack -> ImgAuxMap(depth) -> Img2Img.controlnet
    ImageStack -------------------> Img2Img(default)
    prompt -----------------------> Img2Img(prompt)
    style ------------------------> Img2Img.ip_adapter

    Ports
    -----
    - 'foreground' (Tap)
        Image containing the subject to extract.
    - 'background' (Tap)
        Background image used for the composition.
    - 'prompt' (Tap)
        Prompt payload wired to the final Img2Img pass.
    - 'style' (Tap)
        IP-Adapter reference image(s) used to guide the final Img2Img pass.

    Output
    ------
    The group output is the internal Img2Img node named 'out'.

    Parameters
    ----------
    name : str
        Group name.

    out_spec : SpecInput
        Configuration for the final Img2Img harmonization pass.

    style_scale : IpAdapterScale
        IP-Adapter scale configuration for the style reference.

    fg_layer_feather : int | str, optional
        Feather applied to the subject layer in the stack.

    fg_layer_position : str, optional
        Subject placement anchor/position for ``stack.image(...)``.

    fg_layer_resize : ResizeMode, optional
        Optional resize parameter for the subject layer.

    depth_detect_long_side : int, optional
        Long-side resolution used when computing the depth map.

    depth_conditioning_scale : float, optional
        Conditioning scale used for the Depth ControlNet.

    ip_adapter_model_id : str, optional
        Hugging Face repository identifier for the IP-Adapter model.

    ip_adapter_subfolder : str, optional
        Repository subfolder containing the IP-Adapter weights.

    ip_adapter_weight_name : str, optional
        IP-Adapter weight file name.

    Notes
    -----
    - The depth map is computed from the stacked composite, not from the
      original foreground or background alone.
    - The final pass is a real Img2Img pass: the stack is wired as the init image.
    - The IP-Adapter is configurable through model id, subfolder, and weight name.
    """
    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        fg = Tap(name='foreground')
        bg = Tap(name='background')
        prompt = Tap(name='prompt', strict=False)
        style = Tap(name='style')

        # -------------------
        # Subject cut-out
        # -------------------
        crop = SubjectCrop(
            name='cutout',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
                },
                'params': {
                    'target': 'person',
                    'mode': 'default',
                    'crop_mode': 'trim',
                },
            },
        )

        # -------------------
        # Stack (composite)
        # -------------------
        stack = ImageStack(
            name='stack',
            spec={
                'params': {},
            },
        )

        bg >> stack.image(0)

        fg >> crop
        fg_layer = stack.image(
            1,
            position=fg_layer_position,
            resize=fg_layer_resize,
            feather=fg_layer_feather,
        )
        crop >> fg_layer

        # -------------------
        # Depth control image
        # -------------------
        depth = ImgAuxMap(
            name='depth',
            spec={
                'processor': 'depth_midas',
                'detect_long_side': depth_detect_long_side,
            },
        )

        stack >> depth

        # -------------------
        # Harmonize pass
        # -------------------
        out = Img2Img(
            name='out',
            spec=out_spec,
        )

        stack >> out
        prompt >> out.prompt()

        depth >> out.controlnet.add(
            'diffusers/controlnet-depth-sdxl-1.0',
            conditioning_scale=depth_conditioning_scale,
            key='depth',
        )

        style >> out.ip_adapter.add(
            ip_adapter_model_id,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
            scale=style_scale,
            key='style',
        )

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(fg, bg, prompt, style)

    return g
