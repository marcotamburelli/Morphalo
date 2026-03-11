from __future__ import annotations

from typing import Optional

from stability.dag import NodeGroup
from stability.nodes import Img2Img, Tap
from stability.nodes.common.config_resolve import SpecInput
from stability.nodes.preprocess import ImageStack, SubjectCrop
from stability.nodes.preprocess.image_stack import ResizeMode


def cutout_stack_img2img_group(
    name: str,
    *,
    out_spec: SpecInput,
    fg_layer_feather: int | str = '0.5%',
    fg_layer_position: str = 'center',
    fg_layer_resize: ResizeMode = None,
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
    fg_layer_feather : int, optional
        Feather applied to the subject layer in the stack.
    fg_layer_position : str, optional
        Subject placement anchor/position for stack.image(...).
    fg_layer_corner_radius : int or None, optional
        Optional rounded corner radius for the subject alpha edge (if supported by ImageStack).
    fg_layer_corner_radius : ResizeMode, optional
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
                    # Full-frame RGBA output with alpha mask applied
                    'full_frame': True,
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
