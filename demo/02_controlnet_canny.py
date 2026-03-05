'''
ControlNet (Canny) txt2img demo.

This example introduces the first 'real' lateral conditioning workflow:

- Main flow: an input image is processed into an auxiliary map (Canny edges).
- Lateral wiring: that aux map is injected into Txt2Img as a ControlNet input.
- Prompt is injected laterally through the prompt channel (as in demo 01).

Pipeline structure (conceptual)
-------------------------------

            (lateral prompt)
Prompt  - - - - - - - - - - - - - - - - ┐
                                        v
Source image -> ImgAuxMap(canny) - - -> Txt2Img  -> Output image
                        (lateral ControlNet)

How to run
----------
From the repository root:

    ./bin/run_dag.sh demo.02_controlnet_canny --dag controlnet_canny_txt2img

Requirements
------------
- Provide an input image path (see SOURCE_IMG below).
- Diffusers will download ControlNet weights on first run if not cached.

Notes on supported ControlNets
------------------------------
At the moment, the repo focuses on these SDXL ControlNet models:

- diffusers/controlnet-canny-sdxl-1.0
- diffusers/controlnet-depth-sdxl-1.0

Nothing prevents adding more, but these are the ones used/tested in the demos.
'''

from __future__ import annotations

import os
from pathlib import Path

from stability.dag import DAG
from stability.nodes import FileImage, Prompt, Txt2Img
from stability.nodes.preprocess import ImgAuxMap

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User input: source image
# -----------------------------------------------------------------------------
#
# This image is NOT used as an init image for img2img.
# It is only used to extract a structural signal (Canny edges).
#
# Tip:
# - Keep it roughly aligned with the subject/pose you want.
# - High-contrast line structure generally produces stronger guidance.
#
SOURCE_IMG = '~/images/input_img.png'


# -----------------------------------------------------------------------------
# DAG definition
# -----------------------------------------------------------------------------
with DAG(
    name='controlnet_canny_txt2img',
    out_dir=ROOT / 'outputs' / 'controlnet_canny_txt2img',
) as dag:

    # 1) Load the source image (used only for preprocessing)
    source = FileImage(
        name='source',
        path=SOURCE_IMG,
    )

    # 2) Prompt node (lateral injection into generator)
    #
    # Keep this example in English for clarity. If you want to demonstrate
    # automatic translation, you can set e.g. `lang: ita_Latn` and write the
    # prompt in Italian (see demo 01 notes).
    #
    prompt = Prompt(
        name='prompt',
        spec={
            'prompt': {
                'content': [
                    'A cinematic portrait of a silver-haired mage with violet eyes.',
                    'Realistic skin texture, natural lighting, sharp focus.',
                ],
                'style': [
                    '35mm film look, shallow depth of field, subtle grain.',
                ],
            },
            'negative_prompt': [
                'blurry',
                'low quality',
                'bad anatomy',
                'extra fingers',
            ],
            'lang': 'eng_Latn',
        },
    )

    # 3) Preprocessing node: ImgAuxMap
    #
    # ImgAuxMap converts an input image into an auxiliary conditioning map.
    # Here we use Canny edges. Typical processors include:
    # - 'canny'
    # - 'depth_midas' (if you want a depth map)
    #
    # detect_long_side controls the resolution used for detection.
    #
    canny = ImgAuxMap(
        name='canny',
        spec={
            'processor': 'canny',
            'detect_long_side': 1024,
        },
    )

    # 4) Generator node: Txt2Img
    #
    # This node will switch to a ControlNet-capable pipeline at runtime if
    # one or more ControlNets are declared via `out.controlnet.add(...)`.
    #
    out = Txt2Img(
        name='out',
        spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {
                'steps': 25,
                'cfg': 5.0,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    # Prompt lateral wiring (same concept as demo 01)
    prompt >> out.prompt()

    # ControlNet lateral wiring
    #
    # The chain below has a main-flow feel, but conceptually it prepares a
    # ControlNet conditioning image that is injected laterally into Txt2Img:
    #
    #   source -> canny -> (ControlNet sink on out)
    #
    # `conditioning_scale` controls how strongly the edges constrain generation.
    #
    # Supported / tested model ids (SDXL):
    # - diffusers/controlnet-canny-sdxl-1.0
    # - diffusers/controlnet-depth-sdxl-1.0
    #
    source >> canny >> out.controlnet.add(
        'diffusers/controlnet-canny-sdxl-1.0',
        conditioning_scale=0.5,
        key='canny',
    )

    # Optional: swap to depth ControlNet
    #
    # depth = ImgAuxMap(name='depth', spec={'processor': 'depth_midas', 'detect_long_side': 1024})
    # source >> depth >> out.controlnet.add(
    #     'diffusers/controlnet-depth-sdxl-1.0',
    #     conditioning_scale=0.4,
    #     key='depth',
    # )