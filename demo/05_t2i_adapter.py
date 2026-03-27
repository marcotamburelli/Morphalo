"""
T2I-Adapter (lineart) demo.

This example shows how to inject a T2I-Adapter conditioning signal into a
generation node.

Conceptual graph
----------------
Prompt  - - - - - - - - - - - - - - - - - ┐
                                          v
Source image -> ImgAuxMap(lineart) - - -> Txt2Img -> Output
                        (lateral T2I-Adapter)

How to run
----------
    ./bin/run_dag.sh demo.05_t2i_adapter --dag t2i_adapter_lineart_txt2img

Notes
-----
- T2I-Adapter consumes an *auxiliary map* (e.g. lineart, depth, pose) extracted
  from a source image via `ImgAuxMap`.
- The full list of supported `ImgAuxMap` processors is defined in `img_aux_map.py`
  (see the `MODELS` mapping).
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Prompt, Txt2Img
from morphalo.nodes.preprocess import ImgAuxMap

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# T2I-Adapter overview (SDXL)
# -----------------------------------------------------------------------------
#
# T2I-Adapter is a lightweight adapter network that provides *additional
# conditioning* to Stable Diffusion. Each adapter checkpoint expects a specific
# kind of "control image" (aux map) and is designed to be used with a specific
# base model family (here: SDXL).
#
# In this project, the control image is produced by `ImgAuxMap` using a
# `processor` selected from `img_aux_map.py` (see the `MODELS` mapping).
#
# The following SDXL T2I-Adapter checkpoints (TencentARC) map naturally to
# these `ImgAuxMap` processors:
#
#   Adapter checkpoint                               Expected control image        ImgAuxMap.processor
#   ----------------------------------------------   ---------------------------   -------------------------
#   TencentARC/t2i-adapter-canny-sdxl-1.0            Canny edges (white on black)  'canny'
#
#   TencentARC/t2i-adapter-sketch-sdxl-1.0           PidiNet edges / sketch-like   'scribble_pidinet'
#                                                                                  (or 'softedge_pidinet')
#
#   TencentARC/t2i-adapter-lineart-sdxl-1.0          Lineart (white on black)      'lineart_realistic'
#                                                                                  (or 'lineart_coarse',
#                                                                                      'lineart_anime')
#
#   TencentARC/t2i-adapter-depth-midas-sdxl-1.0      Depth (Midas) grayscale       'depth_midas'
#
#   TencentARC/t2i-adapter-depth-zoe-sdxl-1.0        Depth (Zoe) grayscale         'depth_zoe'
#
#   TencentARC/t2i-adapter-openpose-sdxl-1.0         OpenPose skeleton             'openpose'
#                                                                                  (or 'openpose_full',
#                                                                                      'openpose_face',
#                                                                                      'openpose_hand', etc.)
#
# Notes
# -----
# - "scribble" vs "softedge" variants:
#   - scribble_* tends to produce thinner, more stylized lines
#   - softedge_* tends to produce smoother, more continuous edges
#
# - `conditioning_scale` controls how strongly the adapter constrains generation.
#   Start around 0.6–0.9 for lineart/sketch and adjust by taste.
#
# - These adapters are part of "T2I-Adapter: Learning Adapters to Dig out More
#   Controllable Ability for Text-to-Image Diffusion Models" (Mou et al., 2023).

SOURCE_IMG = '~/images/input_img.png'

with DAG(
    name='t2i_adapter_lineart_txt2img',
    out_dir=ROOT / 'outputs' / 't2i_adapter_lineart_txt2img',
):

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------
    #
    # This source image is used ONLY to compute the aux map (lineart).
    #
    source_img = FileImage(
        name='source_img',
        path=SOURCE_IMG,
    )

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': {
                'content': [
                    'A cinematic portrait of a silver-haired mage with violet eyes.',
                    'Natural skin texture, cinematic lighting, sharp focus.',
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
        },
    )

    # -------------------------------------------------------------------------
    # Generator
    # -------------------------------------------------------------------------
    out = Txt2Img(
        name='out',
        spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {
                'steps': 50,
                'cfg': 9.0,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    # -------------------------------------------------------------------------
    # Preprocessing: build the aux map for the adapter
    # -------------------------------------------------------------------------
    #
    # Processor choices depend on the adapter. For lineart adapters, typical
    # processors are:
    # - 'lineart_realistic'
    # - 'lineart_coarse'
    # - 'lineart_anime'
    #
    # The authoritative list lives in `img_aux_map.py` as the `MODELS` mapping.
    #
    lineart = ImgAuxMap(
        name='lineart',
        spec={
            'processor': 'lineart_realistic',
            'detect_long_side': 1024,
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    # Prompt lateral wiring
    prompt >> out.prompt()

    # T2I-Adapter lateral wiring
    #
    # 1) Build the aux map: source_img -> lineart
    # 2) Inject into Txt2Img via a declared adapter slot:
    #
    # conditioning_scale controls how strongly the adapter constrains generation.
    #
    source_img >> lineart >> out.t2i_adapter.add(
        'TencentARC/t2i-adapter-lineart-sdxl-1.0',
        conditioning_scale=0.9,
        key='sketch',
    )
