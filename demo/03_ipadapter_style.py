"""
IP-Adapter (style) + Img2Img demo.

This example demonstrates IP-Adapter style conditioning in an Img2Img workflow.

Why Img2Img here?
-----------------
A spatial mask is most meaningful when you already have a base image with a
well-defined layout. With Img2Img, the init image provides that spatial anchor,
so applying style "only in a region" is a sensible operation.

Conceptual graph
----------------
Prompt  - - - - - - - - - - - - - - - - ┐
                                        v
Init image --------------------------> Img2Img  -> Output
Style refs (images) - - - - - - - - ->/
Mask (optional)   - - - - - - - - ->/

How to run
----------
    ./bin/run_dag.sh demo.03_ipadapter_style --dag ipadapter_style_img2img
"""

from pathlib import Path

from stability.dag import DAG
from stability.nodes import FileImage, Img2Img, Prompt

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User inputs (local paths, typically gitignored)
# -----------------------------------------------------------------------------
#
# Init image: the base image to be transformed.
#
INIT_IMG = '~/images/init_img.png'

# Style references: one or more images used by IP-Adapter.
#
STYLE_REFS = [
    '~/images/style_ref_01.png',
    '~/images/style_ref_02.png',
]

# Optional mask: apply style only to a region.
# Convention: white = apply, black = ignore.
#
STYLE_MASK = '~/images/style_mask.png'


with DAG(
    name='ipadapter_style_img2img',
    out_dir=ROOT / 'outputs' / 'ipadapter_style_img2img',
):

    # 1) Prompt (lateral injection)
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
    # Inputs
    # -------------------------------------------------------------------------
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    style_refs = FileImage(
        name='style_refs',
        path=STYLE_REFS,
    )

    # Optional mask for regional style application.
    style_mask = FileImage(
        name='style_mask',
        path=STYLE_MASK,
    )

    # -------------------------------------------------------------------------
    # Generator: Img2Img
    # -------------------------------------------------------------------------
    #
    # Img2Img uses the init image as spatial anchor.
    # `strength` controls how much the output can deviate from the init image.
    #
    out = Img2Img(
        name='out',
        spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {
                'steps': 30,
                'cfg': 5.0,
                'strength': 0.6,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    # Main flow: init image -> Img2Img
    init_img >> out

    # Lateral wiring: prompt -> Img2Img
    prompt >> out.prompt()

    # Lateral wiring: IP-Adapter style
    #
    # IP-Adapter for SDXL 1.0 (h94/IP-Adapter, subfolder='sdxl_models'):
    #
    # - ip-adapter_sdxl.bin:
    #     global image embedding from OpenCLIP-ViT-bigG-14
    # - ip-adapter_sdxl_vit-h.bin:
    #     global image embedding from OpenCLIP-ViT-H-14
    # - ip-adapter-plus_sdxl_vit-h.bin:
    #     patch image embeddings from OpenCLIP-ViT-H-14 (usually closer to reference)
    # - ip-adapter-plus-face_sdxl_vit-h.bin:
    #     like "plus", but expects a cropped face reference as condition
    #
    ip_style = out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin',
        # one value per reference image (matches STYLE_REFS length)
        scale=[0.7, 0.7],
        key='style',
    )

    # Wire style references into the adapter slot.
    style_refs >> ip_style

    # Optional: wire a mask into the same slot for regional style application.
    # If you do not need regional control, you can remove these two lines.
    style_mask >> ip_style.mask()
