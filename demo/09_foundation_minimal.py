'''
Foundation nodes demo (minimal).

This module contains three small DAGs, one for each "foundation" node:

1) QwenImage      - text-to-image generation (no input image)
2) QwenImageEdit  - instruction-based image editing (requires one input image)
3) OmniGen        - multimodal generation/editing with wired images referenced in the prompt

How to run
----------
    ./bin/run_dag.sh demo.09_foundation_minimal --dag qwen_image_min
    ./bin/run_dag.sh demo.09_foundation_minimal --dag qwen_image_edit_min
    ./bin/run_dag.sh demo.09_foundation_minimal --dag omnigen_style_transfer_min
'''

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Prompt
from morphalo.nodes.foundation import OmniGen, QwenImage, QwenImageEdit

ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMG = '~/images/input_img.png'
STYLE_REF = '~/images/style_ref_03.jpg'


# -----------------------------------------------------------------------------
# 1) QwenImage (text-to-image)
# -----------------------------------------------------------------------------
#
# QwenImage is a foundation txt2img node. It does not require an upstream image.
# Prompts are injected via the standard prompt channel.
#
with DAG(
    name='qwen_image_min',
    out_dir=ROOT / 'outputs' / 'qwen_image_min',
):

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': [
                'A cinematic portrait of a silver-haired mage with violet eyes.',
                'Natural skin texture, cinematic lighting, sharp focus.',
                '35mm film look, shallow depth of field, subtle grain.',
            ],
            'negative_prompt': [
                'blurry',
                'low quality',
                'bad anatomy',
                'extra fingers',
            ],
        },
    )

    out = QwenImage(
        name='out',
        spec={
            'model': {
                # Defaults to 'Qwen/Qwen-Image' if omitted.
                # 'id': 'Qwen/Qwen-Image',
                'dtype': 'bf16',
                'device_map': 'balanced',
            },
            'params': {
                'steps': 25,

                # Guidance notes (Qwen-Image specific):
                # - true_cfg_scale is only meaningful when a negative_prompt is provided.
                # - guidance_scale may be ignored by non-distilled variants.
                'true_cfg_scale': 4.0,

                # Optional output size (only passed if specified)
                # 'width': 1024,
                # 'height': 1024,
            },
            'seed': 1234,
        },
    )

    prompt >> out.prompt()


# -----------------------------------------------------------------------------
# 2) QwenImageEdit (instruction-based image editing)
# -----------------------------------------------------------------------------
#
# QwenImageEdit requires ONE init image wired into the default input:
#     init_img >> qwen_edit
#
# The prompt is an editing instruction. The model may change pose/viewpoint if
# you ask for it, so keep instructions explicit if you want structure preserved.
#
with DAG(
    name='qwen_image_edit_min',
    out_dir=ROOT / 'outputs' / 'qwen_image_edit_min',
):

    init_img = FileImage(
        name='init_img',
        path=SOURCE_IMG,
    )

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': [
                'Convert the provided image into a live-action cinematic frame.',
                'Preserve the original pose, proportions, and composition.',
                'Natural skin texture, cinematic lighting, sharp focus.',
                '35mm film look, shallow depth of field, subtle grain.',
            ],
            'negative_prompt': [
                'CGI',
                '3D render',
                'cartoon',
                'anime',
                'overly smooth skin',
                'plastic skin',
            ],
        },
    )

    out = QwenImageEdit(
        name='out',
        spec={
            'model': {
                # Defaults to 'Qwen/Qwen-Image-Edit' if omitted.
                # 'id': 'Qwen/Qwen-Image-Edit',
                'dtype': 'bf16',
                'device_map': 'balanced',
            },
            'params': {
                'steps': 25,

                # Primary control knob for instruction strength.
                'true_cfg_scale': 4.0,

                # If width/height are omitted, the output defaults to the input size.
                # 'width': 1024,
                # 'height': 1024,
            },
            'seed': 1234,
        },
    )

    # Main flow: init image -> editor
    init_img >> out

    # Lateral: prompt instruction -> editor
    prompt >> out.prompt()


# -----------------------------------------------------------------------------
# 3) OmniGen (main image + style reference)
# -----------------------------------------------------------------------------
#
# This demo uses:
# - one MAIN image that provides subject, pose, and composition
# - one STYLE reference image that provides character design, outfit, and style
#
# The goal is not a strict pixel-preserving edit, but a guided multimodal
# transformation:
# - keep the subject layout from the main image
# - transfer the character look, clothing design, and visual style from the
#   style reference
#
with DAG(
    name='omnigen_style_transfer_min',
    out_dir=ROOT / 'outputs' / 'omnigen_style_transfer_min',
):

    main_img = FileImage(
        name='main_img',
        path=SOURCE_IMG,
    )

    style_img = FileImage(
        name='style_img',
        path=STYLE_REF,
    )

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': [
                'In image {{img:main}}, transform the main subject using the character design and outfit style from {{img:style}}.',
                'Preserve the original pose, body proportions, framing, and overall composition.',
                'Transfer the hairstyle, clothing, costume details, materials, and overall visual identity.',
                'Keep the scene cinematic, with natural skin texture, sharp focus, realistic lighting, shallow depth of field, and subtle film grain.',
            ],
        },
    )

    out = OmniGen(
        name='out',
        spec={
            'model': {
                # Defaults to 'Shitao/OmniGen-v1-diffusers' if omitted.
                # 'id': 'Shitao/OmniGen-v1-diffusers',
                'device': 'cuda',
                'dtype': 'bf16',
            },
            'params': {
                'steps': 30,
                'guidance_scale': 2.0,
                'img_guidance_scale': 1.6,

                # For edit-like workflows, keeping the input resolution is often useful.
                'img_size_as_out': True,
            },
            'seed': 1234,
        },
    )

    # Wire the structural source image.
    main_img >> out.image.add(key='main')

    # Wire the style / character reference image.
    style_img >> out.image.add(key='style')

    # Lateral: prompt -> OmniGen
    prompt >> out.prompt()
