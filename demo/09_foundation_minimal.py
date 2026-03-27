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
    ./bin/run_dag.sh demo.09_foundation_minimal --dag omnigen_one_image_min
'''

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Prompt
from morphalo.nodes.foundation import OmniGen, QwenImage, QwenImageEdit

ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMG = '~/images/input_img.png'


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
# 3) OmniGen (one wired image + prompt reference)
# -----------------------------------------------------------------------------
#
# OmniGen supports multimodal image inputs wired via `omnigen.image.add(key=...)`.
# The prompt MUST reference wired images using placeholders:
#
#   {{img:<key>}}        -> first image for that key
#   {{img:<key>[idx]}}   -> idx-th image for that key (0-based)
#
# If images are wired but not referenced, OmniGen may raise an error to avoid
# silent conditioning loss.
#
with DAG(
    name='omnigen_one_image_min',
    out_dir=ROOT / 'outputs' / 'omnigen_one_image_min',
):

    main_img = FileImage(
        name='main_img',
        path=SOURCE_IMG,
    )

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': [
                'In image {{img:main}} convert the subject into a live-action cinematic frame.',
                'Preserve the original pose, proportions, and composition.',
                'Natural skin texture, cinematic lighting, sharp focus.',
                '35mm film look, shallow depth of field, subtle grain.',
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

                # If True, uses input image size as output size for image editing.
                # 'img_size_as_out': True,
            },
            'seed': 1234,
        },
    )

    # Wire one image under key 'main'
    main_img >> out.image.add(key='main')

    # Lateral: prompt -> OmniGen
    prompt >> out.prompt()
