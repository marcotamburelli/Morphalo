'''
Foundation nodes demo (minimal).

This module contains five small DAGs covering the minimal foundation workflows:

1) QwenImage      - text-to-image generation (no input image)
2) QwenImageEdit  - instruction-based image editing (requires one input image)
3) QwenImageEditPlus - multi-image instruction editing with ordered image refs
4) QwenImageInpaint - masked inpainting with an AnyCrop-generated mask
5) OmniGen        - multimodal generation/editing with wired images referenced in the prompt

How to run
----------
    ./bin/run_dag.sh demo.10_foundation_minimal --dag qwen_image_min
    ./bin/run_dag.sh demo.10_foundation_minimal --dag qwen_image_edit_min
    ./bin/run_dag.sh demo.10_foundation_minimal --dag qwen_image_edit_plus_min
    ./bin/run_dag.sh demo.10_foundation_minimal --dag qwen_image_inpaint_tshirt_min
    ./bin/run_dag.sh demo.10_foundation_minimal --dag omnigen_style_transfer_min
'''

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Prompt
from morphalo.nodes.foundation import (
    OmniGen,
    QwenImage,
    QwenImageEdit,
    QwenImageEditPlus,
    QwenImageInpaint,
)
from morphalo.nodes.preprocess import AnyCrop

ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMG_1 = '~/images/init_img_3.png'
SOURCE_IMG_2 = '~/images/img_2.jpg'
SOURCE_IMG_3 = '~/images/picture_2.jpg'

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
        path=SOURCE_IMG_1,
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
# 3) QwenImageEditPlus (structure image + face reference)
# -----------------------------------------------------------------------------
#
# QwenImageEditPlus accepts an ordered sequence of one or more input images.
# The model is trained for multi-image editing through image concatenation, so
# prompts should refer to visible subjects/objects rather than "the first image"
# or "the second image". Here the wide composition provides the schema and the
# portrait person provides facial traits.
#
with DAG(
    name='qwen_image_edit_plus_min',
    out_dir=ROOT / 'outputs' / 'qwen_image_edit_plus_min',
):

    structure_img = FileImage(
        name='structure_img',
        path=SOURCE_IMG_2,
    )

    face_ref_img = FileImage(
        name='face_ref_img',
        path=SOURCE_IMG_3,
    )

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': [
                "Take the person in the portrait, wearing a gray sweatshirt.",
                "Then create a full body image of this person, using a pose and clothing as the image of to the seated long-haired girl.",
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

    out = QwenImageEditPlus(
        name='out',
        spec={
            'model': {
                # Defaults to 'Qwen/Qwen-Image-Edit-2509' if omitted.
                # 'id': 'Qwen/Qwen-Image-Edit-2509',
                'dtype': 'bf16',
                'device_map': 'balanced',
            },
            'params': {
                'steps': 40,
                'true_cfg_scale': 4.0,
                # # SOURCE_IMG_2 is 626x417 (~3:2). Keep a matching output canvas
                # # so the edit has less incentive to collapse into SOURCE_IMG_3's
                # # square portrait framing.
                # 'width': 1024,
                # 'height': 672,
                'max_images': 2,
            },
        },
    )

    # First image: schema / composition source.
    structure_img >> out.image.add(idx=0)

    # Second image: facial-traits reference.
    face_ref_img >> out.image.add(idx=1)

    # Lateral: prompt instruction -> editor.
    prompt >> out.prompt()


# -----------------------------------------------------------------------------
# 4) QwenImageInpaint (AnyCrop t-shirt mask + text inpaint)
# -----------------------------------------------------------------------------
#
# AnyCrop localizes the upper garment with an open-vocabulary prompt and emits a
# full-frame inpaint mask. QwenImageInpaint then repaints only that masked
# garment area and asks for a t-shirt with the text "MORPHALO".
#
with DAG(
    name='qwen_image_inpaint_tshirt_min',
    out_dir=ROOT / 'outputs' / 'qwen_image_inpaint_tshirt_min',
):

    init_img = FileImage(
        name='init_img',
        path=SOURCE_IMG_2,
    )

    tshirt_mask = AnyCrop(
        name='tshirt_mask',
        spec={
            'model': {
                'grounding_model': 'IDEA-Research/grounding-dino-tiny',
                'sam_model': 'facebook/sam-vit-large',
            },
            'prompt': 't-shirt. tshirt. tee shirt. shirt. sweatshirt. top.',
            'params': {
                'mode': 'mask',
                'box_margin': 0.10,
                'box_threshold': 0.20,
                'text_threshold': 0.15,
                'select': 'best',
                'dilate_radius': 0,
                'close_radius': 4,
                'smoothing_radius': 0,
            },
        },
    )

    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': [
                't-shirt with text "MORPHALO" printed across the chest.',
            ],
            'negative_prompt': [
                'misspelled text',
                'wrong text',
                'extra letters',
                'garbled typography',
            ],
        },
    )

    out = QwenImageInpaint(
        name='out',
        spec={
            'model': {
                # Defaults to 'Qwen/Qwen-Image' with
                # 'InstantX/Qwen-Image-ControlNet-Inpainting'.
                'dtype': 'bf16',
                'device_map': 'balanced',
            },
            'params': {
                'steps': 30,
                'true_cfg_scale': 4.0,
                'controlnet_conditioning_scale': 1.0,
            },
            'seed': 1234,
        },
    )

    # Base image for the inpaint.
    init_img >> out

    # Full-frame white-on-black t-shirt mask generated by AnyCrop.
    init_img >> tshirt_mask
    tshirt_mask >> out.mask()

    # Lateral: inpaint instruction -> editor.
    prompt >> out.prompt()


# -----------------------------------------------------------------------------
# 5) OmniGen (main image + style reference)
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
        path=SOURCE_IMG_1,
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
