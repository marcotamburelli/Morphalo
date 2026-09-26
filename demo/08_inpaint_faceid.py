"""
Inpaint + FaceID and FaceSwap demos.

This module demonstrates two identity-transfer workflows and one comparative
face-geometry experiment:

1. Inpainting constrained with FaceID PlusV2.
2. Direct face replacement with SimSwap.
3. 3D source-landmark transfer processed as depth, Canny, and lineart.

Conceptual graph
----------------
Prompt  - - - - - - - - - - - - - - - -┐
                                        v
FaceIdEmbedImage (identity embeds) - - > Inpaint  -> Output image
Mask (inpaint region) - - - - - - - - -^
Source image (base)  ------------------->

Target image --------------------------> FaceSwap -> Output image
Identity image ------------------------> FaceSwap.source()

Target image -------------------------> FaceGeometryMap -> Processor map
Source image -------------------------> FaceGeometryMap.source()       |
Original target + prompt + identity ----------------------------------------> Img2Img

How to run
----------
    ./bin/run_dag.sh demo.08_inpaint_faceid --dag inpaint_faceid
    ./bin/run_dag.sh demo.08_inpaint_faceid --dag face_swap
    ./bin/run_dag.sh demo.08_inpaint_faceid --dag face_geometry_maps

Inputs (local paths)
-------------------
- source image: the base image to inpaint or use as the FaceSwap target
- mask image: white = region to inpaint, black = keep original
- face reference: used for FaceID embeddings or as the FaceSwap identity source
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import (FaceIdEmbedImage, FileImage, Img2Img, Inpaint,
                            Prompt)
from morphalo.nodes.preprocess import FaceGeometryMap
from morphalo.nodes.process import FaceSwap

ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMG = '~/images/generated_2.png'
FACE_REF = '~/images/identity_face.png'
STYLE_MASK = '~/images/style_mask.png'

with DAG(
    name='inpaint_faceid',
    out_dir=ROOT / 'outputs' / 'inpaint_faceid',
):

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------
    #
    # Base image to be inpainted.
    source = FileImage(
        name='source',
        path=SOURCE_IMG,
    )

    # Inpaint mask:
    # - white pixels: area to regenerate
    # - black pixels: area to keep from the source image
    mask = FileImage(
        name='mask',
        path=STYLE_MASK,
    )

    # Face reference image used to extract identity embeddings (InsightFace).
    face_emb = FaceIdEmbedImage(
        name='face_emb',
        path=FACE_REF,
    )

    # -------------------------------------------------------------------------
    # Prompt (kept consistent across demos for comparability)
    # -------------------------------------------------------------------------
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
    # Inpaint node
    # -------------------------------------------------------------------------
    #
    # Inpaint is an image-to-image generator that:
    # - takes a source image as the main input
    # - takes a mask specifying which region to regenerate
    # - optionally receives lateral conditioning (FaceID, ControlNet, adapters, ...)
    #
    # `strength` controls how strongly the model is allowed to deviate from the
    # source image inside the masked region.
    #
    out = Inpaint(
        name='out',
        spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {
                'steps': 30,
                'cfg': 3.0,
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

    # Main flow: source image goes into the Inpaint node.
    source >> out

    # Lateral wiring: prompt and mask are injected via dedicated channels.
    prompt >> out.prompt()
    mask >> out.mask()

    # Lateral wiring: identity constraint via FaceID.
    #
    # Use the FaceID PlusV2 weights for SDXL.
    # These variants combine InsightFace embeddings with CLIP image features,
    # typically improving identity preservation.
    #
    face_emb >> out.face_id.add(
        model_id='h94/IP-Adapter-FaceID',
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=1.0,
        # clip_strength=0.5,
        key='id',
    )


# -----------------------------------------------------------------------------
# FaceSwap (direct identity transfer)
# -----------------------------------------------------------------------------
#
# Unlike the FaceID inpainting DAG above, this workflow does not regenerate a
# masked image region with a diffusion model. It directly replaces the largest
# face detected in the target image while preserving the target pose,
# expression, body, and surrounding scene.
#
with DAG(
    name='face_swap',
    out_dir=ROOT / 'outputs' / 'face_swap',
):

    # Image whose face will be replaced.
    target = FileImage(
        name='target',
        path=SOURCE_IMG,
    )

    # Reference image providing the identity transferred onto the target.
    identity = FileImage(
        name='identity',
        path=FACE_REF,
    )

    swap = FaceSwap(
        name='swap',
        spec={
            'model': {
                'device': 'cpu',
                'model_name': 'buffalo_l',
                'det_size': [640, 640],
                'swapper': 'simswap_unofficial_512',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    # Main flow: the target image supplies pose, expression, and composition.
    target >> swap

    # Lateral input: the source attachment supplies the identity to transfer.
    identity >> swap.source()


# -----------------------------------------------------------------------------
# FaceGeometryMap (depth, Canny, and lineart comparison)
# -----------------------------------------------------------------------------
with DAG(
    name='face_geometry_maps',
    out_dir=ROOT / 'outputs' / 'face_geometry_maps',
):

    target = FileImage(
        name='target',
        path=SOURCE_IMG,
    )

    source = FileImage(
        name='source',
        path='~/images/face_x',
    )

    depth_transfer = FaceGeometryMap(
        name='depth_map',
        spec={
            'model': {
                'device': 'cpu',
                'face_landmarker_task': (
                    '~/models/mediapipe/face_landmarker.task'
                ),
            },
            'params': {
                'processor': 'depth_midas',
                'detail_gain': 0.0,
                'processor_device': 'cuda',
                'processor_params': {
                    'detect_resolution': 1024,
                    'image_resolution': 1024,
                },
            },
        },
    )

    canny_transfer = FaceGeometryMap(
        name='canny_map',
        spec={
            'model': {
                'device': 'cpu',
                'face_landmarker_task': (
                    '~/models/mediapipe/face_landmarker.task'
                ),
            },
            'params': {
                'processor': 'canny',
                'detail_gain': 1.75,
                'processor_params': {
                    'low_threshold': 30,
                    'high_threshold': 100,
                    'detect_resolution': 1024,
                    'image_resolution': 1024,
                },
            },
        },
    )

    lineart_transfer = FaceGeometryMap(
        name='lineart_map',
        spec={
            'model': {
                'device': 'cpu',
                'face_landmarker_task': (
                    '~/models/mediapipe/face_landmarker.task'
                ),
            },
            'params': {
                'processor': 'lineart_realistic',
                'detail_gain': 1.25,
                'processor_device': 'cuda',
                'processor_params': {
                    'detect_resolution': 1024,
                    'image_resolution': 1024,
                },
            },
        },
    )

    comparison_prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': {
                'content': ['standing young woman'],
            },
        },
    )

    depth_out = Img2Img(
        name='depth_out',
        spec={
            'model': {
                'path': (
                    '~/models/juggernaut/'
                    'juggernautXL_ragnarokBy.safetensors'
                ),
                'dtype': 'bf16',
                'device': 'cuda',
            },
            'params': {
                'steps': 30,
                'cfg': 0.3,
                'strength': 0.7,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    canny_out = Img2Img(
        name='canny_out',
        spec={
            'model': {
                'path': (
                    '~/models/juggernaut/'
                    'juggernautXL_ragnarokBy.safetensors'
                ),
                'dtype': 'bf16',
                'device': 'cuda',
            },
            'params': {
                'steps': 30,
                'cfg': 0.3,
                'strength': 0.6,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    lineart_out = Img2Img(
        name='lineart_out',
        spec={
            'model': {
                'path': (
                    '~/models/juggernaut/'
                    'juggernautXL_ragnarokBy.safetensors'
                ),
                'dtype': 'bf16',
                'device': 'cuda',
            },
            'params': {
                'steps': 30,
                'cfg': 0.3,
                'strength': 0.6,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    target >> depth_transfer
    source >> depth_transfer.source()
    target >> canny_transfer
    source >> canny_transfer.source()
    target >> lineart_transfer
    source >> lineart_transfer.source()

    # Keep every generation input identical except for map and ControlNet.
    target >> depth_out
    target >> canny_out
    target >> lineart_out
    comparison_prompt >> depth_out.prompt()
    comparison_prompt >> canny_out.prompt()
    comparison_prompt >> lineart_out.prompt()

    depth_transfer >> depth_out.controlnet.add(
        'diffusers/controlnet-depth-sdxl-1.0',
        conditioning_scale=1.0,
        key='depth',
    )
    canny_transfer >> canny_out.controlnet.add(
        'diffusers/controlnet-canny-sdxl-1.0',
        conditioning_scale=1.0,
        key='canny',
    )
    lineart_transfer >> lineart_out.controlnet.add(
        'ShermanG/ControlNet-Standard-Lineart-for-SDXL',
        conditioning_scale=1.0,
        key='lineart',
    )

    # The same source views used to fuse facial geometry provide appearance
    # and identity details through the same face-specific SDXL IP-Adapter.
    source >> depth_out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus-face_sdxl_vit-h.bin',
        scale=0.7,
        key='identity',
    )
    source >> canny_out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus-face_sdxl_vit-h.bin',
        scale=0.7,
        key='identity',
    )
    source >> lineart_out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus-face_sdxl_vit-h.bin',
        scale=0.7,
        key='identity',
    )
