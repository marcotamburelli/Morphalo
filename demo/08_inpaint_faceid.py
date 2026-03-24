"""
Inpaint + FaceID demo.

This example demonstrates how to perform inpainting while constraining identity
using FaceID (PlusV2).

Conceptual graph
----------------
Prompt  - - - - - - - - - - - - - - - -┐
                                        v
FaceIdEmbedImage (identity embeds) - - > Inpaint  -> Output image
Mask (inpaint region) - - - - - - - - -^
Source image (base)  ------------------->

How to run
----------
    ./bin/run_dag.sh demo.08_inpaint_faceid --dag inpaint_faceid

Inputs (local paths)
-------------------
- source image: the base image to inpaint
- mask image: white = region to inpaint, black = keep original
- face reference: used to extract FaceID embeddings
"""

from pathlib import Path

from stability.dag import DAG
from stability.nodes import FaceIdEmbedImage, FileImage, Inpaint, Prompt

ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMG = '~/images/input_img.png'
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
