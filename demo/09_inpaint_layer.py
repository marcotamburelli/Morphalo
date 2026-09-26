"""
Inpaint layer demo.

This example demonstrates a deterministic layer insertion workflow followed by
masked SDXL inpainting for local harmonization.

Conceptual graph
----------------
Base image -----------------------------> ImageStack -----> Inpaint -> Output
                                             ^                ^
Overlay image -> MaskInsertLayer ----------/                |
                 ^                                           |
Upper-garment mask from AnyCrop -----------------------------/

Pipeline
--------
1. Generate an upper-garment mask from the base image with AnyCrop.
2. Fit a rotated rectangle inside that mask with MaskInsertLayer.
3. Insert a prepared text/logo overlay as a full-frame transparent layer.
4. Composite the overlay layer over the base image with ImageStack.
5. Inpaint only the upper-garment mask to harmonize the inserted graphic.

How to run
----------
    ./bin/run_dag.sh demo.09_inpaint_layer --dag inpaint_layer_tshirt

Inputs (local paths)
--------------------
- base image: source image containing the shirt or top
- overlay image: prepared text/logo/graphic asset to insert
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Inpaint
from morphalo.nodes.preprocess import AnyCrop, MaskInsertLayer
from morphalo.nodes.process import ImageStack

ROOT = Path(__file__).resolve().parents[1]

BASE_IMG = '~/images/init_img_2.png'
OVERLAY_IMG = '~/images/text.png'

# Must match the actual base image resolution used by ImageStack / Inpaint.
CANVAS_W = 2048
CANVAS_H = 2048

MODEL_SPEC = {
    'grounding_model': 'IDEA-Research/grounding-dino-tiny',
    'sam_model': 'facebook/sam-vit-large',
}

SDXL_SPEC = {
    'model': {
        'path': '~/models/juggernaut/juggernautXL_ragnarokBy.safetensors',
        'device': 'cuda',
        'dtype': 'bf16',
    },
    'prompt': {
        'content': [
            'a black sleeveless top with a clean printed graphic on the fabric',
            'the graphic is naturally integrated into the shirt',
            'fabric folds, subtle lighting, realistic print texture',
        ],
        'style': [
            'high quality image',
            'natural photographic look',
        ],
    },
    'negative_prompt': [
        'distorted text',
        'garbled letters',
        'extra logos',
        'messy print',
        'low quality',
    ],
    'params': {
        'width': CANVAS_W,
        'height': CANVAS_H,
        'steps': 25,
        'cfg': 4.5,

        # Low strength: harmonize the overlay with the fabric without rewriting it.
        'strength': 0.3,
    },
    'seed': 12345,
}

with DAG(
    name='inpaint_layer_tshirt',
    out_dir=ROOT / 'outputs' / 'inpaint_layer_tshirt',
):
    # -------------------------------------------------------------------------
    # Input sources
    # -------------------------------------------------------------------------
    #
    # Base image containing a visible shirt / top / tank top area.
    # Overlay image is a prepared graphic/text/logo RGBA or RGB asset.
    #
    init_img = FileImage(
        name='init_img',
        path=BASE_IMG,
    )

    overlay = FileImage(
        name='overlay',
        path=OVERLAY_IMG,
    )

    # -------------------------------------------------------------------------
    # Prompt-based upper-garment mask
    # -------------------------------------------------------------------------
    #
    # AnyCrop localizes the upper garment using a flexible open-vocabulary prompt.
    # The output is a full-frame mask aligned to the source image.
    #
    # White = selected garment region.
    # Black = everything else.
    #
    upper_garment_mask = AnyCrop(
        name='upper_garment_mask',
        spec={
            'model': MODEL_SPEC,
            'prompt': (
                'top. black top. tank top. sleeveless top. sleeveless shirt.'
            ),
            'params': {
                'mode': 'mask',
                'box_margin': 0.08,
                'box_threshold': 0.20,
                'text_threshold': 0.15,
                'select': 'best',

                # Keep the mask sharp so the detected placement region stays
                # predictable.
                'dilate_radius': 0,
                'close_radius': 0,
                'smoothing_radius': 0,
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    # -------------------------------------------------------------------------
    # Mask-guided overlay insertion
    # -------------------------------------------------------------------------
    #
    # MaskInsertLayer:
    # - receives the garment mask from AnyCrop,
    # - fits a rotated rectangle aligned to the mask visual axis,
    # - fits the prepared overlay image inside that rectangle,
    # - emits a full-frame RGBA layer,
    # - clips the final layer to the original mask.
    #
    shirt_insert = MaskInsertLayer(
        name='shirt_insert',
        spec={
            'params': {
                'threshold': 128,

                # Content insets inside the selected rectangle.
                'inset_left': '6%',
                'inset_right': '6%',
                'inset_top': '10%',
                'inset_bottom': '10%',

                # Optional fitting helpers.
                # None means: do not connect disconnected foreground components.
                # If enabled, only percentage strings are accepted, e.g. '5%'.
                'max_bridge_distance': None,

                # False means: do not fill internal holes in the fitting mask.
                'fill_holes': False,
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    # -------------------------------------------------------------------------
    # Composite inserted overlay over source image
    # -------------------------------------------------------------------------
    #
    # The inserted overlay is a full-frame RGBA layer, so it can be stacked
    # directly over the original image. ImageStack composites layers in
    # increasing idx order.
    #
    stack = ImageStack(
        name='stack',
        spec={
            'params': {
                'width': CANVAS_W,
                'height': CANVAS_H,
                'background': None,
                'out_mode': 'RGB',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Low-strength Inpaint harmonization
    # -------------------------------------------------------------------------
    #
    # This slightly integrates the printed graphic with the shirt texture, while
    # constraining generation to the upper-garment mask produced above.
    #
    harmonize = Inpaint(
        name='harmonize',
        spec=SDXL_SPEC,
    )

    init_img >> upper_garment_mask

    upper_garment_mask >> shirt_insert
    overlay >> shirt_insert.overlay(
        position='top',
        resize='fit',
    )

    init_img >> stack.image(0)
    shirt_insert >> stack.image(1)

    stack >> harmonize
    upper_garment_mask >> harmonize.mask()
