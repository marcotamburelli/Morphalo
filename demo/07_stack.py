"""
ImageStack demo: crop a subject from one image and composite it onto a background.

This demo shows a simple two-step preprocessing/compositing workflow:

1. load a foreground image containing a person
2. extract the person as a trimmed RGBA cutout with ``SubjectCrop``
3. place the extracted subject over a separate background with ``ImageStack``

Why this demo?
--------------
This is a minimal example of how preprocessing nodes can be chained into a small
compositing pipeline inside the DAG:

- ``FileImage`` provides the input assets
- ``SubjectCrop`` isolates the person from the source image using landmarks and
  a SAM/SAM-HQ segmentation backend
- ``ImageStack`` places multiple image layers on a shared canvas

The resulting workflow is useful as a starting point for:

- quick photobash-style composites
- subject/background replacement
- preparing layered assets for later img2img or inpainting passes
- testing crop quality before more advanced downstream generation steps

Conceptual graph
----------------
Foreground image -> SubjectCrop(target='person') -> ImageStack(layer 1)
Background image --------------------------------> ImageStack(layer 0)

How to run
----------
    ./bin/run_dag.sh demo.07_stack --dag stack

Notes
-----
- The background is placed first and acts as the base canvas.
- The subject is added as a second layer and positioned at the bottom center.
- ``resize='fit'`` preserves aspect ratio while fitting the subject inside the
  target placement logic used by ``ImageStack``.
- ``feather='5px'`` slightly softens the cutout edges to reduce harsh seams.
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import ImageStack, SubjectCrop

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User inputs (local assets, typically gitignored)
# -----------------------------------------------------------------------------
#
# Foreground image containing the subject to extract.
#
INIT_IMG = '~/images/init_img.png'

# Background plate used as the base layer for the composite.
#
BK_IMG = '~/images/bk_img.png'

# Shared model assets used by SubjectCrop.
#
# Notes:
# - sam_model selects the SAM/SAM-HQ backend used for person segmentation.
# - YOLO proposes person boxes. If omitted, SubjectCrop uses its default YOLO
#   model.
# - MediaPipe Pose landmarks stabilize subject-box selection and provide
#   positive prompt points for person-mask candidates.
#
MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'yolo_model': 'yolov8n.pt',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

# Shared crop parameters for SubjectCrop.
#
# `mode='default'`:
#     emit a cropped RGBA image.
#
# `crop_mode='trim'`:
#     return a tight RGBA cutout trimmed to the non-transparent subject mask.
#
# `box_margin`:
#     small margin applied around the initial detected subject box.
#
COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': 0.12,
}


with DAG(
    name='stack',
    out_dir=ROOT / 'outputs' / 'stack',
):
    # -------------------------------------------------------------------------
    # Input sources
    # -------------------------------------------------------------------------
    #
    # `subject_img` is the image from which we extract the foreground person.
    # `bk_img` is the background plate that will receive the composited subject.
    #
    subject_img = FileImage(
        name='subject_img',
        path=INIT_IMG,
    )

    bk_img = FileImage(
        name='bk_img',
        path=BK_IMG,
    )

    # -------------------------------------------------------------------------
    # Foreground extraction
    # -------------------------------------------------------------------------
    #
    # Extract the main person from the foreground image as a trimmed RGBA cutout.
    # This output will later be placed as the second layer in the stack.
    #
    person_crop = SubjectCrop(
        name='person_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Layered compositing
    # -------------------------------------------------------------------------
    #
    # ImageStack composites multiple input images in layer order.
    #
    # In this demo:
    # - layer 0: background plate
    # - layer 1: extracted person cutout
    #
    stack = ImageStack(
        name='stack',
        spec={
            'params': {},
        },
    )

    # Base layer: background image.
    #
    bk_img >> stack.image(0)

    # Foreground layer: extracted subject.
    #
    # `position='center-bottom'` anchors the subject near the lower center of
    # the composition, which is a common placement for full-body characters.
    #
    # `resize='fit'` preserves aspect ratio while fitting the subject inside the
    # stack canvas.
    #
    # `feather='5px'` applies a small edge softening to reduce visible cutout
    # boundaries against the new background.
    #
    subject_img >> person_crop >> stack.image(
        idx=1,
        position='center-bottom',
        resize='fit',
        feather='5px',
    )
