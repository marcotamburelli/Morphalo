"""
SubjectCrop demo: extract person / head / face / eyes from a single image.

This demo shows how to use ``SubjectCrop`` to derive multiple subject-aware crops
from the same input image.

Why this demo?
--------------
``SubjectCrop`` is not just a generic crop node. Depending on the selected
``target``, it combines different detectors/segmenters to isolate a semantically
meaningful region:

- ``person``:
    full person extraction, typically using YOLO + SAM
- ``head``:
    head-oriented crop derived from face landmarks, with hair-friendly framing
- ``face``:
    tighter face crop
- ``eyes``:
    combined crop covering both eyes
- ``left-eye`` / ``right-eye``:
    single-eye crop from MediaPipe facial landmarks

This makes the node useful for workflows such as:

- extracting cutouts for compositing
- preparing localized regions for img2img refinement
- building masks or crops for facial repair / eye refinement
- generating reusable preprocessing assets for downstream DAGs

Conceptual graph
----------------
                 -> SubjectCrop(target='person')    -> person crop
                /
Input image ---+-> SubjectCrop(target='head')      -> head crop
                \
                 +-> SubjectCrop(target='face')    -> face crop
                  \
                   +-> SubjectCrop(target='eyes')      -> eyes crop
                    +-> SubjectCrop(target='left-eye') -> left eye crop
                    +-> SubjectCrop(target='right-eye')-> right eye crop

How to run
----------
    ./bin/run_dag.sh demo.06_crop --dag subject_crop

Notes
-----
- ``crop_mode='trim'`` produces a tight RGBA cutout inside the selected crop box,
  with transparency outside the extracted subject/mask.
- Eye targets rely on MediaPipe landmarks and do not require SAM segmentation.
- Person/head targets typically benefit from SAM + YOLO checkpoints being available.
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import SubjectCrop

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User inputs (local paths, typically gitignored)
# -----------------------------------------------------------------------------
#
# Source image used for all crops in this demo.
#
INIT_IMG = '~/images/init_img.png'

# Shared model assets used by SubjectCrop.
#
# Notes:
# - SAM is required for subject/mask extraction on larger regions such as person.
# - MediaPipe Face Landmarker is required for face/head/eyes targets.
# - The pose landmarker is included here for consistency with your local setup,
#   even if this specific demo mainly exercises face/head/eye-oriented logic.
#
MODEL_SPEC = {
    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

# Shared crop parameters reused across all SubjectCrop nodes in this demo.
#
# `mode='default'`:
#     emit an RGBA crop rather than a full-frame mask.
#
# `crop_mode='trim'`:
#     return the crop box content with transparency outside the selected mask.
#
# `box_margin`:
#     expands the initial detection box before segmentation/cropping.
#
# `expansion`:
#     target-specific enlargement factor, especially meaningful for head/eye crops.
#
COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': 0.12,
    'expansion': 1.2,
}


with DAG(
    name='subject_crop',
    out_dir=ROOT / 'outputs' / 'subject_crop',
):

    # -------------------------------------------------------------------------
    # Input source
    # -------------------------------------------------------------------------
    #
    # FileImage is a pure source node that exposes one existing image path to the DAG.
    #
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    # -------------------------------------------------------------------------
    # Person crop
    # -------------------------------------------------------------------------
    #
    # Extract the largest person in the frame as a trimmed RGBA cutout.
    # This is the most general target and is useful for compositing or
    # person-focused refinement workflows.
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
    # Head crop
    # -------------------------------------------------------------------------
    #
    # Produce a head-oriented crop. Compared to `face`, this usually keeps a
    # larger square region around the face, with more room for hair and framing.
    # Useful when refining portraits while preserving head silhouette/hair volume.
    #
    head_crop = SubjectCrop(
        name='head_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'head',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Face crop
    # -------------------------------------------------------------------------
    #
    # Tighter crop around the face region. Suitable for face-centric refinement,
    # identity-focused conditioning, or facial analysis workflows.
    #
    face_crop = SubjectCrop(
        name='face_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'face',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Eyes crop
    # -------------------------------------------------------------------------
    #
    # Combined crop covering both eyes. This is useful when the downstream task
    # should treat the eyes as a single semantic region, for example local
    # inpainting/refinement of gaze or eyelids.
    #
    # For eye targets, MediaPipe landmarks are typically sufficient, so SAM can
    # be omitted.
    #
    eyes_crop = SubjectCrop(
        name='eyes_crop',
        spec={
            'model': {
                'face_landmarker_task': MODEL_SPEC['face_landmarker_task'],
                'pose_landmarker_task': MODEL_SPEC['pose_landmarker_task'],
            },
            'params': {
                **COMMON_PARAMS,
                'target': 'eyes',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Left eye crop
    # -------------------------------------------------------------------------
    #
    # Crop only the subject's left eye.
    # Note: "left" is in subject perspective, not viewer perspective.
    #
    left_eye_crop = SubjectCrop(
        name='left_eye_crop',
        spec={
            'model': {
                'face_landmarker_task': MODEL_SPEC['face_landmarker_task'],
                'pose_landmarker_task': MODEL_SPEC['pose_landmarker_task'],
            },
            'params': {
                **COMMON_PARAMS,
                'target': 'left-eye',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Right eye crop
    # -------------------------------------------------------------------------
    #
    # Crop only the subject's right eye.
    # Note: "right" is in subject perspective, not viewer perspective.
    #
    right_eye_crop = SubjectCrop(
        name='right_eye_crop',
        spec={
            'model': {
                'face_landmarker_task': MODEL_SPEC['face_landmarker_task'],
                'pose_landmarker_task': MODEL_SPEC['pose_landmarker_task'],
            },
            'params': {
                **COMMON_PARAMS,
                'target': 'right-eye',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------
    #
    # Fan out the same input image into multiple independent SubjectCrop nodes
    # so the outputs can be compared side by side.
    #
    init_img >> [
        person_crop,
        head_crop,
        face_crop,
        eyes_crop,
        left_eye_crop,
        right_eye_crop,
    ]
