r"""
Semantic crop demo: subject, face-detail, and prompt-guided crops from one image.

This demo showcases three complementary crop families:

1) **subject and face-detail crops**
    Extract person/body regions with ``SubjectCrop`` and face-local regions with
    ``FaceCrop``.

2) **full-frame masks**
    Generate positive and negative subject masks aligned with the original image,
    suitable for inpainting or region-constrained editing.

3) **bbox-based crops with aspect ratio control**
    Expand subject regions into rectangular crops with optional ratio targets
    such as ``1:1`` or ``16:9``.

The goal is to demonstrate how specialized preprocessing nodes can provide
reusable, task-oriented outputs for downstream DAGs, and how prompt-based
AnyCrop can extract arbitrary regions such as pants and a blouse.

Why Split The Nodes?
--------------------
``SubjectCrop`` is not a generic geometric crop. It is a *semantic crop node*
that combines detection, pose, hand landmarks, and segmentation to isolate
person/body-level regions.

``FaceCrop`` handles face-local regions using face landmarks and, for the
whole-face target, SAM segmentation.

Depending on the selected node and target, the pipeline may use:

- YOLO-style person detection
- MediaPipe pose landmarks
- MediaPipe face landmarks
- MediaPipe hand landmarks
- SAM segmentation

Supported targets
-----------------
``SubjectCrop``:
- ``person``:
    full subject extraction, typically using YOLO + SAM/SAM-HQ
- ``head``:
    head-oriented region derived from face landmarks, with hair-friendly framing
- ``hands``:
    combined crop covering all visible hands of the selected subject
- ``left-hand`` / ``right-hand``:
    single-hand crops derived from MediaPipe hand landmarks and subject-guided
    masking, using image/viewer perspective

``FaceCrop``:
- ``face``:
    tighter face crop, guided by face landmarks and SAM/SAM-HQ
- ``eyes``:
    combined region covering both eyes
- ``left-eye`` / ``right-eye``:
    single-eye crops derived from MediaPipe face landmarks, using image/viewer
    perspective
- ``eyebrows``:
    combined region covering both eyebrows
- ``left-eyebrow`` / ``right-eyebrow``:
    single-eyebrow crops derived from MediaPipe face landmarks, using
    image/viewer perspective

Defined DAGs
------------

This file defines four demo DAGs:

1) ``subject_crop_masks``
    Produces full-frame positive and negative masks for the whole subject.

2) ``subject_crop``
    Demonstrates subject/body and face-local crops for multiple targets.

3) ``subject_crop_bbox_ratio``
    Demonstrates bbox-based crops with aspect-ratio expansion.

4) ``any_crop_trim``
    Demonstrates open-vocabulary prompt crops using AnyCrop, with trimmed RGBA
    outputs for detected regions such as pants and a blouse.

Conceptual graph (subject masks)
-------------------------------
                 -> SubjectCrop(target='person',
                /               mode='mask')
Input image ---+                  -> positive subject mask
                \
                 +-> SubjectCrop(target='person',
                                 mode='negative-mask')
                                    -> negative subject mask

Conceptual graph (semantic crops)
---------------------------------
                 -> SubjectCrop(target='person')        -> person crop
                /
Input image ---+-> SubjectCrop(target='head')          -> head crop
                \
                 +-> FaceCrop(target='face')           -> face crop
                  \
                   +-> FaceCrop(target='eyes')             -> eyes crop
                    +-> FaceCrop(target='left-eye')        -> left eye crop
                    +-> FaceCrop(target='right-eye')       -> right eye crop
                    +-> FaceCrop(target='eyebrows')        -> eyebrows crop
                    +-> SubjectCrop(target='hands')        -> hands crop
                    +-> SubjectCrop(target='left-hand')    -> left hand crop
                    +-> SubjectCrop(target='right-hand')   -> right hand crop

Conceptual graph (bbox + ratio)
------------------------------
                 -> SubjectCrop(target='person',
                /               crop_mode='bbox')
               /                  -> natural person bbox crop
              /
Input image -+-> SubjectCrop(target='person',
              \                 crop_mode='bbox[1:1]')
               \                  -> square person bbox crop
                \
                 +-> SubjectCrop(target='person',
                 |               crop_mode='bbox[16:9]')
                 |                  -> wide person bbox crop
                 |
                 +-> SubjectCrop(target='head',
                 |               crop_mode='bbox')
                 |                  -> natural head bbox crop
                 |
                 +-> SubjectCrop(target='head',
                 |               crop_mode='bbox[1:1]')
                 |                  -> square head bbox crop
                 |
                 +-> SubjectCrop(target='left-hand',
                                 crop_mode='bbox[1:1]')
                                    -> square left-hand bbox crop

Conceptual graph (prompt crop)
------------------------------
Input image ---+-> AnyCrop(
                   prompt='pants. trousers. jeans. blue jeans.',
                   mode='default',
                   crop_mode='trim'
                 )
                 -> lower-garment crop
                \
                 +-> AnyCrop(
                   prompt='top. tank top. sleeveless top. sleeveless shirt. black top.',
                   mode='default',
                   crop_mode='trim'
                 )
                 -> upper-garment crop

How to run
----------
Run subject masks:
    ./bin/run_dag.sh demo.06_crop --dag subject_crop_masks

Run semantic crops:
    ./bin/run_dag.sh demo.06_crop --dag subject_crop

Run bbox + ratio:
    ./bin/run_dag.sh demo.06_crop --dag subject_crop_bbox_ratio

Run prompt crop with AnyCrop:
    ./bin/run_dag.sh demo.06_crop --dag any_crop_trim

Notes
-----
- ``mode='default'``:
    returns RGBA crops. For masked crop modes, alpha encodes the selected mask.

- ``mode='mask'``:
    returns a full-frame 8-bit mask aligned to the input image.
    White pixels represent the selected region.

- ``mode='negative-mask'``:
    returns the inverse full-frame mask.
    White pixels represent everything except the selected region.

- ``crop_mode='trim'``:
    returns a tight RGBA cutout trimmed to the non-transparent selected mask.

- ``crop_mode='bbox'``:
    returns a rectangular crop including the original background, with fully
    opaque alpha.

- ``crop_mode='bbox[w:h]'``:
    expands the crop toward a target aspect ratio while keeping the selected
    region inside the crop and staying within image bounds.

    The ratio is a target, not a strict guarantee: near image borders, the final
    crop may deviate from the requested ratio.

- Crop metadata uses source-image coordinates. Image pixels are cropped locally
  unless ``crop_mode='full_frame'`` is used.

- Mask outputs are always full-frame and have the same resolution as the input.

- Hand targets combine MediaPipe hand landmarks with a subject-guided SAM mask.

- ``hands`` may contain multiple disconnected regions, one per visible hand.

- Hand detection may fail on heavily deformed, occluded, or poorly delineated
  hands; in those cases a future pose-based fallback may be useful.

- Left/right semantics follow image/viewer perspective, not anatomical subject
  perspective. ``left-eye``, ``left-eyebrow`` and ``left-hand`` mean the region
  on the left side of the image.
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import AnyCrop, FaceCrop, SubjectCrop

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User inputs (local paths, typically gitignored)
# -----------------------------------------------------------------------------
#
# Source image used for all crops in this demo.
#
INIT_IMG = '~/images/init_img_2.png'

# Model assets.
#
# Notes:
# - sam_model selects the SAM/SAM-HQ backend used for segmentation targets.
#
# Full SubjectCrop config: person/head/hand targets may need different model
# families depending on the selected target.
SUBJECT_MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'hand_landmarker_task': '~/models/mediapipe/hand_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

# FaceCrop config: all targets need face landmarks; only target='face' uses SAM.
FACE_MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

# Shared crop parameters reused across SubjectCrop and FaceCrop nodes in this demo.
#
# `mode='default'`:
#     emit an RGBA crop rather than a full-frame mask.
#
# `crop_mode='trim'`:
#     return a tight RGBA cutout trimmed to the non-transparent mask area.
#
# `box_margin`:
#     expands the SAM prompt box for segmentation-backed targets.
#
# `expansion`:
#     target-specific enlargement factor, especially meaningful for head,
#     face-detail, and hand crops.
#
COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': 0.12,
    'expansion': 1.2,
}

# Shared parameters for full-frame mask generation.
#
# Unlike crop mode, these settings operate on the mask itself:
# - dilate_radius: expands the white region
# - close_radius: fills small holes
# - smoothing_radius: softens mask edges (feathering)
# Note:
# `crop_mode` is intentionally omitted here because mask modes always produce
# full-frame outputs aligned to the original image.
MASK_PARAMS = {
    'target': 'person',
    'mode': 'mask',
    'box_margin': 0.12,
    'dilate_radius': 10,
    'close_radius': 4,
    'smoothing_radius': 7,
}

# Negative mask: inverted version of the subject mask.
# Useful when downstream nodes expect "background mask"
# instead of "foreground mask".
NEGATIVE_MASK_PARAMS = {
    **MASK_PARAMS,
    'mode': 'negative-mask',
}


with DAG(
    name='subject_crop_masks',
    out_dir=ROOT / 'outputs' / 'subject_crop_masks',
):
    # -------------------------------------------------------------------------
    # Subject masks
    # -------------------------------------------------------------------------
    #
    # Generate full-frame masks aligned to the original image.
    #
    # - subject_mask:
    #     white = selected subject region
    # - subject_negative_mask:
    #     white = everything except the selected subject region
    #
    # These masks are suitable for inpainting or region-constrained editing.

    init_img_mask = FileImage(
        name='init_img_mask',
        path=INIT_IMG,
    )

    subject_mask = SubjectCrop(
        name='subject_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **MASK_PARAMS,
            },
        },
    )

    subject_negative_mask = SubjectCrop(
        name='subject_negative_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **NEGATIVE_MASK_PARAMS,
            },
        },
    )

    init_img_mask >> [
        subject_mask,
        subject_negative_mask,
    ]

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
            'model': SUBJECT_MODEL_SPEC,
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
    # Produce a head-oriented crop. Compared to FaceCrop(target='face'), this keeps a
    # larger square region around the face, with more room for hair and framing.
    # Useful when refining portraits while preserving head silhouette/hair volume.
    #
    head_crop = SubjectCrop(
        name='head_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
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
    face_crop = FaceCrop(
        name='face_crop',
        spec={
            'model': FACE_MODEL_SPEC,
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
    # Eye targets are landmark-derived and skip SAM entirely.
    #
    eyes_crop = FaceCrop(
        name='eyes_crop',
        spec={
            'model': FACE_MODEL_SPEC,
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
    # Crop only the eye on the left side of the image.
    # Note: "left" is in image/viewer perspective, not anatomical subject
    # perspective.
    #
    left_eye_crop = FaceCrop(
        name='left_eye_crop',
        spec={
            'model': FACE_MODEL_SPEC,
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
    # Crop only the eye on the right side of the image.
    # Note: "right" is in image/viewer perspective, not anatomical subject
    # perspective.
    #
    right_eye_crop = FaceCrop(
        name='right_eye_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-eye',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Eyebrow crops
    # -------------------------------------------------------------------------
    #
    # Eyebrow targets mirror the eye targets: they are landmark-derived,
    # side-specific variants use image/viewer perspective, and SAM is skipped.
    #
    eyebrows_crop = FaceCrop(
        name='eyebrows_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'eyebrows',
            },
        },
    )

    left_eyebrow_crop = FaceCrop(
        name='left_eyebrow_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-eyebrow',
            },
        },
    )

    right_eyebrow_crop = FaceCrop(
        name='right_eyebrow_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-eyebrow',
            },
        },
    )

    # Note:
    # Hand detection relies on MediaPipe and may fail on heavily
    # deformed or occluded hands. In such cases, downstream repair
    # may require fallback strategies.

    # -------------------------------------------------------------------------
    # Hands crop
    # -------------------------------------------------------------------------
    #
    # Combined crop covering all visible hands of the selected subject.
    # Useful when both hands should be refined together as a single semantic
    # region, for example local repair of fingers, pose, or gesture.
    # Output may contain multiple disconnected regions (one per hand).
    #
    hands_crop = SubjectCrop(
        name='hands_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'hands',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Left hand crop
    # -------------------------------------------------------------------------
    #
    # Crop only the hand on the left side of the image.
    # Note: "left" is in image/viewer perspective, not anatomical subject
    # perspective.
    #
    left_hand_crop = SubjectCrop(
        name='left_hand_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-hand',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Right hand crop
    # -------------------------------------------------------------------------
    #
    # Crop only the hand on the right side of the image.
    # Note: "right" is in image/viewer perspective, not anatomical subject
    # perspective.
    #
    right_hand_crop = SubjectCrop(
        name='right_hand_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-hand',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------
    #
    # Fan out the same input image into independent semantic crop nodes so the
    # outputs can be compared side by side.
    #
    init_img >> [
        person_crop,
        head_crop,
        face_crop,
        eyes_crop,
        left_eye_crop,
        right_eye_crop,
        eyebrows_crop,
        left_eyebrow_crop,
        right_eyebrow_crop,
        hands_crop,
        left_hand_crop,
        right_hand_crop,
    ]


with DAG(
    name='subject_crop_bbox_ratio',
    out_dir=ROOT / 'outputs' / 'subject_crop_bbox_ratio',
):
    # -------------------------------------------------------------------------
    # Input source
    # -------------------------------------------------------------------------
    #
    # Reuse the same local image so the differences between crop modes remain
    # directly comparable.
    # Ratio expansion is best-effort:
    # the crop will never exceed image boundaries.
    #
    init_img_bbox = FileImage(
        name='init_img_bbox',
        path=INIT_IMG,
    )

    # -------------------------------------------------------------------------
    # Baseline bbox crop
    # -------------------------------------------------------------------------
    #
    # Natural bbox crop around the detected person, with no aspect-ratio
    # expansion. This is the reference output for comparison.
    #
    person_bbox = SubjectCrop(
        name='person_bbox',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
                'crop_mode': 'bbox',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Square bbox crop
    # -------------------------------------------------------------------------
    #
    # Expand the effective crop box toward a square aspect ratio while keeping
    # the detected subject fully inside the crop and staying within image bounds.
    #
    # The requested ratio is a target, not a hard guarantee: if the image borders
    # leave insufficient room, the final crop may deviate from 1:1.
    #
    person_bbox_square = SubjectCrop(
        name='person_bbox_square',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
                'crop_mode': 'bbox[1:1]',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Wide bbox crop
    # -------------------------------------------------------------------------
    #
    # Expand the crop toward a wide framing. Useful when the person is relatively
    # small in the frame and you want more surrounding context without switching
    # to a full-frame crop.
    #
    person_bbox_wide = SubjectCrop(
        name='person_bbox_wide',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
                'crop_mode': 'bbox[16:9]',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Head bbox crop
    # -------------------------------------------------------------------------
    #
    # Baseline head-oriented bbox crop. Compared to the person crop, this starts
    # from the head framing logic and is useful to inspect how ratio expansion
    # behaves on a smaller semantic target.
    #
    head_bbox = SubjectCrop(
        name='head_bbox',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'head',
                'crop_mode': 'bbox',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Square head bbox crop
    # -------------------------------------------------------------------------
    #
    # Head crop expanded toward a square framing. This is often useful for local
    # portrait repair workflows where downstream nodes expect roughly square
    # crops around the head.
    #
    head_bbox_square = SubjectCrop(
        name='head_bbox_square',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'head',
                'crop_mode': 'bbox[1:1]',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Square left-hand bbox crop
    # -------------------------------------------------------------------------
    #
    # Isolate the subject's left hand and expand the crop toward a square box.
    # This is a practical setup for local hand repair.
    #
    left_hand_bbox_square = SubjectCrop(
        name='left_hand_bbox_square',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-hand',
                'crop_mode': 'bbox[1:1]',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------
    #
    # Fan out the same input image into several bbox-oriented crops so the effect
    # of aspect-ratio-guided expansion can be inspected side by side.
    #
    init_img_bbox >> [
        person_bbox,
        person_bbox_square,
        person_bbox_wide,
        head_bbox,
        head_bbox_square,
        left_hand_bbox_square,
    ]


with DAG(
    name='any_crop_trim',
    out_dir=ROOT / 'outputs' / 'any_crop_trim',
):
    # -------------------------------------------------------------------------
    # Input source
    # -------------------------------------------------------------------------
    #
    # Use the same demo image and show how AnyCrop can localize arbitrary
    # regions from flexible text prompts composed of multiple synonymous or
    # related concepts.
    #
    init_img_any = FileImage(
        name='init_img_any',
        path=INIT_IMG,
    )

    # -------------------------------------------------------------------------
    # Prompt-based pants crop
    # -------------------------------------------------------------------------
    #
    # Use a flexible prompt with multiple related concepts so Grounding DINO
    # can match the strongest visual label among several alternatives.
    #
    any_crop_pants = AnyCrop(
        name='any_crop_pants',
        spec={
            'model': {
                'grounding_model': 'IDEA-Research/grounding-dino-tiny',
                'sam_model': 'facebook/sam-vit-large',
            },
            'prompt': 'pants. trousers. jeans. blue jeans.',
            'params': {
                'mode': 'default',
                'crop_mode': 'trim',
                'box_margin': 0.08,
                'box_threshold': 0.20,
                'text_threshold': 0.15,
                'select': 'best',
            },
        },
    )

    # -------------------------------------------------------------------------
    # Prompt-based upper-body garment crop
    # -------------------------------------------------------------------------
    #
    # Use a flexible prompt for the upper garment as well. This helps when the
    # exact clothing label is uncertain, for example top vs tank top vs
    # sleeveless top.
    #
    any_crop_top = AnyCrop(
        name='any_crop_top',
        spec={
            'model': {
                'grounding_model': 'IDEA-Research/grounding-dino-tiny',
                'sam_model': 'facebook/sam-vit-large',
            },
            'prompt': 'top. black top. tank top. sleeveless top. sleeveless shirt.',
            'params': {
                'mode': 'default',
                'crop_mode': 'trim',
                'box_margin': 0.08,
                'box_threshold': 0.20,
                'text_threshold': 0.15,
                'select': 'best',
            },
        },
    )

    init_img_any >> [
        any_crop_pants,
        any_crop_top,
    ]
