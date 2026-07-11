"""
SubjectCrop and FaceCrop demo.

This split demo covers semantic person/body crops with ``SubjectCrop`` and
face-local crops with ``FaceCrop``.

Defined DAGs
------------
``subject_crop_masks``
    Full-frame positive and negative masks for the detected subject.

``subject_crop``
    Trimmed RGBA crops for person, head, face, eyes, eyebrows and hands.

``subject_crop_bbox_ratio``
    Rectangular bbox crops and aspect-ratio-guided crops.

How to run
----------
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_masks
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_bbox_ratio
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import FaceCrop, SubjectCrop

ROOT = Path(__file__).resolve().parents[1]

INIT_IMG = '~/images/init_img_3.png'

SUBJECT_MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'hand_landmarker_task': '~/models/mediapipe/hand_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

FACE_MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': 0.12,
    'expansion': 1.2,
}

MASK_PARAMS = {
    'target': 'person',
    'mode': 'mask',
    'box_margin': 0.12,
    'dilate_radius': 10,
    'close_radius': 4,
    'smoothing_radius': 7,
}

NEGATIVE_MASK_PARAMS = {
    **MASK_PARAMS,
    'mode': 'negative-mask',
}


with DAG(
    name='subject_crop_masks',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_masks',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    subject_mask = SubjectCrop(
        name='subject_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': MASK_PARAMS,
        },
    )

    subject_negative_mask = SubjectCrop(
        name='subject_negative_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': NEGATIVE_MASK_PARAMS,
        },
    )

    init_img >> [
        subject_mask,
        subject_negative_mask,
    ]


with DAG(
    name='subject_crop',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

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
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_bbox_ratio',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

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

    init_img >> [
        person_bbox,
        person_bbox_square,
        person_bbox_wide,
        head_bbox,
        head_bbox_square,
        left_hand_bbox_square,
    ]
