"""
SubjectCrop and FaceCrop demo.

This split demo covers semantic person/body crops with ``SubjectCrop`` and
face-local crops with ``FaceCrop``.

Defined DAGs
------------
``subject_crop_masks``
    Full-frame positive and negative masks for the detected subject.

``subject_crop``
    Trimmed RGBA crops for person and head.

``subject_crop_head``
    Trimmed RGBA crop for the head.

``subject_crop_hands``
    Trimmed RGBA crops for both hands and image-relative left/right hands.

``subject_crop_arms``
    Trimmed RGBA crops for both arms and image-relative left/right arms.

``subject_crop_legs``
    Trimmed RGBA crops for both legs and image-relative left/right legs.

``subject_crop_feet``
    Trimmed RGBA crops for both feet and image-relative left/right feet.

``face_crop_features``
    Trimmed RGBA FaceCrop outputs for face, eyes and eyebrows, including
    image-relative and explicit anatomical side examples.

``subject_crop_bbox_ratio``
    Rectangular bbox crops and aspect-ratio-guided crops.

How to run
----------
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_masks
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_head
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_hands
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_arms
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_legs
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_feet
    ./bin/run_dag.sh demo.06_A_crop --dag face_crop_features
    ./bin/run_dag.sh demo.06_A_crop --dag subject_crop_bbox_ratio
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import FaceCrop, SubjectCrop

ROOT = Path(__file__).resolve().parents[1]

INIT_IMG = '~/images/init_img_1.png'

SUBJECT_MODEL_SPEC = {
    'yolo_model': 'yolov8n.pt',
    'segment_model': 'facebook/sapiens2-seg-0.4b',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
    'device': 'cuda',
    'dtype': 'bfloat16',
}

FACE_MODEL_SPEC = {
    'sam_model': 'facebook/sam-vit-large',
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': '12%',
}

SHAPE_CLEANUP_PARAMS = {
    'morph_open_radius': 1,
    'fill_holes': 'all',
    'min_component_area': '150',
}

PART_PARAMS = {
    **COMMON_PARAMS,
    'postprocess': SHAPE_CLEANUP_PARAMS,
}

ARM_PARAMS = {
    **PART_PARAMS,
}

LEG_PARAMS = {
    **PART_PARAMS,
}

HEAD_PARAMS = {
    **PART_PARAMS,
}

FOOT_PARAMS = {
    **PART_PARAMS,
}

MASK_PARAMS = {
    'target': 'person',
    'mode': 'mask',
    'box_margin': '12%',
    'dilate_radius': 10,
    'close_radius': 4,
    'smoothing_radius': 7,
}

NEGATIVE_MASK_PARAMS = {
    **MASK_PARAMS,
    'mode': 'negative-mask',
}

DEBUG_SPEC = {
    'save_debug': True,
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
            'debug': DEBUG_SPEC,
        },
    )

    subject_negative_mask = SubjectCrop(
        name='subject_negative_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': NEGATIVE_MASK_PARAMS,
            'debug': DEBUG_SPEC,
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
                **PART_PARAMS,
                'target': 'person',
            },
            'debug': DEBUG_SPEC,
        },
    )

    head_crop = SubjectCrop(
        name='head_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'head',
            },
            'debug': DEBUG_SPEC,
        },
    )

    init_img >> [
        person_crop,
        head_crop,
    ]


with DAG(
    name='subject_crop_head',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_head',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    head_crop = SubjectCrop(
        name='head_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **HEAD_PARAMS,
                'target': 'head',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        head_crop,
    ]


with DAG(
    name='subject_crop_hands',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_hands',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    hands_crop = SubjectCrop(
        name='hands_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'hands',
            },
            'debug': DEBUG_SPEC,
        },
    )

    left_hand_crop = SubjectCrop(
        name='left_hand_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'left-hand',
            },
            'debug': DEBUG_SPEC,
        },
    )

    right_hand_crop = SubjectCrop(
        name='right_hand_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'right-hand',
            },
            'debug': DEBUG_SPEC,
        },
    )

    init_img >> [
        hands_crop,
        left_hand_crop,
        right_hand_crop,
    ]


with DAG(
    name='subject_crop_feet',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_feet',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    feet_crop = SubjectCrop(
        name='feet_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **FOOT_PARAMS,
                'target': 'feet',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_foot_crop = SubjectCrop(
        name='left_foot_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **FOOT_PARAMS,
                'target': 'left-foot',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_foot_crop = SubjectCrop(
        name='right_foot_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **FOOT_PARAMS,
                'target': 'right-foot',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        feet_crop,
        left_foot_crop,
        right_foot_crop,
    ]


with DAG(
    name='face_crop_features',
    out_dir=ROOT / 'outputs' / '06_A_face_crop_features',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    face_crop = FaceCrop(
        name='face_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'face',
                'postprocess': SHAPE_CLEANUP_PARAMS,
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

    anatomical_left_eye_crop = FaceCrop(
        name='anatomical_left_eye_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-left-eye',
            },
        },
    )

    anatomical_right_eye_crop = FaceCrop(
        name='anatomical_right_eye_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-right-eye',
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

    anatomical_left_eyebrow_crop = FaceCrop(
        name='anatomical_left_eyebrow_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-left-eyebrow',
            },
        },
    )

    anatomical_right_eyebrow_crop = FaceCrop(
        name='anatomical_right_eyebrow_crop',
        spec={
            'model': FACE_MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-right-eyebrow',
            },
        },
    )

    init_img >> [
        face_crop,
        eyes_crop,
        left_eye_crop,
        right_eye_crop,
        anatomical_left_eye_crop,
        anatomical_right_eye_crop,
        eyebrows_crop,
        left_eyebrow_crop,
        right_eyebrow_crop,
        anatomical_left_eyebrow_crop,
        anatomical_right_eyebrow_crop,
    ]


with DAG(
    name='subject_crop_arms',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_arms',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    arms_crop = SubjectCrop(
        name='arms_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **ARM_PARAMS,
                'target': 'arms',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_arm_crop = SubjectCrop(
        name='left_arm_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **ARM_PARAMS,
                'target': 'left-arm',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_arm_crop = SubjectCrop(
        name='right_arm_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **ARM_PARAMS,
                'target': 'right-arm',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        arms_crop,
        left_arm_crop,
        right_arm_crop,
    ]


with DAG(
    name='subject_crop_legs',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop_legs',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    legs_crop = SubjectCrop(
        name='legs_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **LEG_PARAMS,
                'target': 'legs',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_leg_crop = SubjectCrop(
        name='left_leg_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **LEG_PARAMS,
                'target': 'left-leg',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_leg_crop = SubjectCrop(
        name='right_leg_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **LEG_PARAMS,
                'target': 'right-leg',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        legs_crop,
        left_leg_crop,
        right_leg_crop,
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
                **PART_PARAMS,
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
                **PART_PARAMS,
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
                **PART_PARAMS,
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
                **PART_PARAMS,
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
                **PART_PARAMS,
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
                **PART_PARAMS,
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
