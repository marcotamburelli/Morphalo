"""
SubjectCrop2 incremental demo.

This temporary demo mirrors the simple target coverage currently implemented by
``SubjectCrop2``. It is intended to grow incrementally and eventually replace
``demo/06_A_crop.py`` when the Sapiens2-based pipeline is complete.

Defined DAGs
------------
``subject_crop2_masks``
    Full-frame positive and negative masks for the detected subject.

``subject_crop2``
    Trimmed RGBA crops for person and head.

``subject_crop2_hands``
    Trimmed RGBA crops for both hands and image-relative left/right hands.

``subject_crop2_feet``
    Trimmed RGBA crops for both feet and image-relative left/right feet.

``subject_crop2_arms``
    Trimmed RGBA crops for both arms and image-relative left/right arms.

``subject_crop2_legs``
    Trimmed RGBA crops for both legs and image-relative left/right legs.

``subject_crop2_bbox_ratio``
    Rectangular bbox crops and aspect-ratio-guided crops.

How to run
----------
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2_masks
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2_hands
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2_feet
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2_arms
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2_legs
    ./bin/run_dag.sh demo.06_A_crop2 --dag subject_crop2_bbox_ratio
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import SubjectCrop2

ROOT = Path(__file__).resolve().parents[1]

INIT_IMG = '~/images/2026-03-11_180553.png'

SUBJECT_MODEL_SPEC = {
    'yolo_model': 'yolov8n.pt',
    'segment_model': 'facebook/sapiens2-seg-0.4b',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
    'device': 'cuda',
    'dtype': 'bfloat16',
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


with DAG(
    name='subject_crop2_masks',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2_masks',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    subject_mask = SubjectCrop2(
        name='subject_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': MASK_PARAMS,
            'debug': {
                'save_debug': True,
            },
        },
    )

    subject_negative_mask = SubjectCrop2(
        name='subject_negative_mask',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': NEGATIVE_MASK_PARAMS,
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        subject_mask,
        subject_negative_mask,
    ]


with DAG(
    name='subject_crop2',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_crop = SubjectCrop2(
        name='person_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'person',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    head_crop = SubjectCrop2(
        name='head_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'head',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        person_crop,
        head_crop,
    ]


with DAG(
    name='subject_crop2_hands',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2_hands',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    hands_crop = SubjectCrop2(
        name='hands_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'hands',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_hand_crop = SubjectCrop2(
        name='left_hand_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'left-hand',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_hand_crop = SubjectCrop2(
        name='right_hand_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'right-hand',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        hands_crop,
        left_hand_crop,
        right_hand_crop,
    ]


with DAG(
    name='subject_crop2_feet',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2_feet',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    feet_crop = SubjectCrop2(
        name='feet_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'feet',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_foot_crop = SubjectCrop2(
        name='left_foot_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'left-foot',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_foot_crop = SubjectCrop2(
        name='right_foot_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
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
    name='subject_crop2_arms',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2_arms',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    arms_crop = SubjectCrop2(
        name='arms_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'arms',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_arm_crop = SubjectCrop2(
        name='left_arm_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'left-arm',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_arm_crop = SubjectCrop2(
        name='right_arm_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
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
    name='subject_crop2_legs',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2_legs',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    legs_crop = SubjectCrop2(
        name='legs_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'legs',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    left_leg_crop = SubjectCrop2(
        name='left_leg_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
                'target': 'left-leg',
            },
            'debug': {
                'save_debug': True,
            },
        },
    )

    right_leg_crop = SubjectCrop2(
        name='right_leg_crop',
        spec={
            'model': SUBJECT_MODEL_SPEC,
            'params': {
                **PART_PARAMS,
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
    name='subject_crop2_bbox_ratio',
    out_dir=ROOT / 'outputs' / '06_A_subject_crop2_bbox_ratio',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_bbox = SubjectCrop2(
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

    person_bbox_square = SubjectCrop2(
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

    head_bbox_square = SubjectCrop2(
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

    left_hand_bbox_square = SubjectCrop2(
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
        head_bbox_square,
        left_hand_bbox_square,
    ]
