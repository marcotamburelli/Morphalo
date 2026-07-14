"""
Sapiens2SegmentCrop demo.

This demo showcases Sapiens2 body-part segmentation with
``Sapiens2SegmentCrop``. Unlike ``FashnSegmentCrop``, this node uses the
Sapiens2 29-class taxonomy, including side-specific labels such as
``left-hand`` and ``right-foot`` in image/viewer perspective, explicit
anatomical targets such as ``anatomical-left-hand``, and fine-grained labels
for socks, shoes, lips, teeth and tongue.

Defined DAGs
------------
``sapiens2_segment_masks``
    Full-frame positive and negative person masks.

``sapiens2_segment_crop``
    Trimmed RGBA crops for semantic Sapiens2 targets such as person, skin,
    clothes, head, hands, feet, side-specific hands/feet, arms, legs and
    explicit target sequences.

``sapiens2_segment_bbox_ratio``
    Rectangular bbox crops and aspect-ratio-guided crops from Sapiens2 masks.

How to run
----------
    ./bin/run_dag.sh demo.06_C_crop --dag sapiens2_segment_masks
    ./bin/run_dag.sh demo.06_C_crop --dag sapiens2_segment_crop
    ./bin/run_dag.sh demo.06_C_crop --dag sapiens2_segment_bbox_ratio
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import Sapiens2SegmentCrop

ROOT = Path(__file__).resolve().parents[1]

INIT_IMG = '~/images/init_img_6.png'

MODEL_SPEC = {
    'segment_model': 'facebook/sapiens2-seg-0.4b',
    'device': 'cuda',
    'dtype': 'bfloat16',
}

COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': '8%',
}

MASK_PARAMS = {
    'target': 'person',
    'mode': 'mask',
    'dilate_radius': 8,
    'close_radius': 3,
    'smoothing_radius': 5,
}

NEGATIVE_MASK_PARAMS = {
    **MASK_PARAMS,
    'mode': 'negative-mask',
}


with DAG(
    name='sapiens2_segment_masks',
    out_dir=ROOT / 'outputs' / '06_C_sapiens2_segment_masks',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_mask = Sapiens2SegmentCrop(
        name='person_mask',
        spec={
            'model': MODEL_SPEC,
            'params': MASK_PARAMS,
            'debug': {
                'save_debug': True,
            },
        },
    )

    background_mask = Sapiens2SegmentCrop(
        name='background_mask',
        spec={
            'model': MODEL_SPEC,
            'params': NEGATIVE_MASK_PARAMS,
            'debug': {
                'save_debug': True,
            },
        },
    )

    init_img >> [
        person_mask,
        background_mask,
    ]


with DAG(
    name='sapiens2_segment_crop',
    out_dir=ROOT / 'outputs' / '06_C_sapiens2_segment_crop',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_crop = Sapiens2SegmentCrop(
        name='person_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
            },
        },
    )

    skin_crop = Sapiens2SegmentCrop(
        name='skin_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'skin',
            },
        },
    )

    clothes_crop = Sapiens2SegmentCrop(
        name='clothes_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'clothes',
            },
        },
    )

    head_crop = Sapiens2SegmentCrop(
        name='head_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'head',
            },
        },
    )

    hands_crop = Sapiens2SegmentCrop(
        name='hands_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'hands',
            },
        },
    )

    left_hand_crop = Sapiens2SegmentCrop(
        name='left_hand_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-hand',
            },
        },
    )

    right_hand_crop = Sapiens2SegmentCrop(
        name='right_hand_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-hand',
            },
        },
    )

    anatomical_left_hand_crop = Sapiens2SegmentCrop(
        name='anatomical_left_hand_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-left-hand',
            },
        },
    )

    anatomical_right_hand_crop = Sapiens2SegmentCrop(
        name='anatomical_right_hand_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-right-hand',
            },
        },
    )

    feet_crop = Sapiens2SegmentCrop(
        name='feet_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'feet',
            },
        },
    )

    left_foot_crop = Sapiens2SegmentCrop(
        name='left_foot_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-foot',
            },
        },
    )

    right_foot_crop = Sapiens2SegmentCrop(
        name='right_foot_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-foot',
            },
        },
    )

    anatomical_left_foot_crop = Sapiens2SegmentCrop(
        name='anatomical_left_foot_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-left-foot',
            },
        },
    )

    anatomical_right_foot_crop = Sapiens2SegmentCrop(
        name='anatomical_right_foot_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'anatomical-right-foot',
            },
        },
    )

    arms_crop = Sapiens2SegmentCrop(
        name='arms_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'arms',
            },
        },
    )

    left_arm_crop = Sapiens2SegmentCrop(
        name='left_arm_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-arm',
            },
        },
    )

    right_arm_crop = Sapiens2SegmentCrop(
        name='right_arm_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-arm',
            },
        },
    )

    legs_crop = Sapiens2SegmentCrop(
        name='legs_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'legs',
            },
        },
    )

    left_leg_crop = Sapiens2SegmentCrop(
        name='left_leg_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'left-leg',
            },
        },
    )

    right_leg_crop = Sapiens2SegmentCrop(
        name='right_leg_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'right-leg',
            },
        },
    )

    mixed_parts_crop = Sapiens2SegmentCrop(
        name='mixed_parts_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': [
                    'head',
                    'left-hand',
                    'right-foot',
                    'upper-clothing',
                    'eyeglass',
                ],
            },
        },
    )

    init_img >> [
        person_crop,
        skin_crop,
        clothes_crop,
        head_crop,
        hands_crop,
        left_hand_crop,
        right_hand_crop,
        anatomical_left_hand_crop,
        anatomical_right_hand_crop,
        feet_crop,
        left_foot_crop,
        right_foot_crop,
        anatomical_left_foot_crop,
        anatomical_right_foot_crop,
        arms_crop,
        left_arm_crop,
        right_arm_crop,
        legs_crop,
        left_leg_crop,
        right_leg_crop,
        mixed_parts_crop,
    ]


with DAG(
    name='sapiens2_segment_bbox_ratio',
    out_dir=ROOT / 'outputs' / '06_C_sapiens2_segment_bbox_ratio',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_bbox = Sapiens2SegmentCrop(
        name='person_bbox',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
                'crop_mode': 'bbox',
            },
        },
    )

    person_bbox_square = Sapiens2SegmentCrop(
        name='person_bbox_square',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
                'crop_mode': 'bbox[1:1]',
            },
        },
    )

    upper_body_bbox = Sapiens2SegmentCrop(
        name='upper_body_bbox',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'upper-body',
                'crop_mode': 'bbox',
            },
        },
    )

    head_bbox_square = Sapiens2SegmentCrop(
        name='head_bbox_square',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'head',
                'crop_mode': 'bbox[1:1]',
            },
        },
    )

    init_img >> [
        person_bbox,
        person_bbox_square,
        upper_body_bbox,
        head_bbox_square,
    ]
