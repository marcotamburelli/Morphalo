"""
HumanSegmentCrop demo.

This demo showcases semantic human parsing with ``HumanSegmentCrop``. Unlike
``SubjectCrop``, this node does not use YOLO, MediaPipe or SAM for the first
implementation phase: it uses the FASHN SegFormer human parser to produce
pixel-wise semantic masks.

Defined DAGs
------------
``human_segment_masks``
    Full-frame positive and negative person masks.

``human_segment_crop``
    Trimmed RGBA crops for semantic parser targets such as person, body, skin,
    clothes, head and an explicit target sequence.

``human_segment_bbox_ratio``
    Rectangular bbox crops and aspect-ratio-guided crops from parser masks.

How to run
----------
    ./bin/run_dag.sh demo.06_B_crop --dag human_segment_masks
    ./bin/run_dag.sh demo.06_B_crop --dag human_segment_crop
    ./bin/run_dag.sh demo.06_B_crop --dag human_segment_bbox_ratio
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import HumanSegmentCrop

ROOT = Path(__file__).resolve().parents[1]

INIT_IMG = '~/images/init_img_1.png'

MODEL_SPEC = {
    'segment_model': 'fashn-ai/fashn-human-parser',
    'device': 'cuda',
    'dtype': 'float16',
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
    name='human_segment_masks',
    out_dir=ROOT / 'outputs' / '06_B_human_segment_masks',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_mask = HumanSegmentCrop(
        name='person_mask',
        spec={
            'model': MODEL_SPEC,
            'params': MASK_PARAMS,
            'debug': {
                'save_debug': True,
            },
        },
    )

    background_mask = HumanSegmentCrop(
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
    name='human_segment_crop',
    out_dir=ROOT / 'outputs' / '06_B_human_segment_crop',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_crop = HumanSegmentCrop(
        name='person_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'person',
            },
        },
    )

    body_crop = HumanSegmentCrop(
        name='body_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'body',
            },
        },
    )

    skin_crop = HumanSegmentCrop(
        name='skin_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'skin',
            },
        },
    )

    clothes_crop = HumanSegmentCrop(
        name='clothes_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'clothes',
            },
        },
    )

    head_crop = HumanSegmentCrop(
        name='head_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'head',
            },
        },
    )

    mixed_parts_crop = HumanSegmentCrop(
        name='mixed_parts_crop',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': [
                    'head',
                    'hands',
                    'feet',
                    'dress',
                ],
            },
        },
    )

    init_img >> [
        person_crop,
        body_crop,
        skin_crop,
        clothes_crop,
        head_crop,
        mixed_parts_crop,
    ]


with DAG(
    name='human_segment_bbox_ratio',
    out_dir=ROOT / 'outputs' / '06_B_human_segment_bbox_ratio',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    person_bbox = HumanSegmentCrop(
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

    person_bbox_square = HumanSegmentCrop(
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

    clothes_bbox = HumanSegmentCrop(
        name='clothes_bbox',
        spec={
            'model': MODEL_SPEC,
            'params': {
                **COMMON_PARAMS,
                'target': 'clothes',
                'crop_mode': 'bbox',
            },
        },
    )

    head_bbox_square = HumanSegmentCrop(
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
        clothes_bbox,
        head_bbox_square,
    ]
