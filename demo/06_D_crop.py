"""
AnyCrop demo.

This split demo covers prompt-guided open-vocabulary crops with ``AnyCrop``.
Use it when the target region is not one of the built-in human/body targets and
can be described by text, such as a garment or prop.

Defined DAGs
------------
``any_crop_trim``
    Trimmed RGBA crops for text-described clothing regions.

How to run
----------
    ./bin/run_dag.sh demo.06_D_crop --dag any_crop_trim
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import AnyCrop

ROOT = Path(__file__).resolve().parents[1]

INIT_IMG = '~/images/init_img_2.png'

MODEL_SPEC = {
    'grounding_model': 'IDEA-Research/grounding-dino-tiny',
    'sam_model': 'facebook/sam-vit-large',
}

COMMON_PARAMS = {
    'mode': 'default',
    'crop_mode': 'trim',
    'box_margin': '8%',
    'box_threshold': 0.20,
    'text_threshold': 0.15,
    'select': 'best',
}


with DAG(
    name='any_crop_trim',
    out_dir=ROOT / 'outputs' / '06_D_any_crop_trim',
):
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    pants_crop = AnyCrop(
        name='pants_crop',
        spec={
            'model': MODEL_SPEC,
            'prompt': 'pants. trousers. jeans. blue jeans.',
            'params': COMMON_PARAMS,
        },
    )

    top_crop = AnyCrop(
        name='top_crop',
        spec={
            'model': MODEL_SPEC,
            'prompt': 'top. black top. tank top. sleeveless top. sleeveless shirt.',
            'params': COMMON_PARAMS,
        },
    )

    init_img >> [
        pants_crop,
        top_crop,
    ]
