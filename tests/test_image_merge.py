from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

from morphalo.dag import DAG, NodeRef
from morphalo.nodes.image_merge import ImageMerge
from morphalo.nodes.image_output import ImageOutputMixin


@dataclass
class ImageSource(ImageOutputMixin, NodeRef):
    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict:
        return {'image': self.id}


def test_image_merge_run_combines_single_and_batch_images_in_input_order(tmp_path):
    with DAG('image_merge_run', out_dir=tmp_path):
        merge = ImageMerge(name='merge', source_names=('a', 'b', 'c'))

    out = merge.run(
        tmp_path,
        input={
            'image:2': {'image': 'c.png'},
            'image:0': {'image': 'a.png'},
            'image:1': {'images': ['b0.png', 'b1.png']},
        },
    )

    assert out['ok'] is True
    assert out['node'] == 'imagemerge'
    assert out['id'] == 'merge'
    assert out['images'] == ['a.png', 'b0.png', 'b1.png', 'c.png']
    assert out['sources'] == ['a', 'b', 'c']
    metadata = Path(out['metadata'])
    assert metadata.parent.name == 'merge'
    assert metadata.suffix == '.json'


def test_image_merge_plus_flattens_chained_expressions_and_wires_inputs(tmp_path):
    with DAG('image_merge_plus', out_dir=tmp_path) as dag:
        a = ImageSource(name='a')
        b = ImageSource(name='b')
        c = ImageSource(name='c')

        merge = a + b + c

    assert isinstance(merge, ImageMerge)
    assert merge.source_names == ('a', 'b', 'c')

    image_merges = [node for node in dag.nodes if isinstance(node, ImageMerge)]
    assert [node.source_names for node in image_merges] == [
        ('a', 'b'),
        ('a', 'b', 'c'),
    ]

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}
    assert {
        ('a', merge.id, 'image:0'),
        ('b', merge.id, 'image:1'),
        ('c', merge.id, 'image:2'),
    }.issubset(edges)


def test_image_merge_plus_reuses_same_ordered_expression_in_scope(tmp_path):
    with DAG('image_merge_reuse', out_dir=tmp_path):
        a = ImageSource(name='a')
        b = ImageSource(name='b')

        first = a + b
        second = a + b
        reversed_merge = b + a

    assert second is first
    assert reversed_merge is not first
    assert first.source_names == ('a', 'b')
    assert reversed_merge.source_names == ('b', 'a')


def test_image_merge_run_rejects_non_contiguous_inputs(tmp_path):
    with DAG('image_merge_gap', out_dir=tmp_path):
        merge = ImageMerge(name='merge')

    with pytest.raises(RuntimeError, match='expected contiguous image inputs'):
        merge.run(
            tmp_path,
            input={
                'image:0': {'image': 'a.png'},
                'image:2': {'image': 'c.png'},
            },
        )


def test_image_merge_run_rejects_payload_without_images(tmp_path):
    with DAG('image_merge_invalid', out_dir=tmp_path):
        merge = ImageMerge(name='merge')

    with pytest.raises(RuntimeError, match='has no image/images payload'):
        merge.run(tmp_path, input={'image:0': {'prompt': 'not an image'}})
