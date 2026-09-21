from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

from morphalo.dag import DAG, NodeRef
from morphalo.nodes.prompt_merge import PromptMerge
from morphalo.nodes.prompt_output import PromptOutputMixin


@dataclass
class PromptSource(PromptOutputMixin, NodeRef):
    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict:
        return {'prompt': self.id, 'negative_prompt': ''}


def test_prompt_merge_run_concatenates_prompt_fields_in_input_order(tmp_path):
    with DAG('prompt_merge_run', out_dir=tmp_path):
        merge = PromptMerge(name='merge', source_names=('a', 'b', 'c'))

    out = merge.run(
        tmp_path,
        input={
            'prompt:2': {
                'prompt': 'pose',
                'negative_prompt': 'low quality',
            },
            'prompt:0': {
                'prompt': 'subject',
                'prompt_2': 'subject style',
                'negative_prompt': 'bad anatomy',
                'negative_prompt_2': 'flat lighting',
            },
            'prompt:1': {
                'prompt': 'wardrobe',
                'prompt_2': 'wardrobe style',
                'negative_prompt_2': 'washed out',
            },
        },
    )

    assert out['ok'] is True
    assert out['node'] == 'promptmerge'
    assert out['id'] == 'merge'
    assert out['prompt'] == 'subject\nwardrobe\npose'
    assert out['prompt_2'] == 'subject style\nwardrobe style'
    assert out['negative_prompt'] == 'bad anatomy, low quality'
    assert out['negative_prompt_2'] == 'flat lighting, washed out'
    assert out['sources'] == ['a', 'b', 'c']
    metadata = Path(out['metadata'])
    assert metadata.parent.name == 'merge'
    assert metadata.suffix == '.json'


def test_prompt_merge_plus_flattens_chained_expressions_and_wires_inputs(tmp_path):
    with DAG('prompt_merge_plus', out_dir=tmp_path) as dag:
        a = PromptSource(name='a')
        b = PromptSource(name='b')
        c = PromptSource(name='c')

        merge = a + b + c

    assert isinstance(merge, PromptMerge)
    assert merge.source_names == ('a', 'b', 'c')

    prompt_merges = [node for node in dag.nodes if isinstance(node, PromptMerge)]
    assert [node.source_names for node in prompt_merges] == [
        ('a', 'b'),
        ('a', 'b', 'c'),
    ]

    edges = {(e.node_from, e.node_to, e.input_id) for e in dag.edges}
    assert {
        ('a', merge.id, 'prompt:0'),
        ('b', merge.id, 'prompt:1'),
        ('c', merge.id, 'prompt:2'),
    }.issubset(edges)


def test_prompt_merge_plus_reuses_same_ordered_expression_in_scope(tmp_path):
    with DAG('prompt_merge_reuse', out_dir=tmp_path):
        a = PromptSource(name='a')
        b = PromptSource(name='b')

        first = a + b
        second = a + b
        reversed_merge = b + a

    assert second is first
    assert reversed_merge is not first
    assert first.source_names == ('a', 'b')
    assert reversed_merge.source_names == ('b', 'a')


def test_prompt_merge_run_rejects_non_contiguous_inputs(tmp_path):
    with DAG('prompt_merge_gap', out_dir=tmp_path):
        merge = PromptMerge(name='merge')

    with pytest.raises(RuntimeError, match='expected contiguous prompt inputs'):
        merge.run(
            tmp_path,
            input={
                'prompt:0': {'prompt': 'a'},
                'prompt:2': {'prompt': 'c'},
            },
        )


def test_prompt_merge_run_rejects_payload_without_prompt_fields(tmp_path):
    with DAG('prompt_merge_invalid', out_dir=tmp_path):
        merge = PromptMerge(name='merge')

    with pytest.raises(RuntimeError, match='has no prompt fields'):
        merge.run(tmp_path, input={'prompt:0': {'image': 'not-a-prompt.png'}})
