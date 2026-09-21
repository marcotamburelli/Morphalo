from __future__ import annotations

import pytest

from morphalo.dag import DAG
from morphalo.nodes.foundation.qwen_image_edit_plus import QwenImageEditPlus
from morphalo.nodes.foundation.wiring import ImageSequenceBundle


def test_image_sequence_bundle_allows_empty_sequence() -> None:
    bundle = ImageSequenceBundle([], input={})

    assert bundle.images == []
    assert bundle.metadata == []


def test_qwen_image_edit_plus_requires_node_level_image_input(tmp_path) -> None:
    with DAG('qwen_image_edit_plus_requires_image', out_dir=tmp_path):
        node = QwenImageEditPlus(
            name='edit',
            spec={'prompt': 'Make a cinematic version of the source image.'},
        )

    with pytest.raises(
        ValueError,
        match='QwenImageEditPlus requires at least one input image',
    ):
        node.run(output_dir=tmp_path, input={})
