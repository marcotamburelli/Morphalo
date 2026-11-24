import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from morphalo.dag.core import NodeGroup
from morphalo.nodes.preprocess import AlphaMask


def test_alpha_mask_uses_opacity_and_keeps_crop_dimensions(tmp_path):
    source_path = tmp_path / 'crop.png'
    crop = Image.new('RGBA', (3, 1))
    crop.putdata([(255, 255, 255, 0), (0, 255, 0, 100), (0, 0, 0, 255)])
    crop.save(source_path)

    with NodeGroup('test'):
        node = AlphaMask(name='mask')
    out = node.run(tmp_path, input={'default': {'image': str(source_path)}})

    with Image.open(out['image']) as mask:
        assert mask.mode == 'L'
        assert mask.size == (3, 1)
        np.testing.assert_array_equal(np.asarray(mask), [[0, 100, 255]])
    assert out['input_size'] == out['output_size'] == [3, 1]
    assert json.loads(Path(out['metadata']).read_text())['image'] == out['image']


def test_alpha_mask_closes_erodes_and_smooths(tmp_path):
    source_path = tmp_path / 'crop.png'
    alpha = np.zeros((25, 25), dtype=np.uint8)
    alpha[5:20, 5:20] = 255
    alpha[12, 12] = 0
    crop = Image.new('RGBA', (25, 25))
    crop.putalpha(Image.fromarray(alpha))
    crop.save(source_path)

    params = {'close_radius': 1, 'dilate_radius': -2, 'smoothing_radius': 1}
    with NodeGroup('test'):
        node = AlphaMask(name='mask', path=source_path, spec={'params': params})
    out = node.run(tmp_path)

    with Image.open(out['image']) as mask:
        assert mask.getpixel((12, 12)) == 255
        assert mask.getpixel((12, 5)) == 0
        assert 0 < mask.getpixel((12, 6)) < 255
    assert out['params'] == params


def test_alpha_mask_rejects_images_without_transparency(tmp_path):
    source_path = tmp_path / 'rgb.png'
    Image.new('RGB', (2, 2)).save(source_path)

    with NodeGroup('test'):
        node = AlphaMask(name='mask', path=source_path)
    with pytest.raises(ValueError, match='must have alpha or transparency data'):
        node.run(tmp_path)
