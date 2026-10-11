import numpy as np
import pytest
from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.process.image_stack import (
    ImageStack, _directional_alpha_ramp, expand_css_values,
)


@pytest.mark.parametrize('value, expected', [
    (3, (3, 3, 3, 3)),
    (('5%',), ('5%',) * 4),
    ((1, '2px'), (1, '2px', 1, '2px')),
    ((1, '2%', 3), (1, '2%', 3, '2%')),
    ([1, 2, 3, 4], (1, 2, 3, 4)),
])
def test_css_edge_expansion(value, expected):
    assert expand_css_values(value) == expected


@pytest.mark.parametrize('value', [(), (1, 2, 3, 4, 5), (1, 'bad')])
@pytest.mark.parametrize('option', ['feather', 'corner_radius'])
def test_stack_rejects_invalid_edge_values(tmp_path, value, option):
    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(name='stack')
        with pytest.raises((ValueError, TypeError)):
            stack.image(0, **{option: value})


def test_directional_feather_uses_axis_specific_percentages():
    alpha = np.asarray(_directional_alpha_ramp(
        100, 50, feather=('20%', '20%', 0, 0), corner_radius=0,
    ))
    assert alpha[0, 0] == 0
    assert alpha[-1, 0] == 255
    assert alpha[-1, -1] == 0
    assert alpha[5, 30] == alpha[30, 89]
    assert alpha[10, 30] == alpha[30, 79] == 255


def test_each_corner_can_round_independently_without_feather():
    for idx, position in enumerate(((0, 0), (0, -1), (-1, -1), (-1, 0))):
        radii = [0, 0, 0, 0]
        radii[idx] = '50%'
        alpha = np.asarray(_directional_alpha_ramp(
            48, 32, feather=0, corner_radius=radii,
        ))
        assert alpha[position] == 0
        for other in ((0, 0), (0, -1), (-1, -1), (-1, 0)):
            if other != position:
                assert alpha[other] == 255
        assert alpha[16, 24] == 255


@pytest.mark.parametrize('source_alpha', [255, 128])
def test_stack_directional_mask_preserves_external_edges_and_source_alpha(tmp_path, source_alpha):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (48, 32), (255, 0, 0, source_alpha)).save(src)
    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(name='stack', spec={'params': {
            'width': 48, 'height': 32, 'background': None,
        }})
        stack.image(0, feather=(0, '10%', '8px', 0), corner_radius=(0, 0, 6, 0))
    out = stack.run(tmp_path, input={'image:0': {'image': str(src)}})
    alpha = np.asarray(Image.open(out['image']))[:, :, 3]
    assert alpha[0, 0] == alpha[0, 20] == alpha[10, 0] == source_alpha
    assert alpha[16, 24] == source_alpha
    assert alpha[16, -1] == alpha[-1, 24] == 0
    assert 0 < alpha[-5, 24] < source_alpha
