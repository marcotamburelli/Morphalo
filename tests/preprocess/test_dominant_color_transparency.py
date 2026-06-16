import numpy as np
import pytest
from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.preprocess import DominantColorTransparency
from morphalo.nodes.preprocess.dominant_color_transparency import (
    dominant_color_alpha,
)


def test_dominant_color_alpha_removes_uniform_surface_and_keeps_text():
    data = np.full((48, 64, 3), 230, dtype=np.uint8)
    data[18:30, 20:44] = 30

    result = dominant_color_alpha(
        Image.fromarray(data, mode='RGB'),
        analysis_clusters=2,
        num_dominant_clusters=1,
        tile_size=64,
        tile_overlap=0.5,
        color_tolerance=5.0,
        feather=2.0,
        sample_stride=1,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[5, 5] == 0
    assert alpha[24, 32] == 255


def test_tiling_distinguishes_same_gray_as_text_and_local_surface():
    data = np.empty((48, 96, 3), dtype=np.uint8)
    data[:, :48] = 235
    data[:, 48:] = 105
    data[18:30, 16:32] = 105
    data[18:30, 64:80] = 15

    result = dominant_color_alpha(
        Image.fromarray(data, mode='RGB'),
        analysis_clusters=2,
        num_dominant_clusters=1,
        tile_size=48,
        tile_overlap=0.25,
        color_tolerance=4.0,
        feather=2.0,
        sample_stride=1,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[5, 10] < 10
    assert alpha[5, 86] < 10
    assert alpha[24, 24] > 240
    assert alpha[24, 72] > 240


def test_dominant_color_alpha_never_increases_existing_alpha():
    data = np.full((16, 16, 4), (200, 200, 200, 80), dtype=np.uint8)
    result = dominant_color_alpha(
        Image.fromarray(data, mode='RGBA'),
        analysis_clusters=1,
        num_dominant_clusters=1,
        tile_size=16,
        dominant_alpha=0.5,
        color_tolerance=1.0,
        feather=0.0,
        sample_stride=1,
    )

    assert np.all(np.asarray(result)[:, :, 3] == 40)


def test_node_writes_rgba_png_and_metadata(tmp_path):
    src = tmp_path / 'page.png'
    Image.new('RGB', (24, 16), color=(220, 220, 220)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = DominantColorTransparency(
            name='extract',
            path=src,
            spec={
                'params': {
                    'analysis_clusters': 1,
                    'num_dominant_clusters': 1,
                    'tile_size': 16,
                }
            },
        )

    out = node.run(tmp_path)

    assert out['output_size'] == [24, 16]
    assert out['dominant_color_transparency']['tile_size'] == 16
    assert Image.open(out['image']).mode == 'RGBA'


def test_node_rejects_more_dominant_than_analysis_clusters(tmp_path):
    src = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(src)

    with DAG('test', out_dir=tmp_path):
        node = DominantColorTransparency(
            name='extract',
            path=src,
            spec={
                'params': {
                    'analysis_clusters': 2,
                    'num_dominant_clusters': 3,
                }
            },
        )

    with pytest.raises(ValueError, match='num_dominant_clusters'):
        node.run(tmp_path)
