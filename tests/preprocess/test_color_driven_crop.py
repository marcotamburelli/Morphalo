import numpy as np
import pytest
from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.preprocess import ColorDrivenCrop
from morphalo.nodes.preprocess.color_driven_crop import (
    color_driven_alpha,
)


def test_color_driven_alpha_removes_uniform_surface_and_keeps_text():
    data = np.full((48, 64, 3), 230, dtype=np.uint8)
    data[18:30, 20:44] = 30

    result = color_driven_alpha(
        Image.fromarray(data, mode='RGB'),
        analysis_clusters=2,
        num_dominant_colors=1,
        tolerance=5.0,
        feather=2.0,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[5, 5] == 0
    assert alpha[24, 32] == 255


def test_tiling_distinguishes_same_gray_as_text_and_local_surface():
    data = np.empty((300, 768, 3), dtype=np.uint8)
    data[:, :384] = 235
    data[:, 384:] = 105
    data[120:180, 72:144] = 105
    data[120:180, 624:696] = 15

    result = color_driven_alpha(
        Image.fromarray(data, mode='RGB'),
        analysis_clusters=2,
        num_dominant_colors=1,
        color_scope='local',
        tolerance=4.0,
        feather=2.0,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[20, 40] < 10
    assert alpha[20, 720] < 10
    assert alpha[150, 108] > 240
    assert alpha[150, 660] > 240


def test_global_scope_does_not_treat_local_dominant_foreground_as_background():
    data = np.full((300, 768, 3), 235, dtype=np.uint8)
    data[40:260, 420:730] = 105

    result = color_driven_alpha(
        Image.fromarray(data, mode='RGB'),
        analysis_clusters=2,
        num_dominant_colors=1,
        tolerance=4.0,
        feather=0.0,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[20, 40] == 0
    assert alpha[150, 560] == 255


def test_color_driven_alpha_never_increases_existing_alpha():
    data = np.full((16, 16, 4), (200, 200, 200, 80), dtype=np.uint8)
    result = color_driven_alpha(
        Image.fromarray(data, mode='RGBA'),
        analysis_clusters=1,
        num_dominant_colors=1,
        strength=0.5,
        tolerance=1.0,
        feather=0.0,
    )

    assert np.all(np.asarray(result)[:, :, 3] == 40)


def test_color_driven_alpha_uses_manual_colors():
    data = np.full((32, 32, 3), 255, dtype=np.uint8)
    data[10:22, 12:20] = 0

    result = color_driven_alpha(
        Image.fromarray(data, mode='RGB'),
        colors=('white', '#ffffff', [255, 255, 255]),
        tolerance=1.0,
        feather=0.0,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[2, 2] == 0
    assert alpha[16, 16] == 255


def test_node_writes_rgba_png_and_metadata(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((24, 32, 3), 220, dtype=np.uint8)
    data[8:16, 10:22] = 30
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'analysis_clusters': 1,
                    'num_dominant_colors': 1,
                    'box_margin': 0.0,
                }
            },
        )

    out = node.run(tmp_path)

    assert out['output_size'] == [12, 8]
    assert out['color_driven_crop']['analysis_clusters'] == 1
    assert out['params']['strength'] == 1.0
    assert out['crop']['bbox_xyxy'] == [10, 8, 22, 16]
    assert Image.open(out['image']).mode == 'RGBA'


def test_node_rejects_more_dominant_than_analysis_clusters(tmp_path):
    src = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'analysis_clusters': 2,
                    'num_dominant_colors': 3,
                }
            },
        )

    with pytest.raises(ValueError, match='num_dominant_colors'):
        node.run(tmp_path)


def test_node_rejects_num_dominant_colors_with_manual_colors(tmp_path):
    src = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'colors': 'white',
                    'num_dominant_colors': 0,
                    'crop_mode': 'full_frame',
                }
            },
        )

    with pytest.raises(ValueError, match='mutually exclusive'):
        node.run(tmp_path)


def test_node_reports_manual_color_count(tmp_path):
    src = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'colors': ['white', 'blue', 'green'],
                    'crop_mode': 'full_frame',
                }
            },
        )

    out = node.run(tmp_path)

    assert out['params']['num_dominant_colors'] == 3


def test_color_policy_include_preserves_matching_colors():
    data = np.full((24, 24, 3), 255, dtype=np.uint8)
    data[8:16, 9:15] = (255, 0, 0)

    result = color_driven_alpha(
        Image.fromarray(data, mode='RGB'),
        colors='red',
        color_policy='include',
        tolerance=1.0,
        feather=0.0,
    )
    alpha = np.asarray(result)[:, :, 3]

    assert alpha[2, 2] == 0
    assert alpha[12, 12] == 255


def test_color_policy_include_mask_selects_matching_colors(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((20, 20, 3), 255, dtype=np.uint8)
    data[5:15, 6:14] = (255, 0, 0)
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'mode': 'mask',
                    'colors': 'red',
                    'color_policy': 'include',
                    'tolerance': 1.0,
                    'feather': 0.0,
                }
            },
        )

    out = node.run(tmp_path)
    mask = np.asarray(Image.open(out['image']))

    assert out['params']['color_policy'] == 'include'
    assert mask[1, 1] == 0
    assert mask[10, 10] == 255


def test_default_bbox_keeps_color_driven_alpha(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((24, 32, 3), 255, dtype=np.uint8)
    data[8:16, 10:22] = 0
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'colors': 'white',
                    'crop_mode': 'bbox[1:1]',
                    'box_margin': 0.0,
                    'tolerance': 1.0,
                    'feather': 0.0,
                }
            },
        )

    out = node.run(tmp_path)
    image = Image.open(out['image'])
    alpha = np.asarray(image)[:, :, 3]

    assert out['output_size'] == [12, 12]
    assert out['crop']['bbox_xyxy'] == [10, 6, 22, 18]
    assert alpha[0, 0] == 0
    assert alpha[6, 6] == 255


def test_mask_ignores_strength(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((20, 20, 3), 255, dtype=np.uint8)
    data[5:15, 6:14] = 0
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'mode': 'mask',
                    'strength': 0.0,
                    'colors': 'white',
                    'tolerance': 1.0,
                    'feather': 0.0,
                }
            },
        )

    out = node.run(tmp_path)
    mask = np.asarray(Image.open(out['image']))

    assert Image.open(out['image']).mode == 'L'
    assert out['output_size'] == [20, 20]
    assert out['crop']['bbox_xyxy'] == [0, 0, 20, 20]
    assert mask[1, 1] == 0
    assert mask[10, 10] == 255


def test_min_component_area_removes_small_selected_islands(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((30, 30, 3), 255, dtype=np.uint8)
    data[10:20, 10:20] = 0
    data[2:4, 2:4] = 0
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'colors': 'white',
                    'box_margin': 0.0,
                    'min_component_area': 4,
                    'tolerance': 1.0,
                    'feather': 0.0,
                }
            },
        )

    out = node.run(tmp_path)

    assert out['crop']['bbox_xyxy'] == [10, 10, 20, 20]


def test_default_box_margin_expands_color_bbox(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((40, 40, 3), 255, dtype=np.uint8)
    data[10:30, 10:30] = 0
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'colors': 'white',
                    'tolerance': 1.0,
                    'feather': 0.0,
                }
            },
        )

    out = node.run(tmp_path)

    assert out['params']['box_margin'] == 0.08
    assert out['crop']['bbox_xyxy'] == [8, 8, 32, 32]


def test_min_component_area_removes_small_islands_from_rgba_alpha(tmp_path):
    src = tmp_path / 'page.png'
    data = np.full((30, 30, 3), 255, dtype=np.uint8)
    data[10:20, 10:20] = 0
    data[2:4, 2:4] = 0
    Image.fromarray(data, mode='RGB').save(src)

    with DAG('test', out_dir=tmp_path):
        node = ColorDrivenCrop(
            name='extract',
            path=src,
            spec={
                'params': {
                    'colors': 'white',
                    'crop_mode': 'full_frame',
                    'min_component_area': 4,
                    'tolerance': 1.0,
                    'feather': 0.0,
                }
            },
        )

    out = node.run(tmp_path)
    alpha = np.asarray(Image.open(out['image']))[:, :, 3]

    assert alpha[2, 2] == 0
    assert alpha[15, 15] == 255
