import pytest
from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.preprocess import FlipImage, ResizeImage, TransposeImage
from morphalo.nodes.preprocess.mask_insert_layer import (
    _place_overlay_on_local_canvas,
    _resize_overlay_to_box,
)


def _save_grid(path):
    img = Image.new('RGB', (3, 2))
    pixels = img.load()
    colors = [
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        [(255, 255, 0), (255, 0, 255), (0, 255, 255)],
    ]
    for y, row in enumerate(colors):
        for x, color in enumerate(row):
            pixels[x, y] = color
    img.save(path)
    return colors


def _pixels(path):
    img = Image.open(path).convert('RGB')
    return [
        [img.getpixel((x, y)) for x in range(img.width)]
        for y in range(img.height)
    ]


def test_transpose_image_clockwise_rotates_quarter_turn(tmp_path):
    src = tmp_path / 'src.png'
    colors = _save_grid(src)

    with DAG('test', out_dir=tmp_path):
        node = TransposeImage(
            name='turn',
            path=src,
            spec={'params': {'direction': 'clockwise'}},
        )

    out = node.run(tmp_path)

    assert out['input_size'] == [3, 2]
    assert out['output_size'] == [2, 3]
    assert out['transpose']['direction'] == 'clockwise'
    assert _pixels(out['image']) == [
        [colors[1][0], colors[0][0]],
        [colors[1][1], colors[0][1]],
        [colors[1][2], colors[0][2]],
    ]


def test_transpose_image_anti_clockwise_rotates_quarter_turn(tmp_path):
    src = tmp_path / 'src.png'
    colors = _save_grid(src)

    with DAG('test', out_dir=tmp_path):
        node = TransposeImage(
            name='turn',
            path=src,
            spec={'params': {'direction': 'anti_clockwise'}},
        )

    out = node.run(tmp_path)

    assert out['output_size'] == [2, 3]
    assert _pixels(out['image']) == [
        [colors[0][2], colors[1][2]],
        [colors[0][1], colors[1][1]],
        [colors[0][0], colors[1][0]],
    ]


@pytest.mark.parametrize(
    ('axis', 'expected'),
    [
        (
            'horizontal',
            [
                [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
                [(0, 255, 255), (255, 0, 255), (255, 255, 0)],
            ],
        ),
        (
            'vertical',
            [
                [(255, 255, 0), (255, 0, 255), (0, 255, 255)],
                [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            ],
        ),
    ],
)
def test_flip_image_mirrors_requested_axis(tmp_path, axis, expected):
    src = tmp_path / 'src.png'
    _save_grid(src)

    with DAG('test', out_dir=tmp_path):
        node = FlipImage(
            name='flip',
            path=src,
            spec={'params': {'axis': axis}},
        )

    out = node.run(tmp_path)

    assert out['input_size'] == [3, 2]
    assert out['output_size'] == [3, 2]
    assert out['flip']['axis'] == axis
    assert _pixels(out['image']) == expected


def test_resize_image_resolves_percentages_against_input_axes(tmp_path):
    src = tmp_path / 'src.png'
    Image.new('RGB', (200, 100), color=(10, 20, 30)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ResizeImage(
            name='resize',
            path=src,
            spec={'params': {'size': ['50%', '80%']}},
        )

    out = node.run(tmp_path)

    assert out['input_size'] == [200, 100]
    assert out['output_size'] == [100, 80]
    assert out['resize']['size'] == ['50%', '80%']
    assert out['resize']['resolved_size'] == [100, 80]
    assert Image.open(out['image']).size == (100, 80)


def test_resize_image_preserves_aspect_ratio_for_missing_axis(tmp_path):
    src = tmp_path / 'src.png'
    Image.new('RGB', (200, 100), color=(10, 20, 30)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ResizeImage(
            name='resize',
            path=src,
            spec={'params': {'size': [50, None]}},
        )

    out = node.run(tmp_path)

    assert out['output_size'] == [50, 25]
    assert Image.open(out['image']).size == (50, 25)


def test_mask_insert_layer_insets_can_expand_content_box():
    overlay = Image.new('RGBA', (10, 10), color=(255, 0, 0, 255))

    resized, content_box = _resize_overlay_to_box(
        overlay,
        box_width=100,
        box_height=80,
        inset_left='-10%',
        inset_right='5%',
        inset_top='10%',
        inset_bottom='0%',
        resize='stretch',
    )

    assert content_box == (-10, 8, 105, 72)
    assert resized.size == (105, 72)


def test_mask_insert_layer_place_overlay_preserves_overflow_by_default():
    overlay = Image.new('RGBA', (40, 20), color=(255, 0, 0, 255))

    canvas, origin = _place_overlay_on_local_canvas(
        overlay,
        box_width=100,
        box_height=80,
        content_box=(-20, 10, 100, 40),
        position='left',
    )

    assert origin == (-20.0, 0.0)
    assert canvas.size == (120, 80)
