import pytest
import numpy as np
from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.preprocess import (
    BoxCrop,
    FlipImage,
    ImageLayerPlacement,
    ImageStack,
    ResizeImage,
    TransposeImage,
)
from morphalo.nodes.preprocess.mask_insert_layer import (
    _place_overlay_on_local_canvas,
    _resize_overlay_to_box,
)
from morphalo.nodes.preprocess.utils import read_spatial_transform


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


def test_resize_image_transform_sink_targets_transform_input(tmp_path):
    with DAG('test', out_dir=tmp_path):
        node = ResizeImage(name='resize', spec={'params': {'size': [10, 10]}})
        sink = node.transform()

    assert sink.target is node
    assert sink.input_id == 'transform'


def test_resize_image_uses_transform_crop_bbox_size(tmp_path):
    src = tmp_path / 'src.png'
    Image.new('RGB', (200, 100), color=(10, 20, 30)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ResizeImage(
            name='resize',
            path=src,
            spec={'params': {'size': [50, 50]}},
        )

    out = node.run(
        tmp_path,
        input={
            'transform': {
                'crop': {
                    'bbox_size': [80, 40],
                }
            }
        },
    )

    assert out['output_size'] == [80, 40]
    assert out['resize']['size'] == [80, 40]
    assert out['resize']['source'] == 'transform'
    assert out['params']['size'] == [50, 50]
    assert Image.open(out['image']).size == (80, 40)


def test_resize_image_accepts_transform_bbox_size_without_params_size(tmp_path):
    src = tmp_path / 'src.png'
    Image.new('RGB', (200, 100), color=(10, 20, 30)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ResizeImage(
            name='resize',
            path=src,
            spec={},
        )

    out = node.run(
        tmp_path,
        input={
            'transform': {
                'placement': {
                    'bbox_size': [None, 25],
                }
            }
        },
    )

    assert out['output_size'] == [50, 25]
    assert out['resize']['size'] == [None, 25]
    assert out['resize']['source'] == 'transform'
    assert out['params']['size'] is None


def test_resize_image_requires_size_or_transform_bbox_size(tmp_path):
    src = tmp_path / 'src.png'
    Image.new('RGB', (200, 100), color=(10, 20, 30)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ResizeImage(
            name='resize',
            path=src,
            spec={},
        )

    with pytest.raises(ValueError, match='missing params.size'):
        node.run(tmp_path)


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


def test_image_stack_brightness_changes_rgb_and_preserves_alpha(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (4, 4), color=(200, 100, 50, 128)).save(src)

    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(
            name='stack',
            spec={
                'params': {
                    'width': 4,
                    'height': 4,
                    'background': None,
                    'out_mode': 'RGBA',
                }
            },
        )
        stack.image(idx=0, brightness=-0.5)

    out = stack.run(
        tmp_path,
        input={'image:0': {'image': str(src)}},
    )
    result = Image.open(out['image']).convert('RGBA')

    assert result.getpixel((2, 2)) == (100, 50, 25, 128)
    assert out['params']['layers'][0]['brightness'] == -0.5


@pytest.mark.parametrize('brightness', [-1.1, 1.1])
def test_image_stack_rejects_brightness_outside_range(tmp_path, brightness):
    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(name='stack')

    with pytest.raises(ValueError, match='brightness'):
        stack.image(idx=0, brightness=brightness)


def test_box_crop_emits_local_anchor_and_global_position(tmp_path):
    src = tmp_path / 'src.png'
    Image.new('RGBA', (20, 20), color=(255, 255, 255, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = BoxCrop(
            name='crop',
            path=src,
            spec={
                'params': {
                    'bbox_format': 'xyxy',
                    'bbox': [4, 6, 14, 16],
                }
            },
        )

    out = node.run(tmp_path)

    assert out['crop']['anchor_xy'] == [5, 5]
    assert out['crop']['position'] == [9, 11]
    assert out['crop']['bbox_size'] == [10, 10]
    assert out['crop']['bbox_xyxy'] == [4, 6, 14, 16]


def test_image_stack_transform_uses_new_crop_contract(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (4, 4), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(
            name='stack',
            spec={
                'params': {
                    'width': 16,
                    'height': 16,
                    'background': None,
                    'out_mode': 'RGBA',
                }
            },
        )
        stack.image(idx=0)

    with pytest.raises(ValueError, match='both crop and placement'):
        stack.run(
            tmp_path,
            input={
                'image:0': {'image': str(src)},
                'transform:0': {
                    'crop': {'anchor_xy': [2, 2]},
                    'placement': {'anchor_xy': [2, 2]},
                },
            },
        )
        stack.image(idx=0)

    out = stack.run(
        tmp_path,
        input={
            'image:0': {'image': str(src)},
            'transform:0': {
                'crop': {
                    'anchor_xy': [2, 2],
                    'position': [10, 8],
                    'bbox_size': [4, 4],
                }
            },
        },
    )
    result = Image.open(out['image']).convert('RGBA')

    assert result.getpixel((7, 6)) == (0, 0, 0, 0)
    assert result.getpixel((8, 6)) == (255, 0, 0, 255)
    assert result.getpixel((11, 9)) == (255, 0, 0, 255)
    assert result.getpixel((12, 10)) == (0, 0, 0, 0)


def test_image_stack_transform_accepts_placement_metadata(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (4, 4), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(
            name='stack',
            spec={
                'params': {
                    'width': 16,
                    'height': 16,
                    'background': None,
                    'out_mode': 'RGBA',
                }
            },
        )
        stack.image(idx=0, position='bottom-right')

    out = stack.run(
        tmp_path,
        input={
            'image:0': {'image': str(src)},
            'transform:0': {
                'placement': {
                    'anchor_xy': [0, 0],
                    'position': [5, 6],
                }
            },
        },
    )
    result = Image.open(out['image']).convert('RGBA')

    assert result.getpixel((4, 6)) == (0, 0, 0, 0)
    assert result.getpixel((5, 6)) == (255, 0, 0, 255)
    assert result.getpixel((8, 9)) == (255, 0, 0, 255)


def test_image_stack_transform_defaults_missing_anchor_to_layer_center(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (4, 4), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(
            name='stack',
            spec={
                'params': {
                    'width': 16,
                    'height': 16,
                    'background': None,
                    'out_mode': 'RGBA',
                }
            },
        )
        stack.image(idx=0, position='bottom-right')

    out = stack.run(
        tmp_path,
        input={
            'image:0': {'image': str(src)},
            'transform:0': {
                'placement': {
                    'position': [8, 8],
                }
            },
        },
    )
    result = Image.open(out['image']).convert('RGBA')

    assert result.getpixel((5, 6)) == (0, 0, 0, 0)
    assert result.getpixel((6, 6)) == (255, 0, 0, 255)
    assert result.getpixel((9, 9)) == (255, 0, 0, 255)
    assert result.getpixel((10, 10)) == (0, 0, 0, 0)


def test_image_stack_transform_rejects_crop_and_placement_together(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (4, 4), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        stack = ImageStack(
            name='stack',
            spec={
                'params': {
                    'width': 16,
                    'height': 16,
                    'background': None,
                    'out_mode': 'RGBA',
                }
            },
        )


def test_read_spatial_transform_rejects_none_anchor():
    with pytest.raises(ValueError, match='anchor_xy cannot be None'):
        read_spatial_transform(
            {'crop': {'anchor_xy': None}},
            node_id='node',
        )


def test_image_layer_placement_emits_geometry_anchor(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (10, 6), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={
                'params': {
                    'anchor': 'bottom-center',
                    'position': ['50%', None],
                    'bbox_size': [20, None],
                }
            },
        )

    out = node.run(tmp_path)

    assert out['input_size'] == [10, 6]
    assert out['placement']['anchor_xy'] == [5.0, 6.0]
    assert out['placement']['position'] == ['50%', None]
    assert out['placement']['bbox_size'] == [20, None]
    assert out['params']['anchor'] == 'bottom-center'


def test_image_layer_placement_alpha_bbox_anchor(tmp_path):
    src = tmp_path / 'layer.png'
    img = Image.new('RGBA', (8, 8), color=(0, 0, 0, 0))
    for y in range(2, 6):
        for x in range(1, 5):
            img.putpixel((x, y), (255, 0, 0, 255))
    img.save(src)

    with DAG('test', out_dir=tmp_path):
        node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={'params': {'anchor': 'alpha-bbox-center'}},
        )

    out = node.run(tmp_path)

    assert out['placement']['anchor_xy'] == [3.0, 4.0]
    assert 'position' not in out['placement']
    assert 'bbox_size' not in out['placement']


def test_image_layer_placement_rejects_fit_cover_bbox_size(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (8, 8), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={'params': {'bbox_size': 'fit'}},
        )

    with pytest.raises(ValueError, match='bbox_size'):
        node.run(tmp_path)


def test_image_layer_placement_can_drive_image_stack(tmp_path):
    src = tmp_path / 'layer.png'
    Image.new('RGBA', (4, 4), color=(255, 0, 0, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        placement_node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={
                'params': {
                    'anchor': 'top-left',
                    'position': [6, 7],
                }
            },
        )
        stack = ImageStack(
            name='stack',
            spec={
                'params': {
                    'width': 16,
                    'height': 16,
                    'background': None,
                    'out_mode': 'RGBA',
                }
            },
        )
        stack.image(idx=0)

    placement = placement_node.run(tmp_path)
    out = stack.run(
        tmp_path,
        input={
            'image:0': {'image': str(src)},
            'transform:0': placement,
        },
    )
    result = Image.open(out['image']).convert('RGBA')

    assert result.getpixel((5, 7)) == (0, 0, 0, 0)
    assert result.getpixel((6, 7)) == (255, 0, 0, 255)
    assert result.getpixel((9, 10)) == (255, 0, 0, 255)


@pytest.mark.parametrize(
    ('anchor', 'expected'),
    [
        ('shoulder-center', [50.0, 20.0]),
        ('hips-center', [50.0, 80.0]),
        ('torso-center', [50.0, 50.0]),
        ('body-center', [50.0, 56.0]),
        ('head-center', [50.0, pytest.approx(28.0 / 3.0)]),
    ],
)
def test_image_layer_placement_human_landmark_anchors(
    tmp_path,
    monkeypatch,
    anchor,
    expected,
):
    src = tmp_path / 'person.png'
    Image.new('RGBA', (100, 100), color=(255, 255, 255, 255)).save(src)

    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[0] = [50, 8]
    pose_xy[2] = [45, 10]
    pose_xy[5] = [55, 10]
    pose_xy[11] = [40, 20]
    pose_xy[12] = [60, 20]
    pose_xy[23] = [45, 80]
    pose_xy[24] = [55, 80]

    def fake_get_pose_landmarker(*, model_asset_path, device):
        assert model_asset_path == 'pose.task'
        assert device == 'cpu'
        return object()

    def fake_pose_landmarks_xy(img_rgb, *, pose_landmarker):
        assert img_rgb.shape == (100, 100, 3)
        assert pose_landmarker is not None
        return pose_xy

    monkeypatch.setattr(
        'morphalo.nodes.preprocess.image_layer_placement.'
        'get_mediapipe_pose_landmarker',
        fake_get_pose_landmarker,
    )
    monkeypatch.setattr(
        'morphalo.nodes.preprocess.image_layer_placement.mp_pose_landmarks_xy',
        fake_pose_landmarks_xy,
    )

    with DAG('test', out_dir=tmp_path):
        node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={
                'model': {
                    'device': 'cpu',
                    'pose_landmarker_task': 'pose.task',
                },
                'params': {'anchor': anchor},
            },
        )

    out = node.run(tmp_path)

    assert out['placement']['anchor_xy'] == expected
    assert out['params']['device'] == 'cpu'
    assert out['params']['pose_landmarker_task'] == 'pose.task'


def test_image_layer_placement_human_anchor_requires_pose_model(tmp_path):
    src = tmp_path / 'person.png'
    Image.new('RGBA', (100, 100), color=(255, 255, 255, 255)).save(src)

    with DAG('test', out_dir=tmp_path):
        node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={'params': {'anchor': 'body-center'}},
        )

    with pytest.raises(ValueError, match='pose_landmarker_task'):
        node.run(tmp_path)


def test_image_layer_placement_human_anchor_fails_on_missing_landmark(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / 'person.png'
    Image.new('RGBA', (100, 100), color=(255, 255, 255, 255)).save(src)

    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[11] = [40, 20]
    pose_xy[23] = [45, 80]
    pose_xy[24] = [55, 80]

    monkeypatch.setattr(
        'morphalo.nodes.preprocess.image_layer_placement.'
        'get_mediapipe_pose_landmarker',
        lambda *, model_asset_path, device: object(),
    )
    monkeypatch.setattr(
        'morphalo.nodes.preprocess.image_layer_placement.mp_pose_landmarks_xy',
        lambda img_rgb, *, pose_landmarker: pose_xy,
    )

    with DAG('test', out_dir=tmp_path):
        node = ImageLayerPlacement(
            name='place',
            path=src,
            spec={
                'model': {'pose_landmarker_task': 'pose.task'},
                'params': {'anchor': 'body-center'},
            },
        )

    with pytest.raises(RuntimeError, match='right_shoulder'):
        node.run(tmp_path)
