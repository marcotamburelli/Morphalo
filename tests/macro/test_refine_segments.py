import pytest

from demo.macro.refine import (
    _band_segment_bboxes_xyxy,
    refine_mask_segments_with_controlnet_group,
)


def test_band_segment_bboxes_make_horizontal_bands_with_percent_bounds():
    boxes = _band_segment_bboxes_xyxy(
        image_size=(100, 200),
        axis='y',
        start='25%',
        end='75%',
        segments=2,
    )

    assert boxes == [
        (0, 50, 100, 100),
        (0, 100, 100, 150),
    ]


def test_band_segment_bboxes_make_vertical_bands_with_overlap_and_reverse():
    boxes = _band_segment_bboxes_xyxy(
        image_size=(100, 50),
        axis='x',
        start=10,
        end=70,
        segments=3,
        overlap=5,
        order='reverse',
    )

    assert boxes == [
        (45, 0, 70, 50),
        (25, 0, 55, 50),
        (10, 0, 35, 50),
    ]


def test_band_segment_bboxes_reject_invalid_range():
    with pytest.raises(ValueError, match='end must be greater'):
        _band_segment_bboxes_xyxy(
            image_size=(100, 50),
            axis='y',
            start='80%',
            end='20%',
            segments=2,
        )


def test_refine_mask_segments_group_exposes_expected_ports():
    group = refine_mask_segments_with_controlnet_group(
        'segmented_refine',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 128),
        segments=2,
        adapter_weight_names=['ip-adapter_sdxl_vit-h.bin'],
        adapter_scales=[0.5],
    )

    assert group.port('in_image').node.name == 'in_image'
    assert group.port('in_mask').node.name == 'in_mask'
    assert group.port('in_prompt').node.name == 'in_prompt'
    assert group.port('style_1').node.name == 'style_1'
    assert group.get_out_node().name == 'out'


def test_refine_mask_segments_group_builds_segment_mask_stack_with_margin():
    group = refine_mask_segments_with_controlnet_group(
        'segmented_refine',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 128),
        start='25%',
        end='75%',
        segments=1,
        bbox_margin=4,
        mask_feather=6,
    )

    crop_mask = next(
        node for node in group.nodes
        if node.name == 'crop_mask_00'
    )
    segment_mask = next(
        node for node in group.nodes
        if node.name == 'segment_mask_00'
    )

    assert crop_mask.spec['params']['bbox'] == [0, 28, 64, 100]
    assert segment_mask.spec['params']['background'] == [0, 0, 0, 255]
    assert segment_mask.spec['params']['out_mode'] == 'RGB'


def test_refine_mask_segments_group_accepts_multiple_controlnets():
    group = refine_mask_segments_with_controlnet_group(
        'segmented_refine',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 128),
        segments=1,
        controlnet_models=[
            'diffusers/controlnet-canny-sdxl-1.0',
            'diffusers/controlnet-depth-sdxl-1.0',
        ],
        controlnet_conditioning_scales=[0.7, 0.35],
        aux_map_specs=[
            {'processor': 'lineart_realistic'},
            {'processor': 'depth_midas'},
        ],
    )

    assert [
        node.name for node in group.nodes
        if node.name.startswith('aux_map_')
    ] == ['aux_map_01', 'aux_map_02']


def test_refine_mask_segments_group_rejects_mismatched_controlnet_lists():
    with pytest.raises(ValueError, match='same length'):
        refine_mask_segments_with_controlnet_group(
            'segmented_refine',
            refine_spec={'params': {'steps': 1}},
            image_size=(64, 128),
            segments=1,
            controlnet_models=[
                'diffusers/controlnet-canny-sdxl-1.0',
                'diffusers/controlnet-depth-sdxl-1.0',
            ],
            controlnet_conditioning_scales=[0.7],
            aux_map_specs=[
                {'processor': 'lineart_realistic'},
                {'processor': 'depth_midas'},
            ],
        )
