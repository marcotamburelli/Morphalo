import pytest

from demo.macro.refine import refine_sliding_tiles_with_controlnet_group


def test_refine_sliding_tiles_group_omits_controlnets_by_default():
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(1, 1),
    )

    assert not any(
        node.name.startswith('aux_map_')
        for node in group.nodes
    )
    assert not any(
        node.name.startswith('crop_aux_')
        for node in group.nodes
    )
    assert all(
        node.lora.specs == []
        for node in group.nodes if node.name.startswith('refine_')
    )


def test_refine_sliding_tiles_group_accepts_multiple_controlnets():
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(1, 1),
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
    assert [
        node.name for node in group.nodes
        if node.name.startswith('crop_aux_')
    ] == ['crop_aux_00_01', 'crop_aux_00_02']


def test_refine_sliding_tiles_group_rejects_partial_controlnet_lists():
    with pytest.raises(ValueError, match='must all be set'):
        refine_sliding_tiles_with_controlnet_group(
            'tiles',
            refine_spec={'params': {'steps': 1}},
            image_size=(64, 64),
            grid=(1, 1),
            controlnet_models=['diffusers/controlnet-canny-sdxl-1.0'],
            controlnet_conditioning_scales=[0.7],
        )


@pytest.mark.parametrize('selection', [None, {}])
def test_tiles_use_all_adapters_by_default(selection):
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(2, 1),
        adapter_weight_names=['upper.bin', 'lower.bin'],
        adapter_scales=[0.7, 0.35],
        adapter_tile_indices=selection,
    )

    for node in group.nodes:
        if node.name.startswith('refine_'):
            assert [spec.weight_name for spec in node.ip_adapter.specs] == [
                'upper.bin', 'lower.bin',
            ]


@pytest.mark.parametrize('global_adapter', [False, True])
def test_tiles_select_and_wire_adapters_in_original_order(global_adapter):
    selection = {0: [0], 1: []}
    if not global_adapter:
        selection[2] = [0, 2]
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(3, 1),
        adapter_weight_names=['upper.bin', 'lower.bin', 'common.bin'],
        adapter_scales=[0.7, 0.35, 0.2],
        adapter_tile_indices=selection,
    )
    tiles = {
        node.name: node for node in group.nodes
        if node.name.startswith('refine_')
    }

    assert [
        (spec.key, spec.weight_name, spec.scale)
        for spec in tiles['refine_00'].ip_adapter.specs
    ] == [('style_1', 'upper.bin', 0.7), ('style_3', 'common.bin', 0.2)]
    assert [spec.key for spec in tiles['refine_01'].ip_adapter.specs] == (
        ['style_3'] if global_adapter else []
    )
    assert [spec.key for spec in tiles['refine_02'].ip_adapter.specs] == ['style_3']
    style_edges = [
        (edge.node_from, edge.node_to)
        for edge in group.edges
        if edge.node_from.endswith(('style_1', 'style_2', 'style_3'))
    ]
    assert len(style_edges) == (4 if global_adapter else 3)
    if not global_adapter:
        assert not any(target.endswith('refine_01') for _, target in style_edges)
    assert not any(
        source.endswith('style_2') for source, _ in style_edges
    )


@pytest.mark.parametrize('selection, message', [
    ({-1: []}, 'invalid adapter index'),
    ({2: []}, 'invalid adapter index'),
    ({True: []}, 'invalid adapter index'),
    ({0: [-1]}, 'invalid tile index'),
    ({0: [2]}, 'invalid tile index'),
    ({0: [True]}, 'invalid tile index'),
    ({0: [0, 0]}, 'duplicate tile indices'),
    ({0: (0,)}, 'must be a list'),
])
def test_tiles_reject_invalid_adapter_mapping(selection, message):
    with pytest.raises(ValueError, match=message):
        refine_sliding_tiles_with_controlnet_group(
            'tiles',
            refine_spec={'params': {'steps': 1}},
            image_size=(64, 64),
            grid=(2, 1),
            adapter_weight_names=['upper.bin', 'lower.bin'],
            adapter_scales=[0.7, 0.35],
            adapter_tile_indices=selection,
        )


def test_tiles_apply_all_loras_independently_of_style_selection():
    declarations = [
        {
            'id': 'org/upper',
            'weight_name': 'upper.safetensors',
        },
        'org/detail',
    ]
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(2, 1),
        adapter_weight_names=['upper.bin'],
        adapter_scales=[0.7],
        adapter_tile_indices={0: [0]},
        lora_models=declarations,
        lora_weights=[0.6, 0.3],
    )
    tiles = [node for node in group.nodes if node.name.startswith('refine_')]

    assert len(tiles) == 2
    assert tiles[1].ip_adapter.specs == []
    for tile in tiles:
        first, second = tile.lora.specs
        assert first.model_id == 'org/upper'
        assert first.weight_name == 'upper.safetensors'
        assert first.adapter_weight == 0.6
        assert second.model_id == 'org/detail'
        assert second.weight_name is None
        assert second.adapter_weight == 0.3
    assert tiles[0].lora.specs[0] is not tiles[1].lora.specs[0]
    assert declarations[0] == {'id': 'org/upper', 'weight_name': 'upper.safetensors'}


@pytest.mark.parametrize('models, weights, message', [
    (['org/model'], None, 'must both be set'),
    (None, [0.5], 'must both be set'),
    (['org/model'], [], 'same length'),
    ([{'weight_name': 'style.safetensors'}], [0.5], 'each lora_models entry'),
    ([{'id': 'org/model', 'unknown': 1}], [0.5], 'each lora_models entry'),
    ([''], [0.5], 'each lora_models entry'),
])
def test_tiles_validate_lora_lists(models, weights, message):
    with pytest.raises(ValueError, match=message):
        refine_sliding_tiles_with_controlnet_group(
            'tiles',
            refine_spec={'params': {'steps': 1}},
            image_size=(64, 64),
            grid=(1, 1),
            lora_models=models,
            lora_weights=weights,
        )
