import pytest

from demo.macro.refine import (
    _tile_edge_smoothing, refine_sliding_tiles_group,
    refine_sliding_tiles_with_controlnet_group,
)
from morphalo.nodes.common.config_resolve import resolve_spec
from morphalo.nodes.prompt_merge import PromptMerge


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


@pytest.mark.parametrize('macro', [
    refine_sliding_tiles_group, refine_sliding_tiles_with_controlnet_group,
])
def test_tiles_only_smooth_internal_canvas_edges(macro):
    group = macro(
        'tiles', refine_spec={}, image_size=(96, 96), grid=(3, 3),
        layer_feather='10%', layer_corner_radius='12%',
    )
    stacks = [node for node in group.nodes if node.name.startswith('stack_') or node.name == 'out']
    crops = [node for node in group.nodes if node.name.startswith('crop_')]
    assert len(stacks) == len(crops) == 9
    for stack, crop in zip(stacks, crops):
        bbox = tuple(crop.spec['params']['bbox'])
        expected = _tile_edge_smoothing(bbox, (96, 96), '10%', '12%')
        assert (stack._layers[1].feather, stack._layers[1].corner_radius) == expected
    assert stacks[0]._layers[1].feather == (0, '10%', '10%', 0)
    assert stacks[0]._layers[1].corner_radius == (0, 0, '12%', 0)
    assert stacks[4]._layers[1].feather == ('10%',) * 4
    assert stacks[-1]._layers[1].feather == ('10%', 0, 0, '10%')


def test_full_canvas_tile_has_no_smoothing():
    assert _tile_edge_smoothing((0, 0, 96, 96), (96, 96), 40, 60) == (
        (0, 0, 0, 0), (0, 0, 0, 0),
    )


def test_tile_smoothing_preserves_mixed_units_on_internal_edges():
    assert _tile_edge_smoothing(
        (0, 20, 50, 80), (100, 100), (1, '2%', 3, '4px'), (5, '6%', 7, '8px'),
    ) == ((1, '2%', 3, 0), (0, '6%', 7, 0))


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


@pytest.mark.parametrize('layered', [False, True])
def test_aux_maps_match_canvas_and_preserve_annotator_params(layered):
    original_spec = {
        'processor': 'lineart_realistic',
        'params': {'detect_resolution': 1024, 'image_resolution': 1024},
        'out_width': 512,
        'out_height': 512,
        'pad_to_multiple_of': 64,
        'output_resize_mode': 'contain',
    }
    spec = [original_spec, {'device': 'cpu'}] if layered else original_spec
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(1000, 750),
        grid=(1, 1),
        controlnet_models=['diffusers/controlnet-canny-sdxl-1.0'],
        controlnet_conditioning_scales=[0.7],
        aux_map_specs=[spec],
    )
    aux_map = next(node for node in group.nodes if node.name == 'aux_map_01')
    resolved = resolve_spec(aux_map.spec)

    assert resolved['out_width'] == 1000
    assert resolved['out_height'] == 750
    assert resolved['pad_to_multiple_of'] == 1
    assert resolved['output_resize_mode'] == 'stretch'
    assert resolved['processor'] == original_spec['processor']
    assert resolved['params'] == original_spec['params']
    if layered:
        assert resolved['device'] == 'cpu'
    assert original_spec['out_width'] == 512
    assert original_spec['out_height'] == 512
    assert original_spec['pad_to_multiple_of'] == 64
    assert original_spec['output_resize_mode'] == 'contain'


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
@pytest.mark.parametrize('key_format', ['index', 'port', 'mixed'])
def test_tiles_select_and_wire_adapters_in_original_order(global_adapter, key_format):
    selection = {0: [0], 1: []}
    if not global_adapter:
        selection[2] = [0, 2]
    if key_format == 'port':
        selection = {f'style_{idx + 1}': tiles for idx, tiles in selection.items()}
    elif key_format == 'mixed':
        selection['style_1'] = selection.pop(0)
    original_selection = {key: list(tiles) for key, tiles in selection.items()}
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
    assert selection == original_selection


@pytest.mark.parametrize('selection, message', [
    ({-1: []}, 'invalid adapter index'),
    ({2: []}, 'invalid adapter index'),
    ({True: []}, 'invalid adapter index'),
    ({0: [-1]}, 'invalid tile index'),
    ({0: [2]}, 'invalid tile index'),
    ({0: [True]}, 'invalid tile index'),
    ({0: [0, 0]}, 'duplicate tile indices'),
    ({0: (0,)}, 'must be a list'),
    ({'style_0': []}, 'invalid adapter port'),
    ({'style_3': []}, 'invalid adapter port'),
    ({'style_01': []}, 'invalid adapter port'),
    ({'unknown': []}, 'invalid adapter port'),
    ({'style_1': [-1]}, 'invalid tile index'),
    ({'style_1': [2]}, 'invalid tile index'),
    ({'style_1': [True]}, 'invalid tile index'),
    ({'style_1': [0, 0]}, 'duplicate tile indices'),
    ({'style_1': (0,)}, 'must be a list'),
    ({0: [0], 'style_1': [1]}, 'duplicate adapter selection'),
    ({'style_1': [], 0: []}, 'duplicate adapter selection'),
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


@pytest.mark.parametrize('selection', [None, {}])
def test_tiles_use_main_prompt_by_default(selection):
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles', refine_spec={}, image_size=(64, 64), grid=(3, 1),
        prompt_tile_indices=selection,
    )
    prompt_edges = [
        edge for edge in group.edges
        if edge.node_to.split('.')[-1].startswith('refine_')
        and edge.input_id == 'prompt:default'
    ]
    assert len(prompt_edges) == 3
    assert all(edge.node_from.split('.')[-1] == 'in_prompt' for edge in prompt_edges)
    assert not any(isinstance(node, PromptMerge) for node in group.nodes)
    assert group.get_out_node().name == 'out'


def test_tiles_merge_dedicated_prompts_without_main_prompt(tmp_path):
    selection = {'prompt_2': [0, 1], 'prompt_1': [1, 2]}
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles', refine_spec={}, image_size=(64, 64), grid=(4, 1),
        prompt_tile_indices=selection,
    )
    for port in ('prompt_1', 'prompt_2', 'in_prompt'):
        assert group.port(port).node.name == port
    prompt_sources = {
        edge.node_to.split('.')[-1]: edge.node_from.split('.')[-1]
        for edge in group.edges
        if edge.input_id == 'prompt:default'
    }
    assert prompt_sources == {
        'refine_00': 'prompt_2',
        'refine_01': 'prompt_merge_01',
        'refine_02': 'prompt_1',
        'refine_03': 'in_prompt',
    }
    merge = next(node for node in group.nodes if isinstance(node, PromptMerge))
    assert [
        (edge.node_from.split('.')[-1], edge.input_id)
        for edge in group.edges if edge.node_to == merge.id
    ] == [('prompt_2', 'prompt:0'), ('prompt_1', 'prompt:1')]
    result = merge.run(tmp_path, input={
        'prompt:0': {'prompt': 'First', 'negative_prompt': 'bad anatomy'},
        'prompt:1': {'prompt': 'Second', 'prompt_2': 'Secondary',
                     'negative_prompt': 'blur'},
    })
    assert result['prompt'] == 'First\nSecond'
    assert result['prompt_2'] == 'First\nSecondary'
    assert result['negative_prompt'] == 'bad anatomy, blur'
    assert selection == {'prompt_2': [0, 1], 'prompt_1': [1, 2]}
    assert group.get_out_node().name == 'out'


@pytest.mark.parametrize('selection', [
    {'prompt_1': [0, 1]},
    {'prompt_1': [], 'prompt_3': [0, 1]},
    {'prompt_1': []},
])
def test_unused_prompt_ports_preserve_single_group_output(selection):
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles', refine_spec={}, image_size=(64, 64), grid=(2, 1),
        prompt_tile_indices=selection,
    )
    assert group.get_out_node().name == 'out'
    assert group.port('in_prompt').node.name == 'in_prompt'
    for port in selection:
        assert group.port(port).node.name == port
    if any(tiles == [0, 1] for tiles in selection.values()):
        assert not any(
            edge.node_from.split('.')[-1] == 'in_prompt'
            and edge.input_id == 'prompt:default'
            for edge in group.edges
        )


@pytest.mark.parametrize('selection, message', [
    ({'in_prompt': [0]}, 'invalid prompt port'),
    ({'prompt_0': [0]}, 'invalid prompt port'),
    ({'prompt_01': [0]}, 'invalid prompt port'),
    ({'prompt_x': [0]}, 'invalid prompt port'),
    ({'style_1': [0]}, 'invalid prompt port'),
    ({1: [0]}, 'invalid prompt port'),
    ({'prompt_1': (0,)}, 'must be a list'),
    ({'prompt_1': [-1]}, 'invalid tile index'),
    ({'prompt_1': [2]}, 'invalid tile index'),
    ({'prompt_1': [True]}, 'invalid tile index'),
    ({'prompt_1': [0, 0]}, 'duplicate tile indices'),
])
def test_tiles_reject_invalid_prompt_mapping(selection, message):
    with pytest.raises(ValueError, match=message):
        refine_sliding_tiles_with_controlnet_group(
            'tiles', refine_spec={}, image_size=(64, 64), grid=(2, 1),
            prompt_tile_indices=selection,
        )


@pytest.mark.parametrize('selection', [None, {}])
def test_tiles_apply_all_loras_by_default(selection):
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
        lora_tile_indices=selection,
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


def test_tiles_select_loras_and_preserve_original_order():
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(3, 1),
        lora_models=['org/upper', 'org/lower', 'org/common'],
        lora_weights=[0.6, 0.4, 0.2],
        lora_tile_indices={0: [0], 1: [], 2: [0, 2]},
    )
    tiles = {
        node.name: node for node in group.nodes
        if node.name.startswith('refine_')
    }

    assert [spec.model_id for spec in tiles['refine_00'].lora.specs] == [
        'org/upper', 'org/common',
    ]
    assert [spec.adapter_weight for spec in tiles['refine_00'].lora.specs] == [
        0.6, 0.2,
    ]
    assert tiles['refine_01'].lora.specs == []
    assert [spec.model_id for spec in tiles['refine_02'].lora.specs] == [
        'org/common',
    ]


def test_unmapped_lora_is_applied_to_every_tile():
    group = refine_sliding_tiles_with_controlnet_group(
        'tiles',
        refine_spec={'params': {'steps': 1}},
        image_size=(64, 64),
        grid=(2, 1),
        lora_models=['org/local', 'org/global'],
        lora_weights=[0.6, 0.2],
        lora_tile_indices={0: [0]},
    )
    tiles = [node for node in group.nodes if node.name.startswith('refine_')]

    assert [spec.model_id for spec in tiles[0].lora.specs] == [
        'org/local', 'org/global',
    ]
    assert [spec.model_id for spec in tiles[1].lora.specs] == ['org/global']


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


@pytest.mark.parametrize('selection, message', [
    ({-1: []}, 'invalid LoRA index'),
    ({2: []}, 'invalid LoRA index'),
    ({True: []}, 'invalid LoRA index'),
    ({0: [-1]}, 'invalid tile index'),
    ({0: [2]}, 'invalid tile index'),
    ({0: [True]}, 'invalid tile index'),
    ({0: [0, 0]}, 'duplicate tile indices'),
    ({0: (0,)}, 'must be a list'),
])
def test_tiles_reject_invalid_lora_mapping(selection, message):
    with pytest.raises(ValueError, match=message):
        refine_sliding_tiles_with_controlnet_group(
            'tiles',
            refine_spec={'params': {'steps': 1}},
            image_size=(64, 64),
            grid=(2, 1),
            lora_models=['org/upper', 'org/lower'],
            lora_weights=[0.6, 0.4],
            lora_tile_indices=selection,
        )
