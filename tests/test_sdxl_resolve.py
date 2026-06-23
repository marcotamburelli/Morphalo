from morphalo.nodes.sdxl_resolve import resolve_common


def test_resolve_common_merges_prompt_layer_without_replacing_params():
    ctx = resolve_common([
        {
            'model': {'id': 'local-test-model', 'device': 'cpu'},
            'params': {
                'steps': 30,
                'cfg': 6.0,
                'strength': 0.7,
                'width': 1024,
                'height': 1024,
            },
            'seed': 123,
        },
        {
            'params': {
                'steps': 50,
                'cfg': 5.0,
                'strength': 0.4,
                'long_side': 1024,
            },
        },
        {
            'prompt': ['S......d'],
            'negative_prompt': ['chair', 'bad quality'],
        },
    ])

    assert ctx.steps == 50
    assert ctx.cfg == 5.0
    assert ctx.strength == 0.4
    assert ctx.long_side == 1024
    assert ctx.spec['prompt'] == ['S......d']
    assert ctx.spec['negative_prompt'] == ['chair', 'bad quality']
