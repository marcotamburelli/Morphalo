from __future__ import annotations

from morphalo.nodes.sdxl_pipe_builder import build_cross_attention_kwargs
from morphalo.nodes.wiring.conditioning import apply_lora, cleanup_adapters
from morphalo.nodes.wiring.lora import LoraBundle, LoraRegistry


class FakePipe:
    def __init__(self):
        self.calls = []
        self.unet = FakeUnet(self.calls)

    def unload_lora_weights(self):
        self.calls.append(('unload_lora_weights',))

    def load_lora_weights(self, model_id, **kwargs):
        self.calls.append(('load_lora_weights', model_id, kwargs))

    def set_adapters(self, adapter_names, adapter_weights):
        self.calls.append(('set_adapters', adapter_names, adapter_weights))

    def lora_state_dict(self, model_id, **kwargs):
        self.calls.append(('lora_state_dict', model_id, kwargs))
        return {'unet.foo.lora.down.weight': 'weight'}, None, None

    def load_lora_into_unet(self, state_dict, **kwargs):
        self.calls.append(('load_lora_into_unet', state_dict, kwargs))


class FakeUnet:
    config = object()

    def __init__(self, calls):
        self.calls = calls

    def set_adapters(self, adapter_names, adapter_weights):
        self.calls.append(('unet.set_adapters', adapter_names, adapter_weights))


class FakeIpBundle:
    with_mask = False

    def build_ip_adapter_masks(self, **kwargs):
        return []


class FakeFaceBundle:
    with_mask = False

    def build_ip_adapter_masks(self, **kwargs):
        return []


class FakeMaskedBundle:
    with_mask = True

    def __init__(self, masks):
        self._masks = masks

    def build_ip_adapter_masks(self, **kwargs):
        return self._masks


def test_apply_lora_loads_weights_and_sets_adapter_weights():
    registry = LoraRegistry(owner=object())
    registry.add(
        'ostris/ikea-instructions-lora-sdxl',
        weight_name='ikea_instructions_xl_v1_5.safetensors',
        adapter_name='ikea',
        adapter_weight=0.7,
    )
    registry.add(
        'lordjia/by-feng-zikai',
        weight_name='fengzikai_v1.0_XL.safetensors',
        adapter_name='feng',
        adapter_weight=0.8,
    )

    pipe = FakePipe()
    bundle = LoraBundle(registry.specs)

    apply_lora(lora_bundle=bundle, pipe=pipe)

    assert pipe.calls == [
        (
            'load_lora_weights',
            'ostris/ikea-instructions-lora-sdxl',
            {
                'adapter_name': 'ikea',
                'weight_name': 'ikea_instructions_xl_v1_5.safetensors',
            },
        ),
        (
            'load_lora_weights',
            'lordjia/by-feng-zikai',
            {
                'adapter_name': 'feng',
                'weight_name': 'fengzikai_v1.0_XL.safetensors',
            },
        ),
        ('set_adapters', ['ikea', 'feng'], [0.7, 0.8]),
    ]


def test_apply_lora_can_skip_text_encoder_for_unet_only_lora():
    registry = LoraRegistry(owner=object())
    registry.add(
        'TonariNoTaku/SDXL_sufficient_nudity',
        weight_name='nudity_v03XL_i1762_prod256n128b2_swn2_offset_e5.safetensors',
        adapter_name='sufficient_nudity',
        adapter_weight=0.6,
        load_text_encoder=False,
    )

    pipe = FakePipe()
    bundle = LoraBundle(registry.specs)

    apply_lora(lora_bundle=bundle, pipe=pipe)

    assert pipe.calls == [
        (
            'lora_state_dict',
            'TonariNoTaku/SDXL_sufficient_nudity',
            {
                'return_lora_metadata': True,
                'weight_name': 'nudity_v03XL_i1762_prod256n128b2_swn2_offset_e5.safetensors',
                'unet_config': pipe.unet.config,
            },
        ),
        (
            'load_lora_into_unet',
            {'unet.foo.lora.down.weight': 'weight'},
            {
                'network_alphas': None,
                'unet': pipe.unet,
                'adapter_name': 'sufficient_nudity',
                'metadata': None,
                '_pipeline': pipe,
            },
        ),
        ('unet.set_adapters', ['sufficient_nudity'], [0.6]),
    ]


def test_cleanup_adapters_unloads_lora_and_ip_adapter_state():
    pipe = FakePipe()
    pipe.unload_ip_adapter = lambda: pipe.calls.append(('unload_ip_adapter',))
    pipe.unet = object()

    cleanup_adapters(pipe)

    assert pipe.calls == [
        ('unload_ip_adapter',),
        ('unload_lora_weights',),
    ]


def test_lora_adds_global_cross_attention_scale():
    registry = LoraRegistry(owner=object())
    registry.add('ostris/super-cereal-sdxl-lora', adapter_name='cereal')
    registry.set_scale(0.9)

    kwargs = build_cross_attention_kwargs(
        ip_bundle=FakeIpBundle(),
        face_bundle=FakeFaceBundle(),
        lora_bundle=LoraBundle(
            registry.specs,
            cross_attention_scale=registry.cross_attention_scale,
        ),
        height=1024,
        width=1024,
        device='cpu',
        dtype=None,
    )

    assert kwargs == {'cross_attention_kwargs': {'scale': 0.9}}


def test_lora_scale_merges_with_ip_adapter_masks():
    masks = ['mask-a']

    registry = LoraRegistry(owner=object())
    registry.add('ostris/super-cereal-sdxl-lora')

    kwargs = build_cross_attention_kwargs(
        ip_bundle=FakeMaskedBundle(masks),
        face_bundle=FakeFaceBundle(),
        lora_bundle=LoraBundle(registry.specs),
        height=1024,
        width=1024,
        device='cpu',
        dtype=None,
    )

    assert kwargs == {
        'cross_attention_kwargs': {
            'ip_adapter_masks': masks,
            'scale': 1.0,
        }
    }
