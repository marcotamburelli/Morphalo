import os

import torch
from diffusers import DiffusionPipeline
from diffusers.loaders import IPAdapterMixin

from stability.core.config import *
from stability.nodes import *


def setup_env():
    os.environ['HF_HOME'] = HF_HOME
    os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = HF_HUB_DISABLE_TELEMETRY


def resolve_common(source_spec: Union[Dict[str, Any], str, Path]):
    spec = resolve_spec(source_spec)

    # model settings
    model = spec.get('model', {})
    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))

    model_path = model.get('path')
    if not model_path:
        raise ValueError('spec.model.path is required')

    # params
    params = spec.get('params', {})
    steps = int(params.get('steps', 30))
    cfg = float(params.get('cfg', params.get('guidance_scale', 6.0)))
    strength = float(params.get('strength', 0.7))
    width = int(params.get('width', 1024))
    height = int(params.get('height', 1024))

    # seed
    seed = resolve_seed(spec.get('seed', 'random'))
    gen = torch.Generator(device=device).manual_seed(seed)

    return spec, device, dtype, model_path, steps, cfg, \
        strength, width, height, seed, gen


def finalize_image_output(
    *,
    node_kind: str,
    node_id: str,
    img_path: Path,
    seed: int,
    params: dict,
    dt_s: float,
    cuda_mem: dict,
    controlnet_specs: List["ControlNetSpec"],
    ip_adapter_specs: List["IpAdapterSpec"],
    model_info: Optional[dict] = None,
) -> dict:
    out = {
        'ok': True,
        'node': node_kind,
        'id': node_id,
        'image': str(img_path),
        'seed': seed,
        'params': params,
        'controlnet': controlnet_meta(controlnet_specs),
        'ip-adapters': ip_adapter_meta(ip_adapter_specs),
        'timing': {'seconds': round(dt_s, 3)},
        'cuda_mem': cuda_mem,
    }

    if model_info is not None:
        out['model'] = model_info

    meta_path = write_json_sidecar(img_path, out)
    out['metadata'] = str(meta_path)
    return out


class ControlNetMixin:
    def __post_init__(self):
        super().__post_init__()
        self.controlnet = ControlNetRegistry(owner=self)
        self.ip_adapter = IpAdapterRegistry(owner=self)

    def build_control_bundles(self, input, device, dtype):
        cn_bundle = ControlNetBundle(
            self.controlnet.specs, dtype=dtype, device=device, input=input
        )
        ip_bundle = IpAdapterBundle(
            self.ip_adapter.specs, dtype=dtype, device=device, input=input
        )
        return cn_bundle, ip_bundle


def apply_ip_adapter(ip_bundle: IpAdapterBundle, pipe: DiffusionPipeline | IPAdapterMixin):
    if ip_bundle.has_ip_adapter:
        pipe.register_modules(image_encoder=ip_bundle.image_encoder)
        pipe.load_ip_adapter(
            ip_bundle.model_id_arg,
            subfolder=ip_bundle.subfolder_arg,
            weight_name=[ip_bundle.weight_names_arg]
        )
        pipe.set_ip_adapter_scale(ip_bundle.scale_arg)
    else:
        pipe.unload_ip_adapter()


def controlnet_meta(controlnet_specs: List[ControlNetSpec]) -> list[dict]:
    return [
        {
            'key': cn.key,
            'model_id': cn.model_id,
            'conditioning_scale': cn.conditioning_scale,
            'input_id': f'controlnet:{cn.key}',
        }
        for cn in controlnet_specs
    ]


def ip_adapter_meta(ip_adapter_specs: List[IpAdapterSpec]) -> list[dict]:
    return [
        {
            'key': ipa.key,
            'model_id': ipa.model_id,
            'weight_name': ipa.weight_name,
            'subfolder': ipa.subfolder,
            'scale': ipa.scale,
            'encoder_key': ipa.encoder_key,
            'encoder_subfolder': ipa.encoder_subfolder,
            'input_id': f'ip-adapter:{ipa.key}',
        }
        for ipa in ip_adapter_specs
    ]


class PromptMixin:
    def __post_init__(self):
        super().__post_init__()
        self.prompt = PromptRegistry(owner=self)
