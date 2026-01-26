import os
from typing import List

import torch
from diffusers import DiffusionPipeline
from diffusers.loaders import IPAdapterMixin

from stability.core.config import *
from stability.nodes.utils import *
from stability.nodes.wiring.controlnet import *
from stability.nodes.wiring.face_id import *
from stability.nodes.wiring.ip_adapter import *
from stability.nodes.wiring.prompt import PromptRegistry
from stability.nodes.wiring.t2i_adapter import *


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
    controlnet_specs: List[ControlNetSpec] = None,
    t2i_adapter_specs: List[T2IAdapterSpec] = None,
    ip_adapter_specs: List['IpAdapterSpec'] = None,
    face_id_specs: List[FaceIdSpec] = None,
    model_info: Optional[dict] = None,
) -> dict:
    out = {
        'ok': True,
        'node': node_kind,
        'id': node_id,
        'image': str(img_path),
        'seed': seed,
        'params': params,
        'timing': {'seconds': round(dt_s, 3)}
    }

    if cuda_mem:
        out['cuda_mem'] = cuda_mem

    if controlnet_specs:
        out['controlnet'] = controlnet_meta(controlnet_specs)

    if t2i_adapter_specs:
        out['t2i-adapters'] = t2i_adapter_meta(t2i_adapter_specs)

    if ip_adapter_specs:
        out['ip-adapters'] = ip_adapter_meta(ip_adapter_specs)

    if face_id_specs:
        out['face-id'] = face_id_meta(face_id_specs)

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
        self.ip_adapter = IpAdapterRegistry(owner=self)
        self.face_id = FaceIdRegistry(owner=self)

    def build_control_bundles(self, input, device, dtype):
        cn_bundle = ControlNetBundle(
            self.controlnet.specs, dtype=dtype, device=device, input=input
        )
        ip_bundle = IpAdapterBundle(
            self.ip_adapter.specs, dtype=dtype, device=device, input=input
        )
        face_bundle = FaceIdBundle(
            self.face_id.specs, dtype=dtype, device=device, input=input
        )
        return cn_bundle, ip_bundle, face_bundle


class T2IAdapterMixin:
    def __post_init__(self):
        super().__post_init__()
        self.t2i_adapter = T2IAdapterRegistry(owner=self)

    def build_t2i_adapter_bundle(self, input: Optional[Dict[str, Dict]], device: str, dtype: torch.dtype) -> T2IAdapterBundle:
        return T2IAdapterBundle(
            adapters=self.t2i_adapter.specs,
            dtype=dtype,
            device=device,
            input=input or {}
        )


def apply_ip_adapter(
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    pipe: DiffusionPipeline | IPAdapterMixin,
    device: str,
    dtype: torch.dtype,
):
    if ip_bundle.has_ip_adapter:
        pipe.register_modules(image_encoder=ip_bundle.image_encoder)
        pipe.load_ip_adapter(
            ip_bundle.model_id_arg,
            subfolder=ip_bundle.subfolder_arg,
            weight_name=ip_bundle.weight_names_arg
        )
        pipe.set_ip_adapter_scale(ip_bundle.scale_arg)
    elif face_bundle.has_face_id:
        if face_bundle.image_encoder is not None:
            pipe.register_modules(image_encoder=face_bundle.image_encoder)

        pipe.load_ip_adapter(
            face_bundle.model_id_arg,
            subfolder=None,
            weight_name=face_bundle.weight_names_arg,
            image_encoder_folder=None
        )
        pipe.set_ip_adapter_scale(face_bundle.scale_arg)

        apply_faceid_clip(
            face_bundle=face_bundle,
            pipe=pipe,
            device=torch.device(device),
            dtype=dtype,
        )

    else:
        pipe.unload_ip_adapter()


def apply_faceid_clip(
    *,
    face_bundle: FaceIdBundle,
    pipe: DiffusionPipeline | IPAdapterMixin,
    device: torch.device,
    dtype: torch.dtype,
    num_images: int = 1,
    clip_strength: float = 1.0,   # 0.0 = “disattiva” visivo senza crash
):
    """
    Inject CLIP image embeddings into the hidden projection layers for
    IP-Adapter FaceID Plus / PlusV2 models.

    This function must be called AFTER FaceID weights are loaded via
    `pipe.load_ip_adapter(...)` and BEFORE invoking the pipeline (`pipe(...)`).

    Notes
    -----
    Diffusers computes CLIP embeddings for IP-Adapters through
    `prepare_ip_adapter_image_embeds(ip_adapter_image=..., ip_adapter_image_embeds=None, ...)`.
    When `ip_adapter_image_embeds` is None, `ip_adapter_image` MUST be a list whose
    length matches the number of loaded IP-Adapters (i.e. the number of projection layers).

    Therefore, when multiple FaceID adapters are loaded, we compute CLIP embeddings
    in one call using a per-adapter list of images, then inject the resulting tensors
    into the corresponding projection layers.

    Parameters
    ----------
    face_bundle:
        FaceID bundle. Only slots marked as `requires_clip=True` (Plus/PlusV2)
        will be injected; other slots are ignored.

    pipe:
        Diffusers pipeline with FaceID IP-Adapters already loaded.

    device, dtype:
        Torch device/dtype used by the pipeline.

    num_images:
        Number of images generated per prompt (equivalent to
        `num_images_per_prompt` in Diffusers).

        In the common case where a single image is generated per prompt,
        this value should be set to 1 (default).

        This parameter is required so Diffusers can correctly replicate
        CLIP embeddings to match the internal batch size. If you are not
        using batching or generating multiple images per prompt, using
        `num_images = 1` is correct and sufficient.

    clip_strength:
        Multiplier applied to computed CLIP embeds before injection.
        Set to 0.0 to effectively neutralize the visual component.
    """
    if not face_bundle.has_face_id:
        return

    entries = face_bundle.clip_images_per_slot  # list[Optional[(images, is_plusv2)]], len = n_face_adapters
    if not entries:
        return

    layers = pipe.unet.encoder_hid_proj.image_projection_layers
    n_layers = len(layers)

    if n_layers != len(face_bundle.weight_names_arg):
        raise RuntimeError(
            f"FaceID adapters/layers mismatch: layers={n_layers} but face weights={len(face_bundle.weight_names_arg)}."
        )

    blank = Image.new("RGB", (224, 224), (0, 0, 0))

    # Build per-adapter image list of length == n_layers.
    # Even for non-clip slots we provide images to satisfy diffusers' length check.
    # (Cost: extra encode; Benefit: perfect alignment + identical semantics to diffusers.)
    ip_adapter_images: list = []
    for j, entry in enumerate(entries):
        if entry is None:
            ip_adapter_images.append([blank])
        else:
            images, _ = entry
            img_list = images if isinstance(images, list) else [images]
            ip_adapter_images.append(img_list)

    # Compute embeds for ALL adapters in one call (diffusers requirement).
    # This returns a list: one tensor per adapter layer, already expanded for CFG and num_images.
    clip_embeds_per_layer = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_images,
        None,
        device,
        num_images,
        True,  # do_classifier_free_guidance; consistent with typical guidance_scale>1 usage
    )

    # Inject only where required (Plus/PlusV2)
    for j, entry in enumerate(entries):
        if entry is None:
            continue

        _images, is_plusv2 = entry
        clip_embeds = clip_embeds_per_layer[j].to(device=device, dtype=dtype)

        if clip_strength != 1.0:
            clip_embeds = clip_embeds * float(clip_strength)

        layer = layers[j]
        layer.clip_embeds = clip_embeds

        if is_plusv2:
            layer.shortcut = False


def build_cross_attention_kwargs(
    *,
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype
):
    cross_attention_kwargs = {}

    if (ip_bundle.with_mask or face_bundle.with_mask):
        ip_masks = ip_bundle.build_ip_adapter_masks(
            height=height,
            width=width,
            device=device,
            dtype=dtype
        )

        face_masks = face_bundle.build_ip_adapter_masks(
            height=height,
            width=width,
            device=device,
            dtype=dtype
        )

        cross_attention_kwargs['cross_attention_kwargs'] = {
            'ip_adapter_masks': ip_masks + face_masks
        }

    return cross_attention_kwargs


def build_pipe_kwargs(
    *,
    cn_bundle: ControlNetBundle,
    t2i_bundle: T2IAdapterBundle = None,
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    height: int,
    width: int,
    device: str,
    dtype,
) -> Dict[str, Any]:
    """
    Assemble kwargs for pipe(...) from all bundles.

    Notes:
      - FaceID Plus/V2 CLIP injection is NOT done here; call apply_faceid_clip(...)
        after load_ip_adapter and before pipe(...).
      - Mask wiring logic is delegated to build_cross_attention_kwargs(...).
    """
    kwargs: Dict[str, Any] = {}

    # ControlNet
    if cn_bundle.has_controlnet:
        kwargs.update({
            'image': cn_bundle.control_image_arg,
            'controlnet_conditioning_scale': cn_bundle.conditioning_scale_arg,
        })

    # T2I-Adapter
    if t2i_bundle is not None and t2i_bundle.has_t2i_adapter:
        kwargs.update({
            'image': t2i_bundle.adapter_image_arg,
            'adapter_conditioning_scale': t2i_bundle.conditioning_scale_arg,
        })

    # IP-Adapter (normal)
    if ip_bundle.has_ip_adapter:
        kwargs['ip_adapter_image'] = ip_bundle.ip_adapter_image

    # FaceID
    if face_bundle.has_face_id:
        kwargs['ip_adapter_image_embeds'] = face_bundle.ip_adapter_image_embeds

    # Cross-attention masks (IP + FaceID), merged inside the helper
    kwargs.update(build_cross_attention_kwargs(
        ip_bundle=ip_bundle,
        face_bundle=face_bundle,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    ))

    return kwargs


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
            'input_id': f'ip-adapter:{ipa.key}',
        }
        for ipa in ip_adapter_specs
    ]


def face_id_meta(face_specs: List[FaceIdSpec]) -> list[dict]:
    return [
        {
            'key': fid.key,
            'model_id': fid.model_id,
            'weight_name': fid.weight_name,
            'subfolder': fid.subfolder,
            'scale': fid.scale,
            'has_mask': fid.has_mask,
            'input_id': f'face_id:{fid.key}',
            'mask_input_id': f'ip_adapter_mask:{fid.key}' if fid.has_mask else None,
        }
        for fid in face_specs
    ]


class PromptMixin:
    def __post_init__(self):
        super().__post_init__()
        self.prompt = PromptRegistry(owner=self)


def t2i_adapter_meta(t2i_adapter_specs: List[T2IAdapterSpec]) -> list[dict]:
    return [
        {
            'key': a.key,
            'model_id': a.model_id,
            'conditioning_scale': a.conditioning_scale,
            'input_id': f't2i-adapter:{a.key}',
        }
        for a in (t2i_adapter_specs or [])
    ]
