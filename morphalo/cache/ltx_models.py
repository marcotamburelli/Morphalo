

import torch
from diffusers import LTXLatentUpsamplePipeline
from diffusers.hooks import apply_group_offloading

from morphalo.cache import CacheKey, ModelCache
from morphalo.cache.models import dtype_key
from third_party.lightricks import LTXConditionPipeline


def get_ltx_condition(*, model_id: str, device: str, dtype: torch.dtype) -> LTXConditionPipeline:
    key = CacheKey(
        kind='ltx_condition',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    pipe.tokenizer.model_max_length = 512
    pipe.tokenizer_max_length = 512

    pipe.enable_attention_slicing("max")

    vae = pipe.vae

    # VAE: reduces the peach during decode (conv3d)
    vae.enable_tiling()
    vae.enable_slicing()

    onload_device = torch.device(device)
    offload_device = torch.device("cpu")

    # Transformer (often is teh biggest component)
    pipe.transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=True,
    )

    # Text encoder: better to have it in blocks
    apply_group_offloading(
        pipe.text_encoder,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=2,
    )

    # VAE: sometimes it's the bottleneck
    apply_group_offloading(
        vae,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
    )

    return ModelCache.put(key, pipe)


def get_ltx_latent_upsample(
        *,
        upscaler_id: str,
        model_id: str,
        device: str,
        dtype: torch.dtype
) -> LTXLatentUpsamplePipeline:
    key = CacheKey(
        kind='ltx_latent_upsample',
        ref=f'{upscaler_id}:{model_id}',
        device=device,
        dtype=dtype_key(dtype)
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    vae = get_ltx_condition(
        model_id=model_id,
        device=device,
        dtype=dtype
    ).vae

    pipe = LTXLatentUpsamplePipeline.from_pretrained(
        upscaler_id,
        vae=vae,
        torch_dtype=dtype,
    ).to(device)

    return ModelCache.put(key, pipe)
