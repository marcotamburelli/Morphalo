
import os
from typing import Optional, Tuple

import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLPipeline
from transformers import (CLIPVisionModelWithProjection, DPTForDepthEstimation,
                          DPTImageProcessor)

from stability.cache import CacheKey, ModelCache


def dtype_key(dtype: torch.dtype) -> str:
    # stable cache key string
    return str(dtype).replace('torch.', '')


def get_sdxl_base_pipe(
    *,
    model_path: str,
    device: str,
    dtype: torch.dtype,
    vae_id: Optional[str] = None,
) -> StableDiffusionXLPipeline:
    model_ref = os.path.expanduser(model_path)

    # include vae_id in the cache key to avoid mismatches
    extra = f'vae={vae_id}' if vae_id else 'vae=<default>'
    key = CacheKey(
        kind='sdxl_base_pipe',
        ref=model_ref,
        device=device,
        dtype=dtype_key(dtype),
        extra=extra
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    vae = get_vae(
        vae_id=vae_id,
        device=device,
        dtype=dtype
    ) if vae_id else None

    pipe = StableDiffusionXLPipeline.from_single_file(
        model_ref,
        torch_dtype=dtype,
        **({'vae': vae} if vae is not None else {}),
    ).to(device)

    return ModelCache.put(key, pipe)


def get_controlnet(*, model_id: str, device: str, dtype: torch.dtype) -> ControlNetModel:
    key = CacheKey(
        kind='controlnet',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    cn = ControlNetModel.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, cn)


def get_vae(*, vae_id: str, device: str, dtype: torch.dtype) -> AutoencoderKL:
    key = CacheKey(
        kind='vae',
        ref=vae_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    vae = AutoencoderKL.from_pretrained(
        vae_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, vae)


def get_depth_estimator(*, model_id: str, device: str) -> Tuple[DPTImageProcessor, DPTForDepthEstimation]:
    proc_key = CacheKey(
        kind='depth_processor',
        ref=model_id,
        device='cpu',
        dtype='na'
    )
    mod_key = CacheKey(
        kind='depth_model',
        ref=model_id,
        device=device,
        dtype='na'
    )

    processor = ModelCache.get(proc_key)
    if processor is None:
        processor = ModelCache.put(
            proc_key, DPTImageProcessor.from_pretrained(model_id)
        )

    model = ModelCache.get(mod_key)
    if model is None:
        model = DPTForDepthEstimation.from_pretrained(model_id).to(device)
        model.eval()
        ModelCache.put(mod_key, model)

    return processor, model


def get_ip_image_encoder(
    *,
    repo_id: str,
    subfolder: str,
    device: str,
    dtype: torch.dtype
) -> CLIPVisionModelWithProjection:
    key = CacheKey(
        kind='ip_image_encoder',
        ref=f'{repo_id}:{subfolder}',
        device=device,
        dtype=dtype_key(dtype)
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    enc = CLIPVisionModelWithProjection.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, enc)
