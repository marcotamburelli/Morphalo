
import os
from typing import Any, Optional, Tuple

import torch
from diffusers import (AutoencoderKL, ControlNetModel, DiffusionPipeline,
                       OmniGenPipeline, StableDiffusionXLPipeline, T2IAdapter)
from transformers import (CLIPVisionModelWithProjection, DPTForDepthEstimation,
                          DPTImageProcessor, pipeline)

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


def get_translator(
    *,
    model_id: str,
    source_lang: str,
    target_lang: str,
    device: str
) -> pipeline:
    key = CacheKey(
        kind='translator',
        ref=f'{model_id}:{source_lang}:{target_lang}',
        device=device,
        dtype='-'
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    translator = pipeline(
        task='translation',
        model=model_id,
        src_lang=source_lang,
        tgt_lang=target_lang,
        device=0 if device == 'cuda' else -1
    )

    return ModelCache.put(key, translator)


def get_t2i_adapter(*, model_id: str, device: str, dtype: torch.dtype) -> T2IAdapter:
    key = CacheKey(
        kind='t2i_adapter',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    adapter = T2IAdapter.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, adapter)


def get_controlnet_aux_annotator(
        *,
        processor: str,
        cls: Any, device: str,
        repo_id: str = 'lllyasviel/Annotators'
):
    """
    Cache wrapper for controlnet-aux annotators that support .from_pretrained(repo_id).

    Note: This only covers "checkpoint=True" annotators (HED, Midas, Openpose, etc.).
    """
    key = CacheKey(
        kind='controlnet_aux_annotator',
        ref=f'{processor}:{repo_id}',
        device=device,
        dtype='na'
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    proc = cls.from_pretrained(repo_id).to(device)
    return ModelCache.put(key, proc)


def get_omnigen(*, model_id: str, device: str, dtype: torch.dtype) -> OmniGenPipeline:
    key = CacheKey(
        kind='omnigen',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    omg = OmniGenPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, omg)

def get_qwen_image(
    *,
    model_id: str,
    dtype: torch.dtype,
    device_map: str = 'balanced',
) -> DiffusionPipeline:
    """
    Load and cache a Qwen-Image Diffusers pipeline.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. "Qwen/Qwen-Image").
    dtype : torch.dtype
        Torch dtype for loading weights.
    device_map : str
        Accelerate device_map for dispatching modules. Typical values:
        "balanced", "auto", "cuda", "cpu".

    Returns
    -------
    DiffusionPipeline
        Cached pipeline instance.
    """
    key = CacheKey(
        kind='qwen_image',
        ref=model_id,
        device=str(device_map),
        dtype=dtype_key(dtype),
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    return ModelCache.put(key, pipe)
