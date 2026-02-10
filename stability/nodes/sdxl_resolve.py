import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch

from stability.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                   resolve_seed, resolve_spec)

ModelSource = Literal['single_file', 'pretrained_id']


@dataclass(frozen=True)
class ResolvedModelRef:
    source: ModelSource
    ref: str  # expanded path OR hf repo id


def resolve_model_ref(model: dict) -> ResolvedModelRef:
    model_id = model.get('id')
    model_path = model.get('path')

    if model_id and model_path:
        raise ValueError(
            "spec.model: choose only one of 'id' or 'path', not both"
        )

    if model_id:
        return ResolvedModelRef(source='pretrained_id', ref=str(model_id))

    if model_path:
        return ResolvedModelRef(source='single_file', ref=os.path.expanduser(str(model_path)))

    raise ValueError("spec.model: one of 'id' or 'path' is required")


@dataclass(frozen=True)
class ModelConfig:
    device: str
    dtype: torch.dtype
    model_ref: ResolvedModelRef
    vae_id: Optional[str]


@dataclass(frozen=True)
class RandomConfig:
    seed: int
    gen: torch.Generator


@dataclass(frozen=True)
class ImageGenerationContext:
    spec: Dict[str, Any]

    # model / runtime
    model: ModelConfig

    # generation params
    steps: int
    cfg: float
    strength: float
    width: int
    height: int

    # randomness
    rng: RandomConfig


def resolve_common(source_spec: SpecInput) -> ImageGenerationContext:
    spec = resolve_spec(source_spec)

    # model
    model = spec.get('model', {})
    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))

    model_ref = resolve_model_ref(model)

    vae_id = spec.get('vae', {}).get('id') if isinstance(
        spec.get('vae'),
        dict
    ) else spec.get('vae')

    model_conf = ModelConfig(
        device=device,
        dtype=dtype,
        model_ref=model_ref,
        vae_id=vae_id,
    )

    # params
    params = spec.get('params', {})
    steps = int(params.get('steps', 30))
    cfg = float(params.get('cfg', params.get('guidance_scale', 6.0)))
    strength = float(params.get('strength', 0.7))
    width = int(params.get('width', 1024))
    height = int(params.get('height', 1024))

    # seed / RNG
    seed = resolve_seed(spec.get('seed', 'random'))
    gen = torch.Generator(device=device).manual_seed(seed)

    return ImageGenerationContext(
        spec=spec,
        model=model_conf,
        steps=steps,
        cfg=cfg,
        strength=strength,
        width=width,
        height=height,
        rng=RandomConfig(
            seed=seed,
            gen=gen,
        )
    )
