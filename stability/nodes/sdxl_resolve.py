from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from stability.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                   resolve_seed, resolve_spec)


@dataclass(frozen=True)
class ModelConfig:
    device: str
    dtype: torch.dtype
    model_path: str
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

    model_path = model.get('path')
    if not model_path:
        raise ValueError('spec.model.path is required')

    vae_id = spec.get('vae', {}).get('id') if isinstance(
        spec.get('vae'),
        dict
    ) else spec.get('vae')

    model_conf = ModelConfig(
        device=device,
        dtype=dtype,
        model_path=model_path,
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
