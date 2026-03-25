from dataclasses import dataclass
from typing import Any, Dict

import torch

from stability.nodes.common.config_resolve import (RandomConfig, SpecLike,
                                                   resolve_dtype, resolve_seed,
                                                   resolve_spec)

_default_model = 'Lightricks/LTX-Video-0.9.7-dev'
_default_upscaler = 'Lightricks/ltxv-spatial-upscaler-0.9.7'


@dataclass(frozen=True)
class LTXModelConfig:
    device: str
    dtype: torch.dtype
    model_id: str
    upscaler_id: str


@dataclass(frozen=True)
class LTXGenerationContext:
    spec: Dict[str, Any]
    model: LTXModelConfig

    # guidance
    guidance_scale: float
    guidance_rescale: float

    # video shape
    height: int
    width: int
    fps: int
    num_frames: int

    # sampling schedule
    steps_main: int
    steps_refine: int
    denoise_strength: float

    # latent scaling
    downscale_factor: float
    upscale_latent_factor: float

    # decode knobs
    decode_timestep: float
    decode_noise_scale: float
    image_cond_noise_scale: float

    # text encoder
    max_sequence_length: int

    frame_index: int

    rng: RandomConfig


def resolve_ltx_common(source_spec: SpecLike) -> LTXGenerationContext:
    spec = resolve_spec(source_spec)

    # model settings
    model = spec.get('model', {})
    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))
    model_id = model.get('id', _default_model)
    upscaler_id = model.get('upscaler', _default_upscaler)

    model_cfg = LTXModelConfig(
        device=device,
        dtype=dtype,
        model_id=model_id,
        upscaler_id=upscaler_id,
    )

    # params
    params = spec.get('params', {})

    # seed / RNG
    seed = resolve_seed(spec.get('seed', 'random'))
    gen = torch.Generator(device=device).manual_seed(seed)

    rng_cfg = RandomConfig(seed=seed, gen=gen)

    return LTXGenerationContext(
        spec=spec,
        model=model_cfg,
        guidance_scale=float(params.get('guidance_scale', 3)),
        guidance_rescale=float(params.get('guidance_rescale', 0.7)),
        height=int(params.get('height', 480)),
        width=int(params.get('width', 832)),
        fps=int(params.get('fps', 24)),
        num_frames=int(params.get('num_frames', 96)),
        steps_main=int(params.get('steps_main', 30)),
        steps_refine=int(params.get('steps_refine', 10)),
        denoise_strength=float(params.get('denoise_strength', 0.4)),
        downscale_factor=float(params.get('downscale_factor', 2 / 3)),
        upscale_latent_factor=float(params.get('upscale_latent_factor', 2)),
        decode_timestep=float(params.get('decode_timestep', 0.05)),
        decode_noise_scale=float(params.get('decode_noise_scale', 0.025)),
        image_cond_noise_scale=float(
            params.get('image_cond_noise_scale', 0.025)
        ),
        max_sequence_length=int(params.get('max_sequence_length', 512)),
        frame_index=int(params.get('frame_index', 0)),
        rng=rng_cfg,
    )
