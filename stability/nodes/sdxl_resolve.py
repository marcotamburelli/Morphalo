import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

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

    long_side: Optional[int]

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

    long_side = params.get('long_side', None)
    if long_side is not None:
        long_side = int(long_side)

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
        long_side=long_side,
        rng=RandomConfig(
            seed=seed,
            gen=gen,
        )
    )


def resolve_image_paths(
    *,
    node_id: str,
    path: Optional[Union[str, Path, List[Union[str, Path]]]],
    input: Optional[Dict[str, Dict]],
    input_key: str = 'default',
) -> List[Path]:
    """
    Resolve one or more image filesystem paths.

    Resolution order:
      1) If `path` is provided (str/Path or list), use it.
      2) Else, read from upstream `input[input_key]` taking `image` or `path`.
         The upstream value may be a single path or a list of paths.

    Returns a list of resolved existing file paths.
    """
    def _as_list(v) -> List[Union[str, Path]]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    # 1) explicit path(s)
    srcs = _as_list(path)

    # 2) upstream path(s)
    if not srcs:
        if not input or input_key not in input:
            raise ValueError(
                f"'{node_id}': missing input '{input_key}'. Provide `path=` or wire an upstream image.")
        up = input[input_key] or {}
        src = up.get('image') or up.get('path')
        srcs = _as_list(src)

    if not srcs:
        up_keys = list((input or {}).get(
            input_key, {}).keys()) if input else []
        raise ValueError(
            f"'{node_id}': no image paths found. Expected `path=` or upstream '{input_key}' with 'image'/'path'. "
            f"Got keys={up_keys}"
        )

    # normalize + validate
    out: List[Path] = []
    for s in srcs:
        if not isinstance(s, (str, Path)):
            raise ValueError(
                f"'{node_id}': invalid path element type: {type(s).__name__} (value={s!r})"
            )
        p = Path(str(s)).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"'{node_id}': file not found: {p}")
        out.append(p)

    return out


def resolve_single_image_path(
    *,
    node_id: str,
    path: Optional[Union[str, Path, List[Union[str, Path]]]],
    input: Optional[Dict[str, Dict]],
    input_key: str = 'default',
) -> Path:
    paths = resolve_image_paths(
        node_id=node_id,
        path=path,
        input=input,
        input_key=input_key,
    )

    if len(paths) != 1:
        raise ValueError(
            f"'{node_id}': expected exactly 1 input image, got {len(paths)}. "
            "Provide a single `path` or wire a single upstream image into 'default'."
        )

    return paths[0]
