from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from morphalo.nodes.common.config_resolve import resolve_dtype, resolve_seed


# Current Qwen Diffusers pipelines used by Morphalo are run through Accelerate
# placement rather than ``pipe.to(device)``. ``balanced`` asks Accelerate to
# distribute pipeline components across available devices while balancing memory
# pressure, which is what makes these large models practical on domestic
# hardware with limited VRAM. Other device maps may become viable later, but
# they should be enabled only after being validated per pipeline.
ALLOWED_DEVICE_MAPS = {'balanced'}


@dataclass(frozen=True)
class QwenModelConfig:
    """
    Resolved model/runtime settings shared by Qwen foundation nodes.
    """

    model_id: str
    dtype: torch.dtype
    device_map: str


@dataclass(frozen=True)
class QwenImageParams:
    """
    Resolved inference parameters for the Qwen text-to-image node.
    """

    steps: int
    guidance_scale: Optional[float]
    true_cfg_scale: Optional[float]
    height: Optional[int]
    width: Optional[int]


@dataclass(frozen=True)
class QwenImageEditParams:
    """
    Resolved inference parameters for the single-image edit node.
    """

    steps: int
    true_cfg_scale: Optional[float]
    height: Optional[int]
    width: Optional[int]


@dataclass(frozen=True)
class QwenImageEditPlusParams:
    """
    Resolved inference parameters for the multi-image edit node.
    """

    steps: int
    true_cfg_scale: float
    height: Optional[int]
    width: Optional[int]
    max_images: Optional[int]


def _model_section(spec: Any) -> Dict[str, Any]:
    return spec.get('model', {}) if isinstance(spec, dict) else {}


def _params_section(spec: Any) -> Dict[str, Any]:
    return spec.get('params', {}) if isinstance(spec, dict) else {}


def _optional_int(params: Dict[str, Any], key: str) -> Optional[int]:
    value = params.get(key, None)
    return None if value is None else int(value)


def _optional_float(params: Dict[str, Any], key: str) -> Optional[float]:
    value = params.get(key, None)
    return None if value is None else float(value)


def validate_qwen_device_map(
    device_map: str,
    *,
    node_name: str,
    supported_by: str,
) -> None:
    """
    Validate the restricted Qwen ``model.device_map`` contract.

    The current Qwen nodes intentionally support only ``'balanced'`` because
    that is the tested Diffusers placement mode for these pipelines in Morphalo.
    ``node_name`` and ``supported_by`` are kept separate so errors can name both
    the failing node and the underlying pipeline/feature that imposes the limit.
    """
    if device_map not in ALLOWED_DEVICE_MAPS:
        raise ValueError(
            f'{node_name} invalid model.device_map={device_map!r}. '
            f"Only 'balanced' is currently supported by {supported_by}."
        )


def resolve_qwen_model_config(
    spec: Any,
    *,
    default_model_id: str,
    node_name: str,
    supported_by: str,
) -> QwenModelConfig:
    """
    Resolve common Qwen model settings from a node spec.

    Reads ``model.id``, ``model.dtype``, and ``model.device_map``; applies the
    node-specific default model id; converts dtype through ``resolve_dtype``; and
    validates the shared device-map restriction.
    """
    model = _model_section(spec)
    device_map = model.get('device_map', 'balanced')
    validate_qwen_device_map(
        device_map,
        node_name=node_name,
        supported_by=supported_by,
    )
    return QwenModelConfig(
        model_id=model.get('id', default_model_id),
        dtype=resolve_dtype(model.get('dtype', 'bf16')),
        device_map=device_map,
    )


def resolve_qwen_image_params(spec: Any) -> QwenImageParams:
    """
    Resolve parameters for ``QwenImage`` while preserving optional kwargs.

    ``guidance_scale``, ``true_cfg_scale``, ``height``, and ``width`` are
    returned as ``None`` when omitted so the node can avoid forwarding unsupported
    or inactive arguments to model variants that do not accept them.
    """
    params = _params_section(spec)
    return QwenImageParams(
        steps=int(params.get('steps', 25)),
        guidance_scale=_optional_float(params, 'guidance_scale'),
        true_cfg_scale=_optional_float(params, 'true_cfg_scale'),
        height=_optional_int(params, 'height'),
        width=_optional_int(params, 'width'),
    )


def resolve_qwen_image_edit_params(spec: Any) -> QwenImageEditParams:
    """
    Resolve parameters for ``QwenImageEdit``.

    ``true_cfg_scale`` and explicit output size remain optional. When omitted,
    the node lets the pipeline use its own defaults, including input-image-sized
    output where applicable.
    """
    params = _params_section(spec)
    return QwenImageEditParams(
        steps=int(params.get('steps', 25)),
        true_cfg_scale=_optional_float(params, 'true_cfg_scale'),
        height=_optional_int(params, 'height'),
        width=_optional_int(params, 'width'),
    )


def resolve_qwen_image_edit_plus_params(spec: Any) -> QwenImageEditPlusParams:
    """
    Resolve parameters for ``QwenImageEditPlus``.

    Plus uses true CFG, matching ``QwenImageEdit``. ``guidance_scale`` is not
    exposed because non guidance-distilled Qwen edit models ignore it and
    Diffusers emits a warning when it is passed. ``max_images`` is validated here
    because it is a node-level guard over the resolved image sequence, not a
    pipeline argument.
    """
    params = _params_section(spec)
    max_images = _optional_int(params, 'max_images')
    if max_images is not None and max_images <= 0:
        raise ValueError(f'params.max_images must be > 0, got {max_images}')

    return QwenImageEditPlusParams(
        steps=int(params.get('steps', 40)),
        true_cfg_scale=float(params.get('true_cfg_scale', 4.0)),
        height=_optional_int(params, 'height'),
        width=_optional_int(params, 'width'),
        max_images=max_images,
    )


def resolve_qwen_seed(spec: Any) -> int:
    """
    Resolve the common Qwen seed field, defaulting to ``'random'``.
    """
    seed_spec = spec.get('seed', 'random') if isinstance(spec, dict) else 'random'
    return resolve_seed(seed_spec)


def qwen_cpu_generator(seed: int) -> torch.Generator:
    """
    Create the CPU torch generator used by Qwen Diffusers pipelines.
    """
    return torch.Generator(device='cpu').manual_seed(seed)


def qwen_stats_device() -> str:
    """
    Return the CUDA stats device used around Qwen pipeline execution.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
