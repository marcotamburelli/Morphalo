import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import torch

from stability.core.spec_loader import load_hocon_spec

SpecLike = Union[Dict[str, Any], str, Path]
SpecInput = Union[SpecLike, Sequence[SpecLike]]


@dataclass(frozen=True)
class RandomConfig:
    seed: int
    gen: torch.Generator


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict = a deep-merged with b (b wins). Lists are replaced."""
    out = dict(a)
    for k, vb in b.items():
        if k in out:
            va = out[k]
            if isinstance(va, dict) and isinstance(vb, dict):
                out[k] = deep_merge(va, vb)
            else:
                # lists + scalars: replace
                out[k] = vb
        else:
            out[k] = vb
    return out


def resolve_spec(spec: SpecInput) -> Dict[str, Any]:
    """
    Resolve a node specification into a single configuration dictionary.

    This function normalizes and composes node specifications provided in
    different forms (in-memory dictionaries or HOCON files) into a single
    plain Python dictionary.

    In addition to single specifications, ``resolve_spec`` supports *layered
    specifications*: when a sequence of specs is provided, each element is
    resolved independently and then merged from left to right using a deep
    dictionary merge. In case of conflicts, values from later (rightmost)
    specifications override earlier ones.

    Dictionary merging rules are:
    - Nested dictionaries are merged recursively.
    - Non-dictionary values (including lists and scalars) are replaced
      entirely by the overriding value.
    - The resulting dictionary is a new object and does not mutate inputs.

    This mechanism enables clean composition of configuration presets, such as
    base model settings, experiment-level overrides, and node-local parameters,
    without duplicating full configuration files.

    Parameters
    ----------
    spec : dict or str or pathlib.Path or sequence of (dict or str or pathlib.Path)
        Node specification to resolve.

        - If a dict is provided, it is used directly.
        - If a str or Path is provided, it is interpreted as a filesystem path
          to a HOCON configuration file and loaded accordingly.
        - If a sequence is provided, each element is resolved in order and
          merged into a single configuration dictionary, with later elements
          overriding earlier ones.

    Returns
    -------
    dict
        A fully resolved configuration dictionary suitable for consumption
        by node execution logic.

    Raises
    ------
    TypeError
        If ``spec`` is not a supported type.

    Notes
    -----
    - Ordering matters when providing a sequence of specifications: parameters
      defined in later elements take precedence over earlier ones.
    - This function is intentionally agnostic to prompt wiring, adapters, or
      other DAG attachments, which are handled separately via node connections.
    """
    if isinstance(spec, dict):
        return spec

    if isinstance(spec, (str, Path)):
        return load_hocon_spec(spec)

    # layered
    if isinstance(spec, Sequence):
        merged: Dict[str, Any] = {}
        for layer in spec:
            d = resolve_spec(layer)  # recursive, layer must be dict/str/path
            merged = deep_merge(merged, d)
        return merged

    raise TypeError(f'Unsupported spec type: {type(spec)}')


def resolve_dtype(dtype: str) -> torch.dtype:
    d = (dtype or 'bf16').lower()
    if d in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if d in ('fp16', 'float16'):
        return torch.float16
    if d in ('fp32', 'float32'):
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype}')


def resolve_seed(seed: Any) -> int:
    if seed is None or seed == 'random':
        return secrets.randbelow(2**31)
    return int(seed)
