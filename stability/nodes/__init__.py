import secrets
from pathlib import Path
from typing import Any, Union

import torch
from PIL import Image


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


def norm_prompt(prompt: dict, joiner: str = '\n') -> str:
    # prompt = spec.get('prompt')
    if prompt is None:
        return ''

    if isinstance(prompt, str):
        return prompt.strip()

    if isinstance(prompt, list):
        return joiner.join(prompt).strip()


def load_init_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')

    p = Path(str(image)).expanduser()

    if not p.exists():
        raise FileNotFoundError(f'Input image not found: {p}')

    return Image.open(p).convert('RGB')
