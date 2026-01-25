import json
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from stability.cache.models import *
from stability.core.spec_loader import load_hocon_spec


def resolve_spec(spec: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """
    Resolve a node specification into a plain dictionary.

    If `spec` is a dict, it is returned as-is.
    If `spec` is a string or Path, it is interpreted as a HOCON file path
    and loaded accordingly.
    """
    if isinstance(spec, dict):
        return spec

    if isinstance(spec, (str, Path)):
        return load_hocon_spec(spec)

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


def norm_prompt(value: Any, *, joiner: str = '\n') -> str:
    """
    Normalize a prompt that can be:
      - str
      - list[str]
    """

    if value is None:
        return ''

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        # filter Nones / non-str defensively
        parts = [str(x).strip()
                 for x in value if x is not None and str(x).strip()]
        return joiner.join(parts).strip()

    # be strict: better to fail early than silently stringify weird objects
    raise TypeError(
        f'Prompt must be str or list[str], got {type(value).__name__}')


def norm_prompt_pair(value: Any, *, joiner: str = '\n') -> Tuple[str, str]:
    """
    Normalize a prompt that can be:
      - str
      - list[str]
      - dict with keys: content/style (each str or list[str])

    Returns (content, style).
    """

    if value is None:
        return '', ''

    # legacy / simple form: treat as content
    if isinstance(value, (str, list)):
        return norm_prompt(value, joiner=joiner), ''

    # structured form
    if isinstance(value, dict):
        content = norm_prompt(value.get('content'), joiner=joiner)
        style = norm_prompt(value.get('style'), joiner=joiner)
        return content, style

    raise TypeError(
        f'Prompt must be str, list[str], or dict, got {type(value).__name__}')


def load_init_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')

    p = Path(str(image)).expanduser()

    if not p.exists():
        raise FileNotFoundError(f'Input image not found: {p}')

    return Image.open(p).convert('RGB')


def make_node_output_path(
    *,
    out_dir: Path,
    node_id: str,
    ext: str = 'png',
    seed: Optional[int] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Generate an output file path for a node execution.

    The path is created under ``out_dir / node_id`` and uses a timestamp-based
    filename. Optional components such as the random seed can be included
    to make the filename more informative while keeping semantics out of
    directory names.

    Parameters
    ----------
    out_dir : Path
        Base output directory of the DAG execution.
    node_id : str
        Identifier of the node producing the output.
    ext : str, optional
        File extension (without leading dot). Default is ``'png'``.
    seed : int, optional
        Optional random seed to include in the filename.
    tag : str, optional
        Optional tag to override the default timestamp-based tag.

    Returns
    -------
    Path
        Full filesystem path where the output file should be written.

    Notes
    -----
    - The directory ``out_dir / node_id`` is created if it does not exist.
    - The default filename format is ``YYYY-MM-DD_HHMMSS[_seedN].<ext>``.
    """
    node_dir = Path(out_dir) / node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = time.strftime('%Y-%m-%d_%H%M%S')

    parts = [tag]
    if seed is not None:
        parts.append(f'seed{seed}')

    filename = '_'.join(parts) + f'.{ext.lstrip('.')}'
    return node_dir / filename


def cuda_prerun(device: str) -> None:
    """
    Prepare CUDA state for timing/memory measurements.

    Resets peak memory stats and synchronizes the device so subsequent timing
    reflects the work of the current run only.
    """
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_sync(device: str) -> None:
    """Synchronize CUDA device if running on CUDA."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def cuda_mem_stats(device: str) -> dict:
    """
    Return CUDA memory statistics in GB, or an empty dict if not on CUDA.
    """
    if torch.device(device).type != 'cuda':
        return {}

    return {
        'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 3),
        'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 3),
        'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def ensure_out_dir(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def save_image(out_dir: Path, *, node_id: str, seed: int, img: Image.Image) -> Path:
    img_path = make_node_output_path(
        out_dir=out_dir,
        node_id=node_id,
        seed=seed
    )

    img.save(img_path)
    return img_path


def write_json_sidecar(out_path: Path, payload: dict) -> Path:
    meta_path = out_path.with_suffix('.json')
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    return meta_path


def infer_image_encoder_subfolder(weight_names: List[str],) -> Optional[CLIPVisionModelWithProjection]:
    def _needs_encoder_for_faceid_plus(weights: list[str]) -> bool:
        # FaceID Plus/PlusV2 SDXL bin -> expects 1280 (common mismatch if using 1664)
        s = ' '.join(w.lower() for w in weights if w)
        return bool(re.search(r'ip-adapter-faceid-(plus|plusv2)_sdxl\.bin', s))

    def _needs_vit_encoder(weights: list[str]) -> bool:
        # If you explicitly use 'sdxl_models/image_encoder', it is 1664 hidden size.
        # Heuristic: weights that mention vit-h / vit_h / vitl are usually paired with that.
        s = ' '.join(w.lower() for w in weights if w)
        return ('vit-h' in s) or ('vit_h' in s) or ('vitl' in s)

    need_fip = _needs_encoder_for_faceid_plus(weight_names)
    need_vit = _needs_vit_encoder(weight_names)

    if need_fip and need_vit:
        raise ValueError(
            'You are mixing IP-Adapter families that require different CLIP image encoders '
            '(1280 vs 1664). Diffusers only supports one `pipe.image_encoder` at a time.'
        )

    if need_fip:
        # If FaceID Plus/PlusV2 SDXL is present -> pick 1280
        return 'models/image_encoder'
    elif need_vit:
        # Otherwise, if you are clearly in the 1664 family:
        return 'models/image_encoder'
    else:
        # Default: safest for FaceID Plus/PlusV2 SDXL is 1280; for generic SDXL you may prefer 1664
        # but I'd keep default conservative to avoid your current crash:
        return 'sdxl_models/image_encoder'
