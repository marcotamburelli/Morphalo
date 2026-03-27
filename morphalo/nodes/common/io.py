import json
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image

from morphalo.core.paths import make_node_output_path


def load_init_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')

    p = Path(str(image)).expanduser()

    if not p.exists():
        raise FileNotFoundError(f'Input image not found: {p}')

    return Image.open(p).convert('RGB')


def load_faceid_embeds(path_or_paths: Union[str, List[str]], device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Load FaceID embeds and normalize to (2, N, D).

    If a list of paths is provided, we concatenate references along N (dim=1),
    preserving the (neg,pos) pairing in dim=0.
    """
    def load_one(p: str) -> torch.Tensor:
        t = torch.load(p, map_location='cpu')
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f'FaceID embeds file {p!r} did not contain a torch.Tensor. Got: {type(t)}')
        if t.ndim != 3 or t.shape[0] != 2 or t.shape[1] < 1:
            raise ValueError(
                f'FaceID embeds must have shape (2, N, D). Got {tuple(t.shape)} from {p!r}')
        return t

    if isinstance(path_or_paths, str):
        t = load_one(path_or_paths)
    else:
        ts = [load_one(p) for p in path_or_paths]
        # concatenate along N dimension
        t = torch.cat(ts, dim=1)  # (2, sum(Ni), D)

    return t.to(device=device, dtype=dtype)


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
    write_json(meta_path, payload=payload)

    return meta_path


def write_json(path: Path, payload: dict):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
