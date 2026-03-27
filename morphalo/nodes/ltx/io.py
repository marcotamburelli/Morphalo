from pathlib import Path
from typing import Optional

import torch
from diffusers.utils import export_to_video

from morphalo.core.paths import make_node_output_path
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.ltx.wiring.ic_lora import ICLoRaSpec


def save_video(
    out_dir: Path,
    *,
    node_id: str,
    seed: int,
    video_out: torch.Tensor,
    fps: int
) -> Path:
    video_path = make_node_output_path(
        out_dir=out_dir,
        node_id=node_id,
        seed=seed,
        ext='mp4'
    )

    export_to_video(video_out, str(video_path), fps=fps)
    return video_path


def finalize_video_output(
    *,
    node_kind: str,
    node_id: str,
    video_path: Path,
    seed: int,
    params: dict,
    dt_s: float,
    cuda_mem: dict,
    input_image: Optional[str] = None,
    input_video: Optional[str] = None,
    ic_lora_specs: Optional[ICLoRaSpec] = None,
    model_info: Optional[dict] = None,
) -> dict:
    out = {
        'ok': True,
        'node': node_kind,
        'id': node_id,
        'video': str(video_path),
        'seed': seed,
        'params': params,
        'timing': {'seconds': round(dt_s, 3)}
    }

    if cuda_mem:
        out['cuda_mem'] = cuda_mem

    if input_image is not None:
        out['input_image'] = input_image

    if input_video is not None:
        out['input_video'] = input_video

    if ic_lora_specs is not None:
        out['ic_lora'] = {
            'model_id': ic_lora_specs.model_id,
            'weight_name': ic_lora_specs.weight_name,
            'adapter_name': ic_lora_specs.adapter_name,
            'adapter_weight': ic_lora_specs.adapter_weight,
        }

    if model_info is not None:
        out['model'] = model_info

    meta_path = write_json_sidecar(video_path, out)
    out['metadata'] = str(meta_path)
    return out
