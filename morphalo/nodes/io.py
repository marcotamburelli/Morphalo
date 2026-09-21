from pathlib import Path
from typing import List, Optional

from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.wiring.controlnet import ControlNetSpec
from morphalo.nodes.wiring.face_id import FaceIdSpec
from morphalo.nodes.wiring.ip_adapter import IpAdapterSpec
from morphalo.nodes.wiring.lora import LoraSpec
from morphalo.nodes.wiring.t2i_adapter import T2IAdapterSpec


def finalize_image_output(
    *,
    node_kind: str,
    node_id: str,
    img_path: Path | List[Path],
    seed: int | List[int],
    params: dict,
    dt_s: float,
    cuda_mem: dict,
    controlnet_specs: List[ControlNetSpec] = None,
    t2i_adapter_specs: List[T2IAdapterSpec] = None,
    ip_adapter_specs: List[IpAdapterSpec] = None,
    face_id_specs: List[FaceIdSpec] = None,
    lora_specs: List[LoraSpec] = None,
    model_info: Optional[dict] = None,
    sampling_scheduler: Optional[str] = None,
) -> dict:
    """
    Build the final output payload and write a single JSON sidecar for the node.

    This helper supports both single-image and batched-image outputs.

    Normalization rules
    -------------------
    Internally, image paths and seeds are normalized to lists.

    Public output rules
    -------------------
    - If exactly one image is present, expose:
      - ``image``
      - ``seed``
    - If more than one image is present, expose:
      - ``images``
      - ``seeds``

    Sidecar naming
    --------------
    Only one JSON sidecar is written per node execution. When multiple images are
    present, the metadata file is anchored to the lexicographically smallest image
    path to keep the naming deterministic and stable.

    Parameters
    ----------
    node_kind : str
        Node type / operator name.
    node_id : str
        Node identifier.
    img_path : Path | list[Path]
        Output image path or list of output image paths.
    seed : int | list[int]
        Seed or list of seeds aligned with ``img_path``.
    params : dict
        Generation parameters to store in metadata.
    dt_s : float
        Execution time in seconds.
    cuda_mem : dict
        CUDA memory statistics.
    controlnet_specs : list[ControlNetSpec], optional
        ControlNet configuration metadata.
    t2i_adapter_specs : list[T2IAdapterSpec], optional
        T2I-Adapter configuration metadata.
    ip_adapter_specs : list[IpAdapterSpec], optional
        IP-Adapter configuration metadata.
    face_id_specs : list[FaceIdSpec], optional
        FaceID configuration metadata.
    lora_specs : list[LoraSpec], optional
        LoRA configuration metadata.
    model_info : dict, optional
        Additional model metadata.
    sampling_scheduler : str, optional
        Sampling scheduler profile used for this generation, when it is part of
        the node spec and distinct from model identity.

    Returns
    -------
    dict
        Final node output payload, including metadata path.

    Raises
    ------
    ValueError
        If image paths or seeds are empty, or if their lengths do not match.
    """
    img_paths = [img_path] if isinstance(img_path, Path) else list(img_path)
    seeds = [seed] if isinstance(seed, int) else list(seed)

    if not img_paths:
        raise ValueError('finalize_image_output: img_path cannot be empty')

    if not seeds:
        raise ValueError('finalize_image_output: seed cannot be empty')

    if len(img_paths) != len(seeds):
        raise ValueError(
            'finalize_image_output: number of image paths must match number '
            f'of seeds, got {len(img_paths)} images and {len(seeds)} seeds'
        )

    out = {
        'ok': True,
        'node': node_kind,
        'id': node_id,
        'params': params,
        'timing': {'seconds': round(dt_s, 3)},
    }

    if len(img_paths) == 1:
        out['image'] = str(img_paths[0])
        out['seed'] = seeds[0]
    else:
        out['images'] = [str(p) for p in img_paths]
        out['seeds'] = seeds
        out['batch'] = len(seeds)

    if cuda_mem:
        out['cuda_mem'] = cuda_mem

    if controlnet_specs:
        out['controlnet'] = controlnet_meta(controlnet_specs)

    if t2i_adapter_specs:
        out['t2i-adapters'] = t2i_adapter_meta(t2i_adapter_specs)

    if ip_adapter_specs:
        out['ip-adapters'] = ip_adapter_meta(ip_adapter_specs)

    if face_id_specs:
        out['face-id'] = face_id_meta(face_id_specs)

    if lora_specs:
        out['loras'] = lora_meta(lora_specs)

    if model_info is not None:
        out['model'] = model_info

    if sampling_scheduler is not None:
        out['sampling_scheduler'] = sampling_scheduler

    meta_anchor = min(img_paths, key=lambda p: str(p))
    meta_path = write_json_sidecar(meta_anchor, out)
    out['metadata'] = str(meta_path)

    return out

def controlnet_meta(controlnet_specs: List[ControlNetSpec]) -> list[dict]:
    return [
        {
            'key': cn.key,
            'model_id': cn.model_id,
            'conditioning_scale': cn.conditioning_scale,
            'input_id': f'controlnet:{cn.key}',
        }
        for cn in controlnet_specs
    ]


def ip_adapter_meta(ip_adapter_specs: List[IpAdapterSpec]) -> list[dict]:
    return [
        {
            'key': ipa.key,
            'model_id': ipa.model_id,
            'weight_name': ipa.weight_name,
            'subfolder': ipa.subfolder,
            'scale': ipa.scale,
            'input_id': f'ip-adapter:{ipa.key}',
        }
        for ipa in ip_adapter_specs
    ]


def face_id_meta(face_specs: List[FaceIdSpec]) -> list[dict]:
    return [
        {
            'key': fid.key,
            'model_id': fid.model_id,
            'weight_name': fid.weight_name,
            'subfolder': fid.subfolder,
            'scale': fid.scale,
            'clip_strength': fid.clip_strength,
            'has_mask': fid.has_mask,
            'input_id': f'face_id:{fid.key}',
            'mask_input_id': f'ip_adapter_mask:{fid.key}' if fid.has_mask else None,
        }
        for fid in face_specs
    ]


def t2i_adapter_meta(t2i_adapter_specs: List[T2IAdapterSpec]) -> list[dict]:
    return [
        {
            'key': a.key,
            'model_id': a.model_id,
            'conditioning_scale': a.conditioning_scale,
            'input_id': f't2i-adapter:{a.key}',
        }
        for a in (t2i_adapter_specs or [])
    ]


def lora_meta(lora_specs: List[LoraSpec]) -> list[dict]:
    return [
        {
            'key': lora.key,
            'model_id': lora.model_id,
            'weight_name': lora.weight_name,
            'adapter_name': lora.adapter_name,
            'adapter_weight': lora.adapter_weight,
        }
        for lora in (lora_specs or [])
    ]
