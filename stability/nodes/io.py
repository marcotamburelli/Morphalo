from pathlib import Path
from typing import List, Optional

from stability.nodes.common.io import write_json_sidecar
from stability.nodes.wiring.controlnet import ControlNetSpec
from stability.nodes.wiring.face_id import FaceIdSpec
from stability.nodes.wiring.ip_adapter import IpAdapterSpec
from stability.nodes.wiring.t2i_adapter import T2IAdapterSpec


def finalize_image_output(
    *,
    node_kind: str,
    node_id: str,
    img_path: Path,
    seed: int,
    params: dict,
    dt_s: float,
    cuda_mem: dict,
    controlnet_specs: List[ControlNetSpec] = None,
    t2i_adapter_specs: List[T2IAdapterSpec] = None,
    ip_adapter_specs: List[IpAdapterSpec] = None,
    face_id_specs: List[FaceIdSpec] = None,
    model_info: Optional[dict] = None,
) -> dict:
    out = {
        'ok': True,
        'node': node_kind,
        'id': node_id,
        'image': str(img_path),
        'seed': seed,
        'params': params,
        'timing': {'seconds': round(dt_s, 3)}
    }

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

    if model_info is not None:
        out['model'] = model_info

    meta_path = write_json_sidecar(img_path, out)
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
