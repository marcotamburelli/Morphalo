
from typing import Any, Dict

import torch

from stability.nodes.wiring.controlnet import ControlNetBundle
from stability.nodes.wiring.face_id import FaceIdBundle
from stability.nodes.wiring.ip_adapter import IpAdapterBundle
from stability.nodes.wiring.t2i_adapter import T2IAdapterBundle


def build_cross_attention_kwargs(
    *,
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype
):
    cross_attention_kwargs = {}

    if (ip_bundle.with_mask or face_bundle.with_mask):
        ip_masks = ip_bundle.build_ip_adapter_masks(
            height=height,
            width=width,
            device=device,
            dtype=dtype
        )

        face_masks = face_bundle.build_ip_adapter_masks(
            height=height,
            width=width,
            device=device,
            dtype=dtype
        )

        cross_attention_kwargs['cross_attention_kwargs'] = {
            'ip_adapter_masks': ip_masks + face_masks
        }

    return cross_attention_kwargs


def build_pipe_kwargs(
    *,
    cn_bundle: ControlNetBundle,
    t2i_bundle: T2IAdapterBundle = None,
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    height: int,
    width: int,
    device: str,
    dtype,
) -> Dict[str, Any]:
    """
    Assemble kwargs for pipe(...) from all bundles.

    Notes:
      - FaceID Plus/V2 CLIP injection is NOT done here; call apply_faceid_clip(...)
        after load_ip_adapter and before pipe(...).
      - Mask wiring logic is delegated to build_cross_attention_kwargs(...).
    """
    kwargs: Dict[str, Any] = {}

    # ControlNet
    if cn_bundle.has_controlnet:
        kwargs.update({
            'image': cn_bundle.control_image_arg,
            'controlnet_conditioning_scale': cn_bundle.conditioning_scale_arg,
        })

    # T2I-Adapter
    if t2i_bundle is not None and t2i_bundle.has_t2i_adapter:
        kwargs.update({
            'image': t2i_bundle.adapter_image_arg,
            'adapter_conditioning_scale': t2i_bundle.conditioning_scale_arg,
        })

    # IP-Adapter (normal)
    if ip_bundle.has_ip_adapter:
        kwargs['ip_adapter_image'] = ip_bundle.ip_adapter_image

    # FaceID
    if face_bundle.has_face_id:
        kwargs['ip_adapter_image_embeds'] = face_bundle.ip_adapter_image_embeds

    # Cross-attention masks (IP + FaceID), merged inside the helper
    kwargs.update(build_cross_attention_kwargs(
        ip_bundle=ip_bundle,
        face_bundle=face_bundle,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    ))

    return kwargs
