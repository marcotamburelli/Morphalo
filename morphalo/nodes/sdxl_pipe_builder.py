
from typing import Any, Dict

import torch

from morphalo.nodes.wiring.controlnet import ControlNetBundle
from morphalo.nodes.wiring.face_id import FaceIdBundle
from morphalo.nodes.wiring.ip_adapter import IpAdapterBundle
from morphalo.nodes.wiring.lora import LoraBundle
from morphalo.nodes.wiring.t2i_adapter import T2IAdapterBundle


def build_cross_attention_kwargs(
    *,
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype,
    lora_bundle: LoraBundle = None,
):
    kwargs = {}
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

        cross_attention_kwargs['ip_adapter_masks'] = ip_masks + face_masks

    if lora_bundle is not None and lora_bundle.has_lora:
        cross_attention_kwargs['scale'] = lora_bundle.cross_attention_scale

    if cross_attention_kwargs:
        kwargs['cross_attention_kwargs'] = cross_attention_kwargs

    return kwargs


def build_pipe_kwargs(
    *,
    cn_bundle: ControlNetBundle,
    t2i_bundle: T2IAdapterBundle = None,
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype,
    init_image_already_passed: bool = False,
    lora_bundle: LoraBundle = None,
) -> Dict[str, Any]:
    """
    Assemble keyword arguments for a Diffusers SDXL pipeline call from the
    various conditioning bundles.

    This helper centralizes the translation between framework-level bundles
    (ControlNet, T2I-Adapter, IP-Adapter, FaceID) and the concrete keyword
    arguments expected by the underlying Diffusers pipeline.

    In particular, it handles the ambiguity around the ``image`` argument
    in SDXL ControlNet pipelines, whose meaning depends on whether an init
    image is already provided by the calling node.

    Parameters
    ----------
    cn_bundle : ControlNetBundle
        Bundle describing the ControlNet configuration and control images.
        If ``cn_bundle.has_controlnet`` is True, a control image and its
        conditioning scale will be added to the returned kwargs.

        - If ``init_image_already_passed`` is False (e.g. txt2img),
          the control image is passed via the ``image`` keyword.
        - If ``init_image_already_passed`` is True (e.g. img2img or inpaint),
          the control image is passed via the ``control_image`` keyword.

    t2i_bundle : T2IAdapterBundle, optional
        Bundle describing T2I-Adapter configuration. In this framework,
        T2I-Adapters are assumed to be txt2img-only. If provided and active,
        adapter-related kwargs are added accordingly.

    ip_bundle : IpAdapterBundle
        Bundle describing IP-Adapter configuration. If active, the reference
        image(s) are passed via ``ip_adapter_image``.

    face_bundle : FaceIdBundle
        Bundle describing FaceID configuration. If active, precomputed image
        embeddings are passed via ``ip_adapter_image_embeds``.

    lora_bundle : LoraBundle, optional
        Bundle describing first-class LoRA configuration. If active, its global
        call-time scale is passed via ``cross_attention_kwargs['scale']``.

    height : int
        Target image height, used for constructing cross-attention masks.

    width : int
        Target image width, used for constructing cross-attention masks.

    device : str
        Device identifier (e.g. ``"cuda"`` or ``"cpu"``), forwarded to
        cross-attention helper utilities.

    dtype : torch.dtype
        Torch dtype used for tensor construction in attention masks.

    init_image_already_passed : bool, default=False
        Whether the calling node will explicitly pass an init image via the
        ``image=`` argument when invoking the pipeline.

        This flag disambiguates the meaning of ``image`` in SDXL ControlNet
        pipelines:
        - False: ``image`` is interpreted as the ControlNet conditioning image
          (txt2img case).
        - True: ``image`` is reserved for the init image, and ControlNet
          conditioning is passed via ``control_image`` instead
          (img2img / inpaint cases).

    Returns
    -------
    Dict[str, Any]
        A dictionary of keyword arguments to be expanded into a Diffusers
        pipeline call.

    Notes
    -----
    - This function must never introduce an ``image`` keyword when
      ``init_image_already_passed`` is True, as that would collide with the
      init image provided by the node.
    - Cross-attention masks for IP-Adapter and FaceID are constructed and
      merged via ``build_cross_attention_kwargs``.
    - This function does not perform any pipeline mutation (e.g. loading
      adapters); it only prepares call-time keyword arguments.
    """
    kwargs: Dict[str, Any] = {}

    # ControlNet
    if cn_bundle.has_controlnet:
        if init_image_already_passed:
            kwargs['control_image'] = cn_bundle.control_image_arg
        else:
            kwargs['image'] = cn_bundle.control_image_arg

        kwargs['controlnet_conditioning_scale'] = cn_bundle.conditioning_scale_arg

    # T2I-Adapter
    if t2i_bundle is not None and t2i_bundle.has_t2i_adapter:
        if init_image_already_passed:
            raise ValueError(
                'T2I-Adapter not supported for img2img/inpaint in this framework.'
            )

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
        lora_bundle=lora_bundle,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    ))

    return kwargs
