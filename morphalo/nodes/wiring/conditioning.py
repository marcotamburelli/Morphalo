from typing import List, Optional, Sequence, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.loaders import IPAdapterMixin, StableDiffusionXLLoraLoaderMixin
from PIL import Image

from morphalo.nodes.wiring.face_id import FaceIdBundle
from morphalo.nodes.wiring.ip_adapter import IpAdapterBundle

SDXLPipeline = DiffusionPipeline | IPAdapterMixin | StableDiffusionXLLoraLoaderMixin


def _as_image_list(x: Union[Image.Image, Sequence[Image.Image]]) -> List[Image.Image]:
    """Normalize a slot image payload to a list of PIL images."""
    if isinstance(x, Image.Image):
        return [x]
    return list(x)


def _aggregate_slot_clip_embeds(
    pipe,
    *,
    slot_index: int,
    slot_images: Union[Image.Image, Sequence[Image.Image]],
    base_images: List[Optional[Image.Image]],
    device: torch.device,
    num_images_per_prompt: int,
    do_cfg: bool,
) -> torch.Tensor:
    """
    Compute and aggregate CLIP embeds for a single FaceID Plus/PlusV2 slot.

    This uses Diffusers `pipe.prepare_ip_adapter_image_embeds(...)` to ensure
    correct preprocessing, CFG duplication, and num_images_per_prompt expansion.

    Parameters
    ----------
    pipe
        A diffusers pipeline instance that implements `prepare_ip_adapter_image_embeds`
        and has FaceID Plus/PlusV2 adapters loaded.
    slot_index
        Index of the FaceID slot (aligned with adapter/projection layer order).
    slot_images
        One PIL image or a sequence of PIL images to be aggregated for this slot.
    base_images
        A list of images aligned with all slots, used as the baseline input to
        `prepare_ip_adapter_image_embeds`. For the current slot, this function will
        temporarily replace `base_images[slot_index]` with each reference image.
    device
        Torch device where embeddings should be placed.
    num_images_per_prompt
        Diffusers generation parameter (images per prompt).
    do_cfg
        Whether classifier-free guidance is enabled.

    Returns
    -------
    torch.Tensor
        Aggregated CLIP embeds for the slot, already multiplied by `strength`.
        The returned tensor matches Diffusers batch expansion for CFG and
        num_images_per_prompt.
    """
    imgs = _as_image_list(slot_images)
    if len(imgs) == 1:
        tmp = list(base_images)
        tmp[slot_index] = imgs[0]
        embeds = pipe.prepare_ip_adapter_image_embeds(
            tmp, None, device, num_images_per_prompt, do_cfg
        )
        return embeds[slot_index]

    # Accumulate per-reference embeds and average.
    acc = None
    for im in imgs:
        tmp = list(base_images)
        tmp[slot_index] = im
        embeds = pipe.prepare_ip_adapter_image_embeds(
            tmp, None, device, num_images_per_prompt, do_cfg
        )
        e = embeds[slot_index]
        acc = e if acc is None else (acc + e)

    out = acc / float(len(imgs))
    return out


def _clear_faceid_clip_state(pipe: SDXLPipeline) -> None:
    """
    Clear manually injected FaceID Plus / PlusV2 CLIP state without guessing
    layer defaults.

    Notes
    -----
    This helper intentionally clears only ``clip_embeds``. It does not reset
    ``shortcut`` because the correct default may depend on the projection layer
    implementation and adapter variant. The subsequent ``unload_ip_adapter()``
    call removes the projection layers themselves.
    """
    encoder_hid_proj = getattr(pipe.unet, 'encoder_hid_proj', None)
    if encoder_hid_proj is None:
        return

    layers = getattr(encoder_hid_proj, 'image_projection_layers', None)
    if not layers:
        return

    for layer in layers:
        if hasattr(layer, 'clip_embeds'):
            layer.clip_embeds = None


def apply_ip_adapter(
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    pipe: SDXLPipeline,
    batch: int,
    device: str,
    dtype: torch.dtype,
):
    """
    Configure IP-Adapter or FaceID conditioning on a diffusion pipeline.

    This function inspects the provided bundles and mutates the given
    pipeline to enable either:

    - standard IP-Adapter conditioning, or
    - FaceID conditioning (including optional CLIP-based conditioning for
      Plus / PlusV2 variants).

    The two modes are mutually exclusive. Passing both bundles as active
    is an error.

    Parameters
    ----------
    ip_bundle : IpAdapterBundle
        Runtime bundle describing standard IP-Adapter configuration.
        If ``has_ip_adapter`` is True, the corresponding adapter(s) are
        loaded and configured on the pipeline.

    face_bundle : FaceIdBundle
        Runtime bundle describing FaceID configuration. Used only if no
        standard IP-Adapter is active. Supports both base FaceID and
        Plus / PlusV2 variants.

    pipe : DiffusionPipeline | IPAdapterMixin | StableDiffusionXLLoraLoaderMixin
        Diffusers pipeline instance to be configured. This function
        mutates the pipeline in-place by loading adapter weights and
        registering required modules.

    batch : int
        Number of images per prompt (``num_images_per_prompt``).

        This value MUST match the batch used when calling the pipeline.
        It is required to correctly prepare CLIP image embeddings for
        FaceID Plus / PlusV2 variants, ensuring alignment with the
        internal batch size (including classifier-free guidance).

    device : str
        Target device for model execution (e.g. ``'cuda'``).

    dtype : torch.dtype
        Data type used for model weights and intermediate tensors.

    Behavior
    --------
    - Always clears manually injected FaceID CLIP state, unloads any previously
      configured IP-Adapter state, and unloads LoRA weights that may have been
      loaded as FaceID side effects before applying new conditioning.

    - If ``ip_bundle.has_ip_adapter`` is True:
        - registers the corresponding image encoder
        - loads IP-Adapter weights via ``load_ip_adapter``
        - sets adapter scale via ``set_ip_adapter_scale``

    - Else if ``face_bundle.has_face_id`` is True:
        - optionally registers the CLIP image encoder for Plus / PlusV2
        - loads FaceID weights via ``load_ip_adapter``
        - sets FaceID scale configuration
        - applies CLIP-based conditioning via ``apply_faceid_clip``

    - Else:
        - leaves the pipeline with no active IP-Adapter / FaceID conditioning.

    Notes
    -----
    - This function performs in-place mutation of the pipeline.
    - The ``batch`` parameter is critical for FaceID Plus / PlusV2:
      mismatched values may lead to tensor shape errors during UNet
      forward passes.
    - IP-Adapter and FaceID conditioning are treated as mutually exclusive
      to avoid conflicts in adapter loading and projection layers.
    """

    if ip_bundle.has_ip_adapter and face_bundle.has_face_id:
        raise ValueError(
            'IP-Adapter and IP-Adapter-FaceID are mutually exclusive.'
        )

    # Always reset adapter state before loading a new adapter configuration.
    # Diffusers IP-Adapter loading mutates the pipeline in-place, FaceID Plus /
    # PlusV2 injects runtime CLIP tensors into projection layers, and some
    # FaceID variants may load auxiliary LoRA adapters.
    _clear_faceid_clip_state(pipe)
    pipe.unload_ip_adapter()
    # TODO: Revisit this when Morphalo adds first-class LoRA support.
    # ``unload_lora_weights()`` clears all LoRA adapters, not only FaceID
    # side-effect LoRAs. This is acceptable for now because Morphalo does not
    # yet expose LoRA as an independent conditioning mechanism. Once it does,
    # LoRA cleanup should become selective or move into the LoRA-specific
    # configuration block.
    pipe.unload_lora_weights()

    if ip_bundle.has_ip_adapter:
        pipe.register_modules(image_encoder=ip_bundle.image_encoder)
        pipe.load_ip_adapter(
            ip_bundle.model_id_arg,
            subfolder=ip_bundle.subfolder_arg,
            weight_name=ip_bundle.weight_names_arg
        )
        pipe.set_ip_adapter_scale(ip_bundle.scale_arg)
    elif face_bundle.has_face_id:
        if face_bundle.image_encoder is not None:
            pipe.register_modules(image_encoder=face_bundle.image_encoder)

        pipe.load_ip_adapter(
            face_bundle.model_id_arg,
            subfolder=None,
            weight_name=face_bundle.weight_names_arg,
            image_encoder_folder=None
        )
        pipe.set_ip_adapter_scale(face_bundle.scale_arg)

        apply_faceid_clip(
            face_bundle=face_bundle,
            pipe=pipe,
            device=torch.device(device),
            dtype=dtype,
            num_images=batch,
        )


def apply_faceid_clip(
    *,
    face_bundle: FaceIdBundle,
    pipe: SDXLPipeline,
    device: torch.device,
    dtype: torch.dtype,
    num_images: int = 1
):
    """
    Inject CLIP image embeddings into the hidden projection layers for
    IP-Adapter FaceID Plus / PlusV2 models.

    This function must be called AFTER FaceID weights are loaded via
    `pipe.load_ip_adapter(...)` and BEFORE invoking the pipeline (`pipe(...)`).

    Notes
    -----
    Diffusers computes CLIP embeddings for IP-Adapters through
    `prepare_ip_adapter_image_embeds(ip_adapter_image=..., ip_adapter_image_embeds=None, ...)`.
    When `ip_adapter_image_embeds` is None, `ip_adapter_image` MUST be a list whose
    length matches the number of loaded IP-Adapters (i.e. the number of projection layers).

    Therefore, when multiple FaceID adapters are loaded, we compute CLIP embeddings
    in one call using a per-adapter list of images, then inject the resulting tensors
    into the corresponding projection layers.

    Parameters
    ----------
    face_bundle:
        FaceID bundle. Only slots marked as `requires_clip=True` (Plus/PlusV2)
        will be injected; other slots are ignored.

    pipe:
        Diffusers pipeline with FaceID IP-Adapters already loaded.

    device, dtype:
        Torch device/dtype used by the pipeline.

    num_images:
        Number of images generated per prompt (equivalent to
        `num_images_per_prompt` in Diffusers).

        In the common case where a single image is generated per prompt,
        this value should be set to 1 (default).

        This parameter is required so Diffusers can correctly replicate
        CLIP embeddings to match the internal batch size. If you are not
        using batching or generating multiple images per prompt, using
        `num_images = 1` is correct and sufficient.
    """
    if not face_bundle.has_face_id:
        return

    # list[Optional[(images, is_plusv2)]], len = n_face_adapters
    entries = face_bundle.clip_images_per_slot

    # entries is aligned with FaceID slots; entries[j] is None for non-clip weights.
    if not entries or all(e is None for e in entries):
        return

    layers = pipe.unet.encoder_hid_proj.image_projection_layers
    n_layers = len(layers)

    if n_layers != len(face_bundle.weight_names_arg):
        raise RuntimeError(
            f'FaceID adapters/layers mismatch: layers={n_layers} but face weights={len(face_bundle.weight_names_arg)}.'
        )

    blank = Image.new('RGB', (224, 224), (0, 0, 0))

    # Build a baseline image list of length == n_layers (ONE image per slot).
    #
    # IMPORTANT:
    # Passing nested lists (list[list[PIL.Image]]) may trigger "multi-reference" code paths
    # in diffusers and lead to shape drift / batch mismatches. We keep the diffusers call
    # strictly on a flat list of images, and explicitly aggregate multi-image slots by
    # averaging the resulting CLIP embeddings.
    # Keep previous semantics; consider making this explicit later.
    do_cfg = True

    base_images: List[Image.Image] = []
    for j, entry in enumerate(entries):
        if entry is None:
            base_images.append(blank)
            continue

        # ClipImg.image is normalized to List[Image.Image]
        imgs = entry.image
        if not imgs:
            raise ValueError(f'FaceID CLIP slot {j} has an empty image list.')
        base_images.append(imgs[0])

    # Compute baseline embeds for ALL adapters in one call.
    clip_embeds_per_layer = pipe.prepare_ip_adapter_image_embeds(
        base_images,
        None,
        device,
        num_images,
        do_cfg,
    )

    # Aggregate multi-image slots (Plus/PlusV2 refinement) by averaging CLIP embeds.
    for j, entry in enumerate(entries):
        if entry is None:
            continue

        imgs = entry.image
        if len(imgs) <= 1:
            continue

        clip_embeds_per_layer[j] = _aggregate_slot_clip_embeds(
            pipe,
            slot_index=j,
            slot_images=imgs,
            base_images=base_images,
            device=device,
            num_images_per_prompt=num_images,
            do_cfg=do_cfg,
        )

    # Inject only where required (Plus/PlusV2)
    for j, entry in enumerate(entries):
        if entry is None:
            continue

        is_plusv2 = entry.is_plusv2
        clip_strength = entry.clip_strength
        clip_embeds = clip_embeds_per_layer[j].to(device=device, dtype=dtype)

        if clip_strength != 1.0:
            clip_embeds = clip_embeds * float(clip_strength)

        layer = layers[j]
        layer.clip_embeds = clip_embeds

        if is_plusv2:
            layer.shortcut = False
