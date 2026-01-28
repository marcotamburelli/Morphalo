
import torch
from diffusers import DiffusionPipeline
from diffusers.loaders import IPAdapterMixin
from PIL import Image

from stability.nodes.wiring.face_id import FaceIdBundle
from stability.nodes.wiring.ip_adapter import IpAdapterBundle


def apply_ip_adapter(
    ip_bundle: IpAdapterBundle,
    face_bundle: FaceIdBundle,
    pipe: DiffusionPipeline | IPAdapterMixin,
    device: str,
    dtype: torch.dtype,
):
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
        )

    else:
        pipe.unload_ip_adapter()


def apply_faceid_clip(
    *,
    face_bundle: FaceIdBundle,
    pipe: DiffusionPipeline | IPAdapterMixin,
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
            f"FaceID adapters/layers mismatch: layers={n_layers} but face weights={len(face_bundle.weight_names_arg)}."
        )

    blank = Image.new("RGB", (224, 224), (0, 0, 0))

    # Build per-adapter image list of length == n_layers.
    # Even for non-clip slots we provide images to satisfy diffusers' length check.
    # (Cost: extra encode; Benefit: perfect alignment + identical semantics to diffusers.)
    ip_adapter_images: list = []
    for j, entry in enumerate(entries):
        if entry is None:
            ip_adapter_images.append([blank])
        else:
            images = entry.image
            img_list = images if isinstance(images, list) else [images]
            ip_adapter_images.append(img_list)

    # Compute embeds for ALL adapters in one call (diffusers requirement).
    # This returns a list: one tensor per adapter layer, already expanded for CFG and num_images.
    clip_embeds_per_layer = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_images,
        None,
        device,
        num_images,
        True,  # do_classifier_free_guidance; consistent with typical guidance_scale>1 usage
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
