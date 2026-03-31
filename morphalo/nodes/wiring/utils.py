from pathlib import PurePosixPath
from typing import List


from pathlib import PurePosixPath
from typing import List


def infer_image_encoder_subfolder(weight_names: List[str]) -> str:
    """
    Infer the CLIP image-encoder subfolder required by a set of IP-Adapter
    or FaceID weight files.

    This function uses an explicit mapping for known SDXL IP-Adapter families
    from the h94 repositories. It intentionally avoids substring heuristics,
    because different SDXL weights may require different CLIP image encoders.

    Supported mapping
    -----------------
    Standard SDXL IP-Adapter
    - ip-adapter_sdxl.bin -> 'sdxl_models/image_encoder'
      Uses OpenCLIP-ViT-bigG-14.
    - ip-adapter_sdxl_vit-h.bin -> 'models/image_encoder'
      Uses OpenCLIP-ViT-H-14.
    - ip-adapter-plus_sdxl_vit-h.bin -> 'models/image_encoder'
      Uses OpenCLIP-ViT-H-14.
    - ip-adapter-plus-face_sdxl_vit-h.bin -> 'models/image_encoder'
      Uses OpenCLIP-ViT-H-14.

    FaceID SDXL
    - ip-adapter-faceid-plus_sdxl.bin -> 'models/image_encoder'
      FaceID Plus requires CLIP image embeddings.
    - ip-adapter-faceid-plusv2_sdxl.bin -> 'models/image_encoder'
      FaceID PlusV2 requires CLIP image embeddings.

    Notes
    -----
    Plain FaceID weights such as ``ip-adapter-faceid_sdxl.bin`` do not require a
    CLIP image encoder and must not be passed to this function.

    If this happens, it likely indicates a bug in the calling code.

    Parameters
    ----------
    weight_names : list[str]
        Weight filenames. Entries may be bare filenames or paths; only the final
        filename component is used for matching.

    Returns
    -------
    str
        Encoder subfolder to pass to ``get_ip_image_encoder(...)``.

    Raises
    ------
    ValueError
        If ``weight_names`` is empty, if an unknown weight filename is found, or
        if the provided weights require different image encoders.

    Notes
    -----
    All weights loaded together in the same node must share the same CLIP image
    encoder, because Diffusers supports only one ``pipe.image_encoder`` at a time.
    """
    if not weight_names:
        raise ValueError(
            'infer_image_encoder_subfolder requires at least one weight name.'
        )

    encoder_by_weight = {
        # Standard SDXL IP-Adapter
        'ip-adapter_sdxl.bin': 'sdxl_models/image_encoder',
        'ip-adapter_sdxl_vit-h.bin': 'models/image_encoder',
        'ip-adapter-plus_sdxl_vit-h.bin': 'models/image_encoder',
        'ip-adapter-plus-face_sdxl_vit-h.bin': 'models/image_encoder',

        # FaceID SDXL variants requiring CLIP
        'ip-adapter-faceid-plus_sdxl.bin': 'models/image_encoder',
        'ip-adapter-faceid-plusv2_sdxl.bin': 'models/image_encoder',
    }

    resolved_subfolders: set[str] = set()
    unknown_weights: list[str] = []

    for raw_name in weight_names:
        filename = PurePosixPath(str(raw_name)).name.lower()
        subfolder = encoder_by_weight.get(filename)

        if subfolder is None:
            unknown_weights.append(str(raw_name))
            continue

        resolved_subfolders.add(subfolder)

    if unknown_weights:
        known = ', '.join(sorted(encoder_by_weight))
        raise ValueError(
            'Unknown IP-Adapter weight name(s): '
            f'{unknown_weights}. '
            'Please extend infer_image_encoder_subfolder(...) explicitly for '
            f'these weights. Known weights: {known}'
        )

    if len(resolved_subfolders) != 1:
        raise ValueError(
            'You are mixing IP-Adapter weights that require different CLIP '
            f'image encoders: {sorted(resolved_subfolders)}. '
            'Diffusers only supports one `pipe.image_encoder` at a time.'
        )

    return next(iter(resolved_subfolders))
