import re
from typing import List


def infer_image_encoder_subfolder(weight_names: List[str],) -> str:
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
