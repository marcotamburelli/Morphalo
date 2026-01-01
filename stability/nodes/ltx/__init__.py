from typing import Tuple
from diffusers import LTXConditionPipeline


def round_to_vae(height: int, width: int, pipe: LTXConditionPipeline) -> Tuple[int, int]:
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)

    return height, width
