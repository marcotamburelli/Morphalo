from typing import Tuple

import cv2
from diffusers import LTXConditionPipeline


def round_to_vae(height: int, width: int, pipe: LTXConditionPipeline) -> Tuple[int, int]:
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)

    return height, width


def read_video_info(cap: cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, w, h, n
