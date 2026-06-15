from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import cv2
import PIL.Image
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from third_party.lightricks import LTXConditionPipeline


def read_video_info(cap: cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, w, h, n


def round_to_vae(height: int, width: int, pipe: LTXConditionPipeline) -> Tuple[int, int]:
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)

    return height, width


def downscale_size(height: int, width: int, factor: float, pipe: LTXConditionPipeline):
    down_h = int(height * factor)
    down_w = int(width * factor)

    return round_to_vae(down_h, down_w, pipe)


def upscale_size(height: int, width: int, factor):
    return int(height * factor), int(width * factor)


def read_video_tensor(video: List[PIL.Image.Image], device: str) -> torch.Tensor:
    """
    Reads a video file and converts it into a torch.Tensor with the shape [F, C, H, W].
    """

    to_tensor_transform = transforms.ToTensor()

    video_tensor = torch.stack([
        to_tensor_transform(img.convert("RGB"))
        for img in video
    ]).to(device)

    return video_tensor
