
from typing import Optional

import torch
from diffusers.utils import export_to_video, load_image, load_video

from stability.nodes.ltx.video_utils import read_video_tensor
from stability.nodes.ltx.wiring.ic_lora import ICLoRaBundle
from third_party.lightricks import LTXConditionPipeline, LTXVideoCondition


def apply_ic_lora(ic_lora_bundle: ICLoRaBundle, pipe: LTXConditionPipeline, device: str) -> Optional[torch.Tensor]:
    if ic_lora_bundle.has_has_ic_lora:
        pipe.load_lora_weights(
            ic_lora_bundle.model_id,
            weight_name=ic_lora_bundle.weight_name,
            adapter_name=ic_lora_bundle.adapter_name
        )

        pipe.set_adapters(
            [ic_lora_bundle.adapter_name],
            [ic_lora_bundle.adapter_weight]
        )

        control_frames = load_video(ic_lora_bundle.video_path)
        return read_video_tensor(control_frames, device=device)
    else:
        pipe.unload_lora_weights()
        return None


def build_image_condition(*, image_path: str, frame_index: int) -> LTXVideoCondition:
    """
    Build an LTX one-frame video condition from a single image.

    The current LTX conditioning API expects a video tensor/sequence. A practical
    workaround is to serialize a one-frame video and reload it as a video object.
    This mirrors your existing approach in the legacy Img2Video node.

    Parameters
    ----------
    image_path : str
        Path to the input keyframe image.
    frame_index : int
        Index of the frame in the condition sequence to anchor the conditioning.

    Returns
    -------
    LTXVideoCondition
        Condition object compatible with ``LTXConditionPipeline``.
    """
    image = load_image(image_path)
    video_1frame = load_video(export_to_video([image]))

    return LTXVideoCondition(video=video_1frame, frame_index=int(frame_index))
