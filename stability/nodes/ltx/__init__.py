from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import PIL.Image
import torch
from diffusers import LTXConditionPipeline
from torchvision import transforms

from stability.dag import AttachmentSink, NodeRef


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


@dataclass
class ICLoRaSpec:
    key: str
    model_id: str
    weight_name: str
    adapter_name: str
    adapter_weight: float


class ICLoRaRegistry:
    """
    Registry for a single Image-Conditioned LoRA (IC-LoRA) attachment.

    This class provides a registry-like interface to attach an IC-LoRA
    configuration to a node in the DAG. Although the interface resembles
    a generic registry, only **one** IC-LoRA specification is supported
    at a time, by design.

    The registry is responsible for:
    - capturing the IC-LoRA specification (model, weights, adapter name)
    - exposing an AttachmentSink to wire a conditioning video into the node
    - producing an ICLoRaSpec that can later be resolved at execution time

    The registry pattern is intentionally preserved for API consistency
    with other adapter registries (e.g. ControlNet, IP-Adapter), even
    though multiple IC-LoRA attachments are currently not supported.

    Notes
    -----
    - Only one IC-LoRA can be registered at a time. Registering a new
      IC-LoRA overwrites the previous specification.
    - This limitation is intentional: multiple IC-LoRA adapters applied
      simultaneously are not meaningful in the current LTX pipeline.
    - The registry abstraction is retained to maintain a uniform
      attachment API across different adapter types.

    See Also
    --------
    ICLoRaSpec
        Dataclass describing the IC-LoRA configuration.
    AttachmentSink
        Sink used to attach upstream video nodes as IC-LoRA inputs.
    """

    def __init__(self, owner: NodeRef):
        """
        Create a new IC-LoRA registry bound to a DAG node.

        Parameters
        ----------
        owner : NodeRef
            The node that owns this registry. Any IC-LoRA attachment
            created through this registry will target this node.
        """

        self._owner = owner
        self._counter = 0
        self._spec: ICLoRaSpec = None

    def __call__(
        self,
        model_id: str,
        weight_name: str,
        adapter_name: str,
        adapter_weight: float,
        key: Optional[str] = None
    ):
        """
        Register an Image-Conditioned LoRA (IC-LoRA) and create an attachment sink.

        This method registers a single IC-LoRA specification for the owning node
        and returns an :class:`AttachmentSink` that can be used to wire an upstream
        video as structural conditioning input.

        Calling this method defines:
        - which IC-LoRA model and weight file to load
        - the adapter name and weight to apply at runtime
        - the input channel (via an attachment sink) that will receive the
        conditioning video

        Only one IC-LoRA can be registered at a time. Invoking this method multiple
        times overwrites the previously registered IC-LoRA specification.

        Parameters
        ----------
        model_id : str
            Hugging Face model identifier or local path for the IC-LoRA model.
        weight_name : str
            Name of the LoRA weight file to load from the model repository.
        adapter_name : str
            Name of the adapter as registered inside the diffusion pipeline.
        adapter_weight : float
            Scalar weight applied to the IC-LoRA adapter during inference.
        key : str, optional
            Optional identifier used to name the attachment sink. If not provided,
            a unique key is generated automatically.

        Returns
        -------
        AttachmentSink
            A sink targeting the owning node. Wiring an upstream video node
            into this sink provides the conditioning video used by the IC-LoRA
            during execution.

        Notes
        -----
        - This method does not perform any model loading or validation.
        The IC-LoRA specification is resolved lazily at execution time.
        - Registering a new IC-LoRA replaces any previously registered one.
        - The returned AttachmentSink must be connected to a node producing
        a video output (e.g. a canny or depth video).
        """
        if key is None:
            self._counter += 1
            key = f'icl{self._counter}'

        self._spec = ICLoRaSpec(
            key=key,
            model_id=model_id,
            weight_name=weight_name,
            adapter_name=adapter_name,
            adapter_weight=adapter_weight
        )

        return AttachmentSink(
            id=f'ic_lora:{key}',
            target=self._owner,
            input_id=f'ic_lora:{key}',
        )

    @property
    def spec(self) -> ICLoRaSpec:
        return self._spec


class ICLoRaBundle:
    def __init__(
        self,
        ic_lora: Optional[ICLoRaSpec],
        *,
        input: Optional[Dict[str, Dict]]
    ):
        self._icl_model_id: Optional[str] = None
        self._icl_weight_name: Optional[str] = None
        self._icl_video: Optional[str] = None
        self._icl_name: Optional[str] = None
        self._icl_weight: Optional[float] = None

        if ic_lora is not None:
            self._has_ic_lora = True
            self._build(
                ic_lora,
                input=input or {}
            )
        else:
            self._has_ic_lora = False

    def _build(self, ic_lora: ICLoRaSpec, *, input: Dict[str, Dict]):
        in_id = f'ic_lora:{ic_lora.key}'
        upstream = input.get(in_id)
        if upstream is None:
            raise ValueError(
                f"Missing IC-LoRa input for '{in_id}'. Did you wire a video into it?")
        # we assume upstream output contains either:
        # - {'video': '/path/to/video.mp4'}  (your stable nodes will do this)
        # - or directly {'path': ...}
        img_path = upstream.get('video') or upstream.get('path')
        if not img_path:
            raise ValueError(
                f"Upstream output for '{in_id}' does not contain a video path")

        self._icl_model_id = ic_lora.model_id
        self._icl_weight_name = ic_lora.weight_name
        self._icl_video = img_path
        self._icl_name = ic_lora.adapter_name
        self._icl_weight = float(ic_lora.adapter_weight)

    @property
    def has_has_ic_lora(self) -> bool:
        return self._has_ic_lora

    @property
    def model_id(self) -> Optional[str]:
        return self._icl_model_id

    @property
    def weight_name(self) -> Optional[str]:
        return self._icl_weight_name

    @property
    def video_path(self) -> Optional[str]:
        return self._icl_video

    @property
    def adapter_name(self) -> Optional[str]:
        return self._icl_name

    @property
    def adapter_weight(self) -> Optional[float]:
        return self._icl_weight
