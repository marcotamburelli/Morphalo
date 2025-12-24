from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import ControlNetModel
from PIL import Image

from stability.cache.models import get_controlnet
from stability.core.spec_loader import load_hocon_spec
from stability.dag import AttachmentSink, NodeRef


def resolve_spec(spec: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """
    Resolve a node specification into a plain dictionary.

    If `spec` is a dict, it is returned as-is.
    If `spec` is a string or Path, it is interpreted as a HOCON file path
    and loaded accordingly.
    """
    if isinstance(spec, dict):
        return spec

    if isinstance(spec, (str, Path)):
        return load_hocon_spec(spec)

    raise TypeError(f'Unsupported spec type: {type(spec)}')


def resolve_dtype(dtype: str) -> torch.dtype:
    d = (dtype or 'bf16').lower()
    if d in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if d in ('fp16', 'float16'):
        return torch.float16
    if d in ('fp32', 'float32'):
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype}')


def resolve_seed(seed: Any) -> int:
    if seed is None or seed == 'random':
        return secrets.randbelow(2**31)
    return int(seed)


def norm_prompt(prompt: dict, joiner: str = '\n') -> str:
    # prompt = spec.get('prompt')
    if prompt is None:
        return ''

    if isinstance(prompt, str):
        return prompt.strip()

    if isinstance(prompt, list):
        return joiner.join(prompt).strip()


def load_init_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')

    p = Path(str(image)).expanduser()

    if not p.exists():
        raise FileNotFoundError(f'Input image not found: {p}')

    return Image.open(p).convert('RGB')


def make_node_output_path(
    *,
    out_dir: Path,
    node_id: str,
    ext: str = 'png',
    seed: Optional[int] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Generate an output file path for a node execution.

    The path is created under ``out_dir / node_id`` and uses a timestamp-based
    filename. Optional components such as the random seed can be included
    to make the filename more informative while keeping semantics out of
    directory names.

    Parameters
    ----------
    out_dir : Path
        Base output directory of the DAG execution.
    node_id : str
        Identifier of the node producing the output.
    ext : str, optional
        File extension (without leading dot). Default is ``'png'``.
    seed : int, optional
        Optional random seed to include in the filename.
    tag : str, optional
        Optional tag to override the default timestamp-based tag.

    Returns
    -------
    Path
        Full filesystem path where the output file should be written.

    Notes
    -----
    - The directory ``out_dir / node_id`` is created if it does not exist.
    - The default filename format is ``YYYY-MM-DD_HHMMSS[_seedN].<ext>``.
    """
    node_dir = Path(out_dir) / node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = time.strftime('%Y-%m-%d_%H%M%S')

    parts = [tag]
    if seed is not None:
        parts.append(f'seed{seed}')

    filename = '_'.join(parts) + f'.{ext.lstrip(".")}'
    return node_dir / filename


@dataclass
class ControlNetSpec:
    key: str
    model_id: str
    conditioning_scale: float


class ControlNetRegistry:
    """
    Registry for ControlNet declarations associated with a DAG node.

    This class is responsible for *declaring* ControlNet inputs at DAG
    construction time. It collects ControlNet specifications and exposes
    an API to create ``AttachmentSink`` objects that can be wired using
    the ``>>`` operator.

    The registry does not load models or process images. Its sole purpose
    is to describe which ControlNets are required by a node and how they
    are connected in the DAG. The actual resolution of images, models,
    and conditioning scales is performed later at execution time by
    ``ControlNetBundle``.

    A ``ControlNetRegistry`` instance is typically owned by a node such
    as ``Txt2Img`` or ``Img2Img`` and is accessed via an attribute
    (e.g. ``node.controlnet``).

    Attributes
    ----------
    specs : list of ControlNetSpec
        Ordered list of ControlNet specifications declared for the node.
        The order is significant and determines the correspondence between
        control images, models, and conditioning scales during execution.
    """

    def __init__(self, owner: NodeRef):
        self._owner = owner
        self._counter = 0
        self._specs: List[ControlNetSpec] = []

    def add(
        self,
        model_id: str,
        *,
        conditioning_scale: float = 1.0,
        key: Optional[str] = None,
    ) -> AttachmentSink:
        """
        Declare a ControlNet input and return an attachment sink.

        This method registers a new ControlNet specification and returns
        an ``AttachmentSink`` that can be wired to an upstream node
        producing a control image.

        The returned sink must be connected using the ``>>`` operator.
        The upstream node output (typically containing an image path)
        will be provided to the owning node under an input identifier
        of the form ``controlnet:<key>``.

        Parameters
        ----------
        model_id : str
            Hugging Face repository identifier or local path for the
            ControlNet weights.
        conditioning_scale : float, optional
            Strength of the ControlNet conditioning. Higher values enforce
            the control signal more strongly. Default is 1.0.
        key : str, optional
            Stable identifier for this ControlNet input. If not provided,
            an incremental key is automatically generated (e.g. ``cn1``,
            ``cn2``).

        Returns
        -------
        AttachmentSink
            A sink object that can be wired with
            ``upstream_node >> sink``.

        Notes
        -----
        - The order in which ``add`` is called determines the order of
          ControlNets during execution.
        - This method only records the declaration of the ControlNet.
          Model loading and image resolution are deferred to runtime.
        """
        if key is None:
            self._counter += 1
            key = f'cn{self._counter}'

        self._specs.append(ControlNetSpec(
            key=key,
            model_id=model_id,
            conditioning_scale=float(conditioning_scale),
        ))

        return AttachmentSink(
            id=f'controlnet:{key}',
            target=self._owner,
            input_id=f'controlnet:{key}',
        )

    @property
    def specs(self) -> List[ControlNetSpec]:
        return self._specs


class ControlNetBundle:
    def __init__(
        self,
            controlnets: List[ControlNetSpec],
            *,
            dtype,
            device,
            input: Optional[Dict[str, Dict]]
    ):
        self._cn_images: List[str] = []
        self._cn_scales: List[float] = []
        self._cn_models: List[ControlNetModel] = []

        if controlnets:
            self._has_controlnet = True
            self._build(
                controlnets,
                dtype=dtype,
                device=device,
                input=input or {}
            )
        else:
            self._has_controlnet = False

    def _build(self, controlnets: List[ControlNetSpec], *, dtype, device,  input: Dict[str, Dict]):

        for cn in controlnets:
            in_id = f"controlnet:{cn.key}"
            upstream = input.get(in_id)
            if upstream is None:
                raise ValueError(
                    f"Missing ControlNet input for '{in_id}'. Did you wire an image into it?")
            # we assume upstream output contains either:
            # - {'image': '/path/to/img.png'}  (your stable nodes will do this)
            # - or directly {'path': ...}
            img_path = upstream.get('image') or upstream.get('path')
            if not img_path:
                raise ValueError(
                    f"Upstream output for '{in_id}' does not contain an image path")

            # diffusers can accept PIL.Image too; here pass path and load later if needed
            self._cn_images.append(img_path)
            self._cn_scales.append(float(cn.conditioning_scale))

            self._cn_models.append(get_controlnet(
                model_id=cn.model_id,
                device=device,
                dtype=dtype
            ))

    @property
    def has_controlnet(self) -> bool:
        return self._has_controlnet

    @property
    def controlnet_arg(self) -> Union[ControlNetModel, List[ControlNetModel]]:
        return self._cn_models if len(self._cn_models) > 1 else self._cn_models[0]

    @property
    def control_image_arg(self):
        control_images = [Image.open(img) for img in self._cn_images]
        return control_images if len(control_images) > 1 else control_images[0]

    @property
    def conditioning_scale_arg(self):
        return self._cn_scales if len(self._cn_scales) > 1 else self._cn_scales[0]
