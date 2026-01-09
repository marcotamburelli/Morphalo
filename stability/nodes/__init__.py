from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import ControlNetModel
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from stability.cache.models import get_controlnet, get_ip_image_encoder
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


def norm_prompt(value: Any, *, joiner: str = '\n') -> str:
    """
    Normalize a prompt that can be:
      - str
      - list[str]
    """

    if value is None:
        return ''

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        # filter Nones / non-str defensively
        parts = [str(x).strip()
                 for x in value if x is not None and str(x).strip()]
        return joiner.join(parts).strip()

    # be strict: better to fail early than silently stringify weird objects
    raise TypeError(
        f'Prompt must be str or list[str], got {type(value).__name__}')


def norm_prompt_pair(value: Any, *, joiner: str = "\n") -> Tuple[str, str]:
    """
    Normalize a prompt that can be:
      - str
      - list[str]
      - dict with keys: content/style (each str or list[str])

    Returns (content, style).
    """

    if value is None:
        return '', ''

    # legacy / simple form: treat as content
    if isinstance(value, (str, list)):
        return norm_prompt(value, joiner=joiner), ''

    # structured form
    if isinstance(value, dict):
        content = norm_prompt(value.get('content'), joiner=joiner)
        style = norm_prompt(value.get('style'), joiner=joiner)
        return content, style

    raise TypeError(
        f'Prompt must be str, list[str], or dict, got {type(value).__name__}')


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


IpAdapterScale = Union[
    float,
    Dict[str, Dict[str, List[float]]],
]


@dataclass
class IpAdapterSpec:
    key: str
    model_id: str
    weight_name: Optional[str] = None
    subfolder: Optional[str] = None
    scale: IpAdapterScale = 1.0
    encoder_key: Optional[str] = None
    encoder_subfolder: Optional[str] = None


def infer_encoder_for_ip_adapter(weight_name: Optional[str]) -> tuple[str, str]:
    wn = (weight_name or '').lower()

    # Euristic: if see 'vit-h' -> vit_h
    if 'vit-h' in wn or 'vit_h' in wn or 'vitl' in wn:
        return 'h94/IP-Adapter', 'models/image_encoder'

    return 'h94/IP-Adapter', 'sdxl_models/image_encoder'


class IpAdapterRegistry:
    """
    Registry for declaring IP-Adapter attachments on a DAG node.

    This class acts as a lightweight declaration layer that allows a
    :class:`~stability.dag.NodeRef` to expose one or more IP-Adapter inputs.
    Each declared adapter is represented internally by an
    :class:`~stability.ip_adapter.IpAdapterSpec` and externally by an
    :class:`~stability.dag.AttachmentSink` that can be wired from upstream
    nodes.

    The registry itself does not load models or execute pipelines. Instead,
    it records the configuration required to later:
    - load IP-Adapter weights into a diffusion pipeline,
    - configure per-adapter or per-block influence via scale settings,
    - associate incoming images with the correct adapter slot at execution time.

    This separation allows nodes to remain declarative, while the runner or
    pipeline builder is responsible for realizing the actual model
    configuration.
    """

    def __init__(self, owner: NodeRef):
        """
        Initialize an IP-Adapter registry for a node.

        Parameters
        ----------
        owner : NodeRef
            The node that owns this registry. All attachment sinks created
            by this registry will target this node.
        """

        self._owner = owner
        self._counter = 0
        self._specs: List[IpAdapterSpec] = []

    def add(
        self,
        model_id: str,
        *,
        scale: IpAdapterScale = 1.0,
        key: Optional[str] = None,
        weight_name: Optional[str] = None,
        subfolder: Optional[str] = None,
        encoder_key: Optional[str] = None,
        encoder_subfolder: Optional[str] = None
    ) -> AttachmentSink:
        """
        Declare a new IP-Adapter input and return an attachment sink.

        This method registers an IP-Adapter configuration and exposes a
        corresponding attachment point that can be connected from upstream
        nodes. The returned sink represents a single IP-Adapter slot, which
        will later receive one or more reference images during DAG execution.

        Parameters
        ----------
        model_id : str
            Hugging Face repository identifier for the IP-Adapter model
            (e.g. ``"h94/IP-Adapter"``).

        scale : IpAdapterScale, optional
            Influence of the IP-Adapter on the diffusion process.
            This may be:
            - a single float for global weighting,
            - a dictionary specifying per-block configuration
              (e.g. ``{"up": {"block_0": [...]}}``),
            Default is ``1.0``.

        key : str, optional
            Logical identifier for this adapter. If not provided, a unique
            key is generated automatically (``"ip1"``, ``"ip2"``, ...).
            The key is used to name the attachment input
            (``"ip_adapter:{key}"``).

        weight_name : str, optional
            Name of the IP-Adapter weight file within the repository.
            If not specified, the default weight defined by the repository
            is used.

        subfolder : str, optional
            Subfolder within the repository containing the IP-Adapter
            weights (e.g. ``"sdxl_models"``).

        encoder_key : str, optional
            Identifier of the image encoder to be used with this IP-Adapter.
            If not provided, an appropriate encoder may be inferred from
            ``weight_name`` (for example, ViT-H encoders for "plus" adapters).

        encoder_subfolder : str, optional
            Subfolder within the encoder repository containing the image
            encoder weights. If not provided, it may be inferred together
            with ``encoder_key``.

        Returns
        -------
        AttachmentSink
            An attachment sink targeting the owning node. Upstream nodes
            can connect to this sink to provide reference images for the
            declared IP-Adapter.

        Notes
        -----
        This method is purely declarative. It does not load any models or
        modify pipelines directly. The collected specifications are intended
        to be consumed later by a runner or pipeline builder that performs
        the actual ``load_ip_adapter`` and ``set_ip_adapter_scale`` calls.
        """
        if key is None:
            self._counter += 1
            key = f'ip{self._counter}'

        if encoder_key is None or encoder_subfolder is None:
            k, sub = infer_encoder_for_ip_adapter(
                weight_name=weight_name
            )
            encoder_key = encoder_key or k
            encoder_subfolder = encoder_subfolder or sub

        self._specs.append(IpAdapterSpec(
            key=key,
            model_id=model_id,
            weight_name=weight_name,
            subfolder=subfolder,
            scale=float(scale),
            encoder_key=encoder_key,
            encoder_subfolder=encoder_subfolder,
        ))

        return AttachmentSink(
            id=f'ip_adapter:{key}',
            target=self._owner,
            input_id=f'ip_adapter:{key}',
        )

    @property
    def specs(self) -> List[IpAdapterSpec]:
        return self._specs


class IpAdapterBundle:
    def __init__(
        self,
        adapters: Optional[List[IpAdapterSpec]],
        *,
        dtype,
        device,
        input: Optional[Dict[str, Dict]]
    ):
        self._images: List[str] = []
        self._scales: List[float] = []
        self._specs: List[IpAdapterSpec] = []
        self._model_id: Optional[str] = None
        self._subfolder: Optional[str] = None
        self._encoder_key: Optional[str] = None
        self._weight_names: List[str] = []
        self._image_encoder: Optional[CLIPVisionModelWithProjection] = None
        if adapters:
            self._has = True
            self._build(
                adapters,
                dtype=dtype,
                device=device,
                input=input or {}
            )
        else:
            self._has = False

    def _build(self, adapters: List[IpAdapterSpec], *, dtype, device, input: Dict[str, Dict]) -> None:
        self._validate(adapters=adapters)

        for ad in adapters:
            in_id = f'ip_adapter:{ad.key}'
            upstream = input.get(in_id)
            if upstream is None:
                raise ValueError(
                    f'Missing IP-Adapter input for {in_id!r}. Did you wire an image into it?')

            img_path = upstream.get('image') or upstream.get('path')
            if not img_path:
                raise ValueError(
                    f'Upstream output for {in_id!r} does not contain an image path')

            self._weight_names.append(ad.weight_name)
            self._images.append(img_path)
            self._scales.append(float(ad.scale))
            self._specs.append(ad)

            self._image_encoder = get_ip_image_encoder(
                repo_id=self._encoder_key,
                subfolder=self._encoder_subfolder,
                device=device,
                dtype=dtype
            )

    def _validate(self, adapters: List[IpAdapterSpec]):
        # Checking unique image encoder
        encoder_keys = {ad.encoder_key for ad in adapters}
        if None in encoder_keys:
            raise ValueError(
                'IpAdapterSpec.encoder_key must be resolved (None found).'
            )

        if len(encoder_keys) != 1:
            # errore leggibile
            details = ', '.join(
                f'{ad.key}:{ad.encoder_key}' for ad in adapters)
            raise ValueError(
                'Incompatible IP-Adapters in the same run: they require different image encoders. '
                f'Got: {details}. '
                'Split into multiple nodes/runs (e.g., first bigG then vit-h) or use adapters that share the same encoder.'
            )

        self._encoder_key = next(iter(encoder_keys))

        encoder_subfolders = {ad.encoder_subfolder for ad in adapters}
        if None in encoder_keys:
            raise ValueError(
                'IpAdapterSpec.encoder_key must be resolved (None found).'
            )

        if len(encoder_subfolders) != 1:
            # errore leggibile
            details = ', '.join(
                f'{ad.key}:{ad.encoder_key}' for ad in adapters)
            raise ValueError(
                'Incompatible IP-Adapters in the same run: they require different image encoders. '
                f'Got: {details}. '
                'Split into multiple nodes/runs (e.g., first bigG then vit-h) or use adapters that share the same encoder.'
            )

        self._encoder_subfolder = next(iter(encoder_subfolders))

        # Checking unique IP Adaper id and subfolder
        model_ids = {ad.model_id for ad in adapters}
        if len(model_ids) != 1:
            raise ValueError(
                f'All IP-Adapters in one node must share the same model_id. Got: {sorted(model_ids)}')
        self._model_id = next(iter(model_ids))

        subfolders = {ad.subfolder for ad in adapters}
        if len(subfolders) != 1:
            raise ValueError(
                'All IP-Adapters in one node must share the same subfolder to be loaded in one call. '
                f'Got: {sorted(subfolders)}'
            )
        self._subfolder = next(iter(subfolders))

    @property
    def has_ip_adapter(self) -> bool:
        return self._has

    @property
    def ip_adapter_image(self) -> List[Any]:
        ip_adapter_image = [
            Image.open(img).convert('RGB') for img in self._images
        ]
        return ip_adapter_image if len(ip_adapter_image) > 1 else ip_adapter_image[0]

    @property
    def scale_arg(self) -> List[float]:
        return self._scales if len(self._scales) > 1 else self._scales[0]

    @property
    def weight_names_arg(self) -> List[str]:
        return self._weight_names if len(self._weight_names) > 1 else self._weight_names[0]

    @property
    def model_id_arg(self) -> str:
        return self._model_id

    @property
    def subfolder_arg(self) -> Optional[str]:
        return self._subfolder

    @property
    def image_encoder(self) -> Optional[CLIPVisionModelWithProjection]:
        return self._image_encoder
