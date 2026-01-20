from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import ControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models import MultiAdapter
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from stability.cache.models import *
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


def norm_prompt_pair(value: Any, *, joiner: str = '\n') -> Tuple[str, str]:
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

    filename = '_'.join(parts) + f'.{ext.lstrip('.')}'
    return node_dir / filename


def cuda_prerun(device: str) -> None:
    """
    Prepare CUDA state for timing/memory measurements.

    Resets peak memory stats and synchronizes the device so subsequent timing
    reflects the work of the current run only.
    """
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_sync(device: str) -> None:
    """Synchronize CUDA device if running on CUDA."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def cuda_mem_stats(device: str) -> dict:
    """
    Return CUDA memory statistics in GB, or an empty dict if not on CUDA.
    """
    if torch.device(device).type != 'cuda':
        return {}

    return {
        'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 3),
        'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 3),
        'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def ensure_out_dir(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def save_image(out_dir: Path, *, node_id: str, seed: int, img: Image.Image) -> Path:
    img_path = make_node_output_path(
        out_dir=out_dir,
        node_id=node_id,
        seed=seed
    )

    img.save(img_path)
    return img_path


def write_json_sidecar(out_path: Path, payload: dict) -> Path:
    meta_path = out_path.with_suffix('.json')
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    return meta_path


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
            in_id = f'controlnet:{cn.key}'
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
    def control_image_arg(self) -> Union[Image.Image, List[Image.Image]]:
        control_images = [Image.open(img) for img in self._cn_images]
        return control_images if len(control_images) > 1 else control_images[0]

    @property
    def conditioning_scale_arg(self) -> Union[float, List[float]]:
        return self._cn_scales if len(self._cn_scales) > 1 else self._cn_scales[0]


# -----------------------------------------------------------------------------
# T2I-Adapter (SDXL Adapter) declarations
# -----------------------------------------------------------------------------

@dataclass
class T2IAdapterSpec:
    """Declarative spec for a single T2I-Adapter attachment."""
    key: str
    model_id: str
    conditioning_scale: float = 1.0


class T2IAdapterRegistry:
    """Registry for declaring T2I-Adapter inputs on a DAG node.

    The registry is purely declarative: it records which adapter(s) a node
    expects and returns :class:`~stability.dag.AttachmentSink` objects that can
    be wired from upstream nodes producing the adapter conditioning image
    (e.g. canny/lineart/sketch/depth maps).

    The actual adapter model loading and conditioning image resolution happen
    at runtime via :class:`~stability.nodes.T2IAdapterBundle`.
    """

    def __init__(self, owner: NodeRef):
        self._owner = owner
        self._counter = 0
        self._specs: List[T2IAdapterSpec] = []

    def add(
        self,
        model_id: str,
        *,
        conditioning_scale: float = 1.0,
        key: Optional[str] = None,
    ) -> AttachmentSink:
        """Declare a new T2I-Adapter input and return an attachment sink."""
        if key is None:
            self._counter += 1
            key = f't2i{self._counter}'

        self._specs.append(T2IAdapterSpec(
            key=key,
            model_id=model_id,
            conditioning_scale=float(conditioning_scale),
        ))

        return AttachmentSink(
            id=f't2i-adapter:{key}',
            target=self._owner,
            input_id=f't2i-adapter:{key}',
        )

    @property
    def specs(self) -> List[T2IAdapterSpec]:
        return self._specs


class T2IAdapterBundle:
    """Runtime bundle resolving declared T2I-Adapters into pipeline args."""

    def __init__(
        self,
        adapters: List[T2IAdapterSpec],
        *,
        dtype,
        device,
        input: Optional[Dict[str, Dict]]
    ):
        self._has = bool(adapters)
        self._images: List[str] = []
        self._scales: List[float] = []
        self._models: List[T2IAdapter] = []
        self._specs: List[T2IAdapterSpec] = []

        if self._has:
            self._build(adapters, dtype=dtype,
                        device=device, input=input or {})

    def _build(self, adapters: List[T2IAdapterSpec], *, dtype, device, input: Dict[str, Dict]) -> None:
        for ad in adapters:
            in_id = f't2i-adapter:{ad.key}'
            upstream = input.get(in_id)
            if upstream is None:
                raise ValueError(
                    f'Missing T2I-Adapter input for {in_id!r}. Did you wire an image into it?'
                )

            img_path = upstream.get('image') or upstream.get('path')
            if not img_path:
                raise ValueError(
                    f'Upstream output for {in_id!r} does not contain an image path'
                )

            self._images.append(img_path)
            self._scales.append(float(ad.conditioning_scale))
            self._models.append(get_t2i_adapter(
                model_id=ad.model_id,
                device=device,
                dtype=dtype
            ))
            self._specs.append(ad)

    @property
    def has_t2i_adapter(self) -> bool:
        return self._has

    @property
    def adapter_model_arg(self) -> Union[T2IAdapter, MultiAdapter]:
        return MultiAdapter(self._models) if len(self._models) > 1 else self._models[0]

    @property
    def adapter_image_arg(self) -> Union[Image.Image, List[Image.Image]]:
        imgs = [Image.open(p).convert('RGB') for p in self._images]
        return imgs if len(imgs) > 1 else imgs[0]

    @property
    def conditioning_scale_arg(self) -> Union[float, List[float]]:
        return self._scales if len(self._scales) > 1 else self._scales[0]

    @property
    def specs(self) -> List[T2IAdapterSpec]:
        return self._specs


IpAdapterScale = Union[
    float,
    List[float],
    Dict[str, Dict[str, List[float]]]
]


@dataclass
class IpAdapterSpec:
    key: str
    model_id: str
    weight_name: str
    subfolder: str
    encoder_key: Optional[str]
    encoder_subfolder: Optional[str]
    scale: IpAdapterScale = 1.0
    has_mask: bool = False


def infer_encoder_for_ip_adapter(weight_name: Optional[str]) -> tuple[str, str]:
    wn = (weight_name or '').lower()

    # Euristic: if see 'vit-h' -> vit_h
    if 'vit-h' in wn or 'vit_h' in wn or 'vitl' in wn:
        return 'h94/IP-Adapter', 'models/image_encoder'

    return 'h94/IP-Adapter', 'sdxl_models/image_encoder'


@dataclass
class IpAdapterAttachmentSink(AttachmentSink):
    """
    Attachment sink representing a single IP-Adapter slot.

    This class is a specialized :class:`~stability.dag.AttachmentSink` returned
    by :meth:`IpAdapterRegistry.add`. It represents the *reference image input*
    for a specific IP-Adapter slot declared on a node.

    Instances of this class are intended to be used in DAG wiring expressions,
    for example::

        ip = node.ip_adapter.add(...)
        image_node >> ip

    In addition to acting as an image attachment point, this sink also provides
    helper methods to declare and wire auxiliary inputs that are *logically
    bound to the same adapter slot*, such as per-image masks.

    Each instance holds a reference to the corresponding
    :class:`~stability.ip_adapter.IpAdapterSpec`, allowing it to update the
    declarative specification (for example, marking that the adapter uses
    masks) when additional attachment points are created.
    """

    key: str
    spec: IpAdapterSpec

    def mask_for(self, idx: int) -> AttachmentSink:
        """
        Declare an indexed mask input for this IP-Adapter slot.

        This method marks the underlying :class:`~stability.ip_adapter.IpAdapterSpec`
        as using masks and returns an :class:`~stability.dag.AttachmentSink`
        representing the mask input ``'ip_adapter_mask:{key}[{idx}]'``.

        The index ``idx`` identifies which mask in a mask list is being attached.
        In typical usage, an upstream source node (e.g. :class:`FileImage`) provides
        a list of mask images and ``idx`` refers to the position within that list::

            masks = FileImage(id='masks', path=[mask0, mask1])
            ip = node.ip_adapter.add(..., key='face')

            masks >> ip.mask_for(0)   # attaches mask0
            masks >> ip.mask_for(1)   # attaches mask1

        The returned sink is a declarative attachment endpoint only; it does not
        read files or preprocess masks. Mask loading and preprocessing are handled
        later by the execution layer (e.g., by building an ``IpAdapterBundle`` and
        passing ``ip_adapter_masks`` to the diffusion pipeline).

        Notes
        -----
        Calling this method has a declarative side effect: it sets
        ``spec.has_mask = True`` on the associated :class:`IpAdapterSpec`.
        """

        self.spec.has_mask = True

        return AttachmentSink(
            id=f'ip_adapter_mask:{self.key}[{idx}]',
            target=self.target,
            input_id=f'ip_adapter_mask:{self.key}[{idx}]',
        )

    def mask(self) -> AttachmentSink:
        self.spec.has_mask = True

        return AttachmentSink(
            id=f'ip_adapter_mask:{self.key}',
            target=self.target,
            input_id=f'ip_adapter_mask:{self.key}',
        )


class IpAdapterRegistry:
    """
    Declarative registry for IP-Adapter attachments on a DAG node.

    This class provides a lightweight declaration layer that allows a
    :class:`~stability.dag.NodeRef` to expose one or more IP-Adapter *slots*
    as attachable inputs in the DAG.

    Each declared adapter slot is:
    - represented internally by an :class:`~stability.ip_adapter.IpAdapterSpec`,
      which stores all configuration required to use the adapter at runtime;
    - exposed externally via an :class:`~stability.ip_adapter.IpAdapterAttachmentSink`
      returned by :meth:`add`, which acts as the image input endpoint for that slot
      and can be wired from upstream nodes.

    The registry itself is purely declarative:
    it does not load models, preprocess images, validate runtime inputs,
    or execute diffusion pipelines. Instead, it records the information
    required for a later execution phase to:
    - load IP-Adapter weights into a diffusion pipeline,
    - configure per-adapter or per-block influence via scale settings,
    - associate incoming reference images (and optional masks) with the
      correct adapter slot.

    This separation keeps DAG nodes side-effect free and easy to reason about,
    while delegating all concrete execution logic to the runner or pipeline
    builder.
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
        weight_name: str,
        subfolder: str,
        scale: IpAdapterScale = 1.0,
        key: Optional[str] = None,
        encoder_key: Optional[str] = None,
        encoder_subfolder: Optional[str] = None
    ) -> IpAdapterAttachmentSink:
        """
        Declare a new IP-Adapter slot on this node and return an attachment sink.

        This method registers an IP-Adapter specification and returns an
        :class:`~stability.ip_adapter.IpAdapterAttachmentSink`, which is a
        specialized :class:`~stability.dag.AttachmentSink` representing the
        *image input endpoint* for that adapter slot.

        The returned sink is intended for DAG wiring, for example::

            ip = node.ip_adapter.add(
                model_id='h94/IP-Adapter',
                weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors',
                subfolder='sdxl_models',
                scale=0.7,
                key='face',
            )

            face_ref_node >> ip

        Since the return type is an ``IpAdapterAttachmentSink``, it also
        exposes helpers to declare and wire additional inputs tied to the
        same adapter slot, such as per-image masks::

            m0 = ip.mask_for(0)
            m1 = ip.mask_for(1)

            mask0_node >> m0
            mask1_node >> m1

        Conceptually:
        - each call to ``add()`` declares **one IP-Adapter slot**,
        - the returned sink corresponds to the owning node input named
          ``'ip_adapter:{key}'``,
        - optional mask sinks correspond to inputs named
          ``'ip_adapter_mask:{key}[{idx}]'``.

        Parameters
        ----------
        model_id : str
            Hugging Face repository identifier for the IP-Adapter model
            (e.g. ``'h94/IP-Adapter'``).

        weight_name : str
            Name of the IP-Adapter weight file within the repository.

        subfolder : str
            Subfolder within the repository containing the IP-Adapter weights
            (e.g. ``'sdxl_models'``).

        scale : IpAdapterScale, optional
            Influence of the IP-Adapter on the diffusion process for this slot.
            It may be:
            - a single float (uniform weighting),
            - a list of floats (per-image weighting when multiple reference images
              are attached to this slot),
            - a dictionary specifying per-block configuration.
            Default is ``1.0``.

        key : str, optional
            Logical identifier for this adapter slot. If not provided, a unique
            key is generated automatically (``'ip1'``, ``'ip2'``, ...).
            The key determines the attachment input name (``'ip_adapter:{key}'``).

        encoder_key : str, optional
            Identifier of the image encoder to be used with this IP-Adapter.
            If not provided, an appropriate encoder may be inferred from
            ``weight_name``.

        encoder_subfolder : str, optional
            Subfolder within the encoder repository containing the image encoder
            weights. If not provided, it may be inferred together with
            ``encoder_key``.

        Returns
        -------
        IpAdapterAttachmentSink
            A specialized attachment sink targeting the owning node. Wire an
            upstream image-producing node into this sink to provide reference
            image(s) for the declared IP-Adapter slot. Use ``mask_for(idx)`` on
            the returned sink to declare mask inputs tied to the same slot.

        Notes
        -----
        This method is purely declarative. It does not load models, preprocess
        images, or modify diffusion pipelines directly. The collected
        specifications and wired inputs are consumed later by a runner or
        pipeline builder that performs the actual ``load_ip_adapter`` and
        ``set_ip_adapter_scale`` calls.
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

        spec = IpAdapterSpec(
            key=key,
            model_id=model_id,
            weight_name=weight_name,
            subfolder=subfolder,
            scale=scale,
            encoder_key=encoder_key,
            encoder_subfolder=encoder_subfolder,
        )
        self._specs.append(spec)

        return IpAdapterAttachmentSink(
            id=f"ip_adapter:{key}",
            target=self._owner,
            input_id=f"ip_adapter:{key}",
            key=key,
            spec=spec
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
        self._images: List[Union[str, List[str]]] = []
        self._scales: List[IpAdapterScale] = []
        self._specs: List[IpAdapterSpec] = []
        self._model_id: Optional[str] = None
        self._subfolder: Optional[str] = None
        self._encoder_key: Optional[str] = None
        self._weight_names: List[str] = []
        self._masks: List[Optional[List[str]]] = []
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

            self._validate_scale_for_slot(
                key=ad.key,
                scale=ad.scale,
                img_path=img_path
            )

            self._weight_names.append(ad.weight_name)
            self._images.append(img_path)
            self._scales.append(float(ad.scale))
            if ad.has_mask:
                n_imgs = len(img_path) if isinstance(img_path, list) else 1
                self._masks.append(self._collect_masks_for_slot(
                    key=ad.key,
                    input=input,
                    n_imgs=n_imgs
                ))
            else:
                self._masks.append(None)
            self._specs.append(ad)

        self._image_encoder = get_ip_image_encoder(
            repo_id=self._encoder_key,
            subfolder=self._encoder_subfolder,
            device=device,
            dtype=dtype
        )

    @staticmethod
    def _collect_masks_for_slot(
        *,
        key: str,
        input: Dict[str, Dict],
        n_imgs: int,
    ) -> List[str]:
        masks: List[Optional[str]] = [None] * n_imgs

        # 1) per-index overrides
        for idx in range(n_imgs):
            in_id = f'ip_adapter_mask:{key}[{idx}]'
            up = input.get(in_id)
            if up is None:
                continue
            p = up.get('image') or up.get('path')
            if not isinstance(p, str) or not p:
                raise ValueError(
                    f'Upstream output for {in_id!r} must contain a single mask path in \'image\' or \'path\'.'
                )
            masks[idx] = p

        # 2) global mask fallback
        global_id = f'ip_adapter_mask:{key}'
        global_up = input.get(global_id)

        if global_up is not None:
            gp = global_up.get('image') or global_up.get('path')

            # gp can be str or list[str]
            if isinstance(gp, str) and gp:
                for i in range(n_imgs):
                    if masks[i] is None:
                        masks[i] = gp

            elif isinstance(gp, list) and all(isinstance(x, str) and x for x in gp):
                if len(gp) == 1:
                    for i in range(n_imgs):
                        if masks[i] is None:
                            masks[i] = gp[0]
                elif len(gp) == n_imgs:
                    for i in range(n_imgs):
                        if masks[i] is None:
                            masks[i] = gp[i]
                else:
                    raise ValueError(
                        f'Global mask input {global_id!r} provides {len(gp)} mask(s), but slot {key!r} '
                        f'has {n_imgs} image(s). Provide 1 mask (broadcast) or exactly {n_imgs} masks.'
                    )
            else:
                raise TypeError(
                    f'Upstream output for {global_id!r} must provide \'image\'/\'path\' as str or list[str]. '
                    f'Got: {type(gp)}'
                )

        # 3) final check
        missing = [i for i, m in enumerate(masks) if m is None]
        if missing:
            raise ValueError(
                f'IP-Adapter {key!r} declared masks but missing mask(s) for indices: {missing}. '
                f'Provide {global_id!r} (global) or per-index inputs ip_adapter_mask:{key}[i].'
            )

        return [m for m in masks if m is not None]

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

    @classmethod
    def _num_images_for_slot(cls, img_path: Union[str, List[str]]) -> int:
        return len(img_path) if isinstance(img_path, list) else 1

    @classmethod
    def _validate_scale_for_slot(
        cls,
        *,
        key: str,
        scale: IpAdapterScale,
        img_path: Union[str, List[str]],
    ) -> None:
        """
        Minimal validation for IP-Adapter scale vs number of images in a slot.

        Rules:
        1) If scale is List[float], it must match the number of images in that slot.
        2) If scale is a per-block dict and the slot has >1 images, raise (ambiguous/undocumented).
        """
        n_imgs = cls._num_images_for_slot(img_path)

        if isinstance(scale, list):
            if len(scale) != n_imgs:
                raise ValueError(
                    f'Invalid IP-Adapter scale for {key!r}: got a per-image scale list of length {len(scale)}, '
                    f'but the slot has {n_imgs} image(s). '
                    'Provide one scale per image, or use a single float.'
                )
            return

        if isinstance(scale, dict):
            if n_imgs > 1:
                raise ValueError(
                    f'Invalid IP-Adapter scale for {key!r}: per-block scale dict is not supported when multiple '
                    f'reference images are attached to the same adapter slot (got {n_imgs} images). '
                    'Use a single reference image for per-block scaling, or use per-image float scales instead.'
                )
            return

        if isinstance(scale, (float, int)):
            return

        raise TypeError(
            f'Invalid scale type for IP-Adapter {key!r}: {type(scale)}. '
            'Expected float, List[float], or per-block dict.'
        )

    def build_ip_adapter_masks(
        self,
        *,
        height: int,
        width: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Returns masks in the format expected by diffusers:
        cross_attention_kwargs={"ip_adapter_masks": masks}

        Output: List[tensor], one per adapter slot.
        Each tensor has shape: (1, num_images_for_that_adapter, H, W)
        """
        if not any(m is not None for m in self._masks):
            return None

        processor = IPAdapterMaskProcessor()
        out: List[torch.Tensor] = []

        for i, ad in enumerate(self._specs):
            mask_paths = self._masks[i]  # Optional[List[str]]

            # How many images are attached to this adapter?
            # IMPORTANT: to fully support diffusers, self._images[i] should be a list[str] (or normalize it to list)
            img_paths = self._images[i]
            if isinstance(img_paths, str):
                img_paths = [img_paths]

            n_imgs = len(img_paths)

            if ad.has_mask and mask_paths is None:
                raise ValueError("declared mask but missing")

            if mask_paths is None:
                # lenient fallback: full-white masks for all images
                pil_masks = [Image.new("L", (width, height), color=255)
                             for _ in range(n_imgs)]
            else:
                if len(mask_paths) != n_imgs:
                    raise ValueError(
                        f"IP-Adapter '{ad.key}' expects {n_imgs} mask(s) (one per image), "
                        f'got {len(mask_paths)}.'
                    )
                pil_masks = [Image.open(p).convert("RGB") for p in mask_paths]

            # doc suggests this path
            m = processor.preprocess(pil_masks, height=height, width=width)
            # doc reshapes (1, N, H, W) dropping the channel dim
            m = m.reshape(1, m.shape[0], m.shape[2], m.shape[3])

            if device is not None:
                m = m.to(device)
            if dtype is not None:
                m = m.to(dtype)

            out.append(m)

        return out

    @property
    def has_ip_adapter(self) -> bool:
        return self._has

    @property
    def ip_adapter_image(self) -> Union[
        Image.Image,
        List[Image.Image],
        List[List[Image.Image]],
    ]:
        per_adapter: List[Union[Image.Image, List[Image.Image]]] = []

        for item in self._images:
            if isinstance(item, str):
                per_adapter.append(Image.open(item).convert('RGB'))
            elif isinstance(item, list):
                per_adapter.append([Image.open(p).convert('RGB')
                                   for p in item])
            else:
                raise TypeError(
                    f'Invalid ip-adapter image entry: expected str or list[str], got {type(item)}'
                )

        # if only one adapter slot, return its payload directly (Image or list[Image])
        if len(per_adapter) == 1:
            # TODO In case the single item is an array then it should return per adapter, otherwise first item.
            return per_adapter if isinstance(per_adapter[0], list) else per_adapter[0]

        # otherwise return the per-adapter list (flat or nested depending on inputs)
        return per_adapter

    @property
    def scale_arg(self) -> Union[IpAdapterScale]:
        return self._scales if len(self._scales) > 1 else self._scales[0]

    @property
    def weight_names_arg(self) -> Union[str, List[str]]:
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


class PromptRegistry:
    """
    Single-slot prompt attachment for a node.

    This registry intentionally exposes a single prompt channel. Wiring more
    than one upstream prompt into the same node will fail DAG validation due
    to duplicate input_id.
    """

    INPUT_ID = 'prompt:default'

    def __init__(self, owner: NodeRef):
        """
        Initialize a prompt registry for a node.

        Parameters
        ----------
        owner : NodeRef
            The node that owns this registry. All attachment sinks created
            by this registry will target this node.
        """

        self._owner = owner
        self._counter = 0

    def __call__(
        self,
        sink_id: Optional[str] = None
    ):
        """
        Create an attachment sink for the prompt input channel.

        This method returns an ``AttachmentSink`` bound to the owning node and
        targeting the fixed prompt input channel (``prompt:default``). The sink
        represents a single wiring point in the DAG DSL.

        Parameters
        ----------
        sink_id : str, optional
            Optional unique identifier for the sink instance. This identifier
            represents the identity of the attachment sink itself (for debugging,
            logging, or future extensions), and is independent of the semantic
            input channel.

            If not provided, a unique sink identifier is automatically generated
            by the registry. Automatically generated identifiers are guaranteed
            to be unique within the lifetime of the registry instance.

        Returns
        -------
        AttachmentSink
            An attachment sink targeting the owning node on the fixed prompt
            input channel (``prompt:default``).

        Notes
        -----
        - The prompt channel is intentionally single-slot: wiring more than one
        upstream prompt into the same node will fail DAG validation due to
        duplicate ``input_id``.
        - The ``sink_id`` identifies the sink instance only and does not affect
        input routing semantics, which are entirely determined by ``input_id``.
        """
        if sink_id is None:
            self._counter += 1
            # unique sink identity (NOT the input channel)
            sink_id = f'sink:prompt:{self._owner.id}:{self._counter}'

        return AttachmentSink(
            id=sink_id,
            target=self._owner,
            input_id=PromptRegistry.INPUT_ID
        )


class PromptBundle:
    def __init__(self, *, spec: Dict[str, Any], input: Optional[Dict[str, Dict]]):
        input = input or {}
        upstream = input.get(PromptRegistry.INPUT_ID) or {}

        # prompts in spec has lower priority
        prompt, prompt_2 = norm_prompt_pair(spec.get('prompt'))
        negative_prompt, negative_prompt_2 = norm_prompt_pair(
            spec.get('negative_prompt'),
            joiner=', '
        )

        self._prompt = upstream.get('prompt') or prompt or ''
        self._prompt_2 = upstream.get('prompt_2') or prompt_2 or None
        self._negative_prompt = upstream.get('negative_prompt') \
            or negative_prompt or ''
        self._negative_prompt_2 = upstream.get('negative_prompt_2') \
            or negative_prompt_2 or None

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def prompt_2(self) -> Optional[str]:
        return self._prompt_2

    @property
    def negative_prompt(self) -> str:
        return self._negative_prompt

    @property
    def negative_prompt_2(self) -> Optional[str]:
        return self._negative_prompt_2
