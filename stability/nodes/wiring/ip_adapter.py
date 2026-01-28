from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from stability.cache.models import get_ip_image_encoder
from stability.dag import AttachmentSink, NodeRef
from stability.nodes.wiring.utils import infer_image_encoder_subfolder

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
    scale: IpAdapterScale = 1.0
    has_mask: bool = False


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

        spec = IpAdapterSpec(
            key=key,
            model_id=model_id,
            weight_name=weight_name,
            subfolder=subfolder,
            scale=scale,
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
        self._weight_names: List[str] = []
        self._masks: List[Optional[List[str]]] = []
        self._with_mask = False

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

        # In case there are masks it should be recorded.
        if any(ad.has_mask for ad in adapters):
            self._with_mask = True

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
            self._scales.append(ad.scale)
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
            repo_id='h94/IP-Adapter',
            subfolder=infer_image_encoder_subfolder(self._weight_names),
            device=device,
            dtype=dtype,
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
    def scale_arg(self) -> List[IpAdapterScale]:
        return self._scales

    @property
    def weight_names_arg(self) -> List[str]:
        return self._weight_names

    @property
    def model_id_arg(self) -> str:
        return self._model_id

    @property
    def subfolder_arg(self) -> Optional[str]:
        return self._subfolder

    @property
    def image_encoder(self) -> Optional[CLIPVisionModelWithProjection]:
        return self._image_encoder

    @property
    def with_mask(self) -> bool:
        return self._with_mask
