import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from morphalo.cache.models import get_ip_image_encoder
from morphalo.dag import NodeRef
from morphalo.dag.core import AttachmentSink
from morphalo.nodes.common.io import load_faceid_embeds
from morphalo.nodes.wiring.utils import infer_image_encoder_subfolder

# For FaceID, scale behaves like IP-Adapter scale:
# float | List[float] | per-block dict (kept for future parity).
FaceIdScale = Union[
    float,
    List[float],
    Dict[str, Dict[str, List[float]]]
]

ClipImgPath = Tuple[Union[str, List[str]], bool]


@dataclass
class ClipImg:
    """
    Container for CLIP reference images used by FaceID Plus / PlusV2.

    Notes
    -----
    We intentionally normalize `image` to a list of PIL images. This avoids
    type-dependent branching and makes multi-reference semantics explicit.
    Aggregation (e.g., mean of CLIP embeddings) should be implemented at the
    injection site (runner / adapter), not here.
    """
    image: List[Image.Image]
    clip_strength: float = 1.0
    is_plusv2: bool = False


@dataclass
class FaceIdSpec:
    """
    Declarative specification for a FaceID adapter slot.

    Notes
    -----
    - `model_id/subfolder/weight_name` are used to load the FaceID IP-Adapter weights.
    - Runtime input is NOT an image: it is a path (or list of paths) to a saved tensor (.pt)
      containing the face-id embeddings (usually extracted with insightface).
    """
    key: str
    model_id: str
    weight_name: str
    subfolder: Optional[str] = None
    scale: FaceIdScale = 1.0
    clip_strength: float = 1.0

    has_mask: bool = False


@dataclass
class FaceIdAttachmentSink(AttachmentSink):
    """
    Attachment sink representing a single FaceID adapter slot.

    The main sink receives FaceID identity embeddings. Additional helper
    methods can declare optional inputs bound to the same FaceID slot, such
    as a CLIP reference image for FaceID Plus / PlusV2.
    """

    key: str
    spec: FaceIdSpec

    def clip(self) -> AttachmentSink:
        """
        Declare an explicit CLIP reference image input for this FaceID slot.

        This input is used only by FaceID Plus / PlusV2 adapters. When provided,
        it overrides the fallback CLIP image exposed by the upstream FaceID
        embedding node.

        The upstream node must provide one of:

        - ``image`` : str
        - ``images`` : list[str]
        - ``path`` : str or list[str]

        Returns
        -------
        AttachmentSink
            A sink bound to this FaceID slot with
            ``input_id=f'face_id_clip:{key}'``.
        """
        return AttachmentSink(
            name=f'face_id_clip:{self.key}',
            target=self.target,
            input_id=f'face_id_clip:{self.key}',
        )

    def mask(self) -> AttachmentSink:
        """
        Declare a slot-level mask input for this FaceID adapter slot.

        This mask applies to the whole FaceID slot. It limits where the FaceID
        adapter contribution is applied on the generation canvas, independently of
        how many identity embeddings or CLIP reference images are associated with
        the slot.

        Unlike standard IP-Adapter masking, FaceID does not support per-reference
        or per-index masks. A FaceID slot accepts exactly one mask, shared by all
        embeddings and CLIP references belonging to that slot.

        Returns
        -------
        AttachmentSink
            A sink expecting a single mask path.

        Input conventions
        -----------------
        The upstream node must provide:

        - ``image`` : str
            Single mask path.

        - ``path`` : str
            Alternative key for the single mask path.

        Notes
        -----
        - The mask is slot-level, not embedding-level.
        - The same mask is used for the FaceID identity component and, when present,
          for the FaceID Plus / PlusV2 CLIP visual side-channel.
        - Lists of masks are intentionally not supported.
        - Per-index masks are intentionally not supported; FaceID masking is kept
          simpler than standard IP-Adapter masking to avoid ambiguous alignment
          between identity embeddings, CLIP references, and projection layers.
        """
        self.spec.has_mask = True

        return AttachmentSink(
            name=f'ip_adapter_mask:{self.key}',
            target=self.target,
            input_id=f'ip_adapter_mask:{self.key}',
        )


class FaceIdRegistry:
    """
    Declarative registry for FaceID attachments on a DAG node.

    Similar to IpAdapterRegistry, but the primary input is embeddings (.pt),
    not images.
    """

    def __init__(self, owner: NodeRef):
        self._owner = owner
        self._counter = 0
        self._specs: List[FaceIdSpec] = []

    def add(
        self,
        model_id: str,
        *,
        weight_name: str,
        subfolder: Optional[str] = None,
        scale: FaceIdScale = 1.0,
        clip_strength: float = 1.0,
        key: Optional[str] = None,
    ) -> FaceIdAttachmentSink:
        """
        Declare a FaceID adapter slot and return its attachment sink.

        This method registers one FaceID-capable IP-Adapter slot on the owning
        node. The slot describes which FaceID adapter weights should be loaded
        at runtime and how strongly they should influence the diffusion process.

        The returned sink represents the main FaceID input for this slot. It is
        wired from an upstream node that provides identity embeddings, usually
        produced by ``FaceIdEmbedImage``.

        For FaceID Plus / PlusV2 variants, the same returned sink also exposes
        helper methods for slot-bound auxiliary inputs:

        - ``.clip()``
            Declares an optional CLIP reference-image input for this FaceID slot.
            This image is used only by FaceID Plus / PlusV2 as the visual
            side-channel. It can be different from the image used to extract the
            FaceID identity embeddings.

        - ``.mask()``
            Declares an optional mask input for this FaceID slot. The mask limits
            where the FaceID / IP-Adapter contribution is applied spatially.

        Conceptually, a FaceID slot may therefore receive:

        1. Identity embeddings
            The main FaceID input, wired directly into the returned sink::

                face_embed_node >> fid

            The upstream payload must contain either:

            - ``embeds`` : str
                Path to a saved FaceID embedding tensor.
            - ``path`` : str
                Alternative path key for the saved embedding tensor.

        2. Optional CLIP reference image(s)
            Used only by FaceID Plus / PlusV2. This input is declared with
            ``fid.clip()``::

                clip_reference_node >> fid.clip()

            The upstream payload must contain one of:

            - ``image`` : str
                Single CLIP reference image.
            - ``images`` : list[str]
                Multiple CLIP reference images for the same slot.
            - ``path`` : str or list[str]
                Alternative path key.

            When multiple CLIP images are provided, they are interpreted as
            multiple visual references for the same FaceID slot. Runtime code may
            aggregate their CLIP embeddings, for example by averaging them.

            If ``fid.clip()`` is not wired, the runtime bundle falls back to the
            image paths exposed by the main FaceID upstream payload. This preserves
            the legacy behavior where the same images used to extract FaceID
            embeddings are also used as CLIP references.

        3. Optional spatial mask
            Declared with ``fid.mask()``::

                mask_node >> fid.mask()

            The upstream payload must contain a single mask path via:

            - ``image`` : str
            - ``path`` : str

            For FaceID, masks are interpreted per slot, not per embedding image.
            A single mask is therefore broadcast to the whole FaceID slot.

        Parameters
        ----------
        model_id : str
            Hugging Face repository id or local path containing the FaceID
            adapter weights.

            Example: ``'h94/IP-Adapter-FaceID'``.

        weight_name : str
            FaceID adapter weight file to load.

            Examples include:

            - ``'ip-adapter-faceid_sdxl.bin'``
            - ``'ip-adapter-faceid-plusv2_sdxl.bin'``

            This value is later forwarded to Diffusers
            ``pipe.load_ip_adapter(..., weight_name=...)``.

        subfolder : str, optional
            Optional subfolder within ``model_id`` containing the weight file.

            For most ``h94/IP-Adapter-FaceID`` weights this is usually ``None``.
            If provided, it is forwarded to Diffusers
            ``pipe.load_ip_adapter(..., subfolder=...)``.

        scale : FaceIdScale, default 1.0
            Adapter strength forwarded to ``pipe.set_ip_adapter_scale(...)``.

            Supported forms are:

            - ``float``
                Uniform strength for this FaceID slot.

            - ``list[float]``
                Per-embedding strength, when multiple FaceID embedding tensors are
                attached to the same slot.

            - ``dict[str, dict[str, list[float]]]``
                Reserved for per-block scale configurations, keeping parity with
                IP-Adapter style scale definitions.

            ``scale`` controls the overall FaceID adapter contribution. It is
            independent from ``clip_strength``.

        clip_strength : float, default 1.0
            Multiplicative factor applied only to the CLIP visual side-channel
            used by FaceID Plus / PlusV2 variants.

            This value does not replace ``scale``. Instead, it controls how much
            the CLIP reference image influences the adapter relative to the
            identity embedding.

            Typical values:

            - ``1.0``
                Use CLIP embeddings as computed by Diffusers.

            - ``0.0``
                Neutralize the visual CLIP component while still injecting a
                correctly shaped tensor required by FaceID Plus / PlusV2.

            - ``0 < value < 1``
                Attenuate the visual reference influence.

            - ``> 1``
                Amplify the visual reference influence. Use carefully, as this
                may increase style or appearance bleed from the CLIP reference.

            If an explicit ``fid.clip()`` input is wired, ``clip_strength``
            applies to that explicit CLIP reference. Otherwise, it applies to the
            fallback CLIP reference obtained from the main FaceID upstream payload.

        key : str, optional
            Stable identifier for this FaceID slot.

            If omitted, a unique key is generated automatically
            (``'fid1'``, ``'fid2'``, ...). The key determines the main DAG input
            channel name:

            ``face_id:{key}``

            It is also used for auxiliary slot-bound inputs such as:

            - ``face_id_clip:{key}``
            - ``ip_adapter_mask:{key}``

        Returns
        -------
        FaceIdAttachmentSink
            FaceID-specific attachment sink targeting the owning node.

            The sink itself is wired to the main FaceID input
            ``face_id:{key}`` and expects an upstream payload containing
            identity embeddings.

            The returned sink also exposes helper methods for optional
            slot-bound inputs:

            - ``clip()``
                Returns a sink for an explicit FaceID Plus / PlusV2 CLIP
                reference image input.

            - ``mask()``
                Returns a sink for a single spatial mask applied to this FaceID
                slot.

        Raises
        ------
        ValueError
            If ``clip_strength`` is not finite or is negative, or if runtime inputs
            are inconsistent with the declared FaceID slot configuration.
        """
        if key is None:
            self._counter += 1
            key = f'fid{self._counter}'

        clip_strength = float(clip_strength)
        if not math.isfinite(clip_strength) or clip_strength < 0.0:
            raise ValueError(
                f'Invalid clip_strength={clip_strength!r}; expected a finite value >= 0.'
            )

        spec = FaceIdSpec(
            key=key,
            model_id=model_id,
            weight_name=weight_name,
            subfolder=subfolder,
            scale=scale,
            clip_strength=clip_strength
        )
        self._specs.append(spec)

        return FaceIdAttachmentSink(
            name=f'face_id:{key}',
            target=self._owner,
            input_id=f'face_id:{key}',
            key=key,
            spec=spec,
        )

    @property
    def specs(self) -> List[FaceIdSpec]:
        return self._specs


class FaceIdBundle:
    """
    Runtime bundle for FaceID slots.

    Provides:
      - `ip_adapter_image_embeds`: list[Tensor] (one per slot) to pass as
        `ip_adapter_image_embeds=[...]` into diffusers pipeline call.
      - `build_ip_adapter_masks(...)`: compatible with diffusers cross_attention_kwargs
        `{"ip_adapter_masks": masks}`, one tensor per slot.
      - `clip_images_per_slot`: optional list of PIL images for Plus/PlusV2, to be used by
        the runner to compute & inject CLIP embeds into hidden projection layers.
    """

    def __init__(
        self,
        adapters: Optional[List[FaceIdSpec]],
        *,
        dtype: torch.dtype,
        device: str,
        input: Optional[Dict[str, Dict]],
    ):
        self._has = bool(adapters)

        self._specs: List[FaceIdSpec] = []
        self._model_id: Optional[str] = None
        self._subfolder: Optional[str] = None
        self._weight_names: List[str] = []
        self._weight_names_requiring_clip: List[str] = []
        self._scales: List[FaceIdScale] = []

        # per slot:
        self._embeds_paths: List[Union[str, List[str]]] = []
        self._embeds: List[torch.Tensor] = []
        self._masks: List[Optional[str]] = []
        self._clip_img_paths: List[Optional[ClipImgPath]] = []
        self._clip_images_per_slot: Optional[List[Optional[ClipImg]]] = None

        self._dtype = dtype
        self._device = device

        self._with_mask = False

        if self._has:
            self._build(
                adapters=adapters,
                dtype=dtype,
                device=device,
                input=input or {}
            )

    def _build(self, adapters: List[FaceIdSpec], *, dtype, device, input: Dict[str, Dict]) -> None:
        self._validate(adapters)

        # In case there are masks it should be recorded.
        if any(ad.has_mask for ad in adapters):
            self._with_mask = True

        for ad in adapters:
            in_id = f'face_id:{ad.key}'
            upstream = input.get(in_id)
            if upstream is None:
                raise ValueError(
                    f'Missing FaceID input for {in_id!r}. Did you wire the embeddings node into it?'
                )

            emb_path = upstream.get('embeds') or upstream.get('path')
            if not emb_path:
                raise ValueError(
                    f"Upstream output for {in_id!r} must contain embeddings path in 'embeds' or 'path'."
                )
            if not isinstance(emb_path, (str, list)):
                raise TypeError(
                    f'Upstream output for {in_id!r} must be str or list[str]. Got: {type(emb_path)}'
                )

            self._validate_scale_for_slot(
                key=ad.key, scale=ad.scale, embeds_path=emb_path)

            self._weight_names.append(ad.weight_name)
            self._scales.append(ad.scale)
            self._embeds_paths.append(emb_path)

            # masks: FaceID uses one optional slot-level mask
            if ad.has_mask:
                self._masks.append(self._collect_mask_for_slot(
                    key=ad.key,
                    input=input,
                ))
            else:
                self._masks.append(None)

            wn = ad.weight_name.lower()
            requires_clip = ('plus' in wn)
            is_plusv2 = ('v2' in wn)

            if requires_clip:
                self._weight_names_requiring_clip.append(ad.weight_name)

            # plus/v2 clip reference images (optional; runner will compute clip embeds)
            if requires_clip:
                clip_paths = self._collect_clip_images_for_slot(
                    key=ad.key,
                    input=input,
                    fallback_upstream=upstream,
                )
                self._clip_img_paths.append((clip_paths, is_plusv2))
            else:
                self._clip_img_paths.append(None)

            self._specs.append(ad)

        # Load embeds tensors (and normalize shapes) eagerly here.
        self._embeds = [
            load_faceid_embeds(
                p,
                device=self._device,
                dtype=self._dtype,
            ) for p in self._embeds_paths
        ]

        if self._weight_names_requiring_clip:
            self._image_encoder = get_ip_image_encoder(
                repo_id='h94/IP-Adapter',
                subfolder=infer_image_encoder_subfolder(
                    self._weight_names_requiring_clip
                ),
                device=device,
                dtype=dtype,
            )
        else:
            self._image_encoder = None

    def _validate(self, adapters: List[FaceIdSpec]) -> None:
        # For now: enforce one model_id/subfolder per node, same as IpAdapterBundle
        model_ids = {ad.model_id for ad in adapters}
        if len(model_ids) != 1:
            raise ValueError(
                f'All FaceID adapters in one node must share the same model_id. Got: {sorted(model_ids)}'
            )
        self._model_id = next(iter(model_ids))

        subfolders = {ad.subfolder for ad in adapters}
        if len(subfolders) != 1:
            raise ValueError(
                'All FaceID adapters in one node must share the same subfolder to be loaded in one call. '
                f'Got: {sorted(subfolders)}'
            )
        self._subfolder = next(iter(subfolders))

    @staticmethod
    def _collect_clip_images_for_slot(
        *,
        key: str,
        input: Dict[str, Dict],
        fallback_upstream: Dict[str, Any],
    ) -> Union[str, List[str]]:
        """
        Collect CLIP reference image paths for a FaceID Plus / PlusV2 slot.

        Priority order:

        1. Explicit slot-bound CLIP input:
        ``face_id_clip:{key}``

        2. Explicit fields in the main FaceID upstream payload:
        ``clip_image`` / ``clip_images``

        3. Legacy fallback fields in the main FaceID upstream payload:
        ``image`` / ``images``

        The fallback deliberately does not use ``path`` because, for FaceID, ``path``
        may refer to the embeddings tensor rather than to an image.
        """
        clip_id = f'face_id_clip:{key}'
        clip_upstream = input.get(clip_id)

        if clip_upstream is not None:
            clip_paths = (
                clip_upstream.get('image')
                or clip_upstream.get('images')
                or clip_upstream.get('path')
            )
        else:
            clip_paths = (
                fallback_upstream.get('clip_image')
                or fallback_upstream.get('clip_images')
                or fallback_upstream.get('image')
                or fallback_upstream.get('images')
            )

        if not clip_paths:
            raise ValueError(
                f"FaceID Plus / PlusV2 slot {key!r} requires CLIP reference image(s). "
                f"Wire an image into 'face_id_clip:{key}', or provide 'clip_image', "
                "'clip_images', 'image', or 'images' in the FaceID upstream payload."
            )

        if isinstance(clip_paths, str):
            return clip_paths

        if isinstance(clip_paths, list):
            if not clip_paths:
                raise ValueError(
                    f"FaceID CLIP input for slot {key!r} is an empty list."
                )
            if not all(isinstance(x, str) and x for x in clip_paths):
                raise TypeError(
                    f"FaceID CLIP input for slot {key!r} must be str or list[str]."
                )
            return clip_paths

        raise TypeError(
            f"FaceID CLIP input for slot {key!r} must be str or list[str], "
            f'got {type(clip_paths).__name__}.'
        )

    @staticmethod
    def _num_items_for_slot(path_or_paths: Union[str, List[str]]) -> int:
        return len(path_or_paths) if isinstance(path_or_paths, list) else 1

    @classmethod
    def _validate_scale_for_slot(
        cls,
        *,
        key: str,
        scale: FaceIdScale,
        embeds_path: Union[str, List[str]],
    ) -> None:
        n = cls._num_items_for_slot(embeds_path)

        if isinstance(scale, list):
            if len(scale) != n:
                raise ValueError(
                    f'Invalid FaceID scale for {key!r}: got a per-item scale list of length {len(scale)}, '
                    f'but the slot has {n} embedding(s).'
                )
            return

        if isinstance(scale, dict):
            if n > 1:
                raise ValueError(
                    f'Invalid FaceID scale for {key!r}: per-block scale dict is not supported when multiple '
                    f'embeddings are attached to the same slot (got {n}).'
                )
            return

        if isinstance(scale, (float, int)):
            return

        raise TypeError(
            f'Invalid scale type for FaceID {key!r}: {type(scale)}. '
            'Expected float, List[float], or per-block dict.'
        )

    @staticmethod
    def _collect_mask_for_slot(
        *,
        key: str,
        input: Dict[str, Dict],
    ) -> str:
        """
        FaceID rule: ONE mask per slot (broadcast to all embeddings in that slot).

        We intentionally only accept the *global* mask wire:
            ip_adapter_mask:{key}

        We reject per-index masks:
            ip_adapter_mask:{key}[i]
        """
        # Hard reject per-index masks if present (fail-fast)
        # (This prevents subtle bugs where someone wires masks per-image.)
        for k in input.keys():
            if k.startswith(f"ip_adapter_mask:{key}["):
                raise ValueError(
                    f"FaceID '{key}' does not support per-index masks ('ip_adapter_mask:{key}[i]'). "
                    f"Provide a single global mask 'ip_adapter_mask:{key}'."
                )

        global_id = f'ip_adapter_mask:{key}'
        up = input.get(global_id)
        if up is None:
            raise ValueError(
                f"Missing FaceID mask input for {global_id!r}. "
                "FaceID expects a single mask per slot."
            )

        p = up.get('image') or up.get('path')

        # No lists: single mask only
        if isinstance(p, list):
            raise ValueError(
                f'Invalid mask for {global_id!r}: got a list, but FaceID mask is unique per slot by design.'
            )

        if not isinstance(p, str) or not p:
            raise TypeError(
                f"Upstream output for {global_id!r} must contain a single mask path in 'image' or 'path'. "
                f'Got: {type(p)}'
            )

        return p

    def build_ip_adapter_masks(
        self,
        *,
        height: int,
        width: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        """
        Build `ip_adapter_masks` for FaceID slots (SDXL).

        Invariants
        ----------
        - One FaceID slot has exactly one spatial mask.
        - The mask is slot-level and is shared by all FaceID embeddings and CLIP
          reference images associated with that slot.
        - Each mask tensor has shape ``(1, 1, H, W)``.
        - Masks are optional unless the spec declares ``has_mask=True``.

        Returns
        -------
        list[torch.Tensor]
            Mask tensors aligned 1:1 with ``self._specs``.
        """
        processor = IPAdapterMaskProcessor()
        out: List[torch.Tensor] = []

        for i, ad in enumerate(self._specs):
            mask_path = self._masks[i]  # Optional[str]

            # Only error if the slot explicitly requires a mask
            if ad.has_mask and mask_path is None:
                raise ValueError(
                    f"FaceID '{ad.key}' declared has_mask=True but no mask was provided "
                    f"(expected input 'ip_adapter_mask:{ad.key}')."
                )

            if mask_path is None:
                # No mask for this slot -> default "full" mask (enabled everywhere)
                pil_mask = Image.new("L", (width, height),  color=255)\
                    .convert("RGB")
            else:
                if not isinstance(mask_path, str) or not mask_path:
                    raise TypeError(
                        f"FaceID '{ad.key}' mask must be a single non-empty path (str). "
                        f'Got: {type(mask_path)}'
                    )
                pil_mask = Image.open(mask_path).convert('RGB')

            # preprocess expects a list of images
            m = processor.preprocess([pil_mask], height=height, width=width)

            # m is typically (1, 1, H, W); enforce exact (1, 1, H, W)
            m = m.reshape(1, 1, m.shape[-2], m.shape[-1])

            if device is not None:
                m = m.to(device)
            if dtype is not None:
                m = m.to(dtype)

            out.append(m)

        return out

    @property
    def has_face_id(self) -> bool:
        return self._has

    @property
    def model_id_arg(self) -> str:
        return self._model_id

    @property
    def subfolder_arg(self) -> Optional[str]:
        return self._subfolder

    @property
    def weight_names_arg(self) -> List[str]:
        return self._weight_names

    @property
    def scale_arg(self) -> List[FaceIdScale]:
        return self._scales

    @property
    def ip_adapter_image_embeds(self) -> List[torch.Tensor]:
        """
        Diffusers expects:
          ip_adapter_image_embeds=[tensor_for_slot0, tensor_for_slot1, ...]

        So we always return a list, even for one slot.
        """
        return self._embeds

    @property
    def clip_images_per_slot(self) -> List[Optional[ClipImg]]:
        """
        Return per-slot CLIP reference images (Plus / PlusV2 only), with per-slot strength.

        This property exposes optional reference images required by FaceID Plus /
        FaceID PlusV2 adapters. Base FaceID adapters do not need CLIP reference images
        and therefore return ``None`` for their slot.

        The returned list is aligned with FaceID slot order (and therefore with the
        FaceID weight order loaded into the pipeline). Each element is either:

        - ``None``:
            The corresponding FaceID slot does not require CLIP injection.

        - ``ClipImg``:
            A dataclass containing:
            - `image`: a PIL.Image or list[PIL.Image] (reference images for this slot)
            - `clip_strength`: float multiplier to apply to computed CLIP embeds
            - `is_plusv2`: whether this slot is a PlusV2 variant (used to set
                `projection_layer.shortcut = False`)

        Downstream Usage
        ----------------
        Typical runner flow for FaceID Plus/PlusV2 is:

        1. Load adapters via `pipe.load_ip_adapter(...)`.
        2. Compute CLIP embeddings and inject into the corresponding projection layers:

        - Build a list of images aligned with loaded adapter layers (Diffusers requires
            list length == number of projection layers when using `ip_adapter_image`).
        - Call:

            ``embeds = pipe.prepare_ip_adapter_image_embeds(ip_adapter_image, None, device,
                                                            num_images_per_prompt, do_cfg)``

        - For each slot that returns `ClipImg`, take `embeds[j]`, multiply by
            `clip_strength`, and assign:

            ``pipe.unet.encoder_hid_proj.image_projection_layers[j].clip_embeds = embeds[j] * clip_strength``

        - If `is_plusv2` is True:

            ``pipe.unet.encoder_hid_proj.image_projection_layers[j].shortcut = False``

        Important Invariants
        --------------------
        - The list order matches FaceID adapter slot order and must stay stable to
        preserve correct association between:
            weight_name[j] <-> masks[j] <-> clip_images_per_slot[j] <-> projection_layer[j]

        - The images returned here are opened and converted to RGB at access time.
        Callers should avoid repeatedly accessing this property in hot loops if they
        want to reduce file I/O (cache at the runner level if needed).

        Returns
        -------
        list[Optional[ClipImg]]
            Per-slot CLIP reference images and metadata for FaceID Plus/PlusV2.

        Raises
        ------
        TypeError
            If internal clip image path entries are neither a string nor list of strings.
        FileNotFoundError
            If any of the stored clip image paths is missing (optional to enforce).
        """
        if self._clip_images_per_slot is not None:
            return self._clip_images_per_slot

        self._clip_images_per_slot: List[Optional[ClipImg]] = []
        for spec, p in zip(self._specs, self._clip_img_paths):
            if p is None:
                self._clip_images_per_slot.append(None)
                continue

            # `p` is expected to be a pair like:
            #   (path: str | list[str], is_plusv2: bool)
            # Normalize to list[str] -> list[PIL.Image].
            paths = p[0]
            if isinstance(paths, str):
                paths_list = [paths]
            elif isinstance(paths, list):
                if not paths:
                    raise ValueError(
                        'Invalid FaceID CLIP image entry: empty path list.'
                    )
                if not all(isinstance(x, str) for x in paths):
                    bad = {type(x) for x in paths if not isinstance(x, str)}
                    raise TypeError(
                        'Invalid FaceID CLIP image entry: expected list[str]. '
                        f'Found non-str elements of types: {sorted([t.__name__ for t in bad])}'
                    )
                paths_list = paths
            else:
                raise TypeError(
                    'Invalid FaceID CLIP image entry: expected str or list[str], '
                    f'got {type(paths)}'
                )

            self._clip_images_per_slot.append(ClipImg(
                image=[Image.open(x).convert('RGB') for x in paths_list],
                is_plusv2=p[1],
                clip_strength=spec.clip_strength,
            ))

        return self._clip_images_per_slot

    @property
    def with_mask(self) -> bool:
        return self._with_mask

    @property
    def image_encoder(self) -> Optional[CLIPVisionModelWithProjection]:
        return self._image_encoder
