from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from stability.cache.models import get_ip_image_encoder
from stability.dag import NodeRef
from stability.nodes.utils import infer_image_encoder_subfolder
from stability.nodes.wiring.ip_adapter import IpAdapterAttachmentSink

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
    image: Union[Image.Image, List[Image.Image]]
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
    ) -> IpAdapterAttachmentSink:
        """
        Declare a FaceID adapter slot and return a DAG sink for wiring.

        This method registers one FaceID "slot" on the owning node. The slot
        describes a single FaceID-capable IP-Adapter weight to be loaded at runtime
        (e.g. FaceID, FaceID Plus, FaceID PlusV2). The returned sink is used to wire
        an upstream node that provides both:

        - identity embeddings (`embeds`: path to a saved tensor), and
        - reference image(s) (`image`: path or list of paths), which are required
        only for FaceID Plus / PlusV2 CLIP injection.

        The slot is later materialized into a runtime bundle (``FaceIdBundle``),
        which is responsible for:
        - collecting the wired embedding paths,
        - loading tensors from disk,
        - collecting optional masks (via IpAdapterAttachmentSink behavior),
        - collecting reference images for CLIP (Plus / PlusV2),
        - producing arguments for `pipe.load_ip_adapter`, `pipe.set_ip_adapter_scale`,
            and `pipe(...)`.

        Parameters
        ----------
        model_id:
            Hugging Face repo id (or local path) containing the FaceID adapter weights.
            Example: ``"h94/IP-Adapter-FaceID"``.

        weight_name:
            File name of the FaceID adapter weights inside the repo (e.g.
            ``"ip-adapter-faceid-plusv2_sdxl.bin"``). This is the value later passed
            to Diffusers ``pipe.load_ip_adapter(weight_name=...)``.

        subfolder:
            Optional subfolder within the repo containing the weight file.
            For most h94 FaceID weights this is typically ``None``.
            If used, it is forwarded to Diffusers `load_ip_adapter(subfolder=...)`.

        scale:
            Adapter strength / scaling, forwarded to `pipe.set_ip_adapter_scale(...)`.
            Supports:
            - float (single scale),
            - list[float] (per-reference scale) if you support multi-reference,
            - dict (reserved for per-block scales, if you keep parity with IP-Adapter).

        clip_strength:
            Multiplicative factor applied to the CLIP embeddings injected for this
            slot (FaceID Plus / PlusV2 only). This controls how much "visual" signal
            from the CLIP embedding influences the adapter relative to the FaceID
            identity embedding.

            - `1.0` (default): use CLIP as computed by Diffusers.
            - `0.0`: effectively neutralize the visual component (while still injecting
            a correctly-shaped tensor to satisfy Plus/PlusV2 execution).
            - `0 < value < 1`: attenuate visual influence.
            - `> 1`: amplify visual influence (use carefully; may reintroduce style bleed).

            Note: this does *not* replace the adapter `scale`; it modulates only the
            CLIP side-channel used by Plus/PlusV2.

        key:
            Optional stable identifier for this slot. If omitted, a unique key is
            auto-generated (``fid1``, ``fid2``, ...). The key determines the input
            channel name in the DAG.

        Returns
        -------
        IpAdapterAttachmentSink
            A sink that can be wired to an upstream node providing FaceID inputs.

            The sink uses input id: ``face_id:{key}`` and expects upstream output
            to contain at least:
            - ``embeds`` or ``path`` : str (path to `.pt` tensor)
            - ``image`` : str or list[str] (reference images), required when this
                adapter weight requires CLIP (Plus/PlusV2)

            Mask wiring is handled by `IpAdapterAttachmentSink` using the standard
            `ip_adapter_mask:{key}` (and optional indexed variants) channels.

        Raises
        ------
        ValueError
            If `clip_strength` is not finite or is negative (recommended to enforce).
            If `key` is invalid (if you choose to validate naming).
        """
        if key is None:
            self._counter += 1
            key = f'fid{self._counter}'

        spec = FaceIdSpec(
            key=key,
            model_id=model_id,
            weight_name=weight_name,
            subfolder=subfolder,
            scale=scale,
            clip_strength=clip_strength
        )
        self._specs.append(spec)

        return IpAdapterAttachmentSink(
            id=f'face_id:{key}',
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

            # masks: mirror the ip_adapter bundle behavior
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
                ip = upstream.get('image')
                if not ip:
                    raise ValueError(
                        f"Upstream output for {in_id!r} must contain an image path in 'image'."
                    )
                self._clip_img_paths.append((ip, is_plusv2))
            else:
                self._clip_img_paths.append(None)

            self._specs.append(ad)

        # Load embeds tensors (and normalize shapes) eagerly here.
        self._embeds = [
            self._load_faceid_embeds(p) for p in self._embeds_paths
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

    def _load_faceid_embeds(self, path_or_paths: Union[str, List[str]]) -> torch.Tensor:
        """
        Load FaceID embeds and normalize to (2, N, D).

        If a list of paths is provided, we concatenate references along N (dim=1),
        preserving the (neg,pos) pairing in dim=0.
        """
        def load_one(p: str) -> torch.Tensor:
            t = torch.load(p, map_location='cpu')
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f'FaceID embeds file {p!r} did not contain a torch.Tensor. Got: {type(t)}')
            if t.ndim != 3 or t.shape[0] != 2 or t.shape[1] < 1:
                raise ValueError(
                    f'FaceID embeds must have shape (2, N, D). Got {tuple(t.shape)} from {p!r}')
            return t

        if isinstance(path_or_paths, str):
            t = load_one(path_or_paths)
        else:
            ts = [load_one(p) for p in path_or_paths]
            # concatenate along N dimension
            t = torch.cat(ts, dim=1)  # (2, sum(Ni), D)

        return t.to(device=self._device, dtype=self._dtype)

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

        Invariants (by design):
        - One FaceID slot corresponds to exactly ONE embedding item (N = 1).
        - Therefore, each mask tensor per slot has shape (1, 1, H, W).
        - Masks are optional unless the spec declares `has_mask=True`.

        Return:
        - A list of mask tensors aligned 1:1 with `self._specs`.
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
            if isinstance(p[0], str):
                # out.append((Image.open(p[0]).convert("RGB"), p[1]))
                self._clip_images_per_slot.append(ClipImg(
                    image=Image.open(p[0]).convert("RGB"),
                    is_plusv2=p[1],
                    clip_strength=spec.clip_strength,
                ))
            elif isinstance(p[0], list):
                self._clip_images_per_slot.append(ClipImg(
                    image=[Image.open(x).convert("RGB") for x in p[0]],
                    is_plusv2=p[1],
                    clip_strength=spec.clip_strength,
                ))
            else:
                raise TypeError(f"Invalid clip image entry: {type(p)}")

        return self._clip_images_per_slot

    @property
    def with_mask(self) -> bool:
        return self._with_mask

    @property
    def image_encoder(self) -> Optional[CLIPVisionModelWithProjection]:
        return self._image_encoder
