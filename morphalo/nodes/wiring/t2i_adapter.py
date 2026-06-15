from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from PIL import Image

from morphalo.cache.models import get_t2i_adapter
from morphalo.dag import AttachmentSink, NodeRef

if TYPE_CHECKING:
    from diffusers import T2IAdapter
    from diffusers.models import MultiAdapter


@dataclass
class T2IAdapterSpec:
    """Declarative spec for a single T2I-Adapter attachment."""
    key: str
    model_id: str
    conditioning_scale: float = 1.0


class T2IAdapterRegistry:
    """Registry for declaring T2I-Adapter inputs on a DAG node.

    The registry is purely declarative: it records which adapter(s) a node
    expects and returns :class:`~morphalo.dag.AttachmentSink` objects that can
    be wired from upstream nodes producing the adapter conditioning image
    (e.g. canny/lineart/sketch/depth maps).

    The actual adapter model loading and conditioning image resolution happen
    at runtime via :class:`~morphalo.nodes.T2IAdapterBundle`.
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
            name=f't2i-adapter:{key}',
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
        from diffusers.models import MultiAdapter

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
