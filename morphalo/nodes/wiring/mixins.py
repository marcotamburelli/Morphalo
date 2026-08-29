from typing import Dict, Optional

import torch

from morphalo.nodes.wiring.controlnet import (ControlNetBundle,
                                              ControlNetRegistry)
from morphalo.nodes.wiring.face_id import FaceIdBundle, FaceIdRegistry
from morphalo.nodes.wiring.ip_adapter import IpAdapterBundle, IpAdapterRegistry
from morphalo.nodes.wiring.lora import LoraBundle, LoraRegistry
from morphalo.nodes.wiring.prompt import PromptRegistry
from morphalo.nodes.wiring.t2i_adapter import (T2IAdapterBundle,
                                               T2IAdapterRegistry)


class ControlNetMixin:
    def __post_init__(self):
        super().__post_init__()
        self.controlnet = ControlNetRegistry(owner=self)
        self.ip_adapter = IpAdapterRegistry(owner=self)
        self.face_id = FaceIdRegistry(owner=self)

    def build_control_bundles(self, input, device, dtype):
        cn_bundle = ControlNetBundle(
            self.controlnet.specs, dtype=dtype, device=device, input=input
        )
        ip_bundle = IpAdapterBundle(
            self.ip_adapter.specs, dtype=dtype, device=device, input=input
        )
        face_bundle = FaceIdBundle(
            self.face_id.specs, dtype=dtype, device=device, input=input
        )
        return cn_bundle, ip_bundle, face_bundle


class LoraMixin:
    def __post_init__(self):
        super().__post_init__()
        self.lora = LoraRegistry(owner=self)

    def build_lora_bundle(self) -> LoraBundle:
        return LoraBundle(
            self.lora.specs,
            cross_attention_scale=self.lora.cross_attention_scale,
        )


class T2IAdapterMixin:
    def __post_init__(self):
        super().__post_init__()
        self.t2i_adapter = T2IAdapterRegistry(owner=self)

    def build_t2i_adapter_bundle(self, input: Optional[Dict[str, Dict]], device: str, dtype: torch.dtype) -> T2IAdapterBundle:
        return T2IAdapterBundle(
            adapters=self.t2i_adapter.specs,
            dtype=dtype,
            device=device,
            input=input or {}
        )


class PromptMixin:
    def __post_init__(self):
        super().__post_init__()
        self.prompt = PromptRegistry(owner=self)
