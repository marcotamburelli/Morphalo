from stability.nodes.ltx.wiring.ic_lora import ICLoRaBundle, ICLoRaRegistry


class IcLoRaMixin:
    def __post_init__(self):
        super().__post_init__()
        self.ic_lora = ICLoRaRegistry(self)

    def build_ic_lora_bundle(self, input):
        return ICLoRaBundle(
            ic_lora=self.ic_lora.spec,
            input=input
        )
