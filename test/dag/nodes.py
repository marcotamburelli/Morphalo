from dataclasses import dataclass
from typing import Dict, Any
from stability.dag import NodeRef, AttachmentSink


@dataclass
class SourceNode(NodeRef):
    value: Any = None

    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict[str, Any]:
        print(f"[{self.id}] SourceNode.run(input={input})")
        return {"value": self.value, "from": self.id}


@dataclass
class PassNode(NodeRef):
    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        print(f"[{self.id}] PassNode.run(input={input})")
        # prende il primo input e lo rilancia
        k, v = next(iter(input.items()))
        return {"value": v["value"], "from": self.id}


@dataclass
class MergeNode(NodeRef):
    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        print(f"[{self.id}] MergeNode.run(input={input})")
        values = {k: v["value"] for k, v in input.items()}
        return {"merged": values, "from": self.id}

    def getAttachmentSink(self, id: str, input_id: str):
        return AttachmentSink(id=id, target=self, input_id=input_id)
