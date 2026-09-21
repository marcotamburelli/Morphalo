"""Nodes controlling runtime execution through executor policies."""

from typing import Dict

from .core import NodeRef

__all__ = ['CudaCooldown']


class CudaCooldown(NodeRef):
    """Recycle the process worker before forwarding the default input.

    Cooldown is handled by ProcessNodeExecutor. With an in-process runner,
    this node only forwards its input. No input produces an empty output.
    """

    @property
    def forces_cuda_cooldown(self) -> bool:
        return True

    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict:
        if not input:
            return {}
        if set(input) != {'default'}:
            raise ValueError('CudaCooldown accepts only the default input')
        return input['default']
