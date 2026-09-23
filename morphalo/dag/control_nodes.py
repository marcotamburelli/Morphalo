"""Nodes controlling runtime execution through executor policies."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from morphalo.core.paths import make_node_output_path

from .core import NodeRef

__all__ = ['CudaCooldown', 'MaterializedPassthrough']

Output = Dict[str, Any]


class MaterializedPassthrough(NodeRef, ABC):
    """Forward one default input and materialize it as this node's output."""

    def _write_output(self, output_dir: str, out: Output) -> Output:
        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='json',
        )
        out_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        out['metadata'] = str(out_path)
        return out

    def _forward_output(self, output_dir: str, upstream: Output) -> Output:
        out: Output = dict(upstream)
        for key in ('metadata', 'id', 'node', 'op', 'from'):
            out.pop(key, None)
        out['ok'] = True
        out['node'] = self.op
        out['id'] = self.id
        return self._write_output(output_dir, out)

    @abstractmethod
    def _missing_input(self, output_dir: str, *, upstream_none: bool) -> Output:
        """Handle a missing input according to the concrete node's policy."""

    def _invalid_input(self, input_keys: list[str]) -> Exception:
        return RuntimeError(
            f'{self.__class__.__name__} {self.id!r} expects exactly one input '
            f"under key 'default', got keys {input_keys}"
        )

    def run(
        self,
        output_dir: str,
        input: Optional[Dict[str, Output]] = None,
    ) -> Output:
        if not input:
            return self._missing_input(output_dir, upstream_none=False)
        if len(input) != 1 or 'default' not in input:
            raise self._invalid_input(sorted(input))

        upstream = input['default']
        if upstream is None:
            return self._missing_input(output_dir, upstream_none=True)
        return self._forward_output(output_dir, upstream)


class CudaCooldown(MaterializedPassthrough):
    """Recycle the process worker before forwarding the default input.

    Cooldown is handled by ProcessNodeExecutor. With an in-process runner,
    this node forwards its input and materializes a JSON output card so that
    downstream nodes can reload it from the DAG cache. No input produces an
    empty output.
    """

    @property
    def forces_cuda_cooldown(self) -> bool:
        return True

    def _missing_input(self, output_dir: str, *, upstream_none: bool) -> Output:
        return {}

    def _invalid_input(self, input_keys: list[str]) -> Exception:
        return ValueError('CudaCooldown accepts only the default input')
