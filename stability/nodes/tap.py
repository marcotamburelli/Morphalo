from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from stability.core.paths import make_node_output_path
from stability.dag import NodeRef
from stability.nodes.common.io import write_json

Output = Dict[str, Any]


@dataclass
class Tap(NodeRef):
    """
    Minimal pass-through checkpoint node.

    Tap forwards the single upstream output under input_id 'default',
    rewrites node metadata to itself, and materializes the result to JSON.

    Contract
    --------
    - Exactly one upstream input is expected.
    - That input must be under key 'default'.
    - Lateral wiring is not supported.

    Raises
    ------
    RuntimeError

    """

    def run(
        self,
        output_dir: str,
        input: Optional[Dict[str, Output]] = None
    ) -> Output:

        if not input:
            raise RuntimeError(
                f'Tap {self.id!r} received no input.'
            )

        if len(input) != 1 or 'default' not in input:
            raise RuntimeError(
                f'Tap {self.id!r} expects exactly one input under key '
                f"'default', got keys {sorted(input)}"
            )

        upstream = input['default']
        if upstream is None:
            raise RuntimeError(
                f'Tap {self.id!r} received None upstream output.'
            )

        # Shallow copy to avoid mutating upstream object.
        out: Output = dict(upstream)

        # Remove previous node metadata if present.
        for k in ('metadata', 'id', 'node', 'op', 'from'):
            out.pop(k, None)

        # Stamp Tap metadata.
        out['ok'] = True
        out['node'] = self.op
        out['id'] = self.id

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='json',
        )

        meta_path = write_json(out_path, out)
        out['metadata'] = str(meta_path)

        return out
