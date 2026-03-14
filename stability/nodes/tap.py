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

    Parameters
    ----------
    strict : bool, default=True
        If True, the node fails when no upstream payload is available.
        If False, missing input produces a minimal materialized payload.
    """

    strict: bool = True

    def _write_output(self, output_dir: str, out: Output) -> Output:
        """
        Materialize the output payload to the node JSON artifact.

        Parameters
        ----------
        output_dir : str
            Directory where the node output is stored.
        out : Output
            Output payload to materialize.

        Returns
        -------
        Output
            The same payload enriched with the metadata file path.
        """
        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='json',
        )

        meta_path = write_json(out_path, out)
        out['metadata'] = str(meta_path)
        return out

    def run(
        self,
        output_dir: str,
        input: Optional[Dict[str, Output]] = None,
    ) -> Output:
        """
        Execute the checkpoint node.

        Parameters
        ----------
        output_dir : str
            Directory where the node JSON artifact is written.
        input : dict[str, Output] | None, optional
            Upstream payload mapping. When present, exactly one entry under
            key 'default' is expected.

        Returns
        -------
        Output
            The forwarded upstream payload with Tap metadata, or a minimal
            materialized payload when running in non-strict mode without
            upstream input.
        """
        if not input:
            if self.strict:
                raise RuntimeError(
                    f'Tap {self.id!r} received no input.'
                )
            return self._write_output(
                output_dir,
                {
                    'ok': True,
                    'node': self.op,
                    'id': self.id,
                },
            )

        if len(input) != 1 or 'default' not in input:
            raise RuntimeError(
                f'Tap {self.id!r} expects exactly one input under key '
                f"'default', got keys {sorted(input)}"
            )

        upstream = input['default']
        if upstream is None:
            if self.strict:
                raise RuntimeError(
                    f'Tap {self.id!r} received None upstream output.'
                )
            return self._write_output(
                output_dir,
                {
                    'ok': True,
                    'node': self.op,
                    'id': self.id,
                },
            )

        out: Output = dict(upstream)

        for k in ('metadata', 'id', 'node', 'op', 'from'):
            out.pop(k, None)

        out['ok'] = True
        out['node'] = self.op
        out['id'] = self.id

        return self._write_output(output_dir, out)
