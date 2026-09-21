from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.io import write_json
from morphalo.nodes.prompt_output import PromptOutputMixin

Output = Dict[str, Any]

_PROMPT_FIELDS = ('prompt', 'prompt_2', 'negative_prompt', 'negative_prompt_2')


def _input_index(input_id: str) -> int:
    """
    Parse a ``prompt:N`` input id and return ``N``.

    ``PromptMerge`` uses numeric input ids so merged prompt bundles can be
    reconstructed in declaration order regardless of dictionary ordering.
    """
    prefix = 'prompt:'
    if not input_id.startswith(prefix):
        raise RuntimeError(
            f'PromptMerge input id must start with {prefix!r}, got {input_id!r}'
        )
    try:
        return int(input_id[len(prefix):])
    except ValueError as exc:
        raise RuntimeError(
            f'PromptMerge input id must end with an integer, got {input_id!r}'
        ) from exc


def _prompt_part(
    node_id: str,
    input_id: str,
    payload: Output,
    field: str,
) -> str:
    """
    Normalize one prompt field from an upstream prompt payload.

    Missing and empty fields are treated as absent. Present fields must already
    be normalized strings because ``PromptMerge`` operates on prompt JSON cards,
    not raw HOCON prompt specifications.
    """
    if field not in payload or payload[field] in (None, ''):
        return ''

    value = payload[field]
    if not isinstance(value, str):
        raise RuntimeError(
            f'PromptMerge {node_id!r} input {input_id!r} field {field!r} '
            f'expected a string, got {type(value).__name__}'
        )
    return value.strip()


def _extract_prompt(node_id: str, input_id: str, payload: Output) -> Dict[str, str]:
    """
    Normalize one upstream payload to canonical prompt fields.

    Upstream nodes may expose any subset of ``prompt``, ``prompt_2``,
    ``negative_prompt`` and ``negative_prompt_2``. Payloads with no prompt
    fields are treated as runtime errors because ``+`` is only meaningful for
    prompt-carrying outputs.
    """
    if not isinstance(payload, dict):
        raise RuntimeError(
            f'PromptMerge {node_id!r} input {input_id!r} expected a dict '
            f'payload, got {type(payload).__name__}'
        )

    parts = {
        field: _prompt_part(node_id, input_id, payload, field)
        for field in _PROMPT_FIELDS
    }
    if not any(parts.values()):
        raise RuntimeError(
            f'PromptMerge {node_id!r} input {input_id!r} has no prompt fields'
        )
    return parts


def _join_prompt_parts(parts: list[str], *, joiner: str) -> str:
    """
    Join non-empty prompt parts using the field-appropriate separator.
    """
    return joiner.join(part for part in parts if part).strip()


@dataclass(kw_only=True)
class PromptMerge(PromptOutputMixin, NodeRef):
    """
    Runtime pass-through node that concatenates upstream prompt payloads.

    Each incoming edge should use an input id of the form ``prompt:N``. At run
    time, the node reads canonical prompt fields from each upstream JSON card
    and emits one combined prompt bundle preserving input order.

    Positive prompt channels (``prompt`` and ``prompt_2``) are concatenated
    with newlines. Negative prompt channels (``negative_prompt`` and
    ``negative_prompt_2``) are concatenated with ``, ``.
    """

    source_names: Tuple[str, ...] = field(default_factory=tuple)

    def prompt(self, index: int) -> AttachmentSink:
        """
        Return the attachment sink for one ordered prompt-merge input.

        Parameters
        ----------
        index : int
            Zero-based position of the upstream payload in the merged prompt
            bundle. The corresponding edge input id is ``prompt:{index}``.

        Returns
        -------
        AttachmentSink
            Sink used by the DSL to wire ``source >> merge.prompt(index)``.
        """
        if index < 0:
            raise ValueError(f'PromptMerge prompt index must be >= 0, got {index}')
        return AttachmentSink(
            name=f'{self.id}.prompt_{index}',
            target=self,
            input_id=f'prompt:{index}',
        )

    def run(
        self,
        output_dir: str,
        input: Optional[Dict[str, Output]] = None,
    ) -> Output:
        if not input:
            raise RuntimeError(f'PromptMerge {self.id!r} received no input.')

        indexed_inputs = sorted(
            ((_input_index(input_id), input_id, payload)
             for input_id, payload in input.items()),
            key=lambda item: item[0],
        )

        expected = list(range(len(indexed_inputs)))
        found = [idx for idx, _, _ in indexed_inputs]
        if found != expected:
            raise RuntimeError(
                f'PromptMerge {self.id!r} expected contiguous prompt inputs '
                f'{expected}, got {found}'
            )

        collected = {field: [] for field in _PROMPT_FIELDS}
        for _, input_id, payload in indexed_inputs:
            parts = _extract_prompt(self.id, input_id, payload)
            for field in _PROMPT_FIELDS:
                collected[field].append(parts[field])

        prompt = _join_prompt_parts(collected['prompt'], joiner='\n')
        prompt_2 = _join_prompt_parts(collected['prompt_2'], joiner='\n')
        negative_prompt = _join_prompt_parts(
            collected['negative_prompt'],
            joiner=', ',
        )
        negative_prompt_2 = _join_prompt_parts(
            collected['negative_prompt_2'],
            joiner=', ',
        )

        out: Output = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
        }
        if prompt_2:
            out['prompt_2'] = prompt_2
        if negative_prompt_2:
            out['negative_prompt_2'] = negative_prompt_2
        if self.source_names:
            out['sources'] = list(self.source_names)

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='json',
        )
        meta_path = write_json(out_path, out)
        out['metadata'] = str(meta_path)
        return out
