from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.io import write_json
from morphalo.nodes.image_output import ImageOutputMixin

Output = Dict[str, Any]


def _input_index(input_id: str) -> int:
    """
    Parse an ``image:N`` input id and return ``N``.

    ``ImageMerge`` uses numeric input ids so merged images can be reconstructed
    in declaration order regardless of dictionary ordering.
    """
    prefix = 'image:'
    if not input_id.startswith(prefix):
        raise RuntimeError(
            f'ImageMerge input id must start with {prefix!r}, got {input_id!r}'
        )
    try:
        return int(input_id[len(prefix):])
    except ValueError as exc:
        raise RuntimeError(
            f'ImageMerge input id must end with an integer, got {input_id!r}'
        ) from exc


def _extract_images(node_id: str, input_id: str, payload: Output) -> list[str]:
    """
    Normalize one upstream payload to a non-empty list of image paths.

    Upstream nodes may expose a single image with ``image`` or a batch with
    ``images``. Missing or empty image payloads are treated as runtime errors
    because ``+`` is only meaningful for image-carrying outputs.
    """
    if not isinstance(payload, dict):
        raise RuntimeError(
            f'ImageMerge {node_id!r} input {input_id!r} expected a dict payload, '
            f'got {type(payload).__name__}'
        )

    if 'images' in payload:
        images = payload['images']
        if not isinstance(images, list) or not images:
            raise RuntimeError(
                f'ImageMerge {node_id!r} input {input_id!r} has invalid '
                f"'images' payload: expected a non-empty list"
            )
        return [str(image) for image in images]

    if 'image' in payload:
        image = payload['image']
        if image in (None, ''):
            raise RuntimeError(
                f'ImageMerge {node_id!r} input {input_id!r} has empty '
                f"'image' payload"
            )
        return [str(image)]

    raise RuntimeError(
        f'ImageMerge {node_id!r} input {input_id!r} has no image/images payload'
    )


@dataclass(kw_only=True)
class ImageMerge(ImageOutputMixin, NodeRef):
    """
    Runtime pass-through node that merges upstream image payloads.

    Each incoming edge should use an input id of the form ``image:N``. At run
    time, the node reads ``image`` or ``images`` from each upstream payload and
    emits one combined ``images`` list preserving input order.
    """

    source_names: Tuple[str, ...] = field(default_factory=tuple)

    def image(self, index: int) -> AttachmentSink:
        """
        Return the attachment sink for one ordered image-merge input.

        Parameters
        ----------
        index : int
            Zero-based position of the upstream payload in the merged image
            list. The corresponding edge input id is ``image:{index}``.

        Returns
        -------
        AttachmentSink
            Sink used by the DSL to wire ``source >> merge.image(index)``.
        """
        if index < 0:
            raise ValueError(f'ImageMerge image index must be >= 0, got {index}')
        return AttachmentSink(
            name=f'{self.id}.image_{index}',
            target=self,
            input_id=f'image:{index}',
        )

    def run(
        self,
        output_dir: str,
        input: Optional[Dict[str, Output]] = None,
    ) -> Output:
        if not input:
            raise RuntimeError(f'ImageMerge {self.id!r} received no input.')

        indexed_inputs = sorted(
            ((_input_index(input_id), input_id, payload)
             for input_id, payload in input.items()),
            key=lambda item: item[0],
        )

        expected = list(range(len(indexed_inputs)))
        found = [idx for idx, _, _ in indexed_inputs]
        if found != expected:
            raise RuntimeError(
                f'ImageMerge {self.id!r} expected contiguous image inputs '
                f'{expected}, got {found}'
            )

        images: list[str] = []
        for _, input_id, payload in indexed_inputs:
            images.extend(_extract_images(self.id, input_id, payload))

        out: Output = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'images': images,
        }
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
