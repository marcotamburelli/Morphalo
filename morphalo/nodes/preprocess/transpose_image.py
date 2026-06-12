from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

TransposeDirection = Literal['clockwise', 'anti_clockwise']


@dataclass
class Config:
    direction: TransposeDirection


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    direction = str(params.get('direction', 'clockwise')).lower()
    if direction not in ('clockwise', 'anti_clockwise'):
        raise ValueError(
            f"'{node_id}': invalid direction={direction!r} "
            "(expected 'clockwise' or 'anti_clockwise')"
        )

    return Config(direction=direction)  # type: ignore[arg-type]


def _transpose_method(direction: TransposeDirection) -> Image.Transpose:
    if direction == 'clockwise':
        return Image.Transpose.ROTATE_270
    if direction == 'anti_clockwise':
        return Image.Transpose.ROTATE_90

    raise ValueError(f'invalid direction={direction!r}')


@dataclass
class TransposeImage(NodeRef):
    """
    Rotate an input image by one quarter turn without resampling.

    ``TransposeImage`` is a purely geometric preprocessing node for discrete
    90-degree image transposition. It does not run model inference, analyze image
    content, or perform arbitrary-angle rotation. The node reads a single
    upstream image, rotates the whole image either clockwise or anti-clockwise,
    and writes the transformed image as a PNG.

    This node is intended for workflows where image orientation should be changed
    explicitly as a DAG step. Since transpose operations are not commutative with
    flips or axis-specific resizing, ordering is controlled by the DAG wiring:
    chain ``TransposeImage``, ``FlipImage``, and ``ResizeImage`` in the sequence
    you want to apply.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default input
        using ``input['default']['image']`` or ``input['default']['path']``.

    spec : dict or str or Path, optional
        Node specification, resolved via ``resolve_spec``.

        Expected structure:

        ``params`` : dict
            ``direction`` : {'clockwise', 'anti_clockwise'}, optional
                Quarter-turn direction. Default is ``'clockwise'``.

                - ``'clockwise'`` rotates the image 90 degrees clockwise.
                - ``'anti_clockwise'`` rotates the image 90 degrees
                  anti-clockwise.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the transposed image.

        ``input_size`` : list[int]
            Original image size as ``[width, height]``.

        ``output_size`` : list[int]
            Transposed output image size as ``[width, height]``.

        ``transpose`` : dict
            Transpose metadata, including the resolved direction.

    Notes
    -----
    - The transform uses PIL transpose constants rather than arbitrary-angle
      rotation, so no interpolation is introduced.
    - The output image is saved as PNG and preserves the input image mode.
    - This node is deterministic and has no model dependencies.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )

        img = Image.open(img_path)
        input_width, input_height = img.size
        transposed = img.transpose(_transpose_method(cfg.direction))
        out_width, out_height = transposed.size

        out_dir = Path(output_dir)
        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        transposed.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'image': str(out_path),
            'input_size': [int(input_width), int(input_height)],
            'output_size': [int(out_width), int(out_height)],
            'transpose': {
                'direction': cfg.direction,
            },
            'params': {
                'direction': cfg.direction,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
