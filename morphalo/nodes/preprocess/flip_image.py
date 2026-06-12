from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

FlipAxis = Literal['horizontal', 'vertical']


@dataclass
class Config:
    axis: FlipAxis


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    axis = str(params.get('axis', 'horizontal')).lower()
    if axis not in ('horizontal', 'vertical'):
        raise ValueError(
            f"'{node_id}': invalid axis={axis!r} "
            "(expected 'horizontal' or 'vertical')"
        )

    return Config(axis=axis)  # type: ignore[arg-type]


def _transpose_method(axis: FlipAxis) -> Image.Transpose:
    if axis == 'horizontal':
        return Image.Transpose.FLIP_LEFT_RIGHT
    if axis == 'vertical':
        return Image.Transpose.FLIP_TOP_BOTTOM

    raise ValueError(f'invalid axis={axis!r}')


@dataclass
class FlipImage(NodeRef):
    """
    Mirror an input image along one image axis.

    ``FlipImage`` is a purely geometric preprocessing node for horizontal or
    vertical mirroring. It does not run model inference, analyze image content,
    or resize the image. The node reads a single upstream image, flips the whole
    image along the requested axis, and writes the transformed image as a PNG.

    This node is intended for workflows where mirroring should be an explicit
    DAG step. Since flips are not commutative with quarter-turn transposition or
    axis-specific resizing, ordering is controlled by the DAG wiring: chain
    ``FlipImage``, ``TransposeImage``, and ``ResizeImage`` in the sequence you
    want to apply.

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
            ``axis`` : {'horizontal', 'vertical'}, optional
                Flip axis. Default is ``'horizontal'``.

                - ``'horizontal'`` mirrors the image left-to-right.
                - ``'vertical'`` mirrors the image top-to-bottom.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the flipped image.

        ``input_size`` : list[int]
            Original image size as ``[width, height]``.

        ``output_size`` : list[int]
            Output image size as ``[width, height]``. Flips preserve size.

        ``flip`` : dict
            Flip metadata, including the resolved axis.

    Notes
    -----
    - The transform uses PIL transpose constants, so no interpolation is
      introduced.
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
        flipped = img.transpose(_transpose_method(cfg.axis))
        out_width, out_height = flipped.size

        out_dir = Path(output_dir)
        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        flipped.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'image': str(out_path),
            'input_size': [int(input_width), int(input_height)],
            'output_size': [int(out_width), int(out_height)],
            'flip': {
                'axis': cfg.axis,
            },
            'params': {
                'axis': cfg.axis,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
