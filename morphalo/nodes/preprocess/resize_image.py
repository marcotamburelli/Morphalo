from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import SizeExpr, resolve_size_expr
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


ResizeSize = tuple[Optional[SizeExpr], Optional[SizeExpr]]


@dataclass
class Config:
    size: ResizeSize


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    size_value = params.get('size')
    if size_value is None:
        raise ValueError(f"'{node_id}': missing params.size")

    if not isinstance(size_value, (list, tuple)):
        raise TypeError(
            f"'{node_id}': params.size must be a list or tuple, "
            f'got {type(size_value).__name__}'
        )

    if len(size_value) != 2:
        raise ValueError(
            f"'{node_id}': params.size must contain exactly 2 values, "
            f'got {len(size_value)}'
        )

    width_expr, height_expr = size_value
    if width_expr is None and height_expr is None:
        raise ValueError(f"'{node_id}': params.size cannot be [None, None]")

    return Config(size=(width_expr, height_expr))


def _resolve_resize_size(
    size: ResizeSize,
    *,
    input_width: int,
    input_height: int,
) -> tuple[int, int]:
    width_expr, height_expr = size

    if width_expr is not None and height_expr is not None:
        return (
            resolve_size_expr(width_expr, reference=input_width),
            resolve_size_expr(height_expr, reference=input_height),
        )

    if width_expr is not None:
        out_width = resolve_size_expr(width_expr, reference=input_width)
        scale = float(out_width) / float(input_width)
        out_height = max(1, int(round(input_height * scale)))
        return out_width, out_height

    if height_expr is not None:
        out_height = resolve_size_expr(height_expr, reference=input_height)
        scale = float(out_height) / float(input_height)
        out_width = max(1, int(round(input_width * scale)))
        return out_width, out_height

    raise ValueError('resize size cannot be [None, None]')


def _json_size(size: ResizeSize) -> list[Optional[Union[int, str]]]:
    return [size[0], size[1]]


@dataclass
class ResizeImage(NodeRef):
    """
    Resize an input image to an explicit target size.

    ``ResizeImage`` is a purely geometric preprocessing node. It does not run
    model inference, analyze image content, or change the image mode. The node
    reads a single upstream image, resolves a target size, resamples the whole
    image, and writes the resized image as a PNG.

    This node is intended for workflows where image geometry should be changed
    explicitly as a DAG step, for example:

    - downscale a generated image before using it as an auxiliary input;
    - normalize an image to a required width or height while preserving aspect
      ratio;
    - intentionally stretch an image by providing both target dimensions.

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
            ``size`` : list[int | str | None] or tuple[int | str | None, ...]
                Target size as ``[width, height]``.

                Each non-``None`` value may be:

                - ``int``: absolute pixels;
                - ``'<number>px'``: absolute pixels;
                - ``'<number>%'``: percentage of the current upstream image
                  dimension on the same axis.

                If both dimensions are provided, the image is resized exactly to
                that size and the original aspect ratio may change. If one
                dimension is ``None``, the missing dimension is computed from the
                original aspect ratio. ``[None, None]`` is invalid.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the resized image.

        ``input_size`` : list[int]
            Original image size as ``[width, height]``.

        ``output_size`` : list[int]
            Resolved output image size as ``[width, height]``.

        ``resize`` : dict
            Resize metadata, including the user-provided size expression and the
            resolved pixel size.

    Notes
    -----
    - Percentage sizes are resolved against the current upstream image, not
      against a canvas or generation target.
    - The output image is saved as PNG and preserves the input image mode.
    - Resampling uses PIL's Lanczos filter.
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
        out_width, out_height = _resolve_resize_size(
            cfg.size,
            input_width=input_width,
            input_height=input_height,
        )

        resized = img.resize((out_width, out_height), resample=Image.LANCZOS)

        out_dir = Path(output_dir)
        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        resized.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'image': str(out_path),
            'input_size': [int(input_width), int(input_height)],
            'output_size': [int(out_width), int(out_height)],
            'resize': {
                'size': _json_size(cfg.size),
                'resolved_size': [int(out_width), int(out_height)],
            },
            'params': {
                'size': _json_size(cfg.size),
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
