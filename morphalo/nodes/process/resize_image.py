from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (SizeSpec, SpatialTransform,
                                             merge_spatial_transform,
                                             read_spatial_transform,
                                             resolve_size_expr,
                                             validate_size_expr)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


@dataclass
class Config:
    size: Optional[SizeSpec]
    long_side: Optional[Union[int, str]]
    short_side: Optional[Union[int, str]]


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    size_value = params.get('size')
    long_side = params.get('long_side')
    short_side = params.get('short_side')

    provided = [
        name for name, value in (
            ('params.size', size_value),
            ('params.long_side', long_side),
            ('params.short_side', short_side),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError(
            f"'{node_id}': use only one resize mode, got {provided}"
        )

    if long_side is not None:
        validate_size_expr(long_side)
    if short_side is not None:
        validate_size_expr(short_side)

    return Config(
        size=None if size_value is None else _read_size_value(
            size_value,
            node_id=node_id,
            name='params.size',
        ),
        long_side=long_side,
        short_side=short_side,
    )


def _read_size_value(value: Any, *, node_id: str, name: str) -> SizeSpec:
    """
    Validate a two-axis resize size expression.

    ``ResizeImage`` accepts the same shape from either ``params.size`` or
    transform metadata ``bbox_size``. Keeping the validation in one helper makes
    both sources behave identically.
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"'{node_id}': {name} must be a list or tuple, "
            f'got {type(value).__name__}'
        )

    if len(value) != 2:
        raise ValueError(
            f"'{node_id}': {name} must contain exactly 2 values, "
            f'got {len(value)}'
        )

    width_expr, height_expr = value
    if width_expr is None and height_expr is None:
        raise ValueError(f"'{node_id}': {name} cannot be [None, None]")

    if width_expr is not None:
        validate_size_expr(width_expr)
    if height_expr is not None:
        validate_size_expr(height_expr)

    return (width_expr, height_expr)


def _resolve_resize_size(
    size: SizeSpec,
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


def _resolve_side_resize_size(
    side: Union[int, str],
    *,
    mode: Literal['long', 'short'],
    input_width: int,
    input_height: int,
) -> tuple[int, int]:
    reference = (
        max(input_width, input_height)
        if mode == 'long'
        else min(input_width, input_height)
    )
    target_side = resolve_size_expr(side, reference=reference)
    scale = float(target_side) / float(reference)
    out_width = max(1, int(round(input_width * scale)))
    out_height = max(1, int(round(input_height * scale)))
    return out_width, out_height


def _json_size(size: SizeSpec) -> list[Optional[Union[int, str]]]:
    return [size[0], size[1]]


@dataclass
class ResizeImage(NodeRef):
    """
    Resize an input image to an explicit target size.

    ``ResizeImage`` is a purely geometric image-processing node. It does not run
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
            ``size`` : list[int | str | None] or tuple[int | str | None, ...], optional
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

                This value may be omitted when a transform input provides
                ``crop.bbox_size`` or ``placement.bbox_size``.

            ``long_side`` : int or str, optional
                Resize proportionally so the longest side matches this value.

            ``short_side`` : int or str, optional
                Resize proportionally so the shortest side matches this value.

                ``size``, ``long_side`` and ``short_side`` are mutually exclusive.

    Transform input
    ---------------
    ``ResizeImage`` exposes :meth:`transform`, an optional metadata input that
    accepts the same ``crop`` / ``placement`` blocks used by ``ImageStack``.
    When the connected metadata contains ``bbox_size``, that size has
    precedence over ``params.size``.

    This is useful for workflows where an image is generated or refined at a
    different resolution and then must be normalized back to the size of an
    upstream crop before being placed by ``ImageStack``.

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
    - Transform ``bbox_size`` values use the same resolution rules as
      ``params.size``.
    - The output image is saved as PNG and preserves the input image mode.
    - Resampling uses PIL's Lanczos filter.
    - This node is deterministic and has no model dependencies.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def transform(self) -> AttachmentSink:
        """
        Declare optional crop / placement metadata used to resolve output size.

        The upstream node should produce either a ``crop`` or ``placement``
        dictionary. If that dictionary contains ``bbox_size``, the value
        overrides ``params.size`` for this resize operation.

        Returns
        -------
        AttachmentSink
            Sink bound to this node with ``input_id='transform'``.
        """
        return AttachmentSink(
            name=f'resize_transform:{self.id}',
            target=self,
            input_id='transform',
        )

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)
        default_spatial = SpatialTransform(bbox_size=cfg.size)
        transform_spatial = read_spatial_transform(
            None if input is None else input.get('transform'),
            node_id=node_id,
        )
        spatial = merge_spatial_transform(default_spatial, transform_spatial)

        size = None if spatial is None else spatial.bbox_size
        side_mode = None
        side_value = None
        if size is None and transform_spatial is None:
            if cfg.long_side is not None:
                side_mode = 'long'
                side_value = cfg.long_side
            elif cfg.short_side is not None:
                side_mode = 'short'
                side_value = cfg.short_side

        size_source = (
            'transform'
            if transform_spatial is not None
            and transform_spatial.bbox_size is not None
            else f'params.{side_mode}_side'
            if side_mode is not None
            else 'params.size'
        )

        if size is None and side_value is None:
            raise ValueError(
                f"'{node_id}': missing params.size, params.long_side or "
                "params.short_side, and no transform bbox_size was provided"
            )

        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )

        img = Image.open(img_path)
        input_width, input_height = img.size
        if side_value is not None:
            out_width, out_height = _resolve_side_resize_size(
                side_value,
                mode=side_mode,
                input_width=input_width,
                input_height=input_height,
            )
        else:
            out_width, out_height = _resolve_resize_size(
                size,
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
                'size': None if size is None else _json_size(size),
                'long_side': cfg.long_side,
                'short_side': cfg.short_side,
                'source': size_source,
                'resolved_size': [int(out_width), int(out_height)],
            },
            'params': {
                'size': None if cfg.size is None else _json_size(cfg.size),
                'long_side': cfg.long_side,
                'short_side': cfg.short_side,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
