from dataclasses import dataclass, field
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import SizeExpr, resolve_size_expr, validate_size_expr
from morphalo.nodes.preprocess.utils.mask_ops import prepare_output_mask
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

BBoxFormat = Literal['xyxy', 'xywh', 'xyl']


@dataclass
class Config:
    bbox_format: BBoxFormat
    bbox: Optional[tuple[SizeExpr, ...]]
    scale_transform_to_image: bool = False
    mode: str = 'default'
    dilate_radius: int = 0
    close_radius: int = 0
    smoothing_radius: int = 0


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': invalid mode={mode!r}")

    mask_params = {
        name: int(params.get(name, 0))
        for name in ('dilate_radius', 'close_radius', 'smoothing_radius')
    }
    for name, value in mask_params.items():
        if value < 0:
            raise ValueError(f"'{node_id}': {name} must be >= 0, got {value}")

    bbox_format = str(params.get('bbox_format', 'xyxy')).lower()
    if bbox_format not in ('xyxy', 'xywh', 'xyl'):
        raise ValueError(
            f"'{node_id}': invalid bbox_format={bbox_format!r} "
            "(expected 'xyxy', 'xywh', or 'xyl')"
        )

    bbox_value = params.get('bbox')
    if bbox_value is None:
        return Config(
            bbox_format=bbox_format,  # type: ignore[arg-type]
            bbox=None,
            scale_transform_to_image=bool(params.get('scale_transform_to_image', False)),
            mode=mode,
            **mask_params,
        )

    if not isinstance(bbox_value, (list, tuple)):
        raise TypeError(
            f"'{node_id}': params.bbox must be a list or tuple, "
            f'got {type(bbox_value).__name__}'
        )

    expected_len = 3 if bbox_format == 'xyl' else 4
    if len(bbox_value) != expected_len:
        raise ValueError(
            f"'{node_id}': bbox_format={bbox_format!r} expects "
            f'{expected_len} values, got {len(bbox_value)}'
        )

    bbox_values = []
    for value in bbox_value:
        try:
            parsed = value if isinstance(value, str) else int(value)
            validate_size_expr(parsed, allow_unitless=True, allow_negative=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"'{node_id}': invalid params.bbox coordinate {value!r}; "
                'expected pixels or a string ending in px or %'
            ) from exc
        bbox_values.append(parsed)
    bbox = tuple(bbox_values)

    return Config(
        bbox_format=bbox_format,  # type: ignore[arg-type]
        bbox=bbox,
        scale_transform_to_image=bool(params.get('scale_transform_to_image', False)),
        mode=mode,
        **mask_params,
    )


def _bbox_to_xyxy(
    bbox: tuple[SizeExpr, ...],
    *,
    bbox_format: BBoxFormat,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a supported bbox representation to end-exclusive xyxy coordinates.

    Parameters
    ----------
    bbox : tuple[int or str, ...]
        Pixel or percentage values in the format specified by ``bbox_format``.
    bbox_format : {'xyxy', 'xywh', 'xyl'}
        Input bbox format.
    width, height : int
        Input image dimensions used to resolve percentages. A percentage side
        length in ``xyl`` refers to the longer dimension.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box as ``(x1, y1, x2, y2)`` with end-exclusive max coordinates.
    """
    dimensions = (width, height, max(width, height)) if bbox_format == 'xyl' else (
        width, height, width, height,
    )
    values = tuple(
        resolve_size_expr(
            value, reference=dimension, min_size=None,
            allow_unitless=True, allow_negative=True, round_result=False,
        )
        for value, dimension in zip(bbox, dimensions)
    )
    if any(not isfinite(value) for value in values):
        raise ValueError('bbox coordinates must be finite')
    if bbox_format == 'xyxy':
        x1, y1, x2, y2 = values
    elif bbox_format == 'xywh':
        x1, y1, bw, bh = values
        x2, y2 = x1 + bw, y1 + bh
    elif bbox_format == 'xyl':
        x1, y1, side = values
        x2, y2 = x1 + side, y1 + side
    else:
        raise ValueError(f'invalid bbox_format={bbox_format!r}')

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid bbox: {(x1, y1, x2, y2)!r}')
    return floor(x1), floor(y1), ceil(x2), ceil(y2)


def _clamp_bbox_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Clamp an xyxy bbox to image bounds.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        End-exclusive bbox coordinates.
    width : int
        Source image width.
    height : int
        Source image height.

    Returns
    -------
    tuple[int, int, int, int]
        Clamped bbox in end-exclusive xyxy coordinates.

    Raises
    ------
    ValueError
        If the clamped bbox is empty or invalid.
    """
    out_x1 = max(0, min(width, int(x1)))
    out_y1 = max(0, min(height, int(y1)))
    out_x2 = max(0, min(width, int(x2)))
    out_y2 = max(0, min(height, int(y2)))

    if out_x2 <= out_x1 or out_y2 <= out_y1:
        raise ValueError(
            'Invalid bbox after clamping: '
            f'{(out_x1, out_y1, out_x2, out_y2)!r}'
        )

    return out_x1, out_y1, out_x2, out_y2


def _read_transform_bbox_xyxy(
    transform: Optional[dict[str, Any]],
    *,
    node_id: str,
) -> Optional[tuple[int, int, int, int]]:
    """
    Read ``crop.bbox_xyxy`` from an optional transform input.

    ``BoxCrop`` consumes the concrete crop bounds rather than the higher-level
    placement metadata used by nodes such as ``ResizeImage`` or ``ImageStack``.
    """
    if transform is None:
        return None

    if not isinstance(transform, dict):
        raise TypeError(
            f'{node_id}: transform must be a dict, '
            f'got {type(transform).__name__}.'
        )

    crop = transform.get('crop')
    if crop is None:
        return None

    if not isinstance(crop, dict):
        raise TypeError(
            f'{node_id}: transform.crop must be a dict, '
            f'got {type(crop).__name__}.'
        )

    bbox = crop.get('bbox_xyxy')
    if bbox is None:
        return None

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f'{node_id}: transform.crop.bbox_xyxy must be a 4-item list '
            f'or tuple, got {bbox!r}.'
        )

    return tuple(int(v) for v in bbox)


@dataclass
class BoxCrop(NodeRef):
    """
    Crop a rectangular region or generate its full-frame inpaint mask.

    ``BoxCrop`` is a purely geometric preprocessing node. It does not run object
    detection, segmentation, pose estimation, or any semantic image analysis.
    The caller provides an explicit bounding box either through ``params.bbox``
    or through the optional ``transform`` input.

    This node is intended for local refinement workflows where a small region of
    an already generated image must be processed at a larger resolution, for
    example:

    - crop a small problematic hand, weapon handle, emblem, eye, ornament, or
      mechanical detail;
    - run ``Img2Img`` on the cropped region with ``params.long_side`` set to a
      larger value;
    - reinsert the refined crop into the original image using ``ImageStack`` and
      the emitted ``crop`` metadata.

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
            ``mode`` : {'default', 'mask', 'negative-mask'}, optional
                Default is ``'default'``: save the rectangular image crop.
                ``'mask'`` saves a full-frame grayscale mask, white inside the
                selected rectangle and black outside. ``'negative-mask'``
                saves the inverse. White marks the region to repaint.

            ``bbox_format`` : {'xyxy', 'xywh', 'xyl'}, optional
                Format of ``bbox``. Default is ``'xyxy'``.

                Supported formats are:

                - ``'xyxy'``:
                  ``bbox = [x1, y1, x2, y2]``.
                  Coordinates are interpreted as end-exclusive crop bounds.

                - ``'xywh'``:
                  ``bbox = [x, y, width, height]``.
                  ``x`` and ``y`` are the top-left corner.

                - ``'xyl'``:
                  ``bbox = [x, y, side]``.
                  Square crop using top-left corner plus side length.

            ``bbox`` : list[int or str] or tuple, optional
                Manual bounding box values in the selected format. This is
                required unless a transform input provides ``crop.bbox_xyxy``.
                Numbers are pixels; strings may specify pixels (``'12px'``)
                or percentages (``'75%'``), mixed within the same bbox.
                X coordinates and widths refer to the input image width;
                Y coordinates and heights refer to its height. In ``xyl``,
                a percentage side length refers to the longer image dimension.
                Minimum bounds are rounded down and maximum bounds up after
                resolving units and computing the rectangle, then clamped.
                For example, ``bbox_format='xywh'`` with
                ``bbox=['0px', '75%', '100%', '25%']`` crops the bottom quarter.

            ``scale_transform_to_image`` : bool, optional
                Convert a bbox received through the ``transform`` input from
                the coordinate space described by ``transform.input_size`` to
                the current image's coordinate space. Default is ``False``.

                Set this in ``spec['params']`` when the bbox was obtained from
                a resized version of the full image. The transform must include
                ``input_size = [width, height]`` with two positive dimensions
                describing the image on which the bbox was measured, not the
                size of the cropped region.

                X coordinates are multiplied by current width / source width;
                Y coordinates by current height / source height. Minimum bounds
                are rounded down and maximum bounds up, then clamped to the
                current image. Only coordinates are scaled; the image itself
                is not resized. This option does not affect ``params.bbox``.

            ``dilate_radius`` : int, optional
                Non-negative dilation radius in pixels for mask outputs.
                Default: ``0``. Expands the selected rectangle before inversion.

            ``close_radius`` : int, optional
                Non-negative morphological closing radius in pixels for mask
                outputs. Default: ``0``.

            ``smoothing_radius`` : int, optional
                Non-negative Gaussian smoothing radius in pixels for mask
                outputs. Default: ``0``.

                Mask processing order is closing, dilation, smoothing, then
                optional inversion for ``mode='negative-mask'``. These three
                parameters do not affect ``mode='default'`` or bbox metadata.

    Inputs
    ------
    default : dict, optional
        Upstream image payload. Required only when ``path`` is omitted.

    transform : dict, optional
        Upstream spatial metadata. When present and containing
        ``crop.bbox_xyxy``, those coordinates override ``params.bbox``. This is
        useful for cropping one image with the geometry emitted by another node,
        for example cropping a full-frame mask to match a ``SubjectCrop`` ROI.

        ``crop.bbox_xyxy`` contains end-exclusive ``[x1, y1, x2, y2]`` bounds;
        ``params.bbox_format`` does not apply to these coordinates. By default,
        they are used directly as pixels in the current image. When
        ``params.scale_transform_to_image=True``, ``input_size`` is also required
        and identifies their source coordinate space.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the cropped image or full-frame mask, depending on ``mode``.

        ``mode`` : str
            Output mode: ``'default'``, ``'mask'`` or ``'negative-mask'``.

        ``input_size``, ``output_size`` : list[int]
            Input and output dimensions as ``[width, height]``. Mask outputs
            preserve the input dimensions.

        ``original_bbox_xyxy`` : list[int]
            Bbox in current-image xyxy coordinates before clamping, after any
            transform scaling.

        ``bbox_xyxy`` : list[int]
            Effective clamped bbox used to crop the image.

        ``crop`` : dict
            Geometry of the selected rectangle before mask processing. In
            default mode this is crop reinsertion metadata; a full-frame mask
            is already aligned to the input image and needs no reinsertion:

            - ``anchor_xy``: local anchor inside the cropped image.
            - ``position``: source-image position where ``anchor_xy`` should be
              placed when reconstructing the original geometry.
            - ``bbox_size``: width and height of the effective crop.
            - ``bbox_xyxy``: effective clamped bbox.

    Examples
    --------
    Generate a mask for inpainting the bottom quarter of the input image::

        region_mask = BoxCrop(
            name='bottom_mask',
            spec={'params': {
                'bbox_format': 'xywh',
                'bbox': ['0px', '75%', '100%', '25%'],
                'mode': 'mask',
                'smoothing_radius': 8,
            }},
        )
        original_image >> region_mask
        original_image >> inpaint
        region_mask >> inpaint.mask()

    Suppose ``original.png`` is 1600 x 1200 pixels, and a region was detected on
    an 800 x 600 resized copy. The detected bbox is ``[100, 50, 300, 250]`` in
    that copy's coordinates. To crop the same region from the original image::

        crop = BoxCrop(
            name='crop_original',
            path='original.png',
            spec={'params': {'scale_transform_to_image': True}},
        )
        result = crop.run(
            output_dir='outputs',
            input={
                'transform': {
                    'input_size': [800, 600],
                    'crop': {'bbox_xyxy': [100, 50, 300, 250]},
                },
            },
        )

    Both scale factors are 2, so ``result['bbox_xyxy']`` is
    ``[200, 100, 600, 500]`` and the saved crop is 400 x 400 pixels. With the
    default ``False``, the bbox would instead remain ``[100, 50, 300, 250]``
    in original-image pixels, selecting a different, smaller region.

    In a DAG, wire the full image to the default input and the node that produced
    the proxy's crop metadata to the transform attachment::

        original_image >> crop
        proxy_crop >> crop.transform()

    Here ``original_image`` supplies the full-size image and ``proxy_crop``
    supplies ``crop.bbox_xyxy`` plus ``input_size`` for the resized full image.
    ``path`` can be omitted when the image is supplied through this wiring.

    Notes
    -----
    - If both ``params.bbox`` and ``transform.crop.bbox_xyxy`` are provided, the
      transform bbox wins.
    - Transform scaling assumes both images show the same full frame, related
      only by resizing. It does not compensate for padding, cropping, rotation,
      or translation in the proxy image.
    - All crop metadata is computed after clamping the bbox to the source image
      bounds. This guarantees that ``crop.anchor_xy``, ``crop.position`` and
      ``crop.bbox_size`` are coherent with the image actually written by the
      node.
    - Output is saved as PNG. Default mode preserves the input image mode;
      mask modes produce full-frame grayscale ``L`` images.
    - This node is deterministic and has no model dependencies.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def transform(self) -> AttachmentSink:
        """
        Declare optional crop metadata used to resolve the crop rectangle.

        The upstream node should produce a ``crop`` dictionary containing
        ``bbox_xyxy``. When this input is wired, it overrides ``params.bbox`` and
        lets ``BoxCrop`` crop one image using the geometry emitted by another
        node, for example a full-frame mask using the bbox from ``SubjectCrop``.

        Set ``spec['params']['scale_transform_to_image'] = True`` to convert
        coordinates from a resized full-image proxy. The upstream payload must
        then also contain ``input_size = [width, height]`` for that proxy.

        Returns
        -------
        AttachmentSink
            Sink bound to this node with ``input_id='transform'``.
        """
        return AttachmentSink(
            name=f'box_crop_transform:{self.id}',
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
        transform_bbox = _read_transform_bbox_xyxy(
            None if input is None else input.get('transform'),
            node_id=node_id,
        )

        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )

        img = Image.open(img_path)
        width, height = img.size

        if transform_bbox is not None:
            orig_x1, orig_y1, orig_x2, orig_y2 = transform_bbox
            if cfg.scale_transform_to_image:
                source_size = input['transform'].get('input_size')
                if (
                    not isinstance(source_size, (list, tuple))
                    or len(source_size) != 2
                    or any(not isinstance(v, (int, float)) or v <= 0 for v in source_size)
                ):
                    raise ValueError(
                        f'{node_id}: scale_transform_to_image requires '
                        'transform.input_size with two positive dimensions'
                    )
                source_width, source_height = source_size
                orig_x1 = floor(orig_x1 * width / source_width)
                orig_y1 = floor(orig_y1 * height / source_height)
                orig_x2 = ceil(orig_x2 * width / source_width)
                orig_y2 = ceil(orig_y2 * height / source_height)
            bbox_source = 'transform'
            bbox_format = 'xyxy'
            bbox_value = [orig_x1, orig_y1, orig_x2, orig_y2]
        elif cfg.bbox is not None:
            orig_x1, orig_y1, orig_x2, orig_y2 = _bbox_to_xyxy(
                cfg.bbox,
                bbox_format=cfg.bbox_format,
                width=width,
                height=height,
            )
            bbox_source = 'params'
            bbox_format = cfg.bbox_format
            bbox_value = list(cfg.bbox)
        else:
            raise ValueError(
                f"'{node_id}': missing params.bbox and no transform "
                'crop.bbox_xyxy was provided'
            )

        out_x1, out_y1, out_x2, out_y2 = _clamp_bbox_xyxy(
            orig_x1,
            orig_y1,
            orig_x2,
            orig_y2,
            width=width,
            height=height,
        )

        if cfg.mode == 'default':
            output_image = img.crop((out_x1, out_y1, out_x2, out_y2))
        else:
            selected_mask = np.zeros((height, width), dtype=bool)
            selected_mask[out_y1:out_y2, out_x1:out_x2] = True
            out_mask = prepare_output_mask(
                selected_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )
            if cfg.mode == 'negative-mask':
                out_mask = 255 - out_mask
            output_image = Image.fromarray(out_mask, mode='L')

        out_dir = Path(output_dir)

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        output_image.save(out_path)

        b_width = int(out_x2 - out_x1)
        b_height = int(out_y2 - out_y1)
        anchor_x = int((out_x1 + out_x2) // 2)
        anchor_y = int((out_y1 + out_y2) // 2)
        local_anchor_x = int(anchor_x - out_x1)
        local_anchor_y = int(anchor_y - out_y1)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'image': str(out_path),
            'mode': cfg.mode,
            'input_size': [width, height],
            'output_size': [output_image.width, output_image.height],
            'bbox_source': bbox_source,
            'bbox_format': bbox_format,
            'bbox': bbox_value,
            'original_bbox_xyxy': [
                int(orig_x1),
                int(orig_y1),
                int(orig_x2),
                int(orig_y2),
            ],
            'bbox_xyxy': [
                int(out_x1),
                int(out_y1),
                int(out_x2),
                int(out_y2),
            ],
            'crop': {
                'anchor_xy': [local_anchor_x, local_anchor_y],
                'position': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
                'bbox_xyxy': [
                    int(out_x1),
                    int(out_y1),
                    int(out_x2),
                    int(out_y2),
                ],
            },
            'params': {
                'bbox_format': cfg.bbox_format,
                'bbox': None if cfg.bbox is None else list(cfg.bbox),
                'mode': cfg.mode,
                'scale_transform_to_image': cfg.scale_transform_to_image,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'smoothing_radius': cfg.smoothing_radius,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
