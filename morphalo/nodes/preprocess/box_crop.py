from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

BBoxFormat = Literal['xyxy', 'xywh', 'xyl']


@dataclass
class Config:
    bbox_format: BBoxFormat
    bbox: tuple[int, ...]


def _read_cfg(spec: dict, node_id: str) -> Config:
    params = spec.get('params', {})

    bbox_format = str(params.get('bbox_format', 'xyxy')).lower()
    if bbox_format not in ('xyxy', 'xywh', 'xyl'):
        raise ValueError(
            f"'{node_id}': invalid bbox_format={bbox_format!r} "
            "(expected 'xyxy', 'xywh', or 'xyl')"
        )

    bbox_value = params.get('bbox')
    if bbox_value is None:
        raise ValueError(f"'{node_id}': missing params.bbox")

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

    bbox = tuple(int(v) for v in bbox_value)

    return Config(
        bbox_format=bbox_format,  # type: ignore[arg-type]
        bbox=bbox,
    )


def _bbox_to_xyxy(
    bbox: tuple[int, ...],
    *,
    bbox_format: BBoxFormat,
) -> tuple[int, int, int, int]:
    """
    Convert a supported bbox representation to end-exclusive xyxy coordinates.

    Parameters
    ----------
    bbox : tuple[int, ...]
        Bounding box values in the format specified by ``bbox_format``.
    bbox_format : {'xyxy', 'xywh', 'xyl'}
        Input bbox format.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box as ``(x1, y1, x2, y2)`` with end-exclusive max coordinates.
    """
    if bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        return x1, y1, x2, y2

    if bbox_format == 'xywh':
        x, y, bw, bh = bbox
        return x, y, x + bw, y + bh

    if bbox_format == 'xyl':
        x, y, side = bbox
        return x, y, x + side, y + side

    raise ValueError(f'invalid bbox_format={bbox_format!r}')


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


@dataclass
class BoxCrop(NodeRef):
    """
    Crop an exact rectangular region from an input image using manual coordinates.

    ``BoxCrop`` is a purely geometric preprocessing node. It does not run object
    detection, segmentation, pose estimation, or any semantic image analysis.
    The caller provides an explicit bounding box, and the node crops that region
    from the source image.

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

            ``bbox`` : list[int] or tuple[int, ...]
                Manual bounding box values in the selected format.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the cropped image.

        ``original_bbox_xyxy`` : list[int]
            User-provided bbox normalized to xyxy coordinates before clamping.

        ``bbox_xyxy`` : list[int]
            Effective clamped bbox used to crop the image.

        ``crop`` : dict
            Reinsertion metadata computed from the effective clamped crop:

            - ``anchor_xy``: local anchor inside the cropped image.
            - ``position``: source-image position where ``anchor_xy`` should be
              placed when reconstructing the original geometry.
            - ``bbox_size``: width and height of the effective crop.
            - ``bbox_xyxy``: effective clamped bbox.

    Notes
    -----
    - All crop metadata is computed after clamping the bbox to the source image
      bounds. This guarantees that ``crop.anchor_xy``, ``crop.position`` and
      ``crop.bbox_size`` are coherent with the image actually written by the
      node.
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
        width, height = img.size

        orig_x1, orig_y1, orig_x2, orig_y2 = _bbox_to_xyxy(
            cfg.bbox,
            bbox_format=cfg.bbox_format,
        )

        out_x1, out_y1, out_x2, out_y2 = _clamp_bbox_xyxy(
            orig_x1,
            orig_y1,
            orig_x2,
            orig_y2,
            width=width,
            height=height,
        )

        cropped = img.crop((out_x1, out_y1, out_x2, out_y2))

        out_dir = Path(output_dir)

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        cropped.save(out_path)

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
            'bbox_format': cfg.bbox_format,
            'bbox': list(cfg.bbox),
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
                'bbox': list(cfg.bbox),
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
