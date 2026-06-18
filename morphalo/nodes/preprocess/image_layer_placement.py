from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, cast

import numpy as np
from PIL import Image

from morphalo.cache.models import get_mediapipe_pose_landmarker
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (PositionSpec, SizeSpec,
                                             validate_size_expr)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.human import mp_pose_landmarks_xy

AnchorName = Literal[
    'top-left',
    'top-center',
    'top-right',
    'center-left',
    'center',
    'center-right',
    'bottom-left',
    'bottom-center',
    'bottom-right',
    'alpha-center',
    'alpha-bbox-center',
    'alpha-top-center',
    'alpha-bottom-center',
    'body-center',
    'torso-center',
    'shoulder-center',
    'hips-center',
    'head-center',
]

HUMAN_ANCHORS = {
    'body-center',
    'torso-center',
    'shoulder-center',
    'hips-center',
    'head-center',
}


@dataclass(frozen=True)
class Config:
    """
    Validated runtime configuration for ``ImageLayerPlacement``.

    ``anchor`` selects the local point computed from the input image.
    ``position`` and ``bbox_size`` are optional because they intentionally
    override only the matching ``ImageStack.image`` parameters when present.
    ``device`` is kept for MediaPipe cache-key consistency.
    ``pose_landmarker_task`` is required only for human landmark anchors.
    """

    device: str
    anchor: AnchorName
    position: Optional[PositionSpec]
    bbox_size: Optional[SizeSpec]
    pose_landmarker_task: Optional[str]


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Read and validate the node specification.

    Parameters
    ----------
    spec : dict
        Resolved node specification.

    node_id : str
        Node identifier used in validation errors.

    Returns
    -------
    Config
        Validated placement configuration.
    """
    params = spec.get('params', {})
    model = spec.get('model', {})

    device = str(model.get('device', 'cuda'))

    anchor = str(params.get('anchor', 'center')).lower().strip()
    valid: set[AnchorName] = {
        'top-left',
        'top-center',
        'top-right',
        'center-left',
        'center',
        'center-right',
        'bottom-left',
        'bottom-center',
        'bottom-right',
        'alpha-center',
        'alpha-bbox-center',
        'alpha-top-center',
        'alpha-bottom-center',
        'body-center',
        'torso-center',
        'shoulder-center',
        'hips-center',
        'head-center',
    }

    if anchor not in valid:
        raise ValueError(
            f"'{node_id}': invalid anchor={anchor!r}; expected one of "
            f'{sorted(valid)!r}'
        )

    anchor = cast(AnchorName, anchor)

    position = params.get('position')
    if position is not None:
        position = _read_position(position, node_id=node_id)

    bbox_size = params.get('bbox_size', params.get('resize'))
    if bbox_size is not None:
        bbox_size = _read_bbox_size(bbox_size, node_id=node_id)

    pose_landmarker_task = model.get('pose_landmarker_task')
    if anchor in HUMAN_ANCHORS:
        if not pose_landmarker_task:
            raise ValueError(
                f"'{node_id}': anchor={anchor!r} requires "
                "'model.pose_landmarker_task' (MediaPipe .task path)"
            )
        pose_landmarker_task = str(pose_landmarker_task)

    return Config(
        device=device,
        anchor=anchor,
        position=position,
        bbox_size=bbox_size,
        pose_landmarker_task=pose_landmarker_task,
    )


def _read_position(value: Any, *, node_id: str) -> PositionSpec:
    """
    Validate a placement target accepted by ``ImageStack``.

    A string value is forwarded as an ImageStack anchor name such as
    ``'center'`` or ``'bottom-right'``. A two-item list/tuple is interpreted as
    explicit canvas coordinates for the layer anchor; each component may be an
    integer, ``'<number>px'``, ``'<number>%'`` or ``None``.
    """
    if isinstance(value, str):
        return value

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"'{node_id}': params.position must be a string anchor or a "
            f'2-item list/tuple, got {value!r}'
        )

    x, y = value
    for component in (x, y):
        if component is not None:
            validate_size_expr(component)

    return (x, y)


def _read_bbox_size(value: Any, *, node_id: str) -> SizeSpec:
    """
    Validate an optional target layer size.

    ``ImageLayerPlacement`` intentionally accepts only tuple-style resize values
    because it does not know the eventual ImageStack canvas. Stack-relative modes
    such as ``'fit'`` and ``'cover'`` belong on ``ImageStack.image`` itself.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"'{node_id}': params.bbox_size must be a 2-item list/tuple, "
            f'got {value!r}'
        )

    width, height = value
    if width is None and height is None:
        raise ValueError(
            f"'{node_id}': params.bbox_size cannot be [None, None]"
        )

    if width is not None:
        validate_size_expr(width)
    if height is not None:
        validate_size_expr(height)

    return (width, height)


def _geometry_anchor(
    anchor: AnchorName,
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    """
    Resolve a named geometric anchor inside an image rectangle.

    Coordinates are expressed in the local image frame used by PIL: ``(0, 0)``
    is the top-left image corner and ``(width, height)`` is the bottom-right
    outer corner. This convention matches ImageStack placement, where an anchor
    can sit on the outside edge of a layer.
    """
    if anchor == 'center':
        return float(width) / 2.0, float(height) / 2.0

    xs = {
        'left': 0.0,
        'center': float(width) / 2.0,
        'right': float(width),
    }
    ys = {
        'top': 0.0,
        'center': float(height) / 2.0,
        'bottom': float(height),
    }

    vertical, horizontal = anchor.split('-')
    return xs[horizontal], ys[vertical]


def _alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """
    Return the tight bbox of visible pixels in an image alpha channel.

    Any alpha value greater than zero is considered visible. The result uses
    end-exclusive ``xyxy`` coordinates.
    """
    alpha = np.asarray(image.convert('RGBA'))[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError('image has no visible alpha pixels')
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _alpha_anchor(anchor: AnchorName, image: Image.Image) -> tuple[float, float]:
    """
    Resolve an alpha-derived local anchor.

    Supported modes are:

    - ``alpha-bbox-center``: center of the tight visible alpha bbox.
    - ``alpha-top-center``: midpoint of the top edge of that bbox.
    - ``alpha-bottom-center``: midpoint of the bottom edge of that bbox.
    - ``alpha-center``: alpha-weighted centroid of visible pixels.

    The weighted centroid uses pixel centers, i.e. pixel ``(x, y)`` contributes
    at ``(x + 0.5, y + 0.5)``.
    """
    x1, y1, x2, y2 = _alpha_bbox(image)

    if anchor == 'alpha-bbox-center':
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
    if anchor == 'alpha-top-center':
        return (float(x1 + x2) / 2.0, float(y1))
    if anchor == 'alpha-bottom-center':
        return (float(x1 + x2) / 2.0, float(y2))

    alpha = np.asarray(image.convert('RGBA'))[:, :, 3].astype(np.float64)
    total = float(alpha.sum())
    if total <= 0.0:
        raise ValueError('image has no visible alpha pixels')

    yy, xx = np.indices(alpha.shape, dtype=np.float64)
    return (
        float(((xx + 0.5) * alpha).sum() / total),
        float(((yy + 0.5) * alpha).sum() / total),
    )


def _require_pose_point(
    pose_xy: np.ndarray,
    index: int,
    *,
    name: str,
) -> np.ndarray:
    """
    Return a required MediaPipe pose landmark as a float coordinate.

    ``mp_pose_landmarks_xy`` encodes missing or weak landmarks as ``(-1, -1)``.
    Human semantic anchors are intentionally strict: if a required point is not
    available, the placement node fails instead of silently falling back to a
    different anchor.
    """
    point = pose_xy[index]
    if point[0] < 0 or point[1] < 0:
        raise RuntimeError(f'MediaPipe pose landmark {name!r} is missing.')
    return point.astype(np.float64)


def _mean_pose_points(
    pose_xy: np.ndarray,
    indices: list[int],
    *,
    names: list[str],
) -> np.ndarray:
    """
    Average a required group of MediaPipe pose landmarks.

    All listed landmarks must be present. This helper is used for anchors whose
    meaning depends on symmetric body pairs, such as shoulders and hips.
    """
    pts = [
        _require_pose_point(pose_xy, index, name=name)
        for index, name in zip(indices, names)
    ]
    return np.mean(np.stack(pts, axis=0), axis=0)


def _head_center_from_pose(pose_xy: np.ndarray) -> np.ndarray:
    """
    Estimate the local center of the head from MediaPipe pose head landmarks.

    MediaPipe Pose exposes lightweight face/head landmarks at indices 0..10
    (nose, eyes, ears and mouth corners). The centroid uses the valid landmarks
    in that group, but requires the nose and at least three valid head points so
    the result is anchored to a real detected head instead of a sparse accident.
    """
    _require_pose_point(pose_xy, 0, name='nose')
    head_pts = []
    for index in range(0, 11):
        point = pose_xy[index]
        if point[0] >= 0 and point[1] >= 0:
            head_pts.append(point.astype(np.float64))

    if len(head_pts) < 3:
        raise RuntimeError(
            'MediaPipe pose landmarks are insufficient for head-center '
            f'(got {len(head_pts)} valid head landmarks).'
        )

    return np.mean(np.stack(head_pts, axis=0), axis=0)


def _human_anchor(anchor: AnchorName, pose_xy: np.ndarray) -> tuple[float, float]:
    """
    Resolve a semantic human anchor from MediaPipe pose landmarks.

    Supported anchors:

    - ``shoulder-center``: midpoint between left and right shoulders.
    - ``hips-center``: midpoint between left and right hips.
    - ``torso-center``: midpoint between shoulder-center and hips-center.
    - ``body-center``: torso point biased slightly toward the hips, useful as a
      stable placement pivot for full-body layers.
    - ``head-center``: centroid of valid pose head landmarks.
    """
    shoulder_center = None
    hip_center = None

    if anchor in {'body-center', 'torso-center', 'shoulder-center'}:
        shoulder_center = _mean_pose_points(
            pose_xy,
            [11, 12],
            names=['left_shoulder', 'right_shoulder'],
        )

    if anchor in {'body-center', 'torso-center', 'hips-center'}:
        hip_center = _mean_pose_points(
            pose_xy,
            [23, 24],
            names=['left_hip', 'right_hip'],
        )

    if anchor == 'shoulder-center':
        point = shoulder_center
    elif anchor == 'hips-center':
        point = hip_center
    elif anchor == 'torso-center':
        point = (shoulder_center + hip_center) / 2.0
    elif anchor == 'body-center':
        point = (0.4 * shoulder_center) + (0.6 * hip_center)
    elif anchor == 'head-center':
        point = _head_center_from_pose(pose_xy)
    else:
        raise ValueError(f'Unsupported human anchor {anchor!r}.')

    return float(point[0]), float(point[1])


def _resolve_anchor(
    anchor: AnchorName,
    image: Image.Image,
    *,
    device: str = 'cuda',
    pose_landmarker_task: Optional[str] = None,
) -> tuple[float, float]:
    """
    Resolve a geometric, alpha-derived or human-landmark anchor for ``image``.
    """
    if anchor.startswith('alpha-'):
        return _alpha_anchor(anchor, image)
    if anchor in HUMAN_ANCHORS:
        if not pose_landmarker_task:
            raise ValueError(
                f"anchor={anchor!r} requires model.pose_landmarker_task"
            )
        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=pose_landmarker_task,
            device=device,
        )
        image_rgb = np.asarray(image.convert('RGB'))
        pose_xy = mp_pose_landmarks_xy(
            image_rgb,
            pose_landmarker=pose_landmarker,
        )
        return _human_anchor(anchor, pose_xy)
    return _geometry_anchor(anchor, width=image.width, height=image.height)


def _json_position(
    position: Optional[PositionSpec],
) -> Optional[Union[str, list[Any]]]:
    """
    Convert an internal position value to a JSON-serializable representation.
    """
    if position is None or isinstance(position, str):
        return position
    return [position[0], position[1]]


def _json_size(size: Optional[SizeSpec]) -> Optional[list[Any]]:
    """
    Convert an internal size value to a JSON-serializable representation.
    """
    if size is None:
        return None
    return [size[0], size[1]]


@dataclass
class ImageLayerPlacement(NodeRef):
    """
    Produce ImageStack placement metadata for an existing image layer.

    ``ImageLayerPlacement`` is a metadata-only preprocessing node. It reads an
    input image, computes a local anchor point inside that image, and emits a
    ``placement`` block compatible with ``ImageStack.image(...).transform()``.

    The node does not write or modify an image. It is useful when an image
    already exists and should be positioned by a meaningful local point such as
    ``'top-center'``, ``'bottom-center'`` or ``'alpha-bbox-center'``.

    Placement contract
    ------------------
    The node emits a ``placement`` metadata block. ``ImageStack`` consumes this
    block when it is wired into ``ImageStack.image(...).transform()``.

    The core fields are:

    ``anchor_xy`` : list[float]
        Local point inside the attached layer image. This is the pivot used for
        placement, rotation and resize propagation.

    ``position`` : str or list, optional
        Canvas position where ``anchor_xy`` should land. If omitted,
        ``ImageStack`` keeps the position declared on ``image(...)``.

    ``bbox_size`` : list, optional
        Target layer size. If omitted, ``ImageStack`` keeps the resize declared
        on ``image(...)``.

    Geometric anchors
    -----------------
    Geometric anchors are resolved against the whole input image:

    - ``'top-left'``, ``'top-center'``, ``'top-right'``
    - ``'center-left'``, ``'center'``, ``'center-right'``
    - ``'bottom-left'``, ``'bottom-center'``, ``'bottom-right'``

    Only the documented canonical anchor names are accepted. Short aliases such as
    ``'top'`` or ``'left'`` are intentionally not supported, so configuration files
    remain explicit and predictable.

    Alpha anchors
    -------------
    Alpha anchors use the visible pixels of the input image:

    - ``'alpha-bbox-center'``:
      center of the tight bbox containing pixels with alpha greater than zero;
    - ``'alpha-top-center'``:
      midpoint of the top edge of that visible bbox;
    - ``'alpha-bottom-center'``:
      midpoint of the bottom edge of that visible bbox;
    - ``'alpha-center'``:
      alpha-weighted centroid of visible pixels.

    These anchors are useful for transparent layers, for example a glyph,
    decal, logo, or cutout inside a larger transparent canvas.

    Human landmark anchors
    ----------------------
    Human anchors use MediaPipe Pose landmarks and require
    ``model.pose_landmarker_task``. They are deliberately strict: if MediaPipe
    cannot detect the required landmarks, the node raises an error instead of
    falling back to a geometric anchor.

    ``model.device`` defaults to ``'cuda'`` for consistency with the other
    MediaPipe users in the project. In this node it is passed to the model cache
    key; MediaPipe itself is not using it here as a CUDA execution selector.

    Supported human anchors are:

    - ``'shoulder-center'``:
      midpoint between left and right shoulders;
    - ``'hips-center'``:
      midpoint between left and right hips;
    - ``'torso-center'``:
      midpoint between shoulder-center and hips-center;
    - ``'body-center'``:
      torso point biased slightly toward the hips, useful as a stable pivot for
      full-body layers;
    - ``'head-center'``:
      centroid of detected MediaPipe head landmarks.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default
        input using ``input['default']['image']`` or ``input['default']['path']``.

    spec : dict or str or Path, optional
        Node specification, resolved via ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``device`` : str, optional
                Logical device string used for cache-key consistency with other
                MediaPipe calls. Default: ``'cuda'``.

            ``pose_landmarker_task`` : str, optional
                MediaPipe Pose Landmarker ``.task`` path. Required only when
                ``params.anchor`` is one of the human landmark anchors.

        ``params`` : dict
            ``anchor`` : str, optional
                Local anchor to compute. Must be one of the documented canonical
                anchor names. Default: ``'center'``.

            ``position`` : str or list[int | str | None, int | str | None], optional
                Canvas placement target to forward to ImageStack. If omitted,
                ImageStack uses the layer's declared ``position``.

            ``bbox_size`` : list[int | str | None, int | str | None], optional
                Target layer size to forward to ImageStack. If omitted,
                ImageStack uses the layer's declared ``resize``. ``'fit'`` and
                ``'cover'`` are intentionally not supported here.

            ``resize`` : list[int | str | None, int | str | None], optional
                Alias for ``bbox_size``.

    Outputs
    -------
    dict
        Metadata dictionary, also written as a JSON sidecar.

        ``placement`` contains:

        - ``anchor_xy``: resolved local anchor.
        - ``position``: present only when configured.
        - ``bbox_size``: present only when configured.

    Notes
    -----
    - Geometric and alpha anchors are deterministic and have no model
      dependencies.
    - Human anchors load MediaPipe Pose only when requested.
    - It does not produce an ``image`` output; wire the original image to
      ``ImageStack.image(...)`` and wire this node to ``.transform()``.
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

        with Image.open(img_path) as image:
            image_rgba = image.convert('RGBA')
            width, height = image_rgba.size
            anchor_xy = _resolve_anchor(
                cfg.anchor,
                image_rgba,
                device=cfg.device,
                pose_landmarker_task=cfg.pose_landmarker_task,
            )

        placement = {
            'anchor_xy': [float(anchor_xy[0]), float(anchor_xy[1])],
        }
        if cfg.position is not None:
            placement['position'] = _json_position(cfg.position)
        if cfg.bbox_size is not None:
            placement['bbox_size'] = _json_size(cfg.bbox_size)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'input_size': [int(width), int(height)],
            'placement': placement,
            'params': {
                'anchor': cfg.anchor,
                'position': _json_position(cfg.position),
                'bbox_size': _json_size(cfg.bbox_size),
                **({} if cfg.pose_landmarker_task is None else {
                    'device': cfg.device,
                    'pose_landmarker_task': cfg.pose_landmarker_task,
                }),
            },
        }

        meta_path = make_node_output_path(
            out_dir=output_dir,
            node_id=node_id,
            ext='json',
        )
        meta_path = write_json_sidecar(meta_path, out)
        out['metadata'] = str(meta_path)
        return out
