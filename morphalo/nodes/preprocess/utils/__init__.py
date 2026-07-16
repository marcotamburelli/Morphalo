import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Union

import cv2
import numpy as np

SizeExpr = int | str
SizeSpec = tuple[Optional[SizeExpr], Optional[SizeExpr]]
PositionCoord = Optional[SizeExpr]
PositionSpec = Union[str, tuple[PositionCoord, PositionCoord]]
ResizeMode = SizeSpec | Literal['fit', 'cover'] | None

CropModeName = Literal['bbox', 'trim', 'full_frame']
SpatialTransformKind = Literal['crop', 'placement']
TargetSpec = str | list[str] | tuple[str, ...]


@dataclass(frozen=True)
class SpatialTransform:
    """
    Parsed crop or placement spatial transform metadata.

    The object represents the common spatial metadata contract shared by
    preprocessing nodes that consume ``crop`` or ``placement`` dictionaries. It
    only performs structural normalization; each consumer remains responsible
    for deciding which fields are required for its own operation.

    Parameters
    ----------
    kind : {'crop', 'placement'} or None, optional
        Metadata block that provided the values. ``None`` is useful for local
        defaults assembled by a consuming node.
    anchor_xy : tuple[float, float] or None, optional
        Local anchor point in the attached image/layer coordinate frame.
    position : str or tuple[Any, Any] or None, optional
        Canvas placement specification. Consumers resolve the concrete meaning.
    bbox_size : tuple[int | str | None, int | str | None] or None, optional
        Target size specification. Components follow the standard preprocessing
        size-expression convention and may be resolved by the consumer.
    """
    kind: Optional[SpatialTransformKind] = None
    anchor_xy: Optional[tuple[float, float]] = None
    position: Optional[PositionSpec] = None
    bbox_size: Optional[SizeSpec] = None


@dataclass(frozen=True)
class CropModeSpec:
    """
    Normalized crop-mode configuration.

    Parameters
    ----------
    mode : {'bbox', 'trim', 'full_frame'}
        Base crop mode.
    ratio : tuple[int, int] | None, optional
        Desired aspect ratio for bbox-guided crops, expressed as ``(w, h)``.

        This is only meaningful when ``mode == 'bbox'``.
        The ratio is a target, not a hard constraint: the final crop is expanded
        toward the requested ratio as much as possible while keeping the target
        fully inside the crop and staying within image bounds.
    raw : str
        Original user-provided crop-mode string, preserved for reporting.
    """
    mode: CropModeName
    ratio: Optional[tuple[int, int]] = None
    raw: str = 'trim'


def resolve_segment_target_labels(
    target: TargetSpec,
    *,
    labels: dict[str, int],
    composite_targets: dict[str, tuple[str, ...]],
    node_id: str,
    taxonomy_name: str,
) -> list[str]:
    """
    Resolve a semantic segmentation target into de-duplicated atomic labels.

    Semantic crop nodes commonly expose two kinds of public targets:

    - atomic labels, which map one-to-one to class ids emitted by a parser;
    - composite aliases, which describe a useful region as a stable ordered
      collection of atomic labels.

    This helper centralizes the validation and expansion logic shared by those
    nodes. ``target`` may be a single label or alias string, a list of strings,
    or a tuple of strings. Atomic labels pass through unchanged, composite
    aliases expand to their member labels, and duplicate atomic labels are
    removed while preserving first-seen order. Preserving order keeps metadata
    deterministic and makes downstream debug output easier to compare.

    Parameters
    ----------
    target : str | list[str] | tuple[str, ...]
        Raw user target specification from node parameters.

    labels : dict[str, int]
        Atomic taxonomy labels accepted by the node, keyed by public label name
        and valued by parser class id. Only keys are used by this function, but
        the full mapping is accepted so callers can pass their canonical label
        table directly.

    composite_targets : dict[str, tuple[str, ...]]
        Composite aliases accepted by the node. Each alias must expand to
        atomic label names present in ``labels``. Invalid composite definitions
        raise ``ValueError`` so taxonomy mistakes fail close to the caller.

    node_id : str
        Node identifier used to make validation errors actionable inside a DAG.

    taxonomy_name : str
        Human-readable taxonomy name included in validation errors, for example
        ``'FASHN'`` or ``'Sapiens2'``.

    Returns
    -------
    list[str]
        Stable, de-duplicated atomic label names resolved from ``target``.

    Raises
    ------
    ValueError
        If ``target`` has an unsupported type, a sequence is empty, any sequence
        item is not a string, a target name is unknown, or a composite alias
        references a label outside ``labels``.
    """
    valid_targets = tuple(labels.keys()) + tuple(composite_targets.keys())
    expected = ', '.join(repr(t) for t in valid_targets)

    if isinstance(target, str):
        items = [target]
    elif isinstance(target, (list, tuple)):
        if not target:
            raise ValueError(f"'{node_id}': target sequence cannot be empty")
        items = list(target)
    else:
        raise ValueError(
            f"'{node_id}': invalid target={target!r} "
            f"(expected a string, list of strings, or tuple of strings; "
            f"valid {taxonomy_name} values: {expected})"
        )

    resolved: list[str] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, str):
            raise ValueError(
                f"'{node_id}': invalid target item={item!r} "
                f"(expected a string; valid {taxonomy_name} values: {expected})"
            )

        if item in labels:
            expanded = (item,)
        elif item in composite_targets:
            expanded = composite_targets[item]
        else:
            raise ValueError(
                f"'{node_id}': invalid target={item!r} "
                f"(expected one of: {expected})"
            )

        for label in expanded:
            if label not in labels:
                raise ValueError(
                    f"'{node_id}': composite target={item!r} expands to "
                    f"unknown {taxonomy_name} label={label!r}"
                )
            if label not in seen:
                resolved.append(label)
                seen.add(label)

    return resolved


def positive_points_for_sam(
    *,
    xy: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_points: Optional[int] = None,
) -> tuple[Optional[list[list[float]]], Optional[list[int]]]:
    """
    Build positive SAM prompt points from valid landmarks inside a bbox.
    """
    x1, y1, x2, y2 = bbox
    points: list[list[float]] = []

    for px, py in xy[:, :2]:
        if px < 0 or py < 0:
            continue

        if not (x1 <= px < x2 and y1 <= py < y2):
            continue

        points.append([float(px), float(py)])

    if not points:
        return None, None

    if max_points is not None and max_points <= 0:
        raise ValueError(f'max_points must be > 0, got {max_points!r}.')

    if max_points is not None and len(points) > max_points:
        idxs = np.linspace(
            0,
            len(points) - 1,
            num=int(max_points),
            dtype=np.int64,
        )
        points = [points[int(i)] for i in idxs]

    labels = [1] * len(points)

    return points, labels


def parse_crop_mode(value: Any, *, node_id: str) -> CropModeSpec:
    s = str(value).strip().lower()

    if s in ('bbox', 'trim', 'full_frame'):
        return CropModeSpec(mode=s, ratio=None, raw=s)

    m = re.fullmatch(r'bbox\[(\d+):(\d+)\]', s)
    if m is None:
        raise ValueError(
            f"'{node_id}': invalid crop_mode={value!r} "
            "(expected 'bbox', 'bbox[w:h]', 'trim', or 'full_frame')"
        )

    rw = int(m.group(1))
    rh = int(m.group(2))

    if rw <= 0 or rh <= 0:
        raise ValueError(
            f"'{node_id}': invalid crop_mode={value!r} "
            '(ratio terms must be > 0)'
        )

    return CropModeSpec(mode='bbox', ratio=(rw, rh), raw=s)


def validate_size_expr(
    size_expr: SizeExpr,
    *,
    allow_unitless: bool = False,
    allow_negative: bool = False,
) -> None:
    if isinstance(size_expr, int):
        if not allow_negative and size_expr < 0:
            raise ValueError('size expression must be >= 0')
        return

    if not isinstance(size_expr, str):
        raise ValueError(
            f'invalid size expression type {type(size_expr).__name__}; '
            'expected int or str'
        )

    s = size_expr.strip().lower()
    sign = r'-?' if allow_negative else ''
    suffix = r'(px|%)?' if allow_unitless else r'(px|%)'
    pattern = rf'^{sign}\d+(\.\d+)?{suffix}$'

    if not re.match(pattern, s):
        raise ValueError(
            f'invalid size expression value "{size_expr}". '
            'Expected formats: int, "<number>px", "<number>%".'
        )


def read_spatial_transform(
    transform: Optional[dict[str, Any]],
    *,
    node_id: str,
    input_name: str = 'transform',
) -> Optional[SpatialTransform]:
    """
    Read crop or placement spatial transform metadata.

    Parameters
    ----------
    transform : dict or None
        Upstream transform metadata. The dictionary must contain at most one of
        ``'crop'`` or ``'placement'``.
    node_id : str
        Node identifier used in error messages.
    input_name : str, optional
        Input name used in error messages.

    Returns
    -------
    SpatialTransform or None
        Parsed spatial transform metadata, or ``None`` if no crop/placement
        metadata is available.

    Notes
    -----
    This function validates and normalizes the common metadata structure, but it
    does not enforce consumer-specific requirements. For example, ``ImageStack``
    can use ``anchor_xy`` when present, while ``ResizeImage`` only cares about
    ``bbox_size``.
    """
    if transform is None:
        return None

    if not isinstance(transform, dict):
        raise TypeError(
            f'{node_id}: {input_name} must be a dict, '
            f'got {type(transform).__name__}.'
        )

    transform_items = [
        (kind, transform[kind])
        for kind in ('crop', 'placement')
        if kind in transform
    ]

    if not transform_items:
        return None

    if len(transform_items) > 1:
        raise ValueError(
            f'{node_id}: {input_name} contains both crop and placement '
            'metadata; expected only one.'
        )

    kind, transform_prop = transform_items[0]

    if not isinstance(transform_prop, dict):
        raise TypeError(
            f'{node_id}: {input_name}.{kind} must be a dict, '
            f'got {type(transform_prop).__name__}.'
        )

    anchor_xy = None
    if 'anchor_xy' in transform_prop:
        raw_anchor_xy = transform_prop.get('anchor_xy')
        if raw_anchor_xy is None:
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.anchor_xy cannot be None. '
                'Omit the key if no anchor is provided.'
            )

        if (
            not isinstance(raw_anchor_xy, (list, tuple))
            or len(raw_anchor_xy) != 2
        ):
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.anchor_xy must be '
                f'a 2-item list or tuple, got {raw_anchor_xy!r}.'
            )

        anchor_xy = (
            float(raw_anchor_xy[0]),
            float(raw_anchor_xy[1]),
        )

    position = None
    if 'position' in transform_prop:
        raw_position = transform_prop.get('position')
        if raw_position is None:
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.position cannot be None. '
                'Omit the key to keep the node default position.'
            )

        if isinstance(raw_position, str):
            position = raw_position

        elif isinstance(raw_position, (list, tuple)) and len(raw_position) == 2:
            position = (
                raw_position[0],
                raw_position[1],
            )

        else:
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.position must be '
                f'a string anchor or a 2-item list/tuple, got {raw_position!r}.'
            )

    bbox_size = None
    if 'bbox_size' in transform_prop:
        raw_bbox_size = transform_prop.get('bbox_size')
        if raw_bbox_size is None:
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.bbox_size cannot be None. '
                'Omit the key to keep the node default size.'
            )

        if (
            not isinstance(raw_bbox_size, (list, tuple))
            or len(raw_bbox_size) != 2
        ):
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.bbox_size must be '
                f'a 2-item list or tuple, got {raw_bbox_size!r}.'
            )

        bw, bh = raw_bbox_size
        if bw is None and bh is None:
            raise ValueError(
                f'{node_id}: {input_name}.{kind}.bbox_size cannot be '
                '[None, None].'
            )

        if bw is not None:
            validate_size_expr(bw)
        if bh is not None:
            validate_size_expr(bh)

        bbox_size = (bw, bh)

    return SpatialTransform(
        kind=kind,
        anchor_xy=anchor_xy,
        position=position,
        bbox_size=bbox_size,
    )


def merge_spatial_transform(
    base: Optional[SpatialTransform],
    override: Optional[SpatialTransform],
) -> Optional[SpatialTransform]:
    """
    Merge two spatial transform objects using non-``None`` override values.

    ``base`` usually represents node-local defaults, while ``override`` usually
    comes from :func:`read_spatial_transform`. The returned object keeps every
    base value unless the override explicitly provides that field.
    """
    if base is None:
        return override
    if override is None:
        return base

    return SpatialTransform(
        kind=override.kind if override.kind is not None else base.kind,
        anchor_xy=(
            override.anchor_xy
            if override.anchor_xy is not None
            else base.anchor_xy
        ),
        position=(
            override.position
            if override.position is not None
            else base.position
        ),
        bbox_size=(
            override.bbox_size
            if override.bbox_size is not None
            else base.bbox_size
        ),
    )


def resolve_size_expr(
    size_expr: SizeExpr,
    *,
    max_size: Optional[int] = None,
    reference: Optional[int] = None,
    min_size: int = 1,
    allow_unitless: bool = False,
) -> int:
    """
    Resolve a pixel or percentage size expression to pixels.
    """
    ref = reference if reference is not None else max_size
    if ref is None:
        raise TypeError('resolve_size_expr requires max_size or reference')
    if ref < 0:
        raise ValueError(f'reference must be >= 0, got {ref}')

    validate_size_expr(size_expr, allow_unitless=allow_unitless)

    if isinstance(size_expr, int):
        return max(min_size, size_expr)

    s = size_expr.strip().lower()
    if s.endswith('px'):
        return max(min_size, int(round(float(s[:-2]))))
    if s.endswith('%'):
        pct = max(0.0, float(s[:-1])) / 100.0
        return max(min_size, int(round(ref * pct)))

    return max(min_size, int(round(float(s))))


def validate_percentage_size_expr(
    value: Optional[SizeExpr],
    *,
    node_id: str,
    name: str,
) -> Optional[str]:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"'{node_id}': invalid {name}={value!r}; "
            "expected None or a percentage string such as '5%'."
        )

    s = value.strip().lower()
    if not s.endswith('%'):
        raise ValueError(
            f"'{node_id}': invalid {name}={value!r}; "
            "expected None or a percentage string such as '5%'."
        )

    try:
        raw = float(s[:-1])
    except ValueError as exc:
        raise ValueError(
            f"'{node_id}': invalid {name}={value!r}; "
            "expected None or a percentage string such as '5%'."
        ) from exc

    if raw < 0:
        raise ValueError(
            f"'{node_id}': invalid {name}={value!r}; expected >= 0%."
        )

    return s


def expand_bbox_toward_ratio(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    full_w: int,
    full_h: int,
    ratio: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f'Invalid bbox: {(x1, y1, x2, y2)!r}')

    rw, rh = ratio
    target_ratio = float(rw) / float(rh)

    bw = int(x2 - x1)
    bh = int(y2 - y1)

    if bw <= 0 or bh <= 0:
        raise RuntimeError(f'Invalid bbox size: {(bw, bh)!r}')

    current_ratio = float(bw) / float(bh)

    if current_ratio >= target_ratio:
        crop_w = bw
        crop_h = int(math.ceil(float(crop_w) / target_ratio))
    else:
        crop_h = bh
        crop_w = int(math.ceil(float(crop_h) * target_ratio))

    crop_w = max(crop_w, bw)
    crop_h = max(crop_h, bh)

    crop_w = min(crop_w, full_w)
    crop_h = min(crop_h, full_h)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    out_x1 = int(math.floor(cx - crop_w / 2.0))
    out_y1 = int(math.floor(cy - crop_h / 2.0))
    out_x2 = out_x1 + crop_w
    out_y2 = out_y1 + crop_h

    if out_x1 < 0:
        out_x2 -= out_x1
        out_x1 = 0
    if out_x2 > full_w:
        shift = out_x2 - full_w
        out_x1 -= shift
        out_x2 = full_w

    if out_y1 < 0:
        out_y2 -= out_y1
        out_y1 = 0
    if out_y2 > full_h:
        shift = out_y2 - full_h
        out_y1 -= shift
        out_y2 = full_h

    out_x1 = max(0, out_x1)
    out_y1 = max(0, out_y1)
    out_x2 = min(full_w, out_x2)
    out_y2 = min(full_h, out_y2)

    if out_x1 > x1:
        needed = out_x1 - x1
        grow = min(needed, full_w - out_x2)
        out_x1 -= needed
        out_x2 += grow
        out_x1 = max(0, out_x1)
        out_x2 = min(full_w, out_x2)

    if out_x2 < x2:
        needed = x2 - out_x2
        grow = min(needed, out_x1)
        out_x2 += needed
        out_x1 -= grow
        out_x1 = max(0, out_x1)
        out_x2 = min(full_w, out_x2)

    if out_y1 > y1:
        needed = out_y1 - y1
        grow = min(needed, full_h - out_y2)
        out_y1 -= needed
        out_y2 += grow
        out_y1 = max(0, out_y1)
        out_y2 = min(full_h, out_y2)

    if out_y2 < y2:
        needed = y2 - out_y2
        grow = min(needed, out_y1)
        out_y2 += needed
        out_y1 -= grow
        out_y1 = max(0, out_y1)
        out_y2 = min(full_h, out_y2)

    if out_x2 <= out_x1 or out_y2 <= out_y1:
        raise RuntimeError(
            f'Failed to derive a valid expanded crop bbox from {(x1, y1, x2, y2)!r}.'
        )

    if not (out_x1 <= x1 and x2 <= out_x2 and out_y1 <= y1 and y2 <= out_y2):
        raise RuntimeError(
            'Expanded crop bbox does not fully contain the original bbox.'
        )

    return int(out_x1), int(out_y1), int(out_x2), int(out_y2)


def round_up(x: int, m: int) -> int:
    return int((x + m - 1) // m * m)


def resize_long_side_rgb(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if long_side is None or long_side <= 0:
        return rgb
    scale = float(long_side) / float(max(h, w))
    if scale == 1.0:
        return rgb
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    return cv2.resize(rgb, (nw, nh), interpolation=interp)


def fit_to_target_rgb(
    rgb: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    keep_aspect: bool = True,
    pad_to_multiple_of: int = 64,
) -> np.ndarray:
    if pad_to_multiple_of and pad_to_multiple_of > 1:
        out_w = round_up(out_w, pad_to_multiple_of)
        out_h = round_up(out_h, pad_to_multiple_of)

    if not keep_aspect:
        return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    h, w = rgb.shape[:2]
    scale = min(out_w / float(w), out_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized

    return canvas


def validate_min_component_area(
    value: Any,
    *,
    node_id: str,
    param_name: str = 'min_component_area',
) -> None:
    """
    Validate a connected-component area cleanup threshold.

    The shared preprocessing convention accepts either an absolute pixel area or
    a percentage string. Absolute values are interpreted as areas in pixels.
    Percentage strings, for example ``'1%'``, are validated here and resolved by
    :func:`resolve_min_component_area` against an image size later.

    Parameters
    ----------
    value : int, float, str or None
        User-provided area threshold. ``None`` and ``0`` both mean "disabled" to
        consumers. The string ``'biggest'`` means "keep only the largest
        component" for consumers that support it.

    node_id : str
        Node identifier used to produce contextual validation errors.

    param_name : str, optional
        Parameter name to include in error messages. This lets nodes reuse the
        same validation logic while preserving their public configuration names.

    Raises
    ------
    ValueError
        If ``value`` is negative, not numeric, or not one of the supported
        types.
    """
    if value is None:
        return

    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"'{node_id}': {param_name} must be >= 0")
        return

    if isinstance(value, str):
        s = value.strip()
        if s == 'biggest':
            return
        try:
            number = float(s[:-1]) if s.endswith('%') else float(s)
        except ValueError as exc:
            raise ValueError(
                f"'{node_id}': invalid {param_name}={value!r}"
            ) from exc

        if number < 0:
            raise ValueError(f"'{node_id}': {param_name} must be >= 0")
        return

    raise ValueError(
        f"'{node_id}': {param_name} must be an int, float, "
        'percentage string, or None'
    )


def resolve_min_component_area(
    value: Any,
    *,
    width: int,
    height: int,
) -> int:
    """
    Resolve a component-area cleanup threshold to pixels.

    Numeric values are interpreted directly as pixel areas. Percentage strings
    are interpreted linearly on the image long side and then converted to an
    area, so ``'1%'`` on a 1024x768 image becomes roughly
    ``(1024 * 0.01) ** 2`` pixels.

    Parameters
    ----------
    value : int, float, str or None
        Area threshold to resolve. ``None`` resolves to ``0``.

    width : int
        Image width used for percentage thresholds.

    height : int
        Image height used for percentage thresholds.

    Returns
    -------
    int
        Resolved non-negative pixel area threshold.
    """
    if value is None:
        return 0

    if isinstance(value, str):
        s = value.strip()
        if s == 'biggest':
            raise ValueError(
                "resolve_min_component_area does not accept 'biggest'; "
                "handle it before resolving numeric thresholds."
            )
        if s.endswith('%'):
            pct = float(s[:-1]) / 100.0
            side = max(width, height) * pct
            return int(round(side * side))
        return int(round(float(s)))

    return int(round(float(value)))


def read_shape_cleanup_config(
    value: Any,
    *,
    node_id: str,
    default_min_component_area: Any = 0,
) -> dict[str, Any]:
    """
    Read shared crop shape-cleanup configuration.
    """
    if value is None:
        cfg: dict[str, Any] = {}
    elif isinstance(value, dict):
        cfg = dict(value)
    else:
        raise ValueError(f"'{node_id}': postprocess must be a dictionary")

    fill_holes = cfg.get('fill_holes', 0)
    if isinstance(fill_holes, str):
        s = fill_holes.strip()
        if s != 'all':
            validate_min_component_area(
                s,
                node_id=node_id,
                param_name='postprocess.fill_holes',
            )
            fill_holes = s
    else:
        validate_min_component_area(
            fill_holes,
            node_id=node_id,
            param_name='postprocess.fill_holes',
        )

    try:
        morph_open_radius = int(cfg.get('morph_open_radius', 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{node_id}': postprocess.morph_open_radius must be an integer"
        ) from exc
    if morph_open_radius < 0:
        raise ValueError(
            f"'{node_id}': postprocess.morph_open_radius must be >= 0"
        )

    min_component_area = cfg.get(
        'min_component_area',
        default_min_component_area,
    )
    validate_min_component_area(
        min_component_area,
        node_id=node_id,
        param_name='postprocess.min_component_area',
    )

    return {
        'fill_holes': fill_holes,
        'morph_open_radius': morph_open_radius,
        'min_component_area': min_component_area,
    }
