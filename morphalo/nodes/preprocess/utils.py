import cv2
import math
import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Literal, Optional

SizeExpr = int | str

CropModeName = Literal['bbox', 'trim', 'full_frame']

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


def tight_alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError('Trimmed crop has no non-transparent pixels.')

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


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


def postprocess_mask(
    mask: np.ndarray,
    *,
    dilate_radius: int = 0,
    close_radius: int = 0,
    smoothing_radius: int = 0,
) -> np.ndarray:
    """
    Post-process a binary/soft mask to make it suitable for inpainting and diffusion.

    Steps (optional, in order):
      1. Morphological closing (fills holes, connects thin gaps)
      2. Dilation (expand mask outward)
      3. Gaussian smoothing (soft edges)

    Parameters
    ----------
    mask : np.ndarray
        Input mask. Accepted formats:
          - bool mask
          - float mask 0..1
          - uint8 mask 0..255

    dilate_radius : int
        Radius in pixels used to expand the mask outward.

    close_radius : int
        Radius in pixels used for morphological closing (fill holes).

    smoothing_radius : int
        Radius in pixels used for Gaussian blur (edge feathering).

    Returns
    -------
    np.ndarray
        Soft mask uint8 in range 0..255.
    """

    if mask.ndim != 2:
        raise ValueError("Mask must be HxW")

    # normalize to uint8 0..255
    if mask.dtype == bool:
        m = mask.astype(np.uint8) * 255
    elif np.issubdtype(mask.dtype, np.floating):
        m = np.clip(mask * 255, 0, 255).astype(np.uint8)
    else:
        m = mask.astype(np.uint8)

    # --- closing (fills holes) ---
    if close_radius > 0:
        k = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # --- dilation (expand mask) ---
    if dilate_radius > 0:
        k = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)

    # --- smoothing / feather ---
    if smoothing_radius > 0:
        k = 2 * smoothing_radius + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=0, sigmaY=0)

    return np.clip(m, 0, 255).astype(np.uint8)
