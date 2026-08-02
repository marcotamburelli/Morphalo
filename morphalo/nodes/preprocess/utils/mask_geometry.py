import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import cv2
import numpy as np

from morphalo.nodes.preprocess.utils.geometry import (expand_clip_bbox,
                                                      segment_capsule_mask,
                                                      tight_mask_bbox)

AnatomicalSide = Literal['anatomical-left', 'anatomical-right']
LimbSelection = Literal[
    'both',
    'anatomical-left',
    'anatomical-right',
]


@dataclass(frozen=True)
class ResolvedLandmark:
    """
    Pose landmark resolved in full-frame image coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Full-frame image coordinates with shape ``(2,)``.

    xyz_px : np.ndarray or None
        Full-frame pseudo-3D image-space coordinates with shape ``(3,)``.
        The first two components correspond to image coordinates, while the
        third component contains MediaPipe relative depth scaled to pixels.
        ``None`` indicates that the depth information is unavailable or
        invalid, while the two-dimensional landmark remains usable.
    """

    xy: np.ndarray
    xyz_px: Optional[np.ndarray] = None


LandmarkResolver = Callable[[str], Optional[ResolvedLandmark]]

# Arm radius probing treats the shoulder as a slightly wider anchor than the
# last visible semantic support pixel.
_ARM_SHOULDER_RADIUS_EXPANSION = 1.15

# Pose-landmark arm rotation is normalized around the vertical image axis; values
# above this threshold are wrapped before fitting tube/cap geometry.
_ARM_RADIUS_VERTICAL_AXIS_THRESHOLD_DEGREES = 280.0

# Arm tube radius guardrails. The observed radius is measured from visible
# shoulder-to-elbow semantic support, then expanded and clipped by shoulder width
# and first-segment length. Shoulder-width bounds may use pseudo-3D width when
# the caller provides it.
_ARM_FALLBACK_RADIUS_SHOULDER_WIDTH_RATIO = 0.35
_ARM_MAX_RADIUS_SHOULDER_WIDTH_RATIO = 0.30
_ARM_MAX_RADIUS_UPPER_ARM_LENGTH_RATIO = 0.75

# Leg radius probing gives hip support a small margin so the tube keeps enough
# proximal attachment before it is clipped by pelvis/segment guardrails.
_LEG_HIP_RADIUS_EXPANSION = 1.10

# Leg tube radius guardrails. The observed radius is measured by probing the
# semantic support outward from the hip, then expanded and clipped by these
# anatomical bounds. Pelvis ratios use the projected 2D hip-to-hip width; the
# segment ratio uses the projected hip-to-knee length.
_LEG_FALLBACK_RADIUS_PELVIS_WIDTH_RATIO = 0.32
_LEG_MIN_RADIUS_PELVIS_WIDTH_RATIO = 0.40
_LEG_MAX_RADIUS_PELVIS_WIDTH_RATIO = 0.65
_LEG_MAX_RADIUS_HIP_KNEE_LENGTH_RATIO = 0.45

# Arm shoulder caps use an ellipse: the minor axis is bounded by shoulder width
# and by the major axis so the cap remains broad enough without swallowing torso.
_ARM_SHOULDER_CAP_MINOR_WIDTH_RATIO = 0.20
_ARM_SHOULDER_CAP_MINOR_RATIO = 0.70

# Proximal masks include a small allowance behind the anatomical half-plane so
# slightly noisy landmarks do not cut off useful shoulder/hip attachment pixels.
_LIMB_PROXIMAL_HALFPLANE_BACKTRACK_RATIO = 0.35

# Distal leg support is trimmed shortly beyond the ankle to keep shoes/feet from
# stretching the leg tube while preserving a little boundary context.
_LEG_AFTER_ANKLE_TRIM_RADIUS_RATIO = 0.25

# Absolute radius guardrails used when proportional measurements collapse on
# tiny crops or degenerate landmark configurations.
_LIMB_MIN_RADIUS_PX = 8.0
_LIMB_FALLBACK_MIN_RADIUS_PX = 12.0
_LIMB_RADIUS_SEARCH_MIN_PX = 2.0

# Limb geometry crops add a narrow margin around the semantic support and
# generated tube/cap masks so downstream topology has boundary context.
_LIMB_GEOMETRY_CROP_EXPANSION_DEFAULT = 0.12


def estimate_mask_centerline_alignment_angle(
    mask: np.ndarray,
    *,
    min_row_width_fraction: float = 0.25,
    trim_top_fraction: float = 0.10,
    trim_bottom_fraction: float = 0.05,
) -> float:
    """
    Estimate the rectangle alignment angle from a binary mask.

    The function estimates the visual slant of the mask by fitting a line to
    row midpoints. The returned value is expressed in the same convention used
    by ``_fit_rect_at_angle``: it is the angle passed directly to the fitting
    function.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    min_row_width_fraction : float, default=0.25
        Minimum row width, expressed as a fraction of the maximum foreground row
        width. Rows narrower than this threshold are ignored.
    trim_top_fraction : float, default=0.10
        Fraction of valid centerline samples removed from the top before fitting.
    trim_bottom_fraction : float, default=0.05
        Fraction of valid centerline samples removed from the bottom before fitting.

    Returns
    -------
    float
        Alignment angle in degrees, using the same convention expected by
        ``_fit_rect_at_angle``.
    """

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    ys = []
    xs = []
    widths = []

    h, _ = mask.shape[:2]

    for y in range(h):
        row_xs = np.flatnonzero(mask[y])
        if row_xs.size == 0:
            continue

        x_left = float(row_xs[0])
        x_right = float(row_xs[-1])
        width = x_right - x_left + 1.0

        ys.append(float(y))
        xs.append(0.5 * (x_left + x_right))
        widths.append(width)

    if len(xs) < 2:
        return 0.0

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    widths_arr = np.asarray(widths, dtype=np.float64)

    max_width = float(np.max(widths_arr))
    if max_width <= 0.0:
        return 0.0

    keep = widths_arr >= float(min_row_width_fraction) * max_width

    xs_arr = xs_arr[keep]
    ys_arr = ys_arr[keep]

    if xs_arr.size < 2:
        return 0.0

    order = np.argsort(ys_arr)
    xs_arr = xs_arr[order]
    ys_arr = ys_arr[order]

    n = xs_arr.size
    top_cut = int(round(n * float(trim_top_fraction)))
    bottom_cut = int(round(n * float(trim_bottom_fraction)))

    end = n - bottom_cut
    if end <= top_cut + 1:
        top_cut = 0
        end = n

    xs_arr = xs_arr[top_cut:end]
    ys_arr = ys_arr[top_cut:end]

    if xs_arr.size < 2:
        return 0.0

    slope, _ = np.polyfit(ys_arr, xs_arr, deg=1)

    return float(np.degrees(np.arctan(float(slope))))


def estimate_mask_pca_alignment_angle(mask: np.ndarray) -> float:
    """
    Estimate the rectangle alignment angle from the PCA major axis of a mask.

    The function treats foreground pixels as a 2D point cloud and computes the
    principal component with maximum variance. The returned angle represents the
    slant of that principal axis using the same convention expected by the
    rectangle fitting code: ``0`` means a vertical axis in image coordinates,
    positive values mean that the axis moves toward the right while going from
    top to bottom.

    This estimator is purely geometric. It does not infer the semantic
    orientation of the represented object; it only measures the dominant axis of
    the visible foreground distribution.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.

    Returns
    -------
    float
        Alignment angle in degrees. Returns ``0.0`` if the mask does not contain
        enough foreground pixels to estimate a stable principal axis.
    """

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    ys, xs = np.where(mask)

    if xs.size < 2:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)

    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    major = eigvecs[:, int(np.argmax(eigvals))]

    # Convert the PCA vector to the alignment convention used by the fitting
    # code. The vector is expressed as (dx, dy); atan2(dx, dy) measures slant
    # relative to the vertical image axis.
    angle = float(np.degrees(np.arctan2(float(major[0]), float(major[1]))))

    # PCA eigenvectors have arbitrary sign, so normalize equivalent axes to the
    # [-90, 90] range.
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0

    return angle


def _sample_indices_evenly(
    n: int,
    *,
    sample_count: int,
) -> np.ndarray:
    """
    Sample evenly spaced indices from a sequence.

    Parameters
    ----------
    n : int
        Number of available samples.
    sample_count : int
        Maximum number of indices to return.

    Returns
    -------
    np.ndarray
        Integer indices with shape ``(M,)``.
    """

    if n <= 0:
        return np.empty((0,), dtype=np.int64)

    if sample_count <= 0 or n <= sample_count:
        return np.arange(n, dtype=np.int64)

    return np.linspace(
        0,
        n - 1,
        num=int(sample_count),
        dtype=np.int64,
    )


def _resolve_median_window(
    n: int,
    *,
    divisor: int = 5,
) -> int:
    """
    Resolve an odd median smoothing window for sampled points.

    Parameters
    ----------
    n : int
        Number of sampled points.
    divisor : int, default=5
        Divisor used to derive the smoothing window from ``n``.

    Returns
    -------
    int
        Odd median window size. Returns ``1`` when smoothing should be disabled.
    """

    if n < 3:
        return 1

    window = max(3, int(round(n / int(divisor))))

    if window % 2 == 0:
        window += 1

    if window > n:
        window = n if n % 2 == 1 else n - 1

    return max(1, int(window))


def _median_smooth_1d(
    values: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """
    Smooth a one-dimensional sequence with a centered median filter.

    Parameters
    ----------
    values : np.ndarray
        Input values with shape ``(N,)``.
    window : int
        Median window size. Values less than or equal to ``1`` disable smoothing.

    Returns
    -------
    np.ndarray
        Smoothed values with shape ``(N,)``.
    """

    if values.ndim != 1:
        raise ValueError('values must be one-dimensional.')

    values = values.astype(np.float64)

    if window <= 1 or values.size < 3:
        return values

    window = min(int(window), int(values.size))

    if window % 2 == 0:
        window += 1

    if window > values.size:
        window = values.size if values.size % 2 == 1 else values.size - 1

    if window <= 1:
        return values

    pad = window // 2
    padded = np.pad(
        values,
        (pad, pad),
        mode='edge',
    )

    out = np.empty(values.shape, dtype=np.float64)

    for i in range(values.size):
        out[i] = float(np.median(padded[i:i + window]))

    return out


def _fit_poly_slope_line_1d(
    independent: np.ndarray,
    dependent: np.ndarray,
    *,
    sample_count: int = 16,
    degree: int = 2,
    slope_trim_fraction: float = 0.0,
    max_abs_slope: float = 1.0,
    min_slope_samples: int = 2,
) -> Tuple[float, float]:
    """
    Fit an equivalent first-degree line using a smoothed polynomial slope estimate.

    The input sequence is sorted by the independent coordinate, sampled evenly,
    smoothed with a median filter on the dependent coordinate, and fitted with a
    low-degree polynomial. The returned slope is estimated from the polynomial
    derivative.

    Derivative samples whose absolute slope is greater than ``max_abs_slope`` are
    treated as incompatible with the expected border direction and ignored. If
    too few compatible samples remain, the samples with the smallest absolute
    slopes are kept as a fallback.

    The returned intercept is chosen so that the equivalent line passes through
    the median compatible sampled point. This gives a stable first-degree
    approximation while allowing gently curved borders to contribute their local
    average direction rather than forcing a single global linear fit.

    Parameters
    ----------
    independent : np.ndarray
        Independent coordinate values.
    dependent : np.ndarray
        Dependent coordinate values.
    sample_count : int, default=16
        Maximum number of evenly spaced samples used for fitting.
    degree : int, default=2
        Polynomial degree used to estimate the local slope. Values greater than
        the number of available samples minus one are automatically reduced.
    slope_trim_fraction : float, default=0.0
        Fraction of sampled points removed from both ends before evaluating the
        polynomial derivative. This is disabled by default because incompatible
        slopes are filtered explicitly through ``max_abs_slope``.
    max_abs_slope : float, default=1.0
        Maximum accepted absolute derivative. ``1.0`` corresponds to a 45-degree
        limit in the local border parameterization.
    min_slope_samples : int, default=2
        Minimum number of derivative samples kept for estimating the equivalent
        line.

    Returns
    -------
    tuple[float, float]
        Slope and intercept of the equivalent first-degree line.

    Raises
    ------
    ValueError
        If input arrays are not one-dimensional or do not have the same length.
    RuntimeError
        If too few points are available to estimate a stable line.
    """

    if independent.ndim != 1 or dependent.ndim != 1:
        raise ValueError('independent and dependent must be one-dimensional.')

    if independent.size != dependent.size:
        raise ValueError(
            'independent and dependent must have the same length.'
        )

    if independent.size < 2:
        raise RuntimeError('At least two points are required to fit a line.')

    if degree < 1:
        raise ValueError(f'degree must be at least 1, got {degree!r}.')

    if not 0.0 <= float(slope_trim_fraction) < 0.5:
        raise ValueError(
            'slope_trim_fraction must be in the range [0, 0.5).'
        )

    if max_abs_slope <= 0.0:
        raise ValueError(
            f'max_abs_slope must be greater than zero, got {max_abs_slope!r}.'
        )

    if min_slope_samples < 1:
        raise ValueError(
            f'min_slope_samples must be at least 1, got {min_slope_samples!r}.'
        )

    order = np.argsort(independent)
    independent = independent[order]
    dependent = dependent[order]

    idx = _sample_indices_evenly(
        independent.size,
        sample_count=sample_count,
    )

    x = independent[idx].astype(np.float64)
    y = dependent[idx].astype(np.float64)

    if x.size < 2:
        raise RuntimeError('At least two sampled points are required.')

    smooth_window = _resolve_median_window(y.size)
    y = _median_smooth_1d(
        y,
        window=smooth_window,
    )

    fit_degree = min(int(degree), int(x.size) - 1)

    if fit_degree < 1:
        raise RuntimeError('At least two sampled points are required.')

    coeff = np.polyfit(x, y, deg=fit_degree)
    deriv = np.polyder(coeff)

    n = x.size
    cut = int(round(n * float(slope_trim_fraction)))
    start = cut
    end = n - cut

    if end <= start:
        start = 0
        end = n

    x_eval = x[start:end]
    y_eval = y[start:end]

    if x_eval.size == 0:
        raise RuntimeError('Cannot estimate polynomial slope.')

    slopes = np.polyval(deriv, x_eval).astype(np.float64)

    compatible = np.abs(slopes) <= float(max_abs_slope)

    if int(np.count_nonzero(compatible)) >= int(min_slope_samples):
        keep_idx = np.flatnonzero(compatible)
    else:
        keep_count = min(int(min_slope_samples), int(slopes.size))
        keep_idx = np.argsort(np.abs(slopes))[:keep_count]

    if keep_idx.size == 0:
        raise RuntimeError('Cannot estimate compatible polynomial slope.')

    kept_slopes = slopes[keep_idx]
    kept_x = x_eval[keep_idx]
    kept_y = y_eval[keep_idx]

    slope = float(np.median(kept_slopes))

    # Build an equivalent first-degree line through the median compatible point.
    x_anchor = float(np.median(kept_x))
    y_anchor = float(np.median(kept_y))
    intercept = y_anchor - slope * x_anchor

    return slope, float(intercept)


def _intersect_x_of_y_with_y_of_x(
    *,
    x_slope: float,
    x_intercept: float,
    y_slope: float,
    y_intercept: float,
) -> Tuple[float, float]:
    """
    Intersect a vertical-like line with a horizontal-like line.

    The vertical-like line is represented as ``x = a*y + b``.
    The horizontal-like line is represented as ``y = c*x + d``.

    Parameters
    ----------
    x_slope : float
        Slope ``a`` of the vertical-like line.
    x_intercept : float
        Intercept ``b`` of the vertical-like line.
    y_slope : float
        Slope ``c`` of the horizontal-like line.
    y_intercept : float
        Intercept ``d`` of the horizontal-like line.

    Returns
    -------
    tuple[float, float]
        Intersection point ``(x, y)``.

    Raises
    ------
    RuntimeError
        If the two fitted lines are numerically degenerate.
    """

    denom = 1.0 - float(y_slope) * float(x_slope)

    if abs(denom) < 1e-6:
        raise RuntimeError('Cannot intersect nearly parallel fitted lines.')

    y = (
        float(y_slope) * float(x_intercept)
        + float(y_intercept)
    ) / denom
    x = float(x_slope) * y + float(x_intercept)

    return float(x), float(y)


def estimate_mask_side_alignment_angle(
    mask: np.ndarray,
    *,
    min_row_width_fraction: float = 0.25,
    trim_top_fraction: float = 0.10,
    trim_bottom_fraction: float = 0.05,
    sample_divisor: int = 20,
) -> float:
    """
    Estimate the alignment angle from fitted left and right mask borders.

    The function approximates the visible vertical sides of the mask by fitting
    two border lines:

    - left border as ``x = a*y + b``;
    - right border as ``x = a*y + b``.

    The returned alignment angle is computed from the median line between those
    two fitted borders. This estimator ignores the top and bottom borders, which
    makes it useful when necklines, lower cuts, straps, or noisy horizontal
    extremities make the full quadrilateral frame unstable.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    min_row_width_fraction : float, default=0.25
        Minimum row foreground width as a fraction of the maximum foreground row
        width. Narrower rows are ignored when fitting left/right borders.
    trim_top_fraction : float, default=0.10
        Fraction of valid rows trimmed from the top before fitting.
    trim_bottom_fraction : float, default=0.05
        Fraction of valid rows trimmed from the bottom before fitting.
    sample_divisor : int, default=20
        Divisor used to derive the number of sampled border points after
        filtering and trimming, with a minimum of two points.

    Returns
    -------
    float
        Alignment angle in degrees, using the same convention expected by
        ``_fit_rect_at_angle``.

    Raises
    ------
    RuntimeError
        If the side-border estimate cannot be computed robustly.
    """

    if sample_divisor <= 0:
        raise ValueError(
            f'sample_divisor must be greater than zero, got {sample_divisor!r}.'
        )

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    h, _ = mask.shape[:2]

    row_ys = []
    left_xs = []
    right_xs = []
    row_widths = []

    for y in range(h):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue

        left = float(xs[0])
        right = float(xs[-1])
        width = right - left + 1.0

        row_ys.append(float(y))
        left_xs.append(left)
        right_xs.append(right)
        row_widths.append(width)

    if len(row_ys) < 2:
        raise RuntimeError('Not enough mask side-border samples.')

    row_ys_arr = np.asarray(row_ys, dtype=np.float64)
    left_arr = np.asarray(left_xs, dtype=np.float64)
    right_arr = np.asarray(right_xs, dtype=np.float64)
    row_widths_arr = np.asarray(row_widths, dtype=np.float64)

    max_width = float(np.max(row_widths_arr))
    if max_width <= 0.0:
        raise RuntimeError('Degenerate mask side-border samples.')

    keep = row_widths_arr >= float(min_row_width_fraction) * max_width

    row_ys_arr = row_ys_arr[keep]
    left_arr = left_arr[keep]
    right_arr = right_arr[keep]

    if row_ys_arr.size < 2:
        raise RuntimeError('Not enough filtered side-border samples.')

    order = np.argsort(row_ys_arr)
    row_ys_arr = row_ys_arr[order]
    left_arr = left_arr[order]
    right_arr = right_arr[order]

    n = row_ys_arr.size
    top_cut = int(round(n * float(trim_top_fraction)))
    bottom_cut = int(round(n * float(trim_bottom_fraction)))

    end = n - bottom_cut
    if end <= top_cut + 1:
        top_cut = 0
        end = n

    row_ys_arr = row_ys_arr[top_cut:end]
    left_arr = left_arr[top_cut:end]
    right_arr = right_arr[top_cut:end]

    if row_ys_arr.size < 2:
        raise RuntimeError('Not enough trimmed side-border samples.')

    sample_count = max(2, int(round(row_ys_arr.size / sample_divisor)))

    left_slope, _ = _fit_poly_slope_line_1d(
        row_ys_arr,
        left_arr,
        sample_count=sample_count,
    )
    right_slope, _ = _fit_poly_slope_line_1d(
        row_ys_arr,
        right_arr,
        sample_count=sample_count,
    )

    mid_slope = 0.5 * (float(left_slope) + float(right_slope))

    return float(np.degrees(np.arctan(mid_slope)))


def estimate_mask_horizontal_border_alignment_angle(
    mask: np.ndarray,
    *,
    min_col_height_fraction: float = 0.25,
    trim_left_fraction: float = 0.10,
    trim_right_fraction: float = 0.10,
    sample_divisor: int = 20,
) -> float:
    """
    Estimate the alignment angle from fitted top and bottom mask borders.

    The function approximates the visible horizontal sides of the mask by fitting
    two border lines:

    - top border as ``y = a*x + b``;
    - bottom border as ``y = a*x + b``.

    The returned alignment angle is derived from the median top/bottom border
    direction. This estimator can be useful when top and bottom edges are more
    reliable than left/right sides.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    min_col_height_fraction : float, default=0.25
        Minimum column foreground height as a fraction of the maximum foreground
        column height. Shorter columns are ignored when fitting top/bottom
        borders.
    trim_left_fraction : float, default=0.10
        Fraction of valid columns trimmed from the left before fitting.
    trim_right_fraction : float, default=0.10
        Fraction of valid columns trimmed from the right before fitting.
    sample_divisor : int, default=20
        Divisor used to derive the number of sampled border points after
        filtering and trimming, with a minimum of two points.

    Returns
    -------
    float
        Alignment angle in degrees, using the same convention expected by
        ``_fit_rect_at_angle``.

    Raises
    ------
    RuntimeError
        If the horizontal-border estimate cannot be computed robustly.
    """

    if sample_divisor <= 0:
        raise ValueError(
            f'sample_divisor must be greater than zero, got {sample_divisor!r}.'
        )

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    _, w = mask.shape[:2]

    col_xs = []
    top_ys = []
    bottom_ys = []
    col_heights = []

    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size == 0:
            continue

        top = float(ys[0])
        bottom = float(ys[-1])
        height = bottom - top + 1.0

        col_xs.append(float(x))
        top_ys.append(top)
        bottom_ys.append(bottom)
        col_heights.append(height)

    if len(col_xs) < 2:
        raise RuntimeError('Not enough mask horizontal-border samples.')

    col_xs_arr = np.asarray(col_xs, dtype=np.float64)
    top_arr = np.asarray(top_ys, dtype=np.float64)
    bottom_arr = np.asarray(bottom_ys, dtype=np.float64)
    col_heights_arr = np.asarray(col_heights, dtype=np.float64)

    max_height = float(np.max(col_heights_arr))
    if max_height <= 0.0:
        raise RuntimeError('Degenerate mask horizontal-border samples.')

    keep = col_heights_arr >= float(min_col_height_fraction) * max_height

    col_xs_arr = col_xs_arr[keep]
    top_arr = top_arr[keep]
    bottom_arr = bottom_arr[keep]

    if col_xs_arr.size < 2:
        raise RuntimeError('Not enough filtered horizontal-border samples.')

    order = np.argsort(col_xs_arr)
    col_xs_arr = col_xs_arr[order]
    top_arr = top_arr[order]
    bottom_arr = bottom_arr[order]

    n = col_xs_arr.size
    left_cut = int(round(n * float(trim_left_fraction)))
    right_cut = int(round(n * float(trim_right_fraction)))

    end = n - right_cut
    if end <= left_cut + 1:
        left_cut = 0
        end = n

    col_xs_arr = col_xs_arr[left_cut:end]
    top_arr = top_arr[left_cut:end]
    bottom_arr = bottom_arr[left_cut:end]

    if col_xs_arr.size < 2:
        raise RuntimeError('Not enough trimmed horizontal-border samples.')

    sample_count = max(2, int(round(col_xs_arr.size / sample_divisor)))

    top_slope, _ = _fit_poly_slope_line_1d(
        col_xs_arr,
        top_arr,
        sample_count=sample_count,
    )
    bottom_slope, _ = _fit_poly_slope_line_1d(
        col_xs_arr,
        bottom_arr,
        sample_count=sample_count,
    )

    mid_slope = 0.5 * (float(top_slope) + float(bottom_slope))

    return float(np.degrees(np.arctan(mid_slope)))


def estimate_mask_quad_alignment_angle(
    mask: np.ndarray,
    *,
    min_row_width_fraction: float = 0.25,
    min_col_height_fraction: float = 0.25,
    trim_row_fraction: float = 0.10,
    trim_col_fraction: float = 0.10,
    sample_divisor: int = 20,
) -> float:
    """
    Estimate the alignment angle from a fitted quadrilateral frame.

    The function approximates the visible mask shape with four fitted border
    lines:

    - left and right borders are fitted as ``x = a*y + b``;
    - top and bottom borders are fitted as ``y = a*x + b``.

    The four intersections define an approximate quadrilateral frame. The
    returned alignment angle is the angle of the line connecting the midpoint of
    the top side to the midpoint of the bottom side.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask with shape ``(H, W)``.
    min_row_width_fraction : float, default=0.25
        Minimum row foreground width as a fraction of the maximum foreground row
        width. Narrower rows are ignored when fitting left/right borders.
    min_col_height_fraction : float, default=0.25
        Minimum column foreground height as a fraction of the maximum foreground
        column height. Shorter columns are ignored when fitting top/bottom
        borders.
    trim_row_fraction : float, default=0.10
        Fraction of valid rows trimmed from both top and bottom before fitting
        left/right borders.
    trim_col_fraction : float, default=0.10
        Fraction of valid columns trimmed from both left and right before fitting
        top/bottom borders.
    sample_divisor : int, default=20
        Divisor used to derive the number of sampled border points after filtering
        and trimming. For example, with ``sample_divisor=20`` roughly one point every
        twenty valid border samples is used for each fitted line, with a minimum of
        two points.

    Returns
    -------
    float
        Alignment angle in degrees, using the same convention expected by
        ``_fit_rect_at_angle``.

    Raises
    ------
    RuntimeError
        If the quadrilateral frame cannot be estimated robustly.
    """

    if sample_divisor <= 0:
        raise ValueError(
            f'sample_divisor must be greater than zero, got {sample_divisor!r}.'
        )

    if mask.ndim != 2:
        raise ValueError('mask must have shape (H, W).')

    h, w = mask.shape[:2]

    row_ys = []
    left_xs = []
    right_xs = []
    row_widths = []

    for y in range(h):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue

        left = float(xs[0])
        right = float(xs[-1])
        width = right - left + 1.0

        row_ys.append(float(y))
        left_xs.append(left)
        right_xs.append(right)
        row_widths.append(width)

    col_xs = []
    top_ys = []
    bottom_ys = []
    col_heights = []

    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size == 0:
            continue

        top = float(ys[0])
        bottom = float(ys[-1])
        height = bottom - top + 1.0

        col_xs.append(float(x))
        top_ys.append(top)
        bottom_ys.append(bottom)
        col_heights.append(height)

    if len(row_ys) < 2 or len(col_xs) < 2:
        raise RuntimeError('Not enough mask border samples.')

    row_ys_arr = np.asarray(row_ys, dtype=np.float64)
    left_arr = np.asarray(left_xs, dtype=np.float64)
    right_arr = np.asarray(right_xs, dtype=np.float64)
    row_widths_arr = np.asarray(row_widths, dtype=np.float64)

    col_xs_arr = np.asarray(col_xs, dtype=np.float64)
    top_arr = np.asarray(top_ys, dtype=np.float64)
    bottom_arr = np.asarray(bottom_ys, dtype=np.float64)
    col_heights_arr = np.asarray(col_heights, dtype=np.float64)

    max_row_width = float(np.max(row_widths_arr))
    max_col_height = float(np.max(col_heights_arr))

    if max_row_width <= 0.0 or max_col_height <= 0.0:
        raise RuntimeError('Degenerate mask border samples.')

    row_keep = (
        row_widths_arr
        >= float(min_row_width_fraction) * max_row_width
    )
    col_keep = (
        col_heights_arr
        >= float(min_col_height_fraction) * max_col_height
    )

    row_ys_arr = row_ys_arr[row_keep]
    left_arr = left_arr[row_keep]
    right_arr = right_arr[row_keep]

    col_xs_arr = col_xs_arr[col_keep]
    top_arr = top_arr[col_keep]
    bottom_arr = bottom_arr[col_keep]

    if row_ys_arr.size < 2 or col_xs_arr.size < 2:
        raise RuntimeError('Not enough filtered border samples.')

    row_order = np.argsort(row_ys_arr)
    row_ys_arr = row_ys_arr[row_order]
    left_arr = left_arr[row_order]
    right_arr = right_arr[row_order]

    col_order = np.argsort(col_xs_arr)
    col_xs_arr = col_xs_arr[col_order]
    top_arr = top_arr[col_order]
    bottom_arr = bottom_arr[col_order]

    row_n = row_ys_arr.size
    row_cut = int(round(row_n * float(trim_row_fraction)))
    row_start = row_cut
    row_end = row_n - row_cut

    if row_end <= row_start + 1:
        row_start = 0
        row_end = row_n

    row_ys_arr = row_ys_arr[row_start:row_end]
    left_arr = left_arr[row_start:row_end]
    right_arr = right_arr[row_start:row_end]

    col_n = col_xs_arr.size
    col_cut = int(round(col_n * float(trim_col_fraction)))
    col_start = col_cut
    col_end = col_n - col_cut

    if col_end <= col_start + 1:
        col_start = 0
        col_end = col_n

    col_xs_arr = col_xs_arr[col_start:col_end]
    top_arr = top_arr[col_start:col_end]
    bottom_arr = bottom_arr[col_start:col_end]

    if row_ys_arr.size < 2 or col_xs_arr.size < 2:
        raise RuntimeError('Not enough trimmed border samples.')

    row_sample_count = max(2, int(round(row_ys_arr.size / sample_divisor)))
    col_sample_count = max(2, int(round(col_xs_arr.size / sample_divisor)))

    left_slope, left_intercept = _fit_poly_slope_line_1d(
        row_ys_arr,
        left_arr,
        sample_count=row_sample_count,
    )
    right_slope, right_intercept = _fit_poly_slope_line_1d(
        row_ys_arr,
        right_arr,
        sample_count=row_sample_count,
    )

    top_slope, top_intercept = _fit_poly_slope_line_1d(
        col_xs_arr,
        top_arr,
        sample_count=col_sample_count,
    )
    bottom_slope, bottom_intercept = _fit_poly_slope_line_1d(
        col_xs_arr,
        bottom_arr,
        sample_count=col_sample_count,
    )

    top_left = _intersect_x_of_y_with_y_of_x(
        x_slope=left_slope,
        x_intercept=left_intercept,
        y_slope=top_slope,
        y_intercept=top_intercept,
    )
    top_right = _intersect_x_of_y_with_y_of_x(
        x_slope=right_slope,
        x_intercept=right_intercept,
        y_slope=top_slope,
        y_intercept=top_intercept,
    )
    bottom_left = _intersect_x_of_y_with_y_of_x(
        x_slope=left_slope,
        x_intercept=left_intercept,
        y_slope=bottom_slope,
        y_intercept=bottom_intercept,
    )
    bottom_right = _intersect_x_of_y_with_y_of_x(
        x_slope=right_slope,
        x_intercept=right_intercept,
        y_slope=bottom_slope,
        y_intercept=bottom_intercept,
    )

    mid_top_x = 0.5 * (top_left[0] + top_right[0])
    mid_top_y = 0.5 * (top_left[1] + top_right[1])

    mid_bottom_x = 0.5 * (bottom_left[0] + bottom_right[0])
    mid_bottom_y = 0.5 * (bottom_left[1] + bottom_right[1])

    dx = float(mid_bottom_x - mid_top_x)
    dy = float(mid_bottom_y - mid_top_y)

    if abs(dy) < 1e-6:
        raise RuntimeError('Degenerate quadrilateral vertical axis.')

    return float(np.degrees(np.arctan2(dx, dy)))


@dataclass(frozen=True)
class LimbCropGeometry:
    """
    Geometry for one semantic-prior limb crop.

    The object contains no model-specific state. It describes the full-frame
    masks and bboxes needed by a limb crop/refinement pipeline:

    - ``tube_mask`` is the domain in which the limb body is refined;
    - ``skeleton_mask`` is the expected proximal-to-distal centerline;
    - ``synthetic_barrier_mask`` closes the proximal and distal ends;
    - ``proximal_circle_mask`` restores the joint attachment area;
    - ``proximal_quadrant_mask`` restores the outward joint quadrant;
    - ``semantic_region_mask`` is the broad semantic region before limb-geometry
      clipping;
    - ``semantic_support_mask`` is the broad semantic region restricted by the
      limb geometry and is used to derive the local crop;
    - ``base_bbox`` tightly encloses the semantic support;
    - ``crop_bbox`` is the expanded local processing crop.

    All masks use full-frame coordinates. Callers may crop them using
    ``crop_bbox`` before invoking local refinement.
    """

    side: AnatomicalSide
    proximal_name: str
    middle_name: str
    distal_name: str

    proximal_point: np.ndarray
    middle_point: Optional[np.ndarray]
    distal_point: Optional[np.ndarray]

    base_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]

    semantic_region_mask: np.ndarray
    semantic_support_mask: np.ndarray
    tube_mask: np.ndarray
    skeleton_mask: np.ndarray
    synthetic_barrier_mask: np.ndarray
    proximal_circle_mask: np.ndarray
    proximal_quadrant_mask: np.ndarray

    tube_radius: float


@dataclass(frozen=True)
class _LimbRadiusBounds:
    """
    Guardrails for a measured limb radius.

    ``fallback_radius`` is used only when semantic probing cannot produce a
    usable observed radius. ``search_limit`` is the maximum outward distance
    worth probing: anything beyond it would be clipped to ``max_radius`` anyway.
    """

    min_radius: float
    max_radius: float
    fallback_radius: float
    search_limit: float


def build_arm_crop_geometries(
    *,
    resolve_landmark: LandmarkResolver,
    image_shape: tuple[int, ...],
    semantic_region: np.ndarray,
    which: LimbSelection = 'both',
    expansion: float = 1.0,
    crop_expansion: float = _LIMB_GEOMETRY_CROP_EXPANSION_DEFAULT,
) -> list[LimbCropGeometry]:
    """
    Build semantic-prior crop geometries for one or both anatomical arms.

    Required landmark names are:

    - ``left_shoulder``, ``left_elbow``, ``left_wrist``;
    - ``right_shoulder``, ``right_elbow``, ``right_wrist``;
    - optionally ``nose`` as a torso/head-side reference.

    ``semantic_region`` should be a broad mask that contains arm skin and the
    relevant clothing/torso context. It is not treated as the final arm mask.

    Parameters
    ----------
    resolve_landmark : LandmarkResolver
        Callable that resolves landmark names to full-frame image coordinates and
        optional pseudo-3D image-space coordinates. Two-dimensional coordinates
        are used for raster geometry. Shoulder-width measurements used by radius
        fallback and clamp bounds use pseudo-3D coordinates when valid for both
        shoulders and fall back to two-dimensional coordinates otherwise.

    image_shape : tuple[int, ...]
        Source image shape. The first two dimensions are used as ``(H, W)``.

    semantic_region : np.ndarray
        Full-frame boolean or binary semantic support region. It should include
        arm skin, upper clothing, and torso context before geometric
        restriction.

    which : {'both', 'anatomical-left', 'anatomical-right'}, default='both'
        Anatomical arm side or sides to build.

    expansion : float, default=1.0
        Multiplicative expansion applied to the estimated tube/joint radius.

    crop_expansion : float, default=_LIMB_GEOMETRY_CROP_EXPANSION_DEFAULT
        Proportional margin applied to the tight support bbox to derive the
        processing crop bbox.

    Returns
    -------
    list[LimbCropGeometry]
        One geometry per requested anatomical arm that can be derived from the
        available landmarks and semantic support.

    Raises
    ------
    ValueError
        If input shapes, side selection, or expansion values are invalid.

    RuntimeError
        If required shoulder landmarks are missing/degenerate or no requested
        arm geometry can be derived.
    """
    h, w = _validate_limb_geometry_inputs(
        image_shape=image_shape,
        semantic_region=semantic_region,
        which=which,
        expansion=expansion,
        crop_expansion=crop_expansion,
    )
    semantic_bool = semantic_region.astype(bool, copy=False)

    left_shoulder_landmark = resolve_landmark('left_shoulder')
    right_shoulder_landmark = resolve_landmark('right_shoulder')

    left_shoulder = _landmark_xy(left_shoulder_landmark)
    right_shoulder = _landmark_xy(right_shoulder_landmark)

    if left_shoulder is None or right_shoulder is None:
        raise RuntimeError(
            'Arm crop geometry requires both shoulder landmarks.'
        )

    shoulder_axis = right_shoulder - left_shoulder
    shoulder_width = float(np.linalg.norm(shoulder_axis))
    if shoulder_width < 4.0:
        raise RuntimeError(
            'Shoulder landmarks are too close for arm crop geometry.'
        )

    shoulder_width_for_bounds = _landmark_distance(
        left_shoulder_landmark,
        right_shoulder_landmark,
    )
    if shoulder_width_for_bounds is None:
        shoulder_width_for_bounds = shoulder_width

    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    nose = _limb_point(resolve_landmark, 'nose')
    head_reference = (
        nose
        if nose is not None
        else shoulder_mid + np.asarray([0.0, -shoulder_width], dtype=np.float32)
    )

    yy, xx = np.mgrid[0:h, 0:w]
    shoulder_axis_len = max(shoulder_width, 1e-6)

    head_cross = _cross2(
        shoulder_axis,
        head_reference - shoulder_mid,
    )
    head_side = 1.0 if head_cross >= 0.0 else -1.0

    rel_mid_x = xx - float(shoulder_mid[0])
    rel_mid_y = yy - float(shoulder_mid[1])
    grid_cross_from_shoulders = (
        float(shoulder_axis[0]) * rel_mid_y
        - float(shoulder_axis[1]) * rel_mid_x
    )

    geometries: list[LimbCropGeometry] = []

    for side in _selected_limb_sides(which):
        prefix = 'left' if side == 'anatomical-left' else 'right'

        shoulder = (
            left_shoulder if side == 'anatomical-left' else right_shoulder
        )
        other_shoulder = (
            right_shoulder if side == 'anatomical-left' else left_shoulder
        )
        elbow = _limb_point(resolve_landmark, f'{prefix}_elbow')
        wrist = _limb_point(resolve_landmark, f'{prefix}_wrist')

        segments = _limb_chain_segments(shoulder, elbow, wrist)
        if not segments:
            continue

        radius = _estimate_arm_proximal_band_radius(
            semantic_bool,
            side=side,
            proximal=shoulder,
            opposite_proximal=other_shoulder,
            middle=elbow,
            image_shape=image_shape,
            bounds_width=shoulder_width_for_bounds,
            fallback_scale=_ARM_FALLBACK_RADIUS_SHOULDER_WIDTH_RATIO,
            min_scale=_ARM_SHOULDER_CAP_MINOR_WIDTH_RATIO,
            max_width_scale=_ARM_MAX_RADIUS_SHOULDER_WIDTH_RATIO,
            max_segment_scale=_ARM_MAX_RADIUS_UPPER_ARM_LENGTH_RATIO,
            expansion_constant=_ARM_SHOULDER_RADIUS_EXPANSION,
        )

        radius *= max(1.0, float(expansion))

        tube_mask = _build_limb_tube_mask(
            image_shape=image_shape,
            segments=segments,
            radius=radius,
        )
        skeleton_mask = build_limb_skeleton_mask(
            image_shape=image_shape,
            segments=segments,
            points=(shoulder, elbow, wrist),
        )

        shoulder_cap_axis = (
            elbow - shoulder
            if elbow is not None
            else shoulder - other_shoulder
        )
        (
            shoulder_cap_major_radius,
            shoulder_cap_minor_radius,
        ) = _arm_shoulder_cap_radii(
            tube_radius=radius,
            shoulder_width=shoulder_width,
        )
        # The proximal barrier is drawn on the nominal cap, while the workspace
        # cap gets a small 2px margin. This gives the dilated barrier room to
        # separate regions instead of coinciding with the workspace boundary.
        proximal_circle_mask = _ellipse_mask(
            xx=xx,
            yy=yy,
            center=shoulder,
            major_axis=shoulder_cap_axis,
            major_radius=shoulder_cap_major_radius + 2.0,
            minor_radius=shoulder_cap_minor_radius + 2.0,
        )

        head_halfplane = (
            head_side * grid_cross_from_shoulders
        ) >= (
            -_LIMB_PROXIMAL_HALFPLANE_BACKTRACK_RATIO
            * radius
            * shoulder_axis_len
        )

        outward_vector = shoulder - other_shoulder
        outward_length = float(np.linalg.norm(outward_vector))
        if outward_length < 1e-6:
            continue

        outward_unit = outward_vector / outward_length
        rel_x = xx - float(shoulder[0])
        rel_y = yy - float(shoulder[1])
        outward_projection = (
            rel_x * float(outward_unit[0])
            + rel_y * float(outward_unit[1])
        )
        outward_halfplane = (
            outward_projection
            >= -_LIMB_PROXIMAL_HALFPLANE_BACKTRACK_RATIO * radius
        )

        proximal_quadrant_mask = (
            semantic_bool
            & head_halfplane
            & outward_halfplane
        )

        semantic_support_mask = (
            semantic_bool
            & (
                tube_mask
                | proximal_circle_mask
                | proximal_quadrant_mask
            )
        )
        if not np.any(semantic_support_mask):
            continue

        proximal_barrier_mask = _ellipse_outline_mask(
            image_shape=image_shape,
            center=shoulder,
            major_axis=shoulder_cap_axis,
            major_radius=shoulder_cap_major_radius,
            minor_radius=shoulder_cap_minor_radius,
        )
        distal_barrier_mask = _build_limb_distal_barrier(
            image_shape=image_shape,
            middle=elbow,
            distal=wrist,
            radius=radius,
        )
        synthetic_barrier_mask = (
            proximal_barrier_mask
            | distal_barrier_mask
        )

        base_bbox, crop_bbox = _limb_support_bboxes(
            semantic_support_mask=semantic_support_mask,
            image_shape=image_shape,
            crop_expansion=crop_expansion,
        )

        geometries.append(
            LimbCropGeometry(
                side=side,
                proximal_name='shoulder',
                middle_name='elbow',
                distal_name='wrist',
                proximal_point=shoulder,
                middle_point=elbow,
                distal_point=wrist,
                base_bbox=base_bbox,
                crop_bbox=crop_bbox,
                semantic_region_mask=semantic_bool,
                semantic_support_mask=semantic_support_mask,
                tube_mask=tube_mask,
                skeleton_mask=skeleton_mask,
                synthetic_barrier_mask=synthetic_barrier_mask,
                proximal_circle_mask=proximal_circle_mask,
                proximal_quadrant_mask=proximal_quadrant_mask,
                tube_radius=radius,
            )
        )

    if not geometries:
        raise RuntimeError(
            'Could not derive geometry for any requested arm.'
        )

    return geometries


def build_leg_crop_geometries(
    *,
    resolve_landmark: LandmarkResolver,
    image_shape: tuple[int, ...],
    semantic_region: np.ndarray,
    which: LimbSelection = 'both',
    expansion: float = 1.0,
    crop_expansion: float = _LIMB_GEOMETRY_CROP_EXPANSION_DEFAULT,
) -> list[LimbCropGeometry]:
    """
    Build semantic-prior crop geometries for one or both anatomical legs.

    Required landmark names are:

    - ``left_hip``, ``left_knee``, ``left_ankle``;
    - ``right_hip``, ``right_knee``, ``right_ankle``.

    Shoulder landmarks and ``nose`` are optional torso-side references.
    ``semantic_region`` should contain leg skin and lower clothing context.

    Parameters
    ----------
    resolve_landmark : LandmarkResolver
        Callable that resolves landmark names to full-frame image coordinates and
        optional pseudo-3D image-space coordinates. Two-dimensional coordinates
        are used for raster geometry. Leg radius fallback and clamp bounds
        currently use the projected two-dimensional hip distance; optional
        pseudo-3D coordinates are not used by this geometry builder.

    image_shape : tuple[int, ...]
        Source image shape. The first two dimensions are used as ``(H, W)``.

    semantic_region : np.ndarray
        Full-frame boolean or binary semantic support region. It should include
        leg skin and lower clothing before geometric restriction.

    which : {'both', 'anatomical-left', 'anatomical-right'}, default='both'
        Anatomical leg side or sides to build.

    expansion : float, default=1.0
        Multiplicative expansion applied to the estimated tube/joint radius.

    crop_expansion : float, default=_LIMB_GEOMETRY_CROP_EXPANSION_DEFAULT
        Proportional margin applied to the tight support bbox to derive the
        processing crop bbox.

    Returns
    -------
    list[LimbCropGeometry]
        One geometry per requested anatomical leg that can be derived from the
        available landmarks and semantic support.

    Raises
    ------
    ValueError
        If input shapes, side selection, or expansion values are invalid.

    RuntimeError
        If required hip landmarks are missing/degenerate or no requested leg
        geometry can be derived.
    """
    h, w = _validate_limb_geometry_inputs(
        image_shape=image_shape,
        semantic_region=semantic_region,
        which=which,
        expansion=expansion,
        crop_expansion=crop_expansion,
    )
    semantic_bool = semantic_region.astype(bool, copy=False)

    left_hip_landmark = resolve_landmark('left_hip')
    right_hip_landmark = resolve_landmark('right_hip')

    left_hip = _landmark_xy(left_hip_landmark)
    right_hip = _landmark_xy(right_hip_landmark)

    if left_hip is None or right_hip is None:
        raise RuntimeError(
            'Leg crop geometry requires both hip landmarks.'
        )

    pelvis_axis = right_hip - left_hip
    pelvis_width = float(np.linalg.norm(pelvis_axis))

    if pelvis_width < 4.0:
        raise RuntimeError(
            'Hip landmarks are too close for leg crop geometry.'
        )

    pelvis_mid = 0.5 * (left_hip + right_hip)
    pelvis_axis_len = max(pelvis_width, 1e-6)

    torso_references = [
        p
        for p in (
            _limb_point(resolve_landmark, 'left_shoulder'),
            _limb_point(resolve_landmark, 'right_shoulder'),
        )
        if p is not None
    ]
    if torso_references:
        torso_reference = np.mean(
            np.stack(torso_references, axis=0),
            axis=0,
        )
    else:
        torso_reference = _limb_point(resolve_landmark, 'nose')

    if torso_reference is None:
        torso_reference = (
            pelvis_mid
            + np.asarray([0.0, -pelvis_width], dtype=np.float32)
        )

    torso_cross = _cross2(
        pelvis_axis,
        torso_reference - pelvis_mid,
    )

    yy, xx = np.mgrid[0:h, 0:w]
    rel_mid_x = xx - float(pelvis_mid[0])
    rel_mid_y = yy - float(pelvis_mid[1])
    grid_cross_from_pelvis = (
        float(pelvis_axis[0]) * rel_mid_y
        - float(pelvis_axis[1]) * rel_mid_x
    )

    if abs(torso_cross) <= 1e-3:
        lower_halfplane = np.ones((h, w), dtype=bool)
    else:
        lower_side = -1.0 if torso_cross >= 0.0 else 1.0
        lower_halfplane = (
            lower_side * grid_cross_from_pelvis
        ) >= -_LIMB_PROXIMAL_HALFPLANE_BACKTRACK_RATIO * pelvis_axis_len

    geometries: list[LimbCropGeometry] = []

    for side in _selected_limb_sides(which):
        prefix = 'left' if side == 'anatomical-left' else 'right'

        hip = left_hip if side == 'anatomical-left' else right_hip
        other_hip = right_hip if side == 'anatomical-left' else left_hip
        knee = _limb_point(resolve_landmark, f'{prefix}_knee')
        ankle = _limb_point(resolve_landmark, f'{prefix}_ankle')

        segments = _limb_chain_segments(hip, knee, ankle)
        if not segments:
            continue

        radius = _estimate_limb_proximal_radius(
            semantic_bool,
            proximal=hip,
            opposite_proximal=other_hip,
            middle=knee,
            image_shape=image_shape,
            bounds_width=pelvis_width,
            fallback_scale=_LEG_FALLBACK_RADIUS_PELVIS_WIDTH_RATIO,
            min_scale=_LEG_MIN_RADIUS_PELVIS_WIDTH_RATIO,
            max_width_scale=_LEG_MAX_RADIUS_PELVIS_WIDTH_RATIO,
            max_segment_scale=_LEG_MAX_RADIUS_HIP_KNEE_LENGTH_RATIO,
            expansion_constant=_LEG_HIP_RADIUS_EXPANSION,
        )

        radius *= max(1.0, float(expansion))

        tube_mask = _build_limb_tube_mask(
            image_shape=image_shape,
            segments=segments,
            radius=radius,
        )
        skeleton_mask = build_limb_skeleton_mask(
            image_shape=image_shape,
            segments=segments,
            points=(hip, knee, ankle),
        )

        # The proximal barrier is drawn on the nominal cap, while the workspace
        # cap gets a small 2px margin. This gives the dilated barrier room to
        # separate regions instead of coinciding with the workspace boundary.
        proximal_circle_mask = _circle_mask(
            xx=xx,
            yy=yy,
            center=hip,
            radius=radius + 2.0,
        )

        outward_vector = hip - other_hip
        outward_length = float(np.linalg.norm(outward_vector))
        if outward_length < 1e-6:
            continue

        outward_unit = outward_vector / outward_length
        rel_x = xx - float(hip[0])
        rel_y = yy - float(hip[1])
        outward_projection = (
            rel_x * float(outward_unit[0])
            + rel_y * float(outward_unit[1])
        )
        outward_halfplane = (
            outward_projection
            >= -_LIMB_PROXIMAL_HALFPLANE_BACKTRACK_RATIO * radius
        )

        if abs(torso_cross) <= 1e-3:
            limb_lower_halfplane = lower_halfplane
        else:
            lower_side = -1.0 if torso_cross >= 0.0 else 1.0
            limb_lower_halfplane = (
                lower_side * grid_cross_from_pelvis
            ) >= (
                -_LIMB_PROXIMAL_HALFPLANE_BACKTRACK_RATIO
                * radius
                * pelvis_axis_len
            )

        proximal_quadrant_mask = (
            semantic_bool
            & limb_lower_halfplane
            & outward_halfplane
        )

        semantic_support_mask = (
            semantic_bool
            & (
                tube_mask
                | proximal_circle_mask
                | proximal_quadrant_mask
            )
        )

        if ankle is not None and knee is not None:
            lower_leg_vector = ankle - knee
            lower_leg_length = float(np.linalg.norm(lower_leg_vector))
            if lower_leg_length >= 2.0:
                lower_leg_unit = lower_leg_vector / lower_leg_length
                after_ankle = (
                    (xx - float(ankle[0])) * float(lower_leg_unit[0])
                    + (yy - float(ankle[1])) * float(lower_leg_unit[1])
                ) > radius * _LEG_AFTER_ANKLE_TRIM_RADIUS_RATIO
                semantic_support_mask &= ~after_ankle
                semantic_support_mask |= (
                    semantic_bool & proximal_circle_mask
                )

        if not np.any(semantic_support_mask):
            continue

        proximal_barrier_mask = _circle_outline_mask(
            image_shape=image_shape,
            center=hip,
            radius=radius,
        )
        distal_barrier_mask = _build_limb_distal_barrier(
            image_shape=image_shape,
            middle=knee,
            distal=ankle,
            radius=radius,
        )
        synthetic_barrier_mask = (
            proximal_barrier_mask
            | distal_barrier_mask
        )

        base_bbox, crop_bbox = _limb_support_bboxes(
            semantic_support_mask=semantic_support_mask,
            image_shape=image_shape,
            crop_expansion=crop_expansion,
        )

        geometries.append(
            LimbCropGeometry(
                side=side,
                proximal_name='hip',
                middle_name='knee',
                distal_name='ankle',
                proximal_point=hip,
                middle_point=knee,
                distal_point=ankle,
                base_bbox=base_bbox,
                crop_bbox=crop_bbox,
                semantic_region_mask=semantic_bool,
                semantic_support_mask=semantic_support_mask,
                tube_mask=tube_mask,
                skeleton_mask=skeleton_mask,
                synthetic_barrier_mask=synthetic_barrier_mask,
                proximal_circle_mask=proximal_circle_mask,
                proximal_quadrant_mask=proximal_quadrant_mask,
                tube_radius=radius,
            )
        )

    if not geometries:
        raise RuntimeError(
            'Could not derive geometry for any requested leg.'
        )

    return geometries


def _validate_limb_geometry_inputs(
    *,
    image_shape: tuple[int, ...],
    semantic_region: np.ndarray,
    which: LimbSelection,
    expansion: float,
    crop_expansion: float,
) -> tuple[int, int]:
    """
    Validate shared limb-geometry inputs and return image dimensions.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Source image shape. The first two dimensions must be positive
        ``(H, W)`` values.
    semantic_region : np.ndarray
        Full-frame semantic support mask aligned with ``image_shape``.
    which : LimbSelection
        Requested anatomical side selection.
    expansion : float
        Radius expansion factor. Must be greater than zero.
    crop_expansion : float
        Crop bbox expansion factor. Must be non-negative.

    Returns
    -------
    tuple[int, int]
        Image height and width as integers.

    Raises
    ------
    ValueError
        If dimensions, mask shape, side selection, or expansion values are
        invalid.
    """
    if len(image_shape) < 2:
        raise ValueError(
            f'Invalid image_shape={image_shape!r}; expected at least (H, W).'
        )

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(
            f'Invalid image dimensions {(h, w)!r}.'
        )

    if semantic_region.shape[:2] != (h, w):
        raise ValueError(
            'semantic_region shape does not match image_shape: '
            f'{semantic_region.shape[:2]!r} != {(h, w)!r}.'
        )

    if which not in (
        'both',
        'anatomical-left',
        'anatomical-right',
    ):
        raise ValueError(
            f'Invalid which={which!r}; expected anatomical side or "both".'
        )

    if float(expansion) <= 0.0:
        raise ValueError(
            f'Invalid expansion={expansion!r}; expected > 0.'
        )

    if float(crop_expansion) < 0.0:
        raise ValueError(
            f'Invalid crop_expansion={crop_expansion!r}; expected >= 0.'
        )

    return h, w


def _selected_limb_sides(which: LimbSelection) -> list[AnatomicalSide]:
    """
    Expand a limb side selection to concrete anatomical sides.

    Parameters
    ----------
    which : LimbSelection
        ``'both'`` or one anatomical side.

    Returns
    -------
    list[AnatomicalSide]
        Anatomical sides to process in deterministic left/right order.
    """
    if which == 'both':
        return ['anatomical-left', 'anatomical-right']
    return [which]


def _limb_point(
    resolve_landmark: LandmarkResolver,
    name: str,
) -> Optional[np.ndarray]:
    """
    Resolve and validate one full-frame limb landmark.

    Parameters
    ----------
    resolve_landmark : LandmarkResolver
        Callable that resolves landmark names to image coordinates and optional
        pseudo-3D coordinates.

    name : str
        Landmark name to resolve.

    Returns
    -------
    np.ndarray or None
        ``float32`` image coordinates with shape ``(2,)``, or ``None`` when
        the landmark is unavailable, malformed, non-finite, or outside the
        valid positive image-coordinate domain.
    """
    return _landmark_xy(resolve_landmark(name))


def _landmark_xy(
    landmark: Optional[ResolvedLandmark],
) -> Optional[np.ndarray]:
    """
    Return validated two-dimensional coordinates from a resolved landmark.

    Parameters
    ----------
    landmark : ResolvedLandmark or None
        Resolved landmark to inspect.

    Returns
    -------
    np.ndarray or None
        ``float32`` coordinates with shape ``(2,)``, or ``None`` when invalid.
    """
    if landmark is None:
        return None

    xy = np.asarray(
        landmark.xy,
        dtype=np.float32,
    ).reshape(-1)

    if xy.size != 2 or not np.all(np.isfinite(xy)):
        return None

    if float(xy[0]) < 0.0 or float(xy[1]) < 0.0:
        return None

    return xy.copy()


def _landmark_distance(
    first: Optional[ResolvedLandmark],
    second: Optional[ResolvedLandmark],
) -> Optional[float]:
    """
    Measure the distance between two resolved pose landmarks.

    Pseudo-3D image-space coordinates are used only when they are valid for
    both landmarks. Otherwise, the function falls back to their full-frame
    two-dimensional image coordinates.

    Parameters
    ----------
    first : ResolvedLandmark or None
        First resolved landmark.

    second : ResolvedLandmark or None
        Second resolved landmark.

    Returns
    -------
    float or None
        Euclidean landmark distance. The value is measured in pseudo-3D
        image-space pixels when valid depth information is available for both
        landmarks, and in two-dimensional image pixels otherwise. ``None`` is
        returned when either landmark or its required two-dimensional
        coordinates are invalid.
    """
    if first is None or second is None:
        return None

    first_xy = _landmark_xy(first)
    second_xy = _landmark_xy(second)

    if first_xy is None or second_xy is None:
        return None

    first_xyz = first.xyz_px
    second_xyz = second.xyz_px

    if first_xyz is not None and second_xyz is not None:
        first_xyz_array = np.asarray(
            first_xyz,
            dtype=np.float32,
        ).reshape(-1)
        second_xyz_array = np.asarray(
            second_xyz,
            dtype=np.float32,
        ).reshape(-1)

        if (
            first_xyz_array.size == 3
            and second_xyz_array.size == 3
            and np.all(np.isfinite(first_xyz_array))
            and np.all(np.isfinite(second_xyz_array))
        ):
            return float(np.linalg.norm(
                second_xyz_array - first_xyz_array
            ))

    return float(np.linalg.norm(
        second_xy - first_xy
    ))


def _limb_chain_segments(
    proximal: np.ndarray,
    middle: Optional[np.ndarray],
    distal: Optional[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build non-degenerate segment pairs from a proximal-to-distal chain.

    Parameters
    ----------
    proximal : np.ndarray
        Required proximal landmark.
    middle : np.ndarray | None
        Optional middle landmark.
    distal : np.ndarray | None
        Optional distal landmark.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Consecutive chain segments whose length is at least two pixels.
    """
    chain = [proximal]
    if middle is not None:
        chain.append(middle)
    if distal is not None:
        chain.append(distal)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(chain, chain[1:]):
        if float(np.linalg.norm(end - start)) >= 2.0:
            segments.append((start, end))

    return segments


def _arm_shoulder_cap_radii(
    *,
    tube_radius: float,
    shoulder_width: float,
) -> tuple[float, float]:
    """
    Return ``(major_radius, minor_radius)`` for the shoulder cap ellipse.
    """
    minor_radius = max(
        float(tube_radius),
        8.0,
        float(shoulder_width) * _ARM_SHOULDER_CAP_MINOR_WIDTH_RATIO,
    )
    major_radius = (
        minor_radius
        / max(float(_ARM_SHOULDER_CAP_MINOR_RATIO), 1e-6)
    )
    return (
        float(major_radius),
        float(minor_radius),
    )


def _estimate_limb_proximal_radius(
    semantic_region: np.ndarray,
    *,
    proximal: np.ndarray,
    opposite_proximal: np.ndarray,
    middle: Optional[np.ndarray],
    image_shape: tuple[int, ...],
    bounds_width: float,
    fallback_scale: float,
    min_scale: float,
    max_width_scale: float,
    max_segment_scale: float,
    expansion_constant: float,
) -> float:
    """
    Estimate a limb tube radius near the proximal joint.

    The estimate probes the semantic support outward from the proximal landmark
    and clamps the observed support width using anatomical fallback limits.
    This is the single-point strategy used by legs, where the hip/pelvis area is
    the intended proximal-width reference.

    Parameters
    ----------
    semantic_region : np.ndarray
        Full-frame boolean semantic support mask.
    proximal : np.ndarray
        Proximal joint coordinates.
    opposite_proximal : np.ndarray
        Opposite-side proximal joint coordinates.
    middle : np.ndarray | None
        Middle limb joint coordinates, used to limit the radius relative to the
        first limb segment.
    image_shape : tuple[int, ...]
        Source image shape.
    bounds_width : float
        Already-resolved anatomical proximal-pair width in pixels used only for
        fallback and min/max radius bounds. Callers may pass a pseudo-3D
        image-space shoulder/hip distance here. The semantic probe still uses
        the 2D projected landmarks because masks live in image space.
    fallback_scale : float
        Fraction of proximal-pair width used when semantic probing fails.
    min_scale : float
        Minimum radius fraction of proximal-pair width. This is the canonical
        lower bound for the proximal joint radius; callers should pass the
        shoulder or pelvis ratio they want instead of applying a second minimum
        after this function returns.
    max_width_scale : float
        Maximum radius fraction of proximal-pair width.
    max_segment_scale : float
        Maximum radius fraction of first segment length.
    expansion_constant : float
        Multiplier applied to the probed outward support distance.

    Returns
    -------
    float
        Estimated radius in pixels.
    """
    external = proximal - opposite_proximal
    external_len = float(np.linalg.norm(external))
    bounds = _limb_radius_bounds(
        proximal=proximal,
        opposite_proximal=opposite_proximal,
        middle=middle,
        bounds_width=bounds_width,
        fallback_scale=fallback_scale,
        min_scale=min_scale,
        max_width_scale=max_width_scale,
        max_segment_scale=max_segment_scale,
        expansion_constant=expansion_constant,
    )
    if external_len < 1.0:
        return bounds.fallback_radius

    external_unit = external / external_len
    last_inside = _probe_last_semantic_distance_outward(
        semantic_region=semantic_region,
        image_shape=image_shape,
        origin=proximal,
        outward_unit=external_unit,
        max_probe=bounds.search_limit,
    )

    if last_inside is None or last_inside < 2.0:
        return bounds.fallback_radius

    expanded = last_inside * float(expansion_constant)
    return float(np.clip(expanded, bounds.min_radius, bounds.max_radius))


def _estimate_arm_proximal_band_radius(
    semantic_region: np.ndarray,
    *,
    side: AnatomicalSide,
    proximal: np.ndarray,
    opposite_proximal: np.ndarray,
    middle: Optional[np.ndarray],
    image_shape: tuple[int, ...],
    bounds_width: float,
    fallback_scale: float,
    min_scale: float,
    max_width_scale: float,
    max_segment_scale: float,
    expansion_constant: float,
) -> float:
    """
    Estimate the upper-arm tube radius from the visible semantic support.

    Two geometric regimes are used.

    In the normal regime, semantic support is measured along cross sections
    perpendicular to the shoulder-to-elbow segment. Only the anatomical outward
    half of each cross section is inspected. This follows the visible upper-arm
    orientation while preventing measurements from extending through the torso.

    In the strongly internally rotated regime, the segment-normal strategy is
    disabled because its outward normal may point through the torso when the arm
    overlaps the body. Semantic support is instead measured from the vertical
    image-space axis passing through the shoulder, again only toward the
    anatomical outside of the body.

    The observed semantic distance is expanded and then clamped using the shared
    anatomical radius bounds. If the required landmarks or semantic measurements
    are unavailable, the bounded anatomical fallback radius is returned.

    Parameters
    ----------
    semantic_region : np.ndarray
        Full-frame boolean semantic support mask.

    side : AnatomicalSide
        Anatomical arm side. The side is used to normalize shoulder-to-elbow
        rotation so both arms follow the same angular convention.

    proximal : np.ndarray
        Shoulder coordinates in full-frame image space.

    opposite_proximal : np.ndarray
        Opposite shoulder coordinates in full-frame image space.

    middle : np.ndarray or None
        Elbow coordinates in full-frame image space.

    image_shape : tuple[int, ...]
        Source image shape.

    bounds_width : float
        Perspective-aware shoulder width used only for fallback and clamp
        bounds.

    fallback_scale : float
        Fraction of shoulder width used when semantic measurement fails.

    min_scale : float
        Minimum radius as a fraction of shoulder width.

    max_width_scale : float
        Maximum radius as a fraction of shoulder width.

    max_segment_scale : float
        Maximum radius as a fraction of projected upper-arm length.

    expansion_constant : float
        Multiplier applied to the measured semantic distance.

    Returns
    -------
    float
        Estimated arm tube radius in pixels.
    """
    bounds = _limb_radius_bounds(
        proximal=proximal,
        opposite_proximal=opposite_proximal,
        middle=middle,
        bounds_width=bounds_width,
        fallback_scale=fallback_scale,
        min_scale=min_scale,
        max_width_scale=max_width_scale,
        max_segment_scale=max_segment_scale,
        expansion_constant=expansion_constant,
    )

    if middle is None:
        return bounds.fallback_radius

    upper_arm = np.asarray(
        middle,
        dtype=np.float32,
    ) - np.asarray(
        proximal,
        dtype=np.float32,
    )

    upper_arm_length = float(np.linalg.norm(upper_arm))
    if upper_arm_length < 2.0:
        return bounds.fallback_radius

    rotation = _arm_rotation_degrees(
        shoulder=proximal,
        opposite_shoulder=opposite_proximal,
        elbow=middle,
    )

    band_half_height = 0.5 * bounds.min_radius

    if rotation > _ARM_RADIUS_VERTICAL_AXIS_THRESHOLD_DEGREES:
        observed_radius = _estimate_arm_radius_from_shoulder_vertical_axis(
            semantic_region,
            shoulder=proximal,
            opposite_shoulder=opposite_proximal,
            image_shape=image_shape,
            max_probe=bounds.search_limit,
            band_half_height=band_half_height,
        )
    else:
        observed_radius = _estimate_arm_radius_from_outward_normals(
            semantic_region,
            shoulder=proximal,
            opposite_shoulder=opposite_proximal,
            elbow=middle,
            image_shape=image_shape,
            max_probe=bounds.search_limit,
        )

    if observed_radius is None or observed_radius < 2.0:
        return bounds.fallback_radius

    expanded_radius = (
        float(observed_radius)
        * float(expansion_constant)
    )

    return float(np.clip(
        expanded_radius,
        bounds.min_radius,
        bounds.max_radius,
    ))


def _arm_rotation_degrees(
    *,
    shoulder: np.ndarray,
    opposite_shoulder: np.ndarray,
    elbow: np.ndarray,
) -> float:
    """
    Return the shoulder-to-elbow rotation in a shoulder-relative frame.

    The angle is measured from the shoulder axis pointing toward the opposite
    shoulder. Positive rotation proceeds toward the upper side of the image.

    The resulting convention is:

    - ``0`` or ``360`` degrees points toward the opposite shoulder;
    - ``90`` degrees points upward;
    - ``180`` degrees points anatomically outward;
    - ``270`` degrees points downward.

    Because the reference axis is derived from the shoulder pair, the same
    convention applies automatically to both anatomical arms.

    Parameters
    ----------
    shoulder : np.ndarray
        Target shoulder coordinates in full-frame image space.

    opposite_shoulder : np.ndarray
        Opposite shoulder coordinates in full-frame image space.

    elbow : np.ndarray
        Target elbow coordinates in full-frame image space.

    Returns
    -------
    float
        Rotation angle normalized to the range ``[0, 360)``.

    Raises
    ------
    RuntimeError
        If the shoulder axis or upper-arm segment is degenerate.
    """
    shoulder_xy = np.asarray(
        shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]
    opposite_xy = np.asarray(
        opposite_shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]
    elbow_xy = np.asarray(
        elbow,
        dtype=np.float32,
    ).reshape(-1)[:2]

    inward = opposite_xy - shoulder_xy
    inward_length = float(np.linalg.norm(inward))
    if inward_length < 1.0:
        raise RuntimeError(
            'Shoulder axis is too short to measure arm rotation.'
        )

    upper_arm = elbow_xy - shoulder_xy
    upper_arm_length = float(np.linalg.norm(upper_arm))
    if upper_arm_length < 1.0:
        raise RuntimeError(
            'Upper-arm segment is too short to measure arm rotation.'
        )

    inward_unit = inward / inward_length

    first_perpendicular = np.asarray(
        [
            float(inward_unit[1]),
            -float(inward_unit[0]),
        ],
        dtype=np.float32,
    )
    second_perpendicular = -first_perpendicular

    # Select the shoulder-axis perpendicular that points toward the upper side
    # of the image.
    upward_unit = (
        first_perpendicular
        if float(first_perpendicular[1]) <= float(second_perpendicular[1])
        else second_perpendicular
    )

    inward_projection = float(np.dot(
        upper_arm,
        inward_unit,
    ))
    upward_projection = float(np.dot(
        upper_arm,
        upward_unit,
    ))

    angle = math.degrees(
        math.atan2(
            upward_projection,
            inward_projection,
        )
    )

    return float(angle % 360.0)


def _estimate_arm_radius_from_outward_normals(
    semantic_region: np.ndarray,
    *,
    shoulder: np.ndarray,
    opposite_shoulder: np.ndarray,
    elbow: np.ndarray,
    image_shape: tuple[int, ...],
    max_probe: float,
) -> Optional[float]:
    """
    Measure upper-arm semantic width along outward segment normals.

    Sample origins are distributed along the shoulder-to-elbow segment. At each
    origin, the function constructs the two perpendicular directions of the
    upper-arm segment and selects the one most aligned with the anatomical
    outside defined by the shoulder pair. Semantic support is then probed only
    along that outward normal.

    Parameters
    ----------
    semantic_region : np.ndarray
        Full-frame semantic support mask.

    shoulder : np.ndarray
        Target shoulder coordinates.

    opposite_shoulder : np.ndarray
        Opposite shoulder coordinates.

    elbow : np.ndarray
        Target elbow coordinates.

    image_shape : tuple[int, ...]
        Source image shape.

    max_probe : float
        Maximum semantic probing distance.

    Returns
    -------
    float or None
        Largest valid outward semantic distance observed along the upper-arm
        band, or ``None`` when no usable measurement is found.
    """
    shoulder_xy = np.asarray(
        shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]
    opposite_xy = np.asarray(
        opposite_shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]
    elbow_xy = np.asarray(
        elbow,
        dtype=np.float32,
    ).reshape(-1)[:2]

    upper_arm = elbow_xy - shoulder_xy
    upper_arm_length = float(np.linalg.norm(upper_arm))
    if upper_arm_length < 2.0:
        return None

    outward = shoulder_xy - opposite_xy
    outward_length = float(np.linalg.norm(outward))
    if outward_length < 1.0:
        return None

    upper_arm_unit = upper_arm / upper_arm_length
    outward_unit = outward / outward_length

    first_normal = np.asarray(
        [
            -float(upper_arm_unit[1]),
            float(upper_arm_unit[0]),
        ],
        dtype=np.float32,
    )
    second_normal = -first_normal

    probe_direction = (
        first_normal
        if float(np.dot(first_normal, outward_unit)) >= 0.0
        else second_normal
    )

    sample_count = max(
        2,
        int(round(upper_arm_length)) + 1,
    )

    probe_origins = [
        shoulder_xy + upper_arm * float(fraction)
        for fraction in np.linspace(
            0.0,
            1.0,
            num=sample_count,
            endpoint=False,
            dtype=np.float32,
        )
    ]

    observed_radius: Optional[float] = None

    for origin in probe_origins:
        distance = _probe_last_semantic_distance_outward(
            semantic_region=semantic_region,
            image_shape=image_shape,
            origin=origin,
            outward_unit=probe_direction,
            max_probe=max_probe,
        )

        if distance is None:
            continue

        if observed_radius is None or distance > observed_radius:
            observed_radius = float(distance)

    return observed_radius


def _estimate_arm_radius_from_shoulder_vertical_axis(
    semantic_region: np.ndarray,
    *,
    shoulder: np.ndarray,
    opposite_shoulder: np.ndarray,
    image_shape: tuple[int, ...],
    max_probe: float,
    band_half_height: float,
) -> Optional[float]:
    """
    Measure the maximum outward semantic distance from the shoulder vertical axis.

    Horizontal probes originate from a short vertical band centered on the
    shoulder and extend only toward the anatomical outside of the body.

    The band extends by ``band_half_height`` pixels above and below the shoulder.
    This prevents semantic widening near the elbow or overlapping forearm from
    determining the arm tube radius.

    Parameters
    ----------
    semantic_region : np.ndarray
        Full-frame semantic support mask.

    shoulder : np.ndarray
        Target shoulder coordinates.

    opposite_shoulder : np.ndarray
        Opposite shoulder coordinates, used to identify the anatomical outer
        side of the body.

    image_shape : tuple[int, ...]
        Source image shape.

    max_probe : float
        Maximum horizontal probing distance from the shoulder vertical axis.

    band_half_height : float
        Half-height of the vertical probe band centered on the shoulder.

    Returns
    -------
    float or None
        Maximum valid outward semantic distance from the shoulder vertical axis,
        or ``None`` when no usable semantic support is found.
    """
    shoulder_xy = np.asarray(
        shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]
    opposite_xy = np.asarray(
        opposite_shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]

    outward_x = float(
        shoulder_xy[0] - opposite_xy[0]
    )
    if abs(outward_x) < 1.0:
        return None

    outward_unit = np.asarray(
        [math.copysign(1.0, outward_x), 0.0],
        dtype=np.float32,
    )

    probe_origins = _arm_proximal_band_probe_origins(
        shoulder=shoulder_xy,
        band_half_height=band_half_height,
        image_shape=image_shape,
    )

    observed_radius: Optional[float] = None

    for origin in probe_origins:
        distance = _probe_last_semantic_distance_outward(
            semantic_region=semantic_region,
            image_shape=image_shape,
            origin=origin,
            outward_unit=outward_unit,
            max_probe=max_probe,
        )

        if distance is None:
            continue

        if observed_radius is None or distance > observed_radius:
            observed_radius = float(distance)

    return observed_radius


def _limb_radius_bounds(
    *,
    proximal: np.ndarray,
    opposite_proximal: np.ndarray,
    middle: Optional[np.ndarray],
    bounds_width: float,
    fallback_scale: float,
    min_scale: float,
    max_width_scale: float,
    max_segment_scale: float,
    expansion_constant: float,
) -> _LimbRadiusBounds:
    """
    Build explicit radius guardrails from anatomical dimensions.

    ``bounds_width`` is supplied by the caller and is not part of semantic
    measurement. It only constrains the final observed radius and provides a
    fallback when the semantic probe cannot find a useful support width.
    """
    projected_width = float(
        np.linalg.norm(proximal - opposite_proximal)
    )
    first_segment_length = (
        float(np.linalg.norm(middle - proximal))
        if middle is not None
        else projected_width
    )
    safe_bounds_width = (
        float(bounds_width)
        if math.isfinite(float(bounds_width)) and float(bounds_width) > 1.0
        else projected_width
    )
    min_radius = max(
        _LIMB_MIN_RADIUS_PX,
        safe_bounds_width * float(min_scale),
    )
    max_radius = max(
        min_radius + 1.0,
        min(
            safe_bounds_width * float(max_width_scale),
            first_segment_length * float(max_segment_scale),
        ),
    )
    fallback_radius = float(np.clip(
        max(
            _LIMB_FALLBACK_MIN_RADIUS_PX,
            safe_bounds_width * float(fallback_scale),
        ),
        min_radius,
        max_radius,
    ))
    search_limit = max(
        _LIMB_RADIUS_SEARCH_MIN_PX,
        max_radius / max(float(expansion_constant), 1e-6),
    )
    return _LimbRadiusBounds(
        min_radius=float(min_radius),
        max_radius=float(max_radius),
        fallback_radius=fallback_radius,
        search_limit=float(search_limit),
    )


def _arm_proximal_band_probe_origins(
    *,
    shoulder: np.ndarray,
    band_half_height: float,
    image_shape: tuple[int, ...],
) -> list[np.ndarray]:
    """
    Return dense probe origins along a vertical band centered on the shoulder.

    The band extends by ``band_half_height`` pixels above and below the shoulder.
    One origin is generated for each image row intersected by the band.

    Parameters
    ----------
    shoulder : np.ndarray
        Shoulder coordinates in full-frame image space.

    band_half_height : float
        Vertical distance covered above and below the shoulder.

    image_shape : tuple[int, ...]
        Source image shape used to clip the band to valid image coordinates.

    Returns
    -------
    list[np.ndarray]
        Probe origins ordered from the top to the bottom of the shoulder band.
    """
    h, _ = image_shape[:2]

    shoulder_xy = np.asarray(
        shoulder,
        dtype=np.float32,
    ).reshape(-1)[:2]

    safe_half_height = max(
        0.0,
        float(band_half_height),
    )

    y_start = max(
        0,
        int(math.floor(
            float(shoulder_xy[1]) - safe_half_height
        )),
    )
    y_end = min(
        h - 1,
        int(math.ceil(
            float(shoulder_xy[1]) + safe_half_height
        )),
    )

    return [
        np.asarray(
            [float(shoulder_xy[0]), float(y)],
            dtype=np.float32,
        )
        for y in range(y_start, y_end + 1)
    ]


def _probe_last_semantic_distance_outward(
    *,
    semantic_region: np.ndarray,
    image_shape: tuple[int, ...],
    origin: np.ndarray,
    outward_unit: np.ndarray,
    max_probe: float,
) -> Optional[float]:
    """
    Probe outward from one image-space origin and return the last semantic hit.
    """
    h, w = image_shape[:2]
    origin_xy = np.asarray(origin, dtype=np.float32).reshape(-1)[:2]

    last_inside: Optional[float] = None
    for distance in np.linspace(
        0.0,
        max_probe,
        num=max(8, int(np.ceil(max_probe)) + 1),
    ):
        point = origin_xy + outward_unit * float(distance)
        px = int(round(float(point[0])))
        py = int(round(float(point[1])))

        if px < 0 or px >= w or py < 0 or py >= h:
            break

        if bool(semantic_region[py, px]):
            last_inside = float(distance)
        elif last_inside is not None:
            break

    return last_inside


def _build_limb_tube_mask(
    *,
    image_shape: tuple[int, ...],
    segments: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
) -> np.ndarray:
    """
    Rasterize limb chain segments into a tube mask.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Source image shape.
    segments : list[tuple[np.ndarray, np.ndarray]]
        Proximal-to-distal limb segments.
    radius : float
        Capsule radius in pixels.

    Returns
    -------
    np.ndarray
        Boolean full-frame tube mask.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    for start, end in segments:
        mask |= segment_capsule_mask(
            image_shape,
            start,
            end,
            radius,
        )

    return mask


def build_limb_skeleton_mask(
    *,
    image_shape: tuple[int, ...],
    segments: list[tuple[np.ndarray, np.ndarray]],
    points: tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ],
) -> np.ndarray:
    """
    Rasterize a one-pixel limb skeleton through chain segments and joints.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Source image shape.
    segments : list[tuple[np.ndarray, np.ndarray]]
        Proximal-to-distal limb segments.
    points : tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
        Proximal, middle, and distal landmarks. Available points are reinforced
        as small filled circles.

    Returns
    -------
    np.ndarray
        Boolean full-frame skeleton mask.
    """
    h, w = image_shape[:2]
    skeleton = np.zeros((h, w), dtype=np.uint8)

    for start, end in segments:
        cv2.line(
            skeleton,
            tuple(int(value) for value in np.rint(start)),
            tuple(int(value) for value in np.rint(end)),
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )

    for point in points:
        if point is None:
            continue
        cv2.circle(
            skeleton,
            tuple(int(value) for value in np.rint(point)),
            1,
            255,
            thickness=-1,
            lineType=cv2.LINE_8,
        )

    return skeleton > 0


def _build_limb_distal_barrier(
    *,
    image_shape: tuple[int, ...],
    middle: Optional[np.ndarray],
    distal: Optional[np.ndarray],
    radius: float,
) -> np.ndarray:
    """
    Build the synthetic distal barrier for limb flood-fill.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Source image shape.
    middle : np.ndarray | None
        Middle joint coordinates.
    distal : np.ndarray | None
        Distal joint coordinates.
    radius : float
        Barrier half-width/radius in pixels.

    Returns
    -------
    np.ndarray
        Boolean full-frame barrier mask. The distal barrier is a transverse
        line when middle and distal landmarks are available.
    """
    h, w = image_shape[:2]
    barriers = np.zeros((h, w), dtype=np.uint8)

    if middle is not None and distal is not None:
        distal_axis = distal - middle
        distal_axis_length = float(np.linalg.norm(distal_axis))

        if distal_axis_length >= 2.0:
            distal_unit = distal_axis / distal_axis_length
            distal_normal = np.asarray(
                [-distal_unit[1], distal_unit[0]],
                dtype=np.float32,
            )
            barrier_start = distal - distal_normal * radius
            barrier_end = distal + distal_normal * radius

            cv2.line(
                barriers,
                tuple(int(value) for value in np.rint(barrier_start)),
                tuple(int(value) for value in np.rint(barrier_end)),
                255,
                thickness=1,
                lineType=cv2.LINE_8,
            )

    return barriers > 0


def _circle_mask(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Rasterize a filled circle from precomputed coordinate grids.

    Parameters
    ----------
    xx : np.ndarray
        Grid of x coordinates.
    yy : np.ndarray
        Grid of y coordinates.
    center : np.ndarray
        Circle center as ``(x, y)``.
    radius : float
        Circle radius in pixels.

    Returns
    -------
    np.ndarray
        Boolean circle mask with the same shape as ``xx`` and ``yy``.
    """
    return (
        np.hypot(
            xx - float(center[0]),
            yy - float(center[1]),
        )
        <= float(radius)
    )


def _ellipse_mask(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    major_axis: np.ndarray,
    major_radius: float,
    minor_radius: float,
) -> np.ndarray:
    """
    Rasterize a filled ellipse from precomputed coordinate grids.
    """
    major = np.asarray(major_axis, dtype=np.float32).reshape(-1)
    major_length = float(np.linalg.norm(major[:2]))
    if major.size < 2 or major_length < 1e-6:
        return _circle_mask(
            xx=xx,
            yy=yy,
            center=center,
            radius=major_radius,
        )

    major_unit = major[:2] / major_length
    minor_unit = np.asarray(
        [
            -float(major_unit[1]),
            float(major_unit[0]),
        ],
        dtype=np.float32,
    )
    rel_x = xx - float(center[0])
    rel_y = yy - float(center[1])
    major_projection = (
        rel_x * float(major_unit[0])
        + rel_y * float(major_unit[1])
    )
    minor_projection = (
        rel_x * float(minor_unit[0])
        + rel_y * float(minor_unit[1])
    )

    safe_major_radius = max(float(major_radius), 1e-6)
    safe_minor_radius = max(float(minor_radius), 1e-6)
    return (
        (major_projection / safe_major_radius) ** 2
        + (minor_projection / safe_minor_radius) ** 2
    ) <= 1.0


def _circle_outline_mask(
    *,
    image_shape: tuple[int, ...],
    center: np.ndarray,
    radius: float,
    thickness: int = 1,
) -> np.ndarray:
    """
    Rasterize a circular outline.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        mask,
        tuple(int(value) for value in np.rint(center)),
        max(1, int(round(float(radius)))),
        255,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_8,
    )
    return mask > 0


def _ellipse_outline_mask(
    *,
    image_shape: tuple[int, ...],
    center: np.ndarray,
    major_axis: np.ndarray,
    major_radius: float,
    minor_radius: float,
    thickness: int = 1,
) -> np.ndarray:
    """
    Rasterize an elliptical outline.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    major = np.asarray(major_axis, dtype=np.float32).reshape(-1)
    major_length = float(np.linalg.norm(major[:2]))
    if major.size < 2 or major_length < 1e-6:
        return _circle_outline_mask(
            image_shape=image_shape,
            center=center,
            radius=major_radius,
            thickness=thickness,
        )

    angle_degrees = math.degrees(
        math.atan2(
            float(major[1]),
            float(major[0]),
        )
    )
    cv2.ellipse(
        mask,
        tuple(int(value) for value in np.rint(center)),
        (
            max(1, int(round(float(major_radius)))),
            max(1, int(round(float(minor_radius)))),
        ),
        angle_degrees,
        0.0,
        360.0,
        255,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_8,
    )
    return mask > 0


def _limb_support_bboxes(
    *,
    semantic_support_mask: np.ndarray,
    image_shape: tuple[int, ...],
    crop_expansion: float,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]:
    """
    Derive tight and expanded bboxes for a limb semantic support mask.

    Parameters
    ----------
    semantic_support_mask : np.ndarray
        Boolean full-frame support mask.
    image_shape : tuple[int, ...]
        Source image shape.
    crop_expansion : float
        Proportional expansion applied to the tight support bbox.

    Returns
    -------
    tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
        ``(base_bbox, crop_bbox)`` as end-exclusive boxes.

    Raises
    ------
    RuntimeError
        If the support mask is empty or bbox expansion becomes invalid.
    """
    h, w = image_shape[:2]
    base_bbox = tight_mask_bbox(
        semantic_support_mask.astype(np.uint8)
    )
    crop_bbox = expand_clip_bbox(
        *base_bbox,
        w,
        h,
        float(crop_expansion),
    )
    return base_bbox, crop_bbox


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the 2D scalar cross product.

    Parameters
    ----------
    a : np.ndarray
        First vector with at least two coordinates.
    b : np.ndarray
        Second vector with at least two coordinates.

    Returns
    -------
    float
        Scalar value ``a.x * b.y - a.y * b.x``.
    """
    return float(a[0] * b[1] - a[1] * b[0])
