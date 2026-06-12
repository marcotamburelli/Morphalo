from typing import Tuple

import numpy as np


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
