from __future__ import annotations

import cv2
import numpy as np


def compare_face_masks(
    target_mask: np.ndarray,
    synthetic_mask: np.ndarray,
) -> dict[str, dict[str, float | int | list[float | int]] | float]:
    """Compare the pixel extents of target and synthetic face masks."""
    target = np.asarray(target_mask, dtype=bool)
    synthetic = np.asarray(synthetic_mask, dtype=bool)

    if target.ndim != 2 or synthetic.shape != target.shape:
        raise ValueError('Face masks must have matching (H, W) shapes')

    def geometry(mask: np.ndarray) -> dict[str, float | int | list[float | int]]:
        y, x = np.nonzero(mask)

        if x.size == 0:
            raise ValueError('Face mask must not be empty')

        min_x = int(np.min(x))
        max_x = int(np.max(x))
        min_y = int(np.min(y))
        max_y = int(np.max(y))

        return {
            'bbox': [min_x, min_y, max_x, max_y],
            'width': max_x - min_x + 1,
            'height': max_y - min_y + 1,
            'area': int(np.count_nonzero(mask)),
            'center': [
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
            ],
        }

    target_geometry = geometry(target)
    synthetic_geometry = geometry(synthetic)
    target_center = target_geometry['center']
    synthetic_center = synthetic_geometry['center']

    return {
        'target': target_geometry,
        'synthetic': synthetic_geometry,
        'target_to_synthetic_width_ratio': (
            target_geometry['width'] / synthetic_geometry['width']
        ),
        'target_to_synthetic_height_ratio': (
            target_geometry['height'] / synthetic_geometry['height']
        ),
        'target_to_synthetic_area_ratio': (
            target_geometry['area'] / synthetic_geometry['area']
        ),
        'synthetic_center_offset': [
            synthetic_center[0] - target_center[0],
            synthetic_center[1] - target_center[1],
        ],
    }


def sample_landmark_luminance(
    image_gray: np.ndarray,
    landmarks_xy: np.ndarray,
    *,
    blur_sigma: float,
) -> np.ndarray:
    """Sample robust low-frequency luminance at landmark coordinates."""
    gray = np.asarray(image_gray, dtype=np.uint8)
    points = np.asarray(landmarks_xy, dtype=np.float32)

    if gray.ndim != 2:
        raise ValueError('Luminance image must have shape (H, W)')

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Luminance landmarks must have shape (N, 2)')

    smoothed = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(0.5, float(blur_sigma)),
        sigmaY=max(0.5, float(blur_sigma)),
        borderType=cv2.BORDER_REPLICATE,
    )
    coordinates = np.vstack((points[:, 0], points[:, 1])).astype(np.float32)
    sampled = cv2.remap(
        smoothed,
        coordinates[0][None, :],
        coordinates[1][None, :],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return sampled[0].astype(np.float32)


def sharpen_face_luminance(
    face_gray: np.ndarray,
    face_mask: np.ndarray,
    *,
    detail_gain: float,
    blur_sigma: float = 1.5,
) -> np.ndarray:
    """Apply a mask-aware unsharp filter to a synthetic luminance face."""
    face = np.asarray(face_gray, dtype=np.float32)
    mask = np.asarray(face_mask, dtype=bool)

    if face.ndim != 2 or mask.shape != face.shape:
        raise ValueError('Synthetic face and mask must have matching shapes')

    if detail_gain < 0.0:
        raise ValueError('Detail gain must be non-negative')

    if detail_gain == 0.0 or not np.any(mask):
        return np.asarray(face_gray, dtype=np.uint8).copy()

    weights = cv2.GaussianBlur(
        mask.astype(np.float32),
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    weighted_face = cv2.GaussianBlur(
        face * mask,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    local_blur = weighted_face / np.maximum(weights, 1e-6)
    sharpened = face + detail_gain * (face - local_blur)
    result = face.copy()
    result[mask] = np.clip(sharpened[mask], 0.0, 255.0)

    return np.rint(result).astype(np.uint8)


def _largest_mask_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest external contour as floating-point XY coordinates."""
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError('Face mask does not contain a contour')

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    return contour.astype(np.float32)


def _prepare_closed_contour(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed contour and its cumulative arc lengths."""
    points = np.asarray(contour, dtype=np.float32)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))

    if cumulative[-1] <= 0.0:
        raise ValueError('Face mask contour is degenerate')

    return closed, cumulative


def _sample_closed_contour(
    closed: np.ndarray,
    cumulative: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Sample a closed contour at normalized arc-length positions."""
    distance = np.mod(positions, 1.0) * cumulative[-1]
    segment = np.searchsorted(cumulative, distance, side='right') - 1
    segment = np.clip(segment, 0, len(cumulative) - 2)
    segment_length = cumulative[segment + 1] - cumulative[segment]
    fraction = (distance - cumulative[segment]) / np.maximum(
        segment_length,
        1e-6,
    )

    return (
        closed[segment] * (1.0 - fraction[:, None])
        + closed[segment + 1] * fraction[:, None]
    )


def _align_contour_phase(
    outer_closed: np.ndarray,
    outer_cumulative: np.ndarray,
    inner_closed: np.ndarray,
    inner_cumulative: np.ndarray,
    sample_count: int,
) -> float:
    """Find the cyclic inner-contour offset nearest to the outer contour."""
    positions = np.arange(sample_count, dtype=np.float32) / sample_count
    outer = _sample_closed_contour(
        outer_closed,
        outer_cumulative,
        positions,
    )
    inner = _sample_closed_contour(
        inner_closed,
        inner_cumulative,
        positions,
    )
    errors = np.empty(sample_count, dtype=np.float64)

    for shift in range(sample_count):
        shifted = np.roll(inner, shift, axis=0)
        errors[shift] = np.mean(np.sum((outer - shifted) ** 2, axis=1))

    return -float(np.argmin(errors)) / float(sample_count)


def warp_face_luminance_background(
    image_gray: np.ndarray,
    erase_mask: np.ndarray,
    insert_mask: np.ndarray,
    *,
    face_span: float,
    feather_radius: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Close the face corona with a smooth local coordinate deformation."""
    from scipy.spatial import cKDTree

    gray = np.asarray(image_gray, dtype=np.uint8)
    erase = np.asarray(erase_mask, dtype=bool)
    insert = np.asarray(insert_mask, dtype=bool)

    if gray.ndim != 2 or erase.shape != gray.shape or insert.shape != gray.shape:
        raise ValueError('Luminance image and masks must have matching shapes')

    if not np.isfinite(face_span) or face_span <= 0.0:
        raise ValueError('Face span must be a positive finite number')

    if feather_radius < 0:
        raise ValueError('Feather radius must be non-negative')

    if feather_radius:
        size = feather_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        inner = cv2.erode(insert.astype(np.uint8), kernel) > 0
        if not np.any(inner):
            inner = insert.copy()
    else:
        inner = insert.copy()

    if not np.any(erase & ~inner):
        return gray.copy(), np.zeros_like(gray, dtype=np.float32), {}

    outer_contour = _largest_mask_contour(erase)
    inner_contour = _largest_mask_contour(inner)

    # Keep both contours in the same winding direction before establishing
    # their cyclic correspondence.
    outer_area = cv2.contourArea(outer_contour, oriented=True)
    inner_area = cv2.contourArea(inner_contour, oriented=True)
    if np.sign(outer_area) != np.sign(inner_area):
        inner_contour = inner_contour[::-1].copy()

    outer_closed, outer_cumulative = _prepare_closed_contour(outer_contour)
    inner_closed, inner_cumulative = _prepare_closed_contour(inner_contour)
    control_spacing = max(2.0, face_span * 0.015)
    control_count = max(
        32,
        int(np.ceil(outer_cumulative[-1] / control_spacing)),
    )
    inner_phase = _align_contour_phase(
        outer_closed,
        outer_cumulative,
        inner_closed,
        inner_cumulative,
        control_count,
    )
    positions = np.arange(control_count, dtype=np.float32) / control_count
    outer_points = _sample_closed_contour(
        outer_closed,
        outer_cumulative,
        positions,
    )
    inner_points = _sample_closed_contour(
        inner_closed,
        inner_cumulative,
        positions + inner_phase,
    )
    control_displacement = outer_points - inner_points
    influence_radius = max(4, int(round(face_span * 0.25)))
    smoothing_sigma = max(0.5, face_span * 0.008)
    height, width = gray.shape
    min_x = max(0, int(np.floor(np.min(inner_points[:, 0]))) - influence_radius)
    max_x = min(
        width,
        int(np.ceil(np.max(inner_points[:, 0]))) + influence_radius + 1,
    )
    min_y = max(0, int(np.floor(np.min(inner_points[:, 1]))) - influence_radius)
    max_y = min(
        height,
        int(np.ceil(np.max(inner_points[:, 1]))) + influence_radius + 1,
    )
    grid_y, grid_x = np.mgrid[min_y:max_y, min_x:max_x]
    coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distance, nearest = cKDTree(inner_points).query(coordinates, workers=-1)
    attenuation = np.clip(1.0 - distance / influence_radius, 0.0, 1.0)
    attenuation = attenuation * attenuation * (3.0 - 2.0 * attenuation)
    raw_displacement = control_displacement[nearest] * attenuation[:, None]
    local_shape = grid_x.shape
    raw_x = raw_displacement[:, 0].reshape(local_shape).astype(np.float32)
    raw_y = raw_displacement[:, 1].reshape(local_shape).astype(np.float32)
    smooth_x = cv2.GaussianBlur(
        raw_x,
        (0, 0),
        sigmaX=smoothing_sigma,
        sigmaY=smoothing_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    smooth_y = cv2.GaussianBlur(
        raw_y,
        (0, 0),
        sigmaX=smoothing_sigma,
        sigmaY=smoothing_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    local_attenuation = attenuation.reshape(local_shape).astype(np.float32)
    displacement_x = smooth_x
    displacement_y = smooth_y
    map_x = grid_x.astype(np.float32) + displacement_x
    map_y = grid_y.astype(np.float32) + displacement_y
    warped_local = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    result = gray.copy()
    active = local_attenuation > 0.0
    result_region = result[min_y:max_y, min_x:max_x]
    result_region[active] = warped_local[active]
    displacement = np.zeros_like(gray, dtype=np.float32)
    displacement_region = displacement[min_y:max_y, min_x:max_x]
    displacement_region[active] = np.hypot(
        displacement_x[active],
        displacement_y[active],
    )

    derivative_x_x = np.gradient(displacement_x, axis=1)
    derivative_x_y = np.gradient(displacement_x, axis=0)
    derivative_y_x = np.gradient(displacement_y, axis=1)
    derivative_y_y = np.gradient(displacement_y, axis=0)
    jacobian = (
        (1.0 + derivative_x_x) * (1.0 + derivative_y_y)
        - derivative_x_y * derivative_y_x
    )
    active_jacobian = jacobian[active]
    min_jacobian = (
        float(np.min(active_jacobian)) if active_jacobian.size else 1.0
    )

    return (
        result,
        displacement,
        {
            'control_spacing': float(control_spacing),
            'control_count': control_count,
            'influence_radius': influence_radius,
            'smoothing_sigma': float(smoothing_sigma),
            'max_displacement': float(np.max(displacement)),
            'min_jacobian': min_jacobian,
        },
    )


def composite_face_luminance(
    background_gray: np.ndarray,
    face_gray: np.ndarray,
    face_mask: np.ndarray,
    *,
    feather_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a synthetic luminance face into a prepared grayscale image."""
    background = np.asarray(background_gray, dtype=np.float32)
    face = np.asarray(face_gray, dtype=np.float32)
    mask = np.asarray(face_mask, dtype=bool)

    if background.ndim != 2 or face.shape != background.shape:
        raise ValueError('Luminance surfaces must have matching (H, W) shapes')

    if mask.shape != background.shape:
        raise ValueError('Face mask must match the luminance surfaces')

    if feather_radius <= 0:
        alpha = mask.astype(np.float32)
    else:
        distance = cv2.distanceTransform(
            mask.astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        alpha = np.clip(distance / float(feather_radius), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    composed = background * (1.0 - alpha) + face * alpha

    return np.clip(composed, 0.0, 255.0).astype(np.uint8), alpha


def gray_to_rgb(image_gray: np.ndarray) -> np.ndarray:
    """Repeat a grayscale image over three RGB channels."""
    gray = np.asarray(image_gray, dtype=np.uint8)

    if gray.ndim != 2:
        raise ValueError('Grayscale image must have shape (H, W)')

    return np.repeat(gray[:, :, None], 3, axis=2)
