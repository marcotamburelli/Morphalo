from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


def _estimate_landmark_similarity(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> np.ndarray:
    """
    Estimate a 3D similarity transform between corresponding landmarks.

    The transform contains translation, uniform scale, and a proper 3D
    rotation. It deliberately excludes reflection and non-rigid deformation so
    that the relative shape and expression of the source landmarks remain
    unchanged.

    Parameters
    ----------
    source_xyz : numpy.ndarray
        Source landmarks with shape ``(N, 3)``.
    target_xyz : numpy.ndarray
        Target landmarks with shape ``(N, 3)`` and matching topology.

    Returns
    -------
    numpy.ndarray
        Homogeneous matrix with shape ``(4, 4)`` mapping source to target.

    Raises
    ------
    ValueError
        If the landmark arrays have incompatible shapes.
    RuntimeError
        If the source geometry is degenerate or the transform is not finite.
    """
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            'Source and target landmarks must have matching (N, 3) shapes'
        )

    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    source_variance = np.mean(np.sum(source_centered ** 2, axis=1))

    if source_variance <= np.finfo(np.float64).eps:
        raise RuntimeError('Could not estimate transform from degenerate landmarks')

    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    orientation = np.ones(3, dtype=np.float64)

    if np.linalg.det(left @ right_t) < 0:
        orientation[-1] = -1.0

    rotation = left @ np.diag(orientation) @ right_t
    scale = float(np.dot(singular_values, orientation) / source_variance)
    translation = target_center - scale * (rotation @ source_center)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation

    if not np.all(np.isfinite(matrix)):
        raise RuntimeError('Could not estimate a finite landmark transform')

    return matrix.astype(np.float32)


def _transform_landmarks(
    landmarks_xyz: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply a homogeneous 3D transform to a landmark set.

    Parameters
    ----------
    landmarks_xyz : numpy.ndarray
        Pixel-equivalent landmarks with shape ``(N, 3)``.
    matrix : numpy.ndarray
        Homogeneous transform with shape ``(4, 4)``.

    Returns
    -------
    numpy.ndarray
        Transformed floating-point landmarks with shape ``(N, 3)``.
    """
    landmarks = np.asarray(landmarks_xyz, dtype=np.float32)
    transform = np.asarray(matrix, dtype=np.float32)

    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError('Landmarks must have shape (N, 3)')

    if transform.shape != (4, 4):
        raise ValueError('Transform must have shape (4, 4)')

    homogeneous = np.column_stack(
        (landmarks, np.ones(landmarks.shape[0], dtype=np.float32))
    )
    transformed = homogeneous @ transform.T

    return transformed[:, :3]


def _yaw_degrees_from_transform(matrix: np.ndarray) -> float:
    """
    Extract signed yaw from a MediaPipe facial transformation matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        MediaPipe facial transformation matrix with shape ``(4, 4)``.

    Returns
    -------
    float
        Signed yaw angle in degrees.

    Raises
    ------
    ValueError
        If the matrix does not have shape ``(4, 4)``.
    RuntimeError
        If a finite proper rotation cannot be recovered.
    """
    transform = np.asarray(matrix, dtype=np.float64)

    if transform.shape != (4, 4):
        raise ValueError('Face transform must have shape (4, 4)')

    left, _, right_t = np.linalg.svd(transform[:3, :3])
    rotation = left @ right_t

    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t

    yaw = np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2]))

    if not np.isfinite(yaw):
        raise RuntimeError('Could not derive yaw from face transform')

    return float(yaw)


def _depth_yaw_weight(yaw_deg: float) -> float:
    """Return the agreed depth confidence for an absolute source yaw."""
    yaw = abs(float(yaw_deg))

    if yaw >= 80.0:
        return 0.0

    if yaw <= 60.0:
        return max(0.1, yaw / 60.0)

    return max(0.1, (80.0 - yaw) / 20.0)


def _fuse_source_landmarks(
    source_landmarks: list[np.ndarray],
    source_yaws: list[float],
) -> tuple[np.ndarray, list[np.ndarray], int, list[float]]:
    """
    Normalize source faces to the most frontal view and fuse their depth.

    Parameters
    ----------
    source_landmarks : list[numpy.ndarray]
        Source XYZ landmark sets with matching ``(N, 3)`` shapes.
    source_yaws : list[float]
        Original signed yaw angle for each source image.

    Returns
    -------
    tuple
        Fused XYZ landmarks, normalized source sets, base index, and depth
        weights in source order.

    Raises
    ------
    ValueError
        If no source is provided or list lengths and landmark shapes differ.

    Notes
    -----
    X and Y deliberately come from the source with minimum absolute yaw. A
    future optimization may average sufficiently frontal X/Y observations and
    weight them according to pitch.
    """
    if not source_landmarks:
        raise ValueError('At least one source landmark set is required')

    if len(source_landmarks) != len(source_yaws):
        raise ValueError('Source landmarks and yaw lists must have equal length')

    shapes = {np.asarray(points).shape for points in source_landmarks}

    if len(shapes) != 1:
        raise ValueError('All source landmark sets must have matching shapes')

    base_index = int(np.argmin(np.abs(np.asarray(source_yaws))))
    base_xyz = np.asarray(source_landmarks[base_index], dtype=np.float32)
    normalized = []

    for index, landmarks in enumerate(source_landmarks):
        points = np.asarray(landmarks, dtype=np.float32)

        if index == base_index:
            normalized.append(points.copy())
            continue

        to_base = _estimate_landmark_similarity(points, base_xyz)
        normalized.append(_transform_landmarks(points, to_base))

    weights = [_depth_yaw_weight(yaw) for yaw in source_yaws]
    fused = base_xyz.copy()

    # TODO: Fuse X/Y across low-yaw views, weighting their reliability by pitch.
    if len(normalized) > 1 and sum(weights) > 0.0:
        depth_values = np.stack([points[:, 2] for points in normalized])
        fused[:, 2] = np.average(depth_values, axis=0, weights=weights)

    return fused, normalized, base_index, weights


def _draw_landmarks(
    image_rgb: np.ndarray,
    landmarks_xy: np.ndarray,
    *,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """
    Draw visible landmark points on an RGB image copy.

    Parameters
    ----------
    image_rgb : numpy.ndarray
        Background image with shape ``(H, W, 3)``.
    landmarks_xy : numpy.ndarray
        Pixel-space landmarks with shape ``(N, 2)``.
    color : tuple[int, int, int]
        RGB point color.

    Returns
    -------
    numpy.ndarray
        RGB image containing the landmark overlay.
    """
    import cv2

    output = image_rgb.copy()
    height, width = output.shape[:2]
    radius = max(1, int(round(min(height, width) / 512.0)))

    for x, y in np.asarray(landmarks_xy, dtype=np.float32):
        px = int(round(float(x)))
        py = int(round(float(y)))

        if 0 <= px < width and 0 <= py < height:
            cv2.circle(output, (px, py), radius, color, -1, cv2.LINE_AA)

    return output


def draw_geometry_alignment(
    image_rgb: np.ndarray,
    target_xy: np.ndarray,
    synthetic_xy: np.ndarray,
    target_mask: np.ndarray,
    synthetic_mask: np.ndarray,
) -> np.ndarray:
    """Overlay target and transformed synthetic face geometry for debugging."""
    import cv2

    target = np.asarray(target_mask, dtype=bool)
    synthetic = np.asarray(synthetic_mask, dtype=bool)

    if target.shape != image_rgb.shape[:2] or synthetic.shape != target.shape:
        raise ValueError('Geometry masks must match the debug image')

    output = _draw_landmarks(
        image_rgb,
        target_xy,
        color=(0, 255, 255),
    )
    output = _draw_landmarks(
        output,
        synthetic_xy,
        color=(255, 0, 255),
    )
    target_contours, _ = cv2.findContours(
        target.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    synthetic_contours, _ = cv2.findContours(
        synthetic.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    thickness = max(1, int(round(min(output.shape[:2]) / 512.0)))
    cv2.drawContours(
        output,
        target_contours,
        -1,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.drawContours(
        output,
        synthetic_contours,
        -1,
        (255, 64, 64),
        thickness,
        cv2.LINE_AA,
    )

    return output


def _mediapipe_face_triangles() -> list[tuple[int, int, int]]:
    """Return triangle indices encoded by MediaPipe's tessellation edges."""
    import mediapipe as mp

    connections = (
        mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    )
    triangles = []

    # MediaPipe stores every triangle as three consecutive closed edges. Use
    # the unique vertices rather than depending on a particular edge winding.
    for offset in range(0, len(connections), 3):
        edges = connections[offset:offset + 3]
        vertices = []

        for edge in edges:
            for vertex in (int(edge.start), int(edge.end)):
                if vertex not in vertices:
                    vertices.append(vertex)

        if len(vertices) == 3:
            triangles.append(tuple(vertices))

    return triangles


def _triangle_rasters(
    image_shape: tuple[int, ...],
    points: np.ndarray,
    triangles: list[tuple[int, int, int]],
) -> Iterator[tuple[np.ndarray, tuple[slice, slice], np.ndarray]]:
    """Yield valid mesh triangles and their pixel-space barycentric weights."""
    height, width = image_shape[:2]

    for triangle in triangles:
        indices = np.asarray(triangle, dtype=np.int32)

        if np.any(indices < 0) or np.any(indices >= points.shape[0]):
            continue

        vertices = points[indices]
        x1, y1 = vertices[0, :2]
        x2, y2 = vertices[1, :2]
        x3, y3 = vertices[2, :2]
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        if abs(float(denominator)) < 1e-6:
            continue

        min_x = max(0, int(np.floor(np.min(vertices[:, 0]))))
        max_x = min(width - 1, int(np.ceil(np.max(vertices[:, 0]))))
        min_y = max(0, int(np.floor(np.min(vertices[:, 1]))))
        max_y = min(height - 1, int(np.ceil(np.max(vertices[:, 1]))))

        if min_x > max_x or min_y > max_y:
            continue

        grid_y, grid_x = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        weight1 = (
            (y2 - y3) * (grid_x - x3)
            + (x3 - x2) * (grid_y - y3)
        ) / denominator
        weight2 = (
            (y3 - y1) * (grid_x - x3)
            + (x1 - x3) * (grid_y - y3)
        ) / denominator
        weight3 = 1.0 - weight1 - weight2
        weights = np.stack((weight1, weight2, weight3))
        inside = np.all(weights >= -1e-5, axis=0)

        if not np.any(inside):
            continue

        region = (
            slice(min_y, max_y + 1),
            slice(min_x, max_x + 1),
        )

        yield indices, region, weights


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill mesh openings that do not connect to the image boundary."""
    import cv2

    binary = np.asarray(mask, dtype=np.uint8)
    padded = cv2.copyMakeBorder(
        binary,
        1,
        1,
        1,
        1,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    exterior = (1 - padded).copy()
    flood_mask = np.zeros(
        (exterior.shape[0] + 2, exterior.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(exterior, flood_mask, (0, 0), 0)
    holes = exterior[1:-1, 1:-1] > 0

    return np.asarray(mask, dtype=bool) | holes


def face_mesh_mask(
    image_shape: tuple[int, ...],
    landmarks_xy: np.ndarray,
    *,
    triangles: Optional[list[tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Rasterize the complete projected face mesh as a boolean mask."""
    points = np.asarray(landmarks_xy, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError('Face landmarks must have shape (N, 2) or (N, 3)')

    mesh_triangles = (
        _mediapipe_face_triangles() if triangles is None else triangles
    )
    coverage = np.zeros(image_shape[:2], dtype=bool)

    for _, region, weights in _triangle_rasters(
        image_shape,
        points,
        mesh_triangles,
    ):
        coverage[region] |= np.all(weights >= -1e-5, axis=0)

    # Eyes and lips are intentional openings in the MediaPipe tessellation,
    # but face replacement needs one continuous region enclosing the complete
    # projected facial surface.
    return _fill_mask_holes(coverage)


def draw_luminance_map(
    image_shape: tuple[int, ...],
    landmarks_xyz: np.ndarray,
    vertex_luminance: np.ndarray,
    *,
    triangles: Optional[list[tuple[int, int, int]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize vertex luminance over a projected MediaPipe face mesh."""
    import cv2

    height, width = image_shape[:2]
    points = np.asarray(landmarks_xyz, dtype=np.float32)
    luminance = np.asarray(vertex_luminance, dtype=np.float32).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Luminance landmarks must have shape (N, 3)')

    if luminance.shape[0] != points.shape[0]:
        raise ValueError('Vertex luminance must match the landmark count')

    mesh_triangles = (
        _mediapipe_face_triangles() if triangles is None else triangles
    )
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    output = np.zeros((height, width), dtype=np.float32)

    for indices, region, weights in _triangle_rasters(
        image_shape,
        points,
        mesh_triangles,
    ):
        vertices = points[indices]
        weight1, weight2, weight3 = weights
        inside = np.all(weights >= -1e-5, axis=0)

        interpolated_z = (
            weight1 * vertices[0, 2]
            + weight2 * vertices[1, 2]
            + weight3 * vertices[2, 2]
        )
        interpolated_luminance = (
            weight1 * luminance[indices[0]]
            + weight2 * luminance[indices[1]]
            + weight3 * luminance[indices[2]]
        )
        z_region = z_buffer[region]
        output_region = output[region]
        visible = inside & (interpolated_z < z_region)
        z_region[visible] = interpolated_z[visible]
        output_region[visible] = interpolated_luminance[visible]

    face_mask = face_mesh_mask(
        image_shape,
        points[:, :2],
        triangles=mesh_triangles,
    )
    missing = face_mask & ~np.isfinite(z_buffer)

    # MediaPipe leaves openings around eyes and lips. Their luminance should be
    # a continuous low-frequency surface, not holes exposed to the background.
    if np.any(missing):
        output = cv2.inpaint(
            output,
            missing.astype(np.uint8) * 255,
            inpaintRadius=3.0,
            flags=cv2.INPAINT_TELEA,
        )

    return np.clip(output, 0.0, 255.0).astype(np.uint8), face_mask
