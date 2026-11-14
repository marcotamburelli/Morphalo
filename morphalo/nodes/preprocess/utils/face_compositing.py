from __future__ import annotations

import cv2
import numpy as np


# Width of the radial transition beyond the larger of the old/new silhouettes,
# as a fraction of face_span. Its outer endpoint stays fixed. Larger values
# spread the deformation across more surrounding detail; smaller values make
# the transition more concentrated. The pixel floor handles small faces.
WARP_INFLUENCE_FRACTION = 0.25
MIN_WARP_INFLUENCE_RADIUS_PX = 4

# Sample angles densely enough for roughly one-pixel spacing at the furthest
# contour vertex. Radii are interpolated periodically between these rays.
RADIAL_SAMPLE_SPACING_PX = 1.0
MIN_RADIAL_SAMPLE_COUNT = 64

# Taper warp strength toward blocked angular directions over this arc length
# at the furthest face contour. The blocked rays and their adjacent angular
# intervals stay exactly at identity; tapering only affects nearby valid rays.
OCCLUSION_TRANSITION_ARC_PX = 4.0

# Fixed inward offset of the new silhouette, independent of face size and
# blending. Increase only enough to hide residual original-face edge pixels.
INSERT_EROSION_RADIUS_PX = 2

# Blend width as a fraction of the combined face span. This changes the alpha
# transition only; it must not alter the radial boundary correspondence.
FEATHER_RADIUS_FRACTION = 0.025
MIN_FEATHER_RADIUS_PX = 2


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


def _radial_mask_contour(mask: np.ndarray) -> np.ndarray:
    """Extract one nondegenerate external silhouette, ignoring interior holes."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(contours) != 1 or cv2.contourArea(contours[0]) <= 0:
        raise ValueError(
            'Radial face warp requires one nondegenerate silhouette per mask')
    return contours[0].reshape(-1, 2).astype(np.float64)


def _radial_contour_radii(
    contour: np.ndarray,
    center: np.ndarray,
    angles: np.ndarray,
    *,
    outermost: bool,
) -> np.ndarray:
    """
    Intersect rays from a shared center with the contour's line segments.

    Select the last positive intersection for an original silhouette, so its
    radial envelope contains the entire old face. Select the first intersection
    for an eroded destination silhouette, so the radial interior stays inside
    the new face even when the ray exits and reenters it. Multiple crossings,
    including pixel stair-steps and deeper concavities, are valid in both cases.
    Every ray must have at least one positive intersection. Process rays in
    batches to bound temporary ray-by-segment arrays.
    """
    starts = contour - center
    ends = np.roll(starts, -1, axis=0)
    segments = ends - starts
    numerators = starts[:, 0] * segments[:, 1] - starts[:, 1] * segments[:, 0]
    radii = np.empty(len(angles), dtype=np.float64)

    for offset in range(0, len(angles), 128):
        batch = angles[offset:offset + 128]
        dx = np.cos(batch)[:, None]
        dy = np.sin(batch)[:, None]
        start_side = dx * starts[:, 1] - dy * starts[:, 0]
        denominator = dx * segments[:, 1] - dy * segments[:, 0]
        distances = np.divide(
            numerators,
            denominator,
            out=np.full(denominator.shape, np.inf),
            where=np.abs(denominator) > 1e-12,
        )
        # Include segment endpoints and tangent vertex hits, not only strict
        # side crossings. A tangent can be the furthest original silhouette
        # point even though the ray does not leave/reenter there. Duplicate
        # vertex hits are harmless because selection uses min/max distances.
        segment_fraction = np.divide(
            -start_side,
            denominator,
            out=np.full(denominator.shape, np.inf),
            where=np.abs(denominator) > 1e-12,
        )
        hits = (
            (segment_fraction >= -1e-9)
            & (segment_fraction <= 1 + 1e-9)
            & (distances > 0)
            & np.isfinite(distances)
        )
        # For a boundary segment collinear with the ray, choose the relevant
        # positive endpoint; the entire interval belongs to the boundary.
        collinear = (np.abs(denominator) <=
                     1e-12) & (np.abs(start_side) <= 1e-9)
        if np.any(collinear):
            start_projection = dx * starts[:, 0] + dy * starts[:, 1]
            end_projection = dx * ends[:, 0] + dy * ends[:, 1]
            endpoints = np.stack((start_projection, end_projection))
            if outermost:
                projected = np.max(endpoints, axis=0)
            else:
                projected = np.min(
                    np.where(endpoints > 0, endpoints, np.inf), axis=0)
            valid_collinear = collinear & (
                projected > 0) & np.isfinite(projected)
            distances = np.where(valid_collinear, projected, distances)
            hits |= valid_collinear
        if np.any(~np.any(hits, axis=1)):
            raise ValueError(
                'Radial face warp found a ray without a positive boundary intersection')

        if outermost:
            selected = np.max(np.where(hits, distances, -np.inf), axis=1)
        else:
            selected = np.min(np.where(hits, distances, np.inf), axis=1)
        radii[offset:offset + len(batch)] = selected

    return radii


def _occlusion_ray_weights(
    occlusion: np.ndarray,
    center: np.ndarray,
    angles: np.ndarray,
    original_radii: np.ndarray,
    inserted_radii: np.ndarray,
    max_radius: float,
) -> tuple[np.ndarray, int]:
    """
    Block occluded contour pairs and taper only their valid angular neighbors.

    Parameters
    ----------
    occlusion : numpy.ndarray
        Boolean target-space protection mask, with shape ``(H, W)``.
    center : numpy.ndarray
        Shared original/new silhouette center in image XY coordinates.
    angles : numpy.ndarray
        Sorted unique ray angles in [0, 2pi), sampled from the full silhouettes.
    original_radii, inserted_radii : numpy.ndarray
        Selected source and eroded-destination radii for each ray. Endpoints
        are checked against occlusion using nearest-pixel coordinates.
    max_radius : float
        Furthest contour distance, converting the internal pixel arc width to
        an angle for the transition near blocked directions.

    Returns
    -------
    numpy.ndarray
        Per-ray strength in [0, 1]. Blocked rays and adjacent sample intervals
        stay at zero; valid neighbors gradually return to full warp strength.
    int
        Number of contour pairs directly blocked by occluded endpoints, before
        expanding the zero-weight intervals for interpolation.
    """
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    blocked = np.zeros(len(angles), dtype=bool)

    for radii in (original_radii, inserted_radii):
        points = np.rint(center + directions * radii[:, None]).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, occlusion.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, occlusion.shape[0] - 1)
        blocked |= occlusion[points[:, 1], points[:, 0]]

    count = int(blocked.sum())

    if not count:
        return np.ones(len(angles)), count

    # Zero both ends of any interval touching a blocked ray. Otherwise angular
    # interpolation between a blocked sample and a valid sample would revive
    # displacement inside that excluded interval. Circular rolls handle 0/2pi.
    zero = blocked | np.roll(blocked, 1) | np.roll(blocked, -1)
    anchors = angles[zero]
    wrapped = np.concatenate(
        ([anchors[-1] - 2 * np.pi], anchors, [anchors[0] + 2 * np.pi])
    )
    indices = np.searchsorted(wrapped, angles, side='right')
    distance = np.minimum(
        angles - wrapped[indices - 1],
        wrapped[indices] - angles
    )
    transition_angle = OCCLUSION_TRANSITION_ARC_PX / max_radius
    weight = np.clip(distance / transition_angle, 0., 1.)

    return weight * weight * (3. - 2. * weight), count


def warp_face_background(
    image: np.ndarray,
    original_mask: np.ndarray,
    insert_mask: np.ndarray,
    *,
    face_span: float,
    erosion_radius: int = INSERT_EROSION_RADIUS_PX,
    occlusion_mask: np.ndarray | None = None,
    collect_diagnostics: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int]]:
    """
    Fit an original face silhouette inside a new one using a local radial warp.

    Use the pixel centroid of ``original_mask | insert_mask`` as a shared center.
    Erode the insertion mask by ``erosion_radius`` to put the old face boundary
    beneath the replacement and its feathered edge. The center must lie inside
    both masks and the eroded insertion silhouette; invalid placement fails
    rather than selecting an alternative center. On each ray, use the furthest
    original-boundary intersection and the nearest eroded-destination boundary
    intersection. Multiple crossings are expected for concavities and raster
    stair-steps; they do not require an intersection-spread tolerance.

    On each ray, A is the furthest original radius and B is the first eroded
    insertion radius. The source radial envelope contains the whole original
    silhouette, while the destination radial interior stays inside the new
    silhouette. This conservative correspondence can deform deep concavities
    more strongly; it does not reproduce every indentation exactly. A third
    radius C equals max(A, B) plus the transition width derived from
    ``face_span``. In the inverse map used by OpenCV, B
    samples A, C samples itself, and the interval [B, C] samples [A, C]
    linearly. Thus the original silhouette moves under the new face while the
    surrounding image reconnects continuously to an unchanged outer boundary.
    The interior [0, B] samples [0, A] to keep the map continuous at the center.

    Rays include uniform angular samples and all silhouette vertex directions,
    preserving narrow raster tips. Radii are interpolated periodically between
    sampled angles. Radial mappings are strictly increasing: neither contraction nor expansion reverses their
    order. No later Gaussian smoothing changes the boundary correspondence.
    With an occlusion mask, sample both contours before checking occlusion.
    A protected endpoint blocks the entire ray. Blocked directions retain the
    identity map; nearby valid directions taper toward them. The center and
    silhouette geometry are never recomputed from occlusion-cut masks. Protected
    pixels are kept unchanged here; callers must also restore them after pasting
    the face. Inverse sampling can still carry occluder colors into other pixels.

    Beyond C, pixels remain unchanged. RGB channels share one remap; no color
    conversion is performed. Boundary accuracy is limited by mask rasterization,
    angular sampling, and image interpolation.

    Parameters
    ----------
    image : numpy.ndarray
        Background with shape ``(H, W)`` or ``(H, W, 3)`` in the 0..255 range.
        Values are converted to ``uint8`` before sampling.
    original_mask : numpy.ndarray
        Boolean ``(H, W)`` original face mask, without dilation.
        The furthest external-boundary hit defines the radial source envelope.
    insert_mask : numpy.ndarray
        Boolean ``(H, W)`` support of the face to insert. After erosion, the
        nearest boundary hit defines a radial interior covered by this face.
    face_span : float
        Positive face extent in pixels, scaling the width of the fixed outer
        transition beyond both silhouettes.
    erosion_radius : int, optional
        Fixed insertion-mask erosion radius in pixels, independent of feathering.
        Default: ``INSERT_EROSION_RADIUS_PX`` (2 pixels). Erosion must leave
        the shared center inside the destination silhouette.
    occlusion_mask : numpy.ndarray or None, optional
        Boolean ``(H, W)`` target-space mask; True protects a pixel. Default: None.
        Either occluded sampled endpoint invalidates its direction. Transition
        width is controlled internally by ``OCCLUSION_TRANSITION_ARC_PX``.
    collect_diagnostics : bool, optional
        Return displacement magnitudes and radial Jacobian statistics.
        Default: ``True``. Disabling diagnostics does not change output pixels.

    Returns
    -------
    numpy.ndarray
        Warped uint8 image with the input shape and channel order.
    numpy.ndarray or None
        Displacement magnitude in pixels, shape ``(H, W)``, or None when disabled.
    dict[str, float | int]
        Shared center, ray count, transition width, maximum displacement, and
        minimum analytic radial Jacobian. With an active occlusion mask, also
        includes the blocked-ray count. The Jacobian describes the radial map
        outside protected pixels, not the boundary of their restoration.
        Empty when disabled or the warp is identity.

    Raises
    ------
    ValueError
        If image/mask dimensions or geometric parameters are invalid, erosion
        removes the shared center, or a silhouette is unsuitable for radial
        correspondence. Masks must be nonempty, have a single nondegenerate
        external contour, and supply a positive intersection on every ray.
        Multiple boundary intersections are allowed.
    """
    image = np.asarray(image, dtype=np.uint8)
    original = np.asarray(original_mask, dtype=bool)
    insert = np.asarray(insert_mask, dtype=bool)

    occlusion = None
    if occlusion_mask is not None:
        occlusion = np.asarray(occlusion_mask, dtype=bool)

        if occlusion.shape != image.shape[:2]:
            raise ValueError(
                'Occlusion mask must match the image height and width')

        if not np.any(occlusion):
            occlusion = None

    if image.ndim != 2 and (image.ndim != 3 or image.shape[2] != 3):
        raise ValueError('Image must have shape (H, W) or (H, W, 3)')
    if original.shape != image.shape[:2] or insert.shape != image.shape[:2]:
        raise ValueError('Masks must match the image height and width')
    if not np.isfinite(face_span) or face_span <= 0.0:
        raise ValueError('Face span must be a positive finite number')
    if erosion_radius < 0:
        raise ValueError('Erosion radius must be non-negative')
    if not np.any(original) or not np.any(insert):
        raise ValueError('Radial face warp requires nonempty masks')

    y, x = np.nonzero(original | insert)
    center = np.array([x.mean(), y.mean()])
    inner = insert.copy()
    if erosion_radius:
        size = erosion_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        inner = cv2.erode(insert.astype(np.uint8), kernel) > 0

    if not np.any(inner):
        raise ValueError(
            'Eroded insertion mask is empty; shared center cannot lie inside it')

    contours = [_radial_mask_contour(mask) for mask in (original, inner)]
    center_x, center_y = np.rint(center).astype(int)
    for mask, contour in zip((original, inner), contours):
        if (not mask[center_y, center_x] or cv2.pointPolygonTest(
            contour.astype(np.float32), tuple(center), False,
        ) <= 0):
            raise ValueError(
                'Shared center of the mask union must lie inside both face '
                'masks, including the eroded insertion mask'
            )

    if np.array_equal(original, inner):
        displacement = (
            np.zeros(original.shape,
                     dtype=np.float32) if collect_diagnostics else None
        )
        return image.copy(), displacement, {}

    # Bound angular sample spacing at the furthest silhouette point, then pair
    # both boundaries at exactly the same angles rather than by arc length.
    max_radius = max(np.linalg.norm(contour - center, axis=1).max()
                     for contour in contours)
    ray_count = max(
        MIN_RADIAL_SAMPLE_COUNT,
        int(np.ceil(2 * np.pi * max_radius / RADIAL_SAMPLE_SPACING_PX)),
    )
    # Include the four image-axis directions exactly, preserving silhouette
    # extrema on those rays instead of interpolating across a narrow pixel tip.
    ray_count = ((ray_count + 3) // 4) * 4
    uniform_angles = np.arange(ray_count) * (2 * np.pi / ray_count)
    # Include every contour vertex direction as well: uniform samples alone
    # can miss a narrow raster tip and underestimate the original silhouette.
    # These extra samples preserve narrow extrema between uniform rays.
    vertex_angles = [
        np.mod(np.arctan2((contour - center)
               [:, 1], (contour - center)[:, 0]), 2 * np.pi)
        for contour in contours
    ]
    angles = np.unique(np.concatenate([uniform_angles, *vertex_angles]))
    ray_count = len(angles)
    # The original must fit entirely under the replacement: use its outer
    # radial envelope and the destination's inner radial region respectively.
    original_radii = _radial_contour_radii(
        contours[0], center, angles, outermost=True,
    )
    inserted_radii = _radial_contour_radii(
        contours[1], center, angles, outermost=False,
    )

    ray_weights = None
    blocked_ray_count = 0
    if occlusion is not None:
        ray_weights, blocked_ray_count = _occlusion_ray_weights(
            occlusion, center, angles, original_radii, inserted_radii, max_radius,
        )

    influence_radius = max(
        MIN_WARP_INFLUENCE_RADIUS_PX,
        int(round(face_span * WARP_INFLUENCE_FRACTION)),
    )
    height, width = image.shape[:2]
    min_x = max(0, int(x.min()) - influence_radius - 1)
    max_x = min(width, int(x.max()) + influence_radius + 2)
    min_y = max(0, int(y.min()) - influence_radius - 1)
    max_y = min(height, int(y.max()) + influence_radius + 2)
    grid_y, grid_x = np.mgrid[min_y:max_y, min_x:max_x]
    relative_x = grid_x - center[0]
    relative_y = grid_y - center[1]
    radius = np.hypot(relative_x, relative_y)
    theta = np.arctan2(relative_y, relative_x)
    original_radius = np.interp(
        theta, angles, original_radii, period=2 * np.pi)
    inserted_radius = np.interp(
        theta, angles, inserted_radii, period=2 * np.pi)
    fixed_radius = np.maximum(
        original_radius, inserted_radius) + influence_radius

    # Inverse radial correspondence: output B samples input A, output C samples
    # C. Because C > max(A, B), the annular slope (C-A)/(C-B) stays positive.
    inside = radius <= inserted_radius
    active = radius < fixed_radius
    annular_slope = (fixed_radius - original_radius) / \
        (fixed_radius - inserted_radius)
    source_radius = np.where(
        inside,
        radius * original_radius / inserted_radius,
        original_radius + (radius - inserted_radius) * annular_slope,
    )
    angular_weight = None

    if ray_weights is not None:
        angular_weight = np.interp(
            theta, angles, ray_weights, period=2 * np.pi)
        source_radius = radius + angular_weight * (source_radius - radius)
        active &= angular_weight > 0
        active &= ~occlusion[min_y:max_y, min_x:max_x]

    source_radius = np.where(active, source_radius, radius)
    scale = np.divide(
        source_radius, radius,
        out=original_radius / inserted_radius,
        where=radius > 0,
    )
    map_x = (center[0] + relative_x * scale).astype(np.float32)
    map_y = (center[1] + relative_y * scale).astype(np.float32)
    warped_local = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    result = image.copy()
    region = result[min_y:max_y, min_x:max_x]
    region[active] = warped_local[active]
    if not collect_diagnostics:
        return result, None, {}

    displacement = np.zeros(original.shape, dtype=np.float32)
    displacement[min_y:max_y, min_x:max_x] = np.abs(source_radius - radius)
    # A radial map preserves angle. Its determinant is (source_radius/radius)
    # times the radial derivative, including the center's scale-squared limit.
    derivative = np.where(inside, original_radius /
                          inserted_radius, annular_slope)
    if angular_weight is not None:
        derivative = 1. + angular_weight * (derivative - 1.)

    jacobian = scale * derivative

    return result, displacement, {
        'center_x': float(center[0]),
        'center_y': float(center[1]),
        'ray_count': ray_count,
        'influence_radius': influence_radius,
        'max_displacement': float(displacement.max()),
        'min_jacobian': float(jacobian[active].min()) if np.any(active) else 1.,
        **({'blocked_ray_count': blocked_ray_count} if occlusion is not None else {}),
    }


def composite_face(
    background: np.ndarray,
    face: np.ndarray,
    face_mask: np.ndarray,
    *,
    feather_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Feather a rendered face into a grayscale or RGB background.

    Compute one alpha field from distance to the face-mask boundary. Use it
    directly for grayscale images or broadcast it across the RGB channels.
    Both inputs must use the same channel order; no color conversion occurs.

    Parameters
    ----------
    background : numpy.ndarray
        Prepared background with shape ``(H, W)`` or ``(H, W, 3)``.
    face : numpy.ndarray
        Rendered face with the same shape as the background, in the 0..255 range.
    face_mask : numpy.ndarray
        Boolean ``(H, W)`` support of the rendered face.
    feather_radius : int
        Inward feathering distance in pixels. Nonpositive values use a hard mask.

    Returns
    -------
    numpy.ndarray
        Composite clipped to 0..255 and converted to uint8, preserving image
        shape. Conversion truncates fractional values in both input modes.
    numpy.ndarray
        Float32 alpha field with shape ``(H, W)``, shared by all RGB channels.

    Raises
    ------
    ValueError
        If image shapes differ, the image layout is unsupported, or the mask
        does not match the image height and width.
    """
    background = np.asarray(background, dtype=np.float32)
    face = np.asarray(face, dtype=np.float32)
    mask = np.asarray(face_mask, dtype=bool)

    if background.ndim != 2 and (background.ndim != 3 or background.shape[2] != 3):
        raise ValueError('Images must have shape (H, W) or (H, W, 3)')

    if face.shape != background.shape:
        raise ValueError('Face and background must have matching shapes')

    if mask.shape != background.shape[:2]:
        raise ValueError('Face mask must match the image height and width')

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

    # Keep alpha spatial in the return value; only expand its last axis for
    # broadcasting the identical blend weights across RGB channels.
    weights = alpha[:, :, None] if background.ndim == 3 else alpha
    composed = background * (1.0 - weights) + face * weights

    return np.clip(composed, 0.0, 255.0).astype(np.uint8), alpha


def gray_to_rgb(image_gray: np.ndarray) -> np.ndarray:
    """Repeat a grayscale image over three RGB channels."""
    gray = np.asarray(image_gray, dtype=np.uint8)

    if gray.ndim != 2:
        raise ValueError('Grayscale image must have shape (H, W)')

    return np.repeat(gray[:, :, None], 3, axis=2)
