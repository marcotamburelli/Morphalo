"""
Geometry operations connecting 3DDFA-V3 inference to an RGB face composite.

Images are aligned to the model's 224-pixel crop before coefficient inference.
Source and target coefficients are combined, then the decoded mesh is rasterized
and mapped back to the target image for composition. Diagnostic renders use the
same coordinate transforms as the face inserted into the final image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from third_party.tddfa_v3.runtime import COEFFICIENT_SLICES
from morphalo.nodes.preprocess.utils.color_ops import match_lab_color_statistics
from morphalo.nodes.preprocess.utils.face_compositing import (
    FEATHER_RADIUS_FRACTION, INSERT_EROSION_RADIUS_PX, MIN_FEATHER_RADIUS_PX,
    composite_face, warp_face_background,
)


# Upstream alignment uses 224x224 inputs and focal length 1015 at center 112.
# A 512-pixel diagnostic raster adds sampling resolution, not inferred detail.
RENDER_SIZE = 512

# Exclude raster-edge and occlusion-boundary pixels from color statistics.
# This affects only the sampling region, never visibility or warp geometry.
COLOR_TRANSFER_BORDER_RADIUS_PX = 2


@dataclass(frozen=True)
class AlignedFace:
    """
    Reconstruction input and exact crop-to-image coordinate transforms.

    Attributes
    ----------
    rgb : numpy.ndarray
        Aligned uint8 RGB input with shape ``(224, 224, 3)``.
    image_to_crop : numpy.ndarray
        Homogeneous transform from original pixels to aligned crop pixels.
    crop_to_image : numpy.ndarray
        Inverse homogeneous transform.
    landmarks : numpy.ndarray
        Five original-image landmarks: eyes, nose, and mouth corners.
    """

    rgb: np.ndarray
    image_to_crop: np.ndarray
    crop_to_image: np.ndarray
    landmarks: np.ndarray


def align_reconstruction_face(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    reference: np.ndarray,
) -> AlignedFace:
    """
    Prepare an RGB image for coefficient inference using upstream alignment.

    Convert the MediaPipe mesh into the five facial references expected by
    3DDFA-V3 and fit them to its canonical 3D reference. The fitted scale and
    translation determine a 224-by-224 crop with the framing used by the
    regressor. This changes image framing; it does not frontalize the face.

    Return both the crop and pixel-coordinate transforms so a later render can
    be placed back into the full image. This function prepares source images
    and the target in the same way; inference itself happens in the runtime.

    Parameters
    ----------
    rgb : numpy.ndarray
        Full source or target image with shape ``(H, W, 3)`` and dtype ``uint8``.
    landmarks : numpy.ndarray
        MediaPipe landmarks with at least 468 rows. X/Y must be original-image
        pixel coordinates, with Y increasing downward; Z is not used for fitting.
    reference : numpy.ndarray
        Canonical five-point XYZ reference with shape ``(5, 3)``, ordered as
        the two eye centers, nose, and two mouth corners.

    Returns
    -------
    AlignedFace
        Input crop and transforms preserving upstream integer resize/crop rules.
        Only the landmark provider changes; image colors remain RGB in [0, 255].

    Raises
    ------
    ValueError
        If landmarks or the derived crop geometry are invalid.
    """
    from third_party.tddfa_v3.preprocess import POS, align_img

    points = np.asarray(landmarks, dtype=np.float64)
    if (points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2
            or not np.isfinite(points).all()):
        raise ValueError(
            'Alignment requires at least 468 finite MediaPipe landmarks'
        )

    # Match the order of the canonical reference: eye centers, nose, mouth
    # corners. Averaging each eye's inner/outer corners approximates the eye
    # centers expected by upstream without depending on iris landmarks.
    # The full MediaPipe mesh is only used to locate these five references;
    # the reconstruction network infers geometry from the resulting RGB crop.
    five = np.stack((
        points[[33, 133], :2].mean(axis=0),
        points[[362, 263], :2].mean(axis=0),
        points[1, :2], points[61, :2], points[291, :2],
    ))
    height, width = rgb.shape[:2]
    # MediaPipe pixels have downward-positive Y. POS fits the canonical 3D
    # points to 2D observations in the upstream upward-positive convention.
    upward = five.copy()
    upward[:, 1] = height - 1 - upward[:, 1]
    _, fitted_scale = POS(upward.T, np.asarray(reference).T)
    # 102 is the upstream rescale_factor, fixing face size in the 224 crop.
    # Repeat the scale calculation here to reject unusable resize dimensions
    # before align_img allocates the resized image.
    scale = float(102.0 / fitted_scale)
    resized = np.array([width, height]) * scale
    if (not np.isfinite(scale)
            or scale <= 0
            or resized.min() < 2
            or resized.max() > 16384):
        raise ValueError(
            'Detected face cannot produce a valid reconstruction crop'
        )
    parameters, crop, _, _ = align_img(
        img=Image.fromarray(rgb),
        lm=upward,
        lm3D=reference,
    )
    _, _, scale, tx, ty = parameters

    # Reproduce upstream integer resize dimensions and crop offsets, rather
    # than using the unrounded scale. 112 is half the 224-pixel crop size.
    # These transforms link original image pixels to the actual crop frame.
    resized_width, resized_height = int(width * scale), int(height * scale)
    left = int(resized_width / 2 - 112 + (tx - width / 2) * scale)
    top = int(resized_height / 2 - 112 + (height / 2 - ty) * scale)

    matrix = np.array([
        [resized_width / width, 0, -left],
        [0, resized_height / height, -top],
        [0, 0, 1],
    ], dtype=np.float64)

    return AlignedFace(
        rgb=np.array(crop, dtype=np.uint8),
        image_to_crop=matrix,
        crop_to_image=np.linalg.inv(matrix),
        landmarks=five,
    )


def transfer_reconstruction_parameters(
    source_coefficients: list[np.ndarray],
    target: np.ndarray,
    preserve: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine source identity with the target's pose, placement, and lighting.

    All source vectors share the same morphable-model basis, so their
    arithmetic mean supplies one identity estimate across views. Start from a
    copy of the target vector and replace its identity group unconditionally.
    Expression and albedo are replaced only when listed in ``preserve``;
    rotation, translation, and illumination always remain those of the target.

    The mixed vector is decoded to build the transferred face. The full source
    mean is also returned for diagnostics; its pose and lighting groups are not
    used to place or illuminate the transferred face. Inputs are not modified.

    Parameters
    ----------
    source_coefficients : list[numpy.ndarray]
        Source vectors with shape ``(257,)`` in the same canonical model basis.
    target : numpy.ndarray
        Target coefficient vector with shape ``(257,)``.
    preserve : tuple[str, ...], optional
        Additional source groups: expression and albedo. Default: empty tuple.
        Identity is always transferred and is not an accepted option here.

    Returns
    -------
    numpy.ndarray
        Source mean vector; view-specific groups are diagnostic only.
    numpy.ndarray
        Mixed vector. Identity equals the source mean. Rotation, translation,
        and illumination equal target independently of the selected options.

    Raises
    ------
    ValueError
        If groups or coefficient shapes/values violate the transfer contract.
    """
    if (not isinstance(preserve, (tuple, list))
            or not all(isinstance(group, str) for group in preserve)
            or set(preserve) - {'expression', 'albedo'}):
        raise ValueError(
            'preserve accepts only expression and albedo; identity is implicit'
        )
    sources = np.asarray(source_coefficients, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if (sources.ndim != 2 or sources.shape[1:] != (257,) or not len(sources)
            or target.shape != (257,) or not np.isfinite(sources).all()
            or not np.isfinite(target).all()):
        raise ValueError('Expected finite source and target 257-value vectors')

    # Blend coefficients in the shared model basis, before mesh decoding.
    # Replacing only the selected slices preserves all other target groups.
    mean = sources.mean(axis=0)
    mixed = target.copy()
    identity = COEFFICIENT_SLICES['identity']

    mixed[identity] = mean[identity]

    for group in preserve:
        indices = COEFFICIENT_SLICES[group]
        mixed[indices] = mean[indices]

    return mean, mixed


def render_reconstruction(
    decoded: dict[str, np.ndarray],
    triangles: np.ndarray,
    size: int = RENDER_SIZE,
    *,
    render_fields: tuple[str, ...] = ('color', 'shape'),
) -> dict[str, np.ndarray]:
    """
    Turn a decoded face mesh into crop-space images and a visibility mask.

    Projected vertices are already expressed in the model's 224-pixel image
    frame. Scale their X/Y coordinates to the requested square raster, then
    fill each triangle while retaining the nearest camera-space surface at
    each pixel. Perspective-correct interpolation uses camera Z to interpolate
    both depth and any requested vertex colors.

    ``color`` supplies the target-lit face used in the RGB composite. ``shape``
    supplies a uniform-gray material under fixed studio lighting for inspecting
    geometry in debug images. With neither field requested, rasterization still
    provides the target visibility mask needed to remove the original face,
    without interpolating color attributes. The raster is mapped to the full
    image separately by :func:`reconstruction_to_image`.

    Parameters
    ----------
    decoded : dict[str, numpy.ndarray]
        ``screen_vertices`` contains projected X/Y in the 224-pixel crop frame;
        ``camera_vertices`` supplies positive camera Z, with smaller Z nearer
        the camera. ``colors`` and ``gray_colors`` supply RGB vertex values in
        ``[0, 1]`` when their corresponding render fields are requested.
    triangles : numpy.ndarray
        Zero-based mesh indices with shape ``(T, 3)``.
    size : int, optional
        Square raster size. Default: ``512``.
    render_fields : tuple[str, ...], optional
        Requested RGB fields: ``color`` and/or ``shape``. Default: both.
        An empty tuple computes visibility and depth without color interpolation.

    Returns
    -------
    dict[str, numpy.ndarray]
        Requested uint8 RGB fields, boolean ``mask``, and camera
        ``depth`` (NaN off mesh). Attributes use perspective-correct barycentrics.
        Both triangle orientations are supported; nearer surfaces win.
    """
    xy = np.asarray(
        decoded['screen_vertices'],
        dtype=np.float64
    ) * (size / 224.0)
    depth = np.asarray(decoded['camera_vertices'], dtype=np.float64)[:, 2]
    field_keys = {'color': 'colors', 'shape': 'gray_colors'}
    features = (
        np.concatenate(
            [decoded[field_keys[name]] for name in render_fields],
            axis=1,
        )
        if render_fields else None
    )

    image = np.zeros((size, size, 3 * len(render_fields)), dtype=np.float32)
    buffer = np.full((size, size), np.inf)
    mesh = np.asarray(triangles, dtype=np.int64)

    vertices = xy[mesh]

    low = np.maximum(np.floor(vertices.min(axis=1)).astype(int), 0)
    high = np.minimum(np.ceil(vertices.max(axis=1)).astype(int), size - 1)

    # Process only triangles whose clipped bounding boxes overlap the raster.
    # Each iteration samples pixels in that box, tests triangle membership,
    # and updates the Z-buffer and requested colors where this surface wins.
    for ti in np.flatnonzero(np.all(low <= high, axis=1)):
        indices = mesh[ti]

        # A triangle crossing or behind the camera cannot be interpolated
        # with this positive-depth projection. Such triangles are skipped.
        if np.any(depth[indices] <= 0):
            continue

        (x1, y1), (x2, y2), (x3, y3) = vertices[ti]
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        # The denominator is signed twice-area. Its sign handles either
        # winding; a near-zero area cannot cover a stable triangle in pixels.
        if abs(denominator) < 1e-8:
            continue

        min_x, min_y = low[ti]
        max_x, max_y = high[ti]
        grid_y, grid_x = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        # Screen-space barycentric weights locate each sample relative to
        # the three vertices; nonnegative weights identify interior pixels.
        first = (
            (y2 - y3) * (grid_x - x3) + (x3 - x2) * (grid_y - y3)
        ) / denominator
        second = (
            (y3 - y1) * (grid_x - x3) + (x1 - x3) * (grid_y - y3)
        ) / denominator

        weights = np.stack((first, second, 1 - first - second))
        inside = np.all(weights >= -1e-6, axis=0)
        # Under perspective projection, interpolate reciprocal depth:
        # z = 1 / sum(weight_i / z_i). The same terms, normalized by their
        # sum, give the perspective-correct weights for vertex colors.
        inverse = weights / depth[indices, None, None]
        total = inverse.sum(axis=0)
        candidate = np.divide(
            1.,
            total,
            out=np.full_like(total, np.inf),
            where=total > 0,
        )

        region = (slice(min_y, max_y + 1), slice(min_x, max_x + 1))
        # Overlapping triangles may project onto the same pixel. Keep the
        # smallest positive camera Z, so hidden surfaces do not overwrite it.
        update = inside & (candidate < buffer[region])

        if not np.any(update):
            continue

        if features is not None:
            correct_weights = inverse[:, update] / total[update]
            image[region][update] = correct_weights.T @ features[indices]

        buffer[region][update] = candidate[update]

    mask = np.isfinite(buffer)
    buffer[~mask] = np.nan
    color = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)

    return {
        'mask': mask, 'depth': buffer,
        **{
            name: color[:, :, index * 3:(index + 1) * 3]
            for index, name in enumerate(render_fields)
        },
    }


def reconstruction_to_image(
    raster: dict[str, np.ndarray],
    alignment: AlignedFace,
    shape: tuple,
) -> dict[str, np.ndarray]:
    """
    Place crop-space face renders into their original image coordinates.

    Inference sees a 224-pixel aligned crop, while rasterization may use a
    different resolution. Combine that resolution ratio with the alignment
    transform to sample the raster for every destination pixel in the full
    image. The resulting color and mask share the target image's coordinate
    frame and can therefore be used directly for composition.

    Only present RGB fields are transformed. Depth remains in the crop raster
    and is not included in the returned dictionary. With a visibility-only
    input, this function returns just the original-image face mask.

    Parameters
    ----------
    raster : dict[str, numpy.ndarray]
        Requested square RGB renders and their mask.
    alignment : AlignedFace
        Target or source crop transform used for reconstruction.
    shape : tuple[int, ...]
        Destination image shape, beginning with height and width.

    Returns
    -------
    dict[str, numpy.ndarray]
        Original-sized requested RGB renders and boolean mask.
    """
    height, width = shape[:2]
    # OpenCV's inverse-map mode expects destination-to-source coordinates:
    # original image -> 224 crop -> raster. Scaling the whole affine matrix
    # also scales its translation into raster pixels.
    matrix = alignment.image_to_crop[:2] * (raster['mask'].shape[0] / 224.0)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    output = {
        name: cv2.warpAffine(
            raster[name],
            matrix,
            (width, height),
            flags=flags,
        ) for name in ('color', 'shape') if name in raster
    }

    # Sample the mask with exactly the same transform and interpolation as
    # the RGB fields, then threshold its 0..255 coverage back to boolean.
    # This keeps the inserted face and its support aligned at the boundary.
    mask = cv2.warpAffine(
        raster['mask'].astype(np.uint8) * 255,
        matrix,
        (width, height),
        flags=flags,
    )

    output['mask'] = mask >= 128
    return output


def compose_reconstructed_face(
    target: np.ndarray,
    original_mask: np.ndarray,
    inserted: dict[str, np.ndarray],
    *,
    occlusion_mask: np.ndarray | None = None,
    color_transfer: float = 0.0,
    collect_diagnostics: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Insert the transferred RGB face while closing gaps around its boundary.

    The original target face and the transferred mesh can have different
    silhouettes. Simply pasting the new render can leave a band of the old
    face visible around it. Radially map the unmodified original silhouette
    beneath the insertion silhouette eroded by ``INSERT_EROSION_RADIUS_PX``.
    Use each ray's furthest original intersection to include every old-face
    protrusion, and its nearest destination intersection to stay under the
    new face even when either contour has concavities.
    Rays share the centroid of the mask union; it must be inside both masks,
    including the eroded destination, or placement is rejected. The original
    silhouette is thus fitted under the render instead of merely displaced
    near its edge. A fixed contour beyond both silhouettes anchors a local,
    continuous transition to the unchanged surrounding image.

    Optionally match the rendered colors to the unwarped original target in
    Lab space before blending. Estimate mean/std only from the common interior
    of both masks, excluding occlusion and a small boundary margin. Apply the
    partial correction to the entire visible render, including areas outside
    that sampling intersection. The caller must retain target albedo when this
    option is enabled; source-albedo preservation is incompatible at node level.

    Blend the new face onto that warped background with a soft edge whose
    width is controlled separately by ``FEATHER_RADIUS_FRACTION`` and its pixel
    minimum. Changing feathering does not change the mask erosion. One
    deformation map and one alpha field are shared across the RGB channels.
    An optional target-space occlusion mask blocks warp directions whose
    sampled endpoints are hidden. Full geometric masks still determine the
    center and contours; destination erosion precedes visibility clipping.
    Paste the new face only outside occlusion, then restore the protected
    original target pixels after composition, including its feathered edge.

    This operation deforms existing pixels; it does not generate new hair or
    skin texture. Further image generation can refine the composite. Protecting
    an occluder does not prevent its colors from being sampled elsewhere.

    Parameters
    ----------
    target : numpy.ndarray
        Original target RGB image.
    original_mask : numpy.ndarray
        Target reconstruction mask in the original image frame.
    inserted : dict[str, numpy.ndarray]
        Mixed reconstruction color and mask in that same frame.
    occlusion_mask : numpy.ndarray or None, optional
        Boolean ``(H, W)`` mask aligned with target; True protects original pixels.
        Default: None. This limits visibility rather than changing inferred mesh.
    color_transfer : float, optional
        Lab mean/std matching strength in [0, 1]. Default: 0 disables matching.
        Reference colors come from ``target`` before the background warp.
        Empty sampling interiors leave the render unchanged.
    collect_diagnostics : bool, optional
        Compute displacement and Jacobian statistics. Default: ``True``.

    Returns
    -------
    numpy.ndarray
        RGB composite, suitable as conditioning input or reconstruction preview.
    dict
        Erosion/feather radii in pixels and warp diagnostics. ``warp`` is empty
        when ``collect_diagnostics=False``. Enabled color transfer adds its
        sample count and whether a correction was applied.

    Raises
    ------
    RuntimeError
        If either reconstruction mask contains no visible face.
    ValueError
        If the shared center is outside either face mask after erosion, a mask
        lacks a single valid external contour, a ray has no intersection, or
        ``color_transfer`` is nonfinite or outside [0, 1].
    """
    if not np.isfinite(color_transfer) or not 0.0 <= color_transfer <= 1.0:
        raise ValueError('color_transfer must be finite and in [0, 1]')

    occlusion = None
    if occlusion_mask is not None:
        occlusion = np.asarray(occlusion_mask, dtype=bool)
        if occlusion.shape != target.shape[:2]:
            raise ValueError(
                'Occlusion mask must match the target height and width')

    insert_mask = inserted['mask']
    if not np.any(insert_mask) or not np.any(original_mask):
        raise RuntimeError('Reconstruction render contains no visible face')

    y, x = np.nonzero(original_mask | insert_mask)
    span = float(max(np.ptp(x), np.ptp(y), 1))
    # Feathering controls alpha only. The small fixed destination erosion
    # controls how far the original silhouette retreats under the new face.
    feather_radius = max(
        MIN_FEATHER_RADIUS_PX,
        int(round(span * FEATHER_RADIUS_FRACTION)),
    )

    # RGB is sampled in one pass; geometry and blend weights depend only on
    # masks, so they are computed once and shared by all three channels.
    warped, _, parameters = warp_face_background(
        target,
        original_mask,
        insert_mask,
        face_span=span,
        erosion_radius=INSERT_EROSION_RADIUS_PX,
        occlusion_mask=occlusion,
        collect_diagnostics=collect_diagnostics,
    )

    visible_mask = insert_mask if occlusion is None else insert_mask & ~occlusion
    face_color = inserted['color']
    color_metadata = {}

    if color_transfer > 0.0:
        # Estimate using the unwarped original face, not the deformed corona
        # (which can contain hair/clothing). Remove both raster boundaries and
        # a margin near occluders, whose colors must not influence the match.
        sampling = np.asarray(original_mask, dtype=bool) & visible_mask
        margin = COLOR_TRANSFER_BORDER_RADIUS_PX
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * margin + 1, 2 * margin + 1),
        )
        sampling = cv2.erode(sampling.astype(np.uint8), kernel) > 0
        sample_pixels = int(sampling.sum())

        if sample_pixels:
            # Limit Lab conversions to the visible-face bounding box. Alpha
            # controls application; statistics_mask independently controls fitting.
            ys, xs = np.nonzero(visible_mask)
            region = (
                slice(ys.min(), ys.max() + 1),
                slice(xs.min(), xs.max() + 1),
            )
            rgba = np.dstack((
                face_color[region],
                visible_mask[region].astype(np.uint8) * 255,
            ))
            corrected = match_lab_color_statistics(
                Image.fromarray(rgba),
                Image.fromarray(target[region]),
                strength=color_transfer,
                statistics_mask=sampling[region],
            )

            face_color = face_color.copy()
            face_color[region] = np.asarray(corrected.convert('RGB'))

        color_metadata = {
            'color_transfer': {
                'applied': bool(sample_pixels),
                'sample_pixels': sample_pixels,
            },
        }

    composed, _ = composite_face(
        warped,
        face_color,
        visible_mask,
        feather_radius=feather_radius,
    )

    if occlusion is not None:
        # Restore after both background deformation and face blending: restoring
        # before the paste would allow the rendered face to overwrite occluders.
        composed[occlusion] = target[occlusion]

    return composed, {
        'erosion_radius': INSERT_EROSION_RADIUS_PX,
        'feather_radius': feather_radius,
        'warp': parameters,
        **color_metadata,
    }
