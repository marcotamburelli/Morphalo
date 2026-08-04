import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class CannyEndpointDebugMark:
    x: int
    y: int
    color: tuple[int, int, int]


@dataclass(frozen=True)
class CannyAcceptedSegmentDebug:
    start_xy: tuple[int, int]
    end_xy: tuple[int, int]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class CannyPairExtensionDebug:
    endpoint_marks: list[CannyEndpointDebugMark]
    accepted_segments: list[CannyAcceptedSegmentDebug]
    accepted_extension_mask: np.ndarray
    accepted_endpoint_mask: np.ndarray
    rejection_marks: list[tuple[int, int, tuple[int, int, int]]]


# Canny preprocessing
# -------------------
# The limb Canny map is intentionally built from a single grayscale image
# instead of separate LAB luminance/chromatic maps. In the tested crops, LAB
# ``a`` and ``b`` often contributed little signal while adding opportunities
# for fabric texture and color noise to become internal partition barriers.
#
# ``LIMB_CANNY_REFERENCE_SCALE_PX`` is the limb/crop scale at which the
# original POC values are used unchanged. The runtime scale comes from the
# visible landmark-chain length when available, otherwise from the local mask
# bbox. Scaling only geometric parameters keeps the preprocessing comparable
# across resolutions without making high-resolution crops prohibitively costly.
LIMB_CANNY_REFERENCE_SCALE_PX = 400.0
#
# ``LIMB_CANNY_GAUSSIAN_KERNEL_AT_REFERENCE`` is the odd square blur kernel at
# the reference scale. It grows/shrinks with limb scale, is forced to remain
# odd as OpenCV requires, and is capped by
# ``LIMB_CANNY_GAUSSIAN_KERNEL_MAX``. Larger kernels average away narrow
# wrinkles, denim seams, compression artifacts, and small texture strokes before
# they can become barriers. Too much blur can soften weak useful boundaries, so
# the cap is deliberately low.
LIMB_CANNY_GAUSSIAN_KERNEL_AT_REFERENCE = 9
LIMB_CANNY_GAUSSIAN_KERNEL_MAX = 11
#
# The direct Canny path deliberately keeps preprocessing lighter than
# ``build_smoothed_canny_edge_map``. A small scale-aware blur suppresses dense
# fabric texture before Canny without the stronger region flattening introduced
# by mean-shift.
LIMB_CANNY_DIRECT_GAUSSIAN_KERNEL_AT_REFERENCE = 5
LIMB_CANNY_DIRECT_GAUSSIAN_KERNEL_MAX = 7
#
# ``LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS`` is OpenCV ``pyrMeanShiftFiltering``
# ``sp`` at runtime. The value is derived from
# ``LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS_AT_REFERENCE`` and capped by
# ``LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS_MAX`` because mean-shift is relatively
# expensive. Larger values merge wider nearby regions and reduce local texture;
# smaller values keep finer structure.
LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS_AT_REFERENCE = 9
LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS_MAX = 15
#
# ``LIMB_CANNY_MEAN_SHIFT_COLOR_RADIUS`` is OpenCV ``pyrMeanShiftFiltering``
# ``sr``. It sets how similar grayscale values must be to collapse into the
# same local mode. Smaller values are more sensitive and preserve subtle
# boundaries; larger values flatten stronger tonal differences.
LIMB_CANNY_MEAN_SHIFT_COLOR_RADIUS = 3
#
# ``LIMB_CANNY_MEAN_SHIFT_MAX_LEVEL`` controls the pyramid level used by
# mean-shift. Higher values let the filter reason over coarser image structure,
# which can flatten larger regions but may also erase small useful contours.
LIMB_CANNY_MEAN_SHIFT_MAX_LEVEL = 1
#
# ``LIMB_CANNY_LOW_THRESHOLD`` and ``LIMB_CANNY_HIGH_THRESHOLD`` define the
# Canny hysteresis thresholds.
#
# Pixels whose gradient exceeds the high threshold always start an edge.
# Pixels between the low and high thresholds are retained only when connected
# to a strong edge; otherwise they are discarded. This allows weak sections of
# an otherwise continuous contour to survive while rejecting isolated weak
# texture responses.
#
# Because Gaussian blur and mean-shift already suppress much of the fine
# texture, relatively low thresholds can be used without producing excessive
# spurious edges.
LIMB_CANNY_LOW_THRESHOLD = 10
LIMB_CANNY_HIGH_THRESHOLD = 60
#
# Thresholds used by the direct Canny path. They are intentionally higher than
# the smoothed-path thresholds because this path no longer performs mean-shift
# texture flattening before edge extraction.
LIMB_CANNY_DIRECT_LOW_THRESHOLD = 25
LIMB_CANNY_DIRECT_HIGH_THRESHOLD = 85
#
# Barrier morphology
# ------------------
#
# ``LIMB_CANNY_MIN_EDGE_COMPONENT_MAX_DIM_RATIO`` removes isolated visual-edge
# components whose bounding-box maximum side is smaller than this fraction of
# the local Canny scale. The cleanup runs before endpoint-aware topology
# correction and deliberately ignores synthetic barriers: touching a trusted
# synthetic closure does not make a tiny visual fragment reliable.
#
# Using the bounding-box maximum side instead of component area preserves thin
# but sufficiently long contour fragments while removing compact texture specks.
LIMB_CANNY_MIN_EDGE_COMPONENT_MAX_DIM_RATIO = 0.02
#
# ``LIMB_CANNY_EDGE_CLOSE_RADIUS`` controls morphological closing applied to
# image-derived visual barriers before dilation. Closing may reconnect narrow
# local gaps in a contour, but larger values can also join unrelated nearby
# edges. A value of zero disables this operation.
LIMB_CANNY_EDGE_CLOSE_RADIUS = 0
#
# ``LIMB_CANNY_EDGE_DILATE_RADIUS`` controls the strengthening applied
# independently to visual, semantic, and synthetic barriers before their union
# is used for 8-connected region partitioning. A radius of one prevents common
# diagonal free-space leaks around one-pixel barriers. A value of zero disables
# dilation.
LIMB_CANNY_EDGE_DILATE_RADIUS = 1


@dataclass(frozen=True)
class CannyEdgeMap:
    """
    Canny edge products derived from a source image.

    Parameters
    ----------
    raw_mask : np.ndarray
        Skeletonized Canny edge mask before component cleanup.

    cleaned_mask : np.ndarray
        Edge mask after removal of insignificant connected components.

    processing_domain : np.ndarray
        Binary domain used to clip the extracted edges.

    scale : float
        Runtime geometric scale used to resolve proportional parameters.
    """

    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    processing_domain: np.ndarray
    scale: float


@dataclass(frozen=True)
class CannyBarrierSet:
    """
    Strengthened barriers used for connected-region partitioning.

    Parameters
    ----------
    visual_mask : np.ndarray
        Strengthened image-derived visual barriers after optional anatomical
        correction and visual closing.

    semantic_mask : np.ndarray
        Strengthened barriers derived from semantic-mask boundaries.

    synthetic_mask : np.ndarray
        Strengthened explicit artificial barriers such as proximal and distal
        closures.

    partition_mask : np.ndarray
        Boolean union of the strengthened visual, semantic, and synthetic
        barriers.
    """

    visual_mask: np.ndarray
    semantic_mask: np.ndarray
    synthetic_mask: np.ndarray
    partition_mask: np.ndarray


@dataclass(frozen=True)
class RegionMosaic:
    """
    Connected free-space partition of a binary workspace.

    Parameters
    ----------
    labels : np.ndarray
        Integer component label for every free-space pixel. Label zero denotes
        barriers or pixels outside the workspace.

    workspace_mask : np.ndarray
        Geometric domain partitioned by the barriers.

    barrier_mask : np.ndarray
        Final barrier mask used for partitioning.

    boundary_labels : frozenset[int]
        Labels of regions touching the workspace boundary.

    region_count : int
        Number of foreground regions, excluding label zero.
    """

    labels: np.ndarray
    workspace_mask: np.ndarray
    barrier_mask: np.ndarray
    boundary_labels: frozenset[int]
    region_count: int


def neighbors8(
    y: int,
    x: int,
    *,
    shape: tuple[int, int],
    mask: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """
    Return in-bounds 8-neighbor coordinates in deterministic scan order.

    When ``mask`` is provided, only foreground neighbors are returned.
    """
    height, width = shape
    result: list[tuple[int, int]] = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            ny = y + dy
            nx = x + dx

            if not (0 <= ny < height and 0 <= nx < width):
                continue

            if mask is not None and not mask[ny, nx]:
                continue

            result.append((ny, nx))

    return result


def thin_binary_edges(edge_mask: np.ndarray) -> np.ndarray:
    """
    Return a one-pixel skeleton of a binary edge mask.

    Semantic-mask boundaries can become two or more pixels thick when adjacent
    label interiors contribute both sides of a contour. Thinning restores a
    traversable 8-connected line graph before endpoint logic walks the edge.
    """
    edge = np.asarray(edge_mask).astype(bool)
    if edge.ndim != 2:
        raise ValueError('edge_mask must be HxW')
    if not np.any(edge):
        return np.zeros_like(edge, dtype=bool)
    return skeletonize(edge).astype(bool)


# --------------------------------------------------------------------------
# Scale utilities
# --------------------------------------------------------------------------


def resolve_limb_canny_scale(
    *,
    input_mask: np.ndarray,
    limb_chain_length: float,
) -> float:
    """
    Resolve the proportional scale used by Canny topology corrections.

    Parameters
    ----------
    input_mask : np.ndarray
        Local boolean limb mask used as fallback support.
    limb_chain_length : float
        Proximal-to-distal landmark-chain length in pixels.

    Returns
    -------
    float
        Positive scale in pixels. A valid landmark-chain length is preferred;
        otherwise the maximum side of the local mask bbox is used.
    """
    chain_length = float(limb_chain_length)
    if np.isfinite(chain_length) and chain_length > 1.0:
        return chain_length

    ys, xs = np.where(input_mask.astype(bool))
    if xs.size > 0:
        return float(
            max(
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
        )

    return float(max(input_mask.shape[:2]))


def scaled_int(
    scale: float,
    ratio: float,
) -> int:
    """
    Convert a proportional Canny parameter to integer pixels by truncation.

    Parameters
    ----------
    scale : float
        Runtime limb scale in pixels.
    ratio : float
        Scale fraction.

    Returns
    -------
    int
        Truncated pixel value. Values below one naturally disable optional
        correction stages that require a positive radius or length.
    """
    value = float(scale) * float(ratio)
    if not np.isfinite(value):
        return 0
    return int(value)


def _scaled_reference_int(
    *,
    scale: float,
    reference_scale: float,
    value_at_reference: int,
    min_value: int,
    max_value: int,
) -> int:
    """
    Scale an integer preprocessing parameter from a reference limb size.
    """
    if reference_scale <= 0.0:
        return int(value_at_reference)

    scale_factor = float(scale) / float(reference_scale)
    value = int(round(float(value_at_reference) * scale_factor))
    return max(
        int(min_value),
        min(int(max_value), value),
    )


def _nearest_odd_not_above(
    value: int,
    *,
    min_value: int,
    max_value: int,
) -> int:
    """
    Clamp a value and return an odd integer without exceeding the upper bound.
    """
    value = max(
        int(min_value),
        min(int(max_value), int(value)),
    )

    if value % 2 == 1:
        return value

    if value + 1 <= max_value:
        return value + 1

    return max(int(min_value), value - 1)


# --------------------------------------------------------------------------
# Generic edge cleanup
# --------------------------------------------------------------------------


def remove_small_edge_segments(
    edges: np.ndarray,
    *,
    min_max_dim: int,
) -> np.ndarray:
    """
    Remove short isolated Canny edge segments.

    Components are computed with 8-connectivity. A component is removed when the
    maximum side of its bounding box is less than ``min_max_dim``. This favors
    preserving thin but long Canny fragments while dropping tiny texture specks.
    """
    edge_mask = np.asarray(edges)

    if edge_mask.ndim != 2:
        raise ValueError(
            'edges must be two-dimensional, '
            f'got shape {edge_mask.shape!r}.'
        )

    if min_max_dim <= 0:
        return edge_mask.astype(bool)

    cm = (edge_mask != 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return edge_mask.astype(bool)

    keep = np.zeros(num, dtype=bool)
    keep[0] = False

    for label in range(1, num):
        max_dim = max(
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        if max_dim >= int(min_max_dim):
            keep[label] = True

    return keep[labels]


# --------------------------------------------------------------------------
# Canny extraction
# --------------------------------------------------------------------------


def build_canny_edge_map(
    *,
    local_rgb: np.ndarray,
    local_mask: np.ndarray,
    limb_chain_length: float,
) -> CannyEdgeMap:
    """
    Extract and clean a skeletonized Canny edge map from the original image.

    This experimental implementation runs Canny on the grayscale source image
    after only a light scale-aware Gaussian blur. It intentionally avoids the
    stronger mean-shift flattening used by ``build_smoothed_canny_edge_map`` while
    still suppressing dense clothing texture before it becomes endpoint-heavy
    topology.
    """
    source_rgb = np.asarray(local_rgb)
    input_mask = np.asarray(local_mask)

    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise ValueError(
            'local_rgb must have shape (height, width, 3), '
            f'got {source_rgb.shape!r}.'
        )

    if input_mask.ndim != 2:
        raise ValueError(
            'local_mask must be two-dimensional, '
            f'got shape {input_mask.shape!r}.'
        )

    if source_rgb.shape[:2] != input_mask.shape:
        raise ValueError(
            'local_rgb and local_mask must have matching spatial dimensions, '
            f'got {source_rgb.shape[:2]!r} and {input_mask.shape!r}.'
        )

    input_mask = input_mask.astype(bool)
    canny_scale = resolve_limb_canny_scale(
        input_mask=input_mask,
        limb_chain_length=limb_chain_length,
    )

    if not np.any(input_mask):
        empty = np.zeros_like(input_mask, dtype=bool)
        return CannyEdgeMap(
            raw_mask=empty,
            cleaned_mask=empty.copy(),
            processing_domain=input_mask,
            scale=canny_scale,
        )

    canny_min_edge_component_max_dim = int(math.ceil(
        canny_scale * LIMB_CANNY_MIN_EDGE_COMPONENT_MAX_DIM_RATIO
    ))
    gaussian_kernel_size = _nearest_odd_not_above(
        _scaled_reference_int(
            scale=canny_scale,
            reference_scale=LIMB_CANNY_REFERENCE_SCALE_PX,
            value_at_reference=(
                LIMB_CANNY_DIRECT_GAUSSIAN_KERNEL_AT_REFERENCE
            ),
            min_value=3,
            max_value=LIMB_CANNY_DIRECT_GAUSSIAN_KERNEL_MAX,
        ),
        min_value=3,
        max_value=LIMB_CANNY_DIRECT_GAUSSIAN_KERNEL_MAX,
    )
    gaussian_sigma = max(
        1,
        int(round(float(gaussian_kernel_size) / 4.0)),
    )

    gray = cv2.cvtColor(
        source_rgb.astype(np.uint8, copy=False),
        cv2.COLOR_RGB2GRAY,
    )
    gray = cv2.GaussianBlur(
        gray,
        (gaussian_kernel_size, gaussian_kernel_size),
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
    )
    canny = cv2.Canny(
        gray,
        LIMB_CANNY_DIRECT_LOW_THRESHOLD,
        LIMB_CANNY_DIRECT_HIGH_THRESHOLD,
        L2gradient=True,
    )

    from skimage.morphology import skeletonize

    canny = skeletonize(canny > 0).astype(bool)
    canny &= input_mask

    cleaned_canny = remove_small_edge_segments(
        canny,
        min_max_dim=canny_min_edge_component_max_dim,
    )

    return CannyEdgeMap(
        raw_mask=canny,
        cleaned_mask=cleaned_canny,
        processing_domain=input_mask,
        scale=canny_scale,
    )


def build_smoothed_canny_edge_map(
    *,
    local_rgb: np.ndarray,
    local_mask: np.ndarray,
    limb_chain_length: float,
) -> CannyEdgeMap:
    """
    Extract and clean a skeletonized Canny edge map inside a binary domain.

    The function converts the supplied RGB image to grayscale, suppresses small
    tonal and textural variations, extracts Canny edges, skeletonizes them to a
    one-pixel representation, clips them to ``local_mask``, and removes
    insignificant connected edge fragments.

    Spatial preprocessing parameters are scaled from ``limb_chain_length``.
    When that value is invalid or too small, the maximum side of the foreground
    bounding box of ``local_mask`` is used as a fallback scale. If the mask is
    empty, the maximum image dimension becomes the final fallback.

    The processing pipeline is:

    1. normalize ``local_mask`` to a boolean processing domain;
    2. resolve the runtime geometric scale;
    3. derive the Gaussian kernel size, Gaussian sigma, mean-shift spatial
       radius, and minimum accepted edge-component extent from that scale;
    4. convert the source image from RGB to grayscale;
    5. apply Gaussian blur to attenuate narrow wrinkles, texture strokes,
       compression artifacts, and small local intensity variations;
    6. apply pyramidal mean-shift filtering to flatten nearby grayscale regions
       while preserving stronger structural transitions;
    7. run Canny edge detection with the configured hysteresis thresholds;
    8. skeletonize the binary Canny result to a one-pixel edge topology;
    9. clip the skeletonized edges to the processing domain;
    10. remove connected edge components whose bounding-box maximum dimension
        is below the scale-dependent threshold.

    This function performs only image-derived edge extraction and initial
    cleanup. It does not:

    - restrict edges by distance from an anatomical skeleton;
    - bridge interrupted contours;
    - connect terminal endpoints;
    - create distal or proximal closures;
    - merge semantic or synthetic barriers;
    - apply barrier strengthening morphology;
    - partition a workspace into connected regions;
    - select or validate target regions.

    Those operations belong to later edge-refinement, barrier-construction, and
    region-partition stages.

    Parameters
    ----------
    local_rgb : np.ndarray
        Local RGB source image with shape ``(height, width, 3)``.

        The image is converted to ``uint8`` without copying when possible before
        OpenCV preprocessing. Values are therefore expected to already represent
        standard 8-bit RGB intensities in the range ``0..255``.

        The spatial dimensions must match ``local_mask``.

    local_mask : np.ndarray
        Two-dimensional binary mask defining the processing domain.

        Non-zero pixels are treated as foreground. Skeletonized Canny pixels
        outside this domain are discarded before connected-component cleanup.

        The mask normally represents a semantic or geometric support region, but
        this function does not interpret its meaning.

    limb_chain_length : float
        Preferred geometric scale in pixels.

        For limb processing, this is normally the summed length of the visible
        proximal-to-distal landmark chain. A finite value greater than one pixel
        is used directly.

        When the value is not finite or is less than or equal to one, the scale
        is derived from the maximum side of the foreground bounding box of
        ``local_mask``. If the mask is empty, the maximum spatial dimension of
        the mask is used.

        The resolved scale controls only spatial parameters. Intensity-domain
        parameters such as the Canny thresholds and mean-shift color radius
        remain unchanged.

    Returns
    -------
    CannyEdgeMap
        Structured edge-extraction result containing:

        ``raw_mask``
            Boolean one-pixel Canny skeleton after clipping to
            ``processing_domain`` and before small-component cleanup.

        ``cleaned_mask``
            Boolean edge mask after removing connected components whose
            bounding-box maximum dimension is below the scale-dependent
            threshold.

        ``processing_domain``
            Boolean normalized copy or view of ``local_mask`` used to clip the
            edge map.

        ``scale``
            Resolved positive geometric scale in pixels used to derive spatial
            preprocessing and cleanup parameters.

    Raises
    ------
    ValueError
        If ``local_rgb`` is not a three-dimensional RGB image, if
        ``local_mask`` is not two-dimensional, or if their spatial dimensions
        differ.

    Notes
    -----
    The returned masks preserve the local image coordinate system.

    Connected edge components are evaluated with 8-connectivity. Cleanup uses
    the maximum side of each component bounding box rather than its area. This
    preserves thin but sufficiently long contour fragments while removing
    compact texture specks.

    Gaussian blur and mean-shift filtering deliberately precede Canny detection.
    Their purpose is not to produce a semantically simplified image, but to
    reduce small internal variations that would otherwise become barriers during
    later region partitioning.

    The edge skeleton is clipped before component cleanup. Consequently, a
    visual contour crossing the boundary of ``local_mask`` may be split into
    multiple local components, and each resulting component is evaluated
    independently.

    Empty processing domains return a ``CannyEdgeMap`` whose ``raw_mask`` and
    ``cleaned_mask`` are empty boolean arrays with the same spatial shape as
    ``local_mask``.
    """

    source_rgb = np.asarray(local_rgb)
    input_mask = np.asarray(local_mask)

    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise ValueError(
            'local_rgb must have shape (height, width, 3), '
            f'got {source_rgb.shape!r}.'
        )

    if input_mask.ndim != 2:
        raise ValueError(
            'local_mask must be two-dimensional, '
            f'got shape {input_mask.shape!r}.'
        )

    if source_rgb.shape[:2] != input_mask.shape:
        raise ValueError(
            'local_rgb and local_mask must have matching spatial dimensions, '
            f'got {source_rgb.shape[:2]!r} and {input_mask.shape!r}.'
        )

    input_mask = input_mask.astype(bool)

    if not np.any(input_mask):
        empty = np.zeros_like(input_mask, dtype=bool)
        return CannyEdgeMap(
            raw_mask=empty,
            cleaned_mask=empty.copy(),
            processing_domain=input_mask,
            scale=resolve_limb_canny_scale(
                input_mask=input_mask,
                limb_chain_length=limb_chain_length,
            ),
        )

    canny_scale = resolve_limb_canny_scale(
        input_mask=input_mask,
        limb_chain_length=limb_chain_length,
    )
    canny_min_edge_component_max_dim = int(math.ceil(
        canny_scale * LIMB_CANNY_MIN_EDGE_COMPONENT_MAX_DIM_RATIO
    ))

    gaussian_kernel_size = _nearest_odd_not_above(
        _scaled_reference_int(
            scale=canny_scale,
            reference_scale=LIMB_CANNY_REFERENCE_SCALE_PX,
            value_at_reference=LIMB_CANNY_GAUSSIAN_KERNEL_AT_REFERENCE,
            min_value=3,
            max_value=LIMB_CANNY_GAUSSIAN_KERNEL_MAX,
        ),
        min_value=3,
        max_value=LIMB_CANNY_GAUSSIAN_KERNEL_MAX,
    )
    gaussian_sigma = max(
        1,
        int(round(float(gaussian_kernel_size) / 4.0)),
    )
    mean_shift_spatial_radius = _scaled_reference_int(
        scale=canny_scale,
        reference_scale=LIMB_CANNY_REFERENCE_SCALE_PX,
        value_at_reference=(
            LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS_AT_REFERENCE
        ),
        min_value=3,
        max_value=LIMB_CANNY_MEAN_SHIFT_SPATIAL_RADIUS_MAX,
    )

    gray = cv2.cvtColor(
        source_rgb.astype(np.uint8, copy=False),
        cv2.COLOR_RGB2GRAY,
    )
    gray = cv2.GaussianBlur(
        gray,
        (gaussian_kernel_size, gaussian_kernel_size),
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
    )
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    flattened_bgr = cv2.pyrMeanShiftFiltering(
        gray_bgr,
        sp=mean_shift_spatial_radius,
        sr=LIMB_CANNY_MEAN_SHIFT_COLOR_RADIUS,
        maxLevel=LIMB_CANNY_MEAN_SHIFT_MAX_LEVEL,
    )
    flattened_gray = cv2.cvtColor(flattened_bgr, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(
        flattened_gray,
        LIMB_CANNY_LOW_THRESHOLD,
        LIMB_CANNY_HIGH_THRESHOLD,
        L2gradient=True,
    )

    from skimage.morphology import skeletonize

    canny = skeletonize(canny > 0).astype(bool)
    canny &= input_mask

    cleaned_canny = remove_small_edge_segments(
        canny,
        min_max_dim=canny_min_edge_component_max_dim,
    )

    return CannyEdgeMap(
        raw_mask=canny,
        cleaned_mask=cleaned_canny,
        processing_domain=input_mask,
        scale=canny_scale,
    )


# --------------------------------------------------------------------------
# Anatomical edge refinement
# --------------------------------------------------------------------------


def bridge_consistent_edge_endpoints(
    edges: np.ndarray,
    *,
    allowed_domain: np.ndarray,
    max_gap: float,
    tangent_radius: int = 6,
    min_facing_alignment: float = 0.70,
    min_parallelism: float = 0.80,
    min_allowed_fraction: float = 0.90,
    bridge_thickness: int = 1,
) -> np.ndarray:
    """
    Bridge short gaps between geometrically consistent edge endpoints.

    The function reconnects fragmented contours without applying a blind
    morphological closing to the complete edge mask. Candidate endpoint pairs
    are accepted only when their local outgoing directions are mutually
    compatible with the gap between them.

    The processing pipeline is:

    1. find one-pixel edge endpoints with exactly one connected neighbor;
    2. estimate the outgoing tangent at each endpoint from its local component;
    3. generate endpoint pairs within ``max_gap``;
    4. reject pairs whose tangents do not face each other;
    5. reject pairs whose local contour directions are not approximately
       collinear;
    6. reject bridges that leave ``allowed_domain`` for too much of their
       length;
    7. sort accepted pairs by geometric cost and connect them greedily.

    Each endpoint is used at most once. This prevents one noisy endpoint from
    producing several artificial branches.

    Parameters
    ----------
    edges : np.ndarray
        Two-dimensional boolean or binary one-pixel edge mask. Non-zero pixels
        are treated as edge pixels.
    allowed_domain : np.ndarray
        Two-dimensional boolean mask defining where synthetic bridge pixels may
        be created. This will normally be restricted to the limb-local input
        mask neighborhood and to a maximum distance from the limb skeleton.
    max_gap : float
        Maximum Euclidean distance in pixels between endpoints that may be
        connected. Values less than or equal to zero disable bridging.
    tangent_radius : int, default=6
        Maximum local graph distance, measured in skeleton pixels, used to
        estimate the outgoing direction of each endpoint. Larger values produce
        more stable directions on smooth contours but may be less local around
        strongly curved edges.
    min_facing_alignment : float, default=0.70
        Minimum cosine alignment required between each endpoint's outgoing
        tangent and the direction toward the opposite endpoint. ``1.0`` means
        exact alignment, ``0.0`` means perpendicular.
    min_parallelism : float, default=0.80
        Minimum absolute cosine similarity between the two local contour
        tangents. The absolute value is used because the tangents point outward
        from opposite sides of the same gap and are therefore expected to have
        opposite signs.
    min_allowed_fraction : float, default=0.90
        Minimum fraction of rasterized bridge pixels that must fall inside
        ``allowed_domain``.
    bridge_thickness : int, default=1
        Thickness in pixels of accepted synthetic bridge segments.

    Returns
    -------
    np.ndarray
        Boolean edge mask containing the original edges plus accepted synthetic
        bridge segments.

    Raises
    ------
    ValueError
        If input masks are not two-dimensional, have different shapes, or if
        configuration values are outside their valid ranges.

    Notes
    -----
    This is a geometric heuristic. It cannot distinguish semantically unrelated
    contours that happen to be locally collinear, so ``allowed_domain`` should
    remain conservative.
    """
    edge_mask = np.asarray(edges).astype(bool)
    domain_mask = np.asarray(allowed_domain).astype(bool)

    if edge_mask.ndim != 2:
        raise ValueError(
            f'edges must be two-dimensional, got shape {edge_mask.shape!r}.'
        )

    if domain_mask.ndim != 2:
        raise ValueError(
            'allowed_domain must be two-dimensional, '
            f'got shape {domain_mask.shape!r}.'
        )

    if edge_mask.shape != domain_mask.shape:
        raise ValueError(
            'edges and allowed_domain must have the same shape, got '
            f'{edge_mask.shape!r} and {domain_mask.shape!r}.'
        )

    if max_gap <= 0:
        return edge_mask.copy()

    if tangent_radius < 1:
        raise ValueError(
            f'tangent_radius must be >= 1, got {tangent_radius!r}.'
        )

    if not (0.0 <= min_facing_alignment <= 1.0):
        raise ValueError(
            'min_facing_alignment must be in [0, 1], got '
            f'{min_facing_alignment!r}.'
        )

    if not (0.0 <= min_parallelism <= 1.0):
        raise ValueError(
            f'min_parallelism must be in [0, 1], got {min_parallelism!r}.'
        )

    if not (0.0 <= min_allowed_fraction <= 1.0):
        raise ValueError(
            'min_allowed_fraction must be in [0, 1], got '
            f'{min_allowed_fraction!r}.'
        )

    if bridge_thickness < 1:
        raise ValueError(
            f'bridge_thickness must be >= 1, got {bridge_thickness!r}.'
        )

    if not np.any(edge_mask):
        return edge_mask.copy()

    def _trace_endpoint_tangent(
        endpoint_y: int,
        endpoint_x: int,
        component_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate the direction pointing outward from a skeleton endpoint.

        A breadth-first traversal follows the local component away from the
        endpoint. The endpoint-to-farthest-local-point vector points inward
        along the existing contour, so its inverse is the outgoing tangent
        across the missing gap.
        """
        start = (endpoint_y, endpoint_x)
        queue: list[tuple[tuple[int, int], int]] = [(start, 0)]
        visited = {start}

        farthest = start
        farthest_distance = 0

        queue_index = 0
        while queue_index < len(queue):
            (current_y, current_x), graph_distance = queue[queue_index]
            queue_index += 1

            if graph_distance > farthest_distance:
                farthest = (current_y, current_x)
                farthest_distance = graph_distance

            if graph_distance >= tangent_radius:
                continue

            for neighbor in neighbors8(
                current_y,
                current_x,
                shape=component_mask.shape,
                mask=component_mask,
            ):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, graph_distance + 1))

        if farthest_distance < 2:
            return None

        inward = np.asarray(
            [
                float(farthest[1] - endpoint_x),
                float(farthest[0] - endpoint_y),
            ],
            dtype=np.float32,
        )
        inward_length = float(np.linalg.norm(inward))

        if inward_length < 1e-6:
            return None

        return -inward / inward_length

    def _line_pixels(
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rasterize a one-pixel line and return its x/y pixel coordinates.
        """
        line_mask = np.zeros_like(edge_mask, dtype=np.uint8)

        cv2.line(
            line_mask,
            start_xy,
            end_xy,
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )

        ys, xs = np.where(line_mask > 0)
        return xs, ys

    component_count, component_labels = cv2.connectedComponents(
        edge_mask.astype(np.uint8),
        connectivity=8,
    )

    endpoints: list[dict[str, object]] = []

    for component_id in range(1, component_count):
        component_mask = component_labels == component_id
        ys, xs = np.where(component_mask)

        if xs.size < 3:
            continue

        for y, x in zip(ys.tolist(), xs.tolist()):
            neighbor_count = len(neighbors8(
                y,
                x,
                shape=component_mask.shape,
                mask=component_mask,
            ))

            if neighbor_count != 1:
                continue

            tangent = _trace_endpoint_tangent(
                y,
                x,
                component_mask,
            )

            if tangent is None:
                continue

            endpoints.append({
                'x': int(x),
                'y': int(y),
                'component_id': int(component_id),
                'tangent': tangent,
            })

    if len(endpoints) < 2:
        return edge_mask.copy()

    candidates: list[
        tuple[
            float,
            int,
            int,
            tuple[int, int],
            tuple[int, int],
        ]
    ] = []

    for first_idx in range(len(endpoints)):
        first = endpoints[first_idx]
        first_xy = np.asarray(
            [float(first['x']), float(first['y'])],
            dtype=np.float32,
        )
        first_tangent = np.asarray(
            first['tangent'],
            dtype=np.float32,
        )

        for second_idx in range(first_idx + 1, len(endpoints)):
            second = endpoints[second_idx]

            # Bridging endpoints from the same component can be useful for a
            # contour interrupted by a small gap. It is therefore deliberately
            # allowed rather than rejected here.
            second_xy = np.asarray(
                [float(second['x']), float(second['y'])],
                dtype=np.float32,
            )
            gap_vector = second_xy - first_xy
            gap_length = float(np.linalg.norm(gap_vector))

            if gap_length < 1.0 or gap_length > float(max_gap):
                continue

            gap_direction = gap_vector / gap_length

            second_tangent = np.asarray(
                second['tangent'],
                dtype=np.float32,
            )

            first_alignment = float(
                np.dot(first_tangent, gap_direction)
            )
            second_alignment = float(
                np.dot(second_tangent, -gap_direction)
            )
            parallelism = abs(float(
                np.dot(first_tangent, second_tangent)
            ))

            if first_alignment < min_facing_alignment:
                continue

            if second_alignment < min_facing_alignment:
                continue

            if parallelism < min_parallelism:
                continue

            start_xy = (
                int(first['x']),
                int(first['y']),
            )
            end_xy = (
                int(second['x']),
                int(second['y']),
            )

            line_xs, line_ys = _line_pixels(
                start_xy,
                end_xy,
            )

            if line_xs.size == 0:
                continue

            allowed_fraction = float(np.mean(
                domain_mask[line_ys, line_xs]
            ))

            if allowed_fraction < min_allowed_fraction:
                continue

            # Shorter gaps and better aligned tangents receive a lower cost.
            # Endpoint pairs are later accepted greedily in ascending order.
            distance_cost = gap_length / float(max_gap)
            facing_cost = (
                (1.0 - first_alignment)
                + (1.0 - second_alignment)
            )
            parallel_cost = 0.5 * (1.0 - parallelism)
            domain_cost = 0.5 * (1.0 - allowed_fraction)

            score = (
                distance_cost
                + facing_cost
                + parallel_cost
                + domain_cost
            )

            candidates.append((
                score,
                first_idx,
                second_idx,
                start_xy,
                end_xy,
            ))

    if not candidates:
        return edge_mask.copy()

    candidates.sort(key=lambda item: item[0])

    bridged = edge_mask.astype(np.uint8) * 255
    used_endpoints: set[int] = set()

    for (
        _,
        first_idx,
        second_idx,
        start_xy,
        end_xy,
    ) in candidates:
        if (
            first_idx in used_endpoints
            or second_idx in used_endpoints
        ):
            continue

        bridge_mask = np.zeros_like(bridged)

        cv2.line(
            bridge_mask,
            start_xy,
            end_xy,
            255,
            thickness=bridge_thickness,
            lineType=cv2.LINE_8,
        )

        # Enforce the domain at pixel level as well. The fraction test permits
        # small rasterization imperfections, while this final clipping prevents
        # bridge pixels from being created outside the allowed region.
        bridge_mask[~domain_mask] = 0

        if not np.any(bridge_mask):
            continue

        bridged = cv2.bitwise_or(
            bridged,
            bridge_mask,
        )

        used_endpoints.add(first_idx)
        used_endpoints.add(second_idx)

    return bridged > 0


def _terminal_branch_direction(
    branch_pixels: list[tuple[int, int]],
) -> Optional[np.ndarray]:
    """
    Return the outgoing direction of a traced terminal edge branch.

    Parameters
    ----------
    branch_pixels : list of tuple[int, int]
        Pixel coordinates ordered from the terminal endpoint toward the interior
        of the connected edge component. Coordinates use image-array order
        ``(y, x)``.

    Returns
    -------
    np.ndarray or None
        Unit vector in image-coordinate order ``(x, y)`` pointing from the
        component interior back through the endpoint, i.e. along the natural
        prolongation direction. ``None`` is returned when fewer than two branch
        pixels are provided or when the endpoint and interior pixel collapse to
        the same coordinate.
    """
    if len(branch_pixels) < 2:
        return None

    endpoint_y, endpoint_x = branch_pixels[0]
    inner_y, inner_x = branch_pixels[-1]
    outgoing = np.asarray(
        [
            float(endpoint_x - inner_x),
            float(endpoint_y - inner_y),
        ],
        dtype=np.float32,
    )
    outgoing_length = float(np.linalg.norm(outgoing))
    if outgoing_length <= 1e-6:
        return None

    return outgoing / outgoing_length


def prolong_endpoint(
    *,
    branch_pixels: list[tuple[int, int]],
    allowed_domain: np.ndarray,
    stop_mask: np.ndarray,
    thickness: int = 1,
    max_length: Optional[float] = None,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """
    Extend one endpoint along its terminal-branch direction.

    The outgoing direction is derived from ``branch_pixels`` with
    ``_terminal_branch_direction``. The ray marches inside ``allowed_domain`` from
    the first branch pixel and stops at the first contact with ``stop_mask`` or
    at the last in-domain pixel before the workspace boundary. The returned mask
    contains the rasterized segment from the endpoint to that stop pixel and is
    clipped to ``allowed_domain``.

    Parameters
    ----------
    branch_pixels : list[tuple[int, int]]
        Terminal branch pixels ordered from endpoint toward the component
        interior, expressed in image-array order ``(y, x)``. The first pixel is
        used as the endpoint start coordinate.
    allowed_domain : np.ndarray
        Binary workspace in which extension pixels may be drawn.
    stop_mask : np.ndarray
        Binary mask of existing structural or synthetic barriers that should stop
        the ray. The start pixel itself is ignored.
    thickness : int, default=1
        Rasterized segment thickness in pixels.
    max_length : float or None, optional
        Optional upper bound on the ray length. When omitted, the image diagonal
        is used.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int] | None]
        Boolean extension mask and the selected stop coordinate. ``None`` is
        returned when no drawable extension can be produced.
    """
    domain = np.asarray(allowed_domain).astype(bool)
    blockers = np.asarray(stop_mask).astype(bool)

    if domain.ndim != 2:
        raise ValueError(
            'allowed_domain must be two-dimensional, '
            f'got shape {domain.shape!r}.'
        )
    if blockers.shape != domain.shape:
        raise ValueError(
            'stop_mask must have the same shape as allowed_domain, got '
            f'{blockers.shape!r} and {domain.shape!r}.'
        )

    height, width = domain.shape
    if len(branch_pixels) < 1:
        return np.zeros_like(domain, dtype=bool), None

    start_y, start_x = branch_pixels[0]
    start_x = int(start_x)
    start_y = int(start_y)
    if not (0 <= start_x < width and 0 <= start_y < height):
        return np.zeros_like(domain, dtype=bool), None
    if not domain[start_y, start_x]:
        return np.zeros_like(domain, dtype=bool), None

    direction = _terminal_branch_direction(branch_pixels)
    if direction is None:
        return np.zeros_like(domain, dtype=bool), None
    max_steps = int(math.ceil(
        float(max_length)
        if max_length is not None
        else math.hypot(float(width), float(height))
    ))
    if max_steps <= 0:
        return np.zeros_like(domain, dtype=bool), None

    last_xy = (start_x, start_y)
    visited: set[tuple[int, int]] = {(start_x, start_y)}
    stop_xy: Optional[tuple[int, int]] = None

    for step in range(1, max_steps + 1):
        point_x = float(start_x) + float(direction[0]) * float(step)
        point_y = float(start_y) + float(direction[1]) * float(step)
        x = int(round(point_x))
        y = int(round(point_y))

        if (x, y) in visited:
            continue
        visited.add((x, y))

        if not (0 <= x < width and 0 <= y < height):
            stop_xy = last_xy
            break

        if not domain[y, x]:
            stop_xy = last_xy
            break

        last_xy = (x, y)

        if blockers[y, x] and (x, y) != (start_x, start_y):
            stop_xy = (x, y)
            break

    if stop_xy is None:
        stop_xy = last_xy

    if stop_xy == (start_x, start_y):
        return np.zeros_like(domain, dtype=bool), None

    extension_u8 = np.zeros_like(domain, dtype=np.uint8)
    cv2.line(
        extension_u8,
        (start_x, start_y),
        stop_xy,
        255,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_8,
    )
    extension_u8[~domain] = 0

    extension_mask = extension_u8 > 0
    if not np.any(extension_mask):
        return np.zeros_like(domain, dtype=bool), None

    return extension_mask, stop_xy


def _prolong_unconnected_limb_endpoints(
    *,
    active_endpoints: list[dict[str, object]],
    unconnected_indices: list[int],
    result_u8: np.ndarray,
    domain_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    synthetic_stop_mask: np.ndarray,
    bridge_thickness: int,
    debug_color: tuple[int, int, int],
    rejection_color: tuple[int, int, int],
    accepted_extension_mask: np.ndarray,
    accepted_endpoint_mask: np.ndarray,
    accepted_segments: list[CannyAcceptedSegmentDebug],
    rejection_marks: list[tuple[int, int, tuple[int, int, int]]],
) -> np.ndarray:
    """
    Prolong unpaired limb endpoints along their stored terminal tangents.
    """
    for index in unconnected_indices:
        endpoint = active_endpoints[index]
        branch_pixels = list(endpoint['branch_pixels'])
        if not branch_pixels:
            continue
        endpoint_y, endpoint_x = branch_pixels[0]
        start_xy = (
            int(endpoint_x),
            int(endpoint_y),
        )
        stop_mask = (result_u8 > 0) | synthetic_stop_mask
        extension_mask, target_xy = prolong_endpoint(
            branch_pixels=branch_pixels,
            allowed_domain=domain_mask,
            stop_mask=stop_mask,
            thickness=bridge_thickness,
        )

        if target_xy is None or not np.any(extension_mask):
            continue

        if np.any(extension_mask & forbidden_mask):
            rejection_marks.append((
                int(endpoint_x),
                int(endpoint_y),
                rejection_color,
            ))
            continue

        extension_u8 = extension_mask.astype(np.uint8) * 255
        result_u8 = cv2.bitwise_or(
            result_u8,
            extension_u8,
        )
        accepted_extension_mask |= extension_mask
        accepted_segments.append(CannyAcceptedSegmentDebug(
            start_xy=start_xy,
            end_xy=target_xy,
            color=debug_color,
        ))
        if (
            0 <= start_xy[0] < accepted_endpoint_mask.shape[1]
            and 0 <= start_xy[1] < accepted_endpoint_mask.shape[0]
        ):
            accepted_endpoint_mask[start_xy[1], start_xy[0]] = 255

    return result_u8


def connect_limb_edge_endpoints(
    edges: np.ndarray,
    *,
    limb_skeleton_mask: np.ndarray,
    allowed_domain: np.ndarray,
    anatomical_segments: list[tuple[np.ndarray, np.ndarray]],
    min_contour_parallelism: float,
    ignore_intersection_radius: int = 3,
    intersection_blocker_dilation_radius: int = 2,
    min_allowed_fraction: float = 0.90,
    bridge_thickness: int = 1,
    synthetic_stop_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, CannyPairExtensionDebug]:
    """
    Connect compatible terminal Canny endpoints with direct edge segments.

    The function expects a one-pixel Canny skeleton. Every degree-1 endpoint is
    projected onto the ordered anatomical segment chain. Pairing traverses the
    endpoints from proximal to distal, starting from the second projected
    endpoint. For each current endpoint, eligible still-unpaired endpoints at or
    slightly behind its anatomical-chain position are tested from nearest to
    farthest in image space.

    The small forward chain-position tolerance compensates for local ordering
    ambiguities introduced when image endpoints are projected onto a piecewise
    anatomical chain, especially near joints and chain endpoints. Setting the
    tolerance to zero restores strict proximal-to-distal ordering.

    Endpoints belonging to one-step terminal branches are discarded before
    ordering. This suppresses the most obvious skeletonization spurs without
    dropping short but potentially meaningful branches created near Canny
    bifurcations.

    An endpoint is consumed only when a pair is accepted. Endpoints that fail to
    find a partner remain available and may be selected as proximal candidates by
    later endpoints. This creates a progressively established pairing frontier
    and avoids unrestricted searches into the unprocessed distal portion of the
    chain.

    Endpoint pairs are accepted only when the bridge segment and both endpoint
    terminal tangents are directionally coherent with the local equidistant
    contour induced by the anatomical skeleton. Existing one-pixel barriers and
    the anatomical skeleton are dilated only for the crossing test, so geometric
    intersections are rejected even when the rasterized segment misses the exact
    skeleton pixel.

    Parameters
    ----------
    edges : np.ndarray
        Two-dimensional one-pixel Canny edge skeleton.
    limb_skeleton_mask : np.ndarray
        Binary anatomical limb skeleton used for chain ordering, intersection
        rejection, and construction of the synthetic equidistant contour.
    allowed_domain : np.ndarray
        Binary domain inside which accepted bridge pixels may be drawn.
    anatomical_segments : list of tuple of np.ndarray
        Ordered proximal-to-distal anatomical chain segments. Projections may
        extend before the proximal endpoint of the first segment and beyond the
        distal endpoint of the last segment.
    min_contour_parallelism : float
        Minimum absolute directional alignment between the local synthetic
        equidistant contour and each direction that must follow it: the candidate
        bridge segment and the terminal tangent of each endpoint. Values range
        from zero for perpendicular directions to one for parallel directions.
    ignore_intersection_radius : int, default=3
        Radius around candidate endpoints excluded from bridge-intersection tests.
    intersection_blocker_dilation_radius : int, default=2
        Radius used to dilate existing Canny edges and the anatomical skeleton
        during crossing detection.
    min_allowed_fraction : float, default=0.90
        Minimum fraction of candidate-segment pixels that must lie inside
        ``allowed_domain``.
    bridge_thickness : int, default=1
        Thickness of accepted bridge segments in pixels.
    synthetic_stop_mask : np.ndarray or None, optional
        Optional synthetic barrier mask that can stop natural endpoint
        prolongation, such as proximal and distal closure barriers.

    Returns
    -------
    tuple of np.ndarray and CannyPairExtensionDebug
        The completed binary edge mask and the diagnostic information collected
        during endpoint pairing and natural endpoint prolongation.
    """
    edge_mask = np.asarray(edges).astype(bool)
    skeleton_mask = np.asarray(limb_skeleton_mask).astype(bool)
    domain_mask = np.asarray(allowed_domain).astype(bool)

    if edge_mask.ndim != 2:
        raise ValueError(
            f'edges must be two-dimensional, got {edge_mask.shape!r}.'
        )

    if not (
        edge_mask.shape
        == skeleton_mask.shape
        == domain_mask.shape
    ):
        raise ValueError(
            'edges, limb_skeleton_mask, and allowed_domain must have the same '
            f'shape, got {edge_mask.shape!r}, {skeleton_mask.shape!r}, '
            f'{domain_mask.shape!r}.'
        )
    if (
        not np.any(edge_mask)
        or not np.any(skeleton_mask)
    ):
        return edge_mask.copy(), CannyPairExtensionDebug(
            endpoint_marks=[],
            accepted_segments=[],
            accepted_extension_mask=np.zeros_like(edge_mask, dtype=bool),
            accepted_endpoint_mask=np.zeros_like(edge_mask, dtype=np.uint8),
            rejection_marks=[],
        )

    height, width = edge_mask.shape
    ignore_intersection_radius = max(0, int(ignore_intersection_radius))
    intersection_blocker_dilation_radius = max(
        0,
        int(intersection_blocker_dilation_radius),
    )

    debug_color_short_terminal_branch = (0, 0, 255)
    debug_color_candidate = (255, 0, 255)
    debug_color_raw_endpoint = (255, 255, 255)
    debug_color_pair_intersection = (255, 0, 0)
    debug_color_pair_domain = (0, 255, 0)
    debug_color_pair_parallelism = (0, 215, 255)
    debug_color_projection_skeleton = (180, 0, 255)
    debug_color_accepted_pair = (0, 128, 255)
    debug_color_accepted_prolongation = (255, 128, 0)

    def _degree(
        y: int,
        x: int,
        mask: np.ndarray,
    ) -> int:
        return len(neighbors8(y, x, shape=mask.shape, mask=mask))

    def _trace_terminal_branch(
        endpoint_y: int,
        endpoint_x: int,
        component_mask: np.ndarray,
    ) -> tuple[float, list[tuple[int, int]]]:
        previous: Optional[tuple[int, int]] = None
        current = (endpoint_y, endpoint_x)
        length = 0.0
        pixels = [current]

        while True:
            current_y, current_x = current
            neighbors = [
                neighbor
                for neighbor in neighbors8(
                    current_y,
                    current_x,
                    shape=component_mask.shape,
                    mask=component_mask,
                )
                if neighbor != previous
            ]

            if not neighbors:
                break

            if previous is not None and len(neighbors) != 1:
                break

            next_pixel = neighbors[0]
            next_y, next_x = next_pixel
            length += math.hypot(
                float(next_x - current_x),
                float(next_y - current_y),
            )
            pixels.append(next_pixel)

            next_degree = _degree(
                next_y,
                next_x,
                component_mask,
            )
            if next_degree != 2:
                break

            previous = current
            current = next_pixel

        return length, pixels

    def _valid_anatomical_segments() -> list[np.ndarray]:
        result: list[np.ndarray] = []
        chain_offset = 0.0

        for start, end in anatomical_segments:
            start_xy = np.asarray(start, dtype=np.float32).reshape(-1)
            end_xy = np.asarray(end, dtype=np.float32).reshape(-1)

            if start_xy.size < 2 or end_xy.size < 2:
                continue

            if not (
                np.all(np.isfinite(start_xy[:2]))
                and np.all(np.isfinite(end_xy[:2]))
            ):
                continue

            vector = end_xy[:2] - start_xy[:2]
            length = float(np.linalg.norm(vector))

            if length <= 1e-6:
                continue

            result.append(np.asarray(
                [
                    float(start_xy[0]),
                    float(start_xy[1]),
                    float(vector[0]),
                    float(vector[1]),
                    length,
                    chain_offset,
                ],
                dtype=np.float32,
            ))
            chain_offset += length

        return result

    anatomical_vectors = _valid_anatomical_segments()

    distance_to_skeleton = cv2.distanceTransform(
        (~skeleton_mask).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )

    def _sample_distance_to_skeleton(
        point_xy: np.ndarray,
    ) -> Optional[float]:
        x = int(round(float(point_xy[0])))
        y = int(round(float(point_xy[1])))

        if not (0 <= x < width and 0 <= y < height):
            return None

        return float(distance_to_skeleton[y, x])

    def _equidistant_contour_direction(
        start_xy: np.ndarray,
        end_xy: np.ndarray,
    ) -> Optional[np.ndarray]:
        midpoint = 0.5 * (start_xy + end_xy)
        x = int(round(float(midpoint[0])))
        y = int(round(float(midpoint[1])))

        if not (1 <= x < width - 1 and 1 <= y < height - 1):
            return None

        center_distance = float(distance_to_skeleton[y, x])
        if center_distance <= 1e-6:
            return None

        gradient_x = float(distance_to_skeleton[y, x + 1]) - float(
            distance_to_skeleton[y, x - 1]
        )
        gradient_y = float(distance_to_skeleton[y + 1, x]) - float(
            distance_to_skeleton[y - 1, x]
        )
        gradient_length = math.hypot(gradient_x, gradient_y)
        if gradient_length <= 1e-6:
            return None

        tangent = np.asarray(
            [
                -gradient_y / gradient_length,
                gradient_x / gradient_length,
            ],
            dtype=np.float32,
        )
        return tangent

    def _direction_alignment(
        first_xy: np.ndarray,
        second_xy: np.ndarray,
    ) -> Optional[float]:
        first = np.asarray(first_xy, dtype=np.float32).reshape(-1)
        second = np.asarray(second_xy, dtype=np.float32).reshape(-1)

        if first.size < 2 or second.size < 2:
            return None

        first = first[:2]
        second = second[:2]
        first_length = float(np.linalg.norm(first))
        second_length = float(np.linalg.norm(second))

        if first_length <= 1e-6 or second_length <= 1e-6:
            return None

        return abs(float(np.dot(
            first / first_length,
            second / second_length,
        )))

    def _anatomical_chain_position(
        point_xy: np.ndarray,
    ) -> Optional[float]:
        if not anatomical_vectors:
            return None

        best_distance_sq: Optional[float] = None
        best_position: Optional[float] = None

        last_segment_index = len(anatomical_vectors) - 1

        for segment_index, segment in enumerate(anatomical_vectors):
            seg_start = segment[:2]
            seg_vector = segment[2:4]
            seg_length = float(segment[4])
            chain_offset = float(segment[5])
            direction = seg_vector / seg_length
            t = float(np.dot(point_xy - seg_start, direction))

            # For ordering, extend the anatomical chain beyond the proximal and
            # distal endpoints instead of collapsing out-of-range projections.
            if segment_index == 0:
                lower_t = t
            else:
                lower_t = max(0.0, t)

            if segment_index == last_segment_index:
                projected_t = lower_t
            else:
                projected_t = min(seg_length, lower_t)

            projected = seg_start + direction * projected_t
            delta = point_xy - projected
            distance_sq = float(np.dot(delta, delta))

            if (
                best_distance_sq is None
                or distance_sq < best_distance_sq
            ):
                best_distance_sq = distance_sq
                best_position = chain_offset + projected_t

        return best_position

    def _line_mask(
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
    ) -> np.ndarray:
        mask = np.zeros_like(edge_mask, dtype=np.uint8)
        cv2.line(
            mask,
            start_xy,
            end_xy,
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        return mask > 0

    def _endpoint_ignore_mask(
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
    ) -> np.ndarray:
        ignore = np.zeros_like(edge_mask, dtype=np.uint8)

        if ignore_intersection_radius <= 0:
            return ignore > 0

        for xy in (start_xy, end_xy):
            cv2.circle(
                ignore,
                xy,
                ignore_intersection_radius,
                255,
                thickness=-1,
                lineType=cv2.LINE_8,
            )

        return ignore > 0

    component_count, component_labels = cv2.connectedComponents(
        edge_mask.astype(np.uint8),
        connectivity=8,
    )

    endpoint_marks: list[CannyEndpointDebugMark] = []
    active_endpoints: list[dict[str, object]] = []

    for component_id in range(1, component_count):
        component_mask = component_labels == component_id
        component_ys, component_xs = np.where(component_mask)

        for y, x in zip(component_ys.tolist(), component_xs.tolist()):
            if _degree(y, x, component_mask) != 1:
                continue

            endpoint_marks.append(CannyEndpointDebugMark(
                x=int(x),
                y=int(y),
                color=debug_color_raw_endpoint,
            ))

            branch_length, branch_pixels = _trace_terminal_branch(
                int(y),
                int(x),
                component_mask,
            )

            # Reject only one-step terminal skeleton branches. Longer short
            # branches may occur near real bifurcations and still provide valid
            # endpoint geometry for pairing or natural prolongation.
            if (
                len(branch_pixels) <= 2
                and branch_length <= math.sqrt(2.0) + 1e-6
            ):
                endpoint_marks.append(CannyEndpointDebugMark(
                    x=int(x),
                    y=int(y),
                    color=debug_color_short_terminal_branch,
                ))
                continue

            point_xy = np.asarray(
                [
                    float(x),
                    float(y),
                ],
                dtype=np.float32,
            )
            skeleton_distance = _sample_distance_to_skeleton(point_xy)
            chain_position = _anatomical_chain_position(point_xy)
            terminal_direction = _terminal_branch_direction(branch_pixels)

            if (
                skeleton_distance is None
                or chain_position is None
                or terminal_direction is None
            ):
                continue

            endpoint_marks.append(CannyEndpointDebugMark(
                x=int(x),
                y=int(y),
                color=debug_color_candidate,
            ))
            active_endpoints.append({
                'x': int(x),
                'y': int(y),
                'xy': point_xy,
                'component_id': int(component_id),
                'branch_length': float(branch_length),
                'branch_pixels': tuple(branch_pixels),
                'terminal_direction': terminal_direction,
                'skeleton_distance': float(skeleton_distance),
                'chain_position': float(chain_position),
            })

    if not active_endpoints:
        return edge_mask.copy(), CannyPairExtensionDebug(
            endpoint_marks=endpoint_marks,
            accepted_segments=[],
            accepted_extension_mask=np.zeros_like(edge_mask, dtype=bool),
            accepted_endpoint_mask=np.zeros_like(edge_mask, dtype=np.uint8),
            rejection_marks=[],
        )

    # Keep a stable proximal-to-distal traversal order separate from the set of
    # endpoints that are still available for pairing.
    #
    # ``ordered_indices`` determines when each endpoint becomes the current
    # endpoint. ``available_indices`` contains endpoints that have not already
    # been consumed by an accepted pair. An endpoint that fails to find a partner
    # remains available and may therefore be selected by a later, more distal
    # endpoint.
    ordered_indices = sorted(
        range(len(active_endpoints)),
        key=lambda index: (
            float(active_endpoints[index]['chain_position']),
            float(active_endpoints[index]['skeleton_distance']),
        ),
    )
    available_indices = set(ordered_indices)

    result_u8 = edge_mask.astype(np.uint8) * 255
    accepted_extension_mask = np.zeros_like(edge_mask, dtype=bool)
    accepted_endpoint_mask = np.zeros_like(edge_mask, dtype=np.uint8)
    accepted_segments: list[CannyAcceptedSegmentDebug] = []
    rejection_marks: list[tuple[int, int, tuple[int, int, int]]] = []
    unconnected_indices: list[int] = []

    # Allow a small forward tolerance in anatomical chain position.
    #
    # Projecting endpoints onto a piecewise anatomical chain may slightly invert
    # the order of geometrically adjacent endpoints, especially near joints and
    # chain endpoints. A small positive tolerance keeps those locally ambiguous
    # candidates eligible while the main traversal still proceeds from proximal
    # to distal. Set this value to zero for strict ordering.
    chain_backward_epsilon_px = 5.0

    # Start from the second endpoint in proximal-to-distal order. The first
    # endpoint has no strictly preceding endpoint to search against, although it
    # remains available as a candidate for all following endpoints.
    for traversal_position in range(1, len(ordered_indices)):
        current_index = ordered_indices[traversal_position]

        # The endpoint may already have been consumed as the partner of an earlier
        # current endpoint.
        if current_index not in available_indices:
            continue

        current = active_endpoints[current_index]
        current_xy = np.asarray(
            current['xy'],
            dtype=np.float32,
        )
        current_chain_position = float(current['chain_position'])

        # Pairing proceeds from proximal to distal, but each current endpoint
        # searches in the opposite direction, toward endpoints that are already at
        # or behind its anatomical-chain position.
        #
        # The small positive epsilon also admits endpoints whose projected chain
        # position is slightly more distal because of local projection ambiguity.
        # Such candidates may include an endpoint appearing just after the current
        # endpoint in ``ordered_indices``; this is intentional and remains bounded
        # by ``chain_backward_epsilon_px``.
        candidate_indices = [
            index
            for index in available_indices
            if (
                index != current_index
                and float(active_endpoints[index]['chain_position'])
                <= (
                    current_chain_position
                    + chain_backward_epsilon_px
                )
            )
        ]

        # Test eligible endpoints from nearest to farthest in image space. The
        # first candidate that satisfies every geometric and topological
        # constraint becomes the selected partner.
        candidate_indices.sort(
            key=lambda index: float(np.linalg.norm(
                np.asarray(
                    active_endpoints[index]['xy'],
                    dtype=np.float32,
                )
                - current_xy
            )),
        )

        connected_index: Optional[int] = None
        connected_mask: Optional[np.ndarray] = None
        connected_start_xy: Optional[tuple[int, int]] = None
        connected_end_xy: Optional[tuple[int, int]] = None

        blocker_mask = (result_u8 > 0) | skeleton_mask
        if intersection_blocker_dilation_radius > 0:
            kernel_size = (
                2 * intersection_blocker_dilation_radius + 1
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            blocker_mask = cv2.dilate(
                blocker_mask.astype(np.uint8),
                kernel,
                iterations=1,
            ) > 0

        for candidate_index in candidate_indices:
            candidate = active_endpoints[candidate_index]
            candidate_xy = np.asarray(
                candidate['xy'],
                dtype=np.float32,
            )

            segment = current_xy - candidate_xy
            segment_length = float(np.linalg.norm(segment))

            if segment_length < 1.0:
                continue

            contour_direction = _equidistant_contour_direction(
                candidate_xy,
                current_xy,
            )
            bridge_alignment = None
            candidate_tangent_alignment = None
            current_tangent_alignment = None
            if contour_direction is not None:
                bridge_alignment = _direction_alignment(
                    current_xy - candidate_xy,
                    contour_direction,
                )
                candidate_tangent_alignment = _direction_alignment(
                    np.asarray(
                        candidate['terminal_direction'],
                        dtype=np.float32,
                    ),
                    contour_direction,
                )
                current_tangent_alignment = _direction_alignment(
                    np.asarray(
                        current['terminal_direction'],
                        dtype=np.float32,
                    ),
                    contour_direction,
                )
            if (
                (
                    bridge_alignment is not None
                    and bridge_alignment < float(min_contour_parallelism)
                )
                or (
                    candidate_tangent_alignment is not None
                    and candidate_tangent_alignment
                    < float(min_contour_parallelism)
                )
                or (
                    current_tangent_alignment is not None
                    and current_tangent_alignment
                    < float(min_contour_parallelism)
                )
            ):
                rejection_marks.append((
                    int(current['x']),
                    int(current['y']),
                    debug_color_pair_parallelism,
                ))
                continue

            start_xy = (
                int(candidate['x']),
                int(candidate['y']),
            )
            end_xy = (
                int(current['x']),
                int(current['y']),
            )
            segment_mask = _line_mask(
                start_xy,
                end_xy,
            )

            if not np.any(segment_mask):
                continue

            allowed_fraction = float(np.mean(
                domain_mask[segment_mask]
            ))

            if allowed_fraction < float(min_allowed_fraction):
                rejection_marks.append((
                    int(current['x']),
                    int(current['y']),
                    debug_color_pair_domain,
                ))
                continue

            ignore_mask = _endpoint_ignore_mask(
                start_xy,
                end_xy,
            )

            intersects_existing_edge = bool(np.any(
                segment_mask
                & blocker_mask
                & ~ignore_mask
            ))

            if intersects_existing_edge:
                rejection_marks.append((
                    int(current['x']),
                    int(current['y']),
                    debug_color_pair_intersection,
                ))
                continue

            extension_u8 = np.zeros_like(result_u8)
            cv2.line(
                extension_u8,
                start_xy,
                end_xy,
                255,
                thickness=bridge_thickness,
                lineType=cv2.LINE_8,
            )
            extension_u8[~domain_mask] = 0

            if not np.any(extension_u8):
                rejection_marks.append((
                    int(current['x']),
                    int(current['y']),
                    debug_color_pair_domain,
                ))
                continue

            connected_index = candidate_index
            connected_mask = extension_u8 > 0
            connected_start_xy = start_xy
            connected_end_xy = end_xy
            break

        if (
            connected_index is None
            or connected_mask is None
        ):
            # Do not remove an unmatched endpoint. It remains available as a
            # possible proximal partner for a later endpoint in the traversal.
            continue

        # An accepted pair consumes both endpoints exactly once.
        available_indices.discard(current_index)
        available_indices.discard(connected_index)

        result_u8 = cv2.bitwise_or(
            result_u8,
            connected_mask.astype(np.uint8) * 255,
        )

        accepted_extension_mask |= connected_mask
        accepted_segments.append(CannyAcceptedSegmentDebug(
            start_xy=connected_start_xy or (
                int(active_endpoints[connected_index]['x']),
                int(active_endpoints[connected_index]['y']),
            ),
            end_xy=connected_end_xy or (
                int(current['x']),
                int(current['y']),
            ),
            color=debug_color_accepted_pair,
        ))

        for endpoint in (
            current,
            active_endpoints[connected_index],
        ):
            x = int(endpoint['x'])
            y = int(endpoint['y'])
            if (
                0 <= x < accepted_endpoint_mask.shape[1]
                and 0 <= y < accepted_endpoint_mask.shape[0]
            ):
                accepted_endpoint_mask[y, x] = 255

    # Only endpoints that remain available after the complete traversal are truly
    # unconnected. An endpoint that initially failed as a current endpoint may
    # still have been consumed later as the proximal partner of another endpoint.
    unconnected_indices = sorted(
        available_indices,
        key=lambda index: (
            float(active_endpoints[index]['chain_position']),
            float(active_endpoints[index]['skeleton_distance']),
        ),
    )

    synthetic_stop = (
        None
        if synthetic_stop_mask is None
        else np.asarray(synthetic_stop_mask).astype(bool)
    )
    resolved_synthetic_stop_mask = (
        np.zeros_like(edge_mask, dtype=bool)
        if (
            synthetic_stop is None
            or synthetic_stop.shape != edge_mask.shape
        )
        else synthetic_stop
    )

    result_u8 = _prolong_unconnected_limb_endpoints(
        active_endpoints=active_endpoints,
        unconnected_indices=unconnected_indices,
        result_u8=result_u8,
        domain_mask=domain_mask,
        forbidden_mask=skeleton_mask,
        synthetic_stop_mask=resolved_synthetic_stop_mask,
        bridge_thickness=bridge_thickness,
        debug_color=debug_color_accepted_prolongation,
        rejection_color=debug_color_projection_skeleton,
        accepted_extension_mask=accepted_extension_mask,
        accepted_endpoint_mask=accepted_endpoint_mask,
        accepted_segments=accepted_segments,
        rejection_marks=rejection_marks,
    )

    return result_u8 > 0, CannyPairExtensionDebug(
        endpoint_marks=endpoint_marks,
        accepted_segments=accepted_segments,
        accepted_extension_mask=accepted_extension_mask,
        accepted_endpoint_mask=accepted_endpoint_mask,
        rejection_marks=rejection_marks,
    )


# --------------------------------------------------------------------------
# Barrier construction
# --------------------------------------------------------------------------


def build_canny_partition_barriers(
    *,
    visual_barrier_mask: np.ndarray,
    semantic_barrier_mask: Optional[np.ndarray] = None,
    synthetic_barrier_mask: Optional[np.ndarray] = None,
    close_radius: int = LIMB_CANNY_EDGE_CLOSE_RADIUS,
    dilation_radius: int = LIMB_CANNY_EDGE_DILATE_RADIUS,
) -> CannyBarrierSet:
    """
    Build strengthened barriers for connected-region partitioning.

    Visual, semantic, and synthetic barriers are normalized to boolean masks,
    processed independently, and retained separately in the returned result.
    Their union forms the final partition mask.

    Morphological closing is applied only to visual barriers. It may reconnect
    small breaks in image-derived contours, but it must not alter explicit
    semantic or synthetic topology.

    Dilation is applied independently to every barrier category. Processing the
    categories separately preserves their provenance while ensuring that
    one-pixel diagonal leaks cannot bypass a barrier during 8-connected region
    labeling.

    Parameters
    ----------
    visual_barrier_mask : np.ndarray
        Two-dimensional visual edge mask. Non-zero pixels are treated as
        barriers.

    semantic_barrier_mask : np.ndarray or None, optional
        Two-dimensional mask containing barriers derived from semantic-region
        boundaries. ``None`` is treated as an empty mask.

    synthetic_barrier_mask : np.ndarray or None, optional
        Two-dimensional mask containing explicit artificial barriers, such as
        proximal or distal closure lines. ``None`` is treated as an empty mask.

    close_radius : int, default=0
        Radius of the elliptical morphological-closing kernel applied only to
        visual barriers. Zero disables closing.

    dilation_radius : int, default=1
        Radius of the elliptical dilation kernel applied independently to all
        barrier categories. Zero disables dilation.

    Returns
    -------
    CannyBarrierSet
        Strengthened visual, semantic, and synthetic masks together with their
        final union in ``partition_mask``.

    Raises
    ------
    ValueError
        If a provided mask is not two-dimensional, masks have different shapes,
        or a morphology radius is negative.
    """
    visual = np.asarray(visual_barrier_mask)

    if visual.ndim != 2:
        raise ValueError(
            'visual_barrier_mask must be two-dimensional, '
            f'got shape {visual.shape!r}.'
        )

    close_radius = int(close_radius)
    dilation_radius = int(dilation_radius)

    if close_radius < 0:
        raise ValueError(
            f'close_radius must be >= 0, got {close_radius!r}.'
        )

    if dilation_radius < 0:
        raise ValueError(
            f'dilation_radius must be >= 0, got {dilation_radius!r}.'
        )

    shape = visual.shape

    def normalize_optional_mask(
        mask: Optional[np.ndarray],
        *,
        name: str,
    ) -> np.ndarray:
        """
        Normalize an optional barrier mask to a shape-aligned boolean array.
        """
        if mask is None:
            return np.zeros(shape, dtype=bool)

        normalized = np.asarray(mask)

        if normalized.ndim != 2:
            raise ValueError(
                f'{name} must be two-dimensional, '
                f'got shape {normalized.shape!r}.'
            )

        if normalized.shape != shape:
            raise ValueError(
                f'{name} must match visual_barrier_mask shape {shape!r}, '
                f'got {normalized.shape!r}.'
            )

        return normalized.astype(bool)

    visual = visual.astype(bool)
    semantic = normalize_optional_mask(
        semantic_barrier_mask,
        name='semantic_barrier_mask',
    )
    synthetic = normalize_optional_mask(
        synthetic_barrier_mask,
        name='synthetic_barrier_mask',
    )

    if close_radius > 0:
        kernel_size = 2 * int(close_radius) + 1
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        visual = cv2.morphologyEx(
            visual.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            close_kernel,
        ) > 0

    if dilation_radius > 0:
        kernel_size = 2 * int(dilation_radius) + 1
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        visual = cv2.dilate(
            visual.astype(np.uint8) * 255,
            dilation_kernel,
            iterations=1,
        ) > 0
        semantic = cv2.dilate(
            semantic.astype(np.uint8) * 255,
            dilation_kernel,
            iterations=1,
        ) > 0
        synthetic = cv2.dilate(
            synthetic.astype(np.uint8) * 255,
            dilation_kernel,
            iterations=1,
        ) > 0

    return CannyBarrierSet(
        visual_mask=visual,
        semantic_mask=semantic,
        synthetic_mask=synthetic,
        partition_mask=visual | semantic | synthetic,
    )


# --------------------------------------------------------------------------
# Region topology
# --------------------------------------------------------------------------


def build_region_mosaic(
    *,
    workspace_mask: np.ndarray,
    barrier_mask: np.ndarray,
) -> RegionMosaic:
    """
    Partition a binary workspace into 8-connected free-space regions.

    Barriers remove pixels from the traversable workspace. The remaining
    free-space pixels are labeled with OpenCV connected-component identifiers.
    Label zero is reserved for barriers and pixels outside the workspace.

    The function also identifies regions touching the internal boundary of the
    workspace. These labels describe regions connected to the exterior side of
    the partition domain and can later be rejected by callers that need enclosed
    regions, such as the limb-core reconstruction.

    This function does not decide which regions are semantically valid. It only
    describes the topology produced by the supplied workspace and barriers.

    Parameters
    ----------
    workspace_mask : np.ndarray
        Two-dimensional binary mask defining the complete domain to partition.
        Non-zero pixels belong to the workspace.

    barrier_mask : np.ndarray
        Two-dimensional binary mask defining non-traversable partition
        barriers. Barrier pixels outside the workspace have no effect.

    Returns
    -------
    RegionMosaic
        Region-label map, normalized workspace and barrier masks, number of
        non-zero connected regions, and labels touching the internal workspace
        boundary.

    Raises
    ------
    ValueError
        If either input is not two-dimensional or if their shapes differ.

    Notes
    -----
    Connectivity is eight-neighbor connectivity. Consequently, one-pixel
    barriers should normally be strengthened before this function is called;
    otherwise free-space regions may remain connected diagonally around them.
    """
    workspace = np.asarray(workspace_mask)
    barriers = np.asarray(barrier_mask)

    if workspace.ndim != 2:
        raise ValueError(
            'workspace_mask must be two-dimensional, '
            f'got shape {workspace.shape!r}.'
        )

    if barriers.ndim != 2:
        raise ValueError(
            'barrier_mask must be two-dimensional, '
            f'got shape {barriers.shape!r}.'
        )

    if workspace.shape != barriers.shape:
        raise ValueError(
            'workspace_mask and barrier_mask must have the same shape, got '
            f'{workspace.shape!r} and {barriers.shape!r}.'
        )

    workspace = workspace.astype(bool)
    barriers = barriers.astype(bool) & workspace

    if not np.any(workspace):
        return RegionMosaic(
            labels=np.zeros(workspace.shape, dtype=np.int32),
            workspace_mask=workspace,
            barrier_mask=barriers,
            boundary_labels=frozenset(),
            region_count=0,
        )

    free_space = workspace & ~barriers

    component_count, labels = cv2.connectedComponents(
        free_space.astype(np.uint8),
        connectivity=8,
    )

    labels = labels.astype(np.int32, copy=False)

    # Region labels use 8-neighbor connectivity, so workspace-boundary detection
    # must use the same neighborhood. A cross/ellipse kernel would miss pixels
    # that touch the workspace exterior only diagonally.
    boundary_kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_workspace = cv2.erode(
        workspace.astype(np.uint8),
        boundary_kernel,
        iterations=1,
    ) > 0

    workspace_boundary = workspace & ~eroded_workspace

    boundary_labels = frozenset(
        int(label)
        for label in np.unique(labels[workspace_boundary])
        if int(label) > 0
    )

    return RegionMosaic(
        labels=labels,
        workspace_mask=workspace,
        barrier_mask=barriers,
        boundary_labels=boundary_labels,
        region_count=max(0, int(component_count) - 1),
    )


def region_labels_intersecting_mask(
    mosaic: RegionMosaic,
    mask: np.ndarray,
) -> frozenset[int]:
    """
    Return the region labels intersecting a binary query mask.

    The function examines the connected-region label map stored in ``mosaic``
    and returns every non-zero region label having at least one pixel in common
    with ``mask``.

    Label zero is never returned because it denotes pixels outside the
    workspace or pixels occupied by partition barriers. The function performs
    no semantic interpretation of the intersected regions: the query mask may
    represent an anchor, a prior, a set of seed points rasterized as a mask, an
    exclusion domain, or any other caller-defined spatial condition.

    Parameters
    ----------
    mosaic : RegionMosaic
        Connected free-space partition to query.

        ``mosaic.labels`` must be a two-dimensional integer label map. Positive
        values identify connected free-space regions, while zero identifies
        barriers and pixels outside the workspace.

    mask : np.ndarray
        Two-dimensional binary query mask aligned with ``mosaic.labels``.
        Non-zero pixels participate in the intersection test.

    Returns
    -------
    frozenset[int]
        Immutable set containing every positive region label intersecting at
        least one non-zero pixel of ``mask``.

        An empty set is returned when the query mask is empty or when it
        intersects no labeled free-space region.

    Raises
    ------
    ValueError
        If ``mosaic.labels`` is not two-dimensional, if ``mask`` is not
        two-dimensional, or if their shapes differ.

    Notes
    -----
    The function tests direct pixel intersection only. It does not dilate the
    query mask, compute distances, cross barriers, or infer adjacency between
    regions.

    Callers requiring tolerance around an anchor should construct the desired
    query mask explicitly before invoking this function. Any such tolerance
    should normally respect the free-space topology rather than blindly crossing
    partition barriers.
    """
    labels = np.asarray(mosaic.labels)
    query_mask = np.asarray(mask)

    if labels.ndim != 2:
        raise ValueError(
            'mosaic.labels must be two-dimensional, '
            f'got shape {labels.shape!r}.'
        )

    if query_mask.ndim != 2:
        raise ValueError(
            'mask must be two-dimensional, '
            f'got shape {query_mask.shape!r}.'
        )

    if query_mask.shape != labels.shape:
        raise ValueError(
            'mask must match mosaic.labels shape, got '
            f'{query_mask.shape!r} and {labels.shape!r}.'
        )

    query_mask = query_mask.astype(bool)

    if not np.any(query_mask):
        return frozenset()

    intersected_labels = np.unique(
        labels[query_mask]
    )

    return frozenset(
        int(label)
        for label in intersected_labels
        if int(label) > 0
    )


def region_mask_from_labels(
    mosaic: RegionMosaic,
    labels: Iterable[int],
) -> np.ndarray:
    """
    Build a binary mask from selected region labels.

    The returned mask contains every pixel whose connected-region label appears
    in ``labels``. Pixels outside the workspace, partition barriers, and
    unselected regions remain false.

    This helper performs only label-to-mask conversion. It does not modify the
    region topology, restore barrier pixels, apply semantic support, or validate
    the resulting area.

    Parameters
    ----------
    mosaic : RegionMosaic
        Connected free-space partition containing the integer label map from
        which the output mask is constructed.

    labels : iterable of int
        Positive region labels to include.

        Duplicate labels are accepted and have no additional effect. The
        iterable may be empty, in which case an empty boolean mask is returned.

    Returns
    -------
    np.ndarray
        Two-dimensional boolean mask with the same shape as ``mosaic.labels``.
        Pixels belonging to one of the requested regions are true.

    Raises
    ------
    ValueError
        If ``mosaic.labels`` is not two-dimensional, if a requested label is not
        an integer, or if it lies outside the valid region range.

    Notes
    -----
    Label zero cannot be selected because it represents both partition barriers
    and pixels outside the workspace rather than a connected free-space region.

    Region labels produced by ``cv2.connectedComponents`` are contiguous from
    one through ``mosaic.region_count``. Rejecting labels outside that range
    helps detect accidental use of labels obtained from a different mosaic.
    """
    mosaic_labels = np.asarray(mosaic.labels)

    if mosaic_labels.ndim != 2:
        raise ValueError(
            'mosaic.labels must be two-dimensional, '
            f'got shape {mosaic_labels.shape!r}.'
        )

    selected_labels: set[int] = set()

    for label in labels:
        if not isinstance(label, (int, np.integer)):
            raise ValueError(
                'Region labels must be integers, '
                f'got {label!r}.'
            )

        selected_labels.add(int(label))

    selected_labels = frozenset(selected_labels)

    invalid_labels = sorted(
        label
        for label in selected_labels
        if (
            label < 1
            or label > int(mosaic.region_count)
        )
    )

    if invalid_labels:
        raise ValueError(
            'Region labels must be in the inclusive range '
            f'[1, {int(mosaic.region_count)}], got {invalid_labels!r}.'
        )

    if not selected_labels:
        return np.zeros(
            mosaic_labels.shape,
            dtype=bool,
        )

    return np.isin(
        mosaic_labels,
        np.asarray(
            sorted(selected_labels),
            dtype=mosaic_labels.dtype,
        ),
    )


def internal_region_labels(
    mosaic: RegionMosaic,
) -> frozenset[int]:
    """
    Return labels of regions that do not touch the workspace boundary.

    Internal regions are connected free-space components whose labels are not
    present in ``mosaic.boundary_labels``. These regions are topologically
    enclosed inside the workspace by partition barriers or by combinations of
    barriers and workspace geometry.

    The function does not assume that an internal region belongs to any
    particular semantic object. It only classifies regions according to their
    connectivity with the internal boundary of the workspace.

    Parameters
    ----------
    mosaic : RegionMosaic
        Connected free-space partition whose regions should be classified.

        ``mosaic.region_count`` defines the valid positive label range and
        ``mosaic.boundary_labels`` identifies regions intersecting the workspace
        boundary computed by ``build_region_mosaic``.

    Returns
    -------
    frozenset[int]
        Immutable set containing every positive region label that does not touch
        the workspace boundary.

        An empty set is returned when the mosaic contains no regions or when
        every region touches the workspace boundary.

    Raises
    ------
    ValueError
        If ``mosaic.region_count`` is negative, if
        ``mosaic.boundary_labels`` contains a non-integer value, or if one of its
        labels lies outside the valid positive region range.

    Notes
    -----
    Being internal is a topological property, not a semantic validation.
    Internal regions may still correspond to texture loops, holes, unrelated
    objects, or artifacts. Callers remain responsible for applying anchors,
    positive-point coverage, semantic support, area constraints, or any other
    target-specific selection criteria.
    """
    region_count = int(mosaic.region_count)

    if region_count < 0:
        raise ValueError(
            'mosaic.region_count must be non-negative, '
            f'got {region_count!r}.'
        )

    boundary_label_set: set[int] = set()

    for label in mosaic.boundary_labels:
        if not isinstance(label, (int, np.integer)):
            raise ValueError(
                'mosaic.boundary_labels must contain integers, '
                f'got {label!r}.'
            )

        boundary_label_set.add(int(label))

    boundary_labels = frozenset(boundary_label_set)

    invalid_boundary_labels = sorted(
        label
        for label in boundary_labels
        if (
            label < 1
            or label > region_count
        )
    )

    if invalid_boundary_labels:
        raise ValueError(
            'mosaic.boundary_labels contains labels outside the valid '
            f'range [1, {region_count}], got '
            f'{invalid_boundary_labels!r}.'
        )

    return frozenset(
        label
        for label in range(1, region_count + 1)
        if label not in boundary_labels
    )
