from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ImageColor
from skimage.color import rgb2lab

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (CropModeSpec,
                                             expand_bbox_toward_ratio,
                                             expand_clip_bbox,
                                             parse_crop_mode,
                                             postprocess_mask,
                                             tight_alpha_bbox)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


@dataclass(frozen=True)
class Config:
    """
    Validated runtime configuration for ``ColorDrivenCrop``.

    All color-distance values are expressed in the CIE Lab space produced by
    :func:`skimage.color.rgb2lab`. ``strength`` is an exclusion strength, where
    ``1`` removes non-preserved pixels and ``0`` leaves alpha unchanged.
    """

    analysis_clusters: int
    num_dominant_colors: int
    mode: str
    crop_mode: Optional[CropModeSpec]
    color_policy: str
    color_scope: str
    tolerance: float
    feather: float
    strength: float
    colors: Optional[tuple[tuple[int, int, int], ...]]
    box_margin: float
    min_component_area: Any
    dilate_radius: int
    close_radius: int
    smoothing_radius: int


_DEFAULT_ANALYSIS_CLUSTERS = 6
_TILE_SIZE = 256
_TILE_OVERLAP = 0.5
_SAMPLE_STRIDE = 2
_KMEANS_ITERATIONS = 12


def _is_rgb_triplet(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(channel, (int, float)) for channel in value)
    )


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Read and validate the node's ``params`` configuration.

    Parameters
    ----------
    spec : dict
        Resolved node specification.

    node_id : str
        Node identifier used to produce contextual validation errors.

    Returns
    -------
    Config
        Validated configuration with defaults applied.

    Raises
    ------
    ValueError
        If a count, range, overlap, alpha value, or iteration limit is invalid.
    """
    params = spec.get('params', {})

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': invalid mode={mode!r}")

    crop_mode = (
        parse_crop_mode(params.get('crop_mode', 'trim'), node_id=node_id)
        if mode == 'default'
        else None
    )

    color_policy = str(params.get('color_policy', 'exclude'))
    if color_policy not in ('exclude', 'include'):
        raise ValueError(
            f"'{node_id}': invalid color_policy={color_policy!r}"
        )

    color_scope = str(params.get('color_scope', 'global'))
    if color_scope not in ('global', 'local'):
        raise ValueError(
            f"'{node_id}': invalid color_scope={color_scope!r}"
        )

    colors = _parse_colors(
        params.get('colors', None),
        node_id=node_id,
    )

    if colors is not None and 'num_dominant_colors' in params:
        raise ValueError(
            f"'{node_id}': colors and num_dominant_colors are mutually exclusive"
        )

    if colors is None:
        num_dominant_colors = int(params.get('num_dominant_colors', 1))
    else:
        num_dominant_colors = len(colors)

    analysis_clusters_value = params.get('analysis_clusters', None)
    if analysis_clusters_value is None:
        analysis_clusters = max(
            _DEFAULT_ANALYSIS_CLUSTERS,
            num_dominant_colors,
        )
    else:
        analysis_clusters = int(analysis_clusters_value)

    tolerance = float(params.get('tolerance', 12.0))
    feather = float(params.get('feather', 6.0))
    strength = float(params.get('strength', 1.0))
    box_margin = float(params.get('box_margin', 0.08))
    min_component_area = params.get('min_component_area', 0)
    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))

    if analysis_clusters < 1:
        raise ValueError(
            f"'{node_id}': analysis_clusters must be >= 1"
        )
    if colors is None and num_dominant_colors < 1:
        raise ValueError(
            f"'{node_id}': num_dominant_colors must be >= 1"
        )
    if colors is None and num_dominant_colors > analysis_clusters:
        raise ValueError(
            f"'{node_id}': num_dominant_colors must be <= analysis_clusters "
            'when colors is not provided'
        )
    if tolerance < 0.0:
        raise ValueError(
            f"'{node_id}': tolerance must be >= 0"
        )
    if feather < 0.0:
        raise ValueError(f"'{node_id}': feather must be >= 0")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"'{node_id}': strength must be in [0, 1]"
        )
    if box_margin < 0.0:
        raise ValueError(f"'{node_id}': box_margin must be >= 0")
    _validate_min_component_area(min_component_area, node_id=node_id)
    if dilate_radius < 0:
        raise ValueError(f"'{node_id}': dilate_radius must be >= 0")
    if close_radius < 0:
        raise ValueError(f"'{node_id}': close_radius must be >= 0")
    if smoothing_radius < 0:
        raise ValueError(f"'{node_id}': smoothing_radius must be >= 0")

    return Config(
        analysis_clusters=analysis_clusters,
        num_dominant_colors=num_dominant_colors,
        mode=mode,
        crop_mode=crop_mode,
        color_policy=color_policy,
        color_scope=color_scope,
        tolerance=tolerance,
        feather=feather,
        strength=strength,
        colors=colors,
        box_margin=box_margin,
        min_component_area=min_component_area,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        smoothing_radius=smoothing_radius,
    )


def _parse_color(value: Any, *, node_id: str) -> tuple[int, int, int]:
    """
    Parse one user-provided color into an RGB triplet.
    """
    if isinstance(value, str):
        try:
            rgb = ImageColor.getrgb(value)
        except ValueError as exc:
            raise ValueError(
                f"'{node_id}': invalid color {value!r}"
            ) from exc

        if len(rgb) == 4:
            rgb = rgb[:3]
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    if _is_rgb_triplet(value):
        rgb = tuple(int(channel) for channel in value)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(
                f"'{node_id}': RGB values must be in [0, 255]"
            )
        return rgb

    raise ValueError(
        f"'{node_id}': colors entries must be color names, hex "
        'strings, or RGB triplets'
    )


def _parse_colors(
    value: Any,
    *,
    node_id: str,
) -> Optional[tuple[tuple[int, int, int], ...]]:
    """
    Parse optional manual reference colors.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return (_parse_color(value, node_id=node_id),)

    if _is_rgb_triplet(value):
        return (_parse_color(value, node_id=node_id),)

    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(
            f"'{node_id}': colors must be a color or a non-empty "
            'list of colors'
        )

    return tuple(_parse_color(item, node_id=node_id) for item in value)


def _validate_min_component_area(value: Any, *, node_id: str) -> None:
    """
    Validate the component-area cleanup threshold.
    """
    if value is None:
        return

    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"'{node_id}': min_component_area must be >= 0")
        return

    if isinstance(value, float):
        if value < 0:
            raise ValueError(f"'{node_id}': min_component_area must be >= 0")
        return

    if isinstance(value, str):
        s = value.strip()
        try:
            if s.endswith('%'):
                number = float(s[:-1])
            else:
                number = float(s)
        except ValueError as exc:
            raise ValueError(
                f"'{node_id}': invalid min_component_area={value!r}"
            ) from exc

        if number < 0:
            raise ValueError(f"'{node_id}': min_component_area must be >= 0")
        return

    raise ValueError(
        f"'{node_id}': min_component_area must be an int, float, "
        'percentage string, or None'
    )


def _resolve_min_component_area(
    value: Any,
    *,
    width: int,
    height: int,
) -> int:
    """
    Resolve a pixel or long-side percentage component area threshold.
    """
    if value is None:
        return 0

    if isinstance(value, str):
        s = value.strip()
        if s.endswith('%'):
            pct = float(s[:-1]) / 100.0
            side = max(width, height) * pct
            return int(round(side * side))
        return int(round(float(s)))

    return int(round(float(value)))


def _remove_small_components(
    mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """
    Remove selected connected components whose area is <= ``min_area``.
    """
    if min_area <= 0:
        return mask

    import cv2

    cm = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return mask

    keep = np.zeros(num, dtype=bool)
    keep[0] = False
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > int(min_area)
    return keep[labels]


def _initial_centers(samples: np.ndarray, count: int) -> np.ndarray:
    """
    Select deterministic, well-separated initial K-means centers.

    The first center is the sample farthest from the sample mean. Each following
    center is the sample whose distance from its nearest existing center is
    greatest. This is similar to farthest-point initialization and avoids the
    run-to-run variation of randomized K-means initialization.

    Parameters
    ----------
    samples : np.ndarray
        Lab samples with shape ``(sample_count, 3)``.

    count : int
        Number of centers to select. The caller guarantees that at least this
        many distinct sample colors exist.

    Returns
    -------
    np.ndarray
        Initial centers with shape ``(count, 3)`` and dtype ``float32``.
    """
    center = np.mean(samples, axis=0, keepdims=True)
    distances = np.sum((samples - center) ** 2, axis=1)
    centers = [samples[int(np.argmax(distances))]]

    # Repeatedly choose the sample least represented by the current centers.
    while len(centers) < count:
        current = np.asarray(centers)
        distances = np.min(
            np.sum((samples[:, None, :] - current[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        centers.append(samples[int(np.argmax(distances))])

    return np.asarray(centers, dtype=np.float32)


def _dominant_centers(
    samples: np.ndarray,
    *,
    analysis_clusters: int,
    num_dominant_colors: int,
    iterations: int,
) -> np.ndarray:
    """
    Cluster one tile and return its most populated Lab color centers.

    A small deterministic K-means implementation is used to keep this node free
    from an additional machine-learning dependency. Empty clusters retain their
    previous center. Cluster importance is measured by assigned sample count,
    and only the requested number of most frequent clusters is returned.

    Parameters
    ----------
    samples : np.ndarray
        Visible Lab samples from one tile, shaped ``(sample_count, 3)``.

    analysis_clusters : int
        Maximum number of clusters fitted to the tile. The effective count is
        reduced when the tile contains fewer distinct colors.

    num_dominant_colors : int
        Number of most populated fitted clusters to return.

    iterations : int
        Maximum number of K-means refinement iterations.

    Returns
    -------
    np.ndarray
        Selected dominant centers with shape ``(selected_count, 3)``.

    Raises
    ------
    ValueError
        If ``samples`` is empty.
    """
    if samples.size == 0:
        raise ValueError('cannot analyze an empty tile')

    # Do not request more clusters than the tile can actually represent.
    unique = np.unique(samples, axis=0)
    cluster_count = min(analysis_clusters, len(unique))
    centers = _initial_centers(samples, cluster_count)
    labels = np.zeros(len(samples), dtype=np.intp)

    for _ in range(iterations):
        distances = np.sum(
            (samples[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()

        for index in range(cluster_count):
            members = samples[new_labels == index]
            if len(members):
                new_centers[index] = np.mean(members, axis=0)

        if np.array_equal(labels, new_labels) and np.allclose(
            centers, new_centers
        ):
            centers = new_centers
            labels = new_labels
            break

        centers = new_centers
        labels = new_labels

    # Stable sorting makes ties deterministic and therefore cache-friendly.
    counts = np.bincount(labels, minlength=cluster_count)
    order = np.argsort(-counts, kind='stable')
    selected_count = min(num_dominant_colors, cluster_count)
    return centers[order[:selected_count]]


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    """
    Compute tile origins along one image axis.

    The final tile is explicitly aligned with the end of the axis when a regular
    stride would leave a remainder. Consequently every pixel is covered, even
    when the image dimension is not a multiple of the tile step.

    Parameters
    ----------
    length : int
        Image length along the selected axis.

    tile_size : int
        Requested square tile size. It is clamped to ``length``.

    overlap : float
        Fractional overlap between consecutive tiles in ``[0, 1)``.

    Returns
    -------
    list[int]
        Ordered, zero-based tile start coordinates.
    """
    actual_size = min(length, tile_size)
    if actual_size == length:
        return [0]

    step = max(1, int(round(actual_size * (1.0 - overlap))))
    starts = list(range(0, length - actual_size + 1, step))
    last = length - actual_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_weights(height: int, width: int) -> np.ndarray:
    """
    Build a separable soft blending window for one tile.

    A squared-sine window gives the tile center more influence than its edges.
    Overlapping local alpha estimates can therefore be averaged without hard
    seams. A small positive floor keeps outer image pixels covered when they
    belong only to a border tile.

    Parameters
    ----------
    height : int
        Tile height in pixels.

    width : int
        Tile width in pixels.

    Returns
    -------
    np.ndarray
        Positive ``float32`` weights with shape ``(height, width)``.
    """

    def axis_weights(length: int) -> np.ndarray:
        """Return the one-dimensional squared-sine blending window."""
        if length == 1:
            return np.ones(1, dtype=np.float32)
        positions = (np.arange(length, dtype=np.float32) + 0.5) / length
        return np.sin(np.pi * positions) ** 2

    weights = axis_weights(height)[:, None] * axis_weights(width)[None, :]
    return np.maximum(weights, 1e-4)


def _alpha_factor(
    lab: np.ndarray,
    centers: np.ndarray,
    *,
    tolerance: float,
    feather: float,
    retained_alpha: float,
) -> np.ndarray:
    """
    Convert distance from reference colors into an alpha retention factor.

    Pixels at or below ``tolerance`` receive ``retained_alpha``. Pixels beyond
    ``tolerance + feather`` retain their original alpha. Values in between are
    interpolated with a smoothstep curve. With ``feather=0`` the result is a
    hard threshold.

    Parameters
    ----------
    lab : np.ndarray
        Lab tile with shape ``(height, width, 3)``.

    centers : np.ndarray
        Reference Lab centers with shape ``(cluster_count, 3)``.

    tolerance : float
        Radius around each reference center receiving maximum attenuation.

    feather : float
        Width of the soft transition outside ``tolerance``.

    retained_alpha : float
        Alpha retention factor applied at the low end of the score ramp.

    Returns
    -------
    np.ndarray
        Alpha retention factors in ``[retained_alpha, 1]``, shaped
        ``(height, width)``.
    """
    # A pixel is compared with every reference center and classified by the
    # nearest one. Euclidean Lab distance approximates perceptual distance.
    distance = np.sqrt(
        np.min(
            np.sum((lab[:, :, None, :] - centers[None, None, :, :]) ** 2,
                   axis=3),
            axis=2,
        )
    )

    if feather == 0.0:
        transition = (distance > tolerance).astype(np.float32)
    else:
        transition = np.clip(
            (distance - tolerance) / feather,
            0.0,
            1.0,
        )
        # Smoothstep avoids a visible slope discontinuity at either threshold.
        transition = transition * transition * (3.0 - 2.0 * transition)

    return retained_alpha + (1.0 - retained_alpha) * transition


def _color_driven_arrays(
    image: Image.Image,
    *,
    tolerance: float = 12.0,
    feather: float = 6.0,
    strength: float = 1.0,
    colors: Any = None,
    color_policy: str = 'exclude',
    color_scope: str = 'global',
    num_dominant_colors: int = 1,
    analysis_clusters: int = _DEFAULT_ANALYSIS_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return full-frame RGBA data and a color-selected positive mask.
    """
    if color_policy not in ('exclude', 'include'):
        raise ValueError(
            f"invalid color_policy={color_policy!r}"
        )
    if color_scope not in ('global', 'local'):
        raise ValueError(
            f"invalid color_scope={color_scope!r}"
        )

    rgba = np.asarray(image.convert('RGBA'), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)
    original_alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    height, width = original_alpha.shape
    retained_alpha = 1.0 - strength
    manual_centers = None
    global_centers = None
    if colors is not None:
        colors = _parse_colors(
            colors,
            node_id='color_driven_alpha',
        )
        color_rgb = (
            np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            / 255.0
        )
        manual_centers = rgb2lab(color_rgb.reshape(1, -1, 3)).reshape(-1, 3)
        manual_centers = manual_centers.astype(np.float32)
    elif color_scope == 'global':
        sampled_lab = lab[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE]
        sampled_alpha = original_alpha[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE]
        samples = sampled_lab[sampled_alpha > 0.0].reshape(-1, 3)
        if len(samples) > 0:
            global_centers = _dominant_centers(
                samples,
                analysis_clusters=analysis_clusters,
                num_dominant_colors=num_dominant_colors,
                iterations=_KMEANS_ITERATIONS,
            )

    preserve_score_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    tile_height = min(height, _TILE_SIZE)
    tile_width = min(width, _TILE_SIZE)

    # Accumulate weighted local estimates instead of writing tile results
    # directly, which would expose discontinuities at tile boundaries.
    for top in _tile_starts(height, _TILE_SIZE, _TILE_OVERLAP):
        bottom = top + tile_height
        for left in _tile_starts(width, _TILE_SIZE, _TILE_OVERLAP):
            right = left + tile_width
            tile_lab = lab[top:bottom, left:right]
            tile_alpha = original_alpha[top:bottom, left:right]

            if manual_centers is not None:
                centers = manual_centers
            elif global_centers is not None:
                centers = global_centers
            elif color_scope == 'local':
                sampled_lab = tile_lab[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE]
                sampled_alpha = tile_alpha[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE]
                # Existing transparent pixels carry no reliable visible color
                # and must not influence the local dominant-color model.
                samples = sampled_lab[sampled_alpha > 0.0].reshape(-1, 3)
                if len(samples) == 0:
                    continue

                centers = _dominant_centers(
                    samples,
                    analysis_clusters=analysis_clusters,
                    num_dominant_colors=num_dominant_colors,
                    iterations=_KMEANS_ITERATIONS,
                )
            else:
                continue

            distance_score = _alpha_factor(
                tile_lab,
                centers,
                tolerance=tolerance,
                feather=feather,
                retained_alpha=0.0,
            )
            if color_policy == 'exclude':
                preserve_score = distance_score
            else:
                preserve_score = 1.0 - distance_score

            weights = _tile_weights(tile_height, tile_width)
            preserve_score_sum[top:bottom, left:right] += (
                preserve_score * weights
            )
            weight_sum[top:bottom, left:right] += weights

    # Tiles containing no visible samples contribute nothing. In uncovered
    # locations the neutral factor 1 preserves the original alpha.
    preserve_score = np.divide(
        preserve_score_sum,
        weight_sum,
        out=np.ones_like(preserve_score_sum),
        where=weight_sum > 0.0,
    )
    factor = retained_alpha + (1.0 - retained_alpha) * preserve_score
    selected_mask = (original_alpha > 0.0) & (preserve_score >= 0.999)

    output = rgba.copy()
    output[:, :, 3] = np.round(
        np.clip(original_alpha * factor, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return output, selected_mask


def color_driven_alpha(
    image: Image.Image,
    *,
    tolerance: float = 12.0,
    feather: float = 6.0,
    strength: float = 1.0,
    colors: Any = None,
    color_policy: str = 'exclude',
    color_scope: str = 'global',
    num_dominant_colors: int = 1,
    analysis_clusters: int = _DEFAULT_ANALYSIS_CLUSTERS,
) -> Image.Image:
    """
    Apply color-driven alpha attenuation and return an RGBA image.

    RGB values are preserved exactly. Only alpha is reduced according to the
    selected reference colors or automatically estimated dominant colors.
    """
    output, _ = _color_driven_arrays(
        image,
        tolerance=tolerance,
        feather=feather,
        strength=strength,
        colors=colors,
        color_policy=color_policy,
        color_scope=color_scope,
        num_dominant_colors=num_dominant_colors,
        analysis_clusters=analysis_clusters,
    )
    return Image.fromarray(output, mode='RGBA')


@dataclass
class ColorDrivenCrop(NodeRef):
    """
    Color-driven crop, cutout, and mask generator.

    ``ColorDrivenCrop`` is a deterministic, model-free preprocessing node. It
    selects pixels by color relationship rather than by semantic recognition and
    can write either an RGBA crop/cutout or a full-frame mask.

    The node is color-driven, not semantic-driven. It does not recognize
    objects, text, people, or prompted concepts. Use ``SubjectCrop`` or
    ``AnyCrop`` when the desired selection depends on semantic recognition.

    Selection modes
    ---------------
    If ``colors`` is provided, the node runs in manual deterministic mode.
    Those RGB colors are converted to Lab and used directly as the reference
    color families. ``colors`` accepts PIL/CSS-like color names
    such as ``'white'`` or ``'blue'``, hex strings such as ``'#ffffff'``, and RGB
    triplets such as ``[255, 255, 255]``. When this mode is active,
    ``num_dominant_colors`` is reported as the number of manual colors.

    If ``colors`` is absent, the node runs in automatic mode. It estimates
    dominant reference colors from the visible image. By default this estimate is
    global, which is usually the most intuitive behavior for cropping a larger
    object from a comparatively simple background. Advanced workflows can switch
    to local tile analysis with ``color_scope='local'``.

    ``color_policy`` controls how reference colors are interpreted. With
    ``'exclude'`` they are treated as background/noise to attenuate, and pixels
    far from them are selected. With ``'include'`` they are treated as foreground
    to preserve, and pixels near them are selected. The same policy applies to
    manual colors and automatically estimated dominant colors.

    Output modes
    ------------
    ``mode='default'`` writes an RGBA image. ``crop_mode`` controls only the
    output geometry: ``'trim'`` crops tightly around the selected region,
    ``'bbox'`` keeps a rectangular crop, ``'bbox[w:h]'`` expands that rectangle
    toward an aspect ratio, and ``'full_frame'`` preserves full input geometry.
    Unlike semantic crop nodes, rectangular ``ColorDrivenCrop`` outputs keep the
    color-driven alpha attenuation.

    ``mode='mask'`` writes a full-frame positive mask where selected pixels are
    white. ``mode='negative-mask'`` writes the inverted full-frame mask, where
    the selected region is protected in black and the background is white.

    Alpha behavior
    --------------
    ``strength`` applies only when ``mode='default'`` and controls how much
    non-preserved pixels are attenuated:

    - ``1.0``: non-preserved pixels become fully transparent;
    - ``0.5``: non-preserved pixels retain half of their existing alpha;
    - ``0.0``: alpha remains unchanged.

    Mask outputs ignore ``strength`` and use only the color-driven selected
    region.

    Pixels outside the policy-selected color region receive the strongest
    attenuation. The following ``feather`` Lab-distance units transition
    smoothly back to unchanged alpha.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default
        input using ``input['default']['image']`` or
        ``input['default']['path']``.

    spec : dict or str or Path, optional
        Node specification, resolved through ``resolve_spec``.

        Expected structure:

        ``params`` : dict
            ``mode`` : {'default', 'mask', 'negative-mask'}, optional
                Output type. Default: ``'default'``.

            ``crop_mode`` : {'trim', 'bbox', 'bbox[w:h]', 'full_frame'}, optional
                Applies only when ``mode='default'``. Controls RGBA output
                geometry. Default: ``'trim'``.

            ``color_policy`` : {'exclude', 'include'}, optional
                Interpretation of reference colors. ``'exclude'`` treats them
                as background/noise and selects pixels far from them.
                ``'include'`` treats them as foreground and selects pixels near
                them. Default: ``'exclude'``.

            ``tolerance`` : float, optional
                Lab-distance radius around every reference color. Its meaning is
                interpreted through ``color_policy``. Must be non-negative.
                Default: ``12.0``.

            ``feather`` : float, optional
                Width, in Lab-distance units, of the smooth transition between
                attenuated and unchanged alpha. ``0`` creates a hard threshold.
                Must be non-negative. Default: ``6.0``.

            ``strength`` : float, optional
                Attenuation strength in ``[0, 1]``. ``1`` fully removes
                non-preserved pixels, ``0.5`` attenuates them partially, and
                ``0`` disables alpha changes. Default: ``1.0``.

            ``box_margin`` : float, optional
                Symmetric expansion ratio applied to the color-derived
                selection bbox, expressed as a fraction of bbox size. Applies
                only when ``mode='default'`` and ``crop_mode`` is ``'trim'``,
                ``'bbox'`` or ``'bbox[w:h]'``. Ignored for
                ``crop_mode='full_frame'`` and mask outputs. For
                ``crop_mode='bbox[w:h]'``, the margin is applied before
                aspect-ratio expansion. Typical range: 0.03-0.12. Default:
                ``0.08``.

            ``colors`` : str or list, optional
                Manual reference colors. Accepted entries are PIL/CSS-like
                names, hex strings, and RGB triplets. If omitted, dominant
                reference colors are estimated automatically. Mutually
                exclusive with ``num_dominant_colors``. Default: ``None``.

            ``num_dominant_colors`` : int, optional
                Number of dominant reference color families to use in automatic
                mode. The colors are estimated globally or locally depending on
                ``color_scope``. In manual mode, metadata reports the number of
                colors provided. Mutually exclusive with ``colors``. Must be at
                least ``1`` in automatic mode. Default: ``1``.

            ``min_component_area`` : int, float, str or None, optional
                Remove small selected connected components from the crop/mask
                selection when their area is less than or equal to this
                threshold. In default RGBA mode, removed selected components are
                treated as non-preserved color noise and attenuated according to
                ``strength``. Pixel values are used directly. Percentage strings
                such as ``'1%'`` are interpreted linearly on the image long side
                and converted to area. Default: ``0``.

            ``dilate_radius`` : int, optional
                Mask dilation radius in pixels, used only for ``mode='mask'``
                and ``mode='negative-mask'``. Default: ``0``.

            ``close_radius`` : int, optional
                Morphological closing radius in pixels, used only for full-frame
                mask outputs. Default: ``0``.

            ``smoothing_radius`` : int, optional
                Gaussian smoothing radius in pixels, used only for full-frame
                mask outputs. Default: ``0``.

            ``analysis_clusters`` : int or None, optional
                Advanced automatic-mode K-means cluster limit. If ``None``, an
                internal default is used. Must be at least ``1`` when provided.
                Increase only when the reference surface contains several
                important color families. Default: ``None``.

            ``color_scope`` : {'global', 'local'}, optional
                Advanced automatic-mode analysis scope. ``'global'`` estimates
                dominant reference colors once from the full visible image and
                reuses them everywhere. This is the default and is usually safer
                for larger subjects, because a large object will not become
                background merely by dominating one tile. ``'local'`` estimates
                reference colors independently per tile and blends the results;
                it is useful for text, logos, or markings on uneven surfaces
                such as paper with shadows, fabric, walls, or curved panels.
                Manual ``colors`` are always global. Default: ``'global'``.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The output contains ``image``, ``mode``, ``input_size``,
        ``output_size``, ``color_driven_crop`` with resolved configuration,
        ``params`` as a copy of the same public parameters, and ``crop`` with
        reinsertion metadata compatible with other preprocessing crop nodes.

    Notes
    -----
    - Existing transparency is respected. The operation can only preserve or
      reduce source alpha; it never makes a source pixel more opaque.
    - The algorithm is deterministic and has no learned-model, CUDA, or network
      dependency.
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
            input_size = image.size
            w, h = image.size
            rgba, selected_mask = _color_driven_arrays(
                image,
                tolerance=cfg.tolerance,
                feather=cfg.feather,
                strength=cfg.strength,
                colors=cfg.colors,
                color_policy=cfg.color_policy,
                color_scope=cfg.color_scope,
                num_dominant_colors=cfg.num_dominant_colors,
                analysis_clusters=cfg.analysis_clusters,
            )

        min_component_area = _resolve_min_component_area(
            cfg.min_component_area,
            width=w,
            height=h,
        )
        raw_selected_mask = selected_mask
        selected_mask = _remove_small_components(
            selected_mask,
            min_area=min_component_area,
        )
        removed_selected_mask = raw_selected_mask & ~selected_mask
        if np.any(removed_selected_mask):
            rgba = rgba.copy()
            retained_alpha = 1.0 - cfg.strength
            rgba[:, :, 3][removed_selected_mask] = np.round(
                rgba[:, :, 3][removed_selected_mask].astype(np.float32)
                * retained_alpha
            ).astype(np.uint8)

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=node_id,
            ext='png',
        )

        if cfg.mode == 'default':
            if cfg.crop_mode is None:
                raise RuntimeError(
                    f"{self.id}: crop_mode must be defined when mode='default'"
                )

            out_x1 = 0
            out_y1 = 0
            out_x2 = w
            out_y2 = h

            if cfg.crop_mode.mode == 'full_frame':
                output_image = Image.fromarray(rgba, mode='RGBA')

            else:
                if not np.any(selected_mask):
                    raise RuntimeError(
                        f'ColorDrivenCrop node {self.id!r}: selected mask is empty.'
                    )

                alpha = selected_mask.astype(np.uint8) * 255
                x1, y1, x2, y2 = tight_alpha_bbox(alpha)
                if cfg.box_margin > 0:
                    x1, y1, x2, y2 = expand_clip_bbox(
                        x1,
                        y1,
                        x2,
                        y2,
                        w,
                        h,
                        cfg.box_margin,
                    )

                if cfg.crop_mode.mode == 'bbox':
                    if cfg.crop_mode.ratio is not None:
                        x1, y1, x2, y2 = expand_bbox_toward_ratio(
                            x1,
                            y1,
                            x2,
                            y2,
                            full_w=w,
                            full_h=h,
                            ratio=cfg.crop_mode.ratio,
                        )

                    out_x1 = int(x1)
                    out_y1 = int(y1)
                    out_x2 = int(x2)
                    out_y2 = int(y2)
                    output_image = Image.fromarray(
                        rgba[out_y1:out_y2, out_x1:out_x2, :],
                        mode='RGBA',
                    )

                elif cfg.crop_mode.mode == 'trim':
                    out_x1 = int(x1)
                    out_y1 = int(y1)
                    out_x2 = int(x2)
                    out_y2 = int(y2)
                    output_image = Image.fromarray(
                        rgba[out_y1:out_y2, out_x1:out_x2, :],
                        mode='RGBA',
                    )

                else:
                    raise ValueError(
                        f'{self.id}: invalid crop_mode={cfg.crop_mode.mode!r}'
                    )

        else:
            out_mask_u8 = postprocess_mask(
                selected_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                out_mask_u8 = 255 - out_mask_u8

            output_image = Image.fromarray(out_mask_u8, mode='L')
            out_x1 = 0
            out_y1 = 0
            out_x2 = w
            out_y2 = h

        output_image.save(out_path)

        bbox_w = int(out_x2 - out_x1)
        bbox_h = int(out_y2 - out_y1)
        anchor_x = int(round((out_x1 + out_x2) / 2.0))
        anchor_y = int(round((out_y1 + out_y2) / 2.0))

        params = {
            'mode': cfg.mode,
            'crop_mode': cfg.crop_mode.raw if cfg.crop_mode is not None else None,
            'color_policy': cfg.color_policy,
            'color_scope': cfg.color_scope,
            'tolerance': cfg.tolerance,
            'feather': cfg.feather,
            'strength': cfg.strength,
            'box_margin': cfg.box_margin,
            'colors': (
                [list(color) for color in cfg.colors]
                if cfg.colors is not None
                else None
            ),
            'num_dominant_colors': cfg.num_dominant_colors,
            'analysis_clusters': cfg.analysis_clusters,
            'min_component_area': cfg.min_component_area,
            'dilate_radius': cfg.dilate_radius,
            'close_radius': cfg.close_radius,
            'smoothing_radius': cfg.smoothing_radius,
        }
        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'input_size': [int(input_size[0]), int(input_size[1])],
            'output_size': [int(output_image.width), int(output_image.height)],
            'color_driven_crop': params.copy(),
            'params': params,
            'crop': {
                'anchor_xy': [
                    int(anchor_x - out_x1),
                    int(anchor_y - out_y1),
                ],
                'position': [anchor_x, anchor_y],
                'bbox_size': [bbox_w, bbox_h],
                'bbox_xyxy': [
                    int(out_x1),
                    int(out_y1),
                    int(out_x2),
                    int(out_y2),
                ],
            },
        }
        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)
        return out
