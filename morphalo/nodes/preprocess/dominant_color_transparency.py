from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


@dataclass(frozen=True)
class Config:
    """
    Validated runtime configuration for ``DominantColorTransparency``.

    All color-distance values are expressed in the CIE Lab space produced by
    :func:`skimage.color.rgb2lab`. ``dominant_alpha`` is a multiplicative alpha
    retention factor, not an absolute 8-bit alpha value.
    """

    analysis_clusters: int
    num_dominant_clusters: int
    tile_size: int
    tile_overlap: float
    color_tolerance: float
    feather: float
    dominant_alpha: float
    sample_stride: int
    kmeans_iterations: int


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

    analysis_clusters = int(params.get('analysis_clusters', 6))
    num_dominant_clusters = int(params.get('num_dominant_clusters', 1))
    tile_size = int(params.get('tile_size', 256))
    tile_overlap = float(params.get('tile_overlap', 0.5))
    color_tolerance = float(params.get('color_tolerance', 12.0))
    feather = float(params.get('feather', 6.0))
    dominant_alpha = float(params.get('dominant_alpha', 0.0))
    sample_stride = int(params.get('sample_stride', 2))
    kmeans_iterations = int(params.get('kmeans_iterations', 12))

    if analysis_clusters < 1:
        raise ValueError(
            f"'{node_id}': analysis_clusters must be >= 1"
        )
    if not 1 <= num_dominant_clusters <= analysis_clusters:
        raise ValueError(
            f"'{node_id}': num_dominant_clusters must be between 1 and "
            'analysis_clusters'
        )
    if tile_size < 1:
        raise ValueError(f"'{node_id}': tile_size must be >= 1")
    if not 0.0 <= tile_overlap < 1.0:
        raise ValueError(
            f"'{node_id}': tile_overlap must be in [0, 1)"
        )
    if color_tolerance < 0.0:
        raise ValueError(
            f"'{node_id}': color_tolerance must be >= 0"
        )
    if feather < 0.0:
        raise ValueError(f"'{node_id}': feather must be >= 0")
    if not 0.0 <= dominant_alpha <= 1.0:
        raise ValueError(
            f"'{node_id}': dominant_alpha must be in [0, 1]"
        )
    if sample_stride < 1:
        raise ValueError(f"'{node_id}': sample_stride must be >= 1")
    if kmeans_iterations < 1:
        raise ValueError(
            f"'{node_id}': kmeans_iterations must be >= 1"
        )

    return Config(
        analysis_clusters=analysis_clusters,
        num_dominant_clusters=num_dominant_clusters,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        color_tolerance=color_tolerance,
        feather=feather,
        dominant_alpha=dominant_alpha,
        sample_stride=sample_stride,
        kmeans_iterations=kmeans_iterations,
    )


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
    num_dominant_clusters: int,
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

    num_dominant_clusters : int
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
    selected_count = min(num_dominant_clusters, cluster_count)
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
    color_tolerance: float,
    feather: float,
    dominant_alpha: float,
) -> np.ndarray:
    """
    Convert distance from dominant colors into an alpha retention factor.

    Pixels at or below ``color_tolerance`` receive ``dominant_alpha``. Pixels
    beyond ``color_tolerance + feather`` retain their original alpha. Values in
    between are interpolated with a smoothstep curve. With ``feather=0`` the
    result is a hard threshold.

    Parameters
    ----------
    lab : np.ndarray
        Lab tile with shape ``(height, width, 3)``.

    centers : np.ndarray
        Dominant Lab centers with shape ``(cluster_count, 3)``.

    color_tolerance : float
        Radius around each dominant center receiving maximum attenuation.

    feather : float
        Width of the soft transition outside ``color_tolerance``.

    dominant_alpha : float
        Alpha retention factor applied to pixels matching dominant colors.

    Returns
    -------
    np.ndarray
        Alpha retention factors in ``[dominant_alpha, 1]``, shaped
        ``(height, width)``.
    """
    # A pixel is compared with every selected local cluster and classified by
    # the nearest one. Euclidean Lab distance approximates perceptual distance.
    distance = np.sqrt(
        np.min(
            np.sum((lab[:, :, None, :] - centers[None, None, :, :]) ** 2,
                   axis=3),
            axis=2,
        )
    )

    if feather == 0.0:
        transition = (distance > color_tolerance).astype(np.float32)
    else:
        transition = np.clip(
            (distance - color_tolerance) / feather,
            0.0,
            1.0,
        )
        # Smoothstep avoids a visible slope discontinuity at either threshold.
        transition = transition * transition * (3.0 - 2.0 * transition)

    return dominant_alpha + (1.0 - dominant_alpha) * transition


def dominant_color_alpha(
    image: Image.Image,
    *,
    analysis_clusters: int = 6,
    num_dominant_clusters: int = 1,
    tile_size: int = 256,
    tile_overlap: float = 0.5,
    color_tolerance: float = 12.0,
    feather: float = 6.0,
    dominant_alpha: float = 0.0,
    sample_stride: int = 2,
    kmeans_iterations: int = 12,
) -> Image.Image:
    """
    Apply local dominant-color attenuation and return an RGBA image.

    The image is converted to CIE Lab and divided into overlapping square tiles.
    Each tile fits a deterministic K-means model to a subsampled set of visible
    pixels. Its most populated clusters are treated as local surface colors.
    Pixel distance from those clusters is transformed into an alpha retention
    factor, and overlapping tile estimates are blended with soft spatial
    weights.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image in any PIL mode. It is converted to RGBA internally.

    analysis_clusters : int, default=6
        Maximum number of color clusters fitted independently in each tile.

    num_dominant_clusters : int, default=1
        Number of most populated local clusters attenuated in each tile.

    tile_size : int, default=256
        Width and height of each analysis tile in pixels. For an image dimension
        smaller than this value, the tile is clamped to that dimension.

    tile_overlap : float, default=0.5
        Fractional overlap between neighboring tiles in ``[0, 1)``.

    color_tolerance : float, default=12.0
        Lab-distance radius around a dominant center receiving the strongest
        attenuation.

    feather : float, default=6.0
        Additional Lab-distance interval over which attenuation transitions
        smoothly back to the original alpha. ``0`` produces a hard threshold.

    dominant_alpha : float, default=0.0
        Retention factor for pixels matching dominant colors. ``0`` makes them
        transparent, ``0.5`` halves their existing alpha, and ``1`` disables
        attenuation.

    sample_stride : int, default=2
        Spatial stride used only while fitting tile clusters. The resulting
        model is still evaluated at full output resolution.

    kmeans_iterations : int, default=12
        Maximum deterministic K-means refinement iterations per tile.

    Returns
    -------
    PIL.Image.Image
        Full-size RGBA image. RGB values are preserved exactly; only alpha is
        changed.

    Notes
    -----
    - Fully transparent source pixels are excluded from cluster fitting.
    - The output alpha never exceeds the source alpha.
    - This function expects validated parameters when called directly. The DAG
      node validates all values through :func:`_read_cfg`.
    """
    rgba = np.asarray(image.convert('RGBA'), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)
    original_alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    height, width = original_alpha.shape

    factor_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    tile_height = min(height, tile_size)
    tile_width = min(width, tile_size)

    # Accumulate weighted local estimates instead of writing tile results
    # directly, which would expose discontinuities at tile boundaries.
    for top in _tile_starts(height, tile_size, tile_overlap):
        bottom = top + tile_height
        for left in _tile_starts(width, tile_size, tile_overlap):
            right = left + tile_width
            tile_lab = lab[top:bottom, left:right]
            tile_alpha = original_alpha[top:bottom, left:right]

            sampled_lab = tile_lab[::sample_stride, ::sample_stride]
            sampled_alpha = tile_alpha[::sample_stride, ::sample_stride]
            # Existing transparent pixels carry no reliable visible color and
            # must not influence the local dominant-color model.
            samples = sampled_lab[sampled_alpha > 0.0].reshape(-1, 3)
            if len(samples) == 0:
                continue

            centers = _dominant_centers(
                samples,
                analysis_clusters=analysis_clusters,
                num_dominant_clusters=num_dominant_clusters,
                iterations=kmeans_iterations,
            )
            factor = _alpha_factor(
                tile_lab,
                centers,
                color_tolerance=color_tolerance,
                feather=feather,
                dominant_alpha=dominant_alpha,
            )
            weights = _tile_weights(tile_height, tile_width)
            factor_sum[top:bottom, left:right] += factor * weights
            weight_sum[top:bottom, left:right] += weights

    # Tiles containing no visible samples contribute nothing. In uncovered
    # locations the neutral factor 1 preserves the original alpha.
    factor = np.divide(
        factor_sum,
        weight_sum,
        out=np.ones_like(factor_sum),
        where=weight_sum > 0.0,
    )
    output = rgba.copy()
    output[:, :, 3] = np.round(
        np.clip(original_alpha * factor, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return Image.fromarray(output, mode='RGBA')


@dataclass
class DominantColorTransparency(NodeRef):
    """
    Attenuate locally dominant color families through the image alpha channel.

    ``DominantColorTransparency`` is a deterministic, model-free preprocessing
    node intended to isolate details drawn, printed, carved, or painted on a
    comparatively broad surface. Typical uses include extracting:

    - text printed on a T-shirt;
    - letters or illustrations from a photographed page;
    - glyphs, graffiti, or signs from a wall;
    - a logo or emblem from a slowly varying material;
    - local detail used as a naive inpaint or compositing layer.

    This node is deliberately not a semantic background remover. ``SubjectCrop``
    and ``AnyCrop`` should be used when the desired result depends on recognizing
    a subject or a prompted object. Here, "dominant" means only "one of the most
    frequent local color families".

    Local color model
    -----------------
    A single global dominant color is insufficient for photographed surfaces.
    Illumination, shadows, perspective, material texture, and page curvature may
    cause the same surface to range from white to gray or from bright to dark.
    Conversely, a gray letter in a bright area may have the same RGB value as
    the surface in a darker area.

    To handle this ambiguity, the node builds a spatially varying model:

    1. Convert the source RGB channels to CIE Lab color space.
    2. Divide the image into overlapping square tiles.
    3. Subsample visible pixels in each tile using ``sample_stride``.
    4. Fit up to ``analysis_clusters`` deterministic K-means clusters per tile.
    5. Select the ``num_dominant_clusters`` most populated local clusters.
    6. Measure every tile pixel against its nearest selected Lab center.
    7. Convert that distance into a soft alpha retention factor.
    8. Blend overlapping tile estimates with smooth spatial weights.
    9. Multiply the blended factor by the original image alpha.

    The local treatment means an identical RGB color can be preserved in one
    region and attenuated in another, depending on the color distribution around
    it.

    Alpha behavior
    --------------
    ``dominant_alpha`` controls how much of the original alpha remains for a
    pixel matching a local dominant cluster:

    - ``0.0``: matching pixels become fully transparent;
    - ``0.5``: matching pixels retain half of their existing alpha;
    - ``1.0``: alpha remains unchanged, effectively disabling attenuation.

    Pixels within ``color_tolerance`` of a selected center receive this factor.
    The following ``feather`` units of Lab distance transition smoothly toward a
    factor of ``1.0``. More distant pixels retain their original alpha.

    Existing transparency is respected. The operation can only preserve or
    reduce source alpha; it never makes a source pixel more opaque.

    Example
    -------
    The following node extracts dark printed detail from a photographed page
    whose background changes gradually under uneven illumination:

    .. code-block:: python

        from morphalo.nodes.preprocess import DominantColorTransparency

        page_detail = DominantColorTransparency(
            name='page_detail',
            path='photo_of_page.png',
            spec={
                'params': {
                    'analysis_clusters': 6,
                    'num_dominant_clusters': 1,
                    'tile_size': 256,
                    'tile_overlap': 0.5,
                    'color_tolerance': 12.0,
                    'feather': 6.0,
                    'dominant_alpha': 0.0,
                    'sample_stride': 2,
                    'kmeans_iterations': 12,
                },
            },
        )

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
            ``analysis_clusters`` : int, optional
                Maximum number of color clusters fitted independently inside
                each tile. Increasing this value can separate multiple surface
                tones from foreground detail, but also increases work and may
                split a continuous surface into several families.

                Must be at least ``1``. Default: ``6``.

            ``num_dominant_clusters`` : int, optional
                Number of the most populated fitted clusters attenuated in each
                tile. Use ``1`` for a mostly uniform page, fabric, or wall.
                Increase it when the local surface itself contains several
                frequent tones.

                Must be between ``1`` and ``analysis_clusters``. A high value
                may classify foreground detail as dominant. Default: ``1``.

            ``tile_size`` : int, optional
                Square analysis tile size in pixels. Smaller tiles adapt more
                closely to local illumination but are more likely to treat a
                large letter or mark as locally dominant. Larger tiles provide
                more stable frequency estimates but adapt less to shadows and
                gradients.

                Must be at least ``1``. Default: ``256``.

            ``tile_overlap`` : float, optional
                Fractional overlap between neighboring tiles. Overlap lets the
                node blend several local estimates and reduces visible tile
                boundaries. Larger values improve continuity at additional
                computation cost.

                Expected range: ``[0, 1)``. Default: ``0.5``.

            ``color_tolerance`` : float, optional
                Radius in Lab color-distance units around every selected
                dominant center. Pixels inside this radius receive
                ``dominant_alpha``.

                Larger values remove a broader color range and may affect
                foreground detail. Must be non-negative. Default: ``12.0``.

            ``feather`` : float, optional
                Width, in Lab color-distance units, of the smooth transition
                between ``dominant_alpha`` and unchanged alpha. ``0`` creates a
                hard threshold.

                Must be non-negative. Default: ``6.0``.

            ``dominant_alpha`` : float, optional
                Alpha retention factor for pixels matching dominant colors.
                ``0`` removes them, ``1`` leaves them unchanged, and intermediate
                values produce partial transparency.

                Expected range: ``[0, 1]``. Default: ``0.0``.

            ``sample_stride`` : int, optional
                Pixel stride used while fitting clusters. ``1`` analyzes every
                visible pixel; ``2`` analyzes every second pixel on each axis;
                larger values reduce clustering cost. Regardless of this value,
                the alpha mask is evaluated at full image resolution.

                Must be at least ``1``. Default: ``2``.

            ``kmeans_iterations`` : int, optional
                Maximum K-means refinement iterations for each tile. The
                implementation may stop earlier when assignments and centers
                converge.

                Must be at least ``1``. Default: ``12``.

    Inputs
    ------
    default : dict, optional
        Upstream image payload used when ``path`` is omitted.

        The payload must contain either:

        - ``image`` : str
        - ``path`` : str

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The output contains:

        ``ok`` : bool
            Success flag.

        ``node`` : str
            Operator name, normally ``'DominantColorTransparency'``.

        ``id`` : str
            Node identifier.

        ``input_image`` : str
            Resolved source image path.

        ``image`` : str
            Path to the full-size RGBA PNG. RGB values come from the source
            image; the alpha channel contains the attenuation result.

        ``input_size`` : list[int]
            Source image size as ``[width, height]``.

        ``output_size`` : list[int]
            Output image size as ``[width, height]``. Geometry is unchanged.

        ``dominant_color_transparency`` : dict
            Resolved algorithm configuration. It contains
            ``analysis_clusters``, ``num_dominant_clusters``, ``tile_size``,
            ``tile_overlap``, ``color_tolerance``, ``feather``,
            ``dominant_alpha``, ``sample_stride`` and ``kmeans_iterations``.

        ``params`` : dict
            Copy of the resolved algorithm parameters for consistency with other
            preprocessing nodes.

        ``metadata`` : str
            Path to the JSON sidecar written next to the output image.

    Notes
    -----
    - The output is always a full-frame RGBA PNG with the same dimensions as the
      source image.
    - RGB channels are not recolored, blurred, or normalized. Only alpha changes.
    - The algorithm is deterministic and has no learned-model, CUDA, or network
      dependency.
    - Color clustering uses only source pixels whose existing alpha is greater
      than zero.
    - The method reasons about local color frequency, not object identity,
      typography, depth, or semantic background.
    - A tile containing more foreground detail than surface may classify that
      detail as dominant. Increase ``tile_size`` or reduce
      ``num_dominant_clusters`` in that case.
    - Very small ``tile_size`` values may remove thick letters, large logos, or
      broad painted marks because they dominate individual tiles.
    - Very large ``tile_size`` values approach a global color model and become
      less effective for uneven lighting or curved surfaces.
    - Increasing ``color_tolerance`` and ``feather`` produces broader, softer
      removal. Excessive values may erase foreground colors near the surface
      distribution.
    - If the desired selection depends on semantic recognition, use
      ``SubjectCrop`` or ``AnyCrop`` instead.
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
            result = dominant_color_alpha(
                image,
                analysis_clusters=cfg.analysis_clusters,
                num_dominant_clusters=cfg.num_dominant_clusters,
                tile_size=cfg.tile_size,
                tile_overlap=cfg.tile_overlap,
                color_tolerance=cfg.color_tolerance,
                feather=cfg.feather,
                dominant_alpha=cfg.dominant_alpha,
                sample_stride=cfg.sample_stride,
                kmeans_iterations=cfg.kmeans_iterations,
            )

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=node_id,
            ext='png',
        )
        result.save(out_path)

        params = {
            'analysis_clusters': cfg.analysis_clusters,
            'num_dominant_clusters': cfg.num_dominant_clusters,
            'tile_size': cfg.tile_size,
            'tile_overlap': cfg.tile_overlap,
            'color_tolerance': cfg.color_tolerance,
            'feather': cfg.feather,
            'dominant_alpha': cfg.dominant_alpha,
            'sample_stride': cfg.sample_stride,
            'kmeans_iterations': cfg.kmeans_iterations,
        }
        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'image': str(out_path),
            'input_size': [int(input_size[0]), int(input_size[1])],
            'output_size': [int(result.width), int(result.height)],
            'dominant_color_transparency': params.copy(),
            'params': params,
        }
        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)
        return out
