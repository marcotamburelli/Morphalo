import math
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from morphalo.nodes.preprocess.utils import resolve_min_component_area


def invert_mask_inside_box(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Invert a mask only inside a prompt bounding box.
    """
    x1, y1, x2, y2 = box

    inv = ~mask
    out = np.zeros_like(mask, dtype=bool)
    out[y1:y2, x1:x2] = inv[y1:y2, x1:x2]

    return out


def remove_small_components(
    mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """
    Remove foreground connected components whose area is below a threshold.

    The input is treated as a binary foreground mask: non-zero values are
    foreground, zero values are background. Components are computed with
    8-connectivity. Components with area less than or equal to ``min_area`` are
    removed; larger components are preserved.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional mask. Boolean masks are returned as boolean masks;
        integer masks are returned as boolean masks suitable for indexing or
        conversion back to ``uint8`` by the caller.

    min_area : int
        Maximum area, in pixels, to remove. Values less than or equal to zero
        disable cleanup and return ``mask`` unchanged.

    Returns
    -------
    np.ndarray
        Boolean keep mask when cleanup is applied, or the original ``mask`` when
        ``min_area <= 0``.
    """
    if min_area <= 0:
        return mask

    if mask.ndim != 2:
        raise ValueError('Component mask must be HxW')

    cm = (mask != 0).astype(np.uint8)
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


def remove_small_components_unless_touching(
    mask: np.ndarray,
    *,
    reference_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    Remove small foreground components unless they touch a reference mask.

    This is useful for edge/barrier cleanup after morphological strengthening:
    tiny isolated edge fragments can be discarded, while small fragments that
    connect to synthetic or trusted reference edges are preserved because they
    may help close a meaningful barrier.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or binary foreground mask whose components should be filtered.
    reference_mask : np.ndarray
        Boolean or binary mask. Components intersecting this mask are kept
        regardless of area.
    min_area : int
        Components with area less than ``min_area`` are removed unless they
        touch ``reference_mask``. Values less than or equal to zero disable
        filtering.

    Returns
    -------
    np.ndarray
        Boolean mask with selected components preserved.
    """
    if min_area <= 0:
        return mask.astype(bool)

    if mask.ndim != 2 or reference_mask.ndim != 2:
        raise ValueError('Component masks must be HxW')
    if mask.shape != reference_mask.shape:
        raise ValueError('mask and reference_mask must have the same shape')

    cm = (mask != 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return mask.astype(bool)

    reference = reference_mask.astype(bool)
    keep = np.zeros(num, dtype=bool)
    keep[0] = False

    for label in range(1, num):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area) or bool(np.any(component & reference)):
            keep[label] = True

    return keep[labels]


def labeled_points_inside_mask(
    mask: np.ndarray,
    point_coords: list[list[float]],
    point_labels: list[int],
    *,
    target_label: int = 1,
) -> tuple[int, int]:
    """
    Count labeled point prompts that fall inside a mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or binary mask in the same coordinate system as the points.
    point_coords : list[list[float]]
        Point coordinates as ``[x, y]`` pairs.
    point_labels : list[int]
        Labels aligned with ``point_coords``.
    target_label : int, default=1
        Only points with this label are counted.

    Returns
    -------
    tuple[int, int]
        ``(inside, valid)`` where ``valid`` is the number of in-bounds points
        with ``target_label`` and ``inside`` is how many of them are foreground.
    """
    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')

    inside = 0
    valid = 0
    h, w = mask.shape[:2]
    m = mask.astype(bool)

    for point, label in zip(point_coords, point_labels):
        if int(label) != int(target_label) or len(point) < 2:
            continue

        px = int(round(float(point[0])))
        py = int(round(float(point[1])))
        if not (0 <= px < w and 0 <= py < h):
            continue

        valid += 1
        if m[py, px]:
            inside += 1

    return inside, valid


def reachable_components_from_labeled_points(
    walkable: np.ndarray,
    point_coords: list[list[float]],
    point_labels: list[int],
    *,
    target_label: int = 1,
    fallback_radius: int = 2,
) -> np.ndarray:
    """
    Keep connected walkable components touched by labeled seed points.

    ``walkable`` is treated as a binary map of pixels that a flood-fill may
    traverse. The function finds connected components in that map and returns
    the union of components containing at least one seed point with
    ``target_label``. If a seed lies exactly on a barrier/non-walkable pixel, a
    small neighborhood is searched for the nearest walkable component. This is
    useful with Canny-style barriers, where an edge can run directly through a
    sparse landmark prompt.

    Parameters
    ----------
    walkable : np.ndarray
        Boolean or binary map of traversable pixels.
    point_coords : list[list[float]]
        Seed point coordinates as ``[x, y]`` pairs.
    point_labels : list[int]
        Labels aligned with ``point_coords``.
    target_label : int, default=1
        Only points with this label seed components.
    fallback_radius : int, default=2
        Pixel radius searched when a seed lands on a non-walkable pixel.

    Returns
    -------
    np.ndarray
        Boolean mask containing all selected connected components.
    """
    if walkable.ndim != 2:
        raise ValueError('walkable must be HxW')

    num, labels = cv2.connectedComponents(
        walkable.astype(np.uint8),
        connectivity=8,
    )
    if num <= 1:
        return np.zeros_like(walkable, dtype=bool)

    keep_labels: set[int] = set()
    h, w = walkable.shape[:2]
    radius = max(0, int(fallback_radius))

    for point, label in zip(point_coords, point_labels):
        if int(label) != int(target_label) or len(point) < 2:
            continue

        px = int(round(float(point[0])))
        py = int(round(float(point[1])))
        if not (0 <= px < w and 0 <= py < h):
            continue

        if walkable[py, px]:
            keep_labels.add(int(labels[py, px]))
            continue

        if radius <= 0:
            continue

        y1 = max(0, py - radius)
        y2 = min(h, py + radius + 1)
        x1 = max(0, px - radius)
        x2 = min(w, px + radius + 1)
        nearby = labels[y1:y2, x1:x2]
        nearby_walkable = nearby[nearby > 0]
        if nearby_walkable.size > 0:
            counts = np.bincount(nearby_walkable.reshape(-1))
            keep_labels.add(int(np.argmax(counts)))

    if not keep_labels:
        return np.zeros_like(walkable, dtype=bool)

    out = np.zeros_like(walkable, dtype=bool)
    for label in keep_labels:
        out |= labels == label
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest foreground connected component.
    """
    if mask.ndim != 2:
        raise ValueError('Component mask must be HxW')

    cm = (mask != 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )
    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask.astype(bool)

    target = 1 + int(np.argmax(areas))
    return labels == target


def fill_mask_holes(
    mask: np.ndarray,
    *,
    max_area: int | None,
) -> np.ndarray:
    """
    Fill background holes that do not touch the image border.
    """
    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')

    fg = mask.astype(bool)
    bg = (~fg).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        bg,
        connectivity=8,
    )
    if num <= 1:
        return fg

    h, w = fg.shape[:2]
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[h - 1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, w - 1]).tolist())

    out = fg.copy()
    for label in range(1, num):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if max_area is None or area <= int(max_area):
            out[labels == label] = True

    return out


def cleanup_shape_mask(
    mask: np.ndarray,
    *,
    fill_holes: Any = 0,
    morph_open_radius: int = 0,
    min_component_area: Any = 0,
) -> np.ndarray:
    """
    Clean a crop target shape mask before bbox/alpha/output derivation.

    Processing order:
      1. fill internal holes;
      2. apply morphological opening;
      3. remove small components or keep only the biggest component.
    """
    if mask.ndim != 2:
        raise ValueError('Mask must be HxW')
    if morph_open_radius < 0:
        raise ValueError('morph_open_radius must be >= 0')

    out = mask.astype(bool)
    h, w = out.shape[:2]

    if fill_holes is not None:
        fill_all = isinstance(fill_holes, str) and fill_holes.strip() == 'all'
        fill_area = 0
        if fill_all:
            fill_area = None
        else:
            fill_area = resolve_min_component_area(
                fill_holes,
                width=w,
                height=h,
            )
        if fill_all or int(fill_area) > 0:
            out = fill_mask_holes(out, max_area=fill_area)

    if morph_open_radius > 0:
        k = 2 * int(morph_open_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            kernel,
        ) > 0

    if isinstance(min_component_area, str) and min_component_area.strip() == 'biggest':
        out = keep_largest_component(out)
    else:
        resolved_min_area = resolve_min_component_area(
            min_component_area,
            width=w,
            height=h,
        )
        out = remove_small_components(out, min_area=resolved_min_area)

    return out.astype(bool)


def cleanup_shape_mask_by_parts(
    mask: np.ndarray,
    parts: np.ndarray | Iterable[np.ndarray] | None,
    *,
    fill_holes: Any = 0,
    morph_open_radius: int = 0,
    min_component_area: Any = 0,
) -> np.ndarray:
    """
    Clean a shape mask independently inside logical target parts.

    Composite targets such as both feet, both hands, eyes, or eyebrows may be
    represented by multiple intentionally disconnected shapes. Running
    ``min_component_area='biggest'`` after their union would keep only one
    target. This helper applies the same cleanup to each logical part first,
    then unions the cleaned parts.
    """
    if parts is None:
        return cleanup_shape_mask(
            mask,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    if mask.ndim != 2:
        raise ValueError('Shape mask must be HxW')

    shape = mask.shape
    source_masks: list[np.ndarray] = []

    if isinstance(parts, np.ndarray):
        if parts.shape != shape:
            raise ValueError('Shape part mask must match mask shape')
        pm = (parts != 0).astype(np.uint8)
        num, labels, _, _ = cv2.connectedComponentsWithStats(
            pm, connectivity=8)
        for label in range(1, num):
            source_masks.append(labels == label)
    else:
        for part in parts:
            if part.shape != shape:
                raise ValueError('Shape part mask must match mask shape')
            source_masks.append(part != 0)

    if not source_masks:
        return cleanup_shape_mask(
            mask,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    base = (mask != 0)
    out = np.zeros(shape, dtype=bool)
    hit = False

    for part_mask in source_masks:
        piece = base & part_mask
        if not np.any(piece):
            continue
        hit = True
        out |= cleanup_shape_mask(
            piece,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    if not hit:
        return cleanup_shape_mask(
            mask,
            fill_holes=fill_holes,
            morph_open_radius=morph_open_radius,
            min_component_area=min_component_area,
        )

    return out.astype(bool)


def prepare_output_mask(
    mask: np.ndarray,
    *,
    dilate_radius: int = 0,
    close_radius: int = 0,
    smoothing_radius: int = 0,
) -> np.ndarray:
    """
    Prepare a binary/soft shape mask for mask or negative-mask output.

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


def bridge_consistent_edge_endpoints(
    edges: np.ndarray,
    *,
    allowed_domain: np.ndarray,
    max_gap: float,
    tangent_radius: int = 6,
    min_facing_alignment: float = 0.70,
    min_parallelism: float = 0.65,
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

    1. skeletonize the input edge mask;
    2. find skeleton pixels with exactly one connected neighbor;
    3. estimate the outgoing tangent at each endpoint from its local component;
    4. generate endpoint pairs within ``max_gap``;
    5. reject pairs whose tangents do not face each other;
    6. reject pairs whose local contour directions are not approximately
       collinear;
    7. reject bridges that leave ``allowed_domain`` for too much of their
       length;
    8. sort accepted pairs by geometric cost and connect them greedily.

    Each endpoint is used at most once. This prevents one noisy endpoint from
    producing several artificial branches.

    Parameters
    ----------
    edges : np.ndarray
        Two-dimensional boolean or binary edge mask. Non-zero pixels are
        treated as edge pixels.
    allowed_domain : np.ndarray
        Two-dimensional boolean mask defining where synthetic bridge pixels may
        be created. This will normally be restricted to the arm-local SAM
        domain and to a maximum distance from the arm skeleton.
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
    min_parallelism : float, default=0.65
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
    The function intentionally operates on a skeletonized copy only for
    endpoint and tangent analysis. Accepted bridges are drawn onto the original
    edge mask, preserving the original edge thickness.

    This is a geometric heuristic. It cannot distinguish semantically unrelated
    contours that happen to be locally collinear, so ``allowed_domain`` should
    remain conservative.
    """
    import cv2

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

    def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
        """
        Return a one-pixel morphological skeleton using OpenCV primitives.
        """
        work = mask.astype(np.uint8) * 255
        skeleton = np.zeros_like(work)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3),
        )

        while np.any(work):
            eroded = cv2.erode(work, kernel)
            opened = cv2.dilate(eroded, kernel)
            residue = cv2.subtract(work, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            work = eroded

        return skeleton > 0

    def _neighbors(
        y: int,
        x: int,
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Return valid 8-connected foreground neighbors.
        """
        height, width = mask.shape
        result: list[tuple[int, int]] = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                ny = y + dy
                nx = x + dx

                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                ):
                    result.append((ny, nx))

        return result

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

            for neighbor in _neighbors(
                current_y,
                current_x,
                component_mask,
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

    skeleton = _morphological_skeleton(edge_mask)

    component_count, component_labels = cv2.connectedComponents(
        skeleton.astype(np.uint8),
        connectivity=8,
    )

    endpoints: list[dict[str, object]] = []

    for component_id in range(1, component_count):
        component_mask = component_labels == component_id
        ys, xs = np.where(component_mask)

        if xs.size < 3:
            continue

        for y, x in zip(ys.tolist(), xs.tolist()):
            neighbor_count = len(_neighbors(y, x, component_mask))

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


def extend_consistent_edge_endpoints(
    edges: np.ndarray,
    *,
    arm_skeleton_mask: np.ndarray,
    allowed_domain: np.ndarray,
    min_component_length: int,
    max_extension_length: float,
    tangent_radius: int = 6,
    skeleton_tangent_radius: int = 8,
    min_skeleton_parallelism: float = 0.75,
    snap_radius: int = 2,
    bridge_thickness: int = 1,
    max_component_length: Optional[int] = None,
) -> np.ndarray:
    """
    Extend reliable edge endpoints along their local outgoing tangent.

    This function complements endpoint-to-endpoint bridging. Bridging can close
    a gap only when compatible edge fragments exist on both sides. Extension
    handles cases in which a sufficiently long and coherent contour terminates
    because Canny temporarily loses the boundary, for example across a fabric
    fold, a weak shadow, or a low-contrast section of a sleeve.

    An endpoint is extended only when its outgoing contour tangent is compatible
    with the local direction of the arm skeleton. This favors longitudinal arm
    boundaries and avoids extending most transverse fabric folds or sleeve
    seams.

    The arm skeleton may bend at the elbow. For this reason, compatibility is
    evaluated against all nearby skeleton directions rather than against one
    global shoulder-to-wrist axis. The highest absolute cosine similarity is
    used as the local skeleton parallelism score.

    The processing pipeline is:

    1. skeletonize the visual edge mask;
    2. split the skeleton into connected components;
    3. retain components whose length falls inside the configured range;
    4. find component pixels with exactly one 8-connected neighbor;
    5. estimate the outgoing tangent at each endpoint;
    6. find the nearest point on the arm skeleton;
    7. compare the edge tangent with nearby local skeleton directions;
    8. extend compatible endpoints one pixel at a time;
    9. stop at the configured maximum length, the allowed-domain boundary, or
       another existing edge;
    10. snap the extension to a nearby edge when one is encountered.

    Parameters
    ----------
    edges : np.ndarray
        Two-dimensional boolean or binary visual-edge mask. Non-zero pixels are
        treated as edge pixels.

    arm_skeleton_mask : np.ndarray
        Two-dimensional boolean mask containing the local
        shoulder-to-elbow-to-wrist centerline. It should normally be one pixel
        thick, although the function tolerates a slightly thicker skeleton.

    allowed_domain : np.ndarray
        Boolean domain inside which synthetic extension pixels may be created.
        This should normally be restricted both to the permitted distance from
        the arm skeleton and to the local SAM neighborhood.

    min_component_length : int
        Minimum skeletonized component length required before one of its
        endpoints may be extended. Increasing this value prevents short noisy
        fragments from being treated as reliable contours.

    max_extension_length : float
        Maximum synthetic extension length in pixels. The extension may stop
        earlier if it leaves ``allowed_domain`` or reaches another edge.

    tangent_radius : int, default=6
        Maximum local graph distance used to estimate the outgoing tangent of
        an edge endpoint. Larger values stabilize the tangent on smooth
        contours, while smaller values follow local curvature more closely.

    skeleton_tangent_radius : int, default=8
        Euclidean neighborhood radius around the nearest arm-skeleton point
        used to derive candidate local skeleton directions. Near the elbow,
        this neighborhood may contain directions from both connected segments.

    min_skeleton_parallelism : float, default=0.75
        Minimum absolute cosine similarity between the outgoing edge tangent
        and at least one local arm-skeleton direction. Values closer to ``1.0``
        require stronger parallelism.

    snap_radius : int, default=2
        Search radius around each proposed extension point. When another edge
        is found inside this radius, the synthetic extension is connected to
        the best aligned nearby edge pixel and then stopped.

    bridge_thickness : int, default=1
        Thickness in pixels of synthetic extension segments.

    max_component_length : int or None, optional
        Optional maximum skeletonized component length eligible for extension.
        ``None`` disables the upper bound. An upper bound is usually unnecessary
        because long silhouette contours are often the most trustworthy ones.

    Returns
    -------
    np.ndarray
        Boolean edge mask containing the original edges and all accepted
        endpoint extensions.

    Raises
    ------
    ValueError
        If the input masks have incompatible shapes or dimensions, or if a
        configuration value is outside its valid range.

    Notes
    -----
    Extension follows the outgoing edge tangent, not the arm skeleton itself.
    The skeleton is used only as a plausibility test. Therefore a curved visual
    boundary can retain its locally observed direction while still being
    rejected when it is inconsistent with a longitudinal arm contour.

    This remains a geometric heuristic. A long fabric fold approximately
    parallel to the arm can still satisfy the criteria, so both
    ``min_component_length`` and ``max_extension_length`` should remain
    conservative.
    """
    import cv2

    edge_mask = np.asarray(edges).astype(bool)
    arm_skeleton = np.asarray(arm_skeleton_mask).astype(bool)
    domain_mask = np.asarray(allowed_domain).astype(bool)

    if edge_mask.ndim != 2:
        raise ValueError(
            f'edges must be two-dimensional, got {edge_mask.shape!r}.'
        )

    if arm_skeleton.ndim != 2:
        raise ValueError(
            'arm_skeleton_mask must be two-dimensional, got '
            f'{arm_skeleton.shape!r}.'
        )

    if domain_mask.ndim != 2:
        raise ValueError(
            'allowed_domain must be two-dimensional, got '
            f'{domain_mask.shape!r}.'
        )

    if not (
        edge_mask.shape
        == arm_skeleton.shape
        == domain_mask.shape
    ):
        raise ValueError(
            'edges, arm_skeleton_mask, and allowed_domain must have the '
            f'same shape, got {edge_mask.shape!r}, '
            f'{arm_skeleton.shape!r}, and {domain_mask.shape!r}.'
        )

    if min_component_length < 2:
        raise ValueError(
            'min_component_length must be >= 2, got '
            f'{min_component_length!r}.'
        )

    if (
        max_component_length is not None
        and max_component_length < min_component_length
    ):
        raise ValueError(
            'max_component_length must be None or >= '
            f'min_component_length, got {max_component_length!r}.'
        )

    if max_extension_length <= 0:
        return edge_mask.copy()

    if tangent_radius < 2:
        raise ValueError(
            f'tangent_radius must be >= 2, got {tangent_radius!r}.'
        )

    if skeleton_tangent_radius < 1:
        raise ValueError(
            'skeleton_tangent_radius must be >= 1, got '
            f'{skeleton_tangent_radius!r}.'
        )

    if not (0.0 <= min_skeleton_parallelism <= 1.0):
        raise ValueError(
            'min_skeleton_parallelism must be in [0, 1], got '
            f'{min_skeleton_parallelism!r}.'
        )

    if snap_radius < 0:
        raise ValueError(
            f'snap_radius must be >= 0, got {snap_radius!r}.'
        )

    if bridge_thickness < 1:
        raise ValueError(
            f'bridge_thickness must be >= 1, got {bridge_thickness!r}.'
        )

    if not np.any(edge_mask) or not np.any(arm_skeleton):
        return edge_mask.copy()

    height, width = edge_mask.shape

    def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
        """
        Return a one-pixel morphological skeleton using OpenCV primitives.
        """
        work = mask.astype(np.uint8) * 255
        skeleton = np.zeros_like(work)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3),
        )

        while np.any(work):
            eroded = cv2.erode(work, kernel)
            opened = cv2.dilate(eroded, kernel)
            residue = cv2.subtract(work, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            work = eroded

        return skeleton > 0

    def _neighbors(
        y: int,
        x: int,
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Return foreground pixels in the 8-connected neighborhood.
        """
        result: list[tuple[int, int]] = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                ny = y + dy
                nx = x + dx

                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                ):
                    result.append((ny, nx))

        return result

    def _estimate_outgoing_tangent(
        endpoint_y: int,
        endpoint_x: int,
        component_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate the unit tangent pointing away from an edge component.

        The local component is followed inward from the endpoint. The inverse
        endpoint-to-interior direction is returned as the outgoing tangent.
        """
        start = (endpoint_y, endpoint_x)
        queue: list[tuple[tuple[int, int], int]] = [(start, 0)]
        visited = {start}

        local_points: list[tuple[int, int, int]] = []

        queue_index = 0
        while queue_index < len(queue):
            (current_y, current_x), graph_distance = queue[queue_index]
            queue_index += 1

            local_points.append((
                current_y,
                current_x,
                graph_distance,
            ))

            if graph_distance >= tangent_radius:
                continue

            for neighbor in _neighbors(
                current_y,
                current_x,
                component_mask,
            ):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append((
                    neighbor,
                    graph_distance + 1,
                ))

        sufficiently_distant = [
            item
            for item in local_points
            if item[2] >= 2
        ]

        if not sufficiently_distant:
            return None

        farthest_y, farthest_x, _ = max(
            sufficiently_distant,
            key=lambda item: item[2],
        )

        inward = np.asarray(
            [
                float(farthest_x - endpoint_x),
                float(farthest_y - endpoint_y),
            ],
            dtype=np.float32,
        )

        length = float(np.linalg.norm(inward))
        if length < 1e-6:
            return None

        return -inward / length

    arm_ys, arm_xs = np.where(arm_skeleton)

    arm_points_xy = np.stack(
        [
            arm_xs.astype(np.float32),
            arm_ys.astype(np.float32),
        ],
        axis=1,
    )

    def _nearest_skeleton_index(point_xy: np.ndarray) -> int:
        """
        Return the index of the nearest arm-skeleton pixel.
        """
        delta = arm_points_xy - point_xy[None, :]
        distance_sq = np.sum(delta * delta, axis=1)
        return int(np.argmin(distance_sq))

    def _local_skeleton_parallelism(
        point_xy: np.ndarray,
        edge_tangent: np.ndarray,
    ) -> float:
        """
        Return the strongest local parallelism with the arm skeleton.

        Nearby arm-skeleton vectors are evaluated independently. This allows an
        endpoint close to the elbow to match either the upper-arm direction or
        the forearm direction.
        """
        nearest_idx = _nearest_skeleton_index(point_xy)
        center = arm_points_xy[nearest_idx]

        delta = arm_points_xy - center[None, :]
        distance = np.linalg.norm(delta, axis=1)

        local_mask = (
            (distance >= 2.0)
            & (
                distance
                <= float(skeleton_tangent_radius)
            )
        )

        local_vectors = delta[local_mask]

        if local_vectors.shape[0] == 0:
            return 0.0

        lengths = np.linalg.norm(
            local_vectors,
            axis=1,
            keepdims=True,
        )

        valid = lengths[:, 0] > 1e-6
        local_vectors = local_vectors[valid]
        lengths = lengths[valid]

        if local_vectors.shape[0] == 0:
            return 0.0

        local_directions = local_vectors / lengths

        similarities = np.abs(
            local_directions @ edge_tangent
        )

        return float(np.max(similarities))

    def _find_snap_target(
        current_xy: np.ndarray,
        outgoing_tangent: np.ndarray,
        *,
        original_component_mask: np.ndarray,
        result_mask: np.ndarray,
    ) -> Optional[tuple[int, int]]:
        """
        Find a nearby edge pixel compatible with the extension direction.
        """
        if snap_radius <= 0:
            return None

        center_x = int(round(float(current_xy[0])))
        center_y = int(round(float(current_xy[1])))

        best_target: Optional[tuple[int, int]] = None
        best_score: Optional[tuple[float, float]] = None

        for dy in range(-snap_radius, snap_radius + 1):
            for dx in range(-snap_radius, snap_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                target_x = center_x + dx
                target_y = center_y + dy

                if not (
                    0 <= target_x < width
                    and 0 <= target_y < height
                ):
                    continue

                if not result_mask[target_y, target_x]:
                    continue

                # Ignore pixels belonging to the local source component. Without
                # this guard the extension could immediately snap backward onto
                # the contour from which it originates.
                if original_component_mask[target_y, target_x]:
                    continue

                vector = np.asarray(
                    [float(dx), float(dy)],
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(vector))

                if distance < 1e-6:
                    continue

                direction = vector / distance
                forward_alignment = float(
                    np.dot(outgoing_tangent, direction)
                )

                if forward_alignment <= 0.0:
                    continue

                score = (
                    forward_alignment,
                    -distance,
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_target = (
                        target_x,
                        target_y,
                    )

        return best_target

    edge_skeleton = _morphological_skeleton(edge_mask)

    component_count, component_labels = cv2.connectedComponents(
        edge_skeleton.astype(np.uint8),
        connectivity=8,
    )

    endpoint_candidates: list[
        tuple[
            int,
            int,
            int,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    for component_id in range(1, component_count):
        component_mask = component_labels == component_id
        component_length = int(np.count_nonzero(component_mask))

        if component_length < min_component_length:
            continue

        if (
            max_component_length is not None
            and component_length > max_component_length
        ):
            continue

        component_ys, component_xs = np.where(component_mask)

        for endpoint_y, endpoint_x in zip(
            component_ys.tolist(),
            component_xs.tolist(),
        ):
            if len(_neighbors(
                endpoint_y,
                endpoint_x,
                component_mask,
            )) != 1:
                continue

            outgoing_tangent = _estimate_outgoing_tangent(
                endpoint_y,
                endpoint_x,
                component_mask,
            )

            if outgoing_tangent is None:
                continue

            endpoint_xy = np.asarray(
                [
                    float(endpoint_x),
                    float(endpoint_y),
                ],
                dtype=np.float32,
            )

            skeleton_parallelism = (
                _local_skeleton_parallelism(
                    endpoint_xy,
                    outgoing_tangent,
                )
            )

            if (
                skeleton_parallelism
                < min_skeleton_parallelism
            ):
                continue

            endpoint_candidates.append((
                component_length,
                component_id,
                endpoint_x,
                endpoint_y,
                outgoing_tangent,
            ))

    if not endpoint_candidates:
        return edge_mask.copy()

    # Process the longest observed contours first. When extensions compete for
    # the same area, the most reliable component therefore receives priority.
    endpoint_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    result_u8 = edge_mask.astype(np.uint8) * 255
    result_mask = result_u8 > 0

    max_steps = max(
        1,
        int(math.ceil(float(max_extension_length))),
    )

    for (
        _,
        component_id,
        endpoint_x,
        endpoint_y,
        outgoing_tangent,
    ) in endpoint_candidates:
        component_mask = (
            component_labels == component_id
        )

        extension_points: list[tuple[int, int]] = [
            (endpoint_x, endpoint_y)
        ]

        previous_pixel = (
            endpoint_x,
            endpoint_y,
        )

        snap_target: Optional[tuple[int, int]] = None

        for step in range(1, max_steps + 1):
            current_xy = np.asarray(
                [
                    float(endpoint_x)
                    + float(outgoing_tangent[0]) * step,
                    float(endpoint_y)
                    + float(outgoing_tangent[1]) * step,
                ],
                dtype=np.float32,
            )

            current_x = int(round(float(current_xy[0])))
            current_y = int(round(float(current_xy[1])))

            if not (
                0 <= current_x < width
                and 0 <= current_y < height
            ):
                break

            current_pixel = (
                current_x,
                current_y,
            )

            if current_pixel == previous_pixel:
                continue

            previous_pixel = current_pixel

            if not domain_mask[current_y, current_x]:
                break

            found_target = _find_snap_target(
                current_xy,
                outgoing_tangent,
                original_component_mask=component_mask,
                result_mask=result_mask,
            )

            if found_target is not None:
                snap_target = found_target
                extension_points.append(found_target)
                break

            extension_points.append(current_pixel)

        if len(extension_points) < 2:
            continue

        extension_u8 = np.zeros_like(result_u8)

        for start, end in zip(
            extension_points,
            extension_points[1:],
        ):
            cv2.line(
                extension_u8,
                start,
                end,
                255,
                thickness=bridge_thickness,
                lineType=cv2.LINE_8,
            )

        # Synthetic pixels may never escape the conservative extension domain.
        extension_u8[~domain_mask] = 0

        if not np.any(extension_u8):
            continue

        result_u8 = cv2.bitwise_or(
            result_u8,
            extension_u8,
        )
        result_mask = result_u8 > 0

        if snap_target is not None:
            cv2.line(
                result_u8,
                extension_points[-2],
                snap_target,
                255,
                thickness=bridge_thickness,
                lineType=cv2.LINE_8,
            )
            result_mask = result_u8 > 0

    return result_u8 > 0
