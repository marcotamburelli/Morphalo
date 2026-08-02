import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
)

FOOT_POINTS = (
    ('L knee', 25, (255, 128, 0)),
    ('L ankle', 27, (255, 128, 0)),
    ('L heel', 29, (255, 128, 0)),
    ('L index', 31, (255, 128, 0)),
    ('R knee', 26, (0, 128, 255)),
    ('R ankle', 28, (0, 128, 255)),
    ('R heel', 30, (0, 128, 255)),
    ('R index', 32, (0, 128, 255)),
)


@dataclass(frozen=True)
class CropDebugRegion:
    """
    Target-local debug region with optional prompt/probe points.

    ``point_coords`` stores full-frame ``[x, y]`` points associated with one
    local crop region. ``point_labels`` is aligned with ``point_coords`` and
    follows the SAM-style convention used by the debug overlay: ``1`` marks a
    positive point that belongs to the target, while ``0`` marks a negative
    point that should be excluded.

    ``point_sources`` is optional metadata aligned with ``point_coords``. It
    describes why each point exists, for example ``landmark`` for a direct pose
    point, ``midpoint`` for a synthetic point between landmarks, ``chromatic``
    for the center of an image-color run, or ``fallback`` for a seed recovered
    from the available mask.

    ``probe_point`` is deliberately separate from ``point_coords`` because it
    represents a diagnostic/test point, not necessarily a final positive or
    negative point used by the crop logic.
    """
    side: str
    point_coords: Optional[list[list[float]]] = None
    point_labels: Optional[list[int]] = None
    point_sources: Optional[list[str]] = None
    probe_point: Optional[list[float]] = None


@dataclass(frozen=True)
class CropDebugMaskOverlay:
    """
    One semi-transparent full-frame mask overlay for debug rendering.
    """
    label: str
    mask: np.ndarray
    color: tuple[int, int, int]
    alpha: float = 0.35


@dataclass(frozen=True)
class CropDebugEdgeOverlay:
    """
    One opaque full-frame sparse edge/barrier overlay.
    """
    label: str
    mask: np.ndarray
    color: tuple[int, int, int]


@dataclass(frozen=True)
class LimbTopologyDebugImage:
    """
    Render specification for one limb-topology debug image.

    ``region_mosaic`` and ``endpoint_debug`` are local crop products. When they
    are supplied, ``crop_bbox`` is used to translate them into full-frame
    coordinates before drawing.
    """
    filename: str
    title: str
    crop_bbox: Optional[tuple[int, int, int, int]] = None
    mask_overlays: list[CropDebugMaskOverlay] = field(default_factory=list)
    edge_overlays: list[CropDebugEdgeOverlay] = field(default_factory=list)
    region_mosaic: Any = None
    selected_region_labels: frozenset[int] = frozenset()
    endpoint_debug: Any = None


def _draw_label(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    import cv2

    cv2.putText(
        img,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _pose_point(pose_xy: np.ndarray, idx: int) -> Optional[tuple[int, int]]:
    if idx >= pose_xy.shape[0]:
        return None

    px, py = pose_xy[idx]
    if px < 0 or py < 0:
        return None

    return int(px), int(py)


def _draw_pose_landmarks(img: np.ndarray, pose_xy: np.ndarray) -> None:
    """
    Draw all valid MediaPipe pose landmarks and a lightweight skeleton.
    """
    import cv2

    skeleton_color = (80, 220, 80)
    point_color = (40, 255, 40)

    for i, j in POSE_EDGES:
        p1 = _pose_point(pose_xy, i)
        p2 = _pose_point(pose_xy, j)
        if p1 is not None and p2 is not None:
            cv2.line(img, p1, p2, skeleton_color, 1, cv2.LINE_AA)

    for idx in range(pose_xy.shape[0]):
        p = _pose_point(pose_xy, idx)
        if p is None:
            continue
        cv2.circle(img, p, 3, point_color, -1)


def _draw_foot_landmarks(img: np.ndarray, pose_xy: np.ndarray) -> None:
    """
    Draw foot landmarks with labels and stronger chain lines.
    """
    import cv2

    for label, idx, color in FOOT_POINTS:
        p = _pose_point(pose_xy, idx)
        if p is None:
            continue

        cv2.circle(img, p, 5, color, -1)
        cv2.circle(img, p, 7, (255, 255, 255), 1)
        _draw_label(
            img,
            label,
            (p[0] + 6, p[1] - 6),
            color,
            scale=0.45,
            thickness=1,
        )

    for idxs, color in (
        ((25, 27, 29, 31), (255, 128, 0)),
        ((26, 28, 30, 32), (0, 128, 255)),
    ):
        chain = [_pose_point(pose_xy, idx) for idx in idxs]
        chain = [p for p in chain if p is not None]
        for p1, p2 in zip(chain, chain[1:]):
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)


def _draw_crop_regions(
    img: np.ndarray,
    crop_regions: Sequence[CropDebugRegion],
) -> None:
    """
    Draw target-local crop diagnostics.

    Each region can expose positive/negative points, optional point sources,
    and a probe point. The same small contract is used by SAM-backed crops and
    by semantic-prior limb crops, so the debug overlay can show the geometry
    that drove the final local mask without caring about the segmentation
    backend.
    """
    import cv2

    if not crop_regions:
        return

    for region in crop_regions:
        point_coords = getattr(region, 'point_coords', None) or []
        point_labels = getattr(region, 'point_labels', None) or []
        point_sources = getattr(region, 'point_sources', None) or []

        for idx, (point, label) in enumerate(zip(point_coords, point_labels)):
            px = int(round(float(point[0])))
            py = int(round(float(point[1])))
            is_positive = int(label) == 1
            source = (
                str(point_sources[idx])
                if idx < len(point_sources)
                else ''
            )
            if source == 'chromatic':
                color = (255, 255, 0)
                text = 'chr +'
            elif source == 'landmark':
                color = (0, 255, 0)
                text = 'lm +'
            elif source == 'midpoint':
                color = (0, 255, 0)
                text = 'mid +'
            elif source == 'fallback':
                color = (0, 255, 0)
                text = 'fb +'
            else:
                color = (0, 255, 0) if is_positive else (255, 64, 64)
                text = 'pt +' if is_positive else 'pt -'

            cv2.circle(img, (px, py), 5, color, -1)
            cv2.circle(img, (px, py), 7, (0, 0, 0), 1)
            _draw_label(
                img,
                text,
                (px + 7, py + 14),
                color,
                scale=0.4,
                thickness=1,
            )

        probe = getattr(region, 'probe_point', None)
        if probe is None:
            continue

        px = int(round(float(probe[0])))
        py = int(round(float(probe[1])))
        color = (255, 255, 0)

        cv2.circle(img, (px, py), 6, color, -1)
        cv2.circle(img, (px, py), 8, (0, 0, 0), 1)
        _draw_label(
            img,
            'probe',
            (px + 7, py - 7),
            color,
            scale=0.45,
            thickness=1,
        )


def _draw_hand_landmarks(img: np.ndarray, hands_res: Any) -> None:
    """
    Draw MediaPipe hand landmarks when a hand crop computed them.
    """
    import cv2

    xy = getattr(hands_res, 'xy', None)
    if xy is None:
        return

    hand_edges = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    )
    colors = ((255, 0, 255), (255, 180, 0), (0, 220, 255))

    for hand_idx in range(xy.shape[0]):
        color = colors[hand_idx % len(colors)]
        pts = []

        for idx in range(xy.shape[1]):
            px, py = xy[hand_idx, idx]
            if px < 0 or py < 0:
                pts.append(None)
                continue
            p = (int(px), int(py))
            pts.append(p)
            cv2.circle(img, p, 3, color, -1)

        for i, j in hand_edges:
            if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                cv2.line(img, pts[i], pts[j], color, 1, cv2.LINE_AA)


def _draw_face_landmarks(img: np.ndarray, face_xy: Optional[np.ndarray]) -> None:
    """
    Draw face landmarks when a head crop computed them.
    """
    import cv2

    if face_xy is None:
        return

    for px, py in face_xy:
        if px < 0 or py < 0:
            continue
        cv2.circle(img, (int(px), int(py)), 1, (255, 255, 0), -1)


def _draw_mask_overlay(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    color: tuple[int, int, int] = (255, 0, 255),
    alpha: float = 0.25,
) -> None:
    """
    Overlay a boolean or uint8 mask on an RGB debug image.
    """
    if mask is None:
        return

    m = mask.astype(bool)
    if m.shape[:2] != img.shape[:2]:
        return

    overlay = np.zeros_like(img)
    overlay[m] = color
    img[m] = (
        (1.0 - alpha) * img[m].astype(np.float32) +
        alpha * overlay[m].astype(np.float32)
    ).astype(np.uint8)


def _draw_region_mask_contours(
    img: np.ndarray,
    region_masks: Any,
    *,
    color: tuple[int, int, int] = (255, 128, 0),
) -> None:
    """
    Draw prior/ROI mask contours without filling the debug image.

    Parameters
    ----------
    img : np.ndarray
        RGB debug image modified in place.
    region_masks : Any
        Iterable of full-frame boolean/binary masks. Masks with mismatched
        shape are ignored.
    color : tuple[int, int, int], default=(255, 128, 0)
        RGB contour color.

    Returns
    -------
    None
    """
    import cv2

    if not region_masks:
        return

    for region_mask in region_masks:
        if region_mask is None:
            continue
        if region_mask.shape[:2] != img.shape[:2]:
            continue

        m = region_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            m,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(img, contours, -1, color, 2, cv2.LINE_AA)


def _draw_edge_masks(
    img: np.ndarray,
    edge_masks: Any,
    *,
    color: tuple[int, int, int] = (0, 180, 255),
) -> None:
    """
    Draw sparse edge/barrier masks as colored pixels.

    Parameters
    ----------
    img : np.ndarray
        RGB debug image modified in place.
    edge_masks : Any
        Iterable of full-frame boolean/binary edge masks. Masks with mismatched
        shape are ignored.
    color : tuple[int, int, int], default=(0, 180, 255)
        RGB color used for edge pixels.

    Returns
    -------
    None
    """
    if not edge_masks:
        return

    for edge_mask in edge_masks:
        if edge_mask is None:
            continue
        if edge_mask.shape[:2] != img.shape[:2]:
            continue

        m = edge_mask.astype(bool)
        img[m] = color


def _darken_debug_base(img_rgb: np.ndarray, factor: float = 0.38) -> np.ndarray:
    """
    Return a darkened RGB base image for high-contrast overlays.
    """
    return np.clip(
        img_rgb.astype(np.float32) * float(factor),
        0,
        255,
    ).astype(np.uint8)


def _debug_region_color(label: int) -> tuple[int, int, int]:
    """
    Derive a stable, readable pseudo-random RGB color for a region label.
    """
    value = int(label) * 1103515245 + 12345
    return (
        70 + ((value >> 0) & 0x7F),
        70 + ((value >> 8) & 0x7F),
        70 + ((value >> 16) & 0x7F),
    )


def _draw_region_mosaic_overlay(
    img: np.ndarray,
    *,
    region_mosaic: Any,
    crop_bbox: tuple[int, int, int, int],
    selected_region_labels: frozenset[int],
) -> None:
    """
    Draw a local ``RegionMosaic`` over a full-frame debug image.
    """
    if region_mosaic is None:
        return

    labels = getattr(region_mosaic, 'labels', None)
    if labels is None:
        return

    labels = np.asarray(labels)
    if labels.ndim != 2:
        return

    x1, y1, x2, y2 = crop_bbox
    local_h = min(labels.shape[0], max(0, y2 - y1))
    local_w = min(labels.shape[1], max(0, x2 - x1))
    if local_h <= 0 or local_w <= 0:
        return

    target = img[y1:y1 + local_h, x1:x1 + local_w]
    local_labels = labels[:local_h, :local_w]
    selected = set(int(label) for label in selected_region_labels)

    for label in np.unique(local_labels):
        label = int(label)
        if label <= 0:
            continue

        region = local_labels == label
        if not np.any(region):
            continue

        if label in selected:
            color = (45, 220, 95)
            alpha = 0.48
        else:
            color = _debug_region_color(label)
            alpha = 0.22

        overlay = np.zeros_like(target)
        overlay[region] = color
        target[region] = (
            (1.0 - alpha) * target[region].astype(np.float32)
            + alpha * overlay[region].astype(np.float32)
        ).astype(np.uint8)


def _draw_limb_endpoint_debug(
    img: np.ndarray,
    *,
    endpoint_debug: Any,
    crop_bbox: tuple[int, int, int, int],
) -> None:
    """
    Draw Canny endpoint marks, accepted segments, and rejection marks.
    """
    import cv2

    if endpoint_debug is None:
        return

    x_offset, y_offset, _, _ = crop_bbox

    for segment in getattr(endpoint_debug, 'accepted_segments', []) or []:
        start = getattr(segment, 'start_xy', None)
        end = getattr(segment, 'end_xy', None)
        if start is None or end is None:
            continue
        color = tuple(getattr(segment, 'color', (0, 180, 255)))
        cv2.line(
            img,
            (int(start[0]) + x_offset, int(start[1]) + y_offset),
            (int(end[0]) + x_offset, int(end[1]) + y_offset),
            color,
            1,
            cv2.LINE_8,
        )

    for mark in getattr(endpoint_debug, 'endpoint_marks', []) or []:
        x = int(getattr(mark, 'x')) + x_offset
        y = int(getattr(mark, 'y')) + y_offset
        color = tuple(getattr(mark, 'color', (255, 255, 255)))
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            img[y, x] = color

    for mark in getattr(endpoint_debug, 'rejection_marks', []) or []:
        if len(mark) < 3:
            continue
        x = int(mark[0]) + x_offset
        y = int(mark[1]) + y_offset
        color = tuple(mark[2])
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            img[y, x] = color


def _limb_endpoint_legend_rows(
    endpoint_debug: Any,
) -> list[tuple[str, tuple[int, int, int]]]:
    """
    Return compact legend rows for endpoint debug marker colors.
    """
    if endpoint_debug is None:
        return []

    return [
        ('dot raw endpoint', (255, 255, 255)),
        ('dot candidate endpoint', (255, 0, 255)),
        ('dot short branch endpoint', (0, 0, 255)),
        ('line accepted bridge', (0, 128, 255)),
        ('line accepted prolongation', (255, 128, 0)),
        ('dot rejected: direction', (0, 215, 255)),
        ('dot rejected: domain', (0, 255, 0)),
        ('dot rejected: crossing', (255, 0, 0)),
    ]


def _draw_debug_legend(
    img: np.ndarray,
    rows: Sequence[tuple[str, tuple[int, int, int]]],
) -> None:
    """
    Draw a compact legend in the upper-left corner.
    """
    import cv2

    if not rows:
        return

    x = 8
    y = 18
    row_height = 16
    for label, color in rows:
        cv2.rectangle(
            img,
            (x, y - 9),
            (x + 9, y),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        _draw_label(
            img,
            label,
            (x + 15, y),
            (235, 235, 235),
            scale=0.38,
            thickness=1,
        )
        y += row_height


def write_limb_topology_debug_directory(
    *,
    img_rgb: np.ndarray,
    out_path: Path,
    images: Sequence[LimbTopologyDebugImage],
    overview_path: Optional[Path] = None,
) -> Path:
    """
    Write a directory of limb-topology debug overlays.

    Each image is rendered over a darkened copy of the original RGB frame.
    Masks are alpha-blended, sparse edge/barrier masks are drawn opaquely, local
    region mosaics are translated through their crop bbox, and Canny endpoint
    debug marks are drawn with their existing classification colors. When an
    overview image is supplied, it is moved into the directory as
    ``00_overview.png`` so callers can expose the directory as the single debug
    artifact.
    """
    import cv2

    debug_dir = out_path.with_name(out_path.stem + '_debug')
    debug_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    if overview_path is not None:
        overview_debug_path = debug_dir / '00_overview.png'
        if overview_path != overview_debug_path:
            overview_path.replace(overview_debug_path)
        manifest.append({
            'filename': overview_debug_path.name,
            'title': 'overview',
        })

    image_start_index = 1 if overview_path is not None else 0
    for index, spec in enumerate(images, start=image_start_index):
        dbg = _darken_debug_base(img_rgb)
        legend_rows: list[tuple[str, tuple[int, int, int]]] = []

        if spec.region_mosaic is not None and spec.crop_bbox is not None:
            _draw_region_mosaic_overlay(
                dbg,
                region_mosaic=spec.region_mosaic,
                crop_bbox=spec.crop_bbox,
                selected_region_labels=spec.selected_region_labels,
            )
            legend_rows.append(('selected region', (45, 220, 95)))
            legend_rows.append(('other region', (120, 120, 120)))

        for overlay in spec.mask_overlays:
            _draw_mask_overlay(
                dbg,
                overlay.mask,
                color=overlay.color,
                alpha=overlay.alpha,
            )
            legend_rows.append((overlay.label, overlay.color))

        for overlay in spec.edge_overlays:
            _draw_edge_masks(
                dbg,
                [overlay.mask],
                color=overlay.color,
            )
            legend_rows.append((overlay.label, overlay.color))

        if spec.endpoint_debug is not None and spec.crop_bbox is not None:
            _draw_limb_endpoint_debug(
                dbg,
                endpoint_debug=spec.endpoint_debug,
                crop_bbox=spec.crop_bbox,
            )
            legend_rows.extend(_limb_endpoint_legend_rows(
                spec.endpoint_debug,
            ))

        if spec.crop_bbox is not None:
            x1, y1, x2, y2 = spec.crop_bbox
            cv2.rectangle(
                dbg,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        _draw_label(
            dbg,
            spec.title,
            (8, max(16, img_rgb.shape[0] - 10)),
            (255, 255, 255),
            scale=0.46,
            thickness=1,
        )
        _draw_debug_legend(dbg, legend_rows[:14])

        filename = f'{index:02d}_{spec.filename}'
        Image.fromarray(dbg).save(debug_dir / filename)
        manifest.append({
            'filename': filename,
            'title': spec.title,
        })

    with (debug_dir / 'debug.json').open('w', encoding='utf-8') as handle:
        json.dump(
            {
                'overview': (
                    None if overview_path is None else '00_overview.png'
                ),
                'images': manifest,
            },
            handle,
            indent=2,
        )

    return debug_dir


def write_crop_debug_overlay(
    *,
    img_rgb: np.ndarray,
    out_path: Path,
    target: str,
    pose_xy: np.ndarray,
    crop_prompt_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    hands_res: Any = None,
    face_xy: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    crop_regions: Optional[Sequence[CropDebugRegion]] = None,
    prompt_bbox_label: str = 'person',
    region_masks: Any = None,
    edge_masks: Any = None,
) -> Path:
    """
    Write a crop debug image with bboxes, pose, and optional local landmarks.

    The overlay always includes all valid pose landmarks. It additionally draws
    hand landmarks, face landmarks, target-local crop regions, or mask overlays
    when the caller provides those data. ``crop_prompt_bbox`` is the yellow/cyan
    prompt/search box; for non-person targets callers can override
    ``prompt_bbox_label`` to describe the actual prompt domain.
    """
    import cv2

    dbg = img_rgb.copy()
    bx1, by1, bx2, by2 = crop_prompt_bbox
    tx1, ty1, tx2, ty2 = target_bbox

    _draw_mask_overlay(dbg, mask)
    _draw_region_mask_contours(dbg, region_masks)
    _draw_edge_masks(dbg, edge_masks)
    _draw_pose_landmarks(dbg, pose_xy)

    if target in ('feet', 'left-foot', 'right-foot'):
        _draw_foot_landmarks(dbg, pose_xy)

    if crop_regions:
        _draw_crop_regions(dbg, crop_regions)

    if hands_res is not None:
        _draw_hand_landmarks(dbg, hands_res)
    if face_xy is not None:
        _draw_face_landmarks(dbg, face_xy)

    cv2.rectangle(dbg, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
    cv2.rectangle(dbg, (tx1, ty1), (tx2, ty2), (255, 0, 0), 3)

    _draw_label(
        dbg,
        prompt_bbox_label,
        (bx1 + 4, max(12, by1 - 6)),
        (0, 255, 255),
    )
    _draw_label(dbg, target, (tx1 + 4, max(12, ty1 - 6)), (255, 0, 0))

    dbg_path = out_path.with_name(out_path.stem + '_debug_bbox.png')
    Image.fromarray(dbg).save(dbg_path)
    return dbg_path


def write_mask_debug_overlay(
    *,
    img_rgb: np.ndarray,
    out_path: Path,
    target: str,
    mask: np.ndarray,
    target_bbox: tuple[int, int, int, int],
) -> Path:
    """
    Write a lightweight debug overlay for mask-derived crops.
    """
    import cv2

    dbg = img_rgb.copy()
    tx1, ty1, tx2, ty2 = target_bbox

    _draw_mask_overlay(dbg, mask)
    cv2.rectangle(dbg, (tx1, ty1), (tx2 - 1, ty2 - 1), (255, 0, 0), 3)
    _draw_label(dbg, target, (tx1 + 4, max(12, ty1 - 6)), (255, 0, 0))

    dbg_path = out_path.with_name(out_path.stem + '_debug_bbox.png')
    Image.fromarray(dbg).save(dbg_path)
    return dbg_path
