from pathlib import Path
from typing import Any, Optional

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


def write_crop_debug_overlay(
    *,
    img_rgb: np.ndarray,
    out_path: Path,
    target: str,
    pose_xy: np.ndarray,
    person_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    hands_res: Any = None,
    face_xy: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> Path:
    """
    Write a crop debug image with bboxes, pose, and optional local landmarks.

    The overlay always includes all valid pose landmarks. It additionally draws
    hand landmarks, face landmarks, or mask overlays when the caller provides
    those data.
    """
    import cv2

    dbg = img_rgb.copy()
    bx1, by1, bx2, by2 = person_bbox
    tx1, ty1, tx2, ty2 = target_bbox

    _draw_mask_overlay(dbg, mask)
    _draw_pose_landmarks(dbg, pose_xy)

    if target in ('feet', 'left-foot', 'right-foot'):
        _draw_foot_landmarks(dbg, pose_xy)
    if hands_res is not None:
        _draw_hand_landmarks(dbg, hands_res)
    if face_xy is not None:
        _draw_face_landmarks(dbg, face_xy)

    cv2.rectangle(dbg, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
    cv2.rectangle(dbg, (tx1, ty1), (tx2, ty2), (255, 0, 0), 3)

    _draw_label(dbg, 'person', (bx1 + 4, max(12, by1 - 6)), (0, 255, 255))
    _draw_label(dbg, target, (tx1 + 4, max(12, ty1 - 6)), (255, 0, 0))

    dbg_path = out_path.with_name(out_path.stem + '_debug_bbox.png')
    Image.fromarray(dbg).save(dbg_path)
    return dbg_path
