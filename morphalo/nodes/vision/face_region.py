import math
from typing import Any

import numpy as np

from morphalo.nodes.vision.human import (face_bbox_xyxy_from_landmarks,
                                         mp_face_landmarks)


def head_area_xyxy_from_pose(
    pose_xy: np.ndarray,
    *,
    full_w: int,
    full_h: int,
    expansion: float = 1.6,
) -> tuple[int, int, int, int]:
    """
    Derive a pose-guided upper-body area suitable for downstream face detection.

    Despite the historical function name, the returned region is intentionally
    larger than a tight head crop. When torso landmarks are available, the box
    includes the head, neck, shoulders, and as much upper body as possible,
    typically down to the hips. This preserves context that can help face
    detection on difficult images.

    Parameters
    ----------
    pose_xy : np.ndarray
        Pose landmarks with shape ``(33, 2)`` in full-image coordinates.
        Missing landmarks must be encoded as ``(-1, -1)``.
    full_w : int
        Full image width.
    full_h : int
        Full image height.
    expansion : float, optional
        Expansion multiplier applied to the estimated region size.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box ``(x1, y1, x2, y2)`` in full-image coordinates.

    Raises
    ------
    RuntimeError
        If the pose does not provide enough information to estimate a plausible
        head / upper-body area.
    """
    def _valid(i: int) -> bool:
        return (
            0 <= i < pose_xy.shape[0]
            and pose_xy[i, 0] >= 0
            and pose_xy[i, 1] >= 0
        )

    def _clip_box(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[int, int, int, int]:
        ix1 = max(0, min(full_w, int(math.floor(x1))))
        iy1 = max(0, min(full_h, int(math.floor(y1))))
        ix2 = max(0, min(full_w, int(math.ceil(x2))))
        iy2 = max(0, min(full_h, int(math.ceil(y2))))

        if ix2 <= ix1 or iy2 <= iy1:
            raise RuntimeError('Invalid pose-derived head area.')

        return ix1, iy1, ix2, iy2

    face_ids = [0, 2, 5, 7, 8]  # nose, eyes, ears
    face_pts = [
        pose_xy[i].astype(np.float32)
        for i in face_ids
        if _valid(i)
    ]

    has_torso = all(_valid(i) for i in [11, 12, 23, 24])

    # ------------------------------------------------------------------
    # Best case: torso available -> build a generous upper-body region.
    # ------------------------------------------------------------------
    if has_torso:
        ls = pose_xy[11].astype(np.float32)
        rs = pose_xy[12].astype(np.float32)
        lh = pose_xy[23].astype(np.float32)
        rh = pose_xy[24].astype(np.float32)

        shoulder_center = 0.5 * (ls + rs)
        hip_center = 0.5 * (lh + rh)

        torso_vec = shoulder_center - hip_center
        torso_len = float(np.linalg.norm(torso_vec))
        if torso_len < 1.0:
            raise RuntimeError(
                'Cannot derive head area from pose: torso too small.'
            )

        head_dir = torso_vec / torso_len
        shoulder_width = float(np.linalg.norm(ls - rs))
        hip_width = float(np.linalg.norm(lh - rh))

        # Top anchor: use face center if available, otherwise project above shoulders.
        if face_pts:
            face_pts_arr = np.stack(face_pts, axis=0)
            face_center = np.mean(face_pts_arr, axis=0)
            top_center = 0.7 * face_center + 0.3 * (
                shoulder_center + head_dir * (0.35 * torso_len)
            )

            face_y_min = float(np.min(face_pts_arr[:, 1]))
            top_y = min(face_y_min - 0.35 * shoulder_width,
                        top_center[1] - 0.8 * shoulder_width)
            x_center = float(top_center[0])
        else:
            top_center = shoulder_center + head_dir * (0.55 * torso_len)
            top_y = float(top_center[1] - 0.6 * shoulder_width)
            x_center = float(top_center[0])

        # Bottom: include hips and a little extra below.
        bottom_center = hip_center
        bottom_y = float(bottom_center[1] + 0.25 * torso_len)

        # Width: wide enough for head + shoulders + some torso context.
        region_width = max(
            1.35 * shoulder_width,
            1.15 * hip_width,
            0.9 * torso_len,
        ) * float(expansion)

        half_w = max(12.0, 0.5 * region_width)

        x1 = x_center - half_w
        x2 = x_center + half_w
        y1 = top_y
        y2 = bottom_y

        return _clip_box(x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Fallback: face landmarks only -> build a looser face-centered region.
    # ------------------------------------------------------------------
    if len(face_pts) >= 2:
        face_pts_arr = np.stack(face_pts, axis=0)

        fx_min = float(np.min(face_pts_arr[:, 0]))
        fy_min = float(np.min(face_pts_arr[:, 1]))
        fx_max = float(np.max(face_pts_arr[:, 0]))
        fy_max = float(np.max(face_pts_arr[:, 1]))

        face_w = max(1.0, fx_max - fx_min)
        face_h = max(1.0, fy_max - fy_min)
        face_span = max(face_w, face_h)

        cx = 0.5 * (fx_min + fx_max)
        cy = 0.5 * (fy_min + fy_max)

        # Keep more vertical context below the face.
        half_w = max(8.0, 1.2 * face_span * float(expansion))
        top_pad = 0.9 * face_span * float(expansion)
        bottom_pad = 1.8 * face_span * float(expansion)

        x1 = cx - half_w
        x2 = cx + half_w
        y1 = cy - top_pad
        y2 = cy + bottom_pad

        return _clip_box(x1, y1, x2, y2)

    raise RuntimeError(
        'Cannot derive head area from pose: insufficient face or torso landmarks.'
    )


def crop_head_area_from_pose(
    img_rgb: np.ndarray,
    *,
    pose_xy: np.ndarray,
    expansion: float = 1.6,
) -> tuple[np.ndarray, int, int]:
    """
    Crop a coarse head area from the input image using pose landmarks.

    Parameters
    ----------
    img_rgb : np.ndarray
        Full RGB image with shape ``(H, W, 3)``.
    pose_xy : np.ndarray
        Pose landmarks with shape ``(33, 2)`` in full-image coordinates.
    expansion : float, optional
        Expansion multiplier applied to the estimated head area.

    Returns
    -------
    tuple[np.ndarray, int, int]
        Tuple ``(head_area_rgb, x1, y1)`` where ``x1`` and ``y1`` are the
        top-left offsets of the cropped area in full-image coordinates.
    """
    h, w = img_rgb.shape[:2]

    x1, y1, x2, y2 = head_area_xyxy_from_pose(
        pose_xy,
        full_w=w,
        full_h=h,
        expansion=expansion,
    )

    head_area_rgb = img_rgb[y1:y2, x1:x2, :]
    if head_area_rgb.size == 0:
        raise RuntimeError('Pose-derived head area is empty.')

    return np.ascontiguousarray(head_area_rgb), x1, y1


def square_head_bbox_from_face_bbox(
    x1: int, y1: int, x2: int, y2: int,
    w: int, h: int,
    expansion: float,
) -> tuple[int, int, int, int]:

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid face bbox: {(x1, y1, x2, y2)}")

    fw = x2 - x1
    fh = y2 - y1
    face_size = max(fw, fh)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # ---- upward bias (hair priority) ----
    cy -= 0.15 * face_size

    radius = max(1.0, 0.75 * face_size * float(expansion))

    bx1 = int(math.floor(cx - radius))
    by1 = int(math.floor(cy - radius))
    bx2 = int(math.ceil(cx + radius))
    by2 = int(math.ceil(cy + radius))

    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(w, bx2)
    by2 = min(h, by2)

    if bx2 <= bx1 or by2 <= by1:
        raise RuntimeError(f"Invalid square head bbox: {(bx1, by1, bx2, by2)}")

    return bx1, by1, bx2, by2


def expand_clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float) -> tuple[int, int, int, int]:
    # assume x2,y2 are end-exclusive after this normalization
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bbox: {(x1, y1, x2, y2)}")

    bw, bh = (x2 - x1), (y2 - y1)
    dx = int(round(bw * margin))
    dy = int(round(bh * margin))

    bx1 = max(0, x1 - dx)
    by1 = max(0, y1 - dy)
    bx2 = min(w, x2 + dx)
    by2 = min(h, y2 + dy)

    if bx2 <= bx1 or by2 <= by1:
        raise RuntimeError(f"Invalid expanded bbox: {(bx1, by1, bx2, by2)}")

    return bx1, by1, bx2, by2


def offset_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    dx: int,
    dy: int,
) -> tuple[int, int, int, int]:
    """
    Translate an end-exclusive bounding box by a constant offset.

    Parameters
    ----------
    bbox_xyxy : tuple[int, int, int, int]
        Bounding box ``(x1, y1, x2, y2)`` in local coordinates.
    dx : int
        Horizontal offset to add.
    dy : int
        Vertical offset to add.

    Returns
    -------
    tuple[int, int, int, int]
        Offset bounding box in the translated coordinate system.

    Raises
    ------
    RuntimeError
        If the input bounding box is invalid.
    """
    x1, y1, x2, y2 = bbox_xyxy

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f'Invalid bbox: {bbox_xyxy!r}')

    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def offset_landmarks_xy(
    xy: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> np.ndarray:
    """
    Translate landmark coordinates by a constant offset.

    Parameters
    ----------
    xy : np.ndarray
        Landmark coordinates with shape ``(N, 2)`` in local coordinates.
    dx : int
        Horizontal offset to add.
    dy : int
        Vertical offset to add.

    Returns
    -------
    np.ndarray
        Offset landmark coordinates with shape ``(N, 2)``.

    Raises
    ------
    ValueError
        If ``xy`` does not have shape ``(N, 2)``.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f'Invalid landmark array shape {xy.shape!r}; expected (N, 2).'
        )

    out = xy.astype(np.int32, copy=True)
    out[:, 0] += int(dx)
    out[:, 1] += int(dy)

    return out


def face_landmarks_from_pose_guided_head_area(
    img_rgb: np.ndarray,
    *,
    pose_xy: np.ndarray,
    face_landmarker: Any,
    expansion: float = 1.6,
) -> dict[str, Any]:
    """
    Detect face landmarks inside a pose-guided head area.

    This helper first derives a coarse head / upper-body area from pose
    landmarks, crops that region, then runs face landmark detection inside the
    cropped area. The detected landmarks and face bounding box are returned in
    both local crop coordinates and full-image coordinates.

    Parameters
    ----------
    img_rgb : np.ndarray
        Full RGB image with shape ``(H, W, 3)``.
    pose_xy : np.ndarray
        Pose landmarks with shape ``(33, 2)`` in full-image coordinates.
        Missing landmarks must be encoded as ``(-1, -1)``.
    face_landmarker : Any
        MediaPipe FaceLandmarker instance.
    expansion : float, optional
        Expansion multiplier forwarded to ``crop_head_area_from_pose(...)``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        ``head_area_rgb`` : np.ndarray
            Cropped pose-guided head area.

        ``head_area_xyxy`` : tuple[int, int, int, int]
            Head-area bounding box in full-image coordinates.

        ``head_offset_xy`` : tuple[int, int]
            Top-left crop offset ``(x1, y1)`` in full-image coordinates.

        ``face_xy_local`` : np.ndarray
            Face landmarks in head-area local coordinates.

        ``face_xy_global`` : np.ndarray
            Face landmarks in full-image coordinates.

        ``face_bbox_local`` : tuple[int, int, int, int]
            Face bounding box in head-area local coordinates.

        ``face_bbox_global`` : tuple[int, int, int, int]
            Face bounding box in full-image coordinates.

    Raises
    ------
    RuntimeError
        If the pose-guided crop cannot be derived, is empty, or no valid face
        landmarks can be detected inside the cropped area.
    ValueError
        If the input image or pose landmark array is malformed.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('img_rgb must have shape (H, W, 3).')

    if pose_xy.ndim != 2 or pose_xy.shape != (33, 2):
        raise ValueError(
            f'Invalid pose_xy shape {pose_xy.shape!r}; expected (33, 2).'
        )

    head_area_rgb, x1, y1 = crop_head_area_from_pose(
        img_rgb,
        pose_xy=pose_xy,
        expansion=expansion,
    )

    face_xy_local = mp_face_landmarks(
        head_area_rgb,
        face_landmarker=face_landmarker,
    )

    face_bbox_local = face_bbox_xyxy_from_landmarks(
        face_xy_local,
        head_area_rgb.shape,
    )

    face_xy_global = offset_landmarks_xy(
        face_xy_local,
        dx=x1,
        dy=y1,
    )

    face_bbox_global = offset_bbox_xyxy(
        face_bbox_local,
        dx=x1,
        dy=y1,
    )

    h_head, w_head = head_area_rgb.shape[:2]
    head_area_xyxy = (x1, y1, x1 + w_head, y1 + h_head)

    return {
        'head_area_rgb': head_area_rgb,
        'head_area_xyxy': head_area_xyxy,
        'head_offset_xy': (x1, y1),
        'face_xy_local': face_xy_local,
        'face_xy_global': face_xy_global,
        'face_bbox_local': face_bbox_local,
        'face_bbox_global': face_bbox_global,
    }
