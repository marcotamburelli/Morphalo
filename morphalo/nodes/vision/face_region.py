import numpy as np

from morphalo.nodes.preprocess.utils import offset_bbox_xyxy, offset_landmarks_xy


def mp_face_landmarks(
    img_rgb: np.ndarray,
    *,
    face_landmarker,
) -> np.ndarray:
    """
    Detect face landmarks and return the most prominent face.

    Returns pixel-space landmarks (N, 2).
    """
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    result = face_landmarker.detect(mp_image)
    face_landmarks_list = result.face_landmarks or []

    if not face_landmarks_list:
        raise RuntimeError('MediaPipe found no face landmarks.')

    best_xy = None
    best_area = None

    for landmarks in face_landmarks_list:
        face_xy = np.empty((len(landmarks), 2), dtype=np.int32)

        for i, lm in enumerate(landmarks):
            x = int(round(float(lm.x) * w))
            y = int(round(float(lm.y) * h))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            face_xy[i] = (x, y)

        x1 = np.min(face_xy[:, 0])
        y1 = np.min(face_xy[:, 1])
        x2 = np.max(face_xy[:, 0]) + 1
        y2 = np.max(face_xy[:, 1]) + 1

        if x2 <= x1 or y2 <= y1:
            continue

        area = float((x2 - x1) * (y2 - y1))

        if best_area is None or area > best_area:
            best_area = area
            best_xy = face_xy

    if best_xy is None:
        raise RuntimeError('Could not derive valid face landmarks.')

    return best_xy


def face_bbox_xyxy_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    expansion: float = 1.0,
) -> tuple[int, int, int, int]:
    """
    Compute an end-exclusive face bounding box from face landmarks.
    """
    if face_xy.ndim != 2 or face_xy.shape[1] != 2:
        raise RuntimeError('Invalid face landmarks.')

    if expansion < 0:
        raise ValueError(
            f'Invalid expansion={expansion!r}; expected >= 0.'
        )

    h, w = image_shape[:2]

    x1 = int(np.min(face_xy[:, 0]))
    y1 = int(np.min(face_xy[:, 1]))
    x2 = int(np.max(face_xy[:, 0])) + 1
    y2 = int(np.max(face_xy[:, 1])) + 1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid face bbox.')

    if expansion != 1.0:
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bw *= float(expansion)
        bh *= float(expansion)

        x1 = int(np.floor(cx - bw / 2.0))
        y1 = int(np.floor(cy - bh / 2.0))
        x2 = int(np.ceil(cx + bw / 2.0))
        y2 = int(np.ceil(cy + bh / 2.0))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid face bbox.')

    return x1, y1, x2, y2


def eye_bbox_xyxy_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.2,
) -> tuple[int, int, int, int]:
    """
    Compute eye bbox from face landmarks.
    """
    h, w = image_shape[:2]

    left_ids = list(range(33, 133))
    right_ids = list(range(362, 463))

    if which == 'left':
        ids = left_ids
    elif which == 'right':
        ids = right_ids
    elif which == 'both':
        ids = left_ids + right_ids
    else:
        raise ValueError(f'Invalid which={which!r}')

    pts = face_xy[np.asarray(ids)]

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0])) + 1
    y2 = int(np.max(pts[:, 1])) + 1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw *= expansion
    bh *= expansion

    ex1 = int(cx - bw / 2)
    ex2 = int(cx + bw / 2)
    ey1 = int(cy - bh / 2)
    ey2 = int(cy + bh / 2)

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(w, ex2)
    ey2 = min(h, ey2)

    return ex1, ey1, ex2, ey2


def eye_mask_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float,
) -> np.ndarray:
    """
    Build eye mask from face landmarks.
    """
    import cv2

    h, w = image_shape[:2]

    left_ids = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ]
    right_ids = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ]

    def _points(ids):
        pts = []
        for i in ids:
            x, y = face_xy[int(i)]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])
        return np.asarray(pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    if which in ('left', 'both'):
        pts = _points(left_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if which in ('right', 'both'):
        pts = _points(right_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if expansion > 1.0:
        radius = max(1, int(round((expansion - 1.0) * 8)))
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask = cv2.dilate(mask, k, iterations=1)

    return (mask > 0)


def eyebrow_bbox_xyxy_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.2,
) -> tuple[int, int, int, int]:
    """
    Compute eyebrow bbox from face landmarks.
    """
    h, w = image_shape[:2]

    left_ids = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    right_ids = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]

    if which == 'left':
        ids = left_ids
    elif which == 'right':
        ids = right_ids
    elif which == 'both':
        ids = left_ids + right_ids
    else:
        raise ValueError(f'Invalid which={which!r}')

    pts = face_xy[np.asarray(ids)]

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0])) + 1
    y2 = int(np.max(pts[:, 1])) + 1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw *= expansion
    bh *= expansion

    ex1 = int(cx - bw / 2)
    ex2 = int(cx + bw / 2)
    ey1 = int(cy - bh / 2)
    ey2 = int(cy + bh / 2)

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(w, ex2)
    ey2 = min(h, ey2)

    return ex1, ey1, ex2, ey2


def eyebrow_mask_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float,
) -> np.ndarray:
    """
    Build eyebrow mask from face landmarks.
    """
    import cv2

    h, w = image_shape[:2]

    left_ids = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    right_ids = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]

    def _points(ids):
        pts = []
        for i in ids:
            x, y = face_xy[int(i)]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])
        return np.asarray(pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    if which in ('left', 'both'):
        pts = _points(left_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if which in ('right', 'both'):
        pts = _points(right_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if expansion > 1.0:
        radius = max(1, int(round((expansion - 1.0) * 8)))
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask = cv2.dilate(mask, k, iterations=1)

    return (mask > 0)


def face_landmarks_from_pose_guided_head_area(
    img_rgb: np.ndarray,
    *,
    pose_xy: np.ndarray,
    face_landmarker,
    expansion: float = 1.6,
) -> dict[str, object]:
    """
    Detect face landmarks inside a pose-guided head area.
    """
    from morphalo.nodes.vision.human import crop_head_area_from_pose

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
