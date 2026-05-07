import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PoseLandmarksResult:
    """
    Rich MediaPipe pose result with image-space and world-space coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Array with shape ``(33, 2)`` containing absolute pixel coordinates.
        Invalid landmarks are encoded as ``(-1, -1)``.
    z : np.ndarray
        Array with shape ``(33,)`` containing image-space relative depth.
        Invalid landmarks are encoded as ``np.nan``.
    world_xyz : np.ndarray
        Array with shape ``(33, 3)`` containing world coordinates.
        Invalid landmarks are encoded as ``np.nan``.
    visibility : np.ndarray
        Array with shape ``(33,)`` containing landmark visibility scores.
    presence : np.ndarray
        Array with shape ``(33,)`` containing landmark presence scores.
    valid : np.ndarray
        Boolean array with shape ``(33,)``. A landmark is valid only if both
        visibility and presence pass the configured thresholds and the image-space
        coordinates fall inside the image bounds.
    strong : np.ndarray
        Boolean array with shape ``(33,)``. A landmark is strong only if it is
        valid and also passes the fixed structural robustness thresholds used
        for person-level pose acceptance.
    """
    xy: np.ndarray
    z: np.ndarray
    world_xyz: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    valid: np.ndarray
    strong: np.ndarray


@dataclass(frozen=True)
class HandLandmarksResult:
    """
    Rich MediaPipe hand result for all detected hands.

    Parameters
    ----------
    xy : np.ndarray
        Array with shape ``(H, 21, 2)`` containing absolute pixel coordinates.
        Invalid landmarks are encoded as ``(-1, -1)``.
    z : np.ndarray
        Array with shape ``(H, 21)`` containing image-space relative depth.
        Invalid landmarks are encoded as ``np.nan``.
    world_xyz : np.ndarray
        Array with shape ``(H, 21, 3)`` containing world coordinates.
        Invalid landmarks are encoded as ``np.nan``.
    visibility : np.ndarray
        Visibility-like confidence per landmark, shape ``(H, 21)``.
        Since HandLandmarker does not expose per-landmark visibility/presence
        in the same way as PoseLandmarker, this field is filled with ones for
        detected landmarks and zeros for invalid ones.
    valid : np.ndarray
        Boolean mask with shape ``(H, 21)``.
    handedness : list[str]
        Hand labels, one per detected hand, typically ``'Left'`` or ``'Right'``.
    handedness_score : np.ndarray
        Confidence score for handedness classification, shape ``(H,)``.
    """
    xy: np.ndarray
    z: np.ndarray
    world_xyz: np.ndarray
    visibility: np.ndarray
    valid: np.ndarray
    handedness: list[str]
    handedness_score: np.ndarray


# The two following functions are intentionally not equivalent views of the same data

def mp_pose_landmarks_full(
    img_rgb: np.ndarray,
    pose_landmarker,
    *,
    min_visibility: float = 0.0,
    min_presence: float = 0.0,
) -> PoseLandmarksResult:
    """
    Return a rich MediaPipe pose result for the most prominent detected pose.

    This function provides a *full, analysis-oriented* representation of the
    detected human pose. It exposes all available landmark information,
    including:

    - image-space coordinates (pixel space)
    - relative depth (z)
    - world-space coordinates (metric space)
    - visibility and presence scores
    - validity and strong-landmark flags

    Landmarks are first filtered using the caller-provided thresholds
    (``min_visibility``, ``min_presence``), then additional robustness checks
    are applied to ensure that the result represents a plausible human pose.

    This function is intended for:
    - pose analysis
    - scoring and quality estimation (e.g. ``PersonScorer``)
    - geometric reasoning (joint angles, proportions, consistency checks)

    It is *not* intended to enforce a stable subset of landmarks for downstream
    tasks such as cropping.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image with shape ``(H, W, 3)`` and dtype uint8.
    pose_landmarker : Any
        MediaPipe PoseLandmarker instance.
    min_visibility : float, optional
        Minimum visibility required for a landmark to be considered valid.
    min_presence : float, optional
        Minimum presence required for a landmark to be considered valid.

    Returns
    -------
    PoseLandmarksResult
        Rich pose result containing:
        - image coordinates
        - world coordinates
        - visibility / presence
        - validity flags
        - strong landmark flags

    Raises
    ------
    RuntimeError
        If MediaPipe returns no pose landmarks, or if the detected pose is too
        weak to identify a person reliably.
    ValueError
        If thresholds are outside ``[0, 1]`` or the input image is invalid.
    """
    import mediapipe as mp

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('img_rgb must have shape (H, W, 3).')

    if not (0.0 <= float(min_visibility) <= 1.0):
        raise ValueError(
            f'Invalid min_visibility={min_visibility!r}; expected in [0, 1].'
        )

    if not (0.0 <= float(min_presence) <= 1.0):
        raise ValueError(
            f'Invalid min_presence={min_presence!r}; expected in [0, 1].'
        )

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    result = pose_landmarker.detect(mp_image)

    pose_landmarks_list = getattr(result, 'pose_landmarks', None) or []
    pose_world_landmarks_list = getattr(
        result, 'pose_world_landmarks', None) or []

    if not pose_landmarks_list:
        raise RuntimeError('MediaPipe found no pose landmarks.')

    def _pose_score(landmarks) -> float:
        """
        Select the most useful pose hypothesis.

        The score favors poses with many confident landmarks and a reasonable
        projected extent in image space.
        """
        vis_sum = 0.0
        pres_sum = 0.0
        xs = []
        ys = []

        for lm in landmarks:
            vis_sum += float(getattr(lm, 'visibility', 0.0))
            pres_sum += float(getattr(lm, 'presence', 0.0))
            xs.append(float(lm.x))
            ys.append(float(lm.y))

        span_x = max(0.0, max(xs) - min(xs)) if xs else 0.0
        span_y = max(0.0, max(ys) - min(ys)) if ys else 0.0
        area = span_x * span_y

        return vis_sum + pres_sum + 10.0 * area

    best_idx = max(
        range(len(pose_landmarks_list)),
        key=lambda i: _pose_score(pose_landmarks_list[i]),
    )

    landmarks = pose_landmarks_list[best_idx]
    world_landmarks = (
        pose_world_landmarks_list[best_idx]
        if best_idx < len(pose_world_landmarks_list)
        else None
    )

    if len(landmarks) != 33:
        raise RuntimeError(
            f'Expected 33 pose landmarks, got {len(landmarks)}.'
        )

    xy = np.full((33, 2), -1, dtype=np.int32)
    z = np.full((33,), np.nan, dtype=np.float32)
    world_xyz = np.full((33, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((33,), dtype=np.float32)
    presence = np.zeros((33,), dtype=np.float32)
    valid = np.zeros((33,), dtype=bool)
    strong = np.zeros((33,), dtype=bool)

    for i, lm in enumerate(landmarks):
        x_norm = float(lm.x)
        y_norm = float(lm.y)
        z_rel = float(getattr(lm, 'z', np.nan))
        vis = float(getattr(lm, 'visibility', 0.0))
        pres = float(getattr(lm, 'presence', 0.0))

        visibility[i] = vis
        presence[i] = pres
        z[i] = z_rel

        x_px = int(round(x_norm * w))
        y_px = int(round(y_norm * h))

        in_bounds = (0 <= x_px < w) and (0 <= y_px < h)
        passes_valid = (vis >= min_visibility) and (pres >= min_presence)
        passes_strong = (vis >= 0.5) and (pres >= 0.5)

        if in_bounds and passes_valid:
            xy[i] = (x_px, y_px)
            valid[i] = True

        if in_bounds and passes_strong:
            strong[i] = True

    if world_landmarks is not None:
        m = min(len(world_landmarks), 33)
        for i in range(m):
            wlm = world_landmarks[i]
            world_xyz[i, 0] = float(wlm.x)
            world_xyz[i, 1] = float(wlm.y)
            world_xyz[i, 2] = float(wlm.z)

    # -------------------------------------------------------------
    # Basic robustness checks
    # -------------------------------------------------------------
    strong_count = int(np.count_nonzero(strong))

    if strong_count < 8:
        raise RuntimeError(
            'MediaPipe pose landmarks are too weak to identify a person reliably.'
        )

    left_shoulder = bool(strong[11])
    right_shoulder = bool(strong[12])
    nose = bool(strong[0])

    if not (left_shoulder or right_shoulder or nose):
        raise RuntimeError(
            'MediaPipe pose landmarks lack stable anchors (no shoulders or nose).'
        )

    return PoseLandmarksResult(
        xy=xy,
        z=z,
        world_xyz=world_xyz,
        visibility=visibility,
        presence=presence,
        valid=valid,
        strong=strong,
    )


def mp_pose_landmarks_xy(
    img_rgb: np.ndarray,
    *,
    pose_landmarker,
) -> np.ndarray:
    """
    Detect pose landmarks and return a stable subset of coordinates in pixel space.

    This function is a *decision-oriented wrapper* built for spatial tasks such as
    subject selection and cropping (e.g. ``SubjectCrop``).

    It enforces a stricter and stable interpretation of pose landmarks:

    - Only landmarks with both visibility and presence >= 0.35 are kept
    - Weak or uncertain landmarks are discarded (set to ``(-1, -1)``)
    - Additional structural checks are applied to ensure the pose is usable:
        - at least 8 strong landmarks must be present
        - at least one anchor (shoulder or nose) must be valid

    The goal is to produce a robust and conservative landmark set suitable for:
    - selecting the correct person in multi-person scenes
    - guiding bounding box selection (YOLO + pose)
    - stable cropping of human subjects

    Compared to ``mp_pose_landmarks_full(...)``, this function intentionally
    sacrifices completeness for robustness.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image with shape ``(H, W, 3)`` and dtype uint8.
    pose_landmarker : Any
        MediaPipe PoseLandmarker instance.

    Returns
    -------
    np.ndarray
        Array with shape ``(33, 2)`` containing absolute pixel coordinates.
        Missing or weak landmarks are encoded as ``(-1, -1)``.

    Raises
    ------
    RuntimeError
        If no pose is detected or if the landmark configuration is too weak
        to plausibly represent a person.
    """
    res = mp_pose_landmarks_full(
        img_rgb,
        pose_landmarker,
        min_visibility=0.35,
        min_presence=0.35,
    )

    pts = res.xy.copy()

    strong_count = int(np.count_nonzero(
        (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
    ))

    if strong_count < 8:
        raise RuntimeError(
            'MediaPipe pose landmarks are too weak to identify a person reliably.'
        )

    left_shoulder = bool(pts[11, 0] >= 0 and pts[11, 1] >= 0)
    right_shoulder = bool(pts[12, 0] >= 0 and pts[12, 1] >= 0)
    nose = bool(pts[0, 0] >= 0 and pts[0, 1] >= 0)

    if not (left_shoulder or right_shoulder or nose):
        raise RuntimeError(
            'MediaPipe pose landmarks lack stable anchors (no shoulders or nose).'
        )

    return pts


def mp_hand_landmarks_full(
    img_rgb: np.ndarray,
    hand_landmarker,
    *,
    min_handedness_score: float = 0.0,
) -> HandLandmarksResult:
    """
    Return rich MediaPipe hand landmarks for all detected hands.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image with shape ``(H, W, 3)`` and dtype uint8.
    hand_landmarker : Any
        MediaPipe HandLandmarker instance.
    min_handedness_score : float, optional
        Minimum handedness classification score required to keep a detected hand.

    Returns
    -------
    HandLandmarksResult
        Rich result containing image-space landmarks, relative depth,
        world-space landmarks, handedness labels, and validity masks.

    Raises
    ------
    RuntimeError
        If MediaPipe returns no hand landmarks after filtering.
    ValueError
        If the input image is invalid or thresholds are outside ``[0, 1]``.
    """
    import mediapipe as mp

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('img_rgb must have shape (H, W, 3).')

    if not (0.0 <= float(min_handedness_score) <= 1.0):
        raise ValueError(
            f'Invalid min_handedness_score={min_handedness_score!r}; expected in [0, 1].'
        )

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    result = hand_landmarker.detect(mp_image)

    hand_landmarks_list = getattr(result, 'hand_landmarks', None) or []
    hand_world_landmarks_list = getattr(
        result, 'hand_world_landmarks', None) or []
    handedness_list = getattr(result, 'handedness', None) or []

    if not hand_landmarks_list:
        raise RuntimeError('MediaPipe found no hand landmarks.')

    kept_xy = []
    kept_z = []
    kept_world_xyz = []
    kept_visibility = []
    kept_valid = []
    kept_handedness = []
    kept_handedness_score = []

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if len(hand_landmarks) != 21:
            continue

        hand_handedness = handedness_list[hand_idx] if hand_idx < len(
            handedness_list) else []
        if hand_handedness:
            top_cat = max(hand_handedness, key=lambda c: float(
                getattr(c, 'score', 0.0)))
            hand_label = str(getattr(top_cat, 'category_name', 'Unknown'))
            hand_score = float(getattr(top_cat, 'score', 0.0))
        else:
            hand_label = 'Unknown'
            hand_score = 0.0

        if hand_score < float(min_handedness_score):
            continue

        xy = np.full((21, 2), -1, dtype=np.int32)
        z = np.full((21,), np.nan, dtype=np.float32)
        world_xyz = np.full((21, 3), np.nan, dtype=np.float32)
        visibility = np.zeros((21,), dtype=np.float32)
        valid = np.zeros((21,), dtype=bool)

        for i, lm in enumerate(hand_landmarks):
            x_norm = float(lm.x)
            y_norm = float(lm.y)
            z_rel = float(getattr(lm, 'z', np.nan))

            x_px = int(round(x_norm * w))
            y_px = int(round(y_norm * h))

            in_bounds = (0 <= x_px < w) and (0 <= y_px < h)

            z[i] = z_rel

            if in_bounds:
                xy[i] = (x_px, y_px)
                visibility[i] = 1.0
                valid[i] = True

        if hand_idx < len(hand_world_landmarks_list):
            world_landmarks = hand_world_landmarks_list[hand_idx]
            for i in range(min(len(world_landmarks), 21)):
                wlm = world_landmarks[i]
                world_xyz[i, 0] = float(wlm.x)
                world_xyz[i, 1] = float(wlm.y)
                world_xyz[i, 2] = float(wlm.z)

        strong_count = int(np.count_nonzero(valid))
        wrist_ok = bool(valid[0])

        if strong_count < 8:
            continue

        if not wrist_ok:
            continue

        kept_xy.append(xy)
        kept_z.append(z)
        kept_world_xyz.append(world_xyz)
        kept_visibility.append(visibility)
        kept_valid.append(valid)
        kept_handedness.append(hand_label)
        kept_handedness_score.append(hand_score)

    if not kept_xy:
        raise RuntimeError('MediaPipe found no reliable hand landmarks.')

    return HandLandmarksResult(
        xy=np.stack(kept_xy, axis=0),
        z=np.stack(kept_z, axis=0),
        world_xyz=np.stack(kept_world_xyz, axis=0),
        visibility=np.stack(kept_visibility, axis=0),
        valid=np.stack(kept_valid, axis=0),
        handedness=kept_handedness,
        handedness_score=np.asarray(kept_handedness_score, dtype=np.float32),
    )


def person_bboxes_xyxy(res, node_id: str) -> list[tuple[int, int, int, int]]:
    """
    Return all YOLO person detections as integer xyxy boxes.

    Parameters
    ----------
    res : Any
        Single YOLO prediction result.
    node_id : str
        Node id used for error messages.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Person bounding boxes in ``(x1, y1, x2, y2)`` format.

    Raises
    ------
    RuntimeError
        If YOLO returns no boxes or no person detections.
    """
    if res.boxes is None or len(res.boxes) == 0:
        raise RuntimeError(f'{node_id}: YOLO found no boxes.')

    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.detach().cpu().numpy()

    person_idx = np.where(cls == 0)[0]
    if person_idx.size == 0:
        raise RuntimeError(f"{node_id}: YOLO found no 'person' detections.")

    out = []
    for i in person_idx:
        x1, y1, x2, y2 = xyxy[int(i)].tolist()
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))

    if not out:
        raise RuntimeError(f"{node_id}: YOLO found no valid 'person' boxes.")

    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    return out


def person_bbox_xyxy_from_pose(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    expansion: float = 1.5,
    min_points: int = 5,
) -> tuple[int, int, int, int]:
    """
    Infer a coarse person bbox from MediaPipe pose landmarks.

    This is intended as a fallback when the object detector fails to return
    a usable person box.

    The bbox is based on valid pose landmarks, then expanded to approximate
    the full visible person silhouette. It is intentionally conservative:
    it may include extra background, but should avoid crashing the DAG.

    Parameters
    ----------
    pose_xy : np.ndarray
        Pose landmarks with shape ``(33, 2)`` in full-image coordinates.
        Missing landmarks are encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    expansion : float, optional
        Multiplicative bbox expansion factor.
    min_points : int, optional
        Minimum number of valid landmarks required.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive bbox ``(x1, y1, x2, y2)``.

    Raises
    ------
    RuntimeError
        If there are not enough valid pose landmarks.
    """
    if pose_xy is None:
        raise RuntimeError('Cannot infer person bbox: pose_xy is None.')

    if pose_xy.ndim != 2 or pose_xy.shape[1] != 2:
        raise RuntimeError(
            f'Invalid pose landmark shape {pose_xy.shape!r}; expected (N, 2).'
        )

    h, w = image_shape[:2]

    valid = (
        (pose_xy[:, 0] >= 0) &
        (pose_xy[:, 1] >= 0)
    )
    pts = pose_xy[valid]

    if pts.shape[0] < int(min_points):
        raise RuntimeError(
            f'Cannot infer person bbox from pose: only {pts.shape[0]} valid landmarks.'
        )

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0])) + 1
    y2 = int(np.max(pts[:, 1])) + 1

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid raw pose-derived person bbox.')

    bw = x2 - x1
    bh = y2 - y1

    # Landmarks usually sit inside the body silhouette, not on its contour.
    # Expand more vertically than horizontally, and bias upward slightly to
    # preserve head / hair.
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    ex = max(1.0, float(expansion))
    ey = max(1.0, float(expansion) * 1.15)

    out_w = max(1.0, bw * ex)
    out_h = max(1.0, bh * ey)

    # Upward bias: useful when feet/legs are missing but head must be preserved.
    cy -= 0.06 * out_h

    ox1 = int(math.floor(cx - out_w / 2.0))
    ox2 = int(math.ceil(cx + out_w / 2.0))
    oy1 = int(math.floor(cy - out_h / 2.0))
    oy2 = int(math.ceil(cy + out_h / 2.0))

    ox1 = max(0, min(w - 1, ox1))
    oy1 = max(0, min(h - 1, oy1))
    ox2 = max(ox1 + 1, min(w, ox2))
    oy2 = max(oy1 + 1, min(h, oy2))

    if ox2 <= ox1 or oy2 <= oy1:
        raise RuntimeError('Invalid expanded pose-derived person bbox.')

    return ox1, oy1, ox2, oy2


def resolve_person_bbox_xyxy(
    res,
    node_id: str,
    *,
    pose_xy: Optional[np.ndarray],
    image_shape: tuple[int, ...],
    pose_fallback_expansion: float = 1.35,
) -> tuple[int, int, int, int]:
    """
    Resolve a person bbox using YOLO first, then MediaPipe Pose fallback.

    YOLO is preferred when available because it usually estimates the visible
    person silhouette better. If YOLO fails, pose landmarks are used to infer
    a coarse bbox.
    """
    try:
        person_boxes = person_bboxes_xyxy(res, node_id)
        return select_person_bbox_xyxy(
            person_boxes,
            pose_xy=pose_xy,
        )
    except RuntimeError as exc:
        return person_bbox_xyxy_from_pose(
            pose_xy,
            image_shape,
            expansion=pose_fallback_expansion,
        )


def score_bbox_with_pose(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    pose_xy: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Score a person bbox against pose landmarks.

    Parameters
    ----------
    bbox_xyxy : tuple[int, int, int, int]
        Candidate bbox in full-image coordinates.
    pose_xy : np.ndarray
        Pose landmarks with shape ``(33, 2)`` in full-image coordinates.
        Missing landmarks are encoded as ``(-1, -1)``.

    Returns
    -------
    tuple[float, float, float, float]
        Lexicographic score tuple:
        - torso coverage
        - landmark coverage ratio
        - face landmark coverage ratio
        - area
    """
    x1, y1, x2, y2 = bbox_xyxy

    valid = (
        (pose_xy[:, 0] >= 0) &
        (pose_xy[:, 1] >= 0)
    )
    pts = pose_xy[valid]

    if pts.size == 0:
        area = float((x2 - x1) * (y2 - y1))
        return (0.0, 0.0, 0.0, area)

    inside = (
        (pts[:, 0] >= x1) & (pts[:, 0] < x2) &
        (pts[:, 1] >= y1) & (pts[:, 1] < y2)
    )
    landmark_ratio = float(np.mean(inside))

    torso_ids = [11, 12, 23, 24]
    torso_hits = 0
    torso_total = 0
    for idx in torso_ids:
        px, py = pose_xy[idx]
        if px >= 0 and py >= 0:
            torso_total += 1
            if x1 <= px < x2 and y1 <= py < y2:
                torso_hits += 1
    torso_ratio = float(torso_hits / torso_total) if torso_total > 0 else 0.0

    face_ids = [0, 2, 5, 7, 8]
    face_hits = 0
    face_total = 0
    for idx in face_ids:
        px, py = pose_xy[idx]
        if px >= 0 and py >= 0:
            face_total += 1
            if x1 <= px < x2 and y1 <= py < y2:
                face_hits += 1
    face_ratio = float(face_hits / face_total) if face_total > 0 else 0.0

    area = float((x2 - x1) * (y2 - y1))

    return (
        torso_ratio,
        landmark_ratio,
        face_ratio,
        area,
    )


def select_person_bbox_xyxy(
    person_boxes_xyxy: list[tuple[int, int, int, int]],
    *,
    pose_xy: Optional[np.ndarray],
) -> tuple[int, int, int, int]:
    """
    Select the best person bbox, optionally guided by pose landmarks.

    Parameters
    ----------
    person_boxes_xyxy : list[tuple[int, int, int, int]]
        Candidate person boxes.
    pose_xy : np.ndarray, optional
        Pose landmarks in full-image coordinates.

    Returns
    -------
    tuple[int, int, int, int]
        Selected person bbox.
    """
    if not person_boxes_xyxy:
        raise RuntimeError('No candidate person boxes provided.')

    if pose_xy is None:
        return person_boxes_xyxy[0]

    best_box = None
    best_score = None

    for box in person_boxes_xyxy:
        score = score_bbox_with_pose(box, pose_xy=pose_xy)
        if best_score is None or score > best_score:
            best_score = score
            best_box = box

    if best_box is None:
        raise RuntimeError('Could not select a pose-aligned person bbox.')

    return best_box


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
) -> tuple[int, int, int, int]:
    """
    Compute face bounding box from landmarks.
    """
    if face_xy.ndim != 2 or face_xy.shape[1] != 2:
        raise RuntimeError('Invalid face landmarks.')

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


def hand_bbox_xyxy(
    hand_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    expansion: float = 1.0,
) -> tuple[int, int, int, int]:
    """
    Compute a hand bounding box from a single hand landmark set.

    Parameters
    ----------
    hand_xy : np.ndarray
        Hand landmark coordinates with shape ``(21, 2)`` in pixel space.
        Invalid landmarks must be encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Image shape. Only the first two dimensions are used as ``(H, W)``.
    expansion : float, optional
        Multiplicative expansion factor applied to the raw landmark bbox.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box ``(x1, y1, x2, y2)`` in end-exclusive image coordinates.

    Raises
    ------
    RuntimeError
        If the hand landmarks do not contain enough valid points to define a bbox.
    ValueError
        If the input shape is invalid or ``expansion < 0``.
    """
    if hand_xy.ndim != 2 or hand_xy.shape != (21, 2):
        raise ValueError(
            f'Invalid hand landmark shape {hand_xy.shape!r}; expected (21, 2).'
        )

    if expansion < 0:
        raise ValueError(
            f'Invalid expansion={expansion!r}; expected >= 0.'
        )

    h, w = image_shape[:2]

    valid = (
        (hand_xy[:, 0] >= 0) &
        (hand_xy[:, 1] >= 0)
    )

    pts = hand_xy[valid]
    if pts.shape[0] < 2:
        raise RuntimeError('Not enough valid hand landmarks to define a bbox.')

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0])) + 1
    y2 = int(np.max(pts[:, 1])) + 1

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid raw hand bbox.')

    bw = x2 - x1
    bh = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw = max(1.0, float(bw) * float(expansion))
    bh = max(1.0, float(bh) * float(expansion))

    ex1 = int(math.floor(cx - bw / 2.0))
    ey1 = int(math.floor(cy - bh / 2.0))
    ex2 = int(math.ceil(cx + bw / 2.0))
    ey2 = int(math.ceil(cy + bh / 2.0))

    ex1 = max(0, min(w - 1, ex1))
    ey1 = max(0, min(h - 1, ey1))
    ex2 = max(ex1 + 1, min(w, ex2))
    ey2 = max(ey1 + 1, min(h, ey2))

    if ex2 <= ex1 or ey2 <= ey1:
        raise RuntimeError('Invalid expanded hand bbox.')

    return ex1, ey1, ex2, ey2


def _select_hand_indices(
    hands: HandLandmarksResult,
    *,
    which: str,
) -> list[int]:
    """
    Select hand indices using observer-side semantics with a conservative fallback.

    Selection strategy
    ------------------
    - ``which='both'``:
      return all detected hands.
    - ``which='left'`` or ``which='right'``:
      first try MediaPipe handedness (which follows subject perspective and is
      therefore mapped to observer perspective here).

      If handedness does not yield a match, fall back to horizontal image
      position **only when at least two hands are available**. In that case,
      the leftmost hand in the image is treated as observer-left and the
      rightmost hand as observer-right.

      If fewer than two hands are available and handedness does not match,
      selection fails rather than returning a potentially wrong hand.

    Parameters
    ----------
    hands : HandLandmarksResult
        Rich MediaPipe hand result.
    which : {'left', 'right', 'both'}
        Requested hand selection in observer/image perspective.

    Returns
    -------
    list[int]
        Selected indices into ``hands.xy``.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If no suitable hand can be selected.
    """
    if which not in ('left', 'right', 'both'):
        raise ValueError(
            f"Invalid which={which!r}; expected 'left', 'right', or 'both'."
        )

    n_hands = hands.xy.shape[0]
    if n_hands == 0:
        raise RuntimeError('No hand landmarks available.')

    if which == 'both':
        return list(range(n_hands))

    labels = [str(h).lower() for h in hands.handedness]

    target_label = {
        'left': 'right',   # observer -> subject
        'right': 'left',
    }[which]

    selected: list[int] = []

    # First pass: handedness-based selection.
    for i, label in enumerate(labels):
        if label == target_label:
            selected.append(i)

    if selected:
        return selected

    # Fallback: use horizontal image position only if at least two hands
    # are geometrically available. With a single detected hand, returning it
    # would be ambiguous and could silently select the wrong side.
    centers: list[tuple[int, float]] = []

    for i in range(n_hands):
        hand_xy = hands.xy[i]
        valid = (
            (hand_xy[:, 0] >= 0) &
            (hand_xy[:, 1] >= 0)
        )
        pts = hand_xy[valid]
        if pts.shape[0] == 0:
            continue

        cx = float(np.mean(pts[:, 0]))
        centers.append((i, cx))

    if len(centers) < 2:
        raise RuntimeError(
            f'Could not reliably select a hand for which={which!r}: '
            'handedness did not match and fewer than two hands were detected.'
        )

    centers.sort(key=lambda t: t[1])

    if which == 'left':
        return [centers[0][0]]

    return [centers[-1][0]]


def hands_bbox_xyxy_from_landmarks(
    hands: HandLandmarksResult,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.0,
) -> tuple[int, int, int, int]:
    """
    Compute a bounding box covering one or more detected hands.

    Parameters
    ----------
    hands : HandLandmarksResult
        Rich MediaPipe hand result.
    image_shape : tuple[int, ...]
        Image shape. Only the first two dimensions are used as ``(H, W)``.
    which : {'left', 'right', 'both'}
        Which hand(s) to include.

        - ``'left'``:
          subject's left hand
        - ``'right'``:
          subject's right hand
        - ``'both'``:
          union of all detected valid hands
    expansion : float, optional
        Multiplicative expansion factor applied to each per-hand bbox before
        union.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box ``(x1, y1, x2, y2)`` in end-exclusive image coordinates.

    Raises
    ------
    RuntimeError
        If no matching hand can be found.
    ValueError
        If ``which`` is invalid.
    """
    if which not in ('left', 'right', 'both'):
        raise ValueError(
            f"Invalid which={which!r}; expected 'left', 'right', or 'both'."
        )

    n_hands = hands.xy.shape[0]
    if n_hands == 0:
        raise RuntimeError('No hand landmarks available.')

    boxes: list[tuple[int, int, int, int]] = []

    selected_indices = _select_hand_indices(
        hands,
        which=which,
    )

    boxes: list[tuple[int, int, int, int]] = []

    for i in selected_indices:
        try:
            box = hand_bbox_xyxy(
                hands.xy[i],
                image_shape,
                expansion=expansion,
            )
        except RuntimeError:
            continue

        boxes.append(box)

    if not boxes:
        raise RuntimeError(
            f'Could not derive a hand bbox for which={which!r}.'
        )

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid combined hand bbox.')

    return x1, y1, x2, y2


def hands_mask_from_landmarks(
    hands: HandLandmarksResult,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.0,
) -> np.ndarray:
    """
    Build a hand mask from one or more detected hand landmark sets.

    The mask is derived geometrically from MediaPipe hand landmarks by filling
    the convex hull of each selected hand. When multiple hands are selected,
    their hull masks are merged with logical OR.

    Parameters
    ----------
    hands : HandLandmarksResult
        Rich MediaPipe hand result.
    image_shape : tuple[int, ...]
        Image shape. Only the first two dimensions are used as ``(H, W)``.
    which : {'left', 'right', 'both'}
        Which hand(s) to include.

        - ``'left'``:
          subject's left hand
        - ``'right'``:
          subject's right hand
        - ``'both'``:
          union of all detected valid hands
    expansion : float, optional
        Optional geometric expansion of the hand mask.

        A minimal dilation is always applied because the convex hull of hand
        landmarks is typically tighter than the true visible hand silhouette.
        When ``expansion > 1.0``, additional dilation is applied on top of the
        size-aware base dilation.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(H, W)``.

    Raises
    ------
    RuntimeError
        If no matching hand can be found or no valid mask can be built.
    ValueError
        If ``which`` is invalid.
    """
    import cv2

    if which not in ('left', 'right', 'both'):
        raise ValueError(
            f"Invalid which={which!r}; expected 'left', 'right', or 'both'."
        )

    if expansion < 0:
        raise ValueError(
            f'Invalid expansion={expansion!r}; expected >= 0.'
        )

    h, w = image_shape[:2]

    if hands.xy.ndim != 3 or hands.xy.shape[1:] != (21, 2):
        raise ValueError(
            f'Invalid hands.xy shape {hands.xy.shape!r}; expected (H, 21, 2).'
        )

    mask = np.zeros((h, w), dtype=np.uint8)
    selected = 0

    selected_indices = _select_hand_indices(
        hands,
        which=which,
    )

    for i in selected_indices:
        hand_xy = hands.xy[i]
        valid = (
            (hand_xy[:, 0] >= 0) &
            (hand_xy[:, 1] >= 0)
        )

        pts = hand_xy[valid]
        if pts.shape[0] < 3:
            continue

        pts = pts.astype(np.int32, copy=False)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        hull = cv2.convexHull(pts)
        if hull is None or len(hull) < 3:
            continue

        # Build a per-hand local mask first so dilation can depend on
        # the size of this specific hand.
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(local_mask, hull, 255)

        x1 = int(np.min(pts[:, 0]))
        y1 = int(np.min(pts[:, 1]))
        x2 = int(np.max(pts[:, 0])) + 1
        y2 = int(np.max(pts[:, 1])) + 1

        bw = x2 - x1
        bh = y2 - y1
        hand_size = max(bw, bh)

        # Always apply a minimal size-aware dilation because the landmark hull
        # tends to under-cover the visible hand silhouette.
        base_radius = max(1, int(round(0.05 * hand_size)))

        # Expansion adds extra dilation on top of the structural base padding.
        extra_radius = max(
            0,
            int(round(max(0.0, float(expansion) - 1.0) * 0.08 * hand_size))
        )

        radius = base_radius + extra_radius

        if radius > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            )
            local_mask = cv2.dilate(local_mask, k, iterations=1)

        # Merge the per-hand mask into the final union mask.
        mask = np.maximum(mask, local_mask)
        selected += 1

    if selected == 0:
        raise RuntimeError(
            f'Could not derive a hand mask for which={which!r}.'
        )

    return (mask > 0)
