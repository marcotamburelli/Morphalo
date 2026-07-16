import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from morphalo.nodes.preprocess.utils.geometry import (expand_clip_bbox,
                                                      segment_capsule_mask,
                                                      tight_mask_bbox)


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
    Resolve a person bbox using MediaPipe Pose as the primary consistency source.

    YOLO is used as a candidate detector, but a YOLO bbox is accepted only when
    it contains all valid MediaPipe pose landmarks. If YOLO fails, or if the
    selected YOLO bbox is inconsistent with the pose, a fallback bbox is derived
    directly from MediaPipe landmarks.

    Parameters
    ----------
    res : Any
        Single YOLO prediction result.

    node_id : str
        Node id used for error messages.

    pose_xy : np.ndarray or None
        MediaPipe pose landmarks in full-image coordinates.

    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.

    pose_fallback_expansion : float, default=1.35
        Expansion factor used when deriving a bbox directly from pose landmarks.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive bbox ``(x1, y1, x2, y2)``.
    """
    if pose_xy is None:
        person_boxes = person_bboxes_xyxy(res, node_id)
        return select_person_bbox_xyxy(
            person_boxes,
            pose_xy=None,
        )

    try:
        person_boxes = person_bboxes_xyxy(res, node_id)
        yolo_bbox = select_person_bbox_xyxy(
            person_boxes,
            pose_xy=pose_xy,
        )

        if bbox_contains_all_valid_pose_points(
            yolo_bbox,
            pose_xy=pose_xy,
        ):
            return yolo_bbox

    except RuntimeError:
        pass

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


def bbox_contains_all_valid_pose_points(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    pose_xy: np.ndarray,
) -> bool:
    """
    Return True if a bbox contains all valid MediaPipe pose landmarks.

    Parameters
    ----------
    bbox_xyxy : tuple[int, int, int, int]
        Candidate bbox in full-image coordinates.

    pose_xy : np.ndarray
        Pose landmarks with shape ``(33, 2)`` in full-image coordinates.
        Missing landmarks are encoded as ``(-1, -1)``.

    Returns
    -------
    bool
        True if all valid landmarks fall inside the bbox.
    """
    if pose_xy is None:
        return False

    x1, y1, x2, y2 = bbox_xyxy

    valid = (
        (pose_xy[:, 0] >= 0) &
        (pose_xy[:, 1] >= 0)
    )
    pts = pose_xy[valid]

    if pts.shape[0] == 0:
        return False

    inside = (
        (pts[:, 0] >= x1) &
        (pts[:, 0] < x2) &
        (pts[:, 1] >= y1) &
        (pts[:, 1] < y2)
    )

    return bool(np.all(inside))


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


def head_area_xyxy_from_pose(
    pose_xy: np.ndarray,
    *,
    full_w: int,
    full_h: int,
    expansion: float = 1.6,
) -> tuple[int, int, int, int]:
    """
    Derive a pose-guided upper-body area suitable for downstream face detection.
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

    face_ids = [0, 2, 5, 7, 8]
    face_pts = [
        pose_xy[i].astype(np.float32)
        for i in face_ids
        if _valid(i)
    ]

    has_torso = all(_valid(i) for i in [11, 12, 23, 24])

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

        if face_pts:
            face_pts_arr = np.stack(face_pts, axis=0)
            face_center = np.mean(face_pts_arr, axis=0)
            top_center = 0.7 * face_center + 0.3 * (
                shoulder_center + head_dir * (0.35 * torso_len)
            )

            face_y_min = float(np.min(face_pts_arr[:, 1]))
            top_y = min(
                face_y_min - 0.35 * shoulder_width,
                top_center[1] - 0.8 * shoulder_width,
            )
            x_center = float(top_center[0])
        else:
            top_center = shoulder_center + head_dir * (0.55 * torso_len)
            top_y = float(top_center[1] - 0.6 * shoulder_width)
            x_center = float(top_center[0])

        bottom_center = hip_center
        bottom_y = float(bottom_center[1] + 0.25 * torso_len)

        region_width = max(
            1.35 * shoulder_width,
            1.15 * hip_width,
            0.9 * torso_len,
        ) * float(expansion)

        half_w = max(12.0, 0.5 * region_width)

        return _clip_box(
            x_center - half_w,
            top_y,
            x_center + half_w,
            bottom_y,
        )

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

        half_w = max(8.0, 1.2 * face_span * float(expansion))
        top_pad = 0.9 * face_span * float(expansion)
        bottom_pad = 1.8 * face_span * float(expansion)

        return _clip_box(
            cx - half_w,
            cy - top_pad,
            cx + half_w,
            cy + bottom_pad,
        )

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

    def _hand_horizontal_centers(indices: list[int]) -> list[tuple[int, float]]:
        """
        Return ``(idx, bbox_center_x)`` for hands with valid landmarks.
        """
        centers: list[tuple[int, float]] = []

        for i in indices:
            hand_xy = hands.xy[i]
            valid = (
                (hand_xy[:, 0] >= 0) &
                (hand_xy[:, 1] >= 0)
            )
            pts = hand_xy[valid]
            if pts.shape[0] == 0:
                continue

            x1 = float(np.min(pts[:, 0]))
            x2 = float(np.max(pts[:, 0]))
            centers.append((i, 0.5 * (x1 + x2)))

        return centers

    def _pick_extreme_hand(indices: list[int]) -> list[int]:
        """
        Pick exactly one hand by observer-side bbox center.
        """
        centers = _hand_horizontal_centers(indices)

        if not centers:
            return []

        centers = sorted(centers, key=lambda item: item[1])

        if which == 'left':
            return [centers[0][0]]

        return [centers[-1][0]]

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
        return _pick_extreme_hand(selected)

    # Fallback: use horizontal image position only if at least two hands
    # are geometrically available. With a single detected hand, returning it
    # would be ambiguous and could silently select the wrong side.
    geometric = _hand_horizontal_centers(list(range(n_hands)))

    if len(geometric) < 2:
        raise RuntimeError(
            f'Could not reliably select a hand for which={which!r}: '
            'handedness did not match and fewer than two hands were detected.'
        )

    return _pick_extreme_hand([item[0] for item in geometric])


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
          hand appearing on the left side of the image
        - ``'right'``:
          hand appearing on the right side of the image
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
          hand appearing on the left side of the image
        - ``'right'``:
          hand appearing on the right side of the image
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


_FOOT_LANDMARKS = {
    'anatomical-left': {
        'knee': 25,
        'ankle': 27,
        'heel': 29,
        'foot_index': 31,
    },
    'anatomical-right': {
        'knee': 26,
        'ankle': 28,
        'heel': 30,
        'foot_index': 32,
    },
}
_FOOT_LENGTH_FROM_FOREARM_WEIGHT = 0.45
_FOOT_LENGTH_FROM_HIP_WEIGHT = 0.25
_FOOT_LENGTH_FROM_SHOULDER_WEIGHT = 0.20
_FOOT_LENGTH_FROM_HEAD_WEIGHT = 0.10
_FOOT_LENGTH_SAFETY = 1.30
_FOOT_LENGTH_MIN_PX = 18.0
_FOOT_CROSSED_LEG_ANKLE_NEAR_AXIS_RATIO = 0.16
_FOOT_CROSSED_LEG_ANKLE_NEAR_AXIS_MIN_PX = 10.0
_FOOT_CROSSED_LEG_ANKLE_NEAR_AXIS_MAX_PX = 32.0
_ARM_LANDMARKS = {
    'anatomical-left': {
        'shoulder': 11,
        'elbow': 13,
        'wrist': 15,
    },
    'anatomical-right': {
        'shoulder': 12,
        'elbow': 14,
        'wrist': 16,
    },
}
_ARM_SHOULDER_RADIUS_EXPANSION = 1.15
_LEG_LANDMARKS = {
    'anatomical-left': {
        'hip': 23,
        'knee': 25,
        'ankle': 27,
    },
    'anatomical-right': {
        'hip': 24,
        'knee': 26,
        'ankle': 28,
    },
}
_LEG_HIP_RADIUS_EXPANSION = 1.10


@dataclass(frozen=True)
class SamRegion:
    """
    Target-local SAM prompt region.

    This is a generic region contract for target pipelines that segment one or
    more local regions independently before unioning their masks. Foot
    segmentation is the first caller, but the same shape can support future arm
    or leg regions.

    ``base_bbox`` is the target-local geometry used for crop/debug purposes.
    ``prompt_bbox`` is the local SAM search area and may be wider than
    ``base_bbox`` when contextual pixels improve segmentation.

    ``point_coords`` / ``point_labels`` are the normal SAM prompt for the
    region. For feet, ankle and foot_index are positive points, optional
    image-aware positives may be inserted by the caller, and an optional point
    from the opposite foot can be negative when it is clearly outside
    ``base_bbox``.

    ``probe_point`` is deliberately kept separate from the normal prompt.
    Callers can use it for a target-specific probe pass before deciding whether
    it should become a negative prompt. For feet, this is the adaptive
    lower-leg continuity probe.

    Parameters
    ----------
    side : str
        Region side or identity label. For foot regions this is usually
        ``'anatomical-left'`` or ``'anatomical-right'``.
    base_bbox : tuple[int, int, int, int]
        Target-local bbox before prompt expansion. This is the red target/base
        box in debug overlays.
    prompt_bbox : tuple[int, int, int, int]
        Local bbox passed to SAM. This is the cyan prompt/search box in debug
        overlays.
    point_coords : list[list[float]] or None
        SAM point coordinates for the normal region prompt.
    point_labels : list[int] or None
        SAM point labels aligned with ``point_coords``. ``1`` means positive and
        ``0`` means negative.
    probe_point : list[float] or None
        Optional target-specific probe point inside ``prompt_bbox``.
    """
    side: str
    base_bbox: tuple[int, int, int, int]
    prompt_bbox: tuple[int, int, int, int]
    point_coords: Optional[list[list[float]]]
    point_labels: Optional[list[int]]
    probe_point: Optional[list[float]]


@dataclass(frozen=True)
class LimbRegionGeometry:
    """
    Geometry required to segment and refine one limb region.

    Parameters
    ----------
    sam_region : SamRegion
        SAM crop and point prompts for the limb.
    prior_mask : np.ndarray
        Coarse full-frame mask used only to neutralize pixels before SAM.
    skeleton_mask : np.ndarray
        Thin full-frame proximal-to-distal centerline used to filter Canny
        edges by distance from the expected limb axis.
    synthetic_barrier_mask : np.ndarray
        Thin full-frame artificial barriers closing the limb at proximal and
        distal joints. These pixels are used only as flood-fill barriers.
    joint_circle_mask : np.ndarray
        Filled full-frame proximal-joint restoration domain. Only pixels
        selected by SAM inside this circle are restored after flood-fill.
    joint_quadrant_mask : np.ndarray
        Full-frame external proximal-joint quadrant. Only the portion also
        selected by SAM is restored after flood-fill.
    tube_radius : float
        Radius used to construct the coarse limb tube.
    """
    sam_region: SamRegion
    prior_mask: np.ndarray
    skeleton_mask: np.ndarray
    synthetic_barrier_mask: np.ndarray
    joint_circle_mask: np.ndarray
    joint_quadrant_mask: np.ndarray
    tube_radius: float


def _valid_pose_point(pose_xy: np.ndarray, idx: int) -> Optional[np.ndarray]:
    """
    Return a valid MediaPipe pose point as ``float32`` or ``None``.

    Parameters
    ----------
    pose_xy : np.ndarray
        Pose landmarks in image coordinates, with shape ``(N, 2)``.
        Missing or weak landmarks are expected to be encoded as ``(-1, -1)``.
    idx : int
        Landmark index to read.

    Returns
    -------
    np.ndarray or None
        The selected point as ``(x, y)`` with dtype ``float32`` when the
        landmark exists and is non-negative; otherwise ``None``.
    """
    if idx >= pose_xy.shape[0]:
        return None

    p = pose_xy[idx]
    if p[0] < 0 or p[1] < 0:
        return None

    return p.astype(np.float32, copy=False)


def _select_arm_sides(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
) -> list[str]:
    """
    Select anatomical arm sides using image/viewer target semantics.
    """
    if which not in ('left', 'right', 'both'):
        raise ValueError(
            f"Invalid which={which!r}; expected 'left', 'right', or 'both'."
        )

    sides = list(_ARM_LANDMARKS.keys())
    if which == 'both':
        return sides

    visible: list[tuple[float, str]] = []
    for side, ids in _ARM_LANDMARKS.items():
        shoulder = _valid_pose_point(pose_xy, ids['shoulder'])
        if shoulder is not None:
            visible.append((float(shoulder[0]), side))

    if not visible:
        raise RuntimeError('No valid shoulder landmarks for arm target.')

    visible.sort(key=lambda item: item[0])
    if len(visible) >= 2:
        return [visible[0][1] if which == 'left' else visible[-1][1]]

    _, w = image_shape[:2]
    center_x, side = visible[0]
    if which == 'left' and center_x <= 0.5 * float(w):
        return [side]
    if which == 'right' and center_x >= 0.5 * float(w):
        return [side]

    raise RuntimeError(
        f'Could not reliably select {which} arm from a single visible shoulder.'
    )


def arm_segments_from_landmarks(
    pose_xy: np.ndarray,
    *,
    side: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return valid pose landmark segments for one anatomical arm side.
    """
    if side not in _ARM_LANDMARKS:
        raise ValueError(f'Unknown arm side {side!r}.')

    ids = _ARM_LANDMARKS[side]
    shoulder = _valid_pose_point(pose_xy, ids['shoulder'])
    elbow = _valid_pose_point(pose_xy, ids['elbow'])
    wrist = _valid_pose_point(pose_xy, ids['wrist'])

    if shoulder is None:
        return []

    chain = [shoulder]
    if elbow is not None:
        chain.append(elbow)
    if wrist is not None:
        chain.append(wrist)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(chain, chain[1:]):
        if float(np.linalg.norm(end - start)) >= 2.0:
            segments.append((start, end))

    return segments


def _estimate_shoulder_radius(
    silhouette: np.ndarray,
    *,
    shoulder: np.ndarray,
    other_shoulder: np.ndarray,
    elbow: Optional[np.ndarray],
    image_shape: tuple[int, ...],
) -> float:
    """
    Estimate shoulder-cap radius by probing the person silhouette outward.
    """
    h, w = image_shape[:2]
    shoulder_width = float(np.linalg.norm(shoulder - other_shoulder))
    fallback = max(12.0, shoulder_width * 0.35)

    external = shoulder - other_shoulder
    external_len = float(np.linalg.norm(external))
    if external_len < 1.0:
        return fallback

    external = external / external_len
    max_probe = max(fallback * 2.0, shoulder_width * 0.9)
    last_inside: Optional[float] = None

    for dist in np.linspace(0.0, max_probe, num=max(8, int(max_probe))):
        p = shoulder + external * float(dist)
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        if px < 0 or px >= w or py < 0 or py >= h:
            break

        if silhouette[py, px]:
            last_inside = float(dist)
        elif last_inside is not None:
            break

    if last_inside is None or last_inside < 2.0:
        return fallback

    upper_arm_len = (
        float(np.linalg.norm(elbow - shoulder))
        if elbow is not None
        else shoulder_width * 0.75
    )
    min_radius = max(8.0, shoulder_width * 0.18)
    max_radius = max(
        min_radius + 1.0,
        min(shoulder_width * 0.70, upper_arm_len * 0.75),
    )
    expanded = last_inside * _ARM_SHOULDER_RADIUS_EXPANSION
    return float(np.clip(expanded, min_radius, max_radius))


def _arm_sam_points_for_side(
    pose_xy: np.ndarray,
    *,
    side: str,
    prior_mask: np.ndarray,
    prompt_bbox: tuple[int, int, int, int],
    additional_positive_points: Optional[list[list[float]]] = None,
) -> tuple[Optional[list[list[float]]], Optional[list[int]]]:
    """
    Build SAM point prompts for one arm region.
    """
    ids = _ARM_LANDMARKS[side]
    points: list[list[float]] = []
    labels: list[int] = []
    h, w = prior_mask.shape[:2]

    for name in ('shoulder', 'elbow', 'wrist'):
        p = _valid_pose_point(pose_xy, ids[name])
        if p is None:
            continue
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        if 0 <= px < w and 0 <= py < h and prior_mask[py, px]:
            points.append([float(p[0]), float(p[1])])
            labels.append(1)

    if additional_positive_points:
        for point in additional_positive_points:
            if len(point) < 2:
                continue
            px, py = point[:2]
            px_i = int(round(float(px)))
            py_i = int(round(float(py)))
            if 0 <= px_i < w and 0 <= py_i < h and prior_mask[py_i, px_i]:
                points.append([float(px), float(py)])
                labels.append(1)

    shoulder = _valid_pose_point(pose_xy, ids['shoulder'])
    if shoulder is not None:
        other_side = (
            'anatomical-right'
            if side == 'anatomical-left'
            else 'anatomical-left'
        )
        other_shoulder = _valid_pose_point(
            pose_xy,
            _ARM_LANDMARKS[other_side]['shoulder'],
        )
        left_shoulder = _valid_pose_point(
            pose_xy,
            _ARM_LANDMARKS['anatomical-left']['shoulder'],
        )
        right_shoulder = _valid_pose_point(
            pose_xy,
            _ARM_LANDMARKS['anatomical-right']['shoulder'],
        )
        negative_candidates = []
        if other_shoulder is not None:
            negative_candidates.append(other_shoulder)
        if left_shoulder is not None and right_shoulder is not None:
            negative_candidates.append((left_shoulder + right_shoulder) * 0.5)

        px1, py1, px2, py2 = prompt_bbox
        for p in negative_candidates:
            px = int(round(float(p[0])))
            py = int(round(float(p[1])))
            if not (px1 <= px < px2 and py1 <= py < py2):
                continue
            if 0 <= px < w and 0 <= py < h and prior_mask[py, px]:
                continue
            points.append([float(p[0]), float(p[1])])
            labels.append(0)

    if not any(label == 1 for label in labels):
        ys, xs = np.where(prior_mask)
        if xs.size > 0:
            points.append([float(xs.mean()), float(ys.mean())])
            labels.append(1)

    if not points:
        return None, None

    return points, labels


def arm_regions_from_landmarks(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    silhouette: np.ndarray,
    which: str,
    expansion: float = 1.0,
    additional_positive_points_by_side: Optional[
        dict[str, list[list[float]]]
    ] = None,
) -> list[LimbRegionGeometry]:
    """
    Build the geometry required to segment one or more arm regions.

    The function converts MediaPipe shoulder, elbow, and wrist landmarks into
    independent arm-local segmentation geometries.

    For each selected arm, it builds:

    - a coarse filled prior used only to neutralize pixels before SAM;
    - a thin shoulder-to-elbow-to-wrist skeleton;
    - thin synthetic barriers closing the arm at the shoulder and wrist;
    - a filled shoulder circle that can be restored after Canny refinement;
    - an external shoulder quadrant used to restore only the portion selected
      by SAM;
    - a generic ``SamRegion`` containing the local crop and point prompts.

    The coarse prior is not a final arm mask and must not be restored after
    flood-fill. It exists only to restrict the visual content shown to SAM.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe Pose landmarks in full-image coordinates. Missing landmarks
        must be encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only height and width are used.
    silhouette : np.ndarray
        Full-frame boolean person silhouette.
    which : {'left', 'right', 'both'}
        Requested arm selection in image/viewer perspective.
    expansion : float, default=1.0
        Multiplicative expansion applied to the estimated arm radius.
        Values below ``1.0`` are treated as ``1.0``.
    additional_positive_points_by_side : dict or None, optional
        Additional positive SAM points keyed by anatomical side.

    Returns
    -------
    list[LimbRegionGeometry]
        One complete geometry object for every selected arm.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If the shoulder landmarks or arm geometry are insufficient.
    """
    import cv2

    h, w = image_shape[:2]
    silhouette_bool = silhouette.astype(bool)

    selected_sides = _select_arm_sides(
        pose_xy,
        image_shape,
        which=which,
    )

    left_shoulder = _valid_pose_point(
        pose_xy,
        _ARM_LANDMARKS['anatomical-left']['shoulder'],
    )
    right_shoulder = _valid_pose_point(
        pose_xy,
        _ARM_LANDMARKS['anatomical-right']['shoulder'],
    )

    if left_shoulder is None or right_shoulder is None:
        raise RuntimeError(
            'Arm targets require both shoulder landmarks.'
        )

    shoulder_axis = right_shoulder - left_shoulder
    shoulder_width = float(np.linalg.norm(shoulder_axis))

    if shoulder_width < 4.0:
        raise RuntimeError(
            'Shoulder landmarks are too close for arm geometry.'
        )

    shoulder_axis_len = max(shoulder_width, 1e-6)
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)

    head_point = _valid_pose_point(pose_xy, 0)
    if head_point is None:
        head_point = shoulder_mid + np.asarray(
            [0.0, -shoulder_width],
            dtype=np.float32,
        )

    yy, xx = np.mgrid[0:h, 0:w]

    # Determine which side of the shoulder axis contains the head.
    head_cross = (
        float(shoulder_axis[0])
        * float(head_point[1] - shoulder_mid[1])
        - float(shoulder_axis[1])
        * float(head_point[0] - shoulder_mid[0])
    )
    head_side = 1.0 if head_cross >= 0.0 else -1.0

    rel_mid_x = xx - float(shoulder_mid[0])
    rel_mid_y = yy - float(shoulder_mid[1])
    grid_cross_from_shoulders = (
        float(shoulder_axis[0]) * rel_mid_y
        - float(shoulder_axis[1]) * rel_mid_x
    )

    arm_geometries: list[LimbRegionGeometry] = []

    for side in selected_sides:
        ids = _ARM_LANDMARKS[side]

        shoulder = _valid_pose_point(
            pose_xy,
            ids['shoulder'],
        )
        elbow = _valid_pose_point(
            pose_xy,
            ids['elbow'],
        )
        wrist = _valid_pose_point(
            pose_xy,
            ids['wrist'],
        )

        if shoulder is None:
            continue

        other_shoulder = (
            right_shoulder
            if side == 'anatomical-left'
            else left_shoulder
        )

        segments = arm_segments_from_landmarks(
            pose_xy,
            side=side,
        )
        if not segments:
            continue

        radius = _estimate_shoulder_radius(
            silhouette_bool,
            shoulder=shoulder,
            other_shoulder=other_shoulder,
            elbow=elbow,
            image_shape=image_shape,
        )
        radius *= max(1.0, float(expansion))

        # -------------------------------------------------------------
        # Arm tube
        # -------------------------------------------------------------
        tube_mask = np.zeros((h, w), dtype=bool)

        for start, end in segments:
            tube_mask |= segment_capsule_mask(
                image_shape,
                start,
                end,
                radius,
            )

        # -------------------------------------------------------------
        # Thin centerline used to filter Canny components by distance.
        # -------------------------------------------------------------
        skeleton_u8 = np.zeros((h, w), dtype=np.uint8)

        for start, end in segments:
            start_xy = tuple(
                int(value)
                for value in np.rint(start)
            )
            end_xy = tuple(
                int(value)
                for value in np.rint(end)
            )

            cv2.line(
                skeleton_u8,
                start_xy,
                end_xy,
                255,
                thickness=1,
                lineType=cv2.LINE_8,
            )

        for point in (shoulder, elbow, wrist):
            if point is None:
                continue

            point_xy = tuple(
                int(value)
                for value in np.rint(point)
            )

            cv2.circle(
                skeleton_u8,
                point_xy,
                1,
                255,
                thickness=-1,
                lineType=cv2.LINE_8,
            )

        skeleton_mask = skeleton_u8 > 0

        # -------------------------------------------------------------
        # Filled shoulder circle.
        #
        # This is trusted geometric support and may be restored after the
        # Canny/flood-fill stage.
        # -------------------------------------------------------------
        shoulder_circle_mask = (
            np.hypot(
                xx - float(shoulder[0]),
                yy - float(shoulder[1]),
            )
            <= radius
        )

        # -------------------------------------------------------------
        # External shoulder quadrant.
        #
        # First keep the half-plane on the head side of the shoulder line.
        # Then keep the outward side of the perpendicular passing through the
        # selected shoulder.
        # -------------------------------------------------------------
        head_halfplane = (
            head_side * grid_cross_from_shoulders
        ) >= -0.35 * radius * shoulder_axis_len

        outward_vector = shoulder - other_shoulder
        outward_length = float(np.linalg.norm(outward_vector))

        if outward_length < 1e-6:
            continue

        outward_unit = outward_vector / outward_length

        rel_shoulder_x = xx - float(shoulder[0])
        rel_shoulder_y = yy - float(shoulder[1])

        outward_projection = (
            rel_shoulder_x * float(outward_unit[0])
            + rel_shoulder_y * float(outward_unit[1])
        )

        outward_halfplane = (
            outward_projection >= -0.35 * radius
        )

        shoulder_quadrant = (
            head_halfplane
            & outward_halfplane
        )

        # Only actual person-silhouette pixels in the shoulder quadrant may be
        # restored from the SAM result later.
        shoulder_quadrant_mask = (
            silhouette_bool
            & shoulder_quadrant
        )

        # -------------------------------------------------------------
        # Additional torso-side rejection used only for the coarse SAM prior.
        # -------------------------------------------------------------
        first_segment_start, first_segment_end = segments[0]
        upper_arm_axis = first_segment_end - first_segment_start
        upper_arm_axis_len = float(np.linalg.norm(upper_arm_axis))

        torso_center = shoulder_mid

        if upper_arm_axis_len >= 2.0:
            cross_to_torso = (
                float(upper_arm_axis[0])
                * float(torso_center[1] - shoulder[1])
                - float(upper_arm_axis[1])
                * float(torso_center[0] - shoulder[0])
            )

            cross_grid = (
                float(upper_arm_axis[0]) * rel_shoulder_y
                - float(upper_arm_axis[1]) * rel_shoulder_x
            )

            if abs(cross_to_torso) > 1e-3:
                torso_side = (
                    1.0
                    if cross_to_torso > 0.0
                    else -1.0
                )

                arm_outer_halfplane = (
                    torso_side * cross_grid
                    <= radius * upper_arm_axis_len
                )
            else:
                arm_outer_halfplane = np.ones(
                    (h, w),
                    dtype=bool,
                )
        else:
            arm_outer_halfplane = np.ones(
                (h, w),
                dtype=bool,
            )

        silhouette_branch = (
            shoulder_quadrant_mask
            & arm_outer_halfplane
        )

        # -------------------------------------------------------------
        # Coarse prior shown to SAM.
        #
        # This mask is used only to neutralize pixels outside the plausible arm
        # region. It must not be restored after flood-fill.
        # -------------------------------------------------------------
        arm_support_mask = (
            silhouette_bool
            & (
                tube_mask
                | shoulder_circle_mask
            )
        )

        sam_prior_mask = (
            silhouette_branch
            | arm_support_mask
        )

        # Remove geometry extending past the wrist.
        if wrist is not None and elbow is not None:
            forearm_vector = wrist - elbow
            forearm_length = float(
                np.linalg.norm(forearm_vector)
            )

            if forearm_length >= 2.0:
                forearm_unit = (
                    forearm_vector / forearm_length
                )

                after_wrist = (
                    (
                        xx - float(wrist[0])
                    ) * float(forearm_unit[0])
                    + (
                        yy - float(wrist[1])
                    ) * float(forearm_unit[1])
                ) > radius * 0.20

                sam_prior_mask &= ~after_wrist

                # Restore only the portion of the shoulder circle that belongs to the
                # segmented person silhouette.
                sam_prior_mask |= (
                    shoulder_circle_mask
                    & silhouette_bool
                )

        if not np.any(sam_prior_mask):
            continue

        # -------------------------------------------------------------
        # Synthetic flood-fill barriers.
        #
        # These are thin lines only. They are never restored as mask regions.
        # -------------------------------------------------------------
        synthetic_barriers_u8 = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        # Shoulder barrier:
        #
        # Close the arm flood-fill domain with the circumference of the same
        # shoulder circle that will later be restored as trusted geometric support.
        # The filled circle and its boundary have deliberately different roles:
        #
        # - the circumference is a flood-fill barrier;
        # - the filled circle is added back to the refined arm mask afterward.
        shoulder_center_xy = tuple(
            int(value)
            for value in np.rint(shoulder)
        )
        shoulder_radius_px = max(
            1,
            int(round(radius)),
        )

        cv2.circle(
            synthetic_barriers_u8,
            shoulder_center_xy,
            shoulder_radius_px,
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )

        # Wrist barrier: perpendicular to the final forearm segment.
        if elbow is not None and wrist is not None:
            forearm_vector = wrist - elbow
            forearm_length = float(
                np.linalg.norm(forearm_vector)
            )

            if forearm_length >= 2.0:
                forearm_unit = (
                    forearm_vector / forearm_length
                )
                forearm_normal = np.asarray(
                    [
                        -forearm_unit[1],
                        forearm_unit[0],
                    ],
                    dtype=np.float32,
                )

                wrist_barrier_start = (
                    wrist - forearm_normal * radius
                )
                wrist_barrier_end = (
                    wrist + forearm_normal * radius
                )

                cv2.line(
                    synthetic_barriers_u8,
                    tuple(
                        int(value)
                        for value in np.rint(
                            wrist_barrier_start
                        )
                    ),
                    tuple(
                        int(value)
                        for value in np.rint(
                            wrist_barrier_end
                        )
                    ),
                    255,
                    thickness=1,
                    lineType=cv2.LINE_8,
                )

        synthetic_barrier_mask = (
            synthetic_barriers_u8 > 0
        )

        # -------------------------------------------------------------
        # SAM region and point prompts.
        # -------------------------------------------------------------
        base_bbox = tight_mask_bbox(
            sam_prior_mask.astype(np.uint8)
        )

        prompt_bbox = expand_clip_bbox(
            *base_bbox,
            w,
            h,
            max(
                0.08,
                min(
                    0.30,
                    0.12 * max(
                        1.0,
                        float(expansion),
                    ),
                ),
            ),
        )

        point_coords, point_labels = (
            _arm_sam_points_for_side(
                pose_xy,
                side=side,
                prior_mask=sam_prior_mask,
                prompt_bbox=prompt_bbox,
                additional_positive_points=(
                    additional_positive_points_by_side
                    or {}
                ).get(side),
            )
        )

        sam_region = SamRegion(
            side=side,
            base_bbox=base_bbox,
            prompt_bbox=prompt_bbox,
            point_coords=point_coords,
            point_labels=point_labels,
            probe_point=None,
        )

        arm_geometries.append(
            LimbRegionGeometry(
                sam_region=sam_region,
                prior_mask=sam_prior_mask,
                skeleton_mask=skeleton_mask,
                synthetic_barrier_mask=(
                    synthetic_barrier_mask
                ),
                joint_circle_mask=(
                    shoulder_circle_mask
                ),
                joint_quadrant_mask=(
                    shoulder_quadrant_mask
                ),
                tube_radius=float(radius),
            )
        )

    if not arm_geometries:
        missing_landmarks = [
            name
            for name, point in (
                ('shoulder', shoulder),
                ('elbow', elbow),
                ('wrist', wrist),
            )
            if point is None
        ]

        raise RuntimeError(
            f'Could not derive {side} arm geometry: '
            f'missing usable landmarks {missing_landmarks!r}.'
        )

    return arm_geometries


def _select_leg_sides(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
) -> list[str]:
    """
    Select anatomical leg sides using image/viewer target semantics.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe Pose landmarks in full-image coordinates. Missing landmarks
        must be encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only height and width are used.
    which : {'left', 'right', 'both'}
        Requested leg selection in image/viewer perspective.

    Returns
    -------
    list[str]
        Anatomical side identifiers selected for segmentation.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If no usable hip or knee landmarks are available, or if a side-specific
        request cannot be resolved from a single visible leg.
    """
    if which not in ('left', 'right', 'both'):
        raise ValueError(
            f"Invalid which={which!r}; expected 'left', 'right', or 'both'."
        )

    sides = list(_LEG_LANDMARKS.keys())
    if which == 'both':
        return sides

    visible: list[tuple[float, str]] = []
    for side, ids in _LEG_LANDMARKS.items():
        hip = _valid_pose_point(pose_xy, ids['hip'])
        knee = _valid_pose_point(pose_xy, ids['knee'])
        anchor = hip if hip is not None else knee
        if anchor is not None:
            visible.append((float(anchor[0]), side))

    if not visible:
        raise RuntimeError('No valid hip or knee landmarks for leg target.')

    visible.sort(key=lambda item: item[0])
    if len(visible) >= 2:
        return [visible[0][1] if which == 'left' else visible[-1][1]]

    _, w = image_shape[:2]
    center_x, side = visible[0]
    if which == 'left' and center_x <= 0.5 * float(w):
        return [side]
    if which == 'right' and center_x >= 0.5 * float(w):
        return [side]

    raise RuntimeError(
        f'Could not reliably select {which} leg from a single visible leg.'
    )


def leg_segments_from_landmarks(
    pose_xy: np.ndarray,
    *,
    side: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return valid pose landmark segments for one anatomical leg side.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe Pose landmarks in full-image coordinates. Missing landmarks
        must be encoded as ``(-1, -1)``.
    side : str
        Anatomical side key, either ``'anatomical-left'`` or
        ``'anatomical-right'``.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Consecutive valid landmark segments, normally ``hip -> knee`` and
        ``knee -> ankle``. Missing distal landmarks are skipped; if the hip is
        missing, the list is empty.

    Raises
    ------
    ValueError
        If ``side`` is not a supported anatomical leg side.
    """
    if side not in _LEG_LANDMARKS:
        raise ValueError(f'Unknown leg side {side!r}.')

    ids = _LEG_LANDMARKS[side]
    hip = _valid_pose_point(pose_xy, ids['hip'])
    knee = _valid_pose_point(pose_xy, ids['knee'])
    ankle = _valid_pose_point(pose_xy, ids['ankle'])

    if hip is None:
        return []

    chain = [hip]
    if knee is not None:
        chain.append(knee)
    if ankle is not None:
        chain.append(ankle)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(chain, chain[1:]):
        if float(np.linalg.norm(end - start)) >= 2.0:
            segments.append((start, end))

    return segments


def _estimate_hip_radius(
    silhouette: np.ndarray,
    *,
    hip: np.ndarray,
    other_hip: np.ndarray,
    knee: Optional[np.ndarray],
    image_shape: tuple[int, ...],
) -> float:
    """
    Estimate hip-cap radius by probing the person silhouette outward.

    The radius starts from the image-space distance between the selected hip and
    the opposite hip. The function probes outward from the selected hip until it
    leaves the person silhouette, expands that observed distance slightly, and
    clamps it to proportions derived from hip width and upper-leg length.

    Parameters
    ----------
    silhouette : np.ndarray
        Full-frame boolean person silhouette.
    hip : np.ndarray
        Selected hip point in image coordinates.
    other_hip : np.ndarray
        Opposite hip point in image coordinates.
    knee : np.ndarray or None
        Same-side knee point, used to bound the radius when available.
    image_shape : tuple[int, ...]
        Source image shape. Only height and width are used.

    Returns
    -------
    float
        Estimated hip-cap radius in pixels.
    """
    h, w = image_shape[:2]
    hip_width = float(np.linalg.norm(hip - other_hip))
    fallback = max(12.0, hip_width * 0.32)

    external = hip - other_hip
    external_len = float(np.linalg.norm(external))
    if external_len < 1.0:
        return fallback

    external = external / external_len
    max_probe = max(fallback * 2.0, hip_width * 0.85)
    last_inside: Optional[float] = None

    for dist in np.linspace(0.0, max_probe, num=max(8, int(max_probe))):
        p = hip + external * float(dist)
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        if px < 0 or px >= w or py < 0 or py >= h:
            break

        if silhouette[py, px]:
            last_inside = float(dist)
        elif last_inside is not None:
            break

    if last_inside is None or last_inside < 2.0:
        return fallback

    upper_leg_len = (
        float(np.linalg.norm(knee - hip))
        if knee is not None
        else hip_width * 1.1
    )
    min_radius = max(8.0, hip_width * 0.18)
    max_radius = max(
        min_radius + 1.0,
        min(hip_width * 0.65, upper_leg_len * 0.45),
    )
    expanded = last_inside * _LEG_HIP_RADIUS_EXPANSION
    return float(np.clip(expanded, min_radius, max_radius))


def _leg_sam_points_for_side(
    pose_xy: np.ndarray,
    *,
    side: str,
    prior_mask: np.ndarray,
    prompt_bbox: tuple[int, int, int, int],
    additional_positive_points: Optional[list[list[float]]] = None,
) -> tuple[Optional[list[list[float]]], Optional[list[int]]]:
    """
    Build SAM point prompts for one leg region.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe Pose landmarks in full-image coordinates. Missing landmarks
        must be encoded as ``(-1, -1)``.
    side : str
        Anatomical side key, either ``'anatomical-left'`` or
        ``'anatomical-right'``.
    prior_mask : np.ndarray
        Full-frame boolean leg prior used to validate prompt points.
    prompt_bbox : tuple[int, int, int, int]
        End-exclusive SAM prompt bbox. Negative pelvis prompts are retained
        only when they fall inside this bbox and outside ``prior_mask``.
    additional_positive_points : list[list[float]] or None, optional
        Extra positive points, typically chromatic run centers sampled along the
        leg skeleton.

    Returns
    -------
    tuple[list[list[float]] | None, list[int] | None]
        SAM point coordinates and labels. Returns ``(None, None)`` when no
        usable positive or negative prompt can be built.
    """
    ids = _LEG_LANDMARKS[side]
    points: list[list[float]] = []
    labels: list[int] = []
    h, w = prior_mask.shape[:2]

    for name in ('hip', 'knee', 'ankle'):
        p = _valid_pose_point(pose_xy, ids[name])
        if p is None:
            continue
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        if 0 <= px < w and 0 <= py < h and prior_mask[py, px]:
            points.append([float(p[0]), float(p[1])])
            labels.append(1)

    if additional_positive_points:
        for point in additional_positive_points:
            if len(point) < 2:
                continue
            px, py = point[:2]
            px_i = int(round(float(px)))
            py_i = int(round(float(py)))
            if 0 <= px_i < w and 0 <= py_i < h and prior_mask[py_i, px_i]:
                points.append([float(px), float(py)])
                labels.append(1)

    other_side = (
        'anatomical-right'
        if side == 'anatomical-left'
        else 'anatomical-left'
    )
    negative_candidates = []
    other_hip = _valid_pose_point(
        pose_xy,
        _LEG_LANDMARKS[other_side]['hip'],
    )
    left_hip = _valid_pose_point(
        pose_xy,
        _LEG_LANDMARKS['anatomical-left']['hip'],
    )
    right_hip = _valid_pose_point(
        pose_xy,
        _LEG_LANDMARKS['anatomical-right']['hip'],
    )
    if other_hip is not None:
        negative_candidates.append(other_hip)
    if left_hip is not None and right_hip is not None:
        negative_candidates.append((left_hip + right_hip) * 0.5)

    px1, py1, px2, py2 = prompt_bbox
    for p in negative_candidates:
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        if not (px1 <= px < px2 and py1 <= py < py2):
            continue
        if 0 <= px < w and 0 <= py < h and prior_mask[py, px]:
            continue
        points.append([float(p[0]), float(p[1])])
        labels.append(0)

    if not any(label == 1 for label in labels):
        ys, xs = np.where(prior_mask)
        if xs.size > 0:
            points.append([float(xs.mean()), float(ys.mean())])
            labels.append(1)

    if not points:
        return None, None

    return points, labels


def leg_regions_from_landmarks(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    silhouette: np.ndarray,
    which: str,
    expansion: float = 1.0,
    additional_positive_points_by_side: Optional[
        dict[str, list[list[float]]]
    ] = None,
) -> list[LimbRegionGeometry]:
    """
    Build the geometry required to segment one or more leg regions.

    The function converts MediaPipe hip, knee, and ankle landmarks into
    independent leg-local segmentation geometries.

    For each selected leg, it builds:

    - a coarse filled prior used only to neutralize pixels before SAM;
    - a thin hip-to-knee-to-ankle skeleton;
    - thin synthetic barriers closing the leg at the hip and ankle;
    - a filled hip circle that can be restored after Canny refinement;
    - an external hip quadrant derived from the pelvis line, the side opposite
      the torso, and the outward direction of the selected hip;
    - a generic ``SamRegion`` containing the local crop and point prompts.

    The coarse prior is not a final leg mask and must not be restored after
    flood-fill. It exists only to restrict the visual content shown to SAM.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe Pose landmarks in full-image coordinates. Missing landmarks
        must be encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only height and width are used.
    silhouette : np.ndarray
        Full-frame boolean person silhouette.
    which : {'left', 'right', 'both'}
        Requested leg selection in image/viewer perspective.
    expansion : float, default=1.0
        Multiplicative expansion applied to the estimated leg radius. Values
        below ``1.0`` are treated as ``1.0``.
    additional_positive_points_by_side : dict or None, optional
        Additional positive SAM points keyed by anatomical side.

    Returns
    -------
    list[LimbRegionGeometry]
        One complete geometry object for every selected leg.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If the hip landmarks or leg geometry are insufficient.
    """
    import cv2

    h, w = image_shape[:2]
    silhouette_bool = silhouette.astype(bool)

    selected_sides = _select_leg_sides(
        pose_xy,
        image_shape,
        which=which,
    )

    left_hip = _valid_pose_point(
        pose_xy,
        _LEG_LANDMARKS['anatomical-left']['hip'],
    )
    right_hip = _valid_pose_point(
        pose_xy,
        _LEG_LANDMARKS['anatomical-right']['hip'],
    )

    if left_hip is None or right_hip is None:
        raise RuntimeError(
            'Leg targets require both hip landmarks.'
        )

    pelvis_axis = right_hip - left_hip
    pelvis_width = float(np.linalg.norm(pelvis_axis))

    if pelvis_width < 4.0:
        raise RuntimeError(
            'Hip landmarks are too close for leg geometry.'
        )

    pelvis_axis_len = max(pelvis_width, 1e-6)
    pelvis_mid = 0.5 * (left_hip + right_hip)

    yy, xx = np.mgrid[0:h, 0:w]

    rel_mid_x = xx - float(pelvis_mid[0])
    rel_mid_y = yy - float(pelvis_mid[1])
    grid_cross_from_pelvis = (
        float(pelvis_axis[0]) * rel_mid_y
        - float(pelvis_axis[1]) * rel_mid_x
    )

    # Determine which side of the pelvis axis contains the torso.
    #
    # The leg-facing half-plane is defined as the side opposite the torso.
    # Shoulder landmarks are preferred because they remain meaningful even when
    # the legs are bent, crossed, raised, or partially occluded. The head is used
    # as a fallback, followed by the image-space upward direction.
    shoulder_points = [
        point
        for point in (
            _valid_pose_point(pose_xy, 11),
            _valid_pose_point(pose_xy, 12),
        )
        if point is not None
    ]

    if shoulder_points:
        torso_reference = np.mean(
            np.stack(shoulder_points, axis=0),
            axis=0,
        )
    else:
        torso_reference = _valid_pose_point(
            pose_xy,
            0,
        )

    if torso_reference is None:
        torso_reference = pelvis_mid + np.asarray(
            [0.0, -pelvis_width],
            dtype=np.float32,
        )

    torso_cross = (
        float(pelvis_axis[0])
        * float(torso_reference[1] - pelvis_mid[1])
        - float(pelvis_axis[1])
        * float(torso_reference[0] - pelvis_mid[0])
    )

    leg_geometries: list[LimbRegionGeometry] = []

    for side in selected_sides:
        ids = _LEG_LANDMARKS[side]

        hip = _valid_pose_point(pose_xy, ids['hip'])
        knee = _valid_pose_point(pose_xy, ids['knee'])
        ankle = _valid_pose_point(pose_xy, ids['ankle'])

        if hip is None:
            continue

        other_hip = (
            right_hip
            if side == 'anatomical-left'
            else left_hip
        )

        segments = leg_segments_from_landmarks(
            pose_xy,
            side=side,
        )
        if not segments:
            continue

        radius = _estimate_hip_radius(
            silhouette_bool,
            hip=hip,
            other_hip=other_hip,
            knee=knee,
            image_shape=image_shape,
        )
        radius *= max(1.0, float(expansion))

        tube_mask = np.zeros((h, w), dtype=bool)
        for start, end in segments:
            tube_mask |= segment_capsule_mask(
                image_shape,
                start,
                end,
                radius,
            )

        skeleton_u8 = np.zeros((h, w), dtype=np.uint8)
        for start, end in segments:
            cv2.line(
                skeleton_u8,
                tuple(int(value) for value in np.rint(start)),
                tuple(int(value) for value in np.rint(end)),
                255,
                thickness=1,
                lineType=cv2.LINE_8,
            )

        for point in (hip, knee, ankle):
            if point is None:
                continue
            cv2.circle(
                skeleton_u8,
                tuple(int(value) for value in np.rint(point)),
                1,
                255,
                thickness=-1,
                lineType=cv2.LINE_8,
            )

        skeleton_mask = skeleton_u8 > 0

        hip_circle_mask = (
            np.hypot(
                xx - float(hip[0]),
                yy - float(hip[1]),
            )
            <= radius
        )

        # The torso side becomes geometrically ambiguous when the torso reference
        # lies almost exactly on the pelvis axis. In that rare case the lower
        # half-plane constraint is disabled rather than choosing an arbitrary side.
        if abs(torso_cross) <= 1e-3:
            lower_halfplane = np.ones(
                silhouette_bool.shape,
                dtype=bool,
            )
        else:
            lower_side = (
                -1.0
                if torso_cross >= 0.0
                else 1.0
            )

            lower_halfplane = (
                lower_side * grid_cross_from_pelvis
            ) >= -0.35 * radius * pelvis_axis_len

        outward_vector = hip - other_hip
        outward_length = float(np.linalg.norm(outward_vector))

        if outward_length < 1e-6:
            continue

        outward_unit = outward_vector / outward_length

        rel_hip_x = xx - float(hip[0])
        rel_hip_y = yy - float(hip[1])

        outward_projection = (
            rel_hip_x * float(outward_unit[0])
            + rel_hip_y * float(outward_unit[1])
        )

        outward_halfplane = (
            outward_projection >= -0.35 * radius
        )

        hip_quadrant_mask = (
            silhouette_bool
            & lower_halfplane
            & outward_halfplane
        )

        leg_support_mask = (
            silhouette_bool
            & (
                tube_mask
                | hip_circle_mask
            )
        )

        sam_prior_mask = (
            hip_quadrant_mask
            | leg_support_mask
        )

        if ankle is not None and knee is not None:
            lower_leg_vector = ankle - knee
            lower_leg_length = float(np.linalg.norm(lower_leg_vector))

            if lower_leg_length >= 2.0:
                lower_leg_unit = lower_leg_vector / lower_leg_length

                after_ankle = (
                    (
                        xx - float(ankle[0])
                    ) * float(lower_leg_unit[0])
                    + (
                        yy - float(ankle[1])
                    ) * float(lower_leg_unit[1])
                ) > radius * 0.25

                sam_prior_mask &= ~after_ankle
                sam_prior_mask |= (
                    hip_circle_mask
                    & silhouette_bool
                )

        if not np.any(sam_prior_mask):
            continue

        synthetic_barriers_u8 = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        hip_center_xy = tuple(
            int(value)
            for value in np.rint(hip)
        )
        hip_radius_px = max(
            1,
            int(round(radius)),
        )

        cv2.circle(
            synthetic_barriers_u8,
            hip_center_xy,
            hip_radius_px,
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )

        if knee is not None and ankle is not None:
            lower_leg_vector = ankle - knee
            lower_leg_length = float(np.linalg.norm(lower_leg_vector))

            if lower_leg_length >= 2.0:
                lower_leg_unit = lower_leg_vector / lower_leg_length
                lower_leg_normal = np.asarray(
                    [
                        -lower_leg_unit[1],
                        lower_leg_unit[0],
                    ],
                    dtype=np.float32,
                )

                ankle_barrier_start = (
                    ankle - lower_leg_normal * radius
                )
                ankle_barrier_end = (
                    ankle + lower_leg_normal * radius
                )

                cv2.line(
                    synthetic_barriers_u8,
                    tuple(
                        int(value)
                        for value in np.rint(
                            ankle_barrier_start
                        )
                    ),
                    tuple(
                        int(value)
                        for value in np.rint(
                            ankle_barrier_end
                        )
                    ),
                    255,
                    thickness=1,
                    lineType=cv2.LINE_8,
                )

        synthetic_barrier_mask = (
            synthetic_barriers_u8 > 0
        )

        base_bbox = tight_mask_bbox(
            sam_prior_mask.astype(np.uint8)
        )

        prompt_bbox = expand_clip_bbox(
            *base_bbox,
            w,
            h,
            max(
                0.08,
                min(
                    0.30,
                    0.12 * max(
                        1.0,
                        float(expansion),
                    ),
                ),
            ),
        )

        point_coords, point_labels = (
            _leg_sam_points_for_side(
                pose_xy,
                side=side,
                prior_mask=sam_prior_mask,
                prompt_bbox=prompt_bbox,
                additional_positive_points=(
                    additional_positive_points_by_side
                    or {}
                ).get(side),
            )
        )

        sam_region = SamRegion(
            side=side,
            base_bbox=base_bbox,
            prompt_bbox=prompt_bbox,
            point_coords=point_coords,
            point_labels=point_labels,
            probe_point=None,
        )

        leg_geometries.append(
            LimbRegionGeometry(
                sam_region=sam_region,
                prior_mask=sam_prior_mask,
                skeleton_mask=skeleton_mask,
                synthetic_barrier_mask=(
                    synthetic_barrier_mask
                ),
                joint_circle_mask=(
                    hip_circle_mask
                ),
                joint_quadrant_mask=(
                    hip_quadrant_mask
                ),
                tube_radius=float(radius),
            )
        )

    if not leg_geometries:
        raise RuntimeError(
            f'Could not derive leg geometry for which={which!r}.'
        )

    return leg_geometries


def _point_segment_distance(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Return the 2D distance between ``point`` and segment ``a -> b``.

    Parameters
    ----------
    point : np.ndarray
        Point to measure, as ``(x, y)`` image coordinates.
    a : np.ndarray
        First endpoint of the segment.
    b : np.ndarray
        Second endpoint of the segment.

    Returns
    -------
    float
        Euclidean distance from ``point`` to the closest point on segment
        ``a -> b``. If the segment is degenerate, returns distance to ``a``.
    """
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))

    if ab_len_sq <= 1e-6:
        return float(np.linalg.norm(point - a))

    t = float(np.dot(point - a, ab) / ab_len_sq)
    t = min(1.0, max(0.0, t))
    closest = a + ab * t
    return float(np.linalg.norm(point - closest))


def _segments_intersect_2d(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> bool:
    """
    Return whether two 2D line segments intersect.

    Parameters
    ----------
    a1 : np.ndarray
        First endpoint of the first segment.
    a2 : np.ndarray
        Second endpoint of the first segment.
    b1 : np.ndarray
        First endpoint of the second segment.
    b2 : np.ndarray
        Second endpoint of the second segment.

    Returns
    -------
    bool
        ``True`` when the two closed segments intersect or touch; otherwise
        ``False``.
    """
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) -
                     (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(float(p[0]), float(r[0])) <= float(q[0]) <= max(float(p[0]), float(r[0])) and
            min(float(p[1]), float(r[1])) <= float(
                q[1]) <= max(float(p[1]), float(r[1]))
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0):
        return True

    eps = 1e-6
    return (
        (abs(o1) <= eps and on_segment(a1, b1, a2)) or
        (abs(o2) <= eps and on_segment(a1, b2, a2)) or
        (abs(o3) <= eps and on_segment(b1, a1, b2)) or
        (abs(o4) <= eps and on_segment(b1, a2, b2))
    )


def _lower_legs_are_crossed(pose_xy: np.ndarray) -> bool:
    """
    Return whether the projected lower-leg segments cross in image space.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.

    Returns
    -------
    bool
        ``True`` when left and right knee->ankle segments intersect in the
        image projection, or when their horizontal ordering flips between knees
        and ankles. Missing lower-leg landmarks return ``False``.
    """
    left = _FOOT_LANDMARKS['anatomical-left']
    right = _FOOT_LANDMARKS['anatomical-right']
    left_knee = _valid_pose_point(pose_xy, left['knee'])
    left_ankle = _valid_pose_point(pose_xy, left['ankle'])
    right_knee = _valid_pose_point(pose_xy, right['knee'])
    right_ankle = _valid_pose_point(pose_xy, right['ankle'])

    if (
        left_knee is None
        or left_ankle is None
        or right_knee is None
        or right_ankle is None
    ):
        return False

    if _segments_intersect_2d(left_knee, left_ankle, right_knee, right_ankle):
        return True

    # Perspective and landmark noise can make the segments miss by a few pixels.
    # A horizontal order flip between knees and ankles is still a useful signal
    # that the lower legs are crossed in the image projection.
    knee_dx = float(left_knee[0] - right_knee[0])
    ankle_dx = float(left_ankle[0] - right_ankle[0])
    return knee_dx * ankle_dx < 0.0


def _foot_ankle_contaminated_by_crossed_leg(
    pose_xy: np.ndarray,
    *,
    side: str,
) -> bool:
    """
    Return whether a foot ankle is likely contaminated by the other crossed leg.

    This is a conservative prompt-safety check. When lower legs are crossed and
    the target ankle lies close to the other leg's knee -> ankle axis, positive
    points around the ankle or color samples along ankle -> foot_index may
    describe the other leg rather than the target foot. In that case the foot
    prompt should rely on foot_index and stronger opposite-foot negatives.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    side : {'anatomical-left', 'anatomical-right'}
        Target foot side whose ankle should be checked.

    Returns
    -------
    bool
        ``True`` when lower legs look crossed and the target ankle is close
        enough to the other leg's knee->ankle axis to make ankle-based foot
        prompts risky. ``False`` means the normal ankle + foot_index prompt can
        be used.
    """
    if not _lower_legs_are_crossed(pose_xy):
        return False

    if side == 'anatomical-left':
        target_ids = _FOOT_LANDMARKS['anatomical-left']
        other_ids = _FOOT_LANDMARKS['anatomical-right']
    elif side == 'anatomical-right':
        target_ids = _FOOT_LANDMARKS['anatomical-right']
        other_ids = _FOOT_LANDMARKS['anatomical-left']
    else:
        raise ValueError(f'Unknown foot side {side!r}.')

    ankle = _valid_pose_point(pose_xy, target_ids['ankle'])
    other_knee = _valid_pose_point(pose_xy, other_ids['knee'])
    other_ankle = _valid_pose_point(pose_xy, other_ids['ankle'])

    if ankle is None or other_knee is None or other_ankle is None:
        return False

    other_leg_len = float(np.linalg.norm(other_ankle - other_knee))
    if other_leg_len < 4.0:
        return False

    threshold = min(
        _FOOT_CROSSED_LEG_ANKLE_NEAR_AXIS_MAX_PX,
        max(
            _FOOT_CROSSED_LEG_ANKLE_NEAR_AXIS_MIN_PX,
            _FOOT_CROSSED_LEG_ANKLE_NEAR_AXIS_RATIO * other_leg_len,
        ),
    )
    distance = _point_segment_distance(ankle, other_knee, other_ankle)
    return distance <= threshold


def _foot_points_for_side(
    pose_xy: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    """
    Resolve the available landmark points for one anatomical foot side.

    The preferred geometry uses MediaPipe ``ankle``, ``heel`` and
    ``foot_index`` landmarks. If fewer than two of those points are available,
    the function falls back to a coarse synthetic two-point foot estimate from
    ``knee -> ankle``.

    Heel may contribute to bbox geometry when valid, but it is not necessarily
    used as a SAM positive prompt because it is less stable in practice.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
        Missing landmarks are encoded as ``(-1, -1)``.
    side : {'anatomical-left', 'anatomical-right'}
        Anatomical foot side to resolve using MediaPipe landmark indices.

    Returns
    -------
    np.ndarray
        Array of shape ``(P, 2)`` with ``P >= 2`` containing image-space foot
        points suitable for bbox/mask construction.

    Raises
    ------
    ValueError
        If ``side`` is not a known anatomical foot side.
    RuntimeError
        If insufficient landmarks are available to infer the foot.
    """
    if side not in _FOOT_LANDMARKS:
        raise ValueError(f'Unknown foot side {side!r}.')

    ids = _FOOT_LANDMARKS[side]
    pts: list[np.ndarray] = []

    for name in ('ankle', 'heel', 'foot_index'):
        p = _valid_pose_point(pose_xy, ids[name])
        if p is not None:
            pts.append(p)

    if len(pts) >= 2:
        return np.stack(pts, axis=0)

    # Fallback for weak foot landmarks: if MediaPipe kept the ankle and knee,
    # extrapolate a rough foot direction from the lower leg. This is intentionally
    # coarse; downstream crop expansion should make the result forgiving.
    ankle = _valid_pose_point(pose_xy, ids['ankle'])
    knee = _valid_pose_point(pose_xy, ids['knee'])

    if ankle is not None and knee is not None:
        leg_vec = ankle - knee
        if float(np.linalg.norm(leg_vec)) >= 4.0:
            synthetic_toe = ankle + 0.45 * leg_vec
            return np.stack([ankle, synthetic_toe], axis=0)

    raise RuntimeError(f'Could not derive {side} foot points from pose.')


def _segment_len(
    pose_xy: np.ndarray,
    a: int,
    b: int,
) -> float:
    """
    Return a valid 2D segment length or ``0.0``.
    """
    pa = _valid_pose_point(pose_xy, a)
    pb = _valid_pose_point(pose_xy, b)
    if pa is None or pb is None:
        return 0.0

    length = float(np.linalg.norm(pb - pa))
    if length < 4.0:
        return 0.0

    return length


def _pose_head_height(pose_xy: np.ndarray) -> float:
    """
    Estimate projected head height from pose face anchors.

    MediaPipe pose face anchors cover only the central face/ears and usually
    underestimate the full head, so the observed anchor height is inflated.
    """
    face_pts = [_valid_pose_point(pose_xy, idx) for idx in (0, 2, 5, 7, 8)]
    face_pts = [p for p in face_pts if p is not None]

    if len(face_pts) < 2:
        return 0.0

    arr = np.stack(face_pts, axis=0)
    face_h = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))

    if face_h < 4.0:
        return 0.0

    return 1.8 * face_h


def _estimate_projected_foot_length(
    pose_xy: np.ndarray,
) -> float:
    """
    Estimate projected foot length from stable 2D body proportions.

    The estimate is a weighted average of:

    - max left/right forearm length, treated as 1:1 with foot length;
    - hip width / 2;
    - shoulder width / 3;
    - head height.

    Missing measurements are ignored and the remaining weights are normalized.
    """
    left_forearm = _segment_len(pose_xy, 13, 15)
    right_forearm = _segment_len(pose_xy, 14, 16)
    forearm = max(left_forearm, right_forearm)

    hip_width = _segment_len(pose_xy, 23, 24)
    shoulder_width = _segment_len(pose_xy, 11, 12)
    head_height = _pose_head_height(pose_xy)

    components = []

    if forearm > 0.0:
        components.append((_FOOT_LENGTH_FROM_FOREARM_WEIGHT, forearm))
    if hip_width > 0.0:
        components.append((_FOOT_LENGTH_FROM_HIP_WEIGHT, hip_width / 2.0))
    if shoulder_width > 0.0:
        components.append((
            _FOOT_LENGTH_FROM_SHOULDER_WEIGHT,
            shoulder_width / 3.0,
        ))
    if head_height > 0.0:
        components.append((_FOOT_LENGTH_FROM_HEAD_WEIGHT, head_height))

    if not components:
        length = _FOOT_LENGTH_MIN_PX
    else:
        total_weight = sum(w for w, _ in components)
        length = sum(w * v for w, v in components) / total_weight
        length = max(length, _FOOT_LENGTH_MIN_PX)

    return length


def _expand_foot_box_toward_person_edge(
    box: tuple[int, int, int, int],
    pose_xy: np.ndarray,
    *,
    side: str,
    person_bbox: Optional[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """
    Expand a foot bbox from ankle toward the coarse foot-index direction.

    When the coarse foot direction points toward the nearest compatible edge of
    the person bbox, the prompt box may be extended up to that edge. This favors
    foot completeness over strict locality and may intentionally include extra
    context in difficult perspective views.

    MediaPipe ``heel`` and ``foot_index`` can be noisy, but ``ankle`` is usually
    more stable because it is anchored to the leg. This helper therefore treats
    the ankle as the origin, uses ``ankle -> foot_index`` only as a coarse 2D
    direction, and expands the box so that it contains:

    ``ankle + normalize(ankle -> foot_index) * estimated_foot_length * safety``.

    The estimated foot length is derived from more stable body-scale cues such
    as forearm length, hip width, shoulder width, and head height. The final box
    is clipped to the person bbox.

    Parameters
    ----------
    box : tuple[int, int, int, int]
        Landmark-derived foot bbox.
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    side : {'anatomical-left', 'anatomical-right'}
        Anatomical foot side.
    person_bbox : tuple[int, int, int, int] or None
        YOLO/person bbox used as an upper spatial guard. If omitted, ``box`` is
        returned unchanged.

    Returns
    -------
    tuple[int, int, int, int]
        Possibly expanded end-exclusive foot bbox, clipped to ``person_bbox``.
    """
    if person_bbox is None:
        return box

    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = person_bbox

    if px2 <= px1 or py2 <= py1:
        return box

    clipped_x1 = max(px1, x1)
    clipped_y1 = max(py1, y1)
    clipped_x2 = min(px2, x2)
    clipped_y2 = min(py2, y2)

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return box

    x1, y1, x2, y2 = clipped_x1, clipped_y1, clipped_x2, clipped_y2

    ids = _FOOT_LANDMARKS[side]
    ankle = _valid_pose_point(pose_xy, ids['ankle'])
    tip = _valid_pose_point(pose_xy, ids['foot_index'])

    if ankle is None or tip is None:
        return int(x1), int(y1), int(x2), int(y2)

    foot_vec = tip - ankle
    foot_vec_len = float(np.linalg.norm(foot_vec))
    if foot_vec_len < 4.0:
        return int(x1), int(y1), int(x2), int(y2)

    direction = foot_vec / foot_vec_len
    raw_length = _estimate_projected_foot_length(pose_xy)
    safe_length = raw_length * _FOOT_LENGTH_SAFETY
    estimated_target = ankle + direction * safe_length
    inward_components = {
        'x': bool(
            (direction[0] > 0.0 and ankle[0] < px2) or
            (direction[0] < 0.0 and ankle[0] > px1)
        ),
        'y': bool(
            (direction[1] > 0.0 and ankle[1] < py2) or
            (direction[1] < 0.0 and ankle[1] > py1)
        ),
    }

    left_gap = float(x1 - px1)
    right_gap = float(px2 - x2)
    top_gap = float(y1 - py1)
    bottom_gap = float(py2 - y2)

    estimated_target = np.asarray([
        min(max(float(estimated_target[0]), float(px1)), float(px2)),
        min(max(float(estimated_target[1]), float(py1)), float(py2)),
    ], dtype=np.float32)
    target = estimated_target.copy()

    if direction[0] < 0.0 and left_gap <= right_gap:
        target[0] = float(px1)
    elif direction[0] > 0.0 and right_gap <= left_gap:
        target[0] = float(px2)

    if direction[1] < 0.0 and top_gap <= bottom_gap:
        target[1] = float(py1)
    elif direction[1] > 0.0 and bottom_gap <= top_gap:
        target[1] = float(py2)

    moved = target - ankle
    can_expand = (
        (inward_components['x'] and abs(float(moved[0])) >= 1.0) or
        (inward_components['y'] and abs(float(moved[1])) >= 1.0)
    )

    if can_expand:
        nx1 = max(px1, min(float(x1), float(target[0])))
        ny1 = max(py1, min(float(y1), float(target[1])))
        nx2 = min(px2, max(float(x2), float(target[0])))
        ny2 = min(py2, max(float(y2), float(target[1])))
    else:
        nx1, ny1, nx2, ny2 = float(x1), float(y1), float(x2), float(y2)

    if nx2 <= nx1 or ny2 <= ny1:
        return box

    return (
        int(math.floor(nx1)),
        int(math.floor(ny1)),
        int(math.ceil(nx2)),
        int(math.ceil(ny2)),
    )


def _foot_candidates(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    expansion: float,
    person_bbox: Optional[tuple[int, int, int, int]] = None,
) -> list[tuple[str, tuple[int, int, int, int], np.ndarray]]:
    """
    Build foot candidates for every anatomical side MediaPipe can support.

    Each candidate contains the anatomical side label, an expanded full-image
    bbox, and the raw landmark points used to derive that bbox.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    expansion : float
        Multiplicative expansion applied to each per-foot bbox.
    person_bbox : tuple[int, int, int, int] or None, optional
        Optional person bbox used to grow foot boxes toward the likely body
        boundary when MediaPipe foot landmarks under-estimate toes/heel.

    Returns
    -------
    list[tuple[str, tuple[int, int, int, int], np.ndarray]]
        Candidate list in ``(side, bbox_xyxy, points)`` format. ``side`` is an
        anatomical label, ``bbox_xyxy`` is end-exclusive full-image geometry,
        and ``points`` contains the resolved foot points.

    Raises
    ------
    RuntimeError
        If neither anatomical side yields usable foot geometry.
    """
    candidates: list[tuple[str, tuple[int, int, int, int], np.ndarray]] = []

    for side in ('anatomical-left', 'anatomical-right'):
        try:
            pts = _foot_points_for_side(pose_xy, side=side)
            box = foot_bbox_xyxy(
                pts,
                pose_xy,
                image_shape,
                side=side,
                expansion=expansion,
            )
            box = _expand_foot_box_toward_person_edge(
                box,
                pose_xy,
                side=side,
                person_bbox=person_bbox,
            )
        except RuntimeError:
            continue

        candidates.append((side, box, pts))

    if not candidates:
        raise RuntimeError('Could not derive any foot geometry from pose.')

    return candidates


def _select_foot_candidates(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float,
    person_bbox: Optional[tuple[int, int, int, int]] = None,
) -> list[tuple[str, tuple[int, int, int, int], np.ndarray]]:
    """
    Select foot candidates using observer/image-side semantics.

    MediaPipe foot landmarks are anatomical. Public crop targets use the same
    convention as ``SubjectCrop`` hand targets: ``left`` means left side of the
    image, and ``right`` means right side of the image. Selection is therefore
    based on the horizontal center of each candidate bbox.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    which : {'left', 'right', 'both'}
        Requested foot selection in image/viewer perspective.
    expansion : float
        Multiplicative expansion applied before candidate selection.
    person_bbox : tuple[int, int, int, int] or None, optional
        Optional person bbox used to grow foot boxes toward the likely body
        boundary.

    Returns
    -------
    list[tuple[str, tuple[int, int, int, int], np.ndarray]]
        Selected candidates in ``(side, bbox_xyxy, points)`` format.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If no candidate can be built, or if side-specific selection would be
        ambiguous.
    """
    if which not in ('left', 'right', 'both'):
        raise ValueError(
            f"Invalid which={which!r}; expected 'left', 'right', or 'both'."
        )

    candidates = _foot_candidates(
        pose_xy,
        image_shape,
        expansion=expansion,
        person_bbox=person_bbox,
    )

    if which == 'both':
        return candidates

    # Public left/right semantics are observer/image based, matching
    # SubjectCrop hand targets. MediaPipe landmark names are anatomical, so
    # choose the leftmost/rightmost candidate by image-space center.
    ordered = sorted(
        candidates,
        key=lambda item: 0.5 * (item[1][0] + item[1][2]),
    )

    if len(ordered) >= 2:
        return [ordered[0] if which == 'left' else ordered[-1]]

    # With only one detected foot, accept side-specific selection only when the
    # candidate is on the requested half of the image; otherwise fail loudly
    # instead of silently returning the wrong foot.
    _, box, _ = ordered[0]
    _, w = image_shape[:2]
    cx = 0.5 * (box[0] + box[2])

    if which == 'left' and cx <= 0.5 * w:
        return ordered
    if which == 'right' and cx >= 0.5 * w:
        return ordered

    raise RuntimeError(
        f'Could not reliably select {which} foot from a single detected foot.'
    )


def foot_bbox_xyxy(
    foot_pts: np.ndarray,
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    side: str,
    expansion: float = 1.8,
) -> tuple[int, int, int, int]:
    """
    Compute a generous bbox from MediaPipe foot landmarks.

    ``foot_pts`` is expected to contain at least two image-space points derived
    from ankle, heel, and foot_index. The bbox is computed as a square enclosing
    those raw landmark points, then expanded.

    Parameters
    ----------
    foot_pts : np.ndarray
        Image-space points for one foot, with shape ``(P, 2)`` and ``P >= 2``.
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
        The same-side knee/ankle may be used to estimate a minimum crop size.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used for clipping.
    side : {'anatomical-left', 'anatomical-right'}
        Anatomical side corresponding to ``foot_pts``.
    expansion : float, default=1.8
        Multiplicative expansion factor for the generated bbox.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive bbox ``(x1, y1, x2, y2)`` in full-image coordinates.

    Raises
    ------
    RuntimeError
        If ``foot_pts`` has an invalid shape or cannot define a non-empty bbox.
    """
    if foot_pts.ndim != 2 or foot_pts.shape[1] != 2:
        raise RuntimeError(
            f'Invalid foot point shape {foot_pts.shape!r}; expected (N, 2).'
        )

    if foot_pts.shape[0] < 2:
        raise RuntimeError('At least two foot points are required.')

    h, w = image_shape[:2]

    x1 = float(np.min(foot_pts[:, 0]))
    y1 = float(np.min(foot_pts[:, 1]))
    x2 = float(np.max(foot_pts[:, 0]))
    y2 = float(np.max(foot_pts[:, 1]))

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    raw_w = max(1.0, x2 - x1)
    raw_h = max(1.0, y2 - y1)

    ids = _FOOT_LANDMARKS[side]
    knee = _valid_pose_point(pose_xy, ids['knee'])
    ankle = _valid_pose_point(pose_xy, ids['ankle'])
    shin_len = 0.0
    if knee is not None and ankle is not None:
        shin_len = float(np.linalg.norm(ankle - knee))

    min_size = max(24.0, 0.04 * max(w, h), 0.35 * shin_len)
    base = max(raw_w, raw_h, min_size)

    ex = max(1.0, float(expansion))
    out_w = max(raw_w * ex, base * ex)
    out_h = max(raw_h * ex, base * ex)

    ox1 = int(math.floor(cx - out_w / 2.0))
    ox2 = int(math.ceil(cx + out_w / 2.0))
    oy1 = int(math.floor(cy - out_h / 2.0))
    oy2 = int(math.ceil(cy + out_h / 2.0))

    ox1 = max(0, min(w - 1, ox1))
    oy1 = max(0, min(h - 1, oy1))
    ox2 = max(ox1 + 1, min(w, ox2))
    oy2 = max(oy1 + 1, min(h, oy2))

    if ox2 <= ox1 or oy2 <= oy1:
        raise RuntimeError('Invalid expanded foot bbox.')

    return ox1, oy1, ox2, oy2


def feet_bbox_xyxy_from_landmarks(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.8,
    person_bbox: Optional[tuple[int, int, int, int]] = None,
) -> tuple[int, int, int, int]:
    """
    Compute a bbox covering one or both feet from MediaPipe Pose landmarks.

    ``which`` uses observer/image perspective:

    - ``'left'``: foot appearing on the left side of the image
    - ``'right'``: foot appearing on the right side of the image
    - ``'both'``: union of all detected foot candidates

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
        Missing landmarks are encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    which : {'left', 'right', 'both'}
        Foot selection in image/viewer perspective.
    expansion : float, default=1.8
        Multiplicative expansion applied to each per-foot bbox before union.
    person_bbox : tuple[int, int, int, int] or None, optional
        Optional person bbox used as a spatial guard. When provided, each
        landmark-derived foot box may be expanded toward the likely adjacent
        person-bbox edge, which helps recover toes missed by MediaPipe
        ``foot_index`` landmarks.

    Returns
    -------
    tuple[int, int, int, int]
        End-exclusive bbox ``(x1, y1, x2, y2)`` in full-image coordinates.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If no suitable foot candidate can be derived.
    """
    selected = _select_foot_candidates(
        pose_xy,
        image_shape,
        which=which,
        expansion=expansion,
        person_bbox=person_bbox,
    )

    x1 = min(b[0] for _, b, _ in selected)
    y1 = min(b[1] for _, b, _ in selected)
    x2 = max(b[2] for _, b, _ in selected)
    y2 = max(b[3] for _, b, _ in selected)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid combined foot bbox.')

    return x1, y1, x2, y2


def _foot_sam_points_for_side(
    pose_xy: np.ndarray,
    *,
    side: str,
    base_bbox: Optional[tuple[int, int, int, int]] = None,
    prompt_bbox: Optional[tuple[int, int, int, int]] = None,
    additional_positive_points: Optional[list[list[float]]] = None,
) -> tuple[Optional[list[list[float]]], Optional[list[int]]]:
    """
    Build the normal SAM foot prompt for one anatomical side.

    The landmark positives are ankle and foot_index. The MediaPipe heel landmark
    is not used because in practice it is less stable for prompting SAM.
    If crossed-leg geometry suggests that the ankle is projected onto the other
    leg, the prompt becomes more conservative and uses foot_index as the only
    landmark positive.

    ``additional_positive_points`` lets callers add extra positive prompt hints
    computed by higher-level logic. This helper does not care how those points
    were chosen; it only appends already validated coordinates to the SAM
    prompt.

    When the opposite foot is visible, the first valid point among its ankle,
    heel and foot_index is added as a negative point, but only if that point is
    inside this foot's local ``prompt_bbox`` and outside this foot's
    ``base_bbox``. If it falls inside ``base_bbox``, the two feet may be
    overlapping and the point would be an ambiguous negative; if it falls
    outside ``prompt_bbox``, it is outside SAM's local prompt domain.

    When ``prompt_bbox`` is available, one additional negative point is placed
    in front of the foot, on the ankle -> foot_index ray, one pixel inside the
    prompt bbox. This tells SAM that background beyond the toe/sandal direction
    is not part of the target without putting a negative point near the lower
    leg, where skin continuity can confuse the segmentation.

    The lower-leg point is *not* added here. It is exposed separately as a probe
    by ``_foot_probe_point_for_side`` so callers can decide dynamically
    whether it should become a negative prompt.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    side : {'anatomical-left', 'anatomical-right'}
        Anatomical foot side.
    base_bbox : tuple[int, int, int, int] or None, optional
        Base foot bbox used to decide whether an opposite-foot point is safely
        outside the current foot region.
    prompt_bbox : tuple[int, int, int, int] or None, optional
        Foot-local SAM prompt bbox used to place line-extension negatives one
        pixel inside the box.
    additional_positive_points : list[list[float]] or None, optional
        Extra positive SAM points already validated by the caller.

    Returns
    -------
    tuple[list[list[float]] | None, list[int] | None]
        SAM point coordinates and labels. Returns ``(None, None)`` if no
        positive foot points are available.
    """
    ids = _FOOT_LANDMARKS[side]
    min_axis_len_px = 4.0
    points: list[list[float]] = []
    labels: list[int] = []
    ankle = _valid_pose_point(pose_xy, ids['ankle'])
    foot_index = _valid_pose_point(pose_xy, ids['foot_index'])

    # Crossed legs can project the target ankle onto the other lower leg. In
    # that case ankle and ankle-derived chromatic points may describe the wrong
    # texture, so we intentionally shrink the positive prompt to foot_index.
    # The foot_index landmark is still the best available anchor for the target
    # foot, while opposite-foot negatives below help SAM separate nearby feet.
    ankle_contaminated = _foot_ankle_contaminated_by_crossed_leg(
        pose_xy,
        side=side,
    )

    landmark_positives = (
        (foot_index,)
        if ankle_contaminated
        else (ankle, foot_index)
    )

    for p in landmark_positives:
        if p is not None:
            points.append([float(p[0]), float(p[1])])
            labels.append(1)

    if additional_positive_points and not ankle_contaminated:
        for point in additional_positive_points:
            if len(point) < 2:
                continue

            px, py = point[:2]
            if px < 0 or py < 0:
                continue

            points.append([float(px), float(py)])
            labels.append(1)

    if not points:
        return None, None

    # Forward-axis negative:
    # This point is useful only when we have the complete local geometry:
    #
    # - ankle and foot_index define a reliable foot direction;
    # - prompt_bbox is the exact local SAM domain, so the negative can be placed
    #   one pixel inside the cyan/debug box instead of outside SAM's prompt;
    # - base_bbox is the conservative red/debug foot box, used as a safety guard
    #   so the generated negative is never accepted if it lands on the likely
    #   foot/sandal region.
    #
    # If any of these inputs is missing, skip the synthetic axis negative rather
    # than guessing. The positive prompt points remain valid on their own.
    if (
        ankle is not None
        and foot_index is not None
        and prompt_bbox is not None
        and base_bbox is not None
    ):
        foot_vec = foot_index - ankle
        foot_len = float(np.linalg.norm(foot_vec))

        if foot_len >= min_axis_len_px:
            x1, y1, x2, y2 = prompt_bbox
            min_x = float(x1)
            min_y = float(y1)
            max_x = float(x2 - 1)
            max_y = float(y2 - 1)
            direction = foot_vec / foot_len
            ts: list[float] = []

            # Intersect the forward ankle -> foot_index ray with the prompt
            # bbox, then step one pixel back inside the box. We deliberately do
            # not place the opposite/backward negative near the ankle: on bare
            # legs or sandals it can land on leg/skin and tell SAM to exclude a
            # texture that is continuous with the foot.
            if abs(float(direction[0])) > 1e-6:
                for bx in (min_x, max_x):
                    t = (bx - float(ankle[0])) / float(direction[0])
                    if t <= foot_len:
                        continue
                    y = float(ankle[1]) + float(direction[1]) * t
                    if min_y <= y <= max_y:
                        ts.append(t)

            if abs(float(direction[1])) > 1e-6:
                for by in (min_y, max_y):
                    t = (by - float(ankle[1])) / float(direction[1])
                    if t <= foot_len:
                        continue
                    x = float(ankle[0]) + float(direction[0]) * t
                    if min_x <= x <= max_x:
                        ts.append(t)

            if ts:
                bx1, by1, bx2, by2 = base_bbox
                t = max(0.0, min(ts) - 1.0)

                # If the prompt boundary is too close to foot_index, a forward
                # negative would sit on/near the toes or sandal and become an
                # ambiguous instruction. Require at least 20% of the observed
                # ankle->foot_index distance beyond foot_index.
                if t - foot_len >= 0.20 * foot_len:
                    neg = ankle + direction * float(t)
                    neg = np.asarray([
                        min(max(float(neg[0]), min_x), max_x),
                        min(max(float(neg[1]), min_y), max_y),
                    ], dtype=np.float32)

                    # Do not add the negative if it lands in the red/base foot
                    # box: in that case the prompt bbox is too tight, or the
                    # foot/sandal legitimately reaches that point.
                    if not (bx1 <= neg[0] < bx2 and by1 <= neg[1] < by2):
                        points.append([float(neg[0]), float(neg[1])])
                        labels.append(0)

    # Opposite-foot negative:
    # Add landmarks from the other foot as negative prompts, but only when each
    # point is both:
    #
    # - inside this foot's prompt_bbox, so SAM can actually use it in the local
    #   prompt domain;
    # - outside this foot's base_bbox, so the two feet are not overlapping in a
    #   way that would make the negative ambiguous.
    #
    # In normal geometry, one safe opposite-foot point is enough. In crossed-leg
    # contamination mode, the target ankle is unsafe as a positive, so we use
    # both the other foot's ankle and foot_index as stronger negatives when
    # available.
    #
    # This is intentionally separate from the forward-axis negative above: that
    # one suppresses background beyond the current foot direction, while this
    # one disambiguates nearby or overlapping feet.
    if base_bbox is not None:
        other_side = (
            'anatomical-right'
            if side == 'anatomical-left'
            else 'anatomical-left'
        )
        other_ids = _FOOT_LANDMARKS[other_side]
        bx1, by1, bx2, by2 = base_bbox
        px1, py1, px2, py2 = prompt_bbox if prompt_bbox is not None else (
            0,
            0,
            math.inf,
            math.inf,
        )

        # Normal case: try ankle, heel, then foot_index and stop after the first
        # safe opposite-foot negative. Contaminated-ankle case: use stronger,
        # explicit opposite-foot anchors, but still skip heel because it is the
        # least stable foot landmark in the images that motivated this prompt
        # logic.
        other_negative_names = (
            ('ankle', 'foot_index')
            if ankle_contaminated
            else ('ankle', 'heel', 'foot_index')
        )

        for name in other_negative_names:
            p = _valid_pose_point(pose_xy, other_ids[name])
            if p is None:
                continue

            if not (px1 <= p[0] < px2 and py1 <= p[1] < py2):
                continue

            if bx1 <= p[0] < bx2 and by1 <= p[1] < by2:
                continue

            points.append([float(p[0]), float(p[1])])
            labels.append(0)

            if not ankle_contaminated:
                break

    return points, labels


def _foot_probe_point_for_side(
    pose_xy: np.ndarray,
    *,
    side: str,
    prompt_bbox: tuple[int, int, int, int],
) -> Optional[list[float]]:
    """
    Return a lower-leg probe point inside the foot prompt bbox.

    The probe is not a normal foot prompt. It is a diagnostic point used by
    ``SubjectCrop``:

    1. run SAM with ankle + foot_index to get a foot mask;
    2. run SAM with only this probe point as a positive point;
    3. compare the two masks inside ``prompt_bbox``.

    If the probe mask overlaps the foot mask heavily, SAM sees visual continuity
    between leg and foot (common with bare skin, sandals, flip-flops), so the
    probe should *not* be used as a negative. If overlap is low, there is likely
    a material/edge discontinuity (pants, socks, shoes), and the same point can
    be useful as a negative prompt.

    Geometrically, the point lies on the ray ``ankle -> knee``. We intersect
    that ray with ``prompt_bbox`` and step back by one pixel, guaranteeing the
    point is inside the exact bbox that SAM receives.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    side : {'anatomical-left', 'anatomical-right'}
        Anatomical foot side.
    prompt_bbox : tuple[int, int, int, int]
        End-exclusive foot prompt bbox passed to SAM.

    Returns
    -------
    list[float] or None
        Probe point ``[x, y]`` inside ``prompt_bbox`` when knee and ankle are
        available and define a usable direction; otherwise ``None``.
    """
    ids = _FOOT_LANDMARKS[side]
    ankle = _valid_pose_point(pose_xy, ids['ankle'])
    knee = _valid_pose_point(pose_xy, ids['knee'])

    if ankle is None or knee is None:
        return None

    leg_vec = knee - ankle
    leg_len = float(np.linalg.norm(leg_vec))

    if leg_len < 4.0:
        return None

    x1, y1, x2, y2 = prompt_bbox
    min_x = float(x1)
    min_y = float(y1)
    max_x = float(x2 - 1)
    max_y = float(y2 - 1)

    direction = leg_vec / leg_len
    ts: list[float] = []

    # Parametric ray: p(t) = ankle + direction * t, t >= 0.
    # For each axis, compute the t at which the ray reaches the relevant bbox
    # boundary. The first positive intersection is where the ray exits the box.
    if direction[0] > 0.0:
        ts.append((max_x - float(ankle[0])) / float(direction[0]))
    elif direction[0] < 0.0:
        ts.append((min_x - float(ankle[0])) / float(direction[0]))

    if direction[1] > 0.0:
        ts.append((max_y - float(ankle[1])) / float(direction[1]))
    elif direction[1] < 0.0:
        ts.append((min_y - float(ankle[1])) / float(direction[1]))

    positive_ts = [t for t in ts if t > 0.0]

    if not positive_ts:
        return None

    dist = max(0.0, min(positive_ts) - 1.0)
    probe = ankle + direction * dist

    # Floating point math near diagonal boundaries can land microscopically
    # outside the box, so clamp after stepping back from the edge.
    probe = np.asarray([
        min(max(float(probe[0]), min_x), max_x),
        min(max(float(probe[1]), min_y), max_y),
    ], dtype=np.float32)

    return [float(probe[0]), float(probe[1])]


def foot_sam_regions_from_landmarks(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.8,
    person_bbox: Optional[tuple[int, int, int, int]] = None,
    additional_positive_points_by_side: Optional[dict[str,
                                                      list[list[float]]]] = None,
) -> list[SamRegion]:
    """
    Build foot-local SAM regions.

    The ``which='both'`` path returns one region per selected foot. Callers can
    run SAM independently for each region and union the resulting masks, while
    still using the union of all returned bboxes as the final crop envelope.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    which : {'left', 'right', 'both'}
        Foot selection in image/viewer perspective.
    expansion : float, default=1.8
        Multiplicative expansion used to build each base foot bbox.
    person_bbox : tuple[int, int, int, int] or None, optional
        Optional person bbox used only to expand each local prompt bbox toward
        the likely body/silhouette edge.
    additional_positive_points_by_side : dict[str, list[list[float]]] or None, optional
        Extra positive prompt points keyed by anatomical side. This is intended
        for image-aware prompt enrichment computed outside this geometry helper.

    Returns
    -------
    list[SamRegion]
        One region per selected foot, each containing base bbox, prompt bbox,
        normal SAM points and optional lower-leg probe point.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If no suitable foot region can be derived.
    """
    selected = _select_foot_candidates(
        pose_xy,
        image_shape,
        which=which,
        expansion=expansion,
        person_bbox=None,
    )

    regions: list[SamRegion] = []

    for side, base_bbox, _ in selected:
        prompt_bbox = _expand_foot_box_toward_person_edge(
            base_bbox,
            pose_xy,
            side=side,
            person_bbox=person_bbox,
        )
        point_coords, point_labels = _foot_sam_points_for_side(
            pose_xy,
            side=side,
            base_bbox=base_bbox,
            prompt_bbox=prompt_bbox,
            additional_positive_points=(
                additional_positive_points_by_side or {}
            ).get(side),
        )
        probe_point = _foot_probe_point_for_side(
            pose_xy,
            side=side,
            prompt_bbox=prompt_bbox,
        )
        regions.append(SamRegion(
            side=side,
            base_bbox=base_bbox,
            prompt_bbox=prompt_bbox,
            point_coords=point_coords,
            point_labels=point_labels,
            probe_point=probe_point,
        ))

    return regions


def foot_sam_region_with_prompt_bbox(
    region: SamRegion,
    pose_xy: np.ndarray,
    prompt_bbox: tuple[int, int, int, int],
    additional_positive_points_by_side: Optional[dict[str,
                                                      list[list[float]]]] = None,
) -> SamRegion:
    """
    Return a foot ``SamRegion`` with an updated prompt bbox.

    This helper returns the generic ``SamRegion`` container, but its semantics
    are foot-specific: it recomputes prompt points with
    ``_foot_sam_points_for_side`` and the adaptive probe with
    ``_foot_probe_point_for_side``. It should therefore not be reused for other
    target families such as arms or legs.

    ``SubjectCrop`` may expand foot prompt bboxes after the initial region is
    built. Foot prompt points can depend on the exact prompt box, so they must
    be recomputed instead of copied.

    Parameters
    ----------
    region : SamRegion
        Existing foot SAM region.
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates.
    prompt_bbox : tuple[int, int, int, int]
        New end-exclusive prompt bbox.
    additional_positive_points_by_side : dict[str, list[list[float]]] or None, optional
        Extra positive prompt points keyed by anatomical side.

    Returns
    -------
    SamRegion
        Updated foot region preserving ``side`` and ``base_bbox`` while
        recomputing normal foot prompt points and ``probe_point`` for
        ``prompt_bbox``.
    """
    point_coords, point_labels = _foot_sam_points_for_side(
        pose_xy,
        side=region.side,
        base_bbox=region.base_bbox,
        prompt_bbox=prompt_bbox,
        additional_positive_points=(
            additional_positive_points_by_side or {}
        ).get(region.side),
    )
    probe_point = _foot_probe_point_for_side(
        pose_xy,
        side=region.side,
        prompt_bbox=prompt_bbox,
    )

    return SamRegion(
        side=region.side,
        base_bbox=region.base_bbox,
        prompt_bbox=prompt_bbox,
        point_coords=point_coords,
        point_labels=point_labels,
        probe_point=probe_point,
    )


def feet_sam_points_from_landmarks(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.8,
    person_bbox: Optional[tuple[int, int, int, int]] = None,
) -> tuple[Optional[list[list[float]]], Optional[list[int]]]:
    """
    Build aggregated SAM point prompts for one or both feet.

    This is a compatibility wrapper around ``foot_sam_regions_from_landmarks``
    for callers that only need point prompts and do not need per-foot prompt
    bboxes. The returned points are the normal foot prompts from each selected
    region:

    - ankle and foot_index are positive points;
    - heel is intentionally excluded because it is often less stable in
      MediaPipe Pose foot geometry;
    - foot-axis edge points and an optional opposite-foot point may be added as
      negatives when they are clearly outside the current foot's base bbox.

    The lower-leg probe point is deliberately *not* returned here. It is stored
    on each ``SamRegion`` because callers must first compare the foot mask
    with a separate leg-probe mask before deciding whether that point is safe to
    reuse as a negative prompt.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
        Missing landmarks are encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    which : {'left', 'right', 'both'}
        Foot selection in image/viewer perspective.
    expansion : float, default=1.8
        Multiplicative expansion used to build each base foot bbox.
    person_bbox : tuple[int, int, int, int] or None, optional
        Optional person bbox used to expand local foot prompt bboxes toward the
        likely body/silhouette edge before deriving prompt regions.

    Returns
    -------
    tuple[list[list[float]] | None, list[int] | None]
        Aggregated SAM point coordinates and labels for the selected feet.
        Labels use SAM convention: ``1`` for positive, ``0`` for negative.
        Returns ``(None, None)`` if no prompt point can be built.

    Raises
    ------
    ValueError
        If ``which`` is invalid.
    RuntimeError
        If no suitable foot region can be derived.
    """
    regions = foot_sam_regions_from_landmarks(
        pose_xy,
        image_shape,
        which=which,
        expansion=expansion,
        person_bbox=person_bbox,
    )

    points: list[list[float]] = []
    labels: list[int] = []

    for region in regions:
        points.extend(region.point_coords or [])
        labels.extend(region.point_labels or [])

    if not points:
        return None, None

    return points, labels


def feet_mask_from_landmarks(
    pose_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.8,
    person_bbox: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Build a generous foot mask from MediaPipe Pose landmarks.

    The mask is intentionally approximate. It fills the triangle/hull defined
    by ankle, heel, and foot_index when available; with only two reliable points
    it draws a thick local line and expands it. The resulting mask is meant to
    constrain SAM/person masks to a foot-local region, not to segment toes.

    Note: This helper produces geometry-only guidance and is not the primary
    foot-segmentation path used by the current SubjectCrop implementation.

    Parameters
    ----------
    pose_xy : np.ndarray
        MediaPipe pose landmarks in image coordinates, with shape ``(33, 2)``.
        Missing landmarks are encoded as ``(-1, -1)``.
    image_shape : tuple[int, ...]
        Source image shape. Only ``(H, W)`` are used.
    which : {'left', 'right', 'both'}
        Foot selection in image/viewer perspective.
    expansion : float, default=1.8
        Expansion factor controlling the selected foot candidates and the
        size-aware dilation applied to their masks.
    person_bbox : tuple[int, int, int, int] or None, optional
        Optional person bbox used to expand foot candidates toward the likely
        body boundary before building the local foot mask.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(H, W)``.

    Raises
    ------
    ValueError
        If ``which`` is invalid or ``expansion`` is negative.
    RuntimeError
        If no usable foot mask can be derived.
    """
    import cv2

    if expansion < 0:
        raise ValueError(
            f'Invalid expansion={expansion!r}; expected >= 0.'
        )

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Build one geometric candidate per requested foot. With ``which='both'``
    # the two feet stay independent until the very end, so a wide/uncertain
    # bbox around one foot does not directly swallow the other foot.
    selected = _select_foot_candidates(
        pose_xy,
        image_shape,
        which=which,
        expansion=expansion,
        person_bbox=person_bbox,
    )

    for _, box, pts in selected:
        pts_i = pts.astype(np.int32, copy=True)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, h - 1)

        local_mask = np.zeros((h, w), dtype=np.uint8)

        if pts_i.shape[0] >= 3:
            # When ankle, heel and foot_index are available, the convex hull is
            # the cleanest geometry-only approximation of the visible foot.
            hull = cv2.convexHull(pts_i)
            if hull is not None and len(hull) >= 3:
                cv2.fillConvexPoly(local_mask, hull, 255)
        else:
            # If only two points survived, keep a thick segment rather than
            # inventing a triangle from unreliable anatomy. The later dilation
            # makes this fallback forgiving enough for crop guidance.
            x1, y1, x2, y2 = box
            thickness = max(3, int(round(0.18 * max(x2 - x1, y2 - y1))))
            cv2.line(
                local_mask,
                tuple(int(v) for v in pts_i[0]),
                tuple(int(v) for v in pts_i[1]),
                255,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)

        # The raw MediaPipe foot landmarks are sparse and often sit inside the
        # actual shoe/skin silhouette. Dilation turns the landmark hull/segment
        # into a generous crop-support mask; it is not intended as final
        # pixel-perfect segmentation.
        base_radius = max(2, int(round(0.16 * size)))
        extra_radius = max(
            0,
            int(round(max(0.0, float(expansion) - 1.0) * 0.08 * size)),
        )
        radius = base_radius + extra_radius

        if radius > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            )
            local_mask = cv2.dilate(local_mask, k, iterations=1)

        mask = np.maximum(mask, local_mask)

    if not np.any(mask):
        raise RuntimeError(
            f'Could not derive a foot mask for which={which!r}.'
        )

    return (mask > 0)
