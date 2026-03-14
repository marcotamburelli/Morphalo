import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
from PIL import Image

from stability.cache.models import (get_mediapipe_face_landmarker,
                                    get_mediapipe_pose_landmarker, get_sam,
                                    get_yolo)
from stability.core.paths import make_node_output_path
from stability.dag import NodeRef
from stability.nodes.common.config_resolve import SpecInput, resolve_spec
from stability.nodes.common.io import write_json_sidecar
from stability.nodes.preprocess.utils import postprocess_mask
from stability.nodes.sdxl_resolve import resolve_single_image_path

CropMode = Literal['bbox', 'trim', 'full_frame']


def _tight_alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return the minimal end-exclusive box containing non-zero alpha pixels.

    Parameters
    ----------
    alpha : np.ndarray
        Alpha channel with shape ``(H, W)``.

    Returns
    -------
    tuple[int, int, int, int]
        Tight bounding box ``(x1, y1, x2, y2)`` in local coordinates.

    Raises
    ------
    RuntimeError
        If the alpha channel contains no non-zero pixels.
    """
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError('Trimmed crop has no non-transparent pixels.')

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


@dataclass
class Config:
    device: str
    yolo_model: Optional[str]
    sam_checkpoint: str
    sam_model_type: Optional[str]
    mode: str
    crop_mode: Optional[CropMode]
    conf: float
    box_margin: float
    multimask: bool
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: Optional[str]
    pose_landmarker_task: str
    smoothing_radius: int


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    sam_checkpoint = model.get('sam_checkpoint')
    device = model.get('device', 'cuda')

    if not sam_checkpoint:
        raise ValueError(
            f"'{node_id}': Missing 'sam_checkpoint' from model configuration."
        )

    sam_model_type = model.get('sam_model_type')

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': Invalid mode={mode!r}")

    # It should apply only when mode='default'
    if mode == 'default':
        crop_mode = str(params.get('crop_mode', 'trim'))
        if crop_mode not in ('bbox', 'trim', 'full_frame'):
            raise ValueError(
                f"'{node_id}': invalid crop_mode={crop_mode!r} "
                "(expected 'bbox', 'trim', or 'full_frame')"
            )
    else:
        crop_mode = None

    conf = float(params.get('conf', 0.35))
    box_margin = float(params.get('box_margin', 0.12))
    multimask = bool(params.get('multimask', True))

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))

    target = str(params.get('target', 'person'))
    if target not in ('person', 'face', 'head', 'eyes', 'left-eye', 'right-eye'):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'person', 'face', 'head', 'eyes', 'left-eye', or 'right-eye')"
        )

    expansion = float(params.get('expansion', 1.0))
    if expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} (expected >= 0)"
        )

    save_debug = bool(debug.get('save_debug', False))

    face_landmarker_task = None
    yolo_model = None

    pose_landmarker_task = model.get('pose_landmarker_task')
    if pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': target={target!r} requires model.pose_landmarker_task (MediaPipe .task path)"
        )
    pose_landmarker_task = str(pose_landmarker_task)

    if target in ('person', 'head'):
        yolo_model = str(model.get('yolo_model', 'yolov8n.pt'))

    if target in ('face', 'head', 'eyes', 'left-eye', 'right-eye'):
        # MediaPipe is used for face/head
        face_landmarker_task = model.get('face_landmarker_task')
        if not face_landmarker_task:
            raise ValueError(
                f"'{node_id}': target={target!r} requires model.face_landmarker_task (MediaPipe .task path)"
            )
        face_landmarker_task = str(face_landmarker_task)

    return Config(
        device=device,
        yolo_model=yolo_model,
        sam_checkpoint=str(sam_checkpoint),
        sam_model_type=None if sam_model_type is None else str(sam_model_type),
        mode=mode,
        crop_mode=crop_mode,
        conf=conf,
        box_margin=box_margin,
        multimask=multimask,
        save_debug=save_debug,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        target=target,
        expansion=expansion,
        face_landmarker_task=face_landmarker_task,
        pose_landmarker_task=pose_landmarker_task,
        smoothing_radius=smoothing_radius,
    )


def _person_bboxes_xyxy(res, node_id: str) -> list[tuple[int, int, int, int]]:
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


def _score_bbox_with_pose(
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


def _select_person_bbox_xyxy(
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
        score = _score_bbox_with_pose(box, pose_xy=pose_xy)
        if best_score is None or score > best_score:
            best_score = score
            best_box = box

    if best_box is None:
        raise RuntimeError('Could not select a pose-aligned person bbox.')

    return best_box


def _head_area_xyxy_from_pose(
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


def _get_head_area(
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

    x1, y1, x2, y2 = _head_area_xyxy_from_pose(
        pose_xy,
        full_w=w,
        full_h=h,
        expansion=expansion,
    )

    head_area_rgb = img_rgb[y1:y2, x1:x2, :]
    if head_area_rgb.size == 0:
        raise RuntimeError('Pose-derived head area is empty.')

    return np.ascontiguousarray(head_area_rgb), x1, y1


def _mp_largest_face_bbox_xyxy(
    img_rgb: np.ndarray,
    face_landmarker_task,
    # face_min_score: Optional[float] = None,
) -> tuple[int, int, int, int]:
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    result = face_landmarker_task.detect(mp_image)
    face_landmarks_list = result.face_landmarks or []
    # face_presence_scores = result.face_presence_scores or []

    if not face_landmarks_list:
        raise RuntimeError('MediaPipe found no face landmarks.')

    candidates = []

    for landmarks in face_landmarks_list:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x1 = max(0, int(round(min(xs) * w)))
        y1 = max(0, int(round(min(ys) * h)))
        x2 = min(w, int(round(max(xs) * w)))
        y2 = min(h, int(round(max(ys) * h)))

        if x2 > x1 and y2 > y1:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, x1, y1, x2, y2))

    if not candidates:
        raise RuntimeError('Could not derive valid face bbox from landmarks.')

    # pick largest face
    _, x1, y1, x2, y2 = max(candidates, key=lambda t: t[0])

    return x1, y1, x2, y2


def _mp_eye_bbox_xyxy(
    img_rgb: np.ndarray,
    face_landmarker_task,
    *,
    which: str,
    expansion: float = 1.2,
) -> tuple[int, int, int, int]:
    """
    Derive an eye bounding box from MediaPipe face landmarks.

    This computes a bounding box for either the left eye, right eye, or both
    eyes combined, using landmark indices from the MediaPipe FaceMesh topology.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input image in RGB format (H, W, 3), dtype uint8.
    face_landmarker_task : Any
        MediaPipe FaceLandmarker instance created by
        ``get_mediapipe_face_landmarker``.
    which : {'left', 'right', 'both'}
        Eye selection:
        - 'left': subject's left eye
        - 'right': subject's right eye
        - 'both': combined region covering both eyes
    expansion : float, optional
        Symmetric expansion factor applied to the derived bounding box.
        Values slightly above 1.0 help include eyelids and avoid hard borders.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box coordinates (x1, y1, x2, y2), end-exclusive.

    Notes
    -----
    - 'left' and 'right' refer to the subject perspective (not the viewer).
    - The landmark ranges used here are a pragmatic choice for stable cropping.
      If you need tighter control (e.g., iris-only), use iris landmarks and/or
      a dedicated segmentation step.
    """
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    result = face_landmarker_task.detect(mp_image)
    face_landmarks_list = result.face_landmarks or []
    if not face_landmarks_list:
        raise RuntimeError('MediaPipe found no face landmarks.')

    landmarks = face_landmarks_list[0]

    # MediaPipe FaceMesh landmark ranges (inclusive start, exclusive end)
    left_eye_ids = list(range(33, 133))
    right_eye_ids = list(range(362, 463))

    if which == 'left':
        ids = left_eye_ids
    elif which == 'right':
        ids = right_eye_ids
    elif which == 'both':
        ids = left_eye_ids + right_eye_ids
    else:
        raise ValueError(
            f'Invalid which={which!r} (expected left, right, or both)'
        )

    xs: list[float] = []
    ys: list[float] = []

    for idx in ids:
        lm = landmarks[idx]
        xs.append(float(lm.x))
        ys.append(float(lm.y))

    x1 = max(0, int(round(min(xs) * w)))
    y1 = max(0, int(round(min(ys) * h)))
    x2 = min(w, int(round(max(xs) * w)))
    y2 = min(h, int(round(max(ys) * h)))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid eye bbox derived from landmarks.')

    # Expand around center for softer inpaint borders
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw *= float(expansion)
    bh *= float(expansion)

    ex1 = int(math.floor(cx - bw / 2.0))
    ex2 = int(math.ceil(cx + bw / 2.0))
    ey1 = int(math.floor(cy - bh / 2.0))
    ey2 = int(math.ceil(cy + bh / 2.0))

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(w, ex2)
    ey2 = min(h, ey2)

    if ex2 <= ex1 or ey2 <= ey1:
        raise RuntimeError('Invalid expanded eye bbox.')

    return ex1, ey1, ex2, ey2


def _mp_eye_mask_from_landmarks(
    img_rgb: np.ndarray,
    face_landmarker_task,
    *,
    which: str,
    expansion: float,
) -> np.ndarray:
    """
    Build a full-frame boolean eye mask from MediaPipe face landmarks.

    This is used as an alternative to SAM for eye targets, because small,
    disjoint regions (two eyes) can be hard for box-guided SAM to segment
    reliably.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image (H, W, 3), dtype uint8.
    face_landmarker_task : Any
        MediaPipe FaceLandmarker instance.
    which : {'left', 'right', 'both'}
        Which eye(s) to include.
    expansion : float
        Expansion factor. Values > 1.0 will slightly dilate the mask to include
        eyelids and avoid hard borders.

    Returns
    -------
    np.ndarray
        Full-frame boolean mask (H, W) where True indicates the selected eye region.
    """
    import cv2
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    result = face_landmarker_task.detect(mp_image)
    face_landmarks_list = result.face_landmarks or []
    if not face_landmarks_list:
        raise RuntimeError('MediaPipe found no face landmarks.')

    # Use the first face for now (consistent with existing behavior).
    landmarks = face_landmarks_list[0]

    # Use canonical FaceMesh eye connection sets (no hard-coded indices).
    left_ids = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ]
    right_ids = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ]

    def _points(ids: Sequence[int]) -> np.ndarray:
        pts = []
        for i in ids:
            lm = landmarks[int(i)]
            x = int(round(float(lm.x) * w))
            y = int(round(float(lm.y) * h))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])
        return np.asarray(pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    if which in ('left', 'both'):
        pts = _points(left_ids)
        if pts.size > 0:
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

    if which in ('right', 'both'):
        pts = _points(right_ids)
        if pts.size > 0:
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

    # Optional dilation driven by expansion. Keep it conservative.
    exp = float(expansion)
    if exp > 1.0:
        # Convert expansion factor to a small pixel radius.
        # Example: 1.2 -> ~2 px, 1.5 -> ~4 px (capped).
        radius = int(round(min(8.0, max(0.0, (exp - 1.0) * 10.0))))
        if radius > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask, k, iterations=1)

    return (mask > 0)


def _expand_clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float) -> tuple[int, int, int, int]:
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


def _head_bbox_square_from_face(
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


def _score_sam_mask_with_landmarks(
    mask: np.ndarray,
    pose_xy: np.ndarray,
) -> tuple[int, int]:
    """
    Score a SAM mask using pose landmarks.

    Returns
    -------
    tuple[int, int]
        (num_landmarks_inside, negative_area)

        Higher is better.
    """

    h, w = mask.shape

    valid = (
        (pose_xy[:, 0] >= 0) &
        (pose_xy[:, 1] >= 0)
    )

    pts = pose_xy[valid]

    inside = 0
    for px, py in pts:
        px = int(px)
        py = int(py)

        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            inside += 1

    area = int(mask.sum())

    return (inside, -area)


def _infer_sam_model_type(ckpt: Path, sam_model_type: Optional[str]) -> str:
    if sam_model_type:
        return sam_model_type

    name = ckpt.name.lower()

    if 'vit_h' in name:
        return 'vit_h'
    if 'vit_l' in name:
        return 'vit_l'
    if 'vit_b' in name:
        return 'vit_b'

    return 'vit_h'


def _mp_pose_landmarks_xy(
    img_rgb: np.ndarray,
    *,
    pose_landmarker,
) -> np.ndarray:
    """
    Detect pose landmarks and return stable landmark coordinates in pixel space.

    This variant is intentionally tolerant to partial bodies (e.g. bust shots,
    cropped portraits, or images where hips are outside the frame). It avoids
    rejecting otherwise valid detections while still enforcing a minimal
    geometric structure.

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
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    result = pose_landmarker.detect(mp_image)

    pose_landmarks = result.pose_landmarks or []
    if not pose_landmarks:
        raise RuntimeError('MediaPipe found no pose landmarks.')

    pts = np.full((33, 2), -1, dtype=np.int32)

    strong_count = 0

    for idx, lm in enumerate(pose_landmarks[0]):
        visibility = float(getattr(lm, 'visibility', 1.0))
        presence = float(getattr(lm, 'presence', 1.0))

        # Conservative threshold that works well with cropped images
        if visibility < 0.35 or presence < 0.35:
            continue

        x = int(round(float(lm.x) * w))
        y = int(round(float(lm.y) * h))

        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        pts[idx] = (x, y)
        strong_count += 1

    # -------------------------------------------------------------
    # Basic robustness checks
    # -------------------------------------------------------------

    if strong_count < 8:
        raise RuntimeError(
            'MediaPipe pose landmarks are too weak to identify a person reliably.'
        )

    # Require at least some structural anchors.
    # Shoulders are the most reliable torso landmarks in cropped images.
    left_shoulder = (pts[11, 0] >= 0 and pts[11, 1] >= 0)
    right_shoulder = (pts[12, 0] >= 0 and pts[12, 1] >= 0)

    # Nose is almost always present when the upper body is visible.
    nose = (pts[0, 0] >= 0 and pts[0, 1] >= 0)

    if not (left_shoulder or right_shoulder or nose):
        raise RuntimeError(
            'MediaPipe pose landmarks lack stable anchors (no shoulders or nose).'
        )

    return pts


@dataclass
class SubjectCrop(NodeRef):
    """
    Subject-aware cutout and inpaint-mask generator using YOLO / MediaPipe / SAM.

    ``SubjectCrop`` detects a region of interest (person / face / head / eyes) and
    produces either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image (``mode='mask'`` or
    ``mode='negative-mask'``).

    The node is intended to support workflows such as:

    - extracting subjects for compositing (e.g. with ``ImageStack``),
    - producing inpaint masks for SDXL pipelines,
    - extracting head regions for FaceID / IP-Adapter refinement,
    - refining a small region by cropping, processing it separately, and reinserting
      it at the original coordinates.

    Pipeline
    --------
    The node combines several models to robustly localize a subject region:

    - MediaPipe Pose is always executed first to obtain body landmarks.
      These landmarks provide a geometric prior for locating the subject
      and estimating the head region.

    - YOLO (COCO class 0) proposes candidate person bounding boxes when
      ``target`` is ``'person'`` or ``'head'``.

    - Pose landmarks are used to select the most plausible person bounding
      box among YOLO detections.

    - A coarse *head area* is derived from the pose landmarks. This region
      approximates the subject head location and is used to improve the
      robustness of face and eye detection when the face is small relative
      to the full image.

    - MediaPipe Face Landmarker is executed inside the head area to obtain
      accurate face or eye landmarks.

    - Segment Anything (SAM) is used to produce segmentation masks when
      needed (person / face / head targets).

    ``target='person'``
        - MediaPipe Pose landmarks are computed for the full image.
        - YOLO proposes candidate person bounding boxes.
        - Pose landmarks are used to select the bbox that best matches the
        detected body.
        - The bbox may be expanded using ``box_margin``.
        - SAM segments inside the selected bbox.
        - The final crop region corresponds to the selected person bbox.

    ``target='face'``
        - MediaPipe Pose landmarks are used to estimate a coarse head area.
        - MediaPipe Face Landmarker runs inside this head area.
        - The largest detected face bbox is remapped to full-image coordinates.
        - SAM segments using the face bbox.

    ``target='head'``
        - MediaPipe Pose landmarks are computed.
        - YOLO proposes person bounding boxes.
        - Pose landmarks select the most plausible person bbox.
        - A head area is derived from pose landmarks.
        - Face landmarks are detected within the head area.
        - A square head crop box is computed from the face bbox using
        ``expansion`` and an upward bias to preserve hair.
        - SAM segmentation is still guided by the person bbox for robustness,
          while the crop region corresponds to the derived head box.

    ``target='eyes'``
        - MediaPipe Pose landmarks estimate a head area.
        - MediaPipe Face Landmarker runs inside that head area.
        - Eye landmarks are extracted and converted to a bounding box.
        - A landmark-derived mask is produced directly (SAM is skipped).

    ``target='left-eye'``
        - MediaPipe derives face landmarks.
        - A bounding box is computed from landmark indices corresponding to the
          subject's left eye (subject perspective, not viewer).
        - The box is optionally expanded via ``expansion``.
        - A landmark-derived mask is used directly; SAM is skipped.
        - The final crop region isolates only the left eye.

    ``target='right-eye'``
        - MediaPipe derives face landmarks.
        - A bounding box is computed from landmark indices corresponding to the
          subject's right eye (subject perspective, not viewer).
        - The box is optionally expanded via ``expansion``.
        - A landmark-derived mask is used directly; SAM is skipped.
        - The final crop region isolates only the right eye.

    Head crop geometry
    ------------------
    Let ``face_size = max(face_w, face_h)`` from the landmark-derived face box.

    The head crop box is defined as a square centered on the face, shifted upward
    to preserve hair:

    - vertical shift: ``cy -= 0.15 * face_size``
    - radius: ``radius = 0.75 * face_size * expansion``

    Parameters
    ----------
    id : str, optional
        Node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default input
        (``input['default']['image']`` or ``input['default']['path']``).

    spec : dict or str or Path, optional
        Node specification (inline dict or path to a config file), resolved via
        ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``sam_checkpoint`` : str
                Path to the SAM checkpoint. Required.
            ``sam_model_type`` : {'vit_h', 'vit_l', 'vit_b'}, optional
                SAM backbone type. If omitted, inferred from the checkpoint
                filename.
            ``device`` : str, optional
                Inference device (e.g. ``'cuda'``, ``'cuda:0'``, ``'cpu'``).
                Default: ``'cuda'``.
            ``yolo_model`` : str, optional
                YOLO weights. Required for ``target='person'`` and ``target='head'``.
            ``face_landmarker_task`` : str, optional
                MediaPipe FaceLandmarker ``.task`` path. Required for
                ``target='face'``, ``target='eyes'``, ``target='left-eye'``,
                ``target='right-eye'`` and ``target='head'``.
            ``pose_landmarker_task`` : str
                MediaPipe PoseLandmarker ``.task`` path.
                This model is required because pose landmarks are used to derive
                head regions and to select the correct person bbox.

        ``params`` : dict
            ``target`` : {'person', 'face', 'head', 'eyes', 'left-eye', 'right-eye'}, optional
                Region to extract. Default: ``'person'``.

            ``mode`` : {'default', 'mask', 'negative-mask'}, optional
                Output type:
                - ``'default'``: RGBA cutout.
                - ``'mask'``: full-frame 8-bit mask (white = selected region).
                - ``'negative-mask'``: inverted full-frame mask (white = background).
                Default: ``'default'``.

            ``crop_mode`` : {'bbox', 'trim', 'full_frame'}, optional
                Applies only when ``mode='default'`` and controls the spatial layout
                of the RGBA cutout:

                - ``'bbox'``:
                    Output is the rectangular crop inside the selected bounding box,
                    including the original background; alpha is fully opaque
                    (255 everywhere).

                - ``'trim'``:
                    Output is an RGBA cutout cropped to the selected bounding box,
                    with alpha derived from the mask. The result is then tightly
                    trimmed to the minimal box containing non-transparent pixels.

                - ``'full_frame'``:
                    Same cutout as ``'trim'`` but placed back into a full-size RGBA
                    canvas of the original image dimensions, preserving the original
                    coordinates.

                Default: ``'trim'``.

            ``conf`` : float, optional
                YOLO confidence threshold (only used when YOLO runs).
                Typical range: 0.2-0.6. Default: 0.35.

            ``box_margin`` : float, optional
                Symmetric expansion ratio applied to the selected bounding box before
                SAM, expressed as a fraction of bbox size.
                Typical range: 0.05-0.20. Default: 0.12.

            ``multimask`` : bool, optional
                If True, SAM returns multiple candidate masks and the node selects
                one via a heuristic based on bbox-center inclusion, coverage, and
                SAM score. Default: True.

            ``expansion`` : float, optional
                Head square expansion multiplier for ``target='head'`` and box
                expansion factor for eye targets. Default: 1.0.

            ``dilate_radius`` : int, optional
                Mask dilation radius in pixels (mask modes only).
                Useful to avoid edge artifacts in inpainting. Default: 0.

            ``close_radius`` : int, optional
                Morphological closing radius in pixels (mask modes only).
                Fills small holes and gaps. Default: 0.

            ``smoothing_radius`` : int, optional
                Gaussian smoothing radius in pixels (mask modes only).
                Produces softer mask edges. Default: 0.

        ``debug`` : dict
            ``save_debug`` : bool, optional
                If True, saves a debug image with the selected SAM bbox overlay.
                Default: False.

    Returns
    -------
    dict
        Output metadata dictionary (also written as a JSON sidecar) with:

        ``ok`` : bool
            Success flag.
        ``node`` : str
            Operator name.
        ``id`` : str
            Node identifier.
        ``input_image`` : str
            Source image path.
        ``mode`` : str
            Output mode.
        ``image`` : str
            Output file path (RGBA cutout or mask).
        ``bbox_xyxy`` : list[int]
            Bounding box used to run SAM.
        ``params`` : dict
            Configuration parameters.
        ``crop`` : dict
            Crop metadata useful for reinsertion/compositing:

            ``anchor_xy`` : list[int]
                Center of the selected crop box in absolute source-image coordinates.
            ``bbox_size`` : list[int]
                Width and height of the crop box in pixels: ``[b_width, b_height]``.

        ``metadata`` : str
            JSON sidecar path.

    Notes
    -----
    - Mask outputs are always full-frame and aligned to the original image size.
    - For ``target='head'``, SAM uses the selected person bbox, while the final crop
      region is the derived head box.
    - For eye targets, the mask is derived directly from face landmarks and SAM is
      skipped.
    - 'left-eye' and 'right-eye' refer to the subject perspective.
      In mirrored images this may appear inverted to the viewer.
    - Heavy models (YOLO, SAM, MediaPipe Tasks) are retrieved via the global model
      cache where available.
    - The SAM predictor is created per run because it stores per-image state.
    - If you change code or spec and need fresh outputs, delete the existing sidecar
      JSON to avoid reusing cached results.
    """

    # Either pass a path explicitly, or wire an upstream image into default input.
    path: Optional[Union[str, Path]] = None

    # Optional node spec (device, etc.)
    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        # Local imports to avoid hard deps if node unused
        import cv2
        import torch
        from segment_anything import SamPredictor

        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        # -----------------------------
        # Resolve input image path
        # -----------------------------
        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Load image
        # -----------------------------
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(
                f"SubjectCrop node '{node_id}': cannot read image: {img_path}")

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        eye_which: Optional[str] = None
        eye_mask: Optional[np.ndarray] = None

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        pose_xy = _mp_pose_landmarks_xy(
            img_rgb=img_rgb,
            pose_landmarker=pose_landmarker,
        )

        if cfg.target == 'person':
            # -----------------------------
            # YOLO: find largest 'person' bbox (COCO class 0)
            # -----------------------------
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device
            )[0]

            person_boxes_xyxy = _person_bboxes_xyxy(res, node_id)

            bx1, by1, bx2, by2 = _select_person_bbox_xyxy(
                person_boxes_xyxy,
                pose_xy=pose_xy,
            )

        elif cfg.target == 'head':
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            person_boxes_xyxy = _person_bboxes_xyxy(res, node_id)

            bx1, by1, bx2, by2 = _select_person_bbox_xyxy(
                person_boxes_xyxy,
                pose_xy=pose_xy,
            )

            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            head_area_rgb, a_x, a_y = _get_head_area(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=1.6,
            )

            r_x1, r_y1, r_x2, r_y2 = _mp_largest_face_bbox_xyxy(
                img_rgb=head_area_rgb,
                face_landmarker_task=landmarker,
            )

            fx1 = r_x1 + a_x
            fy1 = r_y1 + a_y
            fx2 = r_x2 + a_x
            fy2 = r_y2 + a_y

            fx1, fy1, fx2, fy2 = _head_bbox_square_from_face(
                fx1, fy1, fx2, fy2, w, h,
                expansion=cfg.expansion,
            )

        elif cfg.target == 'face':
            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            head_area_rgb, a_x, a_y = _get_head_area(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=1.6,
            )

            r_x1, r_y1, r_x2, r_y2 = _mp_largest_face_bbox_xyxy(
                img_rgb=head_area_rgb,
                face_landmarker_task=landmarker,
            )

            bx1 = r_x1 + a_x
            by1 = r_y1 + a_y
            bx2 = r_x2 + a_x
            by2 = r_y2 + a_y

        elif cfg.target in ('eyes', 'left-eye', 'right-eye'):
            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            eye_which = {
                'eyes': 'both',
                'left-eye': 'left',
                'right-eye': 'right',
            }[cfg.target]

            head_area_rgb, a_x, a_y = _get_head_area(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=1.6,
            )

            r_x1, r_y1, r_x2, r_y2 = _mp_eye_bbox_xyxy(
                img_rgb=head_area_rgb,
                face_landmarker_task=landmarker,
                which=eye_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            bx1 = r_x1 + a_x
            by1 = r_y1 + a_y
            bx2 = r_x2 + a_x
            by2 = r_y2 + a_y

            local_eye_mask = _mp_eye_mask_from_landmarks(
                img_rgb=head_area_rgb,
                face_landmarker_task=landmarker,
                which=eye_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            eye_mask = np.zeros((h, w), dtype=bool)
            eye_h, eye_w = local_eye_mask.shape
            eye_mask[a_y:a_y + eye_h, a_x:a_x + eye_w] = local_eye_mask

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        if cfg.box_margin > 0 and cfg.target not in ('eyes', 'left-eye', 'right-eye'):
            bx1, by1, bx2, by2 = _expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.box_margin
            )

        # -----------------------------
        # Mask generation
        # -----------------------------
        if cfg.target in ('eyes', 'left-eye', 'right-eye'):
            # Eye targets: use landmark-derived mask (skip SAM).
            if eye_mask is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': eye_mask not computed."
                )
            mask = eye_mask
            ckpt = Path(str(cfg.sam_checkpoint)).expanduser().resolve()
            model_type = _infer_sam_model_type(ckpt, cfg.sam_model_type)
        else:
            # Default path: SAM box-guided segmentation
            ckpt = Path(str(cfg.sam_checkpoint)).expanduser().resolve()
            if not ckpt.exists() or not ckpt.is_file():
                raise FileNotFoundError(
                    f"SubjectCrop node '{node_id}': SAM checkpoint not found: {ckpt}")

            model_type = _infer_sam_model_type(ckpt, cfg.sam_model_type)

            sam = get_sam(
                checkpoint=str(ckpt),
                model_type=model_type,
                device=cfg.device
            )

            predictor = SamPredictor(sam)
            predictor.set_image(np.ascontiguousarray(img_rgb))

            box = np.array([bx1, by1, bx2, by2], dtype=np.float32)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=box[None, :],
                    multimask_output=bool(cfg.multimask),
                )

            if masks is None or len(masks) == 0:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': SAM returned no masks.")

            # Pick the best mask in a robust way:
            best_i = 0
            best_key = None

            for i in range(len(masks)):

                mi = masks[i].astype(bool)

                if cfg.target == 'person':

                    key = _score_sam_mask_with_landmarks(
                        mi,
                        pose_xy,
                    )

                else:
                    frac = float(mi[by1:by2, bx1:bx2].mean())
                    score_i = float(scores[i]) if scores is not None else 0.0
                    key = (score_i, -abs(frac - 0.35))

                if best_key is None or key > best_key:
                    best_key = key
                    best_i = i

            mask = masks[int(best_i)].astype(bool)

            # ------------------------------------------------------------------
            # SAM may occasionally return the complementary (background) mask
            # when prompted with a bounding box only. In that case the subject
            # appears as a hole in the mask.
            #
            # We detect this situation using MediaPipe pose landmarks: the correct
            # subject mask should contain most valid body landmarks. If fewer than
            # half of them fall inside the predicted mask, we assume SAM returned
            # the background instead.
            #
            # When flipping the mask we must restrict the inversion to the prompt
            # bounding box. Outside that region SAM predictions are undefined and
            # inverting the whole mask would incorrectly mark the entire image as
            # foreground.
            # ------------------------------------------------------------------
            valid = (
                (pose_xy[:, 0] >= 0) &
                (pose_xy[:, 1] >= 0)
            )

            pts = pose_xy[valid]

            inside = 0
            for px, py in pts:
                px = int(px)
                py = int(py)

                if mask[py, px]:
                    inside += 1

            if inside < len(pts) / 2:
                # Invert mask only inside the prompt bbox to avoid turning the entire
                # background of the image into foreground.
                inv = ~mask
                new_mask = np.zeros_like(mask, dtype=bool)
                new_mask[by1:by2, bx1:bx2] = inv[by1:by2, bx1:bx2]
                mask = new_mask

        # --------------------------------------------------
        # Select crop region depending on target
        # --------------------------------------------------

        if cfg.target == 'head':
            # crop strictly around face/head bbox (full-image coordinates)
            crop_x1, crop_y1, crop_x2, crop_y2 = fx1, fy1, fx2, fy2
        else:
            # person mode
            crop_x1, crop_y1, crop_x2, crop_y2 = bx1, by1, bx2, by2

        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_mask.size == 0:
            raise RuntimeError(
                f"SubjectCrop node '{node_id}': empty crop after bbox."
            )

        cm = crop_mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm, connectivity=8
        )

        if num > 1:
            if cfg.target in ('eyes', 'left-eye', 'right-eye'):
                # Keep all components in the crop.
                # For 'eyes' this preserves both eyes (two disjoint blobs).
                # For 'left-eye'/'right-eye' the crop box is expected to isolate a single eye.
                crop_mask = (labels != 0)

            else:
                # Keep a single component for other targets.
                ccx = cm.shape[1] // 2
                ccy = cm.shape[0] // 2
                target = labels[ccy, ccx]

                if target == 0:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    target = 1 + int(np.argmax(areas))

                crop_mask = (labels == target)

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        out_x1 = int(crop_x1)
        out_y1 = int(crop_y1)
        out_x2 = int(crop_x2)
        out_y2 = int(crop_y2)

        if cfg.mode == 'default':
            crop_rgb = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2, :]
            crop_h = int(crop_y2 - crop_y1)
            crop_w = int(crop_x2 - crop_x1)

            if cfg.crop_mode == 'bbox':
                # Include the original background inside the crop; alpha is fully opaque.
                alpha = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                crop_rgba = np.dstack([crop_rgb, alpha])
                Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

            else:
                # 'trim' and 'full_frame' -> alpha of mask
                alpha = (crop_mask.astype(np.uint8) * 255)
                crop_rgba = np.dstack([crop_rgb, alpha])

                if cfg.crop_mode == 'trim':
                    tx1, ty1, tx2, ty2 = _tight_alpha_bbox(alpha)
                    crop_rgba = crop_rgba[ty1:ty2, tx1:tx2, :]

                    out_x1 = int(crop_x1 + tx1)
                    out_y1 = int(crop_y1 + ty1)
                    out_x2 = int(crop_x1 + tx2)
                    out_y2 = int(crop_y1 + ty2)

                    Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

                elif cfg.crop_mode == 'full_frame':
                    full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    full_rgba[crop_y1:crop_y2, crop_x1:crop_x2, :] = crop_rgba

                    # For full-frame outputs, crop metadata must describe the output image
                    # itself in source-image coordinates.
                    out_x1 = 0
                    out_y1 = 0
                    out_x2 = w
                    out_y2 = h

                    Image.fromarray(full_rgba, mode='RGBA').save(out_path)
                else:
                    raise ValueError(
                        f'{self.id}: invalid crop_mode={cfg.crop_mode!r}')

        else:
            # build full-frame mask (same size as source image)
            full_mask = np.zeros((h, w), dtype=np.uint8)

            # put cropped mask back into absolute position
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask.astype(
                np.uint8) * 255

            if cfg.mode == 'negative-mask':
                full_mask = 255 - full_mask

            # post-process (expand + close + smooth)
            full_mask = postprocess_mask(
                full_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            Image.fromarray(full_mask, mode='L').save(out_path)

        # Optional debug bbox overlay
        dbg_path = None
        if cfg.save_debug:
            dbg = img_rgb.copy()
            dbg_x1, dbg_y1, dbg_x2, dbg_y2 = bx1, by1, bx2, by2
            if cfg.target == 'head':
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = fx1, fy1, fx2, fy2
            cv2.rectangle(
                dbg,
                (dbg_x1, dbg_y1),
                (dbg_x2, dbg_y2),
                (255, 0, 0),
                3,
            )
            dbg_path = out_path.with_name(out_path.stem + '_debug_bbox.png')
            Image.fromarray(dbg).save(dbg_path)

        b_width = int(out_x2 - out_x1)
        b_height = int(out_y2 - out_y1)
        anchor_x = int((out_x1 + out_x2) // 2)
        anchor_y = int((out_y1 + out_y2) // 2)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'sam_bbox_xyxy': [int(bx1), int(by1), int(bx2), int(by2)],
            'model': {
                **({} if cfg.yolo_model is None else {'yolo': cfg.yolo_model}),
                'sam_checkpoint': str(ckpt),
                'sam_model_type': model_type,
                **({} if cfg.face_landmarker_task is None else {
                    'face_landmarker_task': cfg.face_landmarker_task,
                }),
                **({} if cfg.pose_landmarker_task is None else {
                    'pose_landmarker_task': cfg.pose_landmarker_task,
                }),
                'device': cfg.device,
            },
            'params': {
                'target': cfg.target,
                'mode': cfg.mode,
                'crop_mode': cfg.crop_mode,
                'conf': cfg.conf,
                'box_margin': cfg.box_margin,
                'multimask': cfg.multimask,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
            },
            'crop': {
                'anchor_xy': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
            }
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
