import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

from stability.cache import CacheKey, ModelCache

# ----------------------------
# Config
# ----------------------------


@dataclass
class ModelPaths:
    pose_task: str
    hand_task: str
    face_task: str

    def __post_init__(self):
        self.pose_task = self._normalize(self.pose_task)
        self.hand_task = self._normalize(self.hand_task)
        self.face_task = self._normalize(self.face_task)

    @staticmethod
    def _normalize(p: str) -> str:
        return str(
            Path(p)
            .expanduser()
            .resolve()
        )


# Edges (segments) to draw the skeleton.
# Pose: I use BlazePose landmarks (33). Here's a 'slim' version (limbs + torso + legs).
POSE33_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                      # torso/hips
    (23, 25), (25, 27), (24, 26), (26, 28),            # legs
]

# Hand: 21 landmark
HAND21_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)])

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5),
                           (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97),
                           (97, 2), (2, 326), (326, 327), (327, 294),
                           (294, 278), (278, 344), (344, 440), (440, 275),
                           (275, 4), (4, 45), (45, 220), (220, 115), (115, 48),
                           (48, 64), (64, 98)])

FACE_CONNECTIONS = [
    FACEMESH_FACE_OVAL,
    FACEMESH_NOSE,
    FACEMESH_LIPS,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
]

# ----------------------------
# Utility drawing
# ----------------------------


def compute_long_side_resize(h: int, w: int, long_side: int) -> Tuple[int, int, float]:
    if long_side <= 0:
        return h, w, 1.0

    cur = max(h, w)
    if cur == long_side:
        return h, w, 1.0

    scale = float(long_side) / float(cur)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    return new_h, new_w, scale


def resize_to(img: np.ndarray, new_h: int, new_w: int, interpolation=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (new_h, new_w):
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def resize_long_side(img: np.ndarray, long_side: int, interpolation=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w, _ = compute_long_side_resize(h, w, long_side)
    return resize_to(img, new_h, new_w, interpolation=interpolation)


def _to_px(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    """MediaPipe Tasks returns normalized coordinates [0..1]"""
    return int(round(x * w)), int(round(y * h))


def draw_edges(img: np.ndarray, pts_xy: List[Tuple[int, int]], edges: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int):
    for a, b in edges:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.line(img, pts_xy[a], pts_xy[b], color,
                     thickness, lineType=cv2.LINE_AA)


def draw_face_connections(img: np.ndarray, pts_xy: List[Tuple[int, int]], connections, color: Tuple[int, int, int], thickness: int):
    for a, b in connections:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.line(img, pts_xy[a], pts_xy[b], color,
                     thickness, lineType=cv2.LINE_AA)


def draw_face_points(img: np.ndarray, pts_xy: List[Tuple[int, int]], connections, radius, color: Tuple[int, int, int], thickness: int = -1):
    for a, b in connections:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.circle(img, pts_xy[a], radius, color,
                       thickness, lineType=cv2.LINE_AA)


class SkeletonExtractor:
    """
    Skeleton extraction and rendering utility based on MediaPipe Tasks.

    ``SkeletonExtractor`` provides a high-level interface to extract human pose,
    hand, and face landmarks from images or video frames using MediaPipe Tasks
    and to render them as a compact skeleton-style overlay suitable for downstream
    processing (e.g. control videos for generative video pipelines).

    The class is designed to be used in both image-based and video-based workflows.
    In video mode, it supports lightweight temporal stabilization and tracking
    for pose and hand landmarks in order to reduce jitter and intermittent
    detection failures commonly observed in low-quality or highly compressed
    footage.

    Typical usage consists of:
      1. Initializing the extractor with the desired MediaPipe models and thresholds.
      2. Feeding frames sequentially (optionally with timestamps in video mode).
      3. Rendering a skeleton overlay with either a preserved background or a
         black background suitable for control inputs.

    The extractor prioritizes robustness and interpretability over visual fidelity:
    pose landmarks are always favored, while hand and face landmarks may be
    filtered, held, or omitted depending on confidence and temporal stability.

    Notes
    -----
    - Input frames are expected in **BGR** color order (OpenCV convention).
    - The rendered output is a visualization of landmarks only; it does not
      return raw landmark coordinates.
    - Face landmarks are rendered without temporal tracking or smoothing in the
      current implementation.
    - The extractor is intended for preprocessing and control generation rather
      than final visualization or biometric analysis.
    """

    def __init__(
        self,
        models: ModelPaths,
        conf_min_pose: float = 0.3,
        conf_min_hand: float = 0.3,
        conf_min_face: float = 0.3,
        max_poses: int = 4,
        max_faces: int = 4,
        max_hands: int = 8,
        running_mode: str = 'VIDEO',  # 'IMAGE' o 'VIDEO'
        smooth: bool = True,
        pose_tracking: bool = True,
        hands_tracking: bool = True,
        alpha: float = 0.85,
    ):
        """
        Initialize a ``SkeletonExtractor`` instance.

        Parameters
        ----------
        models : ModelPaths
            Container holding filesystem paths to the MediaPipe `.task` model assets
            used for pose, hand, and face landmark detection.
        conf_min_pose : float, optional
            Minimum confidence threshold for pose detections. Detections below this
            value are ignored. Default is 0.3.
        conf_min_hand : float, optional
            Minimum confidence threshold for hand detections. Default is 0.3.
        conf_min_face : float, optional
            Minimum confidence threshold for face detections. Default is 0.3.
        max_poses : int, optional
            Maximum number of poses to detect per frame. Default is 4.
        max_faces : int, optional
            Maximum number of faces to detect per frame. Default is 4.
        max_hands : int, optional
            Maximum number of hands to detect per frame. Default is 8.
        running_mode : {"IMAGE", "VIDEO"}, optional
            MediaPipe Tasks running mode. In ``"VIDEO"`` mode, landmark detection
            expects a monotonically increasing timestamp per frame. Default is
            ``"VIDEO"``.
        smooth : bool, optional
            Whether to apply exponential moving average (EMA) smoothing to tracked
            pose and hand landmarks. Default is True.
        pose_tracking : bool, optional
            Enable temporal tracking for pose landmarks. Default is True.
        hands_tracking : bool, optional
            Enable temporal tracking for hand landmarks. Default is True.
        alpha : float, optional
            EMA smoothing factor in the range [0, 1]. Higher values yield smoother
            trajectories at the cost of increased temporal lag. Default is 0.85.

        Raises
        ------
        FileNotFoundError
            If one or more MediaPipe model files specified in ``models`` do not exist.
        ValueError
            If an unsupported ``running_mode`` is specified.
        """
        for p in (models.pose_task, models.hand_task, models.face_task):
            if not os.path.isfile(p):
                raise FileNotFoundError(f'.task model not found: {p}')

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        rm = VisionRunningMode.VIDEO \
            if running_mode.upper() == 'VIDEO' else VisionRunningMode.IMAGE

        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=models.pose_task),
                running_mode=rm,
                num_poses=max_poses,
                min_pose_detection_confidence=conf_min_pose,
                min_pose_presence_confidence=conf_min_pose,
                min_tracking_confidence=conf_min_pose,
            )
        )

        self._hands = mp.tasks.vision.HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=models.hand_task),
                running_mode=rm,
                num_hands=max_hands,
                min_hand_detection_confidence=conf_min_hand,
                min_hand_presence_confidence=conf_min_hand,
                min_tracking_confidence=conf_min_hand,
            )
        )

        self._face = mp.tasks.vision.FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=models.face_task),
                running_mode=rm,
                num_faces=max_faces,
                min_face_detection_confidence=conf_min_face,
                min_face_presence_confidence=conf_min_face,
                min_tracking_confidence=conf_min_face,
            )
        )

        self._running_mode = rm

        # distance threshold in NORMALIZED coords (0..1). Start conservative.
        self._pose_dist_max = 0.06       # ~6% of frame; tune 0.04..0.10

        # --- tracking state
        self._pose_state = {'tracks': {}, 'next_id': 0}
        self._hand_state = {'tracks': {}, 'next_id': 0}

        self._trk_cfg = {
            'pose': dict(enable=pose_tracking, smooth=smooth, alpha=alpha, hold=3, max_tracks=8, max_missing=12, dist_max=0.06),
            'hand': dict(enable=hands_tracking, smooth=smooth, alpha=alpha, hold=2, max_tracks=8, max_missing=10, dist_max=0.08),
        }

    def _ema_landmarks(self, prev_lms, curr_lms, alpha: float):
        """
        Exponential moving average smoothing between two landmark lists.
        prev_lms/curr_lms: list of landmarks with .x .y .z (and optionally .visibility/.presence).
        Returns a new list of landmarks (same type as input items).
        """
        out = []
        for p, c in zip(prev_lms, curr_lms):
            lm = type(c)()
            lm.x = alpha * p.x + (1.0 - alpha) * c.x
            lm.y = alpha * p.y + (1.0 - alpha) * c.y
            lm.z = alpha * p.z + (1.0 - alpha) * c.z

            # Keep current confidence-like fields when available
            if hasattr(c, 'visibility'):
                lm.visibility = c.visibility
            if hasattr(c, 'presence'):
                lm.presence = c.presence

            out.append(lm)
        return out

    def _pose_center(self, lm):
        ls, rs, lh, rh = lm[11], lm[12], lm[23], lm[24]
        return ((ls.x + rs.x + lh.x + rh.x)/4.0, (ls.y + rs.y + lh.y + rh.y)/4.0)

    def _hand_center(self, lm):
        w, i, p = lm[0], lm[5], lm[17]
        return ((w.x + i.x + p.x)/3.0, (w.y + i.y + p.y)/3.0)

    def _track_and_stabilize(
        self,
        curr_items,          # List[List[Landmark]]
        state,               # {'tracks': dict, 'next_id': int}
        cfg,                 # dict con enable/smooth/alpha/hold/max_tracks/max_missing/dist_max
        center_fn,           # function(lm) -> (cx,cy) normalized
    ):
        """
        Track and temporally stabilize a set of landmark groups using greedy matching.

        This helper implements lightweight multi-object tracking and temporal
        stabilization for landmark-based detections (e.g. body poses or hands).
        It assigns persistent track IDs across frames using greedy nearest-neighbor
        matching in normalized image coordinates, optionally applying exponential
        moving average (EMA) smoothing and short-term hold when detections are missing.

        The function is intentionally generic and can be reused for different
        landmark types (pose, hands, face) by providing an appropriate ``center_fn``
        and configuration.

        Parameters
        ----------
        curr_items : list of list of Landmark
            Current-frame detections. Each element represents a single detected
            object (e.g. one person pose or one hand) and is a list of landmarks
            in normalized coordinates. May be an empty list if no detections
            are present in the current frame.
        state : dict
            Mutable tracking state for this landmark type. Expected keys:
            - ``'tracks'`` : dict[int, dict]
                Mapping from track_id to track data. Each track data dict must
                contain:
                - ``'lm'`` : list of Landmark
                    Last stabilized landmark list for this track.
                - ``'center'`` : tuple(float, float)
                    Last known center (cx, cy) in normalized coordinates.
                - ``'miss'`` : int
                    Number of consecutive frames without a matched detection.
            - ``'next_id'`` : int
                Next track ID to assign when creating a new track.
        cfg : dict
            Configuration dictionary controlling tracking and smoothing behavior.
            Expected keys:
            - ``enable`` : bool
                If False, disables tracking and returns current detections without
                temporal association.
            - ``smooth`` : bool
                Whether to apply EMA smoothing to matched tracks.
            - ``alpha`` : float
                EMA smoothing factor in [0, 1]. Higher values increase temporal
                stability but introduce more lag.
            - ``hold`` : int
                Number of frames to keep rendering a track after a missed detection.
            - ``max_tracks`` : int
                Maximum number of simultaneous tracks.
            - ``max_missing`` : int
                Maximum number of consecutive missed frames before a track is dropped.
            - ``dist_max`` : float
                Maximum allowed matching distance (normalized units) between
                centers for associating a detection to an existing track.
        center_fn : callable
            Function computing a representative center for a landmark group.
            Signature: ``center_fn(lm) -> (cx, cy)``, where ``lm`` is a list of
            landmarks and ``cx, cy`` are normalized coordinates in [0, 1].

        Returns
        -------
        dict[int, list of Landmark]
            Dictionary mapping active track IDs to stabilized landmark lists for
            the current frame. Tracks are included if they were matched in the
            current frame or are within the configured hold window.

        Notes
        -----
        - Matching is performed using greedy nearest-neighbor association based
        solely on center distance; it does not guarantee globally optimal
        assignment.
        - Track identity is stable across frames only under moderate motion and
        limited occlusion; frequent crossings may lead to ID swaps.
        - This method does not rely on bounding boxes or appearance features and
        is intended as a lightweight alternative to full multi-object trackers.
        - The returned track IDs are suitable for temporal smoothing and internal
        consistency but are not exposed in the rendered output.
        """
        curr = curr_items or []
        tracks = state['tracks']

        if not cfg['enable']:
            # No tracking: return 'as-is' with synthetic ids by index (not stable)
            return {i: lm for i, lm in enumerate(curr)}

        curr_centers = [center_fn(lm) for lm in curr]

        used = set()
        matched = {}

        # greedy match each existing track to closest unused current item
        for tid, data in list(tracks.items()):
            px, py = data['center']
            best_i, best_d2 = None, 1e9
            for i, (cx, cy) in enumerate(curr_centers):
                if i in used:
                    continue
                dx, dy = px - cx, py - cy
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2, best_i = d2, i

            if best_i is not None and best_d2 <= (cfg['dist_max'] ** 2):
                matched[tid] = best_i
                used.add(best_i)

        # update matched tracks (optional EMA)
        for tid, i in matched.items():
            new_lm = curr[i]
            prev_lm = tracks[tid]['lm']

            if cfg['smooth'] and prev_lm is not None and len(prev_lm) == len(new_lm):
                new_lm = self._ema_landmarks(prev_lm, new_lm, cfg['alpha'])

            tracks[tid]['lm'] = new_lm
            tracks[tid]['center'] = curr_centers[i]
            tracks[tid]['miss'] = 0

        # increment miss for unmatched tracks; drop after max_missing
        dead = []
        for tid, data in tracks.items():
            if tid in matched:
                continue
            data['miss'] += 1
            if data['miss'] > cfg['max_missing']:
                dead.append(tid)
        for tid in dead:
            tracks.pop(tid, None)

        # create new tracks for unused curr items
        for i, lm in enumerate(curr):
            if i in used:
                continue
            if len(tracks) >= cfg['max_tracks']:
                break
            tid = state['next_id']
            state['next_id'] += 1
            tracks[tid] = {'lm': lm, 'center': curr_centers[i], 'miss': 0}

        # output active tracks (hold_missing)
        out = {}
        for tid, data in tracks.items():
            if data['miss'] == 0 or data['miss'] <= cfg['hold']:
                out[tid] = data['lm']
        return out

    def close(self):
        """
        Release MediaPipe resources associated with this extractor.

        This method closes the underlying MediaPipe landmarkers and frees any
        resources they hold. It should be called when the ``SkeletonExtractor`` is
        no longer needed, especially in long-running processes or batch pipelines,
        to avoid resource leaks.

        After calling this method, the extractor instance should not be used to
        process additional frames.

        Returns
        -------
        None
        """
        self._pose.close()
        self._hands.close()
        self._face.close()

    def render_frame(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        thickness: int = 2,
        point_radius: int = 2,
        preserve_bg: bool = False,
        timestamp_ms: int = 0,
        long_side: int | None = None,
    ) -> cv2.typing.MatLike:
        """
        Render a skeleton overlay for a single image or video frame.

        This method runs pose, hand, and face landmark detection on a single input
        frame and renders the detected landmarks as a compact skeleton-style
        visualization. The output image can either preserve the original background
        or be rendered on a black background, making it suitable for use as a
        control signal in downstream video generation pipelines.

        In video mode, this method supports temporal stabilization and tracking for
        pose and hand landmarks. When enabled, detected landmarks are associated
        across frames and optionally smoothed to reduce jitter and short-lived
        detection failures.

        Parameters
        ----------
        frame_bgr : cv2.typing.MatLike
            Input frame in **BGR** color order (OpenCV convention).
        thickness : int, optional
            Line thickness used when drawing pose and hand edges. Default is 2.
        point_radius : int, optional
            Radius used when rendering face landmark points. Default is 2.
        preserve_bg : bool, optional
            If True, the skeleton overlay is drawn on top of the original frame.
            If False, the overlay is drawn on a black background. Default is False.
        timestamp_ms : int, optional
            Frame timestamp in milliseconds. Required when operating in ``"VIDEO"``
            running mode to ensure correct temporal tracking. Ignored in ``"IMAGE"``
            mode. Default is 0.
        long_side : int or None, optional
            If provided, the output image is resized so that its longest side equals
            this value while preserving aspect ratio. If None, no resizing is
            performed. Default is None.

        Returns
        -------
        cv2.typing.MatLike
            Output frame containing the rendered skeleton overlay in BGR format.

        Notes
        -----
        - Pose landmarks are always rendered when detected.
        - Hand and face landmarks may be omitted on frames where detection confidence
        is low or detection fails.
        - For low-resolution or highly compressed videos, hand and face detections
        may be intermittent; pose landmarks are generally more robust.
        - This method does not modify internal state other than updating tracking
        information when tracking is enabled.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == mp.tasks.vision.RunningMode.VIDEO:
            pose_res = self._pose.detect_for_video(mp_image, timestamp_ms)
            hand_res = self._hands.detect_for_video(mp_image, timestamp_ms)
            face_res = self._face.detect_for_video(mp_image, timestamp_ms)
        else:
            pose_res = self._pose.detect(mp_image)
            hand_res = self._hands.detect(mp_image)
            face_res = self._face.detect(mp_image)

        # render
        out = frame_bgr.copy() if preserve_bg else np.zeros_like(frame_bgr)

        POSE_COLOR = (0, 255, 0)
        HAND_COLOR = (255, 0, 0)
        FACE_COLOR = (0, 200, 255)

        pose_tracks = self._track_and_stabilize(
            curr_items=pose_res.pose_landmarks,
            state=self._pose_state,
            cfg=self._trk_cfg['pose'],
            center_fn=self._pose_center
        )
        if pose_tracks:
            # stable ordering: by track_id (important if you ever export per-track later)
            for tid in sorted(pose_tracks.keys()):
                lm = pose_tracks[tid]
                pts = [_to_px(p.x, p.y, w, h) for p in lm]
                draw_edges(
                    img=out,
                    pts_xy=pts,
                    edges=POSE33_EDGES,
                    color=POSE_COLOR,
                    thickness=thickness
                )

        hand_tracks = self._track_and_stabilize(
            curr_items=hand_res.hand_landmarks,
            state=self._hand_state,
            cfg=self._trk_cfg['hand'],
            center_fn=self._hand_center
        )
        if hand_tracks:
            for tid in sorted(hand_tracks.keys()):
                lm = hand_tracks[tid]
                pts = [_to_px(p.x, p.y, w, h) for p in lm]
                draw_edges(
                    img=out,
                    pts_xy=pts,
                    edges=HAND21_EDGES,
                    color=HAND_COLOR,
                    thickness=thickness
                )

        if face_res.face_landmarks:
            for face_lm in face_res.face_landmarks:
                pts = [_to_px(p.x, p.y, w, h) for p in face_lm]
                for conn_set in FACE_CONNECTIONS:
                    draw_face_points(
                        img=out,
                        pts_xy=pts,
                        connections=conn_set,
                        radius=point_radius,
                        color=FACE_COLOR
                    )

        if long_side is not None:
            out = resize_long_side(out, long_side)

        return out


class CannyExtractor:
    """
    Canny edge extractor and renderer for video control maps.
    """

    def render_frame(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        low_threshold: int = 80,
        high_threshold: int = 160,
        blur_ksize: int = 5,
        blur_sigma: float = 0.0,
        aperture_size: int = 3,
        l2_gradient: bool = False,
        dilate: int = 0,
        dilate_iter: int = 1,
        preserve_bg: bool = False,
        invert: bool = False,
        long_side: int | None = None,
    ) -> cv2.typing.MatLike:
        """
        Render a Canny edge overlay for a single frame.

        Parameters
        ----------
        frame_bgr : cv2.typing.MatLike
            Input frame in BGR format.
        low_threshold, high_threshold : int
            Canny thresholds.
        blur_ksize : int
            Gaussian blur kernel size (odd). Use 0 or 1 to disable.
        blur_sigma : float
            Gaussian sigma. 0 lets OpenCV choose automatically.
        aperture_size : int
            Canny Sobel aperture size (3, 5, or 7).
        l2_gradient : bool
            Use a more precise L2 norm for gradient magnitude.
        dilate : int
            If >0, apply morphological dilation to thicken edges.
        dilate_iter : int
            Dilation iterations.
        preserve_bg : bool
            If True, overlay edges on the original frame; else black background.
        invert : bool
            If True, invert edge colors (useful for some pipelines).
        long_side : int or None
            Optional resize of output to a specific long side.

        Returns
        -------
        cv2.typing.MatLike
            Output BGR frame.
        """
        # Ensure we operate on a NumPy array (MatLike may be UMat, etc.)
        frame = np.asarray(frame_bgr)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if blur_ksize and blur_ksize > 1:
            # OpenCV requires odd kernel sizes
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

        # Canny aperture size must be 3, 5, or 7
        if aperture_size not in (3, 5, 7):
            raise ValueError(
                f"aperture_size must be 3, 5, or 7, got {aperture_size}")

        edges = cv2.Canny(
            gray,
            threshold1=int(low_threshold),
            threshold2=int(high_threshold),
            apertureSize=int(aperture_size),
            L2gradient=bool(l2_gradient),
        )

        if dilate and dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=int(dilate_iter))

        if invert:
            edges = 255 - edges

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if preserve_bg:
            out = frame.copy()
            m = edges > 0
            out[m] = (255, 255, 255)
        else:
            out = edges_bgr

        if long_side is not None:
            out = resize_long_side(out, int(long_side))

        return out


@dataclass
class DepthVideoExtractor:
    """
    Depth estimator for video frames using DPT (MiDaS/DPT via HuggingFace).

    This extractor is responsible for:
    - lazy-loading and caching the processor/model
    - running per-frame depth estimation
    - converting depth to a 3-channel BGR control frame
    """

    model_id: str = "Intel/dpt-hybrid-midas"
    device: str = "cuda"
    autocast: bool = True

    # cached per-process handles (avoid repeated lookups in hot loop)
    _processor: Optional[DPTImageProcessor] = field(
        default=None, init=False, repr=False)
    _model: Optional[DPTForDepthEstimation] = field(
        default=None, init=False, repr=False)

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        # Use your existing cache logic (same as get_depth_estimator)
        proc_key = CacheKey(kind="depth_processor",
                            ref=self.model_id, device="cpu", dtype="na")
        mod_key = CacheKey(kind="depth_model", ref=self.model_id,
                           device=self.device, dtype="na")

        processor = ModelCache.get(proc_key)
        if processor is None:
            processor = ModelCache.put(
                proc_key, DPTImageProcessor.from_pretrained(self.model_id))

        model = ModelCache.get(mod_key)
        if model is None:
            model = DPTForDepthEstimation.from_pretrained(
                self.model_id).to(self.device)
            model.eval()
            ModelCache.put(mod_key, model)

        self._processor = processor
        self._model = model

    @torch.no_grad()
    def predict_depth_tensor(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        long_side_infer: int | None = None,
    ) -> torch.Tensor:
        """
        Return raw predicted depth as a (H, W) float tensor on the model device.
        """
        from PIL import Image

        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None

        frame = np.asarray(frame_bgr)
        if long_side_infer is not None:
            frame = resize_long_side(frame, int(long_side_infer))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        pixel_values = self._processor(
            images=pil, return_tensors="pt").pixel_values.to(self.device)

        use_autocast = bool(self.autocast) and str(
            self.device).startswith("cuda")
        if use_autocast:
            with torch.autocast("cuda"):
                depth = self._model(pixel_values).predicted_depth
        else:
            depth = self._model(pixel_values).predicted_depth

        return depth[0].to(dtype=torch.float32)

    @torch.no_grad()
    def render_frame(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        long_side_infer: int | None = None,
        normalize: Literal["per_frame", "global"] = "per_frame",
        invert: bool = False,
        clip_p_low: float = 2.0,
        clip_p_high: float = 98.0,
        global_minmax: Optional[Tuple[float, float]] = None,
    ) -> cv2.typing.MatLike:
        """

        Estimate depth for a single frame and return a 3-channel BGR depth map.

        Parameters
        ----------
        frame_bgr : cv2.typing.MatLike
            Input frame in BGR format.
        long_side_infer : int or None
            Optional resize (long side) applied *before* inference (speed/quality tradeoff).
        normalize : {"per_frame", "global"}
            Normalization strategy for mapping depth to [0, 255].
        invert : bool
            If True, invert depth visualization (near/far swap).
        clip_p_low, clip_p_high : float
            Percentile clipping to reduce outlier influence before normalization.
        global_minmax : (float, float) or None
            If normalize="global", provide (min, max) depth values used for normalization.

        Returns
        -------
        cv2.typing.MatLike
            Depth visualization as BGR uint8 (H, W, 3).
        """
        d = self.predict_depth_tensor(
            frame_bgr,
            long_side_infer=long_side_infer
        )

        if normalize == "global" and global_minmax is not None:
            glo, ghi = global_minmax
            lo = torch.tensor(glo, device=d.device, dtype=d.dtype)
            hi = torch.tensor(ghi, device=d.device, dtype=d.dtype)

            # scale (no hard clamp)
            d = (d - lo) / (hi - lo + 1e-8)

            # soft bound to [0, 1]
            d = torch.clamp(d, 0.0, 1.0)

        else:
            # per-frame robust scaling (no hard clamp)
            ql = float(clip_p_low) / 100.0
            qh = float(clip_p_high) / 100.0

            dq = d.float()  # quantile requires float32
            lo = torch.quantile(dq, ql)
            hi = torch.quantile(dq, qh)

            # scale instead of clamp
            d = (d - lo) / (hi - lo + 1e-8)

            # soft bound
            d = torch.clamp(d, 0.0, 1.0)

        if invert:
            d = 1.0 - d

        img = (d * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return out
