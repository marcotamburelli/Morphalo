import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------
# Config
# ----------------------------

@dataclass
class ModelPaths:
    pose_task: str
    hand_task: str
    face_task: str

# Edges (segmenti) per disegnare scheletro.
# Pose: uso i landmark di BlazePose (33). Qui una versione "snella" (arti + torso + gambe).
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


mp.tasks.vision.FaceDetectorOptions
# ----------------------------
# Utility drawing
# ----------------------------

def _to_px(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    """MediaPipe Tasks restituisce coordinate normalizzate [0..1] (tipicamente)."""
    return int(round(x * w)), int(round(y * h))

def draw_points(img: np.ndarray, pts_xy: List[Tuple[int, int]], radius: int, color: Tuple[int,int,int], thickness: int = -1):
    for (x, y) in pts_xy:
        cv2.circle(img, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)

def draw_edges(img: np.ndarray, pts_xy: List[Tuple[int, int]], edges: List[Tuple[int,int]], color: Tuple[int,int,int], thickness: int):
    for a, b in edges:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.line(img, pts_xy[a], pts_xy[b], color, thickness, lineType=cv2.LINE_AA)

def draw_face_connections(img: np.ndarray, pts_xy: List[Tuple[int,int]], connections, color: Tuple[int,int,int], thickness: int):
    # connections è una set/list di coppie (a,b)
    for a, b in connections:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.line(img, pts_xy[a], pts_xy[b], color, thickness, lineType=cv2.LINE_AA)

def draw_face_points(img: np.ndarray, pts_xy: List[Tuple[int,int]], connections, radius, color: Tuple[int,int,int], thickness: int = -1):
    # connections è una set/list di coppie (a,b)
    for a, b in connections:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.circle(img, pts_xy[a], radius, color, thickness, lineType=cv2.LINE_AA)


# ----------------------------
# Main: run tasks + render
# ----------------------------

def render_skeleton_from_image(
    image_path: str,
    out_path: str,
    models: ModelPaths,
    thickness: int = 2,
    point_radius: int = 2,
    conf_min_pose: float = 0.3,
    conf_min_hand: float = 0.3,
    conf_min_face: float = 0.3,
):
    # 1) Load image
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Immagine non trovata o non leggibile: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 2) Setup MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    # Validazione path modelli (ti evita il solito panico)
    for p in (models.pose_task, models.hand_task, models.face_task):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Modello .task non trovato: {p}")

    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=models.pose_task),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=conf_min_pose,
        min_pose_presence_confidence=conf_min_pose,
        min_tracking_confidence=conf_min_pose,
    )

    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=models.hand_task),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=conf_min_hand,
        min_hand_presence_confidence=conf_min_hand,
        min_tracking_confidence=conf_min_hand,
    )

    face_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=models.face_task),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=conf_min_face,
        min_face_presence_confidence=conf_min_face,
        min_tracking_confidence=conf_min_face,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # 3) Inference
    with PoseLandmarker.create_from_options(pose_opts) as pose:
        pose_res = pose.detect(mp_image)

    with HandLandmarker.create_from_options(hand_opts) as hands:
        hand_res = hands.detect(mp_image)

    with FaceLandmarker.create_from_options(face_opts) as face:
        face_res = face.detect(mp_image)

    # 4) Render overlay
    out = bgr.copy()

    # ---- Pose
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks[0]  # una persona
        pts = [_to_px(p.x, p.y, w, h) for p in lm]
        # linee + punti
        draw_edges(out, pts, POSE33_EDGES, color=(0, 255, 0), thickness=thickness)
        draw_points(out, pts, radius=point_radius, color=(0, 255, 0))

    # ---- Hands
    if hand_res.hand_landmarks:
        for hand_lm in hand_res.hand_landmarks:
            pts = [_to_px(p.x, p.y, w, h) for p in hand_lm]
            draw_edges(out, pts, HAND21_EDGES, color=(255, 0, 0), thickness=thickness)
            draw_points(out, pts, radius=point_radius, color=(255, 0, 0))

    # ---- Face (connessioni “essenziali”)
    if face_res.face_landmarks:
        face_lm = face_res.face_landmarks[0]
        pts = [_to_px(p.x, p.y, w, h) for p in face_lm]

        for conn_set in FACE_CONNECTIONS:
            draw_face_points(out, pts, conn_set, radius=point_radius, color=(0, 200, 255))

        # opzionale: qualche punto
        # draw_points(out, pts[::2], radius=1, color=(0, 200, 255))  # uno ogni 2 per non impastare

    # 5) Save
    ok = cv2.imwrite(out_path, out)
    if not ok:
        raise RuntimeError(f"Impossibile salvare output in: {out_path}")

    print(f"OK: salvato {out_path}")


if __name__ == "__main__":
    # ESEMPIO USO:
    # metti qui i path reali ai tuoi .task
    models = ModelPaths(
        pose_task="/home/marco/models/mediapipe/pose_landmarker_heavy.task",
        hand_task="/home/marco/models/mediapipe/hand_landmarker.task",
        face_task="/home/marco/models/mediapipe/face_landmarker.task",
    )

    render_skeleton_from_image(
        image_path="/home/marco/images/elven_princess.png",
        out_path="./outputs/skeleton_overlay.png",
        models=models,
        thickness=2,
        point_radius=2,
        conf_min_pose=0.3,
        conf_min_hand=0.3,
        conf_min_face=0.3,
    )
