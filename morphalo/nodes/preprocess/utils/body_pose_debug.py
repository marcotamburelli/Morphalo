"""Draw body fitting and MediaPipe crop diagnostics."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from morphalo.nodes.preprocess.utils.body_pose_fit import (
    FIT_SEGMENTS,
    HIPS,
    SHOULDERS,
    BodyLandmarks,
    BodyPoseCrop,
)


# Include the sides of the torso for context around the fitted center line.
BODY_CONNECTIONS = (*FIT_SEGMENTS, (0, 6), (1, 7))
RIG_COLOR = (0, 220, 255)
LANDMARK_COLOR = (255, 65, 205)
WEAK_LANDMARK_COLOR = (255, 170, 40)


def _draw_skeleton(
    canvas: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Draw connected joints in place on an RGB diagnostic image.

    Parameters
    ----------
    canvas : numpy.ndarray
        Writable uint8 RGB canvas.
    points : numpy.ndarray
        Joint positions in BODY_JOINTS order. NaNs mark missing points.
    color : tuple[int, int, int]
        RGB color of the joints and their connecting segments.
    thickness : int
        Line width in pixels.
    """
    shoulder_center = points[list(SHOULDERS)].mean(axis=0)
    hip_center = points[list(HIPS)].mean(axis=0)
    points = np.vstack((points, shoulder_center, hip_center))

    _draw_landmark_graph(canvas, points, BODY_CONNECTIONS, color=color, thickness=thickness)


def _draw_landmark_graph(
    canvas: np.ndarray,
    points: np.ndarray,
    connections: tuple[tuple[int, int], ...],
    *,
    color: tuple[int, int, int],
    thickness: int,
    strong: np.ndarray | None = None,
) -> None:
    """
    Draw finite points and their connecting segments in crop-local pixels.

    Canvas edges clip segments and markers. When supplied, ``strong`` marks
    points that meet the detector's visibility and presence thresholds.
    Other finite points get an orange marker.
    """
    finite = np.isfinite(points).all(axis=1)
    pixels = np.zeros_like(points, dtype=np.int32)
    pixels[finite] = np.rint(points[finite]).clip(-1e8, 1e8).astype(np.int32)
    height, width = canvas.shape[:2]
    rect = (0, 0, width, height)

    for start, end in connections:
        if not (finite[start] and finite[end]):
            continue

        visible, first, second = cv2.clipLine(
            rect,
            tuple(pixels[start]),
            tuple(pixels[end]),
        )

        if visible:
            cv2.line(
                canvas, first, second, color, thickness, cv2.LINE_AA,
            )

    for index in np.flatnonzero(finite):
        x, y = pixels[index]

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                canvas,
                (int(x), int(y)),
                thickness + 2,
                WEAK_LANDMARK_COLOR if strong is not None and not strong[index] else color,
                -1,
                cv2.LINE_AA,
            )


def _draw_pose_legend(
    canvas: np.ndarray,
    legend: list[tuple[str, tuple[int, int, int]]],
    thickness: int,
    *,
    bottom: bool = False,
) -> None:
    """Draw each legend label once over a dark background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, thickness / 5)
    text_sizes = [cv2.getTextSize(label, font, font_scale, 1) for label, _ in legend]
    width = max(size[0] for size, _ in text_sizes)
    height = max(size[1] for size, _ in text_sizes)
    baseline = max(baseline for _, baseline in text_sizes)
    line_height = height + baseline + 3
    panel_height = len(legend) * line_height + 6
    top = max(0, canvas.shape[0] - panel_height - 6) if bottom else 6
    cv2.rectangle(
        canvas, (6, top), (width + 18, top + panel_height),
        (30, 33, 38), -1,
    )

    for row, (label, color) in enumerate(legend):
        position = (12, top + 6 + height + row * line_height)
        cv2.putText(
            canvas, label, position, font, font_scale, color, 1, cv2.LINE_AA,
        )


def draw_body_pose_overlay(
    render: Image.Image,
    rig_xy: np.ndarray,
    landmarks: BodyLandmarks | None,
    *,
    scale: float = 1.0,
    offset: tuple[float, float] = (0, 0),
) -> Image.Image:
    """
    Overlay the rendered skeleton and MediaPipe targets in the same frame.

    Parameters
    ----------
    render : PIL.Image.Image
        Rendered RGBA mannequin on the final canvas.
    rig_xy : numpy.ndarray
        Projected MHR body joints in BODY_JOINTS order.
    landmarks : BodyLandmarks or None
        Target observations, when MediaPipe is available.
    scale : float, optional
        Uniform conversion from target pixels to render pixels.
    offset : tuple[float, float], optional
        Translation in render pixels applied after scaling.

    Returns
    -------
    PIL.Image.Image
        RGB diagnostic over the render, with a dark fill behind transparent
        pixels. Cyan identifies MHR and magenta identifies MediaPipe.
    """
    background = Image.new('RGB', render.size, (30, 33, 38))
    background.paste(render, mask=render.getchannel('A'))
    canvas = np.array(background)
    thickness = max(2, round(min(render.size) / 425))

    _draw_skeleton(
        canvas,
        rig_xy * scale + offset,
        color=RIG_COLOR,
        thickness=thickness + 1,
    )
    legend = [('MHR skeleton', RIG_COLOR)]

    if landmarks is not None:
        _draw_skeleton(
            canvas,
            landmarks.xy * scale + offset,
            color=LANDMARK_COLOR,
            thickness=thickness,
        )
        legend.append(('MediaPipe landmarks', LANDMARK_COLOR))

    _draw_pose_legend(canvas, legend, thickness)

    return Image.fromarray(canvas)


def save_pose_crop_debug(
    crops: dict[str, BodyPoseCrop],
    directory: Path,
) -> dict:
    """
    Save MediaPipe landmark overlays and the unannotated inputs for failed detections.

    Parameters
    ----------
    crops : dict[str, BodyPoseCrop]
        Each attempted MediaPipe crop with its observations or error.
    directory : pathlib.Path
        Existing output directory for the debug images.

    Returns
    -------
    dict
        Target-frame bounds and an overlay or detection error for each crop.
        Rotated crops include their four corners in target coordinates.
        Failed detections include the exact image passed to MediaPipe.
    """
    from mediapipe.tasks.python.vision import PoseLandmarksConnections

    connections = tuple(
        (edge.start, edge.end) for edge in PoseLandmarksConnections.POSE_LANDMARKS
    )
    artifacts = {}

    for region, crop in crops.items():
        image = Image.fromarray(crop.rgb)
        artifact = {'bbox_xyxy': list(crop.box)}
        artifacts[region] = artifact

        if crop.target_corners is not None:
            artifact['corners_xy'] = crop.target_corners.tolist()

        if crop.landmarks is None:
            image_path = directory / f'mediapipe_{region}.png'
            image.save(image_path)
            artifact['image'] = str(image_path)
            artifact['error'] = crop.error
            continue

        landmarks = crop.landmarks
        points = landmarks.xy.astype(np.float32)
        points[~landmarks.valid] = np.nan
        canvas = np.array(image)
        thickness = max(1, round(min(image.size) / 425))
        _draw_landmark_graph(
            canvas, points, connections, color=LANDMARK_COLOR,
            thickness=thickness, strong=landmarks.strong,
        )
        legend = [('MediaPipe landmarks', LANDMARK_COLOR)]

        if np.any(landmarks.valid & ~landmarks.strong):
            legend.append(('Lower visibility or presence', WEAK_LANDMARK_COLOR))

        _draw_pose_legend(canvas, legend, thickness, bottom=True)
        overlay_path = directory / f'mediapipe_{region}_overlay.png'
        Image.fromarray(canvas).save(overlay_path)
        artifact['pose_overlay'] = str(overlay_path)

    return artifacts
