"""Draw a shared body skeleton for predicted and observed joint positions."""

import cv2
import numpy as np
from PIL import Image

from morphalo.nodes.preprocess.utils.body_pose_fit import (
    FIT_SEGMENTS,
    HIPS,
    SHOULDERS,
    BodyLandmarks,
)


# Include the sides of the torso for context around the fitted center line.
BODY_CONNECTIONS = (*FIT_SEGMENTS, (0, 6), (1, 7))
RIG_COLOR = (0, 220, 255)
LANDMARK_COLOR = (255, 65, 205)


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
    finite = np.isfinite(points).all(axis=1)
    pixels = np.zeros_like(points, dtype=np.int32)
    pixels[finite] = np.rint(points[finite]).clip(-1e8, 1e8).astype(np.int32)
    height, width = canvas.shape[:2]
    rect = (0, 0, width, height)

    for start, end in BODY_CONNECTIONS:
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
                color,
                -1,
                cv2.LINE_AA,
            )


def draw_body_pose_overlay(
    render: Image.Image,
    rig_xy: np.ndarray,
    landmarks: BodyLandmarks | None,
) -> Image.Image:
    """
    Overlay the rendered skeleton and MediaPipe targets in the same frame.

    Parameters
    ----------
    render : PIL.Image.Image
        Rendered RGBA mannequin in target-frame coordinates.
    rig_xy : numpy.ndarray
        Projected MHR body joints in BODY_JOINTS order.
    landmarks : BodyLandmarks or None
        Target observations, when MediaPipe is available.

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
        rig_xy,
        color=RIG_COLOR,
        thickness=thickness + 1,
    )
    legend = [('MHR skeleton', RIG_COLOR)]

    if landmarks is not None:
        _draw_skeleton(
            canvas,
            landmarks.xy,
            color=LANDMARK_COLOR,
            thickness=thickness,
        )
        legend.append(('MediaPipe landmarks', LANDMARK_COLOR))

    font_scale = max(0.4, thickness / 5)
    line_height = 14 + 5 * thickness

    for row, (label, color) in enumerate(legend, start=1):
        position = (12, row * line_height)
        cv2.putText(
            canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, 1, cv2.LINE_AA,
        )

    return Image.fromarray(canvas)
