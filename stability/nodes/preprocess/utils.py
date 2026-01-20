import cv2
import numpy as np


def round_up(x: int, m: int) -> int:
    return int((x + m - 1) // m * m)


def resize_long_side_rgb(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if long_side is None or long_side <= 0:
        return rgb
    scale = float(long_side) / float(max(h, w))
    if scale == 1.0:
        return rgb
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(rgb, (nw, nh), interpolation=interp)


def fit_to_target_rgb(
    rgb: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    keep_aspect: bool = True,
    pad_to_multiple_of: int = 64,
) -> np.ndarray:
    if pad_to_multiple_of and pad_to_multiple_of > 1:
        out_w = round_up(out_w, pad_to_multiple_of)
        out_h = round_up(out_h, pad_to_multiple_of)

    if not keep_aspect:
        return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    h, w = rgb.shape[:2]
    scale = min(out_w / float(w), out_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas
