from typing import Tuple

import cv2
import numpy as np


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
