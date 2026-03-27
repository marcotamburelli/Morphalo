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


def postprocess_mask(
    mask: np.ndarray,
    *,
    dilate_radius: int = 0,
    close_radius: int = 0,
    smoothing_radius: int = 0,
) -> np.ndarray:
    """
    Post-process a binary/soft mask to make it suitable for inpainting and diffusion.

    Steps (optional, in order):
      1. Morphological closing (fills holes, connects thin gaps)
      2. Dilation (expand mask outward)
      3. Gaussian smoothing (soft edges)

    Parameters
    ----------
    mask : np.ndarray
        Input mask. Accepted formats:
          - bool mask
          - float mask 0..1
          - uint8 mask 0..255

    dilate_radius : int
        Radius in pixels used to expand the mask outward.

    close_radius : int
        Radius in pixels used for morphological closing (fill holes).

    smoothing_radius : int
        Radius in pixels used for Gaussian blur (edge feathering).

    Returns
    -------
    np.ndarray
        Soft mask uint8 in range 0..255.
    """

    if mask.ndim != 2:
        raise ValueError("Mask must be HxW")

    # normalize to uint8 0..255
    if mask.dtype == bool:
        m = mask.astype(np.uint8) * 255
    elif np.issubdtype(mask.dtype, np.floating):
        m = np.clip(mask * 255, 0, 255).astype(np.uint8)
    else:
        m = mask.astype(np.uint8)

    # --- closing (fills holes) ---
    if close_radius > 0:
        k = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # --- dilation (expand mask) ---
    if dilate_radius > 0:
        k = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)

    # --- smoothing / feather ---
    if smoothing_radius > 0:
        k = 2 * smoothing_radius + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=0, sigmaY=0)

    return np.clip(m, 0, 255).astype(np.uint8)
