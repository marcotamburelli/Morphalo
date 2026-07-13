from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab


@dataclass(frozen=True)
class ChromaticLineSegment:
    """
    Chromatically coherent sub-segment sampled along an image line.

    Parameters
    ----------
    start_xy : tuple[float, float]
        Image-space start point of the retained sub-segment.
    end_xy : tuple[float, float]
        Image-space end point of the retained sub-segment.
    center_xy : tuple[float, float]
        Image-space center point of the retained sub-segment. This is the point
        most useful as a robust positive prompt because it avoids color
        transition boundaries.
    start_sample : int
        Inclusive index of the first line sample in this sub-segment.
    end_sample : int
        Inclusive index of the last line sample in this sub-segment.
    length_px : float
        Geometric length of the sub-segment in pixels.
    mean_lab : tuple[float, float, float]
        Mean CIE Lab color of the sampled pixels in the sub-segment.
    """
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    center_xy: tuple[float, float]
    start_sample: int
    end_sample: int
    length_px: float
    mean_lab: tuple[float, float, float]


def split_segment_by_chromatic_runs(
    img_rgb: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    lab_distance_threshold: float = 14.0,
    min_segment_len_px: float = 3.0,
) -> list[ChromaticLineSegment]:
    """
    Split an image segment into chromatically coherent Lab-color runs.

    The segment is sampled at approximately one-pixel spacing. Consecutive
    samples whose CIE Lab distance exceeds ``lab_distance_threshold`` start a
    new run. Runs shorter than ``min_segment_len_px`` are discarded.

    This is intended for prompt generation along sparse landmark geometry. For
    example, along an ankle -> foot_index line, centers of retained chromatic
    runs can be used as safer positive points than a fixed midpoint: the centers
    tend to avoid boundaries between skin, sandal, shadow and background.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image array with shape ``(H, W, 3)`` and values in ``[0, 255]``.
    start_xy : tuple[float, float]
        Start point ``(x, y)`` in full-image coordinates.
    end_xy : tuple[float, float]
        End point ``(x, y)`` in full-image coordinates.
    lab_distance_threshold : float, default=14.0
        CIE Lab distance above which two adjacent line samples are considered a
        color boundary.
    min_segment_len_px : float, default=3.0
        Minimum geometric sub-segment length to keep.

    Returns
    -------
    list[ChromaticLineSegment]
        Retained sub-segments in start-to-end order.

    Raises
    ------
    ValueError
        If the image, thresholds, or segment geometry are invalid.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('img_rgb must have shape (H, W, 3).')
    if lab_distance_threshold < 0.0:
        raise ValueError('lab_distance_threshold must be >= 0.')
    if min_segment_len_px < 0.0:
        raise ValueError('min_segment_len_px must be >= 0.')

    h, w = img_rgb.shape[:2]
    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    delta = end - start
    segment_len = float(np.linalg.norm(delta))

    if segment_len < 1.0:
        raise ValueError('segment length must be at least 1 pixel.')

    # Sample roughly once per pixel along the continuous landmark segment. The
    # color is read from the nearest image pixel, but all returned geometry stays
    # in continuous image coordinates so callers can use centers directly.
    sample_count = max(2, int(round(segment_len)) + 1)
    t_values = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    sample_xy = start[None, :] + delta[None, :] * t_values[:, None]

    sample_px = np.rint(sample_xy).astype(np.int32)
    sample_px[:, 0] = np.clip(sample_px[:, 0], 0, w - 1)
    sample_px[:, 1] = np.clip(sample_px[:, 1], 0, h - 1)

    rgb_samples = img_rgb[sample_px[:, 1], sample_px[:, 0], :].astype(np.float32)
    lab_samples = rgb2lab((rgb_samples / 255.0).reshape(1, -1, 3))
    lab_samples = lab_samples.reshape(-1, 3).astype(np.float32)

    # Boundaries are placed between adjacent samples whose perceptual color
    # distance jumps enough to suggest a material/texture transition.
    adjacent_dist = np.linalg.norm(np.diff(lab_samples, axis=0), axis=1)
    boundaries = list(np.where(adjacent_dist > lab_distance_threshold)[0] + 1)
    run_starts = [0] + boundaries
    run_stops = boundaries + [sample_count]

    segments: list[ChromaticLineSegment] = []

    for run_start, run_stop in zip(run_starts, run_stops):
        run_end = run_stop - 1
        run_len = float(t_values[run_end] - t_values[run_start]) * segment_len

        if run_len < min_segment_len_px:
            continue

        t_start = float(t_values[run_start])
        t_end = float(t_values[run_end])
        t_center = 0.5 * (t_start + t_end)

        start_point = start + delta * t_start
        end_point = start + delta * t_end
        center_point = start + delta * t_center
        mean_lab = np.mean(lab_samples[run_start:run_stop], axis=0)

        segments.append(ChromaticLineSegment(
            start_xy=(float(start_point[0]), float(start_point[1])),
            end_xy=(float(end_point[0]), float(end_point[1])),
            center_xy=(float(center_point[0]), float(center_point[1])),
            start_sample=int(run_start),
            end_sample=int(run_end),
            length_px=run_len,
            mean_lab=(
                float(mean_lab[0]),
                float(mean_lab[1]),
                float(mean_lab[2]),
            ),
        ))

    return segments
