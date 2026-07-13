import numpy as np

from morphalo.nodes.vision.chromatic_segmentation import (
    split_segment_by_chromatic_runs,
)


def test_split_segment_by_chromatic_runs_filters_short_runs():
    img = np.zeros((1, 9, 3), dtype=np.uint8)
    img[:, 0:3] = (255, 0, 0)
    img[:, 3:7] = (0, 255, 0)
    img[:, 7:9] = (0, 0, 255)

    segments = split_segment_by_chromatic_runs(
        img,
        (0.0, 0.0),
        (8.0, 0.0),
        lab_distance_threshold=20.0,
        min_segment_len_px=3.0,
    )

    assert len(segments) == 1
    assert segments[0].start_xy == (3.0, 0.0)
    assert segments[0].end_xy == (6.0, 0.0)
    assert segments[0].center_xy == (4.5, 0.0)
    assert segments[0].start_sample == 3
    assert segments[0].end_sample == 6
    assert segments[0].length_px == 3.0


def test_split_segment_by_chromatic_runs_keeps_uniform_line():
    img = np.zeros((1, 6, 3), dtype=np.uint8)
    img[:] = (120, 80, 40)

    segments = split_segment_by_chromatic_runs(
        img,
        (0.0, 0.0),
        (5.0, 0.0),
        lab_distance_threshold=10.0,
        min_segment_len_px=3.0,
    )

    assert len(segments) == 1
    assert segments[0].start_xy == (0.0, 0.0)
    assert segments[0].end_xy == (5.0, 0.0)
    assert segments[0].center_xy == (2.5, 0.0)
    assert segments[0].length_px == 5.0
