import numpy as np

from morphalo.nodes.vision.human import (
    HandLandmarksResult,
    hands_bbox_xyxy_from_landmarks,
)


def _hands_with_same_handedness() -> HandLandmarksResult:
    xy = np.full((2, 21, 2), -1, dtype=np.int32)
    xy[0, :3] = np.asarray([
        (0, 20),
        (100, 20),
        (100, 30),
    ])
    xy[1, :3] = np.asarray([
        (70, 20),
        (90, 20),
        (90, 30),
    ])

    z = np.zeros((2, 21), dtype=np.float32)
    world_xyz = np.zeros((2, 21, 3), dtype=np.float32)
    visibility = np.zeros((2, 21), dtype=np.float32)
    valid = xy[:, :, 0] >= 0
    handedness_score = np.asarray([0.9, 0.9], dtype=np.float32)

    return HandLandmarksResult(
        xy=xy,
        z=z,
        world_xyz=world_xyz,
        visibility=visibility,
        valid=valid,
        handedness=['Right', 'Right'],
        handedness_score=handedness_score,
    )


def test_single_left_hand_uses_leftmost_bbox_center_when_labels_tie():
    box = hands_bbox_xyxy_from_landmarks(
        _hands_with_same_handedness(),
        (100, 100, 3),
        which='left',
        expansion=1.0,
    )

    assert box[0] == 0
    assert box[2] == 100


def test_single_right_hand_uses_rightmost_bbox_center_when_labels_tie():
    box = hands_bbox_xyxy_from_landmarks(
        _hands_with_same_handedness(),
        (100, 100, 3),
        which='right',
        expansion=1.0,
    )

    assert box[0] == 70
    assert box[2] == 91
