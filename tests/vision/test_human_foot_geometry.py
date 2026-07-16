import numpy as np

from morphalo.nodes.vision.human import (
    _foot_ankle_contaminated_by_crossed_leg,
    _foot_sam_points_for_side,
    feet_sam_points_from_landmarks,
    foot_sam_regions_from_landmarks,
)


def test_feet_sam_points_exclude_heel_landmark():
    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[27] = (50, 50)  # left ankle
    pose_xy[29] = (10, 10)  # left heel
    pose_xy[31] = (70, 50)  # left foot_index

    points, labels = feet_sam_points_from_landmarks(
        pose_xy,
        (100, 100, 3),
        which='both',
        expansion=1.0,
    )

    assert points == [[50.0, 50.0], [70.0, 50.0]]
    assert labels == [1, 1]


def test_feet_sam_regions_keep_both_feet_separate():
    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[27] = (20, 80)  # left ankle
    pose_xy[31] = (30, 80)  # left foot_index
    pose_xy[28] = (80, 80)  # right ankle
    pose_xy[32] = (70, 80)  # right foot_index

    regions = foot_sam_regions_from_landmarks(
        pose_xy,
        (100, 100, 3),
        which='both',
        expansion=1.0,
        person_bbox=(0, 0, 100, 100),
    )

    assert len(regions) == 2
    assert [region.point_coords for region in regions] == [
        [
            [20.0, 80.0],
            [30.0, 80.0],
            [42.0, 80.0],
        ],
        [
            [80.0, 80.0],
            [70.0, 80.0],
            [57.0, 80.0],
        ],
    ]
    assert [region.point_labels for region in regions] == [
        [1, 1, 0],
        [1, 1, 0],
    ]


def test_feet_sam_region_keeps_leg_probe_separate_when_knee_is_available():
    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[25] = (50, 20)  # left knee
    pose_xy[27] = (50, 60)  # left ankle
    pose_xy[31] = (70, 65)  # left foot_index

    regions = foot_sam_regions_from_landmarks(
        pose_xy,
        (100, 100, 3),
        which='both',
        expansion=1.0,
    )

    assert regions[0].point_coords == [
        [50.0, 60.0],
        [70.0, 65.0],
    ]
    assert regions[0].point_labels == [1, 1]
    assert regions[0].probe_point == [50.0, 51.0]


def test_feet_sam_regions_skip_other_foot_negative_outside_prompt_bbox():
    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[27] = (20, 80)  # left ankle
    pose_xy[31] = (30, 80)  # left foot_index
    pose_xy[28] = (80, 80)  # right ankle
    pose_xy[32] = (70, 80)  # right foot_index

    regions = foot_sam_regions_from_landmarks(
        pose_xy,
        (100, 100, 3),
        which='both',
        expansion=1.0,
    )

    assert regions[0].point_coords == [
        [20.0, 80.0],
        [30.0, 80.0],
    ]
    assert regions[0].point_labels == [1, 1]


def test_feet_sam_regions_accept_additional_positive_points_by_side():
    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[27] = (20, 80)  # left ankle
    pose_xy[31] = (30, 80)  # left foot_index

    regions = foot_sam_regions_from_landmarks(
        pose_xy,
        (100, 100, 3),
        which='both',
        expansion=1.0,
        additional_positive_points_by_side={
            'anatomical-left': [[26.0, 80.0]],
        },
    )

    assert regions[0].point_coords == [
        [20.0, 80.0],
        [30.0, 80.0],
        [26.0, 80.0],
    ]
    assert regions[0].point_labels == [1, 1, 1]


def test_crossed_leg_contamination_uses_foot_index_only():
    pose_xy = np.full((33, 2), -1, dtype=np.int32)
    pose_xy[25] = (60, 20)  # left knee
    pose_xy[27] = (55, 78)  # left ankle, almost on right lower-leg axis
    pose_xy[31] = (50, 95)  # left foot_index
    pose_xy[26] = (40, 20)  # right knee
    pose_xy[28] = (56, 82)  # right ankle
    pose_xy[32] = (60, 95)  # right foot_index

    assert _foot_ankle_contaminated_by_crossed_leg(
        pose_xy,
        side='anatomical-left',
    )

    points, labels = _foot_sam_points_for_side(
        pose_xy,
        side='anatomical-left',
        base_bbox=(45, 75, 53, 100),
        prompt_bbox=(0, 0, 100, 100),
        additional_positive_points=[[50.0, 88.0]],
    )

    assert points == [
        [50.0, 95.0],
        [56.0, 82.0],
        [60.0, 95.0],
    ]
    assert labels == [1, 0, 0]
