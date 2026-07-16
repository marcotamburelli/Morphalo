import numpy as np
import pytest

from morphalo.nodes.preprocess.utils.mask_selection import (
    mask_bbox_center_x,
    select_image_side_mask_candidate,
)


def test_mask_bbox_center_x_uses_end_exclusive_bbox_center():
    mask = np.zeros((4, 8), dtype=bool)
    mask[1:3, 2:5] = True

    assert mask_bbox_center_x(mask) == 3.5


def test_select_image_side_mask_candidate_uses_image_relative_position():
    anatomical_left = np.zeros((2, 6), dtype=bool)
    anatomical_right = np.zeros((2, 6), dtype=bool)
    anatomical_left[:, 4] = True
    anatomical_right[:, 1] = True

    selected = select_image_side_mask_candidate(
        [
            ('anatomical-left-hand', anatomical_left),
            ('anatomical-right-hand', anatomical_right),
        ],
        image_side='left',
        node_id='node',
        target_name='left-hand',
        error_prefix='TestCrop',
    )

    assert selected.selected_index == 1
    assert selected.selected_name == 'anatomical-right-hand'
    assert selected.candidate_centers_x == {
        'anatomical-right-hand': 1.5,
        'anatomical-left-hand': 4.5,
    }


def test_select_image_side_mask_candidate_rejects_opposite_half_singleton():
    anatomical_left = np.zeros((2, 6), dtype=bool)
    anatomical_right = np.zeros((2, 6), dtype=bool)
    anatomical_left[:, 4] = True

    with pytest.raises(RuntimeError, match='only visible candidate'):
        select_image_side_mask_candidate(
            [
                ('anatomical-left-hand', anatomical_left),
                ('anatomical-right-hand', anatomical_right),
            ],
            image_side='left',
            node_id='node',
            target_name='left-hand',
            error_prefix='TestCrop',
        )
