import numpy as np
import pytest

from morphalo.nodes.preprocess.utils.mask_selection import (
    filter_mask_components_by_constraint,
    label_values_intersecting_mask,
    mask_bbox_center_x,
    masks_for_label_values,
    select_mask_components_intersecting_mask,
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


def test_select_mask_components_intersecting_mask_uses_spatial_components():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:4, 1:4] = True
    mask[1:4, 4:6] = True
    mask[4, 6] = True
    selector = np.zeros_like(mask)
    selector[2, 2] = True

    components = select_mask_components_intersecting_mask(
        mask,
        selector_mask=selector,
    )

    assert np.array_equal(components, mask)


def test_label_values_intersecting_mask_returns_touched_values():
    mask = np.zeros((5, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[1:3, 5:7] = True
    mask[3:5, 5:7] = True
    label_map = np.zeros_like(mask, dtype=np.int64)
    label_map[1:3, 1:3] = 8
    label_map[1:3, 5:7] = 8
    label_map[3:5, 5:7] = 13
    selector = np.zeros_like(mask)
    selector[1, 1] = True

    selected = label_values_intersecting_mask(
        mask,
        label_map=label_map,
        selector_mask=selector,
    )

    assert selected == frozenset({8})


def test_masks_for_label_values_keeps_labels_separate():
    mask = np.zeros((5, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[1:3, 5:7] = True
    mask[3:5, 5:7] = True
    label_map = np.zeros_like(mask, dtype=np.int64)
    label_map[1:3, 1:3] = 8
    label_map[1:3, 5:7] = 8
    label_map[3:5, 5:7] = 13

    label_masks = masks_for_label_values(
        mask,
        label_map=label_map,
        label_values={8, 13},
    )

    assert len(label_masks) == 2
    assert np.array_equal(label_masks[0], mask & (label_map == 8))
    assert np.array_equal(label_masks[1], mask & (label_map == 13))


def test_filter_mask_components_by_constraint_keeps_intersecting_components():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:4, 1:4] = True
    mask[1:4, 5:7] = True
    constraint = np.zeros_like(mask)
    constraint[2, 2] = True

    filtered = filter_mask_components_by_constraint(
        mask,
        constraint_mask=constraint,
    )

    assert np.any(filtered[1:4, 1:4])
    assert not np.any(filtered[1:4, 5:7])


def test_filter_mask_components_by_constraint_accepts_multiple_constraints():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:4, 1:4] = True
    mask[1:4, 5:7] = True
    first_constraint = np.zeros_like(mask)
    second_constraint = np.zeros_like(mask)
    second_constraint[2, 5] = True

    filtered = filter_mask_components_by_constraint(
        mask,
        constraint_masks=[first_constraint, second_constraint],
    )

    assert not np.any(filtered[1:4, 1:4])
    assert np.any(filtered[1:4, 5:7])


def test_filter_mask_components_by_constraint_preserves_on_empty_by_default():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:3] = True
    constraint = np.zeros_like(mask)

    filtered = filter_mask_components_by_constraint(
        mask,
        constraint_mask=constraint,
    )

    assert np.array_equal(filtered, mask)
