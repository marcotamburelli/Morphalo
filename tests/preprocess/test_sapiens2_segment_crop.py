import numpy as np

from morphalo.nodes.preprocess.sapiens2_segment_crop import (
    _parse_target_specs,
    _resolve_parsed_target,
    _segment_part_masks,
    SAPIENS2_CLASSES,
)
from morphalo.nodes.preprocess.utils import cleanup_shape_mask_by_parts


def test_sapiens2_label_ids_match_official_taxonomy():
    assert SAPIENS2_CLASSES['background'] == 0
    assert SAPIENS2_CLASSES['face-neck'] == 3
    assert SAPIENS2_CLASSES['left-hand'] == 6
    assert SAPIENS2_CLASSES['right-hand'] == 15
    assert SAPIENS2_CLASSES['tongue'] == 28


def test_target_array_parses_specs_in_order():
    parsed = _parse_target_specs(
        ['head', 'left-hand', 'hands'],
        node_id='sapiens',
    )

    assert [item.name for item in parsed.items] == [
        'head',
        'left-hand',
        'hands',
    ]
    assert parsed.items[1].spec.side_mode == 'image-relative'
    assert [
        candidate.name for candidate in parsed.items[1].spec.candidates
    ] == ['anatomical-left-hand', 'anatomical-right-hand']


def test_parsed_target_resolves_label_ids_in_order():
    parsed = _parse_target_specs(
        ['head', 'left-hand', 'hands'],
        node_id='sapiens',
    )
    segments = np.asarray([[0, 15, 0, 0, 6]], dtype=np.int64)

    resolved = _resolve_parsed_target(
        parsed,
        node_id='sapiens',
        segments=segments,
    )

    assert resolved.labels == [
        'face-neck',
        'hair',
        'lower-lip',
        'upper-lip',
        'lower-teeth',
        'upper-teeth',
        'tongue',
        'right-hand',
        'left-hand',
    ]
    assert resolved.label_ids == [3, 4, 24, 25, 26, 27, 28, 15, 6]


def test_person_alias_excludes_background():
    parsed = _parse_target_specs(
        'person',
        node_id='sapiens',
    )
    segments = np.asarray([[0, 1, 28]], dtype=np.int64)
    resolved = _resolve_parsed_target(
        parsed,
        node_id='sapiens',
        segments=segments,
    )

    assert 'background' not in resolved.labels
    assert 0 not in resolved.label_ids


def test_side_specific_arm_and_leg_aliases_parse_candidate_pairs():
    parsed = _parse_target_specs(
        ['left-arm', 'right-arm', 'left-leg', 'right-leg'],
        node_id='sapiens',
    )

    assert [item.name for item in parsed.items] == [
        'left-arm',
        'right-arm',
        'left-leg',
        'right-leg',
    ]
    assert [
        [candidate.name for candidate in item.spec.candidates]
        for item in parsed.items
    ] == [
        ['anatomical-left-arm', 'anatomical-right-arm'],
        ['anatomical-left-arm', 'anatomical-right-arm'],
        ['anatomical-left-leg', 'anatomical-right-leg'],
        ['anatomical-left-leg', 'anatomical-right-leg'],
    ]


def test_image_relative_target_selects_candidate_by_image_position():
    segments = np.asarray([[0, 15, 0, 0, 6]], dtype=np.int64)

    left = _resolve_parsed_target(
        _parse_target_specs('left-hand', node_id='sapiens'),
        node_id='sapiens',
        segments=segments,
    )
    right = _resolve_parsed_target(
        _parse_target_specs('right-hand', node_id='sapiens'),
        node_id='sapiens',
        segments=segments,
    )
    anatomical = _resolve_parsed_target(
        _parse_target_specs('anatomical-left-hand', node_id='sapiens'),
        node_id='sapiens',
        segments=segments,
    )

    assert left.labels == ['right-hand']
    assert left.label_ids == [15]
    assert left.selected_candidates == ['anatomical-right-hand']
    assert left.side_resolutions[0]['selected_candidate'] == (
        'anatomical-right-hand'
    )

    assert right.labels == ['left-hand']
    assert right.label_ids == [6]
    assert right.selected_candidates == ['anatomical-left-hand']

    assert anatomical.labels == ['left-hand']
    assert anatomical.label_ids == [6]
    assert anatomical.selected_candidates == ['anatomical-left-hand']
    assert anatomical.side_resolutions == []


def test_cleanup_by_segment_parts_preserves_one_component_per_label():
    segments = np.asarray([[0, 6, 0, 15, 0]], dtype=np.int64)
    mask = np.isin(
        segments,
        np.asarray(
            [SAPIENS2_CLASSES['left-hand'], SAPIENS2_CLASSES['right-hand']],
            dtype=np.int64,
        ),
    )

    clean = cleanup_shape_mask_by_parts(
        mask,
        _segment_part_masks(
            segments,
            [SAPIENS2_CLASSES['left-hand'], SAPIENS2_CLASSES['right-hand']],
        ),
        min_component_area='biggest',
    )

    assert clean.tolist() == [[False, True, False, True, False]]
