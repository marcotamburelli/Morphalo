import numpy as np

from morphalo.nodes.preprocess.fashn_segment_crop import _resolve_target_labels
from morphalo.nodes.preprocess.fashn_segment_crop import _segment_part_masks
from morphalo.nodes.preprocess.utils.mask_ops import cleanup_shape_mask_by_parts


def test_target_array_expands_aliases_and_keeps_first_seen_order():
    labels = _resolve_target_labels(
        ['head', 'hair', 'glasses'],
        node_id='human',
    )

    assert labels == ['face', 'hair', 'glasses']


def test_target_tuple_expands_aliases_and_keeps_first_seen_order():
    labels = _resolve_target_labels(
        ('head', 'hair', 'glasses'),
        node_id='human',
    )

    assert labels == ['face', 'hair', 'glasses']


def test_cleanup_by_segment_parts_preserves_one_component_per_label():
    segments = [[0, 1, 0, 2, 0]]

    segments = np.asarray(segments, dtype=np.int64)
    mask = np.isin(segments, np.asarray([1, 2], dtype=np.int64))

    clean = cleanup_shape_mask_by_parts(
        mask,
        _segment_part_masks(segments, [1, 2]),
        min_component_area='biggest',
    )

    assert clean.tolist() == [[False, True, False, True, False]]
