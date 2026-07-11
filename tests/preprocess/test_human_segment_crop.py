from morphalo.nodes.preprocess.human_segment_crop import _resolve_target_labels


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
