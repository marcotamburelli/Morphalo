import numpy as np
import pytest
from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.preprocess import any_crop as any_crop_mod
from morphalo.nodes.preprocess.any_crop import (
    AnyCrop,
    GroundedBox,
    _clean_sam_mask_for_bbox,
    _read_any_crop_cfg,
    _select_grounded_boxes,
)


def test_any_crop_config_accepts_all_selection():
    cfg = _read_any_crop_cfg(
        {'params': {'select': 'all'}},
        node_id='multi',
    )

    assert cfg.select == 'all'


@pytest.mark.parametrize('select', ['best', 'largest', 'center'])
def test_single_selection_strategies_still_return_one_box(select):
    boxes = [
        GroundedBox((0, 0, 2, 2), 0.9, 'best'),
        GroundedBox((3, 3, 9, 9), 0.5, 'largest'),
        GroundedBox((4, 4, 6, 6), 0.4, 'center'),
    ]

    selected = _select_grounded_boxes(
        boxes,
        image_shape=(10, 10, 3),
        select=select,
    )

    assert len(selected) == 1
    assert selected[0].label == select


def test_all_selection_applies_class_agnostic_nms():
    boxes = [
        GroundedBox((0, 0, 10, 10), 0.9, 'eyebrow'),
        GroundedBox((0, 0, 10, 10), 0.8, 'left eyebrow'),
        GroundedBox((20, 0, 30, 10), 0.7, 'right eyebrow'),
    ]

    selected = _select_grounded_boxes(
        boxes,
        image_shape=(40, 40, 3),
        select='all',
    )

    assert [box.label for box in selected] == ['eyebrow', 'right eyebrow']


def test_all_cleanup_preserves_disconnected_components_inside_each_box():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[4, 4] = True
    mask[7, 7] = True

    cleaned = _clean_sam_mask_for_bbox(
        mask,
        bbox=(0, 0, 6, 6),
        keep_all_components=True,
    )

    assert cleaned[1, 1]
    assert cleaned[4, 4]
    assert not cleaned[7, 7]
    assert int(cleaned.sum()) == 2


def test_all_run_uses_union_bbox_and_records_selected_detections(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / 'source.png'
    Image.new('RGB', (12, 8), color=(20, 30, 40)).save(src)

    boxes = [
        GroundedBox((1, 1, 5, 5), 0.9, 'first'),
        GroundedBox((1, 1, 5, 5), 0.8, 'duplicate'),
        GroundedBox((7, 2, 11, 6), 0.7, 'second'),
    ]

    monkeypatch.setattr(
        any_crop_mod,
        'get_grounding_dino',
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        any_crop_mod,
        '_predict_grounding_dino_boxes',
        lambda **kwargs: boxes,
    )
    monkeypatch.setattr(
        any_crop_mod,
        'get_sam',
        lambda **kwargs: (object(), object()),
    )

    def predict_mask(*, img_rgb, bbox, **kwargs):
        mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = True
        return np.asarray([mask]), np.asarray([1.0])

    monkeypatch.setattr(any_crop_mod, 'predict_sam_mask', predict_mask)

    with DAG('test', out_dir=tmp_path):
        node = AnyCrop(
            name='multi',
            path=src,
            spec={
                'prompt': 'first. second.',
                'model': {'device': 'cpu'},
                'params': {
                    'mode': 'default',
                    'crop_mode': 'bbox',
                    'box_margin': 0,
                    'select': 'all',
                },
            },
        )

    out = node.run(tmp_path)

    assert Image.open(out['image']).size == (10, 5)
    assert out['crop']['bbox_xyxy'] == [1, 1, 11, 6]
    assert out['detections']['candidate_count'] == 3
    assert out['detections']['selected_count'] == 2
    assert [
        candidate['idx']
        for candidate in out['detections']['candidates']
        if candidate['selected']
    ] == [0, 2]
    assert [
        segment['label'] for segment in out['segments']
    ] == ['first', 'second']
    assert [
        segment['candidate_idx'] for segment in out['segments']
    ] == [0, 2]
    assert [
        segment['sam_bbox_xyxy'] for segment in out['segments']
    ] == [[1, 1, 5, 5], [7, 2, 11, 6]]
