import numpy as np

from morphalo.nodes.preprocess.face_crop import (_read_cfg,
                                                 _semantic_face_mask)
from morphalo.nodes.preprocess.utils.sapiens2_seg import SAPIENS2_CLASSES
from morphalo.nodes.vision.face_region import (MEDIAPIPE_JAWLINE,
                                               face_side_of_jaw_mask)


def test_face_crop_accepts_anatomical_feature_targets():
    cfg = _read_cfg(
        {
            'model': {
                'face_landmarker_task': '/tmp/face.task',
                'pose_landmarker_task': '/tmp/pose.task',
            },
            'params': {
                'target': 'anatomical-left-eye',
            },
        },
        node_id='face',
    )

    assert cfg.target == 'anatomical-left-eye'
    assert cfg.segment_model == 'facebook/sapiens2-seg-0.4b'


def test_face_crop_accepts_face_neck_target():
    cfg = _read_cfg(
        {
            'model': {
                'face_landmarker_task': '/tmp/face.task',
                'pose_landmarker_task': '/tmp/pose.task',
            },
            'params': {'target': 'face-neck'},
        },
        node_id='face',
    )

    assert cfg.target == 'face-neck'
    assert cfg.chin_margin == 0.03


def test_face_side_of_jaw_mask_preserves_configured_chin_margin():
    landmarks = np.zeros((478, 2), dtype=np.int32)
    landmarks[1] = (50, 35)
    landmarks[10] = (50, 20)
    for index, x in zip(
        MEDIAPIPE_JAWLINE,
        np.linspace(25, 75, len(MEDIAPIPE_JAWLINE)),
    ):
        landmarks[index] = (round(x), 65)
    landmarks[152] = (50, 70)
    support = np.zeros((100, 100), dtype=bool)
    support[5:95, 5:95] = True

    exact = face_side_of_jaw_mask(
        landmarks,
        (100, 100, 3),
        support_mask=support,
        margin_ratio=0.0,
    )
    padded = face_side_of_jaw_mask(
        landmarks,
        (100, 100, 3),
        support_mask=support,
        margin_ratio=0.1,
    )

    assert exact[35, 50]
    assert exact[35, 10]
    assert exact[35, 90]
    assert not exact[73, 50]
    assert padded[73, 50]
    assert not padded[90, 50]


def test_semantic_face_mask_includes_mouth_for_both_targets():
    segments = np.zeros((80, 100), dtype=np.int64)
    segments[15:70, 20:45] = SAPIENS2_CLASSES['face-neck']
    segments[10:35, 70:90] = SAPIENS2_CLASSES['face-neck']
    segments[40:43, 28:37] = SAPIENS2_CLASSES['upper-lip']
    jaw_keep = np.zeros_like(segments, dtype=bool)
    jaw_keep[:55, :] = True

    face_neck_mask, face_neck_parts, labels = _semantic_face_mask(
        segments,
        face_bbox=(18, 12, 48, 58),
        jaw_keep_mask=None,
    )
    face_mask, face_parts, _ = _semantic_face_mask(
        segments,
        face_bbox=(18, 12, 48, 58),
        jaw_keep_mask=jaw_keep,
    )

    assert face_neck_mask[41, 32]
    assert face_neck_mask[65, 30]
    assert len(face_neck_parts) == 2

    assert face_mask[25, 30]
    assert face_mask[41, 32]
    assert not face_mask[65, 30]
    assert face_mask[20, 75]
    assert len(face_parts) == 2
    assert labels == [
        'face-neck',
        'lower-lip',
        'upper-lip',
        'lower-teeth',
        'upper-teeth',
        'tongue',
    ]
