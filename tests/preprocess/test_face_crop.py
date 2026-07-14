from morphalo.nodes.preprocess.face_crop import _read_cfg


def test_face_crop_accepts_anatomical_feature_targets_without_sam():
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
    assert cfg.sam_model is None
