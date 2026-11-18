from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import torch

from morphalo.dag import DAG
from morphalo.nodes.preprocess import body_geometry_map as node_module
from morphalo.nodes.preprocess.utils.body_pose_fit import (
    FIT_ANGLE_NAMES,
    refine_body_pose,
)
from morphalo.nodes.preprocess.utils.body_render import render_body
from third_party.instanthmr.parameters import BodyEstimate, BodyProportions, pack_body


def test_body_pose_fit_preserves_the_clavicle_angles():
    clavicle_angles = {
        f'{side}_clavicle_r{axis}'
        for side in ('l', 'r')
        for axis in ('x', 'y', 'z')
    }

    assert clavicle_angles.isdisjoint(FIT_ANGLE_NAMES)


def test_body_pose_fit_uses_body_joints_without_face_or_hand_landmarks():
    # The synthetic rig has one angle that shifts both elbow centers.
    initial_joints = torch.tensor([
        [-30.0, 35.0, 0.0],
        [30.0, 35.0, 0.0],
        [-35.0, 0.0, 0.0],
        [35.0, 0.0, 0.0],
    ])
    elbow_shift = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
    ])
    character = SimpleNamespace(
        model_parameters_to_joint_parameters=lambda params: params,
        joint_parameters_to_skeleton_state=lambda params: (
            initial_joints + params[0, 6] * elbow_shift
        ).unsqueeze(0),
    )
    rig = SimpleNamespace(
        character_torch=character,
        get_joint_names=lambda: ['l_uparm', 'r_uparm', 'l_lowarm', 'r_lowarm'],
        get_parameter_names=lambda: [None] * 6 + list(FIT_ANGLE_NAMES),
        get_parameter_limits=lambda: torch.tensor([[-1.0, 1.0]] * 130),
    )
    runtime = SimpleNamespace(
        rig=rig,
        tables={
            'scale_mean': torch.zeros(68),
            'scale_comps': torch.zeros(28, 68),
        },
    )

    with torch.inference_mode():
        proportions = BodyProportions(
            torch.ones(1, 45),
            torch.zeros(1, 6),
            torch.zeros(1, 28),
        )
        pose = torch.zeros((1, 130))
        pose[:, 70] = 0.2
        target = BodyEstimate(
            pose,
            proportions,
            torch.tensor([[0.0, 0.0, 4.0]]),
            200.0,
        )

    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, visibility=0.0, presence=0.0)
        for _ in range(33)
    ]
    observed = {11: (241, 239), 12: (271, 239), 13: (249, 256), 14: (284, 256)}

    for index, (x, y) in observed.items():
        landmarks[index] = SimpleNamespace(
            x=x / 512,
            y=y / 512,
            z=0.0,
            visibility=0.95,
            presence=0.95,
        )

    detector = SimpleNamespace(
        detect=lambda image: SimpleNamespace(pose_landmarks=[landmarks]),
    )
    refined, metrics = refine_body_pose(
        runtime=runtime,
        target=target,
        proportions=proportions,
        rgb=np.zeros((512, 512, 3), dtype=np.uint8),
        bbox=(0, 0, 512, 512),
        pose_landmarker=detector,
    )

    assert metrics['status'] == 'fitted'
    assert metrics['joints'] == 4
    assert metrics['after_px'] < metrics['before_px']
    assert refined.proportions is proportions
    assert refined.camera_translation is target.camera_translation
    torch.testing.assert_close(refined.pose[:, 70:], target.pose[:, 70:])


def test_body_geometry_map_selects_the_largest_person_detection(tmp_path):
    boxes = MagicMock()
    boxes.xyxy = torch.tensor([[1, 1, 3, 4], [2, 3, 18, 19]])
    boxes.cls = torch.tensor([0, 0])
    boxes.__len__.return_value = 2
    detector = SimpleNamespace(
        predict=lambda *args, **kwargs: [SimpleNamespace(boxes=boxes)],
    )
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)

    bbox = node_module._person_bbox(
        detector,
        rgb,
        'cpu',
        tmp_path / 'target.png',
    )

    assert bbox == (2, 3, 18, 19)


def test_body_render_keeps_the_nearest_surface_and_transparent_background():
    # Two triangles cover the same pixels with opposing normals and depths.
    near = np.array(((-2, -2, 2), (-2, 3, 2), (3, -2, 2)), dtype=np.float32)
    vertices = np.concatenate((near, near * 2))
    faces = np.array(((0, 1, 2), (3, 5, 4)))
    params = dict(width=20, height=20, focal=8, body_color=(200, 200, 200), background_color=None)
    output = np.array(render_body(vertices, faces, **params))
    front_only = np.array(render_body(vertices, faces[:1], **params))
    back_only = np.array(render_body(vertices, faces[1:], **params))
    np.testing.assert_array_equal(output, front_only)
    np.testing.assert_array_equal(output, np.array(render_body(vertices, faces[::-1], **params)))
    assert output[10, 10, 0] > back_only[10, 10, 0]
    assert tuple(output[0, 0]) == (0, 0, 0, 0)
    assert output[10, 10, 3] == 255


def test_body_geometry_map_averages_all_size_channels_and_keeps_target_pose(tmp_path, monkeypatch):
    def estimate(rgb, bbox):
        value = float(rgb[0, 0, 0])
        return BodyEstimate(
            pose=torch.full((1, 130), value),
            proportions=BodyProportions(
                shape=torch.full((1, 45), value),
                lengths=torch.full((1, 6), value + 10),
                scale=torch.full((1, 28), value + 20),
            ),
            camera_translation=torch.tensor([[value, 0, 2]]),
            focal_length=value * 2,
        )

    def mesh(target, proportions):
        torch.testing.assert_close(target.camera_translation, torch.tensor([[4.0, 0, 2]]))
        torch.testing.assert_close(proportions.shape, torch.full((1, 45), 14.0))
        assert target.focal_length == 8
        tables = {'scale_mean': torch.zeros(68), 'scale_comps': torch.ones((28, 68))}
        packed = pack_body(target.pose, proportions, tables)
        torch.testing.assert_close(packed[:, :130], torch.full((1, 130), 4.0))
        torch.testing.assert_close(packed[:, 130:136], torch.full((1, 6), 24.0))
        torch.testing.assert_close(packed[:, 136:], torch.full((1, 68), 34.0 * 28))
        return np.array(((-2, -2, 2), (-2, 3, 2), (3, -2, 2)), dtype=np.float32)

    runtime = SimpleNamespace(estimate=estimate, mesh=mesh, faces=np.array(((0, 1, 2),)))
    monkeypatch.setattr(node_module, 'get_instanthmr_model', lambda **kwargs: runtime)
    monkeypatch.setattr(node_module, 'get_yolo', lambda **kwargs: None)
    monkeypatch.setattr(node_module, '_person_bbox', lambda *args: (0, 0, 20, 20))
    target, first, second = (tmp_path / f'{value}.png' for value in (4, 12, 16))
    for value, path in zip((4, 12, 16), (target, first, second)):
        Image.new('RGB', (20, 20), (value, 0, 0)).save(path)
    asset = tmp_path / 'mock-weight'
    asset.touch()
    with DAG(name='test_body_geometry_map', out_dir=tmp_path):
        node = node_module.BodyGeometryMap(
            name='body',
            spec={
                'model': {'checkpoint': str(asset), 'mhr_model_pt': str(asset)},
                'params': {'background_color': 'navy'},
            },
        )

    result = node.run(tmp_path, {'default': {'image': target}, 'source': {'images': [first, second]}})
    assert result['source_images'] == [str(first), str(second)]
    with Image.open(result['image']) as image:
        assert image.size == (20, 20)
        assert image.mode == 'RGBA'
        assert image.getpixel((0, 0)) == (0, 0, 128, 255)
