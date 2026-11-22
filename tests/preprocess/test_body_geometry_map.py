from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
import torch

from morphalo.dag import DAG
from morphalo.nodes.preprocess import body_geometry_map as node_module
from morphalo.nodes.preprocess.utils import body_pose_fit
from morphalo.nodes.preprocess.utils.body_pose_fit import (
    BODY_JOINTS,
    LIMB_CHAINS,
    ROOT_ANGLE_NAMES,
    TORSO_ANGLE_NAMES,
    BodyLandmarks,
    detect_body_landmarks,
    refine_body_pose,
)
from morphalo.nodes.preprocess.utils.body_render import render_body
from morphalo.nodes.preprocess.utils.body_shape import adjust_body_volumes
from third_party.instanthmr.parameters import BodyEstimate, BodyProportions, pack_body


FIT_PARAMETER_NAMES = [None] * 3 + [
    *ROOT_ANGLE_NAMES,
    *TORSO_ANGLE_NAMES,
    *(name for _, upper, hinge in LIMB_CHAINS for name in (
        f'{upper}_twist', f'{upper}_ry', f'{upper}_rz', hinge,
    )),
    'l_foot_bend',
    'r_foot_bend',
    'l_ball_bend',
    'r_ball_bend',
]


def test_body_pose_fit_preserves_the_clavicle_angles():
    clavicle_angles = {
        f'{side}_clavicle_r{axis}'
        for side in ('l', 'r')
        for axis in ('x', 'y', 'z')
    }

    assert clavicle_angles.isdisjoint(FIT_PARAMETER_NAMES)


def test_world_xy_alignment_preserves_joint_order_and_depth():
    world = torch.full((len(BODY_JOINTS), 3), torch.nan)
    world[[0, 1, 6, 7]] = torch.tensor([
        [-0.22, -0.35, 0.03], [0.22, -0.38, -0.04],
        [-0.15, 0.0, -0.01], [0.15, 0.03, 0.02],
    ])
    world = body_pose_fit._with_torso_centers(world)
    image = 200 * world[:, :2] + 256
    reflected = world * torch.tensor((-1, 1, 1))
    original = reflected.clone()

    aligned, signs, disagreement = body_pose_fit._align_world_xy_to_image(
        reflected, image,
    )

    assert signs == (-1, 1)
    assert disagreement < 0.1
    torch.testing.assert_close(aligned, world, equal_nan=True)
    torch.testing.assert_close(reflected, original, equal_nan=True)


def test_world_depth_scale_recovers_relative_scale_from_perspective_projection():
    world = torch.full((len(BODY_JOINTS), 3), torch.nan)
    world[[0, 1, 4, 5, 6, 7, 8, 9]] = torch.tensor([
        [-0.22, -0.35, 0.16], [0.22, -0.38, -0.16],
        [-0.58, -0.76, -0.50], [0.40, -0.71, 0.55],
        [-0.15, 0.0, -0.12], [0.15, 0.0, 0.12],
        [-0.35, 0.40, 0.38], [0.28, 0.53, -0.52],
    ])
    world = body_pose_fit._with_torso_centers(world)
    original = world.clone()
    focal_length = 800.0
    image_size = (800, 600)
    xy_scale, depth_scale = 0.45, 0.225
    image = focal_length * (
        torch.tensor((0.18, 0.06)) + xy_scale * world[:, :2]
    ) / (1 + depth_scale * world[:, 2:]) + torch.tensor(image_size) / 2
    calibration = {
        'focal_length': focal_length,
        'image_size': image_size,
        'person_size': 480,
    }

    scale = body_pose_fit._estimate_world_depth_scale(world, image, **calibration)

    assert abs(scale - depth_scale / xy_scale) < 0.04
    torch.testing.assert_close(world, original, equal_nan=True)

    world[:, 2] = 0
    assert body_pose_fit._estimate_world_depth_scale(world, image, **calibration) == 1.0


def test_body_pose_fit_reorients_the_pelvis_and_keeps_anatomical_targets(monkeypatch):
    joints = torch.zeros((len(BODY_JOINTS), 3))
    indices = [0, 1, 6, 7]
    joints[indices, :2] = torch.tensor([
        [-30.0, 35.0], [30.0, 35.0],
        [-20.0, -40.0], [20.0, -40.0],
    ])
    yaw_index = FIT_PARAMETER_NAMES.index('root_ry')

    def skeleton(params):
        yaw = params[0, yaw_index]
        x, y, z = joints.unbind(dim=1)
        rotated = torch.stack((yaw.cos() * x, y, -yaw.sin() * x + z), dim=1)

        return torch.cat((joints[:6], rotated[6:])).unsqueeze(0)

    fit_stage = body_pose_fit._fit_pose_stage
    stages = []

    def capture_stage(**kwargs):
        stages.append(kwargs)

        return fit_stage(**kwargs)

    monkeypatch.setattr(body_pose_fit, '_fit_pose_stage', capture_stage)

    runtime = SimpleNamespace(
        rig=SimpleNamespace(
            character_torch=SimpleNamespace(
                model_parameters_to_joint_parameters=lambda params: params,
                joint_parameters_to_skeleton_state=skeleton,
            ),
            get_joint_names=lambda: [name for _, name in BODY_JOINTS],
            get_parameter_names=lambda: FIT_PARAMETER_NAMES,
            get_parameter_limits=lambda: torch.tensor([[-4.0, 4.0]] * 130),
        ),
        tables={
            'scale_mean': torch.zeros(68),
            'scale_comps': torch.zeros(28, 68),
        },
    )
    proportions = BodyProportions(
        torch.zeros(1, 45),
        torch.zeros(1, 6),
        torch.zeros(1, 28),
    )
    camera = torch.tensor([[0.0, 0.0, 4.0]])
    world = torch.full((len(BODY_JOINTS), 3), torch.nan)
    world[indices] = joints[indices] * torch.tensor((0.01, -0.01, -0.01))
    world[8] = torch.tensor((-0.25, 0.65, -0.2))
    xy = (200 * world[:, :2] / (world[:, 2:] + 4) + 256).numpy()
    original_xy = xy.copy()
    original_world = world.numpy().copy()
    landmarks = BodyLandmarks(xy=xy, world_xyz=world.numpy(), person_size=300)
    pose = torch.zeros((1, 130))
    pose[0, yaw_index] = 2.4

    refined, metrics = refine_body_pose(
        runtime=runtime,
        target=BodyEstimate(pose, proportions, camera, 200.0),
        proportions=proportions,
        landmarks=landmarks,
        image_size=(512, 512),
    )

    assert metrics['method'] == 'mediapipe_3d_retarget'
    assert metrics['root_retargeted']
    assert metrics['after_hips_xyz_deg'] < 10
    assert refined.pose[0, yaw_index].cos() > 0
    assert refined.proportions is proportions
    root_stage, spine_stage, width_stage, limb_stage = stages
    assert [FIT_PARAMETER_NAMES[i] for i in root_stage['angle_indices']] == list(ROOT_ANGLE_NAMES)
    assert [FIT_PARAMETER_NAMES[i] for i in spine_stage['angle_indices']] == list(TORSO_ANGLE_NAMES)
    width_pairs, target_widths, _ = width_stage['width_targets']
    assert width_pairs.tolist() == [[0, 1], [6, 7]]
    assert limb_stage['use_depth_order']
    before_widths, _ = width_stage['project'](width_stage['params'], width_stage['camera'])
    after_widths, _ = limb_stage['project'](limb_stage['params'], limb_stage['camera'])
    first, last = width_pairs.unbind(dim=1)
    before_error = ((before_widths[last] - before_widths[first]).norm(dim=1) - target_widths).norm()
    after_error = ((after_widths[last] - after_widths[first]).norm(dim=1) - target_widths).norm()
    assert after_error <= before_error + 1e-5
    root_segments, _ = root_stage['world_targets']
    assert root_segments[0].tolist() == [6, 7]
    positions, pixels, _, _ = limb_stage['joint_targets']
    assert positions.tolist() == [8]
    np.testing.assert_array_equal(pixels[0].numpy(), original_xy[8])
    world_pairs, directions = limb_stage['world_targets']
    assert world_pairs.tolist() == [[6, 8]]
    expected = torch.nn.functional.normalize(world[8] - world[6], dim=0)
    torch.testing.assert_close(directions[0], expected)
    np.testing.assert_array_equal(landmarks.xy, original_xy)
    np.testing.assert_array_equal(landmarks.world_xyz, original_world)


def test_body_pose_fit_uses_arm_depth_order_independent_of_depth_scale():
    joints = torch.zeros((len(BODY_JOINTS), 3))
    joints[[0, 1], :2] = torch.tensor([[-30.0, 35.0], [30.0, 35.0]])
    joints[[6, 7], :2] = torch.tensor([[-20.0, -40.0], [20.0, -40.0]])
    upper_index = FIT_PARAMETER_NAMES.index('l_uparm_ry')
    hinge_index = FIT_PARAMETER_NAMES.index('l_elbow_bend')
    spine_index = FIT_PARAMETER_NAMES.index('spine_twist0')

    def skeleton(params):
        upper = params[0, upper_index]
        lower = upper + params[0, hinge_index]
        zero = upper.new_zeros(())
        result = joints.clone()
        result[2] = joints[0] + 30 * torch.stack((upper.cos(), zero, -upper.sin()))
        result[4] = result[2] + 25 * torch.stack((lower.cos(), zero, -lower.sin()))

        return result.unsqueeze(0)

    runtime = SimpleNamespace(
        rig=SimpleNamespace(
            character_torch=SimpleNamespace(
                model_parameters_to_joint_parameters=lambda params: params,
                joint_parameters_to_skeleton_state=skeleton,
            ),
            get_joint_names=lambda: [name for _, name in BODY_JOINTS],
            get_parameter_names=lambda: FIT_PARAMETER_NAMES,
            get_parameter_limits=lambda: torch.tensor([[-1.0, 1.0]] * 130),
        ),
        tables={
            'scale_mean': torch.zeros(68),
            'scale_comps': torch.zeros(28, 68),
        },
    )
    proportions = BodyProportions(
        torch.zeros(1, 45),
        torch.zeros(1, 6),
        torch.zeros(1, 28),
    )
    pose = torch.zeros(1, 130)
    pose[0, upper_index] = -0.6
    pose[0, hinge_index] = -0.3
    target = BodyEstimate(pose, proportions, torch.tensor([[0.0, 0.0, 4.0]]), 750.0)
    desired = pose.clone()
    desired[0, upper_index] = 0.6
    desired[0, hinge_index] = 0.4
    indices = [0, 1, 2, 4, 6, 7]
    world = torch.full((len(BODY_JOINTS), 3), torch.nan)
    world[indices] = skeleton(desired)[0, indices] * torch.tensor((0.01, -0.01, -0.01))
    xy = (target.focal_length * world[:, :2] / (world[:, 2:] + 4) + 256).numpy()

    fitted_poses = []

    for depth_scale in (1.0, 3.0):
        observations = world.clone()
        observations[:, 2] *= depth_scale
        refined, diagnostics = refine_body_pose(
            runtime=runtime,
            target=target,
            proportions=proportions,
            landmarks=BodyLandmarks(xy=xy, world_xyz=observations.numpy(), person_size=180),
            image_size=(512, 512),
        )

        assert diagnostics['limbs_retargeted']
        assert diagnostics['after_px'] < diagnostics['before_px']

        if depth_scale == 1.0:
            assert abs(diagnostics['world_depth_scale'] - 1) < 0.01
        else:
            assert 0 < diagnostics['world_depth_scale'] < 0.5

        relative_depth = -skeleton(refined.pose)[0, [0, 2, 4], 2].diff()
        assert (relative_depth > 0).all()
        torch.testing.assert_close(refined.pose[0, spine_index], target.pose[0, spine_index])
        assert refined.proportions is proportions
        fitted_poses.append(refined.pose)

    torch.testing.assert_close(*fitted_poses)


def test_body_pose_fit_uses_body_joints_without_face_or_hand_landmarks():
    # The synthetic rig can move the elbows and adjust ankle flexion.
    initial_joints = torch.tensor([
        [-30.0, 35.0, 0.0],
        [30.0, 35.0, 0.0],
        [-35.0, 0.0, 0.0],
        [35.0, 0.0, 0.0],
        [-50.0, -5.0, 0.0],
        [50.0, -5.0, 0.0],
        [-20.0, -40.0, 0.0],
        [20.0, -40.0, 0.0],
        [-20.0, -80.0, 0.0],
        [20.0, -80.0, 0.0],
        [-20.0, -120.0, 0.0],
        [20.0, -120.0, 0.0],
        [-20.0, -130.0, 0.0],
        [20.0, -130.0, 0.0],
    ])
    elbow_shift = torch.zeros_like(initial_joints)
    elbow_shift[2:4, 0] = 50
    foot_shift = torch.zeros_like(initial_joints)
    foot_shift[12, 0] = 50
    arm_index = FIT_PARAMETER_NAMES.index('l_uparm_twist')
    ankle_index = FIT_PARAMETER_NAMES.index('l_foot_bend')
    toe_index = FIT_PARAMETER_NAMES.index('l_ball_bend')
    other_ankle_index = FIT_PARAMETER_NAMES.index('r_foot_bend')
    character = SimpleNamespace(
        model_parameters_to_joint_parameters=lambda params: params,
        joint_parameters_to_skeleton_state=lambda params: (
            initial_joints + params[0, arm_index] * elbow_shift
            + (params[0, ankle_index] + params[0, other_ankle_index]) * foot_shift
        ).unsqueeze(0),
    )
    rig = SimpleNamespace(
        character_torch=character,
        get_joint_names=lambda: [name for _, name in BODY_JOINTS],
        get_parameter_names=lambda: FIT_PARAMETER_NAMES,
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
        SimpleNamespace(x=np.nan, y=np.nan, visibility=0.0, presence=0.0)
        for _ in range(33)
    ]
    observed = {
        11: (253, 239),
        12: (283, 239),
        13: (260, 256),
        14: (295, 256),
        23: (258, 276),
        24: (278, 276),
        27: (258, 316),
        31: (268, 321),
    }

    for index, (x, y) in observed.items():
        landmarks[index] = SimpleNamespace(
            x=x / 512,
            y=y / 512,
            z=0.0,
            visibility=0.30 if index == 13 else 0.95,
            presence=0.95,
        )

    # Even zero confidence leaves a finite prediction available to the fit.
    landmarks[15] = SimpleNamespace(
        x=0.25,
        y=0.5,
        z=0.0,
        visibility=0.0,
        presence=0.0,
    )
    detector = SimpleNamespace(
        detect=lambda image: SimpleNamespace(pose_landmarks=[landmarks]),
    )
    detections = detect_body_landmarks(
        rgb=np.zeros((512, 512, 3), dtype=np.uint8),
        bbox=(0, 0, 512, 512),
        pose_landmarker=detector,
    )
    np.testing.assert_array_equal(detections.xy[4], [128, 256])
    np.testing.assert_array_equal(detections.xy[2], observed[13])

    refined, metrics = refine_body_pose(
        runtime=runtime,
        target=target,
        proportions=proportions,
        landmarks=detections,
        image_size=(512, 512),
    )

    assert metrics['status'] == 'fitted'
    assert metrics['joints'] == 9
    assert metrics['depth_segments'] == 0
    assert metrics['after_deg'] < metrics['before_deg']
    assert metrics['after_left_foot_deg'] < metrics['before_left_foot_deg']
    assert refined.pose[0, ankle_index] > 0
    torch.testing.assert_close(refined.pose[0, other_ankle_index], target.pose[0, other_ankle_index])
    torch.testing.assert_close(refined.pose[0, toe_index], target.pose[0, toe_index])
    assert metrics['after_pelvis_px'] < metrics['before_pelvis_px']
    assert metrics['after_torso_cm'] == metrics['before_torso_cm']
    assert refined.proportions is proportions
    assert refined.focal_length == target.focal_length
    torch.testing.assert_close(refined.camera_translation[:, 2:], target.camera_translation[:, 2:])
    torch.testing.assert_close(refined.pose[:, 70:], target.pose[:, 70:])

    # Predictions beyond the image edge are kept when MediaPipe reports them.
    landmarks[16] = SimpleNamespace(
        x=1.125,
        y=0.5,
        z=0.0,
        visibility=0.0,
        presence=0.0,
    )
    detections = detect_body_landmarks(
        rgb=np.zeros((512, 512, 3), dtype=np.uint8),
        bbox=(0, 0, 512, 512),
        pose_landmarker=detector,
    )

    np.testing.assert_array_equal(detections.xy[5], [576, 256])


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


def test_body_volume_controls_deform_only_their_respective_regions():
    reference = torch.tensor([
        [8.5, 129.0, 12.0],
        [-8.5, 129.0, 12.0],
        [8.5, 129.0, -12.0],
        [19.0, 94.0, 0.0],
        [-19.0, 94.0, 0.0],
        [0.0, 155.0, 10.0],
    ])
    vertices = reference.unsqueeze(0).clone()
    chest = adjust_body_volumes(vertices, reference, chest_volume=-1, hip_volume=0)
    hips = adjust_body_volumes(vertices, reference, chest_volume=0, hip_volume=1)

    assert (chest[0, :2, 2] < vertices[0, :2, 2]).all()
    torch.testing.assert_close(chest[0, 2:], vertices[0, 2:])
    torch.testing.assert_close(chest[..., :2], vertices[..., :2])
    assert hips[0, 3, 0] > vertices[0, 3, 0]
    assert hips[0, 4, 0] < vertices[0, 4, 0]
    torch.testing.assert_close(hips[0, [0, 1, 2, 5]], vertices[0, [0, 1, 2, 5]])
    torch.testing.assert_close(hips[..., 1:], vertices[..., 1:])
    torch.testing.assert_close(vertices[0], reference)


@pytest.mark.parametrize(
    ('params', 'chest', 'hips'),
    [
        ({}, 0, 0),
        ({'gender': 'm'}, -0.85, -0.45),
        ({'gender': 'masculine'}, -0.85, -0.45),
        ({'gender': 'f'}, 0.65, 0.5),
        ({'gender': 'feminine', 'chest_volume': -1.0, 'hip_volume': 0.25}, -0.35, 0.75),
        ({'chest_volume': 0.5, 'hip_volume': -0.5}, 0.5, -0.5),
    ],
)
def test_body_geometry_map_resolves_volume_presets(params, chest, hips):
    config = node_module._read_cfg(
        {'model': {'mhr_model_pt': 'mhr_model.pt'}, 'params': params},
        'body',
    )

    assert config.chest_volume == pytest.approx(chest)
    assert config.hip_volume == pytest.approx(hips)


@pytest.mark.parametrize(
    ('shape_params', 'deformed'),
    [({}, False), ({'gender': 'f', 'chest_volume': 0.25}, True)],
)
def test_body_geometry_map_averages_all_size_channels_and_keeps_target_pose(
    tmp_path, monkeypatch, shape_params, deformed,
):
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

    def mesh(target, proportions, *, rest_deformation=None):
        torch.testing.assert_close(target.camera_translation, torch.tensor([[4.0, 0, 2]]))
        torch.testing.assert_close(proportions.shape, torch.full((1, 45), 14.0))
        assert target.focal_length == 8
        tables = {'scale_mean': torch.zeros(68), 'scale_comps': torch.ones((28, 68))}
        packed = pack_body(target.pose, proportions, tables)
        torch.testing.assert_close(packed[:, :130], torch.full((1, 130), 4.0))
        torch.testing.assert_close(packed[:, 130:136], torch.full((1, 6), 24.0))
        torch.testing.assert_close(packed[:, 136:], torch.full((1, 68), 34.0 * 28))

        if deformed:
            reference = torch.tensor([[8.5, 129.0, 12.0], [19.0, 94.0, 0.0]])
            vertices = rest_deformation(reference.unsqueeze(0), reference)
            assert vertices[0, 0, 2] > reference[0, 2]
            assert vertices[0, 1, 0] > reference[1, 0]
        else:
            assert rest_deformation is None

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
                'params': {'background_color': 'navy', **shape_params},
            },
        )

    result = node.run(tmp_path, {'default': {'image': target}, 'source': {'images': [first, second]}})
    assert result['source_images'] == [str(first), str(second)]
    with Image.open(result['image']) as image:
        assert image.size == (20, 20)
        assert image.mode == 'RGBA'
        assert image.getpixel((0, 0)) == (0, 0, 128, 255)
