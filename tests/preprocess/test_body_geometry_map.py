from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
import torch

from morphalo.dag import DAG
from morphalo.nodes.preprocess import body_geometry_map as node_module
from morphalo.nodes.preprocess.utils import body_collision
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
from morphalo.nodes.preprocess.utils.body_pose_debug import (
    LANDMARK_COLOR,
    RIG_COLOR,
    draw_body_pose_overlay,
    save_pose_crop_debug,
)
from morphalo.nodes.preprocess.utils.body_render import fit_body_canvas, render_body
from morphalo.nodes.preprocess.utils.body_shape import BodyAxes, adjust_body_volumes
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


@pytest.mark.parametrize('upper_detection', ['complete', 'partial', 'failed'])
def test_split_pose_detection_merges_local_joints_and_preserves_full_frame_fallback(
    monkeypatch, tmp_path, upper_detection,
):
    calls = []
    indices = [index for index, _ in BODY_JOINTS[:-2]]
    full_pixels = np.full((len(BODY_JOINTS), 2), 40)
    full_pixels[6:8, 1] = [58, 62]
    full_world = np.zeros((len(BODY_JOINTS), 3), dtype=np.float32)
    upper_world = full_world.copy()
    upper_world[1] = [0.25, 0.01, 0.03]
    lower_world = full_world.copy()
    lower_world[7:9] = [[0.23, 0.03, -0.08], [0.02, 0.15, -0.1]]

    def detect(crop, detector, *, require_stable_pose, keep_out_of_frame):
        calls.append((crop.shape[0], keep_out_of_frame))
        local_xy = np.full((33, 2), 12)
        world = np.zeros((33, 3), dtype=np.float32)
        valid = np.ones(33, dtype=bool)

        if len(calls) == 1:
            local_xy[indices] = full_pixels[:-2]
            world[indices] = full_world[:-2]
            local_xy[[17, 19, 18, 20]] = 40
        elif len(calls) == 2:
            if upper_detection == 'failed':
                raise RuntimeError('No upper pose')

            local_xy[indices[:2]] = [[8, 10], [24, 16]]
            world[indices] = upper_world[:-2]
            local_xy[[17, 19]] = [[16, 25], [24, 29]]
            world[[17, 19]] = [[0.10, 0.02, 0.04], [0.20, 0.04, 0.08]]
            valid[18] = False
            valid[indices[4]] = False
            valid[indices[1]] = upper_detection == 'complete'
        else:
            local_xy[indices[6:8]] = [[9, 10], [25, 18]]
            world[indices] = lower_world[:-2]
            valid[indices[8]] = False

        return SimpleNamespace(xy=local_xy, world_xyz=world, valid=valid, strong=valid)

    monkeypatch.setattr(body_pose_fit, 'mp_pose_landmarks_full', detect)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    crop_debug = {}
    detection = detect_body_landmarks(
        rgb=rgb,
        bbox=(0, 0, 100, 100),
        pose_landmarker=object(),
        split_pose_detection=True,
        crop_debug=crop_debug,
    )

    assert calls == [(100, True), (60, False), (47, False)]
    expected_shoulders = {
        'complete': [[8, 37], [24, 43]],
        'partial': [[8, 10], [40, 40]],
        'failed': [[40, 40], [40, 40]],
    }
    np.testing.assert_array_equal(detection.xy[:2], expected_shoulders[upper_detection])
    np.testing.assert_array_equal(detection.xy[4], full_pixels[4])
    np.testing.assert_array_equal(detection.xy[5], [40, 40] if upper_detection == 'failed' else [12, 12])
    np.testing.assert_array_equal(detection.xy[6:8], [[9, 56], [25, 64]])
    np.testing.assert_array_equal(detection.xy[8], full_pixels[8])
    np.testing.assert_array_equal(detection.xy[10], [12, 65])
    expected_hand = [40, 40] if upper_detection == 'failed' else [20, 27]
    np.testing.assert_array_equal(detection.xy[14], expected_hand)
    np.testing.assert_array_equal(detection.xy[15], [40, 40])
    np.testing.assert_array_equal(detection.full_frame_xy, full_pixels)
    np.testing.assert_array_equal(detection.world_xyz, full_world)

    vectors = detection.split_world_vectors
    np.testing.assert_array_equal(vectors[1], lower_world[7] - lower_world[6])
    assert np.isnan(vectors[2]).all()
    assert np.isnan(vectors[body_pose_fit.FIT_SEGMENTS.index((6, 8))]).all()
    assert np.isnan(vectors[body_pose_fit.FIT_SEGMENTS.index((4, 14))]).all()

    artifacts = save_pose_crop_debug(crop_debug, tmp_path)
    assert set(artifacts) == {'full', 'upper', 'lower'}
    assert artifacts['lower']['bbox_xyxy'] == [0, 53, 100, 100]
    assert 'image' not in artifacts['lower']
    assert not (tmp_path / 'mediapipe_lower.png').exists()
    assert 'image' not in artifacts['full']
    assert not (tmp_path / 'mediapipe_full.png').exists()
    with Image.open(artifacts['full']['pose_overlay']) as overlay:
        assert overlay.size == (100, 100)
    with Image.open(artifacts['lower']['pose_overlay']) as overlay:
        assert overlay.getpixel((9, 10)) == LANDMARK_COLOR

    if upper_detection == 'failed':
        assert artifacts['upper']['error'] == 'No upper pose'
        assert 'pose_overlay' not in artifacts['upper']
        with Image.open(artifacts['upper']['image']) as original:
            np.testing.assert_array_equal(np.array(original), rgb[:60])
    else:
        assert 'image' not in artifacts['upper']
        assert not (tmp_path / 'mediapipe_upper.png').exists()

    if upper_detection == 'complete':
        np.testing.assert_array_equal(vectors[0], upper_world[1] - upper_world[0])
    else:
        assert np.isnan(vectors[0]).all()


def test_split_pose_detection_is_an_opt_in_boolean():
    spec = {'model': {'mhr_model_pt': 'rig.pt'}}

    assert node_module._read_cfg(spec, 'body').split_pose_detection is False

    spec['params'] = {'split_pose_detection': True}
    assert node_module._read_cfg(spec, 'body').split_pose_detection is True

    spec['params']['split_pose_detection'] = 'true'
    with pytest.raises(ValueError, match='split_pose_detection'):
        node_module._read_cfg(spec, 'body')


@pytest.mark.parametrize(
    ('axis', 'rotation'),
    [
        ((19, 20), None),
        ((20, 20), [[0, -1], [1, 0]]),
        ((-20, -20), [[0, 1], [-1, 0]]),
        ((19, -20), [[-1, 0], [0, -1]]),
    ],
)
def test_split_pose_cut_switches_at_45_degrees(axis, rotation):
    points = np.full((len(BODY_JOINTS), 2), np.nan)
    points[[0, 1]] = [[30, 15], [50, 15]]
    points[[6, 7]] = points[[0, 1]] + axis
    transform = body_pose_fit._torso_crop_transform((0, 0, 100, 60), points)

    if rotation is None:
        assert transform is None
    else:
        np.testing.assert_array_equal(transform[0][:, :2], rotation)


def test_split_pose_detection_rotates_crops_and_maps_observations_back(monkeypatch, tmp_path):
    inputs = []

    def detect(crop, detector, *, require_stable_pose, keep_out_of_frame):
        inputs.append(crop.copy())
        xy = np.full((33, 2), -1)
        world = np.full((33, 3), np.nan, dtype=np.float32)

        if keep_out_of_frame:
            xy[[11, 12, 23, 24]] = [[20, 25], [20, 35], [66, 25], [72, 35]]
        elif len(inputs) == 3:
            xy[[23, 24, 27]] = [[30, 10], [20, 16], [7, 8]]
            world[[23, 24]] = [[0, 0, 0], [0.1, 0.2, 0.3]]

        valid = (xy >= 0).all(axis=1)
        return SimpleNamespace(xy=xy, world_xyz=world, valid=valid, strong=valid)

    monkeypatch.setattr(body_pose_fit, 'mp_pose_landmarks_full', detect)
    rgb = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
    crop_debug = {}
    result = detect_body_landmarks(
        rgb=rgb,
        bbox=(0, 0, 100, 60),
        pose_landmarker=object(),
        split_pose_detection=True,
        crop_debug=crop_debug,
    )

    rectified = np.rot90(rgb, -1)
    np.testing.assert_array_equal(inputs[0], rgb)
    np.testing.assert_array_equal(inputs[1], rectified[:69])
    np.testing.assert_array_equal(inputs[2], rectified[61:])
    np.testing.assert_array_equal(result.xy[6:8], [[71, 25], [77, 35]])
    np.testing.assert_array_equal(result.xy[10], [69, 52])
    hip_vector = result.split_world_vectors[body_pose_fit.FIT_SEGMENTS.index((6, 7))]
    np.testing.assert_allclose(hip_vector, [0.2, -0.1, 0.3])

    artifacts = save_pose_crop_debug(crop_debug, tmp_path)
    assert artifacts['lower']['bbox_xyxy'] == [61, 0, 100, 60]
    np.testing.assert_array_equal(
        artifacts['lower']['corners_xy'], [[61, 60], [61, 0], [100, 0], [100, 60]],
    )
    with Image.open(artifacts['lower']['pose_overlay']) as overlay:
        assert overlay.size == (60, 39)


def test_head_fit_moves_only_neck_and_head_parameters(monkeypatch):
    rest = torch.tensor([
        [0., -4., 12.], [3., 0., 9.], [-3., 0., 9.],
        [7., -2., 0.], [-7., -2., 0.],
    ])
    monkeypatch.setattr(body_pose_fit, 'HEAD_VERTICES', list(range(len(rest))))
    names = ['root_ry', *body_pose_fit.HEAD_ANGLE_NAMES]
    yaw_index = names.index('head_twist')

    def skeleton(parameters):
        half_yaw = parameters[0, yaw_index] / 2
        zero = half_yaw.new_zeros(())
        one = half_yaw.new_ones(())
        return torch.stack((
            zero, zero, zero, zero, half_yaw.sin(), zero, half_yaw.cos(), one,
        )).reshape(1, 1, 8)

    rig = SimpleNamespace(
        get_parameter_names=lambda: names,
        get_parameter_limits=lambda: torch.tensor([[-1., 1.]] * len(names)),
        get_joint_names=lambda: ['c_head'],
        character_torch=SimpleNamespace(
            model_parameters_to_joint_parameters=lambda parameters: parameters,
            joint_parameters_to_skeleton_state=skeleton,
            blend_shape=SimpleNamespace(forward=lambda shape: rest.unsqueeze(0)),
            linear_blend_skinning=SimpleNamespace(
                inverse_bind_pose=torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.]]),
            ),
        ),
    )
    runtime = SimpleNamespace(
        rig=rig, tables={'scale_mean': torch.zeros(68), 'scale_comps': torch.zeros(28, 68)},
    )
    proportions = BodyProportions(torch.zeros(1, 45), torch.zeros(1, 6), torch.zeros(1, 28))
    pose = torch.zeros(1, 130)
    pose[0, 0] = 0.4
    camera = torch.tensor([[0., 0., 2.]])
    target = BodyEstimate(pose, proportions, camera, 800.)
    yaw = torch.tensor(0.4)
    x, y, z = rest.unbind(dim=1)
    rotated = torch.stack((yaw.cos() * x + yaw.sin() * z, y, -yaw.sin() * x + yaw.cos() * z), dim=1)
    xyz = rotated * torch.tensor((0.01, -0.01, -0.01)) + camera
    pixels = (target.focal_length * xyz[:, :2] / xyz[:, 2:] + 256).numpy()

    result, summary = body_pose_fit._fit_head_pose(
        runtime=runtime, target=target, proportions=proportions, head_xy=pixels, image_size=(512, 512),
    )

    assert summary['status'] == 'fitted'
    assert summary['after_px'] < summary['before_px']
    assert result.pose[0, yaw_index] > 0
    torch.testing.assert_close(result.pose[:, :1], pose[:, :1])
    torch.testing.assert_close(result.pose[:, len(names):], pose[:, len(names):])
    torch.testing.assert_close(result.camera_translation, camera)


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


def test_hand_point_and_wrist_angles_participate_in_the_arm_fit():
    observed = torch.zeros(len(BODY_JOINTS), dtype=torch.bool)
    observed[[4, 5, 14]] = True
    projected = torch.zeros(len(body_pose_fit.FIT_SEGMENTS), dtype=torch.bool)
    spatial = projected.clone()
    hand_segment = body_pose_fit.FIT_SEGMENTS.index((4, 14))
    spatial[hand_segment] = True
    fit_stage = MagicMock(side_effect=lambda pose, camera, *_args, **_kwargs: (pose, camera, False))

    body_pose_fit._fit_limbs(
        torch.zeros(1, 130),
        torch.zeros(1, 3),
        fit_stage=fit_stage,
        observed_joints=observed,
        image_segments=projected,
        depth_segments=spatial,
        summary={},
    )

    limb_stage = fit_stage.call_args_list[0]
    assert hand_segment in limb_stage.args[2]
    assert {'l_wrist_ry', 'l_wrist_rz'} <= set(limb_stage.args[3])
    assert {'r_wrist_ry', 'r_wrist_rz'}.isdisjoint(limb_stage.args[3])
    assert limb_stage.kwargs['joint_indices_to_fit'] == (4, 14, 5)
    assert limb_stage.kwargs['use_depth_order']


def test_body_pose_fit_uses_body_joints_without_face_or_hand_observations():
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
        [-58.0, -5.0, 0.0],
        [58.0, -5.0, 0.0],
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


def test_body_canvas_expands_overflowing_edges_at_the_original_scale():
    vertices = np.array(((0, -14, 1), (-11, 0, 1), (2, 5, 1)), dtype=np.float32)
    faces = np.array(((0, 1, 2),))
    frame = {'width': 20, 'height': 20, 'focal': 1}
    colors = {'body_color': (180, 180, 180), 'background_color': None}

    padding = fit_body_canvas(vertices, faces, **frame)
    narrow = render_body(vertices, faces, **frame, **colors)
    expanded = render_body(vertices, faces, **frame, **colors, padding=padding)

    assert padding == (2, 5, 0, 0)
    assert expanded.size == (22, 25)
    assert expanded.getchannel('A').getbbox()[:2] == (1, 1)
    np.testing.assert_array_equal(
        np.array(expanded.crop((2, 5, 22, 25))),
        np.array(narrow),
    )
    assert fit_body_canvas(vertices * (0.5, 0.5, 1), faces, **frame) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ('scale', 'offset', 'rig_pixel', 'landmark_pixel'),
    [
        (1.0, (7, 12), (52, 62), (82, 62)),
        (0.8, (7.25, 20.25), (43, 60), (67, 60)),
    ],
)
def test_body_pose_overlay_scales_and_shifts_both_skeletons(
    scale, offset, rig_pixel, landmark_pixel,
):
    render = Image.new('RGBA', (120, 120))
    rig_xy = np.full((len(BODY_JOINTS), 2), np.nan, dtype=np.float32)
    rig_xy[0] = (45, 50)
    landmark_xy = rig_xy.copy()
    landmark_xy[0] = (75, 50)
    landmarks = BodyLandmarks(landmark_xy, np.zeros((len(BODY_JOINTS), 3)), 100)

    overlay = draw_body_pose_overlay(render, rig_xy, landmarks, scale=scale, offset=offset)

    assert overlay.getpixel(rig_pixel) == RIG_COLOR
    assert overlay.getpixel(landmark_pixel) == LANDMARK_COLOR
    np.testing.assert_array_equal(landmarks.xy[0], (75, 50))


@pytest.fixture
def terminal_contact_geometry():
    vertices = np.array([
        [-0.1, -0.1, 0.95], [0.1, -0.1, 0.95], [0, 0.1, 0.95],
        [-0.1, -0.1, 1.05], [0.1, -0.1, 1.05], [0, 0.1, 1.05],
        [0, 0, 1.02], [0, 0, 1.2], [0, 0, 0.8],
    ], dtype=np.float32)
    region = body_collision.TerminalRegion(
        name='l_hand',
        samples=np.arange(6, 9),
        obstacle_faces=np.array([[0, 2, 1], [3, 4, 5]]),
        joint_indices=(4, 2),
        angles=('wrist_y', 'wrist_z', 'wrist_twist', 'elbow', 'arm_y', 'arm_z'),
    )

    return vertices, region


def test_terminal_collision_distinguishes_penetration_from_occlusion(terminal_contact_geometry):
    vertices, region = terminal_contact_geometry

    collision = body_collision.detect_terminal_collisions(vertices, (region,))['l_hand']

    np.testing.assert_array_equal(collision.indices, [6])
    np.testing.assert_allclose(collision.depths, [0.03], atol=1e-6)


def test_terminal_collision_refines_only_the_penetrating_limb(monkeypatch, terminal_contact_geometry):
    vertices, region = terminal_contact_geometry
    monkeypatch.setattr(body_collision, 'terminal_regions', lambda runtime: (region,))
    monkeypatch.setattr(
        body_collision, 'project_body_joints', lambda **kwargs: np.zeros((len(BODY_JOINTS), 2)),
    )
    rig = SimpleNamespace(
        get_parameter_names=lambda: list(region.angles),
        get_parameter_limits=lambda: torch.tensor([[-1.0, 1.0]] * len(region.angles)),
    )
    proportions = BodyProportions(torch.zeros((1, 45)), torch.zeros((1, 6)), torch.zeros((1, 28)))
    target = BodyEstimate(torch.zeros((1, 130)), proportions, torch.tensor([[0.0, 0.0, 2.0]]), 200)

    def mesh(estimate):
        deformed = vertices.copy()
        deformed[6, 2] += 0.12 * float(estimate.pose[0, 0])

        return deformed

    refined, corrected, diagnostic = body_collision.refine_terminal_collisions(
        runtime=SimpleNamespace(rig=rig),
        target=target,
        proportions=proportions,
        image_size=(200, 200),
        vertices=vertices,
        mesh=mesh,
    )

    assert diagnostic['adjusted'] == ['l_hand']
    assert diagnostic['after'] == {}
    assert corrected[6, 2] >= 1.049
    np.testing.assert_array_equal(corrected[[*range(6), 7, 8]], vertices[[*range(6), 7, 8]])
    torch.testing.assert_close(refined.pose[:, 1:], target.pose[:, 1:])
    assert refined.camera_translation is target.camera_translation


def _rest_shape_reference():
    return torch.tensor([
        [8.5, 129, 17], [-8.5, 129, 17], [8.5, 129, -12],
        [0, 129, 14], [14.5, 129, 10], [-14.5, 129, 10],
        [21, 94, 0], [-21, 94, 0], [18, 82, 0], [-18, 82, 0],
        [18, 102, 0], [-18, 102, 0], [14, 111, 0], [-14, 111, 0],
        [16, 122, 0], [-16, 122, 0], [0, 155, 10],
    ], dtype=torch.float32)


def test_local_body_targets_measure_neutral_profiles_from_the_current_shape():
    reference = _rest_shape_reference()
    vertices = reference.repeat(2, 1, 1)
    vertices[1, :2, 2] += 4
    vertices[1, 6:8, 0] *= 1.1
    vertices[1, 12:14, 0] *= 0.9

    chest = adjust_body_volumes(vertices, reference, chest_volume=0.0)
    hips = adjust_body_volumes(vertices, reference, hip_volume=0.0)
    waist = adjust_body_volumes(vertices, reference, waist_size=1.0)

    torch.testing.assert_close(chest[0, :2], chest[1, :2])
    assert (chest[:, :2, 2] < vertices[:, :2, 2]).all()
    torch.testing.assert_close(chest[:, 2:], vertices[:, 2:])
    torch.testing.assert_close(hips[:, 6:8, 0].abs(), torch.full((2, 2), 18.))
    torch.testing.assert_close(hips[..., 1:], vertices[..., 1:])
    torch.testing.assert_close(waist[:, 12:14, 0].abs(), torch.full((2, 2), 17.1))
    torch.testing.assert_close(waist[..., 1:], vertices[..., 1:])
    torch.testing.assert_close(adjust_body_volumes(vertices, reference), vertices)


def test_global_body_volume_scales_radial_thickness_and_blends_skinning_weights():
    reference = torch.tensor([[2., 3., 4.], [-2., 5., 4.], [2., 8., 4.]])
    axes = BodyAxes(
        origins=torch.zeros(1, 3),
        directions=torch.tensor([[0., 1., 0.]]),
        weights=torch.tensor([[1.], [0.5], [0.]]),
    )

    shaped = adjust_body_volumes(reference.unsqueeze(0), reference, body_volume=1.2, body_axes=axes)

    torch.testing.assert_close(shaped[0], torch.tensor([
        [2.4, 3.0, 4.8], [-2.2, 5.0, 4.4], [2.0, 8.0, 4.0],
    ]))


def test_neck_thickness_scales_only_the_weighted_radial_offsets():
    reference = torch.tensor([[2., 3., 4.], [-2., 5., 4.], [2., 8., 4.]])
    neck_axis = BodyAxes(
        origins=torch.zeros(1, 3),
        directions=torch.tensor([[0., 1., 0.]]),
        weights=torch.tensor([[1.], [0.5], [0.]]),
    )
    shaped = adjust_body_volumes(
        reference.unsqueeze(0), reference, neck_thickness=0.8, neck_axis=neck_axis,
    )

    torch.testing.assert_close(shaped[0], torch.tensor([
        [1.6, 3.0, 3.2], [-1.8, 5.0, 3.6], [2.0, 8.0, 4.0],
    ]))


@pytest.mark.parametrize(
    ('params', 'chest', 'hips', 'waist', 'body_volume'),
    [
        ({}, None, None, None, 1.0),
        ({'gender': 'm'}, 0.0, 0.0, 1.0, 1.0),
        ({'gender': 'f'}, 0.65, None, 0.90, 1.0),
        ({'gender': 'f', 'chest_volume': 0, 'hip_volume': 0.25}, 0.0, 0.25, 0.90, 1.0),
        ({'gender': 'm', 'chest_volume': None, 'waist_size': None}, None, 0.0, None, 1.0),
        ({'chest_volume': 0.5, 'hip_volume': 2.5}, 0.5, 2.5, None, 1.0),
        ({'waist_size': 1.1, 'body_volume': 0.9}, None, None, 1.1, 0.9),
    ],
)
def test_body_geometry_map_resolves_volume_presets(params, chest, hips, waist, body_volume):
    config = node_module._read_cfg(
        {'model': {'mhr_model_pt': 'mhr_model.pt'}, 'params': params},
        'body',
    )

    assert (config.chest_volume, config.hip_volume, config.waist_size) == (chest, hips, waist)
    assert config.body_volume == body_volume
    assert config.neck_thickness == 1.0


def test_neck_thickness_is_independent_of_the_body_multiplier():
    config = node_module._read_cfg(
        {
            'model': {'mhr_model_pt': 'mhr_model.pt'},
            'params': {'gender': 'feminine', 'body_volume': 1.15, 'neck_thickness': 0.9},
        },
        'body',
    )

    assert config.neck_thickness == 0.9
    assert config.body_volume == 1.15


@pytest.mark.parametrize(
    ('params', 'error'),
    [
        ({'body_volume': 1.5}, 'body_volume'),
        ({'body_volume': True}, 'body_volume'),
        ({'neck_thickness': 0.7}, 'neck_thickness'),
        ({'neck_thickness': 1.3}, 'neck_thickness'),
        ({'waist_size': 0}, 'waist_size'),
        ({'hip_volume': float('inf')}, 'hip_volume'),
        ({'chest_volume': -0.01}, 'chest_volume'),
        ({'hip_volume': -0.01}, 'hip_volume'),
    ],
)
def test_body_geometry_map_rejects_invalid_volume_targets(params, error):
    with pytest.raises(ValueError, match=error):
        node_module._read_cfg({'model': {'mhr_model_pt': 'mhr_model.pt'}, 'params': params}, 'body')


@pytest.mark.parametrize('value', [0, -1, True, 1024.5, '1024'])
def test_body_geometry_map_rejects_invalid_long_side(value):
    spec = {'model': {'mhr_model_pt': 'rig.pt'}, 'params': {'long_side': value}}

    with pytest.raises(ValueError, match='params.long_side'):
        node_module._read_cfg(spec, 'body')


@pytest.mark.parametrize(
    ('params', 'deformed', 'target_size', 'output_size', 'padding'),
    [
        ({}, False, (20, 20), (24, 24), (0, 0, 4, 4)),
        ({'gender': 'f', 'chest_volume': 0.25, 'fit_canvas': False}, True,
         (20, 20), (20, 20), (0, 0, 0, 0)),
        ({'long_side': 10, 'fit_canvas': False}, False, (20, 13), (10, 6), (0, 0, 0, 0)),
        ({'long_side': 10}, False, (20, 13), (13, 13), (0, 2, 3, 5)),
        ({'long_side': 32, 'fit_canvas': False}, False, (11, 20), (18, 32), (0, 0, 0, 0)),
    ],
)
def test_body_geometry_map_averages_all_size_channels_and_keeps_target_pose(
    tmp_path, monkeypatch, params, deformed, target_size, output_size, padding,
):
    def estimate(rgb, bbox):
        value = float(rgb[0, 0, 0])

        assert rgb.shape[:2] == target_size[::-1]

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
            reference = _rest_shape_reference()
            vertices = rest_deformation(reference.unsqueeze(0), reference)
            assert vertices[0, 0, 2] < reference[0, 2]
            torch.testing.assert_close(vertices[0, 6:8], reference[6:8])
            torch.testing.assert_close(vertices[0, 12:14, 0].abs(), torch.full((2,), 15.39))
        else:
            assert rest_deformation is None

        return np.array(((-2, -2, 2), (-2, 3, 2), (3, -2, 2)), dtype=np.float32)

    runtime = SimpleNamespace(estimate=estimate, mesh=mesh, faces=np.array(((0, 1, 2),)))
    monkeypatch.setattr(node_module, 'get_instanthmr_model', lambda **kwargs: runtime)
    monkeypatch.setattr(node_module, 'get_yolo', lambda **kwargs: None)
    monkeypatch.setattr(node_module, '_person_bbox', lambda *args: (0, 0, *target_size))
    target, first, second = (tmp_path / f'{value}.png' for value in (4, 12, 16))
    for value, path in zip((4, 12, 16), (target, first, second)):
        Image.new('RGB', target_size, (value, 0, 0)).save(path)
    asset = tmp_path / 'mock-weight'
    asset.touch()
    with DAG(name='test_body_geometry_map', out_dir=tmp_path):
        node = node_module.BodyGeometryMap(
            name='body',
            spec={
                'model': {'checkpoint': str(asset), 'mhr_model_pt': str(asset)},
                'params': {'background_color': 'navy', **params},
            },
        )

    result = node.run(tmp_path, {'default': {'image': target}, 'source': {'images': [first, second]}})
    assert result['source_images'] == [str(first), str(second)]
    assert result['canvas_padding'] == dict(zip(('left', 'top', 'right', 'bottom'), padding))
    with Image.open(result['image']) as image:
        assert image.size == output_size
        assert image.mode == 'RGBA'
        assert image.getpixel((0, 0)) == (0, 0, 128, 255)
