"""Refine MHR body articulation against visible MediaPipe joints."""

from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from morphalo.nodes.preprocess.utils.geometry import expand_clip_bbox
from morphalo.nodes.vision.human import mp_pose_landmarks_full
from third_party.instanthmr.parameters import BodyEstimate, BodyProportions, pack_body
from third_party.instanthmr.runtime import InstantHMRRuntime


# MediaPipe landmark index paired with the corresponding MHR joint name.
BODY_JOINTS = (
    (11, 'l_uparm'),
    (12, 'r_uparm'),
    (13, 'l_lowarm'),
    (14, 'r_lowarm'),
    (15, 'l_wrist'),
    (16, 'r_wrist'),
    (23, 'l_upleg'),
    (24, 'r_upleg'),
    (25, 'l_lowleg'),
    (26, 'r_lowleg'),
    (27, 'l_foot'),
    (28, 'r_foot'),
)
TORSO_ANGLE_NAMES = (
    'spine_twist0', 'spine_lean0', 'spine_bend0',
    'spine_twist1', 'spine_lean1', 'spine_bend1',
)
# The shoulder joints descend from separate clavicles on the same spine joint.
# Keep both clavicle rotations fixed to preserve the 3D shoulder span.
LIMB_ANGLE_NAMES = (
    'uparm_twist', 'uparm_ry', 'uparm_rz', 'elbow_bend',
    'upleg_twist', 'upleg_ry', 'upleg_rz', 'knee_bend',
)
FIT_ANGLE_NAMES = TORSO_ANGLE_NAMES + tuple(
    f'{side}_{angle}' for side in ('l', 'r') for angle in LIMB_ANGLE_NAMES
)


def refine_body_pose(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pose_landmarker,
) -> tuple[BodyEstimate, dict]:
    """
    Fit the articulated body to the target image while retaining its camera.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Loaded network and MHR TorchScript rig.
    target : BodyEstimate
        Initial articulation and camera estimated by InstantHMR.
    proportions : BodyProportions
        Fixed proportions used for both fitting and the final render.
    rgb : numpy.ndarray
        Full-frame target RGB array.
    bbox : tuple[int, int, int, int]
        Target person box in full-frame pixels.
    pose_landmarker : mediapipe.tasks.python.vision.PoseLandmarker
        Detector used on the selected person's crop.

    Returns
    -------
    BodyEstimate
        Target estimate with the refined pose, or the original estimate if
        there is insufficient evidence or the reprojection does not improve.
    dict
        Fit status, number of fitted joints, and weighted reprojection RMS
        before and after fitting in pixels when available.
    """
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = expand_clip_bbox(*bbox, w=width, h=height, margin=0.12)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])

    try:
        landmarks = mp_pose_landmarks_full(
            crop,
            pose_landmarker,
            min_visibility=0.5,
            min_presence=0.5,
            require_stable_pose=False,
        )
    except RuntimeError as exc:
        return target, {'status': 'skipped', 'reason': str(exc)}

    visible = [(index, name) for index, name in BODY_JOINTS if landmarks.valid[index]]

    if len(visible) < 4:
        return target, {
            'status': 'skipped',
            'reason': 'fewer than four visible body joints',
            'joints': len(visible),
        }

    indices, joint_names = zip(*visible)
    points = landmarks.xy[list(indices)] + np.array((x0, y0))
    weights = landmarks.visibility[list(indices)] * landmarks.presence[list(indices)]

    return _fit_body_joints(
        runtime=runtime,
        target=target,
        proportions=proportions,
        joint_names=joint_names,
        points=points,
        weights=weights,
        image_size=(width, height),
        person_size=max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
    )


@torch.inference_mode(False)
@torch.enable_grad()
def _fit_body_joints(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    joint_names: tuple[str, ...],
    points: np.ndarray,
    weights: np.ndarray,
    image_size: tuple[int, int],
    person_size: int,
) -> tuple[BodyEstimate, dict]:
    """
    Optimize a subset of MHR angles against body landmarks in image space.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Supplies the rig, parameter names, limits, and scale tables.
    target : BodyEstimate
        Initial pose and fixed perspective camera.
    proportions : BodyProportions
        Fixed shape and scale coefficients for the rendered subject.
    joint_names : tuple[str, ...]
        MHR joints corresponding to the supplied points, in the same order.
    points : numpy.ndarray
        Visible joint locations of shape ``(N, 2)`` in target-frame pixels.
    weights : numpy.ndarray
        Positive confidence weights of shape ``(N,)``.
    image_size : tuple[int, int]
        Target frame width and height in pixels.
    person_size : int
        Long side of the detected person box, used to normalize residuals.

    Returns
    -------
    BodyEstimate
        Pose with the best accepted fit. Local clavicle, head, wrist, and
        finger rotations retain their initial values, as do the root
        transform and camera.
    dict
        Fit status with the number of joints and reprojection error in pixels.
    """
    rig = runtime.rig
    character = rig.character_torch
    rig_joints = rig.get_joint_names()
    rig_parameters = rig.get_parameter_names()
    joint_indices = [rig_joints.index(name) for name in joint_names]
    angle_indices = [rig_parameters.index(name) for name in FIT_ANGLE_NAMES]

    params = pack_body(target.pose, proportions, runtime.tables).detach()
    # TorchScript appends one zero per shape coefficient before applying the
    # joint transform. Reuse that path directly to skip skinning each step.
    params = torch.cat((params, torch.zeros_like(proportions.shape)), dim=1)
    initial = params[0, angle_indices]
    lower, upper = rig.get_parameter_limits()[angle_indices].unbind(dim=1)
    # Allow the input pose when it already lies outside an authored limit.
    lower = torch.maximum(torch.minimum(lower, initial), initial - 0.7)
    upper = torch.minimum(torch.maximum(upper, initial), initial + 0.7)
    angles = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([angles], lr=0.035)

    camera = target.camera_translation.detach().clone()
    center = params.new_tensor(image_size) / 2
    axis_scale = params.new_tensor((0.01, -0.01, -0.01))
    observations = torch.as_tensor(points, dtype=params.dtype)
    confidence = torch.as_tensor(weights, dtype=params.dtype)
    confidence = confidence / confidence.sum()

    best_cost = float('inf')
    best_error = None
    first_error = None
    best_angles = None

    for step in range(51):
        candidate = params.clone()
        candidate[0, angle_indices] = angles
        joint_params = character.model_parameters_to_joint_parameters(candidate)
        state = character.joint_parameters_to_skeleton_state(joint_params)
        xyz = state[0, joint_indices, :3] * axis_scale + camera
        projected = (
            target.focal_length * xyz[:, :2] / xyz[:, 2:].clamp_min(1e-4)
            + center
        )

        residual = projected - observations
        error = (confidence * residual.square().sum(dim=1)).sum().sqrt().item()
        # A robust reprojection loss limits bad landmark influence. The pose
        # prior discourages rotations that the image barely constrains.
        distances = F.smooth_l1_loss(
            residual / person_size,
            torch.zeros_like(residual),
            reduction='none',
            beta=0.02,
        ).sum(dim=1)
        cost = (confidence * distances).sum() + 0.025 * (angles - initial).square().mean()
        cost_value = cost.item()

        if not torch.isfinite(cost) or not (xyz[:, 2] > 1e-4).all():
            break

        if first_error is None:
            first_error, best_cost = error, cost_value
        elif cost_value < best_cost and error < first_error:
            best_cost, best_error = cost_value, error
            best_angles = angles.detach().clone()

        if step == 50:
            break

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            angles.clamp_(min=lower, max=upper)

    result = {'status': 'skipped', 'joints': len(joint_names)}

    if best_angles is None:
        result['reason'] = 'no improvement in weighted reprojection'

        if first_error is not None:
            result['before_px'] = round(first_error, 2)

        return target, result

    pose = target.pose.detach().clone()
    pose[0, angle_indices] = best_angles
    result.update(
        status='fitted',
        before_px=round(first_error, 2),
        after_px=round(best_error, 2),
    )

    return replace(target, pose=pose), result
