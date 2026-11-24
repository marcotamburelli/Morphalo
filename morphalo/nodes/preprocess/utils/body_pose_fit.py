"""Refine MHR body articulation against MediaPipe joint estimates."""

from dataclasses import dataclass, replace
from math import pi

import cv2
import numpy as np
from scipy.optimize import least_squares
import torch
from torch.nn import functional as F

from morphalo.nodes.preprocess.utils.geometry import expand_clip_bbox
from morphalo.nodes.vision.human import PoseLandmarksResult, mp_pose_landmarks_full
from third_party.instanthmr.parameters import BodyEstimate, BodyProportions, pack_body
from third_party.instanthmr.runtime import InstantHMRRuntime


# MediaPipe landmark indices paired with the corresponding MHR joint names.
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
    # The MHR ball joint is the closest rig reference to MediaPipe's forefoot.
    (31, 'l_ball'),
    (32, 'r_ball'),
    # The index–pinky midpoint follows the middle finger without fitting digits.
    ((17, 19), 'l_middle2'),
    ((18, 20), 'r_middle2'),
)
SHOULDERS = (0, 1)
HIPS = (6, 7)
HAND_SEGMENTS = ((4, 14), (5, 15))
FOOT_SEGMENTS = ((10, 12), (11, 13))
LIMB_CHAINS = (
    ((0, 2, 4), 'l_uparm', 'l_elbow_bend'),
    ((1, 3, 5), 'r_uparm', 'r_elbow_bend'),
    ((6, 8, 10), 'l_upleg', 'l_knee_bend'),
    ((7, 9, 11), 'r_upleg', 'r_knee_bend'),
)
LIMB_TARGET_JOINTS = [
    *(joint for (_, *distal), _, _ in LIMB_CHAINS for joint in distal),
    *(hand for _, hand in HAND_SEGMENTS),
]
LIMB_SEGMENTS = (
    (0, 2), (2, 4), (1, 3), (3, 5),
    (6, 8), (8, 10), (7, 9), (9, 11),
    *HAND_SEGMENTS,
    *FOOT_SEGMENTS,
)
# The last two points are the virtual shoulder and hip midpoints.
SHOULDER_CENTER = len(BODY_JOINTS)
HIP_CENTER = SHOULDER_CENTER + 1
TORSO_SEGMENTS = (SHOULDERS, HIPS, (HIP_CENTER, SHOULDER_CENTER))
FIT_SEGMENTS = (*TORSO_SEGMENTS, *LIMB_SEGMENTS)
WORLD_DIRECTION_WEIGHT = 2.0
IMAGE_DIRECTION_WEIGHT = 0.55
TORSO_JOINT_WEIGHT = 12.0
LIMB_JOINT_WEIGHT = 3.0
FOOT_JOINT_WEIGHT = 2.0
LIMB_DEPTH_ORDER_WEIGHT = 0.08
LIMB_DEPTH_ORDER_MARGIN = 0.02
TORSO_WIDTH_ANGLE_LIMIT = pi / 60
TORSO_WIDTH_WEIGHT = 4.0
SPLIT_POSE_OVERLAP = 0.05
ROOT_ANGLE_NAMES = ('root_rx', 'root_ry', 'root_rz')
TORSO_ANGLE_NAMES = (
    'spine_twist0', 'spine_lean0', 'spine_bend0',
    'spine_twist1', 'spine_lean1', 'spine_bend1',
)
HEAD_ANGLE_NAMES = (
    'neck_twist', 'neck_lean', 'neck_bend', 'head_twist', 'head_lean', 'head_bend',
)
# MediaPipe Pose indices and rigid head vertices for the verified MHR LOD 1 mesh.
# Order is nose, anatomical left/right eye, then anatomical left/right ear.
HEAD_LANDMARKS = [0, 2, 5, 7, 8]
HEAD_VERTICES = [2765, 2619, 5505, 247, 3133]


@dataclass(frozen=True)
class BodyPoseCrop:
    """Exact MediaPipe input pixels, target bounds, and crop-local results."""

    box: tuple[int, int, int, int]
    rgb: np.ndarray
    landmarks: PoseLandmarksResult | None = None
    error: str | None = None
    target_corners: np.ndarray | None = None


@dataclass(frozen=True)
class BodyLandmarks:
    """
    Target body observations in the order defined by ``BODY_JOINTS``.

    Attributes
    ----------
    xy : numpy.ndarray
        Full-frame pixel coordinates of shape ``(16, 2)``. Missing coordinates
        are represented by NaNs. Finite predictions may extend outside the frame.
    world_xyz : numpy.ndarray
        Full-person MediaPipe world landmarks of shape ``(16, 3)`` in its
        hip-relative frame. The MHR rig determines absolute segment lengths.
    person_size : int
        Long side in pixels of the selected person's bounding box.
    full_frame_xy : numpy.ndarray or None
        Original full-person landmarks for calibrating the full-person
        world coordinates when regional detections update ``xy``.
    split_world_vectors : numpy.ndarray or None
        Local 3D segment displacements from the regional detections in
        ``FIT_SEGMENTS`` order. NaNs select the full-person displacement.
    head_xy : numpy.ndarray or None
        Full-person MediaPipe head landmarks in target pixels, in
        ``HEAD_LANDMARKS`` order. NaNs mark points unavailable for head fitting.
    """

    xy: np.ndarray
    world_xyz: np.ndarray
    person_size: int
    full_frame_xy: np.ndarray | None = None
    split_world_vectors: np.ndarray | None = None
    head_xy: np.ndarray | None = None


def detect_body_landmarks(
    *,
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pose_landmarker,
    split_pose_detection: bool = False,
    crop_debug: dict[str, BodyPoseCrop] | None = None,
) -> BodyLandmarks:
    """
    Detect body joints inside the selected person crop.

    Parameters
    ----------
    rgb : numpy.ndarray
        Full-frame target RGB array.
    bbox : tuple[int, int, int, int]
        Target person box in full-frame pixels.
    pose_landmarker : mediapipe.tasks.python.vision.PoseLandmarker
        Detector used on the selected person's crop.
    split_pose_detection : bool, optional
        Redetect the upper and lower body in overlapping crops. Use a
        vertical cut when the torso axis is at least 45 degrees from
        the image vertical. Defaults to a single detection.
    crop_debug : dict[str, BodyPoseCrop] or None, optional
        Receives the bounds and detector output of each attempted crop
        before its observations are mapped to the full frame and merged.

    Returns
    -------
    BodyLandmarks
        Finite predictions from the crop, with confidence scores unused by
        the fitting.

    Raises
    ------
    RuntimeError
        If the selected crop does not yield a valid MediaPipe pose result.
    """
    height, width = rgb.shape[:2]
    crop_box = expand_clip_bbox(*bbox, w=width, h=height, margin=0.12)
    xy, world, head_xy = _detect_body_crop(
        rgb,
        crop_box,
        pose_landmarker,
        keep_out_of_frame=True,
        region='full',
        crop_debug=crop_debug,
    )
    full = BodyLandmarks(
        xy=xy,
        world_xyz=world,
        person_size=max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
        head_xy=head_xy,
    )

    if not split_pose_detection:
        return full

    return _merge_split_landmarks(
        rgb,
        crop_box,
        pose_landmarker,
        full,
        person_size=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
        crop_debug=crop_debug,
    )


def _detect_body_crop(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    pose_landmarker,
    *,
    keep_out_of_frame: bool,
    region: str,
    crop_debug: dict[str, BodyPoseCrop] | None,
    frame_to_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect body joints in a crop and map their pixels back to the target.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image in the frame used for detection.
    box : tuple[int, int, int, int]
        Crop bounds in detection-frame pixels, with exclusive right and bottom.
    pose_landmarker : mediapipe.tasks.python.vision.PoseLandmarker
        Detector used for the crop.
    keep_out_of_frame : bool
        Keep finite image landmarks outside the crop when true.
    region : str
        Name of the crop in the debug output.
    crop_debug : dict[str, BodyPoseCrop] or None
        Receives the crop bounds and result or detection error.
    frame_to_target : numpy.ndarray or None, optional
        Inverse affine transform mapping a rectified frame to target pixels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Target pixels and axis-aligned world landmarks in ``BODY_JOINTS``
        order, followed by head landmark pixels. Missing points use NaNs.
    """
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    target_corners = None
    target_box = box

    if crop_debug is not None and frame_to_target is not None:
        edges = np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1))) - 0.5
        target_corners = edges @ frame_to_target[:, :2].T + frame_to_target[:, 2] + 0.5
        target_box = (
            *np.floor(target_corners.min(axis=0)).astype(int).tolist(),
            *np.ceil(target_corners.max(axis=0)).astype(int).tolist(),
        )

    try:
        landmarks = mp_pose_landmarks_full(
            crop,
            pose_landmarker,
            require_stable_pose=False,
            keep_out_of_frame=keep_out_of_frame,
        )
    except RuntimeError as exc:
        if crop_debug is not None:
            crop_debug[region] = BodyPoseCrop(
                target_box, crop, error=str(exc), target_corners=target_corners,
            )

        raise

    if crop_debug is not None:
        crop_debug[region] = BodyPoseCrop(
            target_box, crop, landmarks=landmarks, target_corners=target_corners,
        )

    sources = np.asarray([
        indices if isinstance(indices, tuple) else (indices, indices)
        for indices, _ in BODY_JOINTS
    ])
    xy = landmarks.xy[sources].mean(axis=1, dtype=np.float32)
    xy += (x0, y0)
    xy[~landmarks.valid[sources].all(axis=1)] = np.nan

    head_xy = landmarks.xy[HEAD_LANDMARKS].astype(np.float32)
    head_xy += (x0, y0)
    head_xy[~landmarks.strong[HEAD_LANDMARKS]] = np.nan
    world = landmarks.world_xyz[sources].mean(axis=1)

    if frame_to_target is not None:
        rotation = frame_to_target[:, :2]
        offset = frame_to_target[:, 2]
        xy = (xy @ rotation.T + offset).astype(np.float32)
        head_xy = (head_xy @ rotation.T + offset).astype(np.float32)
        world[:, :2] = world[:, :2] @ rotation.T

    return xy, world, head_xy


def _torso_crop_transform(
    box: tuple[int, int, int, int],
    points: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """
    Rotate the crop by quarter turns according to the projected torso direction.

    Parameters
    ----------
    box : tuple[int, int, int, int]
        Target-frame bounds of the expanded person crop.
    points : numpy.ndarray
        Full-person landmark pixels in ``BODY_JOINTS`` order.

    Returns
    -------
    tuple[numpy.ndarray, tuple[int, int]] or None
        Forward affine transform and the width and height of the rotated
        bounds. None keeps the original frame for a downward-facing torso
        closer to vertical than horizontal, or an unusable torso axis.
    """
    shoulders = points[list(SHOULDERS)].mean(axis=0)
    hips = points[list(HIPS)].mean(axis=0)
    axis = (hips - shoulders).astype(np.float64)
    length = np.linalg.norm(axis)

    if not np.isfinite(length) or length < 2:
        return None

    dx, dy = axis

    if abs(dx) >= abs(dy):
        dx, dy = np.sign(dx), 0
    elif dy > 0:
        return None
    else:
        dx, dy = 0, -1

    rotation = np.array(((dy, -dx), (dx, dy)))
    x0, y0, x1, y1 = box
    edges = np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1))) - 0.5
    corners = edges @ rotation.T
    origin = corners.min(axis=0)
    size = tuple(np.ceil(corners.max(axis=0) - origin).astype(int).tolist())

    return np.column_stack((rotation, -origin - 0.5)), size


def _merge_split_landmarks(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    pose_landmarker,
    full: BodyLandmarks,
    *,
    person_size: tuple[int, int],
    crop_debug: dict[str, BodyPoseCrop] | None,
) -> BodyLandmarks:
    """
    Replace upper and lower image joints with observations from overlapping crops.

    Choose a horizontal or vertical cut from the dominant direction of the
    shoulder-to-hip axis, preferring a vertical cut at 45 degrees. Place the
    split at the hip estimates and pad the lower crop by 5% of the person box
    extent along the crop frame. Missing regional image joints
    keep the original estimate. When both detections have a complete shoulder
    or hip axis, align its midpoint height to the full-person detection while
    preserving the regional horizontal position and axis direction. Use a
    regional 3D displacement only when both endpoints come from that crop.
    The torso centerline always uses the full-person 3D displacement.

    Parameters
    ----------
    rgb : numpy.ndarray
        Target RGB image.
    box : tuple[int, int, int, int]
        Bounds of the initial person crop in target pixels.
    pose_landmarker : mediapipe.tasks.python.vision.PoseLandmarker
        Detector reused for both regional crops.
    full : BodyLandmarks
        Initial full-person observations used to locate the split and fill
        gaps in the regional detections.
    person_size : tuple[int, int]
        Width and height of the person box before expansion.
    crop_debug : dict[str, BodyPoseCrop] or None
        Receives the crop bounds and regional detection results before merging.

    Returns
    -------
    BodyLandmarks
        Merged image landmarks and optional regional 3D segment displacements.
    """
    hips = full.xy[list(HIPS)]

    if not np.isfinite(hips).all():
        return full

    frame = rgb
    frame_to_target = None
    person_extent = person_size[1]
    transform = _torso_crop_transform(box, full.xy)

    if transform is not None:
        target_to_frame, size = transform
        frame_to_target = cv2.invertAffineTransform(target_to_frame)
        frame = cv2.warpAffine(rgb, target_to_frame, size)
        hips = hips @ target_to_frame[:, :2].T + target_to_frame[:, 2]
        person_extent = np.abs(target_to_frame[1, :2]) @ person_size
        box = (0, 0, *size)

    hips_y = hips[:, 1]
    x0, y0, x1, y1 = box

    if not y0 < hips_y.mean() < y1:
        return full

    upper_end = int(np.ceil(hips_y.mean()))
    lower_start = max(y0, int(np.floor(hips_y.min() - SPLIT_POSE_OVERLAP * person_extent)))
    first_hand = HAND_SEGMENTS[0][1]
    upper_joints = [*range(HIPS[0]), *range(first_hand, len(BODY_JOINTS))]
    lower_joints = list(range(HIPS[0], first_hand))
    regions = (
        ('upper', (x0, y0, x1, upper_end), upper_joints, SHOULDERS),
        ('lower', (x0, lower_start, x1, y1), lower_joints, HIPS),
    )
    merged_xy = full.xy.copy()
    segments = np.asarray(FIT_SEGMENTS)
    segment_ids = np.flatnonzero((segments < len(BODY_JOINTS)).all(axis=1))
    starts, ends = segments[segment_ids].T
    vectors = np.full((len(segments), 3), np.nan, dtype=full.world_xyz.dtype)
    updated = False

    for region, crop_box, joints, torso_axis in regions:

        try:
            xy, world, _ = _detect_body_crop(
                frame,
                crop_box,
                pose_landmarker,
                keep_out_of_frame=False,
                region=region,
                crop_debug=crop_debug,
                frame_to_target=frame_to_target,
            )
        except RuntimeError:
            continue

        accepted = np.zeros(len(BODY_JOINTS), dtype=bool)
        accepted[joints] = np.isfinite(xy[joints]).all(axis=1)

        if frame_to_target is not None:
            height, width = rgb.shape[:2]
            accepted &= ((xy >= 0) & (xy < (width, height))).all(axis=1)

        if not accepted.any():
            continue

        updated = True
        merged_xy[accepted] = xy[accepted]
        axis = list(torso_axis)
        full_axis_y = full.xy[axis, 1]

        if accepted[axis].all() and np.isfinite(full_axis_y).all():
            merged_xy[axis, 1] += full_axis_y.mean() - xy[axis, 1].mean()

        usable = accepted & np.isfinite(world).all(axis=1)
        use_segment = usable[starts] & usable[ends]
        regional_vectors = world[ends[use_segment]] - world[starts[use_segment]]
        nonzero = np.linalg.norm(regional_vectors, axis=1) > 1e-4
        vectors[segment_ids[use_segment][nonzero]] = regional_vectors[nonzero]

    if not updated:
        return full

    return replace(full, xy=merged_xy, full_frame_xy=full.xy, split_world_vectors=vectors)


def refine_body_pose(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    landmarks: BodyLandmarks,
    image_size: tuple[int, int],
    fit_head: bool = False,
) -> tuple[BodyEstimate, dict]:
    """
    Retarget MHR articulation hierarchically to MediaPipe 3D observations.

    Fit the root and spine from the InstantHMR pose and refine projected
    torso widths within a small angular tolerance. Then fit the limbs to
    image locations using MediaPipe depth order and refine ankle flexion.
    Global depth calibration uses the MediaPipe image landmarks and InstantHMR
    focal length. Landmarks match the rig by anatomical side.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Loaded network and MHR TorchScript rig.
    target : BodyEstimate
        Initial articulation and camera estimated by InstantHMR.
    proportions : BodyProportions
        Fixed proportions used for both fitting and the final render.
    landmarks : BodyLandmarks
        Target-frame positions and relative 3D joint estimates from MediaPipe.
    image_size : tuple[int, int]
        Target width and height in pixels.
    fit_head : bool, optional
        Refine the neck and head against MediaPipe face points after the body
        fit. Defaults to false.

    Returns
    -------
    BodyEstimate
        Refined estimate or the original estimate when fitting is skipped.
    dict
        Fit status, counts of usable observations, stage improvement flags,
        and available direction, reprojection, and torso twist diagnostics.
    """
    joint_count = int(np.isfinite(landmarks.xy).all(axis=1).sum())

    if joint_count < 4:
        return target, {
            'status': 'skipped',
            'reason': 'fewer than four usable body joints',
            'joints': joint_count,
        }

    fitted, summary = _fit_body_joints(
        runtime=runtime,
        target=target,
        proportions=proportions,
        landmarks=landmarks,
        image_size=image_size,
    )

    if fit_head and landmarks.head_xy is not None:
        fitted, summary['head_fit'] = _fit_head_pose(
            runtime=runtime,
            target=fitted,
            proportions=proportions,
            head_xy=landmarks.head_xy,
            image_size=image_size,
        )

    return fitted, summary


def _rotate_head_points(quaternion: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate 3D points by an MHR skeleton-state quaternion in XYZW order."""
    axis = quaternion[:3].expand_as(points)
    cross = 2 * torch.linalg.cross(axis, points)

    return points + quaternion[3] * cross + torch.linalg.cross(axis, cross)


@torch.enable_grad()
def _fit_head_pose(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    head_xy: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[BodyEstimate, dict]:
    """
    Align rigid MHR head landmarks in target pixels through neck and head rotations.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Validated MHR LOD 1 rig and its parameter tables.
    target : BodyEstimate
        Pose and camera after the body fitting stages.
    proportions : BodyProportions
        Shape and scale used to render the mannequin.
    head_xy : numpy.ndarray
        Head observations in target pixels in ``HEAD_LANDMARKS`` order.
    image_size : tuple[int, int]
        Full target width and height in pixels.

    Returns
    -------
    tuple[BodyEstimate, dict]
        Updated neck and head rotation parameters and head-fit diagnostics.
    """
    valid = np.isfinite(head_xy).all(axis=1)

    if valid.sum() < 3:
        return target, {'status': 'skipped', 'reason': 'fewer than three head landmarks'}

    rig = runtime.rig
    character = rig.character_torch
    parameter_names = rig.get_parameter_names()
    angle_indices = [parameter_names.index(name) for name in HEAD_ANGLE_NAMES]
    head = rig.get_joint_names().index('c_head')
    params = pack_body(target.pose, proportions, runtime.tables).detach()
    params = torch.cat((params, torch.zeros_like(proportions.shape)), dim=1)
    camera = target.camera_translation
    points = params.new_tensor(head_xy[valid])
    axes = params.new_tensor((0.01, -0.01, -0.01))
    center = params.new_tensor(image_size) / 2

    # All five anchors are rigidly skinned to c_head on the LOD 1 topology.
    with torch.no_grad():
        rest = character.blend_shape.forward(proportions.shape)[0, HEAD_VERTICES][valid]
        bind = character.linear_blend_skinning.inverse_bind_pose[head]
        local = _rotate_head_points(bind[3:7], rest * bind[7]) + bind[:3]

    def project(candidate: torch.Tensor, translation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the selected rigid head anchors from the candidate pose."""
        joints = character.model_parameters_to_joint_parameters(candidate)
        state = character.joint_parameters_to_skeleton_state(joints)[0, head]
        xyz = (_rotate_head_points(state[3:7], local * state[7]) + state[:3]) * axes + translation

        return target.focal_length * xyz[:, :2] / xyz[:, 2:].clamp_min(1e-4) + center, xyz

    initial = params[0, angle_indices]
    limits = rig.get_parameter_limits()[angle_indices].to(params)
    max_change = params.new_tensor((0.26, 0.26, 0.26, 0.52, 0.52, 0.52))
    lower = torch.maximum(torch.minimum(limits[:, 0], initial), initial - max_change)
    upper = torch.minimum(torch.maximum(limits[:, 1], initial), initial + max_change)
    scale = max(float((points.amax(dim=0) - points.amin(dim=0)).norm()), 1.0)
    empty = torch.empty((0, 2), dtype=torch.long, device=params.device)
    candidate, _, improved = _fit_pose_stage(
        project=project,
        params=params,
        camera=camera,
        angle_indices=angle_indices,
        bounds=(lower, upper),
        image_targets=(empty, params.new_empty((0, 2))),
        world_targets=(empty, params.new_empty((0, 3))),
        pelvis_hint=None,
        joint_targets=(torch.arange(len(points), device=params.device), points, scale, 1.0),
        prior_weights=params.new_tensor((0.18, 0.18, 0.18, 0.06, 0.06, 0.06)),
        steps=100,
    )
    with torch.no_grad():
        before_px = (project(params, camera)[0] - points).square().sum(dim=1).mean().sqrt()
        after_px = (project(candidate, camera)[0] - points).square().sum(dim=1).mean().sqrt()

    summary = {
        'status': 'fitted' if improved else 'unchanged',
        'landmarks': len(points),
        'before_px': round(float(before_px), 2),
        'after_px': round(float(after_px), 2),
    }
    pose = candidate[:, :target.pose.shape[1]].clone()

    return replace(target, pose=pose), summary


def _project_rig_joints(
    character,
    params: torch.Tensor,
    joint_indices: list[int],
    camera_translation: torch.Tensor,
    focal_length: float,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run forward kinematics and project selected joint centers.

    Parameters
    ----------
    character : torch.jit.ScriptModule
        TorchScript MHR character with its joint transform and skeleton.
    params : torch.Tensor
        Batched rig parameters with the shape-channel padding expected by the
        TorchScript parameter transform.
    joint_indices : list[int]
        Indices of the MHR joints to project, in output order.
    camera_translation : torch.Tensor
        Translation in camera-space metres.
    focal_length : float
        Full-frame focal length in pixels.
    image_size : tuple[int, int]
        Width and height of the target frame in pixels.

    Returns
    -------
    points : torch.Tensor
        Projected joint centers of shape ``(N, 2)`` in pixels.
    xyz : torch.Tensor
        Camera-space joint centers of shape ``(N, 3)`` in metres.
    """
    joint_params = character.model_parameters_to_joint_parameters(params)
    state = character.joint_parameters_to_skeleton_state(joint_params)
    axis_scale = params.new_tensor((0.01, -0.01, -0.01))
    xyz = state[0, joint_indices, :3] * axis_scale + camera_translation
    center = params.new_tensor(image_size) / 2
    depth = xyz[:, 2:]

    return focal_length * xyz[:, :2] / depth.clamp_min(1e-4) + center, xyz


@torch.inference_mode()
def project_body_joints(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Project the rendered rig's joint centers into the target image.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Supplies the MHR rig and scale tables.
    target : BodyEstimate
        The pose and camera used by the renderer.
    proportions : BodyProportions
        The proportions used by the renderer.
    image_size : tuple[int, int]
        Output width and height in pixels.

    Returns
    -------
    numpy.ndarray
        Pixel coordinates in ``BODY_JOINTS`` order. Points at or behind the
        near plane are represented by NaNs.
    """
    rig = runtime.rig
    names = rig.get_joint_names()
    joint_indices = [names.index(name) for _, name in BODY_JOINTS]
    params = pack_body(target.pose, proportions, runtime.tables)
    params = torch.cat((params, torch.zeros_like(proportions.shape)), dim=1)
    points, xyz = _project_rig_joints(
        rig.character_torch,
        params,
        joint_indices,
        target.camera_translation,
        target.focal_length,
        image_size,
    )
    points[xyz[:, 2] <= 1e-4] = torch.nan

    return points.numpy()


def _with_torso_centers(points: torch.Tensor) -> torch.Tensor:
    """
    Append the shoulder and hip midpoints in that order.

    Parameters
    ----------
    points : torch.Tensor
        Body joint centers of shape ``(16, D)`` in BODY_JOINTS order, where
        D is two for pixel coordinates or three for camera-space coordinates.

    Returns
    -------
    torch.Tensor
        The input joint centers followed by the two torso centers.
    """
    shoulder_center = points[list(SHOULDERS)].mean(dim=0, keepdim=True)
    hip_center = points[list(HIPS)].mean(dim=0, keepdim=True)

    return torch.cat((points, shoulder_center, hip_center))


@torch.no_grad()
def _shoulder_hip_angle_deg(points: torch.Tensor) -> float | None:
    """
    Measure the 3D angle between the anatomical shoulder and hip axes.

    Parameters
    ----------
    points : torch.Tensor
        Joint coordinates in BODY_JOINTS order, with shape ``(N, 3)``.

    Returns
    -------
    float or None
        Angle in degrees within [0, 180], or None if a required joint is
        invalid or an anatomical axis has negligible length.

    Notes
    -----
    The angle measures relative axis orientation rather than an isolated
    axial rotation of the spine. It is invariant to rigid transformations.
    """
    shoulder_axis = points[SHOULDERS[1]] - points[SHOULDERS[0]]
    hip_axis = points[HIPS[1]] - points[HIPS[0]]

    if not (
        torch.isfinite(shoulder_axis).all()
        and torch.isfinite(hip_axis).all()
    ):
        return None

    if shoulder_axis.norm() < 1e-6 or hip_axis.norm() < 1e-6:
        return None

    cosine = F.cosine_similarity(shoulder_axis, hip_axis, dim=0).clamp(-1, 1)

    return round(float(torch.rad2deg(cosine.acos())), 2)


@torch.no_grad()
def _signed_torso_twist_deg(points: torch.Tensor) -> float | None:
    """
    Measure signed lateral-axis twist around the approximate torso centerline.

    Parameters
    ----------
    points : torch.Tensor
        Camera-aligned 3D joints in BODY_JOINTS order, followed by the
        virtual shoulder and hip midpoints.

    Returns
    -------
    float or None
        Signed angle in degrees, approximately within [-180, 180]. A positive
        sign follows the right-hand rule about the hip-to-shoulder centerline.
        None indicates degenerate or missing anatomical axes.

    Notes
    -----
    The two lateral axes are projected onto the plane perpendicular to the
    virtual torso centerline before the signed angle is measured. This is a
    diagnostic for axial twist, not a direct MHR spine-joint parameter: the
    actual spine may be curved and its rotations distributed over joints.
    """
    torso = points[SHOULDER_CENTER] - points[HIP_CENTER]
    shoulders = points[SHOULDERS[1]] - points[SHOULDERS[0]]
    hips = points[HIPS[1]] - points[HIPS[0]]

    if not torch.isfinite(torch.stack((torso, shoulders, hips))).all():
        return None

    if torso.norm() < 1e-6:
        return None

    axis = F.normalize(torso, dim=0)
    shoulder_plane = shoulders - shoulders.dot(axis) * axis
    hip_plane = hips - hips.dot(axis) * axis

    if shoulder_plane.norm() < 1e-6 or hip_plane.norm() < 1e-6:
        return None

    shoulder_plane = F.normalize(shoulder_plane, dim=0)
    hip_plane = F.normalize(hip_plane, dim=0)
    sine = axis.dot(torch.cross(hip_plane, shoulder_plane, dim=0))
    cosine = hip_plane.dot(shoulder_plane)

    return round(float(torch.rad2deg(torch.atan2(sine, cosine))), 2)


def _align_world_xy_to_image(
    world: torch.Tensor,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int], float | None]:
    """
    Align MediaPipe world XY axis signs with image-space joint directions.

    Parameters
    ----------
    world : torch.Tensor
        MediaPipe world-space joint positions of shape ``(18, 3)``, including
        the two virtual torso centers.
    observations : torch.Tensor
        MediaPipe pixel positions with the same joint ordering, shape
        ``(18, 2)``.

    Returns
    -------
    aligned : torch.Tensor
        World-space points with the selected global X/Y signs applied. Z is
        unchanged, preserving MediaPipe's nearer/farther depth convention.
    signs : tuple[int, int]
        Chosen X and Y multipliers, each either +1 or -1.
    image_disagreement_deg : float or None
        Mean angle between world XY segment directions and the corresponding
        image segment directions after sign alignment, in degrees.

    Notes
    -----
    The signs apply globally to X and Y. The world Z values and the anatomical
    ordering are retained. Perspective and landmark estimation can leave
    residual disagreement, which the stage objectives account for through
    their image-space terms.
    """
    pairs = torch.as_tensor(FIT_SEGMENTS, device=world.device)
    starts, ends = pairs.unbind(dim=1)
    image_vectors = observations[ends] - observations[starts]
    world_xy_vectors = world[ends, :2] - world[starts, :2]
    valid = (
        torch.isfinite(image_vectors).all(dim=1)
        & torch.isfinite(world_xy_vectors).all(dim=1)
        & (image_vectors.norm(dim=1) >= 2)
        & (world_xy_vectors.norm(dim=1) > 1e-4)
    )

    if not valid.any():
        return world, (1, 1), None

    target = F.normalize(image_vectors[valid], dim=1)
    reference = world_xy_vectors[valid]
    best_error = float('inf')
    best_signs = (1, 1)

    for signs in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        multiplier = world.new_tensor(signs)
        candidate = F.normalize(reference * multiplier, dim=1)
        agreement = (candidate * target).sum(dim=1).clamp(-1, 1)
        error = float((1 - agreement).mean())

        if error < best_error:
            best_error = error
            best_signs = signs

    aligned = world.clone()
    aligned[:, :2] *= world.new_tensor(best_signs)
    aligned_vectors = aligned[ends[valid], :2] - aligned[starts[valid], :2]
    cosine = (
        F.normalize(aligned_vectors, dim=1) * target
    ).sum(dim=1).clamp(-1, 1)
    disagreement = torch.rad2deg(cosine.acos()).mean()

    return aligned, best_signs, round(float(disagreement), 2)


def _estimate_world_depth_scale(
    world: torch.Tensor,
    observations: torch.Tensor,
    *,
    focal_length: float,
    image_size: tuple[int, int],
    person_size: int,
) -> float:
    """
    Calibrate MediaPipe depth against its observed 2D landmarks.

    Fit positive XY and Z scales under perspective projection, anchoring
    the virtual hip midpoint to its image ray. Scales are expressed in
    units of the camera-to-hip distance, which cancels in their ratio.
    A weak prior favors equal scales when the image provides little depth
    information.

    Parameters
    ----------
    world : torch.Tensor
        MediaPipe 3D joints with aligned X/Y signs and virtual torso centers.
    observations : torch.Tensor
        Corresponding full-frame pixel coordinates and virtual torso centers.
    focal_length : float
        InstantHMR focal length in pixels.
    image_size : tuple[int, int]
        Frame width and height in pixels.
    person_size : int
        Long side of the person box in pixels. Used to normalize image error.

    Returns
    -------
    float
        Positive Z-to-XY scale ratio to apply before normalizing 3D segments,
        limited to ``[0.25, 4.0]``. Defaults to 1.0 with insufficient data.
    """
    world = world.detach().cpu().numpy().astype(np.float64)
    pixels = observations.detach().cpu().numpy().astype(np.float64)
    hip, hip_px = world[HIP_CENTER], pixels[HIP_CENTER]
    valid = (
        np.isfinite(world[:len(BODY_JOINTS)]).all(axis=1)
        & np.isfinite(pixels[:len(BODY_JOINTS)]).all(axis=1)
    )

    if (
        valid.sum() < 4
        or not np.isfinite(hip).all()
        or not np.isfinite(hip_px).all()
        or not np.isfinite(focal_length)
        or focal_length <= 0
    ):
        return 1.0

    offsets = world[:len(BODY_JOINTS)][valid] - hip
    targets = pixels[:len(BODY_JOINTS)][valid]
    xy, z = offsets[:, :2], offsets[:, 2]
    center = np.asarray(image_size, dtype=np.float64) / 2
    hip_ray = (hip_px - center) / focal_length
    xy_energy = np.square(xy).sum()

    if xy_energy <= 1e-8 or np.max(np.abs(z)) <= 1e-4:
        return 1.0

    # Start from the orthographic XY scale, then fit both scales in perspective.
    initial_xy = (xy * (targets - hip_px)).sum() / (focal_length * xy_energy)

    if not np.isfinite(initial_xy) or initial_xy <= 0:
        return 1.0

    # Keep the nearest calibrated point in front of the camera.
    max_depth = 0.95 / max(float(-z.min()), 1e-8)
    initial_depth = min(initial_xy, max_depth)
    image_scale = max(person_size, 1)

    def residual(scales: np.ndarray) -> np.ndarray:
        """Return normalized reprojection errors and the isotropic-scale prior."""
        xy_scale, depth_scale = scales
        rays = (hip_ray + xy_scale * xy) / (1 + depth_scale * z[:, None])
        projected = focal_length * rays + center
        pixel_error = (projected - targets) / image_scale
        scale_prior = 0.02 * np.log(depth_scale / xy_scale)

        return np.append(pixel_error.ravel(), scale_prior)

    result = least_squares(
        residual,
        x0=(initial_xy, initial_depth),
        bounds=((np.finfo(float).tiny, np.finfo(float).tiny), (np.inf, max_depth)),
        loss='soft_l1',
        f_scale=0.02,
        max_nfev=60,
    )

    if not result.success or not np.isfinite(result.x).all():
        return 1.0

    xy_scale, depth_scale = result.x

    return float(np.clip(depth_scale / xy_scale, 0.25, 4.0))


@torch.enable_grad()
def _fit_body_joints(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    landmarks: BodyLandmarks,
    image_size: tuple[int, int],
) -> tuple[BodyEstimate, dict]:
    """
    Retarget MHR articulation from MediaPipe 3D skeletal directions.

    The InstantHMR pose supplies the initial angles. MediaPipe landmarks map
    to their named MHR joints by anatomical side. Body shape and bone lengths
    come from the selected MHR proportions.

    The root rotation first matches the 3D hip and hip-to-shoulder directions.
    Spine controls then align the shoulders and torso while retaining a prior
    on their original curvature. A separate pass refines projected torso
    widths while bounding changes to orientation and twist. Limb positions
    are then fitted in the image, guided by depth order where available.
    The virtual torso-axis length can change as the spine bends. The MHR
    skeleton supplies the individual bone lengths.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Supplies the rig, parameter names, limits, and scale tables.
    target : BodyEstimate
        Initial pose and camera.
    proportions : BodyProportions
        Fixed shape and scale coefficients used for fitting and rendering.
    landmarks : BodyLandmarks
        Full-frame image landmarks and hip-relative 3D landmarks used to
        derive target segment directions.
    image_size : tuple[int, int]
        Frame width and height in pixels.

    Returns
    -------
    BodyEstimate
        Retargeted pose and lateral camera translation after hierarchical IK.
    dict
        Fit status and geometry diagnostics before and after fitting.
        Reprojection RMS in pixels summarizes the final joint alignment.
    """
    rig = runtime.rig
    names = rig.get_joint_names()
    joint_indices = [names.index(name) for _, name in BODY_JOINTS]
    params = pack_body(target.pose, proportions, runtime.tables).detach()

    # The TorchScript rig expects shape-channel padding for forward kinematics.
    params = torch.cat((params, torch.zeros_like(proportions.shape)), dim=1)
    camera = target.camera_translation.detach().clone()
    points = torch.as_tensor(
        landmarks.xy, dtype=params.dtype, device=params.device
    )
    observations = _with_torso_centers(points)
    raw_world = _with_torso_centers(
        torch.as_tensor(
            landmarks.world_xyz,
            dtype=params.dtype,
            device=params.device,
        )
    )
    calibration_points = observations

    if landmarks.full_frame_xy is not None:
        calibration_points = _with_torso_centers(
            torch.as_tensor(
                landmarks.full_frame_xy, dtype=params.dtype, device=params.device,
            )
        )

    # Use the body landmarks for global calibration. Hand points only guide the arm fit.
    calibration_points = calibration_points.clone()
    calibration_points[[hand for _, hand in HAND_SEGMENTS]] = torch.nan

    world, world_xy_signs, world_xy_error = _align_world_xy_to_image(
        raw_world, calibration_points,
    )
    world_depth_scale = _estimate_world_depth_scale(
        world,
        calibration_points,
        focal_length=target.focal_length,
        image_size=image_size,
        person_size=landmarks.person_size,
    )
    world = world * world.new_tensor((1, 1, world_depth_scale))
    endpoints = torch.as_tensor(FIT_SEGMENTS, device=params.device)
    starts, ends = endpoints.unbind(dim=1)
    expected = observations[ends] - observations[starts]
    world_vectors = world[ends] - world[starts]

    if landmarks.split_world_vectors is not None:
        regional_vectors = torch.as_tensor(
            landmarks.split_world_vectors, dtype=params.dtype, device=params.device,
        )
        regional = torch.isfinite(regional_vectors).all(dim=1)
        world_vectors[regional] = regional_vectors[regional] * world.new_tensor(
            (*world_xy_signs, world_depth_scale)
        )

    # A nearly collapsed segment gives no reliable projected direction.
    image_segments = torch.isfinite(expected).all(
        dim=1) & (expected.norm(dim=1) >= 2)
    depth_segments = torch.isfinite(world_vectors).all(
        dim=1) & (world_vectors.norm(dim=1) > 1e-4)
    observed_joints = torch.isfinite(points).all(dim=1)
    placement_enabled = bool(observed_joints[list(HIPS)].all())
    summary = {
        'status': 'skipped',
        'joints': int(observed_joints.sum()),
        'segments': int(image_segments.sum()),
        'depth_segments': int(depth_segments.sum()),
        'method': 'mediapipe_3d_retarget',
        'world_xy_signs': list(world_xy_signs),
        'world_xy_alignment_deg': world_xy_error,
        'world_depth_scale': world_depth_scale,
    }

    if not (
        depth_segments.any()
        or image_segments.any()
        or observed_joints[LIMB_TARGET_JOINTS].any()
    ):
        return target, {**summary, 'reason': 'no usable body observations'}

    def project(
        candidate: torch.Tensor,
        translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pixel and camera-space joints with their torso midpoints."""
        xy, xyz = _project_rig_joints(
            rig.character_torch,
            candidate,
            joint_indices,
            translation,
            target.focal_length,
            image_size,
        )

        return _with_torso_centers(xy), _with_torso_centers(xyz)

    def measure(xy: torch.Tensor, xyz: torch.Tensor) -> dict:
        """
        Report pose-fit errors and projected torso widths at the current stage.

        Parameters
        ----------
        xy : torch.Tensor
            Rendered joint coordinates in pixels, including torso midpoints.
        xyz : torch.Tensor
            Corresponding camera-space joint positions in metres.

        Returns
        -------
        dict
            Image-space, 3D direction, and placement diagnostics, including
            shoulder and hip widths in pixels measured from the current pose.
        """
        residual = xy[:len(BODY_JOINTS)][observed_joints] - \
            points[observed_joints]
        vectors = xyz[ends] - xyz[starts]
        metrics = {
            'px': round(float(residual.square().sum(dim=1).mean().sqrt()), 2),
            'torso_cm': round(float(vectors[2].norm() * 100), 3),
        }

        for name, axis in (('shoulders', SHOULDERS), ('hips', HIPS)):
            metrics[f'{name}_width_px'] = round(
                float((xy[axis[1]] - xy[axis[0]]).norm()), 2
            )

        if image_segments.any():
            projected = xy[ends] - xy[starts]
            agreement = (F.normalize(projected, dim=1) *
                         F.normalize(expected, dim=1)).sum(dim=1)
            angles = torch.rad2deg(agreement.clamp(-1, 1).acos())
            metrics['deg'] = round(float(angles[image_segments].mean()), 2)

            diagnostic_segments = {
                'shoulders_deg': SHOULDERS,
                'hips_deg': HIPS,
                'torso_deg': (HIP_CENTER, SHOULDER_CENTER),
                'left_foot_deg': FOOT_SEGMENTS[0],
                'right_foot_deg': FOOT_SEGMENTS[1],
            }

            for name, segment in diagnostic_segments.items():
                index = FIT_SEGMENTS.index(segment)

                if image_segments[index]:
                    metrics[name] = round(float(angles[index]), 2)

        if depth_segments.any():
            depth = F.normalize(vectors[depth_segments], dim=1)[:, 2]
            expected_depth = F.normalize(
                world_vectors[depth_segments], dim=1)[:, 2]
            metrics['depth_mae'] = round(
                float((depth - expected_depth).abs().mean()), 3)
            world_directions = F.normalize(
                world_vectors[depth_segments], dim=1
            )
            rig_directions = F.normalize(vectors[depth_segments], dim=1)
            agreement = (
                rig_directions * world_directions
            ).sum(dim=1).clamp(-1, 1)
            metrics['world_xyz_deg'] = round(
                float(torch.rad2deg(agreement.acos()).mean()), 2
            )

        for label, index in (('shoulders', 0), ('hips', 1)):
            if depth_segments[index]:
                cosine = F.cosine_similarity(
                    vectors[index], world_vectors[index], dim=0
                )
                metrics[f'{label}_xyz_deg'] = round(
                    float(cosine.clamp(-1, 1).acos().rad2deg()), 2
                )

        if placement_enabled:
            metrics['pelvis_px'] = round(
                float((xy[-1] - observations[-1]).norm()), 2)

        return metrics

    xy, xyz = project(params, camera)

    if not (xyz[:, 2] > 1e-4).all():
        return target, {**summary, 'reason': 'initial body joints are behind the near plane'}

    summary.update(
        {f'before_{name}': value for name, value in measure(xy, xyz).items()}
    )
    summary['mediapipe_torso_twist_deg'] = _shoulder_hip_angle_deg(world)
    summary['before_mhr_torso_twist_deg'] = _shoulder_hip_angle_deg(xyz)
    summary['mediapipe_torso_signed_twist_deg'] = _signed_torso_twist_deg(
        world
    )
    summary['before_mhr_torso_signed_twist_deg'] = _signed_torso_twist_deg(
        xyz
    )

    rig_parameters = rig.get_parameter_names()
    limits = rig.get_parameter_limits().to(
        device=params.device, dtype=params.dtype
    )
    def fit_segment_stage(
        candidate: torch.Tensor,
        translation: torch.Tensor,
        segment_indices: tuple[int, ...],
        angle_names: tuple[str, ...] | list[str],
        *,
        joint_indices_to_fit: tuple[int, ...] = (),
        joint_weight: float = LIMB_JOINT_WEIGHT,
        fit_placement: bool = False,
        steps: int = 160,
        use_depth_order: bool = False,
        width_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Solve rig angles for anatomical MediaPipe 3D and image targets.

        Parameters
        ----------
        candidate, translation : torch.Tensor
            Current MHR parameters and camera translation.
        segment_indices : tuple[int, ...]
            Segment indices into ``FIT_SEGMENTS`` activated in this stage.
        angle_names : tuple[str, ...] or list[str]
            Named MHR rotation controls to optimize. All other controls,
            including body proportions, remain untouched.
        joint_indices_to_fit : tuple[int, ...], optional
            Indices in ``BODY_JOINTS`` for absolute 2D alignment.
        joint_weight : float, optional
            Strength of the image-space joint position objective.
        fit_placement : bool, optional
            Whether to allow a small lateral camera shift toward the pelvis.
        steps : int, optional
            Number of iterative IK updates for this stage.
        use_depth_order : bool, optional
            Use only the sign of observed depth differences. Joint pixels
            supply the directional guidance for the limb stage.
        width_only : bool, optional
            Refine projected shoulder and hip widths within a small angular
            tolerance of the fitted torso.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, bool]
            Updated rig pose, camera, and whether a feasible improvement
            of the stage objective was found.
        """
        if not angle_names:
            return candidate, translation, False

        selected = torch.as_tensor(
            segment_indices, dtype=torch.long, device=params.device
        )
        stage_expected = expected[selected]
        stage_vectors = world_vectors[selected]
        image_mask = image_segments[selected]
        world_mask = depth_segments[selected]

        if width_only:
            world_mask = torch.zeros_like(world_mask)

        observed_positions = torch.as_tensor(
            joint_indices_to_fit, dtype=torch.long, device=params.device
        )
        observed_positions = observed_positions[
            observed_joints[observed_positions]
        ]

        if not (
            image_mask.any() or world_mask.any()
            or observed_positions.numel() or fit_placement
        ):
            return candidate, translation, False

        angle_indices = [rig_parameters.index(name) for name in angle_names]
        max_change = candidate.new_tensor([
            pi if name == 'root_ry' else
            1.5 if name.startswith('root_') else
            1.4 if 'spine_twist' in name else
            0.9 if name.startswith('spine_') else
            pi if name.endswith(('_ry', '_rz', '_twist')) else 1.0
            for name in angle_names
        ])

        if width_only:
            max_change.fill_(2 * TORSO_WIDTH_ANGLE_LIMIT)

        initial = candidate[0, angle_indices]
        lower, upper = limits[angle_indices].unbind(dim=1)

        # Keep the initial pose feasible even outside an authored limit.
        lower = torch.maximum(torch.minimum(
            lower, initial), initial - max_change)
        upper = torch.minimum(torch.maximum(
            upper, initial), initial + max_change)

        prior_weights = candidate.new_tensor([
            0.002 if name.startswith('root_') else
            0.08 if 'spine_bend' in name else
            0.05 if 'spine_lean' in name else
            0.02 if 'spine_twist' in name else
            0.02 if name.endswith('_twist') else 0.003
            for name in angle_names
        ])

        if use_depth_order:
            prior_weights *= 0.1

        return _fit_pose_stage(
            project=project,
            params=candidate,
            camera=translation,
            angle_indices=angle_indices,
            bounds=(lower, upper),
            image_targets=(
                endpoints[selected][image_mask],
                F.normalize(stage_expected[image_mask], dim=1),
            ),
            world_targets=(
                endpoints[selected][world_mask],
                F.normalize(stage_vectors[world_mask], dim=1),
            ),
            joint_targets=(
                observed_positions,
                points[observed_positions],
                max(landmarks.person_size, 1),
                joint_weight,
            ) if observed_positions.numel() else None,
            pelvis_hint=(
                observations[-1],
                landmarks.person_size,
                target.focal_length,
            ) if fit_placement and placement_enabled else None,
            prior_weights=prior_weights,
            steps=steps,
            use_depth_order=use_depth_order,
            width_targets=(
                endpoints[selected][image_mask],
                stage_expected[image_mask].norm(dim=1),
                max(landmarks.person_size, 1),
            ) if width_only else None,
        )

    hip_axis = expected[1]
    hip_axis_visible = bool(torch.isfinite(hip_axis).all() and hip_axis.norm() >= 2)
    params, camera, torso_fitted = _fit_torso(
        params,
        camera,
        fit_stage=fit_segment_stage,
        project=project,
        hip_axis_visible=hip_axis_visible,
        summary=summary,
    )
    params, camera, limbs_fitted = _fit_limbs(
        params,
        camera,
        fit_stage=fit_segment_stage,
        observed_joints=observed_joints,
        image_segments=image_segments,
        depth_segments=depth_segments,
        summary=summary,
    )
    fitted = torso_fitted or limbs_fitted

    if not fitted:
        summary['reason'] = 'no feasible pose improvement'

        return target, summary

    xy, xyz = project(params, camera)
    summary.update({f'after_{name}': value for name,
                   value in measure(xy, xyz).items()})
    summary['after_mhr_torso_twist_deg'] = _shoulder_hip_angle_deg(xyz)
    summary['after_mhr_torso_signed_twist_deg'] = _signed_torso_twist_deg(xyz)
    residual = xy[:len(BODY_JOINTS)][observed_joints] - points[observed_joints]
    summary['after_fit_px'] = round(
        float(residual.square().sum(dim=1).mean().sqrt()),
        2
    )

    for axis, label in ((SHOULDERS, 'shoulders'), (HIPS, 'hips')):
        segment = FIT_SEGMENTS.index(axis)
        start, end = endpoints[segment]

        if image_segments[segment]:
            direction = xy[end] - xy[start]
            agreement = F.cosine_similarity(
                direction,
                expected[segment],
                dim=0
            )
            summary[f'after_{label}_fit_deg'] = round(
                float(agreement.clamp(-1, 1).acos().rad2deg()),
                2
            )

        if depth_segments[segment] and world_vectors[segment, 2].abs() > 1e-4:
            depth_delta = xyz[end, 2] - xyz[start, 2]
            summary[f'after_{label}_depth_order_matches'] = bool(
                depth_delta * world_vectors[segment, 2] > 0
            )

    summary['status'] = 'fitted'
    pose = params[:, :target.pose.shape[1]].clone()

    return replace(target, pose=pose, camera_translation=camera), summary


def _fit_torso(
    params: torch.Tensor,
    camera: torch.Tensor,
    *,
    fit_stage,
    project,
    hip_axis_visible: bool,
    summary: dict,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Fit the pelvis, spine, and projected torso widths in order.

    Parameters
    ----------
    params, camera : torch.Tensor
        Rig parameters and camera at the start of fitting.
    fit_stage : callable
        Optimizes a stage against the observed torso landmarks.
    project : callable
        Returns projected and camera-space joints for a candidate pose.
    hip_axis_visible : bool
        Whether the hip axis constrains the projected width refinement.
    summary : dict
        Receives per-stage diagnostics and the torso geometry before width refinement.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, bool]
        Fitted torso parameters, camera, and whether any stage improved.
    """
    # Align the anatomical pelvis first. The complete 3D right/left hip axis
    # removes the front/back sign ambiguity of an undirected image segment.
    params, camera, root_fitted = fit_stage(
        params,
        camera,
        (1, 2),
        ROOT_ANGLE_NAMES,
        joint_indices_to_fit=HIPS,
        joint_weight=TORSO_JOINT_WEIGHT,
        fit_placement=True,
        steps=220,
    )
    summary['root_retargeted'] = bool(root_fitted)

    # Spine rotations distribute the upper-torso orientation over MHR's two
    # overlapping bending controls. Their original values remain soft priors.
    params, camera, spine_fitted = fit_stage(
        params,
        camera,
        (0, 2),
        TORSO_ANGLE_NAMES,
        joint_indices_to_fit=SHOULDERS,
        joint_weight=TORSO_JOINT_WEIGHT,
        steps=180,
    )
    summary['spine_retargeted'] = bool(spine_fitted)
    torso_xy, torso_xyz = project(params, camera)
    summary['before_width_refinement_signed_twist_deg'] = _signed_torso_twist_deg(torso_xyz)

    for label, (left, right) in (('shoulders', SHOULDERS), ('hips', HIPS)):
        summary[f'before_width_refinement_{label}_px'] = round(
            float((torso_xy[right] - torso_xy[left]).norm()), 2
        )

    width_angles = TORSO_ANGLE_NAMES

    if hip_axis_visible:
        width_angles = (*ROOT_ANGLE_NAMES, *width_angles)

    params, camera, width_fitted = fit_stage(
        params,
        camera,
        (0, 1),
        width_angles,
        steps=80,
        width_only=True,
    )
    summary['torso_width_refined'] = bool(width_fitted)
    return params, camera, root_fitted or spine_fitted or width_fitted



def _fit_limbs(
    params: torch.Tensor,
    camera: torch.Tensor,
    *,
    fit_stage,
    observed_joints: torch.Tensor,
    image_segments: torch.Tensor,
    depth_segments: torch.Tensor,
    summary: dict,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Fit arms, hands, legs, and feet against the MediaPipe observations.

    Parameters
    ----------
    params, camera : torch.Tensor
        Rig parameters and camera after fitting the torso.
    fit_stage : callable
        Optimizes a stage against the selected MediaPipe observations.
    observed_joints : torch.Tensor
        Availability mask for joint positions in the target image.
    image_segments, depth_segments : torch.Tensor
        Availability masks for projected and 3D segment directions.
    summary : dict
        Receives the limb and foot stage diagnostics.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, bool]
        Limb parameters, camera, and whether either stage improved.
    """
    foot_start = len(FIT_SEGMENTS) - len(FOOT_SEGMENTS)
    hand_by_wrist = dict(HAND_SEGMENTS)
    limb_joints = []
    limb_angles = []

    for (proximal, middle, distal), upper, hinge in LIMB_CHAINS:
        # The original axial twists remain regularized, since sparse
        # landmarks do not determine rotation around an individual bone.
        hand = hand_by_wrist.get(distal)
        joints = (middle, distal) if hand is None else (middle, distal, hand)
        available = [
            joint for joint in joints
            if observed_joints[joint]
        ]
        bone_segments = [
            FIT_SEGMENTS.index(pair) for pair in zip((proximal, *joints[:-1]), joints)
        ]

        if available or depth_segments[bone_segments].any():
            limb_joints.extend(available)
            limb_angles.extend(
                f'{upper}_{suffix}' for suffix in ('twist', 'ry', 'rz')
            )
            limb_angles.append(hinge)

            if hand is not None and (
                observed_joints[hand] or depth_segments[bone_segments[-1]]
            ):
                side = upper[0]
                limb_angles.extend((f'{side}_wrist_ry', f'{side}_wrist_rz'))

    if limb_angles:
        params, camera, limbs_fitted = fit_stage(
            params,
            camera,
            tuple(range(len(TORSO_SEGMENTS), foot_start)),
            limb_angles,
            joint_indices_to_fit=tuple(limb_joints),
            joint_weight=LIMB_JOINT_WEIGHT,
            steps=200,
            use_depth_order=bool(limb_joints),
        )
        summary['limbs_retargeted'] = bool(limbs_fitted)
    else:
        limbs_fitted = False
        summary['limbs_retargeted'] = False

    foot_angles = tuple(
        f'{side}_foot_bend'
        for side, segment in zip(('l', 'r'), FOOT_SEGMENTS)
        if (
            image_segments[FIT_SEGMENTS.index(segment)]
            or depth_segments[FIT_SEGMENTS.index(segment)]
        )
    )
    params, camera, feet_fitted = fit_stage(
        params,
        camera,
        tuple(range(foot_start, len(FIT_SEGMENTS))),
        foot_angles,
        joint_indices_to_fit=tuple(
            joint for _, joint in FOOT_SEGMENTS if observed_joints[joint]
        ),
        joint_weight=FOOT_JOINT_WEIGHT,
        steps=100,
    )
    summary['feet_retargeted'] = bool(feet_fitted)

    return params, camera, limbs_fitted or feet_fitted


def _fit_pose_stage(
    *,
    project,
    params: torch.Tensor,
    camera: torch.Tensor,
    angle_indices: list[int],
    bounds: tuple[torch.Tensor, torch.Tensor],
    image_targets: tuple[torch.Tensor, torch.Tensor],
    world_targets: tuple[torch.Tensor, torch.Tensor],
    pelvis_hint: tuple[torch.Tensor, int, float] | None,
    prior_weights: torch.Tensor,
    steps: int,
    joint_targets: tuple[torch.Tensor, torch.Tensor, int, float] | None = None,
    use_depth_order: bool = False,
    width_targets: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Fit selected MHR rotations to the observations for a single stage.

    For every selected anatomical segment, compare its unit direction in
    camera space with the corresponding MediaPipe world direction or its
    depth order. Depending on the stage, the image targets are projected
    directions, joint pixels, or torso widths. MHR forward kinematics
    determines the joint positions.

    A parameter-specific quadratic prior retains details that the sparse
    MediaPipe joints cannot identify, particularly axial bone twists and
    spine curvature. The distance between virtual torso midpoints can vary
    with spine curvature.

    Parameters
    ----------
    project : callable
        Maps rig parameters and camera translation to pixel and camera-space
        joints, including the two torso midpoints.
    params : torch.Tensor
        Starting rig parameters, including fixed body proportions.
    camera : torch.Tensor
        Starting camera translation in metres.
    angle_indices : list[int]
        Rig parameters that this stage may change.
    bounds : tuple[torch.Tensor, torch.Tensor]
        Lower and upper search limits for those parameters.
    image_targets : tuple[torch.Tensor, torch.Tensor]
        Endpoint indices and unit image directions of observed segments.
        The endpoint array can be empty when only 3D data is usable.
    world_targets : tuple[torch.Tensor, torch.Tensor]
        Endpoint indices and unit MediaPipe 3D directions, aligned to the
        camera coordinate convention. The MHR proportions set bone lengths.
    pelvis_hint : tuple or None
        Target hip midpoint, person-box size, and focal length for lateral
        placement. The camera's Z translation stays unchanged.
    prior_weights : torch.Tensor
        Non-negative per-angle regularization weights. Higher weights better
        preserve the InstantHMR initialization for unobservable details.
    steps : int
        Maximum Adam updates for the selected angles and optional camera shift.
    joint_targets : tuple[torch.Tensor, torch.Tensor, int, float] or None
        Joint indices, target pixels, person-box size, and a loss multiplier
        for positional alignment. None disables this loss.
    use_depth_order : bool, optional
        Use observed Z differences only to favor the corresponding depth
        order. The limb joint positions provide the image-space objective.
    width_targets : tuple[torch.Tensor, torch.Tensor, int] or None
        Endpoint pairs, observed spans in pixels, and person-box size for
        projected width refinement. The fitted torso limits pose changes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, bool]
        Best rig parameters and camera translation, and whether the stage
        improved its objective within the applicable constraints.
    """
    image_endpoints, expected_directions = image_targets
    image_starts, image_ends = image_endpoints.unbind(dim=1)
    world_endpoints, world_directions = world_targets
    world_starts, world_ends = world_endpoints.unbind(dim=1)
    lower, upper = bounds
    initial = params[0, angle_indices]
    angles = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([angles], lr=0.035)

    if pelvis_hint is not None:
        pelvis, person_size, focal_length = pelvis_hint
        # Express the shift as a fraction of the detected person's size.
        shift = camera.new_zeros((2,), requires_grad=True)
        shift_scale = max(person_size, 1) * camera[0, 2].clamp_min(0.1) / focal_length
        optimizer.add_param_group({'params': [shift], 'lr': 0.008})

    best_cost = float('inf')
    best_params, best_camera = params, camera
    improved = False

    if joint_targets is not None:
        indices, expected_xy, target_size, joint_weight = joint_targets

    if width_targets is not None:
        width_pairs, expected_width, width_scale = width_targets
        width_starts, width_ends = width_pairs.unbind(dim=1)
        reference_xy, reference_xyz = project(params, camera)
        torso_pairs = torch.as_tensor(TORSO_SEGMENTS, device=params.device)
        torso_starts, torso_ends = torso_pairs.unbind(dim=1)
        reference_projected = reference_xy[torso_ends] - reference_xy[torso_starts]
        projected_valid = reference_projected.norm(dim=1) >= 2
        reference_projected = F.normalize(reference_projected[projected_valid], dim=1)
        reference_axes = F.normalize(reference_xyz[torso_ends] - reference_xyz[torso_starts], dim=1)
        reference_twist = _signed_torso_twist_deg(reference_xyz)
        minimum_agreement = np.cos(TORSO_WIDTH_ANGLE_LIMIT)

    if use_depth_order:
        depth_signs = world_directions[:, 2].sign()
        depth_known = world_directions[:, 2].abs() > 1e-4

    for step in range(steps + 1):
        candidate = params.clone()
        candidate[0, angle_indices] = angles
        translation = camera

        if pelvis_hint is not None:
            translation = camera + F.pad(shift * shift_scale, (0, 1))

        xy, xyz = project(candidate, translation)

        if not (xyz[:, 2] > 1e-4).all():
            break

        cost = (prior_weights * (angles - initial).square()).mean()

        if image_starts.numel() and not use_depth_order and width_targets is None:
            vectors = xy[image_ends] - xy[image_starts]
            agreement = (
                F.normalize(vectors, dim=1) * expected_directions
            ).sum(dim=1).clamp(-1, 1)
            cost = cost + IMAGE_DIRECTION_WEIGHT * (1 - agreement).mean()

        if world_starts.numel():
            vectors = xyz[world_ends] - xyz[world_starts]
            directions = F.normalize(vectors, dim=1)

            if use_depth_order:
                if depth_known.any():
                    signed_depth = directions[depth_known, 2] * depth_signs[depth_known]
                    wrong_order = (LIMB_DEPTH_ORDER_MARGIN - signed_depth).relu()
                    cost = cost + LIMB_DEPTH_ORDER_WEIGHT * wrong_order.mean()
            else:
                agreement = (directions * world_directions).sum(dim=1).clamp(-1, 1)
                cost = cost + WORLD_DIRECTION_WEIGHT * (1 - agreement).mean()

        if width_targets is not None:
            width = (xy[width_ends] - xy[width_starts]).norm(dim=1)
            width_error = (width - expected_width) / width_scale
            cost = cost + TORSO_WIDTH_WEIGHT * width_error.square().mean()
            axes = F.normalize(xyz[torso_ends] - xyz[torso_starts], dim=1)
            axis_agreement = (axes * reference_axes).sum(dim=1).clamp(-1, 1)
            projected = xy[torso_ends[projected_valid]] - xy[torso_starts[projected_valid]]
            projected_agreement = (F.normalize(projected, dim=1) * reference_projected).sum(dim=1)
            midpoint_shift = (xy[-2:] - reference_xy[-2:]) / width_scale
            cost = cost + (1 - axis_agreement).mean() + midpoint_shift.square().mean()

        if joint_targets is not None:
            offset = (xy[indices] - expected_xy) / max(target_size, 1)
            cost = cost + joint_weight * offset.square().sum(dim=1).mean()

        if pelvis_hint is not None:
            offset = (xy[-1] - pelvis) / max(person_size, 1)
            pelvis_loss = F.smooth_l1_loss(
                offset,
                torch.zeros_like(offset),
                reduction='sum',
                beta=0.02,
            )
            cost = cost + 0.8 * pelvis_loss + 0.05 * shift.square().sum()

        cost_value = cost.item()

        if not np.isfinite(cost_value):
            break

        if cost_value < best_cost:
            feasible = True

            if width_targets is not None:
                twist = _signed_torso_twist_deg(xyz)
                feasible = bool((axis_agreement >= minimum_agreement).all())
                feasible &= bool((projected_agreement >= minimum_agreement).all())

                if reference_twist is not None:
                    max_twist_change = np.rad2deg(TORSO_WIDTH_ANGLE_LIMIT)
                    feasible &= (
                        twist is not None
                        and abs((twist - reference_twist + 180) % 360 - 180) <= max_twist_change
                    )

            if feasible:
                best_cost = cost_value

            if feasible and step:
                best_params = candidate.detach()
                best_camera = translation.detach()
                improved = True

        if step == steps:
            break

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            angles.clamp_(min=lower, max=upper)

            if pelvis_hint is not None:
                shift.clamp_(-0.15, 0.15)

    return best_params, best_camera, improved
