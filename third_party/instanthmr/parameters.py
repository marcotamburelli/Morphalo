"""MHR pose packing adapted from InstantHMR's training/mhr_cont.py.

Copyright (c) Meta Platforms, Inc. and affiliates.
Distributed under the SAM License in LICENSE-SAM.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

BODY_ROTATION_GROUPS = (
    (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20),
    (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39),
    (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79),
    (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100),
    (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132),
)
BODY_ROTATION_AXES = [axis for group in BODY_ROTATION_GROUPS for axis in group]
BODY_HINGE_AXES = [
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50,
    51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80,
    81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111,
    116, 117, 118, 119, 120, 121, 122, 123,
]

HAND_JOINT_DOFS = (3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1)
HAND_ROTATION_AXES = [dof == 3 for dof in HAND_JOINT_DOFS for _ in range(dof)]
HAND_ROTATION_CHANNELS = [
    axis for axis in HAND_ROTATION_AXES for _ in range(2)
]
HAND_HINGE_AXES = [not axis for axis in HAND_ROTATION_AXES]
HAND_HINGE_CHANNELS = [not channel for channel in HAND_ROTATION_CHANNELS]

TABLE_NAMES = (
    'flex_bounds',
    'scale_bounds',
    'scale_mean',
    'scale_comps',
    'hand_pose_mean',
    'hand_pose_comps',
    'hand_joint_idxs_left',
    'hand_joint_idxs_right',
)


@dataclass(frozen=True)
class BodyProportions:
    """
    Store the body size coefficients that can be transferred across poses.

    Attributes
    ----------
    shape : torch.Tensor
        MHR identity coefficients of shape ``(B, 45)``.
    lengths : torch.Tensor
        Six bounded skeletal length and width channels of shape ``(B, 6)``.
    scale : torch.Tensor
        Bounded bone-scale PCA coefficients of shape ``(B, 28)``, before
        expansion into the rig's scale channels.
    """

    shape: torch.Tensor
    lengths: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def mean(cls, views: list['BodyProportions']) -> 'BodyProportions':
        """
        Average each coefficient with equal weight across source views.

        Parameters
        ----------
        views : list[BodyProportions]
            Non-empty list of estimates with matching tensor shapes and devices.

        Returns
        -------
        BodyProportions
            Coefficient means in the same basis as the input estimates.
        """

        return cls(
            shape=torch.stack([view.shape for view in views]).mean(dim=0),
            lengths=torch.stack([view.lengths for view in views]).mean(dim=0),
            scale=torch.stack([view.scale for view in views]).mean(dim=0),
        )


@dataclass(frozen=True)
class BodyEstimate:
    """
    Store the parameters inferred from a single image.

    Attributes
    ----------
    pose : torch.Tensor
        Tensor of shape ``(1, 130)`` containing the zero MHR root translation,
        global orientation, and body and hand joint angles.
    proportions : BodyProportions
        Shape and size coefficients for the detected person.
    camera_translation : torch.Tensor
        Translation of shape ``(1, 3)`` in camera-space metres. X points
        right, Y points down, and Z points away from the camera.
    focal_length : float
        Full-frame focal length in pixels.
    """

    pose: torch.Tensor
    proportions: BodyProportions
    camera_translation: torch.Tensor
    focal_length: float


def rotation_to_euler(rotation):
    """
    Decode a 6D rotation to extrinsic XYZ Euler angles.

    Parameters
    ----------
    rotation : torch.Tensor
        First two rotation-matrix columns concatenated along the last
        dimension, with shape ``(..., 6)``.

    Returns
    -------
    torch.Tensor
        Roll, pitch, and yaw in radians with shape ``(..., 3)``. At the Euler
        singularity, yaw is set to zero.
    """
    first, raw_second = rotation.split(3, dim=-1)
    first = F.normalize(first, dim=-1)
    third = F.normalize(torch.cross(first, raw_second, dim=-1), dim=-1)
    second = torch.cross(third, first, dim=-1)

    horizontal = torch.linalg.vector_norm(first[..., :2], dim=-1)
    singular = horizontal < 1e-6
    roll = torch.where(
        singular,
        torch.atan2(-third[..., 1], second[..., 1]),
        torch.atan2(second[..., 2], third[..., 2]),
    )
    pitch = torch.atan2(-first[..., 2], horizontal)
    yaw = torch.where(singular, 0.0, torch.atan2(first[..., 1], first[..., 0]))

    return torch.stack((roll, pitch, yaw), dim=-1)


def _decode_hinges(channels):
    """Convert consecutive sine/cosine pairs to joint angles in radians."""
    sin, cos = channels.unflatten(-1, (-1, 2)).unbind(dim=-1)

    return torch.atan2(sin, cos)


def _decode_hand(channels):
    """Decode 54 continuous channels into 27 joint angles in MHR hand order."""
    rotations = channels[..., HAND_ROTATION_CHANNELS].unflatten(-1, (-1, 6))
    angles = channels.new_empty((*channels.shape[:-1], 27))
    angles[..., HAND_ROTATION_AXES] = rotation_to_euler(rotations).flatten(-2)
    angles[..., HAND_HINGE_AXES] = _decode_hinges(
        channels[..., HAND_HINGE_CHANNELS],
    )

    return angles


def _apply_bounds(values, bounds):
    """Map predictions to the interval defined by a minimum and half-range."""
    minimum, half_range = bounds

    return minimum + half_range * (values.tanh() + 1)


def decode_body(coefficients, tables):
    """
    Decode network coefficients into joint pose and body proportions.

    Parameters
    ----------
    coefficients : torch.Tensor
        Network head output of shape ``(B, 447)``, ordered as global rotation,
        continuous body parameters, shape, bone-scale coefficients, and two
        hand blocks.
    tables : dict[str, torch.Tensor]
        Checkpoint tensors named in ``TABLE_NAMES``, including size bounds,
        hand bases, and hand joint indices.

    Returns
    -------
    pose : torch.Tensor
        Tensor of shape ``(B, 130)`` containing zero root translation,
        orientation, and joint angles with the decoded finger angles inserted.
    proportions : BodyProportions
        Shape and bounded length, width, and bone-scale coefficients.
    """
    root, continuous, shape, scale, hands = coefficients.split(
        (6, 260, 45, 28, 108),
        dim=-1,
    )
    rotations, hinges, lengths = continuous.split((138, 116, 6), dim=-1)

    angles = continuous.new_empty((*continuous.shape[:-1], 133))
    angles[..., BODY_ROTATION_AXES] = rotation_to_euler(
        rotations.unflatten(-1, (-1, 6)),
    ).flatten(-2)
    angles[..., BODY_HINGE_AXES] = _decode_hinges(hinges)

    # The remaining body channels are lengths/widths and unused jaw angles.
    orientation = rotation_to_euler(root)
    pose = torch.cat(
        (torch.zeros_like(orientation), orientation, angles[..., :124]),
        dim=-1,
    )

    hands = hands.unflatten(-1, (2, 54))
    reconstructed_hands = (
        tables['hand_pose_mean'] + hands @ tables['hand_pose_comps']
    )
    left, right = _decode_hand(reconstructed_hands).unbind(dim=-2)
    pose[..., tables['hand_joint_idxs_left']] = left
    pose[..., tables['hand_joint_idxs_right']] = right

    proportions = BodyProportions(
        shape,
        _apply_bounds(lengths, tables['flex_bounds']),
        _apply_bounds(scale, tables['scale_bounds']),
    )

    return pose, proportions


def pack_body(pose, proportions, tables):
    """
    Pack the rigid, articulated, and size channels in MHR rig order.

    Parameters
    ----------
    pose : torch.Tensor
        Root and joint parameters of shape ``(B, 130)`` from ``decode_body``.
    proportions : BodyProportions
        Lengths, widths, and scale coefficients for the body to render.
        The decoder receives the 45 shape coefficients separately.
    tables : dict[str, torch.Tensor]
        Checkpoint tensors containing the 68-channel scale mean and PCA basis.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, 204)`` ordered as pose, six lengths and widths,
        and 68 expanded bone scales.
    """
    bone_scales = (
        tables['scale_mean'] + proportions.scale @ tables['scale_comps']
    )

    return torch.cat((pose, proportions.lengths, bone_scales), dim=-1)
