"""Body surface targets on the unposed MHR LOD 1 mesh."""

from dataclasses import dataclass
from math import tanh

import torch


@dataclass(frozen=True)
class BodyAxes:
    """Bind-pose bone axes and their per-vertex skinning weights."""

    origins: torch.Tensor
    directions: torch.Tensor
    weights: torch.Tensor


def rest_body_axes(rig) -> BodyAxes:
    """
    Build the bind-pose axes used to scale the thickness of the torso and limbs.

    Parameters
    ----------
    rig : torch.jit.ScriptModule
        MHR rig with joint names, bind kinematics and vertex skinning weights.

    Returns
    -------
    BodyAxes
        Bone axes in MHR centimetres. The weights blend between adjacent
        bones and taper the effect into the head, hands and feet.
    """
    groups = [('root', 'c_neck', ('root', 'c_spine', 'l_clavicle', 'r_clavicle'))]

    for side in ('l', 'r'):
        for start, end in (
            ('uparm', 'lowarm'), ('lowarm', 'wrist'),
            ('upleg', 'lowleg'), ('lowleg', 'foot'),
        ):
            bone = f'{side}_{start}'
            groups.append((bone, f'{side}_{end}', (bone,)))

    return _rest_axes(rig, groups)


def rest_neck_axis(rig) -> BodyAxes:
    """
    Build the bind-pose neck axis from the base of the neck to the head joint.

    Parameters
    ----------
    rig : torch.jit.ScriptModule
        MHR rig with bind-pose kinematics and vertex skinning weights.

    Returns
    -------
    BodyAxes
        A single neck axis. Skinning influences taper the effect where the
        neck meets the torso and head.
    """
    return _rest_axes(rig, [('c_neck', 'c_head', ('c_neck',))])


@torch.no_grad()
def _rest_axes(rig, groups: list[tuple[str, str, tuple[str, ...]]]) -> BodyAxes:
    """Resolve named bind-pose axis endpoints and sum skinning weights by group."""
    character = rig.character_torch
    names = rig.get_joint_names()
    params = character.mesh.rest_vertices.new_zeros((1, rig.get_parameter_transform().shape[-1]))
    joints = character.model_parameters_to_joint_parameters(params)
    centers = character.joint_parameters_to_skeleton_state(joints)[0, :, :3]
    indices, weights = rig.get_lbsw()
    origins = centers[[names.index(start) for start, _, _ in groups]]
    ends = centers[[names.index(end) for _, end, _ in groups]]
    directions = torch.nn.functional.normalize(ends - origins, dim=1)
    members = torch.tensor([
        [name.startswith(prefixes) for prefixes in (group[2] for group in groups)]
        for name in names
    ], dtype=weights.dtype, device=weights.device)
    vertex_weights = (members[indices.long()] * weights[..., None]).sum(dim=1)

    return BodyAxes(origins, directions, vertex_weights)


def _radial_offsets(vertices: torch.Tensor, axes: BodyAxes) -> torch.Tensor:
    """Blend perpendicular vertex offsets from the bone axes using skinning weights."""
    relative = vertices.unsqueeze(-2) - axes.origins
    along = (relative * axes.directions).sum(dim=-1, keepdim=True)
    radial = relative - along * axes.directions

    return (radial * axes.weights[..., None]).sum(dim=-2)


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    """Ease a unit influence from zero to one at its boundaries."""
    value = value.clamp(0, 1)

    return value.square() * (3 - 2 * value)


def _chest_profile(vertices: torch.Tensor, reference: torch.Tensor) -> torch.Tensor | None:
    """
    Estimate the front of the rib cage from the sternum and lateral ribs.

    Fit a smooth height profile to each strip on the shaped mesh and
    interpolate between them. The lateral strip and sternum are outside
    the central breast fields.
    """
    x, y, z = reference.unbind(dim=-1)
    lateral = x.abs()
    depth = vertices[..., 2]
    front_band = (y > 114) & (y < 144) & (z > 4)
    sternum = front_band & (lateral < 2)
    ribs = front_band & (lateral > 12) & (lateral < 17)

    if not sternum.any() or not (ribs & (x < 0)).any() or not (ribs & (x > 0)).any():
        return None

    height = ((y - 129) / 12).clamp(-1.5, 1.5)
    basis = torch.stack((torch.ones_like(height), height, height.square()), dim=1)
    ridge = torch.diag(y.new_tensor((0, 0.005, 0.005)))

    def profile(mask: torch.Tensor) -> torch.Tensor:
        """Fit the strip's front depth as a quadratic function of height."""
        design = basis[mask]
        normal = design.T @ design + len(design) * ridge
        coefficients = torch.linalg.solve(normal, design.T @ depth[:, mask].T)

        return (basis @ coefficients).T

    inner = profile(sternum)
    outer = torch.where(x < 0, profile(ribs & (x < 0)), profile(ribs & (x > 0)))
    blend = (lateral / 14.5).clamp(0, 1).square()

    return torch.lerp(inner, outer, blend)


def _side_widths(
    vertices: torch.Tensor,
    reference: torch.Tensor,
    height: float,
) -> torch.Tensor | None:
    """Measure the shaped left and right half-widths in a neutral-topology band."""
    x, y, _ = reference.unbind(dim=-1)
    band = ((y - height).abs() <= 2.5) & (x.abs() > 8) & (x.abs() < 25)
    sides = (band & (x < 0), band & (x > 0))

    if not all(mask.any() for mask in sides):
        return None

    lateral = vertices[..., 0].abs()
    widths = [lateral[:, mask].amax(dim=1) for mask in sides]

    return torch.stack(widths, dim=1)


def adjust_body_volumes(
    vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    *,
    chest_volume: float | None = None,
    hip_volume: float | None = None,
    waist_size: float | None = None,
    body_volume: float = 1.0,
    body_axes: BodyAxes | None = None,
    neck_thickness: float = 1.0,
    neck_axis: BodyAxes | None = None,
) -> torch.Tensor:
    """
    Scale body thickness and fit local shape targets before MHR skinning.

    Parameters
    ----------
    vertices : torch.Tensor
        Batched identity-shaped rest vertices in MHR centimetres.
    reference_vertices : torch.Tensor
        Neutral LOD 1 reference in the same order, with X lateral, Y up and
        positive Z forward. It identifies regions independently of identity.
    chest_volume, hip_volume : float or None, optional
        Nonnegative targets relative to the neutral local profile. Zero
        follows the rib cage or side contour. Positive values add
        prominence. Target offsets approach 4 cm forward
        at the chest and 3.5 cm laterally per hip via ``tanh``. None leaves
        that region without a local target.
    waist_size : float or None, optional
        Target waist half-width as a multiple of the neutral side contour
        interpolated between the upper pelvis and lower ribs. None leaves
        the waist without a local target.
    body_volume : float, optional
        Multiplier for radial thickness around the bind-pose torso and limb
        axes. One preserves estimated thickness. Applied before local targets.
    body_axes : BodyAxes or None, optional
        Bind-pose axes and weights, required when body thickness changes.
    neck_thickness : float, optional
        Multiplier for radial thickness around the bind-pose neck axis.
        One preserves the estimated neck thickness.
    neck_axis : BodyAxes or None, optional
        Bind-pose neck axis and weights, required when neck thickness changes.

    Returns
    -------
    torch.Tensor
        Rest vertices with smooth local transitions. The rig parameters
        and skeleton are unaffected.
    """
    shaped = vertices

    if body_volume != 1.0:
        if body_axes is None:
            raise ValueError('Body axes are required to scale body volume')

        shaped = shaped + (body_volume - 1.0) * _radial_offsets(vertices, body_axes)

    if neck_thickness != 1.0:
        if neck_axis is None:
            raise ValueError('The neck axis is required to scale neck thickness')

        shaped = shaped + (neck_thickness - 1.0) * _radial_offsets(vertices, neck_axis)

    corrected = shaped.clone()
    x, y, z = reference_vertices.unbind(dim=-1)
    lateral = x.abs()

    if chest_volume is not None:
        profile = _chest_profile(shaped, reference_vertices)

        if profile is not None:
            sides = _smoothstep((17 - lateral) / 5)
            height = _smoothstep((y - 112) / 12) * _smoothstep((145 - y) / 10)
            front = _smoothstep((z - 4) / 6)
            weight = sides * height * front
            prominence = (1 - ((lateral - 8.5) / 8.5).square()).clamp_min(0).square()
            target = profile + 4.0 * tanh(chest_volume) * prominence
            corrected[..., 2] += weight * (target - shaped[..., 2])

    if hip_volume is not None or waist_size is not None:
        pelvis = _side_widths(shaped, reference_vertices, 102)
        if hip_volume is not None:
            thighs = _side_widths(shaped, reference_vertices, 82)
            hips = _side_widths(shaped, reference_vertices, 94)

            if pelvis is not None and thighs is not None and hips is not None:
                neutral = torch.lerp(thighs, pelvis, (94 - 82) / (102 - 82))
                shift = neutral + 3.5 * tanh(hip_volume) - hips
                shifts = torch.where(x < 0, shift[:, :1], shift[:, 1:])
                weight = (1 - ((y - 94) / 16).square()).clamp_min(0).square()
                weight *= _smoothstep((lateral - 9) / 9)
                corrected[..., 0] += shifts * x.sign() * weight

        if waist_size is not None:
            ribs = _side_widths(shaped, reference_vertices, 122)
            waist = _side_widths(shaped, reference_vertices, 111)

            if pelvis is not None and ribs is not None and waist is not None:
                neutral = torch.lerp(pelvis, ribs, (111 - 102) / (122 - 102))
                scale = waist_size * neutral / waist.clamp_min(1e-4) - 1
                scales = torch.where(x < 0, scale[:, :1], scale[:, 1:])
                weight = (1 - ((y - 111) / 14).square()).clamp_min(0).square()
                weight *= _smoothstep((27 - lateral) / 6)
                corrected[..., 0] += shaped[..., 0] * scales * weight

    return corrected
