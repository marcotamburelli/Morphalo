"""Optional local volume corrections for the MHR LOD 1 rest mesh."""

from math import tanh

import torch


def adjust_body_volumes(
    vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    *,
    chest_volume: float,
    hip_volume: float,
) -> torch.Tensor:
    """
    Adjust chest and hip volume on the unposed MHR mesh.

    Smooth influence fields are anchored to the neutral LOD 1 topology.
    The chest moves along the forward axis and the hips move laterally.
    Skinning carries these surface changes into the final pose.

    Parameters
    ----------
    vertices : torch.Tensor
        Batched identity-shaped rest vertices, in MHR centimetres.
    reference_vertices : torch.Tensor
        Neutral LOD 1 vertices in the same order, with X to the side, Y up,
        and positive Z forward.
    chest_volume : float
        Unitless signed chest adjustment after adding the preset contribution.
        The forward displacement approaches 4 cm at full influence.
    hip_volume : float
        Unitless signed hip adjustment after adding the preset contribution.
        The lateral displacement approaches 3.5 cm per side at full influence.

    Returns
    -------
    torch.Tensor
        Corrected rest vertices. Adjustments ease toward their maximum
        displacement through ``tanh`` as the magnitude of the controls grows.
    """
    x, y, z = reference_vertices.unbind(dim=-1)
    lateral = x.abs()
    offsets = torch.zeros_like(reference_vertices)

    if chest_volume:
        # Two mirrored fields taper toward the sternum and arms.
        radius_squared = ((lateral - 8.5) / 8.5).square() + ((y - 129) / 14).square()
        front = ((z - 4) / 7).clamp(0, 1)
        front = front.square() * (3 - 2 * front)
        chest_weight = (1 - radius_squared).clamp_min(0).square() * front
        offsets[:, 2] = 4.0 * tanh(chest_volume) * chest_weight

    if hip_volume:
        # Fade into the waist and upper thighs, and concentrate on the sides.
        height_weight = (1 - ((y - 94) / 16).square()).clamp_min(0).square()
        side = ((lateral - 9) / 10).clamp(0, 1)
        side = side.square() * (3 - 2 * side)
        offsets[:, 0] = 3.5 * tanh(hip_volume) * height_weight * side * x.sign()

    return vertices + offsets
