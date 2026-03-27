from typing import Optional

import numpy as np


def clamp01(x: float) -> float:
    """
    Clamp a scalar score to the [0, 1] interval.

    Parameters
    ----------
    x : float
        Input scalar.

    Returns
    -------
    float
        Clamped value.
    """
    return float(max(0.0, min(1.0, x)))


def segment_len_3d(
    world_xyz: np.ndarray,
    valid: np.ndarray,
    a: int,
    b: int,
) -> Optional[float]:
    """
    Return the Euclidean segment length in world space.

    Parameters
    ----------
    world_xyz : np.ndarray
        World-space landmark coordinates with shape ``(33, 3)``.
    valid : np.ndarray
        Boolean validity mask with shape ``(33,)``.
    a : int
        First landmark index.
    b : int
        Second landmark index.

    Returns
    -------
    float | None
        Segment length if both endpoints are valid and finite, otherwise None.
    """
    if not valid[a] or not valid[b]:
        return None

    pa = world_xyz[a]
    pb = world_xyz[b]

    if np.isnan(pa).any() or np.isnan(pb).any():
        return None

    return float(np.linalg.norm(pb - pa))


def safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Safely compute a ratio.

    Parameters
    ----------
    a : float | None
        Numerator.
    b : float | None
        Denominator.

    Returns
    -------
    float | None
        ``a / b`` if both values are usable and ``b`` is non-zero, otherwise None.
    """
    if a is None or b is None or b <= 1e-8:
        return None

    return float(a / b)


def joint_angle_3d(
    world_xyz: np.ndarray,
    valid: np.ndarray,
    a: int,
    b: int,
    c: int,
) -> Optional[float]:
    """
    Return the 3D joint angle ABC in degrees.

    Parameters
    ----------
    world_xyz : np.ndarray
        World-space landmark coordinates with shape ``(33, 3)``.
    valid : np.ndarray
        Boolean validity mask with shape ``(33,)``.
    a : int
        First landmark index.
    b : int
        Vertex landmark index.
    c : int
        Third landmark index.

    Returns
    -------
    float | None
        Angle in degrees if all landmarks are valid and finite, otherwise None.
    """
    if not (valid[a] and valid[b] and valid[c]):
        return None

    pa = world_xyz[a]
    pb = world_xyz[b]
    pc = world_xyz[c]

    if np.isnan(pa).any() or np.isnan(pb).any() or np.isnan(pc).any():
        return None

    v1 = pa - pb
    v2 = pc - pb

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 <= 1e-8 or n2 <= 1e-8:
        return None

    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))

    return float(np.degrees(np.arccos(cosang)))
