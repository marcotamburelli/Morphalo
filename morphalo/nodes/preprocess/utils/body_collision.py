"""Detect and locally refine hand and foot penetration in a posed MHR mesh."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
import torch

from morphalo.nodes.preprocess.utils.body_pose_fit import BODY_JOINTS, project_body_joints
from third_party.instanthmr.parameters import BodyEstimate, BodyProportions
from third_party.instanthmr.runtime import InstantHMRRuntime


SURFACE_TOLERANCE = 0.001
SURFACE_CLEARANCE = 0.002
MAX_TERMINAL_SAMPLES = 320
MAX_LANDMARK_SHIFT_FRACTION = 0.04


@dataclass(frozen=True)
class TerminalRegion:
    """Sampled surface of an extremity and its possible collision faces."""

    name: str
    samples: np.ndarray
    obstacle_faces: np.ndarray
    joint_indices: tuple[int, int]
    angles: tuple[str, ...]


@dataclass(frozen=True)
class TerminalCollision:
    """Penetrating samples and their enclosing surfaces along camera-space Z."""

    region: TerminalRegion
    indices: np.ndarray
    near_points: np.ndarray
    far_points: np.ndarray
    near_normals: np.ndarray
    far_normals: np.ndarray
    depths: np.ndarray

    @property
    def score(self) -> float:
        """Average penetration depth across all samples, including clear points."""
        return float(self.depths.sum() / len(self.region.samples))


def terminal_regions(runtime: InstantHMRRuntime) -> tuple[TerminalRegion, ...]:
    """
    Identify hands, feet and their obstacles from MHR skinning assignments.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Loaded rig and mesh connectivity.

    Returns
    -------
    tuple[TerminalRegion, ...]
        Sample indices, obstacle triangles, pose controls and projected
        joint references for each terminal limb.
    """
    rig = runtime.rig
    indices, weights = (value.numpy() for value in rig.get_lbsw())
    names = np.asarray(rig.get_joint_names())
    bones = names[indices[np.arange(len(weights)), weights.argmax(axis=1)]]
    body_joints = [name for _, name in BODY_JOINTS]
    regions = []

    for side in ('l', 'r'):
        for label, parts, adjacent, joint_names, angle_suffixes in (
            (
                'hand',
                ('wrist', 'thumb', 'index', 'middle', 'ring', 'pinky'),
                'lowarm',
                ('wrist', 'lowarm'),
                ('wrist_ry', 'wrist_rz', 'lowarm_twist', 'elbow_bend', 'uparm_ry', 'uparm_rz'),
            ),
            (
                'foot',
                ('foot', 'ball', 'talocrural', 'subtalar', 'transversetarsal'),
                'lowleg',
                ('foot', 'lowleg'),
                ('foot_bend', 'foot_lean0', 'foot_lean1', 'knee_bend', 'upleg_ry', 'upleg_rz'),
            ),
        ):
            distal = np.zeros(len(names), dtype=bool)
            excluded = np.zeros(len(names), dtype=bool)

            for index, name in enumerate(names):
                if name.startswith(f'{side}_'):
                    local = name[2:]
                    distal[index] = local.startswith(parts)
                    excluded[index] = distal[index] or local.startswith(adjacent)

            samples = np.flatnonzero(np.isin(bones, names[distal]))

            if samples.size > MAX_TERMINAL_SAMPLES:
                samples = samples[np.linspace(0, samples.size - 1, MAX_TERMINAL_SAMPLES, dtype=int)]

            faces = runtime.faces[~np.isin(bones[runtime.faces], names[excluded]).any(axis=1)]

            if samples.size and faces.size:
                regions.append(TerminalRegion(
                    name=f'{side}_{label}',
                    samples=samples,
                    obstacle_faces=faces,
                    joint_indices=tuple(body_joints.index(f'{side}_{name}') for name in joint_names),
                    angles=tuple(f'{side}_{name}' for name in angle_suffixes),
                ))

    return tuple(regions)


def detect_terminal_collisions(
    vertices: np.ndarray,
    regions: tuple[TerminalRegion, ...],
) -> dict[str, TerminalCollision]:
    """
    Find terminal-limb surface samples inside an obstacle in camera space.

    Each ray must see an outward-facing front surface before the sample
    and a back surface after it. Hits on the same side of the sample are
    ordinary occlusion. Near-tangent faces are ignored for depth estimates.

    Parameters
    ----------
    vertices : numpy.ndarray
        Posed mesh vertices in camera-space metres.
    regions : tuple[TerminalRegion, ...]
        Terminal samples and obstacle faces with connected surfaces removed.

    Returns
    -------
    dict[str, TerminalCollision]
        Regions with penetrating samples, their enclosure surfaces, and
        estimated distances to the nearer surface along camera-space Z.
    """
    result = {}

    for region in regions:
        samples = vertices[region.samples]
        triangles = vertices[region.obstacle_faces]
        lower = samples.min(axis=0)[:2]
        upper = samples.max(axis=0)[:2]
        triangles = triangles[
            ((triangles.max(axis=1)[:, :2] >= lower)
             & (triangles.min(axis=1)[:, :2] <= upper)).all(axis=1)
        ]

        if not triangles.size:
            continue

        a = triangles[:, 0]
        ab, ac = triangles[:, 1] - a, triangles[:, 2] - a
        normal = np.cross(ab, ac)
        length = np.linalg.norm(normal, axis=1)
        strong = np.abs(normal[:, 2]) > np.maximum(1e-10, 0.03 * length)
        a, ab, ac = a[strong], ab[strong], ac[strong]
        determinants = normal[strong, 2]
        normals = normal[strong] / length[strong, None]

        if not determinants.size:
            continue

        offset = samples[:, None, :2] - a[None, :, :2]
        u = (offset[..., 0] * ac[:, 1] - offset[..., 1] * ac[:, 0]) / determinants
        v = (ab[:, 0] * offset[..., 1] - ab[:, 1] * offset[..., 0]) / determinants
        hit = (u >= 0) & (v >= 0) & (u + v <= 1)
        z = a[:, 2] + u * ab[:, 2] + v * ac[:, 2]
        z_near = np.where(hit & (z < samples[:, 2:]), z, -np.inf)
        z_far = np.where(hit & (z > samples[:, 2:]), z, np.inf)
        near = z_near.argmax(axis=1)
        far = z_far.argmin(axis=1)
        rows = np.arange(len(samples))
        delta_near = samples[:, 2] - z_near[rows, near]
        delta_far = z_far[rows, far] - samples[:, 2]
        valid = (
            (determinants[near] < 0) & (determinants[far] > 0)
            & np.isfinite(delta_near) & np.isfinite(delta_far)
            & (delta_near > SURFACE_TOLERANCE) & (delta_far > SURFACE_TOLERANCE)
        )

        if not valid.any():
            continue

        near_points, far_points = samples[valid].copy(), samples[valid].copy()
        near_points[:, 2] = z_near[rows[valid], near[valid]]
        far_points[:, 2] = z_far[rows[valid], far[valid]]
        result[region.name] = TerminalCollision(
            region=region,
            indices=region.samples[valid],
            near_points=near_points,
            far_points=far_points,
            near_normals=normals[near[valid]],
            far_normals=normals[far[valid]],
            depths=np.minimum(delta_near[valid], delta_far[valid]),
        )

    return result


def refine_terminal_collisions(
    *,
    runtime: InstantHMRRuntime,
    target: BodyEstimate,
    proportions: BodyProportions,
    image_size: tuple[int, int],
    vertices: np.ndarray,
    mesh: Callable[[BodyEstimate], np.ndarray],
) -> tuple[BodyEstimate, np.ndarray, dict]:
    """
    Reduce terminal collisions with small, bounded changes to the limb pose.

    For each colliding part, try exits through the nearest, front and back
    surfaces. Accept candidates that reduce measured penetration and keep
    the projected wrist or ankle near its position from the pose fitting.

    Parameters
    ----------
    runtime : InstantHMRRuntime
        Rig used to resolve joint limits and project the fitted pose.
    target : BodyEstimate
        Pose to refine, including camera.
    proportions : BodyProportions
        Fixed proportions used for the fit and mesh generation.
    image_size : tuple[int, int]
        Unpadded target width and height in pixels.
    vertices : numpy.ndarray
        Initial posed mesh, with any surface deformations applied.
    mesh : callable
        Regenerates the same surface geometry using a candidate pose.

    Returns
    -------
    tuple[BodyEstimate, numpy.ndarray, dict]
        Best accepted pose and mesh, plus initial and residual collision
        measurements for each affected extremity.
    """
    regions = terminal_regions(runtime)
    collisions = detect_terminal_collisions(vertices, regions)
    initial_collisions = collisions.copy()
    rig = runtime.rig
    names = rig.get_parameter_names()
    limits = rig.get_parameter_limits().numpy()
    initial_pose = target.pose
    initial_xy = project_body_joints(
        runtime=runtime, target=target, proportions=proportions, image_size=image_size,
    ) if collisions else None
    max_shift = MAX_LANDMARK_SHIFT_FRACTION * min(image_size)
    adjusted = set()

    # A second pass uses the enclosure surfaces from the accepted correction.
    for name in (*initial_collisions, *initial_collisions):
        collision = collisions.get(name)

        if collision is None:
            continue

        region = collision.region
        joints = list(region.joint_indices)
        pose_indices = [names.index(angle) for angle in region.angles]
        start = target.pose[0, pose_indices].numpy()
        original = initial_pose[0, pose_indices].numpy()
        max_change = np.array((0.65, 0.65, 0.45, 0.40, 0.35, 0.35))
        lower = np.maximum(np.minimum(limits[pose_indices, 0], original), original - max_change) - start
        upper = np.minimum(np.maximum(limits[pose_indices, 1], original), original + max_change) - start
        variable = (upper - lower) > 1e-5

        if not variable.any() or not np.isfinite(initial_xy[joints]).all():
            continue

        variable_indices = np.asarray(pose_indices)[variable].tolist()
        lower, upper = lower[variable], upper[variable]
        # Start just inside the bounds so a joint already at its limit does
        # not shrink the first trust-region step to floating-point noise.
        margin = np.minimum(0.005, (upper - lower) / 4)
        seed = np.clip(0.0, lower + margin, upper - margin)

        def evaluate(delta: np.ndarray) -> tuple[BodyEstimate, np.ndarray, np.ndarray]:
            """Evaluate bounded limb rotations on the actual skinned mesh."""
            pose = target.pose.clone()
            pose[0, variable_indices] += torch.as_tensor(
                delta, dtype=pose.dtype,
            )
            estimate = replace(target, pose=pose)
            posed_vertices = mesh(estimate)
            xy = project_body_joints(
                runtime=runtime, target=estimate, proportions=proportions, image_size=image_size,
            )

            return estimate, posed_vertices, xy[joints] - initial_xy[joints]

        points = vertices[collision.indices]
        near_first = points[:, 2] - collision.near_points[:, 2] <= collision.far_points[:, 2] - points[:, 2]
        best = None
        best_score = collision.score
        regularization = np.array((0.004, 0.004, 0.006, 0.010, 0.012, 0.012))

        for toward_near in (near_first, np.ones_like(near_first), np.zeros_like(near_first)):
            surface = np.where(toward_near[:, None], collision.near_points, collision.far_points)
            normals = np.where(toward_near[:, None], collision.near_normals, collision.far_normals)

            def residual(delta: np.ndarray) -> np.ndarray:
                """Penalize penetration, screen displacement and large rotations."""
                _, posed_vertices, shift = evaluate(delta)
                separation = ((posed_vertices[collision.indices] - surface) * normals).sum(axis=1)
                penetration = np.maximum(SURFACE_CLEARANCE - separation, 0)
                pixel_penalty = np.nan_to_num(shift, nan=max_shift * 4) / max(max_shift, 1) * 0.012

                return np.concatenate((
                    penetration / np.sqrt(len(separation)),
                    pixel_penalty.ravel(),
                    regularization[variable] * delta,
                ))

            solution = least_squares(
                residual, seed, bounds=(lower, upper),
                jac='3-point', diff_step=0.005, max_nfev=15,
            )
            estimate, posed_vertices, shift = evaluate(solution.x)
            max_displacement = np.linalg.norm(shift, axis=1).max()

            if not np.isfinite(max_displacement) or max_displacement > max_shift:
                continue

            remaining = detect_terminal_collisions(posed_vertices, regions)
            score = remaining[name].score if name in remaining else 0.0

            if any(
                contact.score > (collisions[key].score if key in collisions else 0) + 0.001
                for key, contact in remaining.items() if key != name
            ):
                continue

            # Prefer near-identical projections when residual penetration is comparable.
            score += 0.001 * max_displacement / max(max_shift, 1)

            if score < best_score - SURFACE_TOLERANCE / 10:
                best_score = score
                best = estimate, posed_vertices, remaining

        if best is not None:
            target, vertices, collisions = best
            adjusted.add(name)

    def measures(values: dict[str, TerminalCollision]) -> dict:
        """Summarize observed collisions without serializing the mesh samples."""
        return {
            name: {
                'samples': int(contact.indices.size),
                'max_depth_cm': round(float(contact.depths.max()) * 100, 2),
            }
            for name, contact in values.items()
        }

    return target, vertices, {
        'before': measures(initial_collisions),
        'after': measures(collisions),
        'adjusted': sorted(adjusted),
    }
