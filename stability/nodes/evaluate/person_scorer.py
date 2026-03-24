from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from stability.cache.models import (get_mediapipe_hand_landmarker,
                                    get_mediapipe_pose_landmarker)
from stability.core.paths import make_node_output_path
from stability.dag.core import NodeRef
from stability.nodes.common.config_resolve import SpecInput, resolve_spec
from stability.nodes.common.io import write_json
from stability.nodes.evaluate.helper import (clamp01, joint_angle_3d,
                                             safe_ratio, segment_len_3d)
from stability.nodes.evaluate.img_wiring_mixin import ImgBundle, ImgWiringMixin
from stability.nodes.vision.human import (PoseLandmarksResult,
                                          mp_hand_landmarks_full,
                                          mp_pose_landmarks_full)


@dataclass
class ScorerConfig:
    device: str
    pose_landmarker_task: str
    hand_landmarker_task: str
    w_pose: float
    w_hands: float
    w_feet: float
    score_weight: float


def _read_cfg(spec: dict, node_id: str) -> ScorerConfig:
    model = spec.get('model', {})
    params = spec.get('params', {})

    device = model.get('device', 'cuda')

    pose_landmarker_task = model.get('pose_landmarker_task')
    if pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': model.pose_landmarker_task required (MediaPipe .task path)"
        )
    pose_landmarker_task = str(pose_landmarker_task)

    hand_landmarker_task = model.get('hand_landmarker_task')
    if hand_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': model.hand_landmarker_task required (MediaPipe .task path)"
        )
    hand_landmarker_task = str(hand_landmarker_task)

    w_pose = float(params.get('w_pose', 0.45))
    w_hands = float(params.get('w_hands', 0.40))
    w_feet = float(params.get('w_feet', 0.15))
    score_weight = float(params.get('score_weight', 1.0))

    if w_pose <= 0:
        raise ValueError(
            f"'{node_id}': w_pose must be greater then 0."
        )
    if w_hands <= 0:
        raise ValueError(
            f"'{node_id}': w_hands must be greater then 0."
        )
    if w_feet <= 0:
        raise ValueError(
            f"'{node_id}': w_feet must be greater then 0."
        )
    if score_weight <= 0:
        raise ValueError(
            f"'{node_id}': score_weight must be greater then 0."
        )

    return ScorerConfig(
        device=device,
        pose_landmarker_task=pose_landmarker_task,
        hand_landmarker_task=hand_landmarker_task,
        w_pose=w_pose,
        w_hands=w_hands,
        w_feet=w_feet,
        score_weight=score_weight
    )


def _chain_complete(valid: np.ndarray, *idxs: int) -> bool:
    """
    Return whether all landmarks in a chain are valid.

    Parameters
    ----------
    valid : np.ndarray
        Boolean validity mask with shape ``(33,)``.
    *idxs : int
        Landmark indices belonging to the chain.

    Returns
    -------
    bool
        True if all requested landmarks are valid.
    """
    return all(bool(valid[i]) for i in idxs)


def _ratio_plausibility_score(
    r: Optional[float],
    *,
    ideal: float = 1.0,
    tol_good: float = 0.35,
    tol_bad: float = 1.0,
) -> float:
    """
    Convert an anatomical ratio into a soft plausibility score.

    Parameters
    ----------
    r : float | None
        Observed ratio.
    ideal : float, optional
        Ideal central value.
    tol_good : float, optional
        Absolute deviation around ``ideal`` that still receives full score.
    tol_bad : float, optional
        Absolute deviation at or beyond which the score becomes zero.

    Returns
    -------
    float
        Score in ``[0, 1]``.
    """
    if r is None:
        return 0.5

    d = abs(r - ideal)

    if d <= tol_good:
        return 1.0

    if d >= tol_bad:
        return 0.0

    t = (d - tol_good) / (tol_bad - tol_good)
    return float(1.0 - t)


def _angle_plausibility_score(angle_deg: Optional[float]) -> float:
    """
    Convert a 3D joint angle into a coarse plausibility score.

    This function is intentionally conservative. It does not try to encode
    exact anatomical limits; it only penalizes clearly degenerate or unusable
    configurations.

    Parameters
    ----------
    angle_deg : float | None
        Joint angle in degrees.

    Returns
    -------
    float
        Score in ``[0, 1]``.
    """
    if angle_deg is None:
        return 0.3

    if angle_deg < 5.0:
        return 0.0

    if angle_deg < 15.0:
        return 0.4

    if angle_deg > 175.0:
        return 0.6

    return 1.0


def pose_plausibility_score(res: PoseLandmarksResult) -> dict[str, Any]:
    """
    Compute a structural plausibility score for a MediaPipe pose result.

    The score combines:
    - landmark coverage
    - chain completeness
    - world-space geometry plausibility
    - joint-angle sanity

    Parameters
    ----------
    res : PoseLandmarksResult
        Rich pose result produced by ``mp_pose_landmarks_full(...)``.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - ``score``: final pose score
        - ``scores``: subscore breakdown
        - ``features``: extracted geometric features
        - ``flags``: qualitative warnings
    """
    valid = res.valid
    strong = res.strong
    world_xyz = res.world_xyz

    flags: list[str] = []

    n = int(len(valid))

    core_ids = [0, 11, 12, 23, 24]
    upper_ids = [0, 11, 12, 13, 14, 15, 16]
    lower_ids = [23, 24, 25, 26, 27, 28]

    valid_joint_ratio = float(np.count_nonzero(valid) / n)
    strong_joint_ratio = float(np.count_nonzero(strong) / n)
    core_valid_ratio = float(np.mean(valid[core_ids]))
    upper_body_valid_ratio = float(np.mean(valid[upper_ids]))
    lower_body_valid_ratio = float(np.mean(valid[lower_ids]))

    left_arm_chain_complete = _chain_complete(valid, 11, 13, 15)
    right_arm_chain_complete = _chain_complete(valid, 12, 14, 16)
    left_leg_chain_complete = _chain_complete(valid, 23, 25, 27)
    right_leg_chain_complete = _chain_complete(valid, 24, 26, 28)

    chain_complete_ratio = float(np.mean([
        left_arm_chain_complete,
        right_arm_chain_complete,
        left_leg_chain_complete,
        right_leg_chain_complete,
    ]))

    coverage_score = clamp01(
        0.4 * strong_joint_ratio +
        0.3 * core_valid_ratio +
        0.15 * upper_body_valid_ratio +
        0.15 * lower_body_valid_ratio
    )

    chain_score = clamp01(chain_complete_ratio)

    left_upper_arm = segment_len_3d(world_xyz, valid, 11, 13)
    left_lower_arm = segment_len_3d(world_xyz, valid, 13, 15)
    right_upper_arm = segment_len_3d(world_xyz, valid, 12, 14)
    right_lower_arm = segment_len_3d(world_xyz, valid, 14, 16)

    left_upper_leg = segment_len_3d(world_xyz, valid, 23, 25)
    left_lower_leg = segment_len_3d(world_xyz, valid, 25, 27)
    right_upper_leg = segment_len_3d(world_xyz, valid, 24, 26)
    right_lower_leg = segment_len_3d(world_xyz, valid, 26, 28)

    shoulder_width = segment_len_3d(world_xyz, valid, 11, 12)
    hip_width = segment_len_3d(world_xyz, valid, 23, 24)

    left_arm_total = (
        None if left_upper_arm is None or left_lower_arm is None
        else left_upper_arm + left_lower_arm
    )
    right_arm_total = (
        None if right_upper_arm is None or right_lower_arm is None
        else right_upper_arm + right_lower_arm
    )
    left_leg_total = (
        None if left_upper_leg is None or left_lower_leg is None
        else left_upper_leg + left_lower_leg
    )
    right_leg_total = (
        None if right_upper_leg is None or right_lower_leg is None
        else right_upper_leg + right_lower_leg
    )

    left_arm_ratio = safe_ratio(left_upper_arm, left_lower_arm)
    right_arm_ratio = safe_ratio(right_upper_arm, right_lower_arm)
    left_leg_ratio = safe_ratio(left_upper_leg, left_lower_leg)
    right_leg_ratio = safe_ratio(right_upper_leg, right_lower_leg)

    arm_symmetry_ratio = safe_ratio(left_arm_total, right_arm_total)
    leg_symmetry_ratio = safe_ratio(left_leg_total, right_leg_total)
    shoulder_hip_ratio = safe_ratio(shoulder_width, hip_width)

    geometry_components = [
        _ratio_plausibility_score(left_arm_ratio),
        _ratio_plausibility_score(right_arm_ratio),
        _ratio_plausibility_score(left_leg_ratio),
        _ratio_plausibility_score(right_leg_ratio),
        _ratio_plausibility_score(arm_symmetry_ratio),
        _ratio_plausibility_score(leg_symmetry_ratio),
        _ratio_plausibility_score(
            shoulder_hip_ratio,
            ideal=1.2,
            tol_good=0.5,
            tol_bad=1.5,
        ),
    ]
    geometry_score = clamp01(float(np.mean(geometry_components)))

    elbow_left_angle = joint_angle_3d(world_xyz, valid, 11, 13, 15)
    elbow_right_angle = joint_angle_3d(world_xyz, valid, 12, 14, 16)
    knee_left_angle = joint_angle_3d(world_xyz, valid, 23, 25, 27)
    knee_right_angle = joint_angle_3d(world_xyz, valid, 24, 26, 28)

    angle_components = [
        _angle_plausibility_score(elbow_left_angle),
        _angle_plausibility_score(elbow_right_angle),
        _angle_plausibility_score(knee_left_angle),
        _angle_plausibility_score(knee_right_angle),
    ]
    joint_angle_score = clamp01(float(np.mean(angle_components)))

    world_segments = [
        left_upper_arm,
        left_lower_arm,
        right_upper_arm,
        right_lower_arm,
        left_upper_leg,
        left_lower_leg,
        right_upper_leg,
        right_lower_leg,
        shoulder_width,
        hip_width,
    ]
    world_coverage_ratio = float(np.mean([
        seg is not None for seg in world_segments
    ]))

    geometry_reliability = clamp01(
        0.4 * strong_joint_ratio +
        0.3 * chain_complete_ratio +
        0.3 * world_coverage_ratio
    )

    pose_score = clamp01(
        0.35 * coverage_score +
        0.25 * chain_score +
        0.30 * geometry_reliability * geometry_score +
        0.10 * joint_angle_score
    )

    if coverage_score < 0.4:
        flags.append('low_landmark_coverage')

    if core_valid_ratio < 0.6:
        flags.append('missing_core_anchors')

    if chain_complete_ratio < 0.5:
        flags.append('incomplete_joint_chains')

    if world_coverage_ratio < 0.5:
        flags.append('poor_world_geometry_coverage')

    if geometry_score < 0.4:
        flags.append('implausible_body_geometry')

    return {
        'score': pose_score,
        'scores': {
            'coverage_score': coverage_score,
            'chain_score': chain_score,
            'geometry_score': geometry_score,
            'geometry_reliability': geometry_reliability,
            'joint_angle_score': joint_angle_score,
        },
        'features': {
            'valid_joint_ratio': valid_joint_ratio,
            'strong_joint_ratio': strong_joint_ratio,
            'core_valid_ratio': core_valid_ratio,
            'upper_body_valid_ratio': upper_body_valid_ratio,
            'lower_body_valid_ratio': lower_body_valid_ratio,
            'chain_complete_ratio': chain_complete_ratio,
            'world_coverage_ratio': world_coverage_ratio,
            'left_arm_chain_complete': left_arm_chain_complete,
            'right_arm_chain_complete': right_arm_chain_complete,
            'left_leg_chain_complete': left_leg_chain_complete,
            'right_leg_chain_complete': right_leg_chain_complete,
            'left_arm_ratio': left_arm_ratio,
            'right_arm_ratio': right_arm_ratio,
            'left_leg_ratio': left_leg_ratio,
            'right_leg_ratio': right_leg_ratio,
            'arm_symmetry_ratio': arm_symmetry_ratio,
            'leg_symmetry_ratio': leg_symmetry_ratio,
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'left_upper_arm_len_3d': left_upper_arm,
            'left_lower_arm_len_3d': left_lower_arm,
            'right_upper_arm_len_3d': right_upper_arm,
            'right_lower_arm_len_3d': right_lower_arm,
            'left_upper_leg_len_3d': left_upper_leg,
            'left_lower_leg_len_3d': left_lower_leg,
            'right_upper_leg_len_3d': right_upper_leg,
            'right_lower_leg_len_3d': right_lower_leg,
            'shoulder_width_3d': shoulder_width,
            'hip_width_3d': hip_width,
            'elbow_left_angle_3d': elbow_left_angle,
            'elbow_right_angle_3d': elbow_right_angle,
            'knee_left_angle_3d': knee_left_angle,
            'knee_right_angle_3d': knee_right_angle,
        },
        'flags': flags,
    }


def hand_plausibility_score(res: PoseLandmarksResult) -> dict[str, Any]:
    """
    Compute a refined plausibility score for detected hands.

    The scorer separates:
    - observability: whether the hand is sufficiently visible to be judged
    - topology: whether finger chains and overall structure are readable
    - geometry: whether coarse proportions and articulation look plausible

    Palm-based geometric checks are downweighted when the palm is poorly
    observed, for example when the hand is seen edge-on, strongly foreshortened,
    or partially resting on a surface.

    Hands that are not observable do not contribute to the final hand score.

    Parameters
    ----------
    res : HandLandmarksResult
        Rich MediaPipe hand result.

    Returns
    -------
    dict[str, Any]
        Dictionary containing final score, observability, score breakdown,
        extracted features, and flags.
    """
    valid = res.valid
    world_xyz = res.world_xyz
    xy = res.xy
    handedness = res.handedness

    flags: list[str] = []

    if valid.shape[0] == 0:
        return {
            'score': 0.0,
            'observable': False,
            'scores': {},
            'features': {},
            'flags': ['hands_not_detected'],
        }

    finger_chains = {
        'thumb': (1, 2, 3, 4),
        'index': (5, 6, 7, 8),
        'middle': (9, 10, 11, 12),
        'ring': (13, 14, 15, 16),
        'pinky': (17, 18, 19, 20),
    }

    fingertip_ids = [4, 8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]

    hand_entries = []
    observable_scores = []

    for h in range(valid.shape[0]):
        hv = valid[h]
        hw = world_xyz[h]
        hxy = xy[h]

        wrist_ok = bool(hv[0])
        mcp_valid_ratio = float(np.mean([hv[i] for i in mcp_ids]))
        fingertip_valid_ratio = float(np.mean([hv[i] for i in fingertip_ids]))
        valid_ratio = float(np.count_nonzero(hv) / len(hv))

        # ---------------------------------------------------------
        # Finger chain completeness
        # ---------------------------------------------------------
        finger_chain_scores = {
            name: float(all(bool(hv[i]) for i in chain))
            for name, chain in finger_chains.items()
        }
        chain_complete_ratio = float(
            np.mean(list(finger_chain_scores.values()))
        )

        # ---------------------------------------------------------
        # Image-space observability
        # ---------------------------------------------------------
        valid_xy = hxy[hv]
        if valid_xy.shape[0] >= 3:
            x1 = int(np.min(valid_xy[:, 0]))
            y1 = int(np.min(valid_xy[:, 1]))
            x2 = int(np.max(valid_xy[:, 0])) + 1
            y2 = int(np.max(valid_xy[:, 1])) + 1
            bbox_w = max(0, x2 - x1)
            bbox_h = max(0, y2 - y1)
            bbox_area = float(bbox_w * bbox_h)
        else:
            bbox_area = 0.0

        bbox_area_score = 1.0 if bbox_area >= 400.0 else float(
            bbox_area / 400.0)

        observability_score = clamp01(
            0.30 * valid_ratio +
            0.20 * float(wrist_ok) +
            0.20 * mcp_valid_ratio +
            0.20 * fingertip_valid_ratio +
            0.10 * bbox_area_score
        )

        observable = bool(
            observability_score >= 0.45 and
            wrist_ok and
            mcp_valid_ratio >= 0.50
        )

        # ---------------------------------------------------------
        # Palm geometry
        # ---------------------------------------------------------
        palm_width = segment_len_3d(hw, hv, 5, 17)

        index_len = (
            (segment_len_3d(hw, hv, 5, 6) or 0.0) +
            (segment_len_3d(hw, hv, 6, 7) or 0.0) +
            (segment_len_3d(hw, hv, 7, 8) or 0.0)
        )
        middle_len = (
            (segment_len_3d(hw, hv, 9, 10) or 0.0) +
            (segment_len_3d(hw, hv, 10, 11) or 0.0) +
            (segment_len_3d(hw, hv, 11, 12) or 0.0)
        )
        thumb_len = (
            (segment_len_3d(hw, hv, 1, 2) or 0.0) +
            (segment_len_3d(hw, hv, 2, 3) or 0.0) +
            (segment_len_3d(hw, hv, 3, 4) or 0.0)
        )

        index_to_palm = safe_ratio(
            index_len if index_len > 0 else None,
            palm_width,
        )
        middle_to_palm = safe_ratio(
            middle_len if middle_len > 0 else None,
            palm_width,
        )
        thumb_to_palm = safe_ratio(
            thumb_len if thumb_len > 0 else None,
            palm_width,
        )

        raw_palm_geometry_score = float(np.mean([
            _ratio_plausibility_score(
                index_to_palm,
                ideal=1.0,
                tol_good=0.5,
                tol_bad=1.6,
            ),
            _ratio_plausibility_score(
                middle_to_palm,
                ideal=1.2,
                tol_good=0.5,
                tol_bad=1.8,
            ),
            _ratio_plausibility_score(
                thumb_to_palm,
                ideal=0.8,
                tol_good=0.4,
                tol_bad=1.4,
            ),
        ]))

        # ---------------------------------------------------------
        # Finger / palm spread
        # ---------------------------------------------------------
        fingertip_pts = [
            hw[i] for i in fingertip_ids
            if hv[i] and not np.isnan(hw[i]).any()
        ]
        mcp_pts = [
            hw[i] for i in mcp_ids
            if hv[i] and not np.isnan(hw[i]).any()
        ]

        def _mean_pairwise_dist(points: list[np.ndarray]) -> Optional[float]:
            if len(points) < 2:
                return None

            ds = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    ds.append(float(np.linalg.norm(points[j] - points[i])))

            if not ds:
                return None

            return float(np.mean(ds))

        fingertip_spread = _mean_pairwise_dist(fingertip_pts)
        mcp_spread = _mean_pairwise_dist(mcp_pts)

        fingertip_spread_ratio = safe_ratio(fingertip_spread, palm_width)
        mcp_spread_ratio = safe_ratio(mcp_spread, palm_width)

        spread_score = float(np.mean([
            _ratio_plausibility_score(
                fingertip_spread_ratio,
                ideal=1.2,
                tol_good=0.6,
                tol_bad=2.0,
            ),
            _ratio_plausibility_score(
                mcp_spread_ratio,
                ideal=0.8,
                tol_good=0.5,
                tol_bad=1.8,
            ),
        ]))

        # ---------------------------------------------------------
        # Palm observability / reliability
        # ---------------------------------------------------------
        # If MCP spread is weak or palm width is unstable, palm-based ratios
        # are less trustworthy. In such cases we soften their contribution
        # toward a neutral score instead of treating them as hard evidence
        # of implausible geometry.
        palm_geometry_reliability = clamp01(
            0.45 * mcp_valid_ratio +
            0.35 * clamp01(spread_score) +
            0.20 * float(palm_width is not None)
        )

        palm_geometry_score = (
            palm_geometry_reliability * raw_palm_geometry_score +
            (1.0 - palm_geometry_reliability) * 0.5
        )

        # ---------------------------------------------------------
        # Joint-angle sanity
        # ---------------------------------------------------------
        angle_values = [
            joint_angle_3d(hw, hv, 5, 6, 7),
            joint_angle_3d(hw, hv, 6, 7, 8),
            joint_angle_3d(hw, hv, 9, 10, 11),
            joint_angle_3d(hw, hv, 10, 11, 12),
            joint_angle_3d(hw, hv, 13, 14, 15),
            joint_angle_3d(hw, hv, 14, 15, 16),
            joint_angle_3d(hw, hv, 17, 18, 19),
            joint_angle_3d(hw, hv, 18, 19, 20),
        ]
        angle_scores = [
            0.4 if a is None else (0.0 if a < 5.0 else 1.0)
            for a in angle_values
        ]
        joint_angle_score = float(np.mean(angle_scores))

        # ---------------------------------------------------------
        # Topology and geometry
        # ---------------------------------------------------------
        topology_score = clamp01(
            0.55 * chain_complete_ratio +
            0.25 * mcp_valid_ratio +
            0.20 * fingertip_valid_ratio
        )

        geometry_score = clamp01(
            0.35 * palm_geometry_score +
            0.35 * spread_score +
            0.30 * joint_angle_score
        )

        if observable:
            hand_score = clamp01(
                0.30 * observability_score +
                0.35 * topology_score +
                0.35 * geometry_score
            )
            observable_scores.append(hand_score)
        else:
            hand_score = 0.0

        local_flags = []

        if not observable:
            local_flags.append('hand_not_observable')

        if topology_score < 0.4:
            local_flags.append('poor_hand_topology')

        if spread_score < 0.35:
            local_flags.append('collapsed_finger_spread')

        if palm_geometry_reliability < 0.35:
            local_flags.append('low_palm_observability')
        elif raw_palm_geometry_score < 0.35:
            local_flags.append('suspected_implausible_palm_geometry')

        hand_entries.append({
            'handedness': handedness[h] if h < len(handedness) else 'Unknown',
            'observable': observable,
            'observability_score': observability_score,
            'topology_score': topology_score,
            'geometry_score': geometry_score,
            'score': hand_score,
            'features': {
                'valid_ratio': valid_ratio,
                'mcp_valid_ratio': mcp_valid_ratio,
                'fingertip_valid_ratio': fingertip_valid_ratio,
                'chain_complete_ratio': chain_complete_ratio,
                'bbox_area': bbox_area,
                'palm_width_3d': palm_width,
                'index_to_palm_ratio': index_to_palm,
                'middle_to_palm_ratio': middle_to_palm,
                'thumb_to_palm_ratio': thumb_to_palm,
                'raw_palm_geometry_score': raw_palm_geometry_score,
                'palm_geometry_reliability': palm_geometry_reliability,
                'palm_geometry_score': palm_geometry_score,
                'fingertip_spread_ratio': fingertip_spread_ratio,
                'mcp_spread_ratio': mcp_spread_ratio,
                'finger_chain_scores': finger_chain_scores,
            },
            'flags': local_flags,
        })

    if not observable_scores:
        return {
            'score': 0.0,
            'observable': False,
            'scores': {},
            'features': {
                'num_hands_detected': int(valid.shape[0]),
                'hands': hand_entries,
            },
            'flags': ['hands_not_observable'],
        }

    final_score = float(np.mean(observable_scores))

    for entry in hand_entries:
        flags.extend(entry['flags'])

    if final_score < 0.4:
        flags.append('implausible_hand_geometry')

    return {
        'score': clamp01(final_score),
        'observable': True,
        'scores': {
            'hand_score': clamp01(final_score),
        },
        'features': {
            'num_hands_detected': int(valid.shape[0]),
            'num_hands_observable': int(
                sum(1 for x in hand_entries if x['observable'])
            ),
            'hands': hand_entries,
        },
        'flags': flags,
    }


def foot_plausibility_score(res: PoseLandmarksResult) -> dict[str, Any]:
    """
    Compute a coarse plausibility score for visible feet using pose landmarks.

    The scorer is intentionally conservative because feet are often partially
    occluded, cropped, or covered by shoes. It focuses on three aspects:

    - scale of the foot relative to the tibia
    - plausibility of the foot attachment at the ankle
    - coarse orientation of the foot relative to the lower leg

    Parameters
    ----------
    res : PoseLandmarksResult
        Rich pose result produced by ``mp_pose_landmarks_full(...)``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``score``: aggregated foot score
        - ``observable``: whether at least one foot is observable
        - ``scores``: score breakdown
        - ``features``: extracted features
        - ``flags``: qualitative warnings
    """
    valid = res.valid
    world_xyz = res.world_xyz

    flags: list[str] = []

    sides = {
        'left': (25, 27, 29, 31),   # knee, ankle, heel, foot_index
        'right': (26, 28, 30, 32),
    }

    foot_entries = []

    for side, (knee, ankle, heel, toe) in sides.items():
        foot_observable = all(bool(valid[i]) for i in (knee, ankle, heel, toe))

        if not foot_observable:
            foot_entries.append({
                'side': side,
                'observable': False,
                'score': None,
            })
            continue

        tibia_len = segment_len_3d(world_xyz, valid, knee, ankle)
        foot_len = segment_len_3d(world_xyz, valid, heel, toe)
        ankle_to_heel = segment_len_3d(world_xyz, valid, ankle, heel)
        ankle_to_toe = segment_len_3d(world_xyz, valid, ankle, toe)

        foot_to_tibia_ratio = safe_ratio(foot_len, tibia_len)
        ankle_to_heel_ratio = safe_ratio(ankle_to_heel, tibia_len)
        ankle_to_toe_ratio = safe_ratio(ankle_to_toe, tibia_len)

        # Coarse orientation: angle between lower leg direction and foot axis.
        # We do not impose a narrow anatomical prior; we only penalize clearly
        # degenerate or implausible configurations.
        tibia_foot_angle = joint_angle_3d(world_xyz, valid, knee, ankle, toe)

        foot_scale_score = _ratio_plausibility_score(
            foot_to_tibia_ratio,
            ideal=0.45,
            tol_good=0.20,
            tol_bad=0.80,
        )

        foot_attachment_score = float(np.mean([
            _ratio_plausibility_score(
                ankle_to_heel_ratio,
                ideal=0.20,
                tol_good=0.15,
                tol_bad=0.50,
            ),
            _ratio_plausibility_score(
                ankle_to_toe_ratio,
                ideal=0.35,
                tol_good=0.20,
                tol_bad=0.90,
            ),
        ]))

        foot_orientation_score = _angle_plausibility_score(tibia_foot_angle)

        foot_score = clamp01(float(np.mean([
            foot_scale_score,
            foot_attachment_score,
            foot_orientation_score,
        ])))

        foot_entries.append({
            'side': side,
            'observable': True,
            'tibia_len_3d': tibia_len,
            'foot_len_3d': foot_len,
            'ankle_to_heel_len_3d': ankle_to_heel,
            'ankle_to_toe_len_3d': ankle_to_toe,
            'foot_to_tibia_ratio': foot_to_tibia_ratio,
            'ankle_to_heel_ratio': ankle_to_heel_ratio,
            'ankle_to_toe_ratio': ankle_to_toe_ratio,
            'tibia_foot_angle_3d': tibia_foot_angle,
            'scores': {
                'foot_scale_score': foot_scale_score,
                'foot_attachment_score': foot_attachment_score,
                'foot_orientation_score': foot_orientation_score,
            },
            'score': foot_score,
        })

    observable_scores = [
        x['score']
        for x in foot_entries
        if x['observable'] and x['score'] is not None
    ]

    if not observable_scores:
        return {
            'score': 0.0,
            'observable': False,
            'scores': {},
            'features': {
                'feet': foot_entries,
            },
            'flags': ['feet_not_observable'],
        }

    score = float(np.mean(observable_scores))

    if score < 0.4:
        flags.append('implausible_foot_geometry')

    return {
        'score': score,
        'observable': True,
        'scores': {
            'foot_score': score,
        },
        'features': {
            'feet': foot_entries,
        },
        'flags': flags,
    }


def _aggregate_person_score(
    *,
    pose_score_res: dict[str, Any],
    hand_score_res: dict[str, Any],
    foot_score_res: dict[str, Any],
    cfg: ScorerConfig,
) -> dict[str, Any]:
    flags: list[str] = []
    flags.extend(pose_score_res.get('flags', []))
    flags.extend(hand_score_res.get('flags', []))
    flags.extend(foot_score_res.get('flags', []))

    weighted = []
    total_w = 0.0

    weighted.append(cfg.w_pose * float(pose_score_res['score']))
    total_w += cfg.w_pose

    if hand_score_res.get('observable', False):
        weighted.append(cfg.w_hands * float(hand_score_res['score']))
        total_w += cfg.w_hands

    if foot_score_res.get('observable', False):
        weighted.append(cfg.w_feet * float(foot_score_res['score']))
        total_w += cfg.w_feet

    if total_w <= 1e-8:
        final_score = 0.0
    else:
        final_score = float(sum(weighted) / total_w)

    return {
        'score': clamp01(final_score),
        'weights': {
            'w_pose': cfg.w_pose,
            'w_hands': cfg.w_hands if hand_score_res.get('observable', False) else 0.0,
            'w_feet': cfg.w_feet if foot_score_res.get('observable', False) else 0.0,
        },
        'flags': flags,
    }


@dataclass
class PersonScorer(ImgWiringMixin, NodeRef):
    """
    Evaluate and rank one or more person images based on structural plausibility.

    ``PersonScorer`` analyzes candidate images containing a person and assigns
    each image a score reflecting how structurally plausible it is. The score
    combines pose, hand, and foot plausibility using simple, inspectable
    heuristics.

    The node is designed to operate both as:
    - a standalone selector over multiple generated images
    - a stage in a chain of scorer nodes contributing to a global ranking

    Input wiring
    ------------
    Candidate images are attached explicitly via repeated calls to :meth:`image`.

    Each call to ``image()`` creates a new dynamic input sink:

    - first call  -> ``image:0``
    - second call -> ``image:1``
    - third call  -> ``image:2``
    - ...

    Example:

    >>> scorer = PersonScorer(id='best_person', spec=...)
    >>> node_a >> scorer.image()
    >>> node_b >> scorer.image()
    >>> node_c >> scorer.image()

    Each upstream input must provide either:

    - ``'image'`` : path to an image file
    - or ``'path'`` : path to an image file

    Chained scorers
    ---------------
    If upstream nodes already produced a ``rankings`` structure (i.e. they are
    scorer nodes), this node will:

    - reuse the candidate image set from upstream
    - append its own score contribution to each image
    - recompute the final ranking using all accumulated scores

    In this case, explicit ``image()`` wiring is not required.

    Scoring pipeline
    ----------------
    For each candidate image:

    1. Pose extraction
        Full-body landmarks are extracted using MediaPipe Pose.

    2. Pose scoring
        A plausibility score is computed based on:
        - landmark coverage
        - chain completeness
        - 3D geometric consistency
        - joint angle sanity

    3. Hand scoring
        Hand landmarks are extracted using MediaPipe Hands and evaluated for:
        - finger topology consistency
        - coarse geometric plausibility

        Hands are optional: they contribute only if sufficiently observable.

    4. Foot scoring
        Feet are approximated from pose landmarks (ankle / heel / toe geometry).

        Feet are optional and contribute only if pose is available and the
        relevant landmarks are visible.

    5. Aggregation
        The final person score is computed as a weighted mean of the observable
        components:

            score = weighted_mean(pose, hands?, feet?)

        Components that are not observable are excluded from the aggregation.

    Ranking logic
    -------------
    Each node contributes a score with an associated ``score_weight``.

    The final ranking score for each image is:

        final_score = weighted_mean(all_node_scores)

    where:
        - each scorer node contributes one score
        - each score is weighted by its ``score_weight``

    This allows multiple scorer nodes (e.g. person, face, style) to be chained
    together into a single coherent ranking.

    Outputs
    -------
    dict
        Dictionary containing:

        ``image`` : str
            Path of the best-ranked image.

        ``best_score`` : float
            Final aggregated score of the best image.

        ``rankings`` : dict[str, list[dict]]
            Mapping from node id to score contributions. Each entry is a list of:

                - ``image`` : str
                - ``score`` : float
                - ``score_weight`` : float
                - ``flags`` : list[str], optional
                - ``error`` : str, optional

            Each list represents the contributions of a single scorer node in
            the chain.

        ``results`` : list[dict]
            Per-candidate detailed results for this node. Each entry contains:

                - ``input_image`` : str
                - ``score`` : float
                - ``pose`` : dict
                - ``hands`` : dict
                - ``feet`` : dict
                - ``aggregation`` : dict
                - ``flags`` : list[str]

            Each sub-component includes its own ``observable`` flag, features,
            breakdown, and flags.

    Parameters
    ----------
    spec : dict or str or Path, optional
        Node configuration resolved via ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``pose_landmarker_task`` : str
                Path to MediaPipe PoseLandmarker ``.task`` model.

            ``hand_landmarker_task`` : str
                Path to MediaPipe HandLandmarker ``.task`` model.

            ``device`` : str, optional
                Logical device string used by the model cache.

        ``params`` : dict
            ``w_pose`` : float, optional
                Weight of pose score in local aggregation.

            ``w_hands`` : float, optional
                Weight of hand score when observable.

            ``w_feet`` : float, optional
                Weight of foot score when observable.

            ``score_weight`` : float, optional
                Weight of this node in cross-node ranking aggregation.

    Notes
    -----
    - This node does not generate images; it selects among existing candidates.
    - Hands and feet are treated as optional signals to avoid penalizing images
      where they are occluded or out of frame.
    - Observability is defined at the component level; the aggregated score
      reflects only the available evidence.
    - The scoring system is heuristic and intended for relative ranking, not
      strict anatomical validation.

    Design rationale
    ----------------
    ``PersonScorer`` separates generation from evaluation:

    - upstream nodes generate multiple candidates
    - this node evaluates them using explicit heuristics
    - multiple scorer nodes can be chained to refine selection

    This makes selection reproducible, debuggable, and incrementally
    improvable over time.
    """

    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        hand_landmarker = get_mediapipe_hand_landmarker(
            model_asset_path=cfg.hand_landmarker_task,
            device=cfg.device,
        )
        candidates: list[dict[str, Any]] = []

        bundle = ImgBundle(
            node_id=self.id,
            img_count=self.image.img_count,
            input=input
        )

        for path in bundle.paths:
            with Image.open(path) as im:
                img_rgb = np.asarray(im.convert('RGB'))

            try:
                pose_res = mp_pose_landmarks_full(
                    img_rgb=img_rgb,
                    pose_landmarker=pose_landmarker,
                )
                pose_score_res = pose_plausibility_score(pose_res)
                foot_score_res = foot_plausibility_score(pose_res)
            except RuntimeError:
                pose_score_res = {
                    'score': 0.0,
                    'observable': False,
                    'scores': {},
                    'features': {},
                    'flags': ['pose_not_detected'],
                }
                foot_score_res = {
                    'score': 0.0,
                    'observable': False,
                    'scores': {},
                    'features': {},
                    'flags': ['feet_not_observable_due_to_pose'],
                }

            try:
                hand_res = mp_hand_landmarks_full(
                    img_rgb=img_rgb,
                    hand_landmarker=hand_landmarker,
                )
                hand_score_res = hand_plausibility_score(hand_res)
            except RuntimeError:
                hand_score_res = {
                    'score': 0.0,
                    'observable': False,
                    'scores': {},
                    'features': {},
                    'flags': ['hands_not_detected'],
                }

            agg = _aggregate_person_score(
                pose_score_res=pose_score_res,
                hand_score_res=hand_score_res,
                foot_score_res=foot_score_res,
                cfg=cfg,
            )

            candidate = {
                'input_image': path,
                'score': float(agg['score']),
                'pose': pose_score_res,
                'hands': hand_score_res,
                'feet': foot_score_res,
                'aggregation': agg,
                'flags': agg['flags'],
            }

            candidates.append(candidate)

        rankings = bundle.build_rankings(
            candidates=candidates,
            score_weight=cfg.score_weight,
        )

        best_image, best_score = bundle.calculate_best_image(rankings)

        if best_score <= 0.0 or best_image is None:
            raise RuntimeError(
                f"{self.id}: all candidate images failed scoring."
            )

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'image': best_image,
            'best_score': best_score,
            'model': {
                'device': cfg.device,
                'pose_landmarker_task': cfg.pose_landmarker_task,
                'hand_landmarker_task': cfg.hand_landmarker_task,
            },
            'params': {
                'w_pose': cfg.w_pose,
                'w_hands': cfg.w_hands,
                'w_feet': cfg.w_feet,
                'score_weight': cfg.score_weight,
            },
            'rankings': rankings,
            'results': candidates,
        }

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='json',
        )
        meta_path = write_json(out_path, out)
        out['metadata'] = str(meta_path)

        return out
