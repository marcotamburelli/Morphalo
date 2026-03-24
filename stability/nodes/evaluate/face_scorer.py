from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stability.cache.models import (get_insightface,
                                    get_mediapipe_face_landmarker,
                                    get_mediapipe_pose_landmarker)
from stability.core.paths import make_node_output_path
from stability.dag.core import AttachmentSink, NodeRef
from stability.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                   resolve_spec)
from stability.nodes.common.io import load_faceid_embeds, write_json
from stability.nodes.evaluate.helper import clamp01
from stability.nodes.evaluate.img_wiring_mixin import ImgBundle, ImgWiringMixin
from stability.nodes.vision.face_region import (
    face_landmarks_from_pose_guided_head_area, square_head_bbox_from_face_bbox)
from stability.nodes.vision.human import (eye_bbox_xyxy_from_landmarks,
                                          face_bbox_xyxy_from_landmarks,
                                          mp_face_landmarks,
                                          mp_pose_landmarks_full)


@dataclass
class ScorerConfig:
    device: str
    dtype: str
    face_landmarker_task: str
    pose_landmarker_task: str
    face_region_expansion: float
    insightface_name: str
    insightface_det_size: tuple[int, int]
    w_landmarks: float
    w_eyes: float
    w_size: float
    w_identity: float
    score_weight: float


def _read_cfg(spec: dict, node_id: str) -> ScorerConfig:
    model = spec.get('model', {})
    params = spec.get('params', {})

    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))

    face_landmarker_task = model.get('face_landmarker_task')
    if face_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': model.face_landmarker_task required (MediaPipe .task path)"
        )

    pose_landmarker_task = model.get('pose_landmarker_task')
    if pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': model.pose_landmarker_task required (MediaPipe .task path)"
        )

    face_region_expansion = float(params.get('face_region_expansion', 1.6))
    if face_region_expansion <= 0:
        raise ValueError(
            f"'{node_id}': face_region_expansion must be > 0."
        )

    w_landmarks = float(params.get('w_landmarks', 1.0))
    w_eyes = float(params.get('w_eyes', 0.75))
    w_size = float(params.get('w_size', 0.25))
    w_identity = float(params.get('w_identity', 1.0))
    score_weight = float(params.get('score_weight', 1.0))

    for name, value in [
        ('w_landmarks', w_landmarks),
        ('w_eyes', w_eyes),
        ('w_size', w_size),
        ('w_identity', w_identity),
        ('score_weight', score_weight),
    ]:
        if value <= 0:
            raise ValueError(f"'{node_id}': {name} must be > 0.")

    insightface_name = str(model.get('insightface_name', 'buffalo_l'))
    det_size = model.get('insightface_det_size', [640, 640])
    if not isinstance(det_size, (list, tuple)) or len(det_size) != 2:
        raise ValueError(
            f"'{node_id}': model.insightface_det_size must be a pair [w, h]."
        )

    return ScorerConfig(
        device=device,
        dtype=dtype,
        face_landmarker_task=str(face_landmarker_task),
        pose_landmarker_task=str(pose_landmarker_task),
        face_region_expansion=face_region_expansion,
        insightface_name=insightface_name,
        insightface_det_size=(int(det_size[0]), int(det_size[1])),
        w_landmarks=w_landmarks,
        w_eyes=w_eyes,
        w_size=w_size,
        w_identity=w_identity,
        score_weight=score_weight,
    )


def face_identity_similarity(
    face_embed: torch.Tensor,
    ref_embeds: torch.Tensor,
) -> float:
    """
    Compute similarity between a face embedding and a set of reference embeddings.

    Parameters
    ----------
    face_embed : torch.Tensor
        Shape (D,)
    ref_embeds : torch.Tensor
        Shape (2, N, D)

    Returns
    -------
    float
        Max cosine similarity with positive reference embeddings.
    """
    if ref_embeds.ndim != 3 or ref_embeds.shape[0] != 2:
        raise ValueError(f'Invalid ref_embeds shape: {ref_embeds.shape}')

    refs = ref_embeds[1]  # (N, D)

    face_embed = F.normalize(face_embed, dim=-1)
    refs = F.normalize(refs, dim=-1)

    sims = torch.matmul(refs, face_embed)  # (N,)
    return float(torch.max(sims).item())


def face_landmark_plausibility_score(
    face_xy: np.ndarray,
    *,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    """
    Compute a coarse plausibility score for face landmarks.

    The scorer is intentionally heuristic and ranking-oriented. It checks:

    - whether enough landmarks are present
    - whether the derived face box is non-degenerate
    - whether the face occupies a plausible image area
    - whether the basic vertical ordering of facial regions is sensible
    - whether the face appears roughly symmetric around its center

    Parameters
    ----------
    face_xy : np.ndarray
        Face landmarks in pixel coordinates with shape ``(N, 2)``.
    image_shape : tuple[int, ...]
        Full image shape, typically ``(H, W, 3)``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing score, observability, breakdown, features, and flags.
    """
    flags: list[str] = []

    if face_xy.ndim != 2 or face_xy.shape[1] != 2 or face_xy.shape[0] < 50:
        return {
            'score': 0.0,
            'observable': False,
            'breakdown': {},
            'features': {},
            'flags': ['invalid_face_landmarks'],
        }

    h, w = image_shape[:2]

    x1 = int(np.min(face_xy[:, 0]))
    y1 = int(np.min(face_xy[:, 1]))
    x2 = int(np.max(face_xy[:, 0])) + 1
    y2 = int(np.max(face_xy[:, 1])) + 1

    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    face_area = float(face_w * face_h)
    img_area = float(max(1, h * w))
    area_ratio = face_area / img_area
    aspect_ratio = float(face_w) / float(face_h)

    # MediaPipe canonical regions.
    left_eye_ids = np.arange(33, 133)
    right_eye_ids = np.arange(362, 463)
    nose_ids = np.arange(1, 6)
    mouth_ids = np.arange(61, 292)

    left_eye_y = float(np.mean(face_xy[left_eye_ids, 1]))
    right_eye_y = float(np.mean(face_xy[right_eye_ids, 1]))
    eye_y = 0.5 * (left_eye_y + right_eye_y)
    nose_y = float(np.mean(face_xy[nose_ids, 1]))
    mouth_y = float(np.mean(face_xy[mouth_ids, 1]))

    ordering_ok = float(eye_y < nose_y < mouth_y)
    if ordering_ok < 0.5:
        flags.append('face_feature_ordering_suspicious')

    face_cx = 0.5 * (x1 + x2)
    left_eye_cx = float(np.mean(face_xy[left_eye_ids, 0]))
    right_eye_cx = float(np.mean(face_xy[right_eye_ids, 0]))

    eye_dx_left = abs(face_cx - left_eye_cx)
    eye_dx_right = abs(right_eye_cx - face_cx)
    eye_sym_ratio = min(eye_dx_left, eye_dx_right) / max(
        1.0, max(eye_dx_left, eye_dx_right)
    )

    # Area plausibility:
    # - too tiny -> probably not a useful face candidate
    # - too huge -> suspicious / overly clipped
    if area_ratio < 0.005:
        flags.append('face_too_small')
    if area_ratio > 0.65:
        flags.append('face_too_large')

    area_score = 1.0
    if area_ratio < 0.005:
        area_score = area_ratio / 0.005
    elif area_ratio > 0.65:
        area_score = max(0.0, 1.0 - (area_ratio - 0.65) / 0.35)

    # Aspect plausibility: most faces should remain in a broad, portrait-ish range.
    if aspect_ratio < 0.55 or aspect_ratio > 1.35:
        flags.append('face_aspect_suspicious')

    if aspect_ratio < 0.55:
        aspect_score = max(0.0, aspect_ratio / 0.55)
    elif aspect_ratio > 1.35:
        aspect_score = max(0.0, 1.35 / aspect_ratio)
    else:
        aspect_score = 1.0

    # Vertical spacing sanity.
    eye_to_nose = max(0.0, nose_y - eye_y)
    nose_to_mouth = max(0.0, mouth_y - nose_y)
    spacing_ratio = min(eye_to_nose, nose_to_mouth) / max(
        1.0, max(eye_to_nose, nose_to_mouth)
    )
    if spacing_ratio < 0.35:
        flags.append('face_vertical_spacing_unbalanced')

    breakdown = {
        'area': clamp01(area_score),
        'aspect': clamp01(aspect_score),
        'ordering': clamp01(ordering_ok),
        'symmetry': clamp01(eye_sym_ratio),
        'spacing': clamp01(spacing_ratio),
    }

    # Slightly favor geometry over size.
    score = (
        0.20 * breakdown['area']
        + 0.20 * breakdown['aspect']
        + 0.20 * breakdown['ordering']
        + 0.20 * breakdown['symmetry']
        + 0.20 * breakdown['spacing']
    )

    return {
        'score': clamp01(score),
        'observable': True,
        'breakdown': breakdown,
        'features': {
            'face_bbox_xyxy': (x1, y1, x2, y2),
            'face_w': face_w,
            'face_h': face_h,
            'face_area_ratio': float(area_ratio),
            'aspect_ratio': float(aspect_ratio),
            'eye_y': float(eye_y),
            'nose_y': float(nose_y),
            'mouth_y': float(mouth_y),
            'eye_sym_ratio': float(eye_sym_ratio),
            'spacing_ratio': float(spacing_ratio),
        },
        'flags': flags,
    }


def eye_plausibility_score(
    face_xy: np.ndarray,
    *,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    """
    Compute a coarse plausibility score for the eye regions.

    The scorer derives left and right eye bounding boxes from face landmarks and
    evaluates:

    - presence of both eyes
    - minimum usable size
    - left/right size symmetry
    - horizontal separation relative to face width
    - rough vertical alignment

    Parameters
    ----------
    face_xy : np.ndarray
        Face landmarks in pixel coordinates with shape ``(N, 2)``.
    image_shape : tuple[int, ...]
        Full image shape, typically ``(H, W, 3)``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing score, observability, breakdown, features, and flags.
    """
    flags: list[str] = []

    try:
        left_box = eye_bbox_xyxy_from_landmarks(
            face_xy,
            image_shape,
            which='left',
            expansion=1.0,
        )
        right_box = eye_bbox_xyxy_from_landmarks(
            face_xy,
            image_shape,
            which='right',
            expansion=1.0,
        )
    except Exception as e:
        return {
            'score': 0.0,
            'observable': False,
            'breakdown': {},
            'features': {},
            'flags': [f'eye_bbox_failed: {e}'],
        }

    lx1, ly1, lx2, ly2 = left_box
    rx1, ry1, rx2, ry2 = right_box

    lw = max(1, lx2 - lx1)
    lh = max(1, ly2 - ly1)
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)

    l_area = float(lw * lh)
    r_area = float(rw * rh)
    area_ratio = min(l_area, r_area) / max(1.0, max(l_area, r_area))

    lcx = 0.5 * (lx1 + lx2)
    lcy = 0.5 * (ly1 + ly2)
    rcx = 0.5 * (rx1 + rx2)
    rcy = 0.5 * (ry1 + ry2)

    face_x1 = int(np.min(face_xy[:, 0]))
    face_x2 = int(np.max(face_xy[:, 0])) + 1
    face_y1 = int(np.min(face_xy[:, 1]))
    face_y2 = int(np.max(face_xy[:, 1])) + 1

    face_w = max(1, face_x2 - face_x1)
    face_h = max(1, face_y2 - face_y1)

    eye_dist = abs(rcx - lcx)
    eye_dist_ratio = eye_dist / max(1.0, float(face_w))

    vertical_align = 1.0 - min(1.0, abs(lcy - rcy) / max(1.0, 0.25 * face_h))

    # Tiny eyes usually mean face is too small or landmarks collapsed.
    min_eye_size = min(lw, lh, rw, rh)
    size_score = min(1.0, float(min_eye_size) / 8.0)
    if min_eye_size < 8:
        flags.append('eyes_too_small')

    # Healthy horizontal separation is broad but not extreme.
    if eye_dist_ratio < 0.20 or eye_dist_ratio > 0.70:
        flags.append('eye_separation_suspicious')

    if eye_dist_ratio < 0.20:
        sep_score = max(0.0, eye_dist_ratio / 0.20)
    elif eye_dist_ratio > 0.70:
        sep_score = max(0.0, 0.70 / eye_dist_ratio)
    else:
        sep_score = 1.0

    if area_ratio < 0.45:
        flags.append('eye_size_asymmetry')

    if vertical_align < 0.6:
        flags.append('eye_vertical_misalignment')

    breakdown = {
        'size': clamp01(size_score),
        'symmetry': clamp01(area_ratio),
        'separation': clamp01(sep_score),
        'alignment': clamp01(vertical_align),
    }

    score = (
        0.25 * breakdown['size']
        + 0.25 * breakdown['symmetry']
        + 0.25 * breakdown['separation']
        + 0.25 * breakdown['alignment']
    )

    return {
        'score': clamp01(score),
        'observable': True,
        'breakdown': breakdown,
        'features': {
            'left_eye_bbox_xyxy': left_box,
            'right_eye_bbox_xyxy': right_box,
            'left_eye_area': float(l_area),
            'right_eye_area': float(r_area),
            'eye_area_ratio': float(area_ratio),
            'eye_distance_px': float(eye_dist),
            'eye_distance_ratio': float(eye_dist_ratio),
            'eye_vertical_alignment': float(vertical_align),
        },
        'flags': flags,
    }


def face_size_score(
    face_bbox_xyxy: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    """
    Compute a utility score for face size and crop quality.

    This scorer is intentionally simple and ranking-oriented. It rewards faces
    that are:

    - large enough to be evaluated reliably
    - not excessively large relative to the frame
    - not strongly clipped by image borders

    Parameters
    ----------
    face_bbox_xyxy : tuple[int, int, int, int]
        Face bounding box ``(x1, y1, x2, y2)`` in full-image coordinates.
    image_shape : tuple[int, ...]
        Full image shape, typically ``(H, W, 3)``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing score, observability, breakdown, features, and flags.
    """
    flags: list[str] = []

    h, w = image_shape[:2]
    x1, y1, x2, y2 = face_bbox_xyxy

    if x2 <= x1 or y2 <= y1:
        return {
            'score': 0.0,
            'observable': False,
            'breakdown': {},
            'features': {},
            'flags': ['invalid_face_bbox'],
        }

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = float((bw * bh) / max(1, w * h))
    min_dim = float(min(bw, bh))

    # Practical size for a face scorer.
    min_dim_score = min(1.0, min_dim / 96.0)
    if min_dim < 96.0:
        flags.append('face_resolution_low')

    if area_ratio < 0.01:
        area_score = area_ratio / 0.01
        flags.append('face_frame_coverage_too_small')
    elif area_ratio > 0.45:
        area_score = max(0.0, 1.0 - (area_ratio - 0.45) / 0.35)
        flags.append('face_frame_coverage_too_large')
    else:
        area_score = 1.0

    border_hits = 0
    if x1 <= 0:
        border_hits += 1
    if y1 <= 0:
        border_hits += 1
    if x2 >= w:
        border_hits += 1
    if y2 >= h:
        border_hits += 1

    if border_hits >= 2:
        flags.append('face_heavily_clipped')
    elif border_hits == 1:
        flags.append('face_touches_border')

    border_score = {
        0: 1.0,
        1: 0.75,
        2: 0.45,
        3: 0.2,
        4: 0.0,
    }[min(4, border_hits)]

    breakdown = {
        'resolution': clamp01(min_dim_score),
        'coverage': clamp01(area_score),
        'border': clamp01(border_score),
    }

    score = (
        0.45 * breakdown['resolution']
        + 0.35 * breakdown['coverage']
        + 0.20 * breakdown['border']
    )

    return {
        'score': clamp01(score),
        'observable': True,
        'breakdown': breakdown,
        'features': {
            'face_bbox_xyxy': face_bbox_xyxy,
            'face_w': int(bw),
            'face_h': int(bh),
            'face_area_ratio': float(area_ratio),
            'border_hits': int(border_hits),
            'min_dim': float(min_dim),
        },
        'flags': flags,
    }


def identity_similarity_score(
    head_rgb: np.ndarray,
    *,
    insightface_app: Any,
    ref_embeds: Optional[torch.Tensor],
) -> dict[str, Any]:
    """
    Compute identity similarity against one or more reference FaceID embeddings.

    The score is based on the maximum cosine similarity between the candidate
    face embedding and the positive reference embeddings stored in
    ``ref_embeds[1]``.

    Parameters
    ----------
    head_rgb : np.ndarray
        Cropped head RGB image with shape ``(H, W, 3)``.
    insightface_app : Any
        Prepared InsightFace ``FaceAnalysis`` instance.
    ref_embeds : torch.Tensor or None
        Reference embeddings with shape ``(2, N, D)``, or ``None`` if identity
        matching is disabled.

    Returns
    -------
    dict[str, Any]
        Dictionary containing score, observability, features, and flags.
    """
    if ref_embeds is None:
        return {
            'score': 0.0,
            'observable': False,
            'breakdown': {},
            'features': {},
            'flags': ['no_identity_reference'],
        }

    if ref_embeds.ndim != 3 or ref_embeds.shape[0] < 2:
        raise ValueError(
            f'Invalid ref_embeds shape {tuple(ref_embeds.shape)!r}; expected (2, N, D).'
        )

    import cv2

    bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    faces = insightface_app.get(bgr)

    if not faces:
        return {
            'score': 0.0,
            'observable': False,
            'breakdown': {},
            'features': {},
            'flags': ['identity_face_not_detected'],
        }

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    cand_embed = torch.from_numpy(face.normed_embedding).to(
        device=ref_embeds.device,
        dtype=ref_embeds.dtype,
    )

    sim = float(face_identity_similarity(cand_embed, ref_embeds))

    # Conservative mapping:
    # negative similarities collapse to 0,
    # positive similarities stay in [0, 1].
    score = clamp01(max(0.0, sim))

    return {
        'score': score,
        'observable': True,
        'breakdown': {
            'similarity': score,
        },
        'features': {
            'raw_similarity': float(sim),
            'ref_count': int(ref_embeds.shape[1]),
            'bbox_xyxy': [float(v) for v in face.bbox],
        },
        'flags': [],
    }


def _aggregate_face_score(
    *,
    landmark_score_res: dict[str, Any],
    eye_score_res: dict[str, Any],
    size_score_res: dict[str, Any],
    identity_score_res: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    """
    Aggregate face sub-scores into a final face score.

    Only observable components contribute to the weighted mean.

    Parameters
    ----------
    landmark_score_res : dict[str, Any]
        Face landmark plausibility result.
    eye_score_res : dict[str, Any]
        Eye plausibility result.
    size_score_res : dict[str, Any]
        Face size / crop quality result.
    identity_score_res : dict[str, Any]
        Identity similarity result.
    cfg : Any
        Scorer config exposing:
        - ``w_landmarks``
        - ``w_eyes``
        - ``w_size``
        - ``w_identity``

    Returns
    -------
    dict[str, Any]
        Aggregated score result containing score, weights, and merged flags.
    """
    flags: list[str] = []
    flags.extend(landmark_score_res.get('flags', []))
    flags.extend(eye_score_res.get('flags', []))
    flags.extend(size_score_res.get('flags', []))
    flags.extend(identity_score_res.get('flags', []))

    weighted_scores: list[float] = []
    total_w = 0.0

    used_weights = {
        'w_landmarks': 0.0,
        'w_eyes': 0.0,
        'w_size': 0.0,
        'w_identity': 0.0,
    }

    if landmark_score_res.get('observable', False):
        weighted_scores.append(
            float(cfg.w_landmarks) * float(landmark_score_res['score'])
        )
        total_w += float(cfg.w_landmarks)
        used_weights['w_landmarks'] = float(cfg.w_landmarks)

    if eye_score_res.get('observable', False):
        weighted_scores.append(
            float(cfg.w_eyes) * float(eye_score_res['score'])
        )
        total_w += float(cfg.w_eyes)
        used_weights['w_eyes'] = float(cfg.w_eyes)

    if size_score_res.get('observable', False):
        weighted_scores.append(
            float(cfg.w_size) * float(size_score_res['score'])
        )
        total_w += float(cfg.w_size)
        used_weights['w_size'] = float(cfg.w_size)

    if identity_score_res.get('observable', False):
        weighted_scores.append(
            float(cfg.w_identity) * float(identity_score_res['score'])
        )
        total_w += float(cfg.w_identity)
        used_weights['w_identity'] = float(cfg.w_identity)

    final_score = 0.0 if total_w <= 1e-8 else sum(weighted_scores) / total_w

    return {
        'score': clamp01(float(final_score)),
        'weights': used_weights,
        'flags': flags,
    }


@dataclass
class FaceScorer(ImgWiringMixin, NodeRef):
    """
    Evaluate and rank one or more person images based on face plausibility and,
    optionally, identity similarity.

    ``FaceScorer`` analyzes candidate images containing a person and assigns each
    image a face-centered score reflecting:

    - facial landmark plausibility
    - eye-region plausibility
    - face size / usability within the frame
    - optional similarity to a reference identity embedding

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

    >>> scorer = FaceScorer(id='best_face', spec=...)
    >>> node_a >> scorer.image()
    >>> node_b >> scorer.image()
    >>> node_c >> scorer.image()

    Each upstream input must provide either:

    - ``'image'`` : path to an image file
    - or ``'path'`` : path to an image file

    Identity reference input
    ------------------------
    An optional identity reference can be attached via :meth:`identity`.

    Example:

    >>> ref_node >> scorer.identity()

    The upstream payload connected to ``identity()`` must provide either:

    - ``'embeds'`` : path to a FaceID embeddings file
    - or ``'path'`` : path to a FaceID embeddings file

    The embeddings are loaded through ``load_faceid_embeds(...)`` and are expected
    to follow the project convention with shape ``(2, N, D)``. If multiple
    reference files are provided, they are concatenated along the reference
    dimension.

    If no identity reference is provided, the node still works normally and simply
    disables the identity component of the score.

    Chained scorers
    ---------------
    If upstream nodes already produced a ``rankings`` structure, this node will:

    - reuse the candidate image set from upstream
    - append its own score contribution to each image
    - recompute the final ranking using all accumulated scores

    In this mode, explicit ``image()`` wiring is not required.

    Localization pipeline
    ---------------------
    For each candidate image, face analysis follows a two-stage localization
    strategy:

    1. Pose-guided head-area localization
        A coarse head / upper-body region is derived from pose landmarks and used
        as the preferred region for face landmark detection.

    2. Full-image fallback
        If pose-guided localization fails, face landmarks are detected directly on
        the full image.

    This makes the node more robust when the face is small, partially ambiguous, or
    embedded in a cluttered scene, while still preserving a reasonable fallback
    path.

    Scoring pipeline
    ----------------
    For each candidate image:

    1. Face landmark extraction
        Face landmarks are extracted from the pose-guided head area when possible,
        otherwise from the full image.

    2. Landmark plausibility scoring
        A face plausibility score is computed from coarse geometric heuristics,
        including:
        - face bounding-box plausibility
        - facial region ordering
        - rough symmetry
        - vertical spacing consistency

    3. Eye plausibility scoring
        Left and right eye regions are derived from face landmarks and evaluated
        for:
        - minimum usable size
        - size symmetry
        - horizontal separation
        - vertical alignment

    4. Face size scoring
        The face bounding box is evaluated as a practical utility signal, rewarding
        faces that are:
        - large enough to be evaluated reliably
        - not excessively large relative to the frame
        - not strongly clipped by image borders

    5. Identity similarity scoring (optional)
        If reference FaceID embeddings are available, a square head crop is derived
        from the detected face box and passed to the identity scorer.

        The identity score is based on the maximum similarity between the candidate
        embedding and the positive reference embeddings.

    6. Aggregation
        The final face score is computed as a weighted mean of the observable
        components:

            score = weighted_mean(landmarks, eyes?, size, identity?)

        Components that are not observable are excluded from the aggregation.

    Ranking logic
    -------------
    Each scorer node contributes a score with an associated ``score_weight``.

    The final ranking score for each image is computed by aggregating all score
    contributions accumulated along the scorer chain as a weighted mean.

    This allows multiple scorer nodes, such as:

    - ``PersonScorer``
    - ``FaceScorer``
    - ``PromptStyleScorer``

    to cooperate in a single coherent best-image selection process.

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

            Each list represents the contributions of a single scorer node in the
            chain.

        ``results`` : list[dict]
            Per-candidate detailed results for this node. Each entry contains:

                - ``input_image`` : str
                - ``score`` : float
                - ``landmarks`` : dict
                - ``eyes`` : dict
                - ``size`` : dict
                - ``identity`` : dict
                - ``aggregation`` : dict
                - ``flags`` : list[str]

            Additional debugging fields may also be present, such as:

                - ``face_bbox_xyxy``
                - ``pose``

            Each sub-component includes its own ``observable`` flag, features,
            breakdown, and flags.

    Parameters
    ----------
    spec : dict or str or Path, optional
        Node configuration resolved via ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``face_landmarker_task`` : str
                Path to MediaPipe FaceLandmarker ``.task`` model.

            ``pose_landmarker_task`` : str
                Path to MediaPipe PoseLandmarker ``.task`` model.

            ``insightface_name`` : str, optional
                InsightFace model name used for identity similarity.

            ``insightface_det_size`` : sequence[int], optional
                Detection size passed to InsightFace.

            ``device`` : str, optional
                Logical device string used by the model cache.

            ``dtype`` : str, optional
                Floating-point dtype used when loading FaceID reference embeddings.

        ``params`` : dict
            ``face_region_expansion`` : float, optional
                Expansion factor used when deriving the pose-guided head area.

            ``w_landmarks`` : float, optional
                Weight of landmark plausibility in local aggregation.

            ``w_eyes`` : float, optional
                Weight of eye plausibility in local aggregation.

            ``w_size`` : float, optional
                Weight of face size / crop quality in local aggregation.

            ``w_identity`` : float, optional
                Weight of identity similarity in local aggregation when reference
                embeddings are available.

            ``score_weight`` : float, optional
                Weight of this node in cross-node ranking aggregation.

    Notes
    -----
    - This node does not generate images; it selects among existing candidates.
    - Identity similarity is optional and only contributes when reference
    embeddings are provided upstream.
    - Identity scoring is computed on a head crop derived from the detected face
    box, rather than on the full image.
    - Observability is defined at the component level; the aggregated score
    reflects only the available evidence.
    - The scoring system is heuristic and intended for relative ranking, not
    biometric verification or anatomical validation.

    Design rationale
    ----------------
    ``FaceScorer`` complements broader structural scorers by focusing on the part
    of the image that users usually perceive as most sensitive: the face.

    Its role in the pipeline is to:

    - evaluate whether a face looks geometrically credible
    - reward images where the face is large enough and clean enough to matter
    - optionally favor candidates that better match a desired subject identity

    This makes face-based selection explicit, inspectable, and composable with
    other scorer nodes over time.
    """

    spec: SpecInput = field(default_factory=dict)

    def identity(self) -> AttachmentSink:
        """
        Attach a reference identity embedding for face similarity scoring.

        This method creates an input sink used to provide one or more FaceID
        embeddings representing the desired subject identity.

        The attached upstream node must output either:

        - ``'embeds'`` : path to a serialized FaceID embedding tensor
        - or ``'path'`` : path to a serialized FaceID embedding tensor

        The embeddings are loaded via ``load_faceid_embeds(...)`` and are expected
        to follow the project convention with shape ``(2, N, D)``, where:

        - dim 0 separates negative / positive embeddings
        - dim 1 enumerates multiple reference images
        - dim 2 is the embedding dimension

        If multiple embedding files are provided, they are concatenated along the
        reference dimension (``N``).

        When an identity reference is available, the node computes an additional
        identity similarity score for each candidate image. The score is based on
        the maximum similarity between the candidate face embedding and the
        positive reference embeddings.

        If no identity input is attached, identity scoring is disabled and does not
        contribute to the final score.

        Returns
        -------
        AttachmentSink
            Sink to be connected to an upstream node producing FaceID embeddings.
        """
        return AttachmentSink(
            name=f'identity:{self.id}',
            target=self,
            input_id='identity:default',
        )

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = ImgBundle(
            node_id=node_id,
            img_count=self.image.img_count,
            input=input,
        )

        ref_embeds = self._get_face_embeds(input, cfg)

        face_landmarker = get_mediapipe_face_landmarker(
            model_asset_path=cfg.face_landmarker_task,
            device=cfg.device,
        )

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )

        insightface_app = None
        use_identity = ref_embeds is not None

        if use_identity:
            insightface_app = get_insightface(
                model_name=cfg.insightface_name,
                device=cfg.device,
                det_size=cfg.insightface_det_size,
            )

        candidates: list[dict[str, Any]] = []

        for path in bundle.paths:
            try:
                with Image.open(path) as im:
                    img_rgb = np.asarray(im.convert('RGB'))

                face_xy_global: Optional[np.ndarray] = None
                face_bbox_global: Optional[tuple[int, int, int, int]] = None
                pose_info: Optional[dict[str, Any]] = None

                try:
                    pose_res = mp_pose_landmarks_full(
                        img_rgb,
                        pose_landmarker=pose_landmarker,
                    )
                    pose_info = {
                        'valid': pose_res.valid.tolist(),
                        'xy': pose_res.xy.tolist(),
                        'z': pose_res.z.tolist(),
                        'world_xyz': pose_res.world_xyz.tolist(),
                    }

                    face_det = face_landmarks_from_pose_guided_head_area(
                        img_rgb,
                        pose_xy=pose_res.xy,
                        face_landmarker=face_landmarker,
                        expansion=cfg.face_region_expansion,
                    )
                    face_xy_global = face_det['face_xy_global']
                    face_bbox_global = face_det['face_bbox_global']

                except RuntimeError:
                    # Fallback to full-image face detection if pose-guided crop fails.
                    face_xy_global = mp_face_landmarks(
                        img_rgb,
                        face_landmarker=face_landmarker,
                    )
                    face_bbox_global = face_bbox_xyxy_from_landmarks(
                        face_xy_global,
                        img_rgb.shape,
                    )

                landmark_score_res = face_landmark_plausibility_score(
                    face_xy_global,
                    image_shape=img_rgb.shape,
                )

                eye_score_res = eye_plausibility_score(
                    face_xy_global,
                    image_shape=img_rgb.shape,
                )

                size_score_res = face_size_score(
                    face_bbox_global,
                    image_shape=img_rgb.shape,
                )

                fx1, fy1, fx2, fy2 = face_bbox_global
                h_img, w_img = img_rgb.shape[:2]

                hx1, hy1, hx2, hy2 = square_head_bbox_from_face_bbox(
                    fx1, fy1, fx2, fy2,
                    w_img, h_img,
                    1.2,  # For identity should be a safe choice
                )
                head_rgb = img_rgb[hy1:hy2, hx1:hx2, :]
                if head_rgb.size == 0:
                    raise RuntimeError('Empty head crop for identity.')

                if use_identity:
                    identity_score_res = identity_similarity_score(
                        head_rgb,
                        insightface_app=insightface_app,
                        ref_embeds=ref_embeds,
                    )
                else:
                    identity_score_res = {
                        'score': 0.0,
                        'observable': False,
                        'breakdown': {},
                        'features': {},
                        'flags': ['no_identity_reference'],
                    }

                agg = _aggregate_face_score(
                    landmark_score_res=landmark_score_res,
                    eye_score_res=eye_score_res,
                    size_score_res=size_score_res,
                    identity_score_res=identity_score_res,
                    cfg=cfg,
                )

                candidate = {
                    'input_image': path,
                    'score': float(agg['score']),
                    'landmarks': landmark_score_res,
                    'eyes': eye_score_res,
                    'size': size_score_res,
                    'identity': identity_score_res,
                    'aggregation': agg,
                    'flags': agg['flags'],
                }

                if face_bbox_global is not None:
                    candidate['face_bbox_xyxy'] = list(face_bbox_global)

                if pose_info is not None:
                    candidate['pose'] = pose_info

            except RuntimeError as e:
                candidate = {
                    'input_image': path,
                    'score': 0.0,
                    'landmarks': {
                        'score': 0.0,
                        'observable': False,
                        'breakdown': {},
                        'features': {},
                        'flags': [f'face_runtime_error: {e}'],
                    },
                    'eyes': {
                        'score': 0.0,
                        'observable': False,
                        'breakdown': {},
                        'features': {},
                        'flags': [],
                    },
                    'size': {
                        'score': 0.0,
                        'observable': False,
                        'breakdown': {},
                        'features': {},
                        'flags': [],
                    },
                    'identity': {
                        'score': 0.0,
                        'observable': False,
                        'breakdown': {},
                        'features': {},
                        'flags': [],
                    },
                    'aggregation': {
                        'score': 0.0,
                        'weights': {
                            'w_landmarks': 0.0,
                            'w_eyes': 0.0,
                            'w_size': 0.0,
                            'w_identity': 0.0,
                        },
                        'flags': [f'face_runtime_error: {e}'],
                    },
                    'flags': [f'face_runtime_error: {e}'],
                    'error': str(e),
                }

            candidates.append(candidate)

        rankings = bundle.build_rankings(
            candidates=candidates,
            score_weight=cfg.score_weight,
        )

        best_image, best_score = bundle.calculate_best_image(rankings)

        if best_score <= 0.0 or best_image is None:
            raise RuntimeError(
                f'{self.id}: all candidate images failed scoring.'
            )

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'image': best_image,
            'best_score': best_score,
            'model': {
                'device': cfg.device,
                'dtype': str(cfg.dtype),
                'face_landmarker_task': cfg.face_landmarker_task,
                'pose_landmarker_task': cfg.pose_landmarker_task,
                'insightface_name': cfg.insightface_name,
                'insightface_det_size': list(cfg.insightface_det_size),
            },
            'params': {
                'face_region_expansion': cfg.face_region_expansion,
                'use_identity': use_identity,
                'w_landmarks': cfg.w_landmarks,
                'w_eyes': cfg.w_eyes,
                'w_size': cfg.w_size,
                'w_identity': cfg.w_identity,
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

    @staticmethod
    def _get_face_embeds(
        input: Optional[Dict[str, Dict]],
        cfg: ScorerConfig
    ) -> Optional[torch.Tensor]:
        identity_default = input.get('identity:default') if input else None

        if identity_default:
            emb_path = (
                identity_default.get('embeds')
                or identity_default.get('path')
            )
            if not emb_path:
                raise ValueError(
                    "Upstream output for 'identity:default' must contain embeddings path in 'embeds' or 'path'."
                )
            if not isinstance(emb_path, (str, list)):
                raise TypeError(
                    f"Upstream output for 'identity:default' must be str or list[str]. Got: {type(emb_path)}"
                )
            return load_faceid_embeds(
                emb_path,
                device=cfg.device,
                dtype=cfg.dtype
            )
        else:
            return None
