import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import (get_mediapipe_face_landmarker,
                                   get_mediapipe_hand_landmarker,
                                   get_mediapipe_pose_landmarker, get_sam,
                                   get_yolo)
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.segmentation import predict_sam_mask
from morphalo.nodes.preprocess.utils import (CropModeSpec,
                                             expand_bbox_toward_ratio,
                                             expand_clip_bbox,
                                             invert_mask_inside_box,
                                             parse_crop_mode,
                                             positive_points_for_sam,
                                             postprocess_mask,
                                             tight_alpha_bbox)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.face_region import (face_bbox_xyxy_from_landmarks,
                                               mp_face_landmarks)
from morphalo.nodes.vision.human import (crop_head_area_from_pose,
                                         hands_bbox_xyxy_from_landmarks,
                                         hands_mask_from_landmarks,
                                         mp_hand_landmarks_full,
                                         mp_pose_landmarks_xy,
                                         resolve_person_bbox_xyxy,
                                         square_head_bbox_from_face_bbox)

# Internal geometry constants.
#
# FACE_SEARCH_AREA_EXPANSION defines how generously the pose-derived head area is cropped
# before running the Face Landmarker.
FACE_SEARCH_AREA_EXPANSION = 1.6


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    yolo_model: Optional[str]
    sam_model: Optional[str]
    mode: str
    crop_mode: Optional[CropModeSpec]
    conf: float
    box_margin: float
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: Optional[str]
    hand_landmarker_task: Optional[str]
    pose_landmarker_task: str
    smoothing_radius: int
    min_landmark_fraction: Optional[float]


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'bf16')))

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': Invalid mode={mode!r}")

    # It should apply only when mode='default'
    if mode == 'default':
        crop_mode = parse_crop_mode(
            params.get('crop_mode', 'trim'),
            node_id=node_id,
        )
    else:
        crop_mode = None

    conf = float(params.get('conf', 0.35))
    box_margin = float(params.get('box_margin', 0.12))

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))
    min_landmark_fraction = params.get('min_landmark_fraction', 0.8)
    if min_landmark_fraction is not None:
        min_landmark_fraction = float(min_landmark_fraction)
        if not (0.0 <= min_landmark_fraction <= 1.0):
            raise ValueError(
                f"'{node_id}': invalid min_landmark_fraction={min_landmark_fraction!r} "
                '(expected a float in [0, 1] or None)'
            )

    target = str(params.get('target', 'person'))
    if target not in (
        'person',
        'head',
        'hands',
        'left-hand',
        'right-hand',
    ):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'person', "
            "'head', 'hands', 'left-hand' or 'right-hand'; use FaceCrop for "
            "face, eye, and eyebrow targets)"
        )

    sam_model = str(model.get('sam_model', 'facebook/sam-vit-large'))

    expansion = float(params.get('expansion', 1.0))
    if expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} (expected >= 0)"
        )

    save_debug = bool(debug.get('save_debug', False))

    face_landmarker_task = None
    hand_landmarker_task = None
    yolo_model = None

    pose_landmarker_task = model.get('pose_landmarker_task')
    if pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': Missing 'model.pose_landmarker_task' (MediaPipe .task path)"
        )
    pose_landmarker_task = str(pose_landmarker_task)

    if target in ('person', 'head', 'hands', 'left-hand', 'right-hand'):
        yolo_model = str(model.get('yolo_model', 'yolov8n.pt'))

    if target == 'head':
        # Face landmarks are required for head targets.
        face_landmarker_task = model.get('face_landmarker_task')
        if not face_landmarker_task:
            raise ValueError(
                f"'{node_id}': target={target!r} requires model.face_landmarker_task (MediaPipe .task path)"
            )
        face_landmarker_task = str(face_landmarker_task)

    if target in ('hands', 'left-hand', 'right-hand'):
        hand_landmarker_task = model.get('hand_landmarker_task')
        if not hand_landmarker_task:
            raise ValueError(
                f"'{node_id}': target={target!r} requires model.hand_landmarker_task (MediaPipe .task path)"
            )
        hand_landmarker_task = str(hand_landmarker_task)

    return Config(
        device=device,
        dtype=dtype,
        yolo_model=yolo_model,
        sam_model=sam_model,
        mode=mode,
        crop_mode=crop_mode,
        conf=conf,
        box_margin=box_margin,
        save_debug=save_debug,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        target=target,
        expansion=expansion,
        face_landmarker_task=face_landmarker_task,
        hand_landmarker_task=hand_landmarker_task,
        pose_landmarker_task=pose_landmarker_task,
        smoothing_radius=smoothing_radius,
        min_landmark_fraction=min_landmark_fraction,
    )


@dataclass(frozen=True)
class SamMaskCandidate:
    """
    SAM mask candidate with metadata used by person-mask selection.

    Parameters
    ----------
    mask : np.ndarray
        Full-frame boolean candidate mask.

    sam_score : float
        SAM predicted-IoU score associated with the raw mask.

    source : str
        Prompt strategy that produced the raw mask. Expected values are
        ``'strict'`` and ``'complete'``.

    inverted : bool, default=False
        Whether this candidate is the local inverse of the raw SAM mask.
    """
    mask: np.ndarray
    sam_score: float
    source: str
    inverted: bool = False


def _build_person_sam_candidates(
    *,
    img_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pose_xy: np.ndarray,
    processor: Any,
    model: Any,
    device: str,
    use_landmarks: bool = True,
) -> list[SamMaskCandidate]:
    """
    Build person SAM candidates using strict and complete prompting.

    This helper is used only for ``target='person'`` and intentionally combines
    two prompt strategies:

    strict
        Uses the resolved person bbox together with positive pose landmarks.
        This tends to produce cleaner, more subject-specific masks, but may miss
        weakly supported silhouette regions.

    complete
        Uses the resolved person bbox only. This may preserve a fuller
        silhouette, but is more likely to attach nearby background fragments or
        ambiguous objects.

    For each raw SAM mask, the local inverse inside the prompt bbox is also
    added as a candidate. The inverse candidate keeps the same SAM score because
    it is derived from the same raw prediction, but it is marked with
    ``inverted=True`` so the selector can prefer non-inverted candidates when
    all stronger criteria are tied.

    Returns
    -------
    list[SamMaskCandidate]
        Full-frame boolean candidate masks with selection metadata.
    """
    candidates: list[SamMaskCandidate] = []

    prompt_runs: list[tuple[str, Optional[list[list[float]]],
                            Optional[list[int]]]] = []

    if use_landmarks:
        point_coords, point_labels = positive_points_for_sam(
            xy=pose_xy,
            bbox=bbox,
        )

        if point_coords is not None and point_labels is not None:
            prompt_runs.append(('strict', point_coords, point_labels))

    prompt_runs.append(('complete', None, None))

    for source, run_point_coords, run_point_labels in prompt_runs:
        masks, scores = predict_sam_mask(
            img_rgb=img_rgb,
            bbox=bbox,
            processor=processor,
            model=model,
            device=device,
            point_coords=run_point_coords,
            point_labels=run_point_labels,
        )

        for i, mask in enumerate(masks):
            mi = mask.astype(bool)
            score_i = float(scores[i]) if i < len(scores) else 0.0

            candidates.append(SamMaskCandidate(
                mask=mi,
                sam_score=score_i,
                source=source,
                inverted=False,
            ))

            candidates.append(SamMaskCandidate(
                mask=invert_mask_inside_box(mi, bbox),
                sam_score=score_i,
                source=source,
                inverted=True,
            ))

    return candidates


def _quantize_score(value: float, *, bins: int = 10) -> int:
    """
    Quantize a normalized score into an integer bucket.

    Quantization prevents tiny score differences from dominating later
    tie-breakers such as area preference or prompt source.

    Parameters
    ----------
    value : float
        Score expected in ``[0, 1]``. Values outside the range are clipped.

    bins : int, default=10
        Number of score intervals. The returned bucket is in ``[0, bins]``.

    Returns
    -------
    int
        Quantized score bucket.
    """
    clipped = min(1.0, max(0.0, float(value)))
    return int(round(clipped * float(bins)))


def _select_best_person_sam_mask(
    candidates: list[SamMaskCandidate],
    *,
    bbox: tuple[int, int, int, int],
    pose_xy: np.ndarray,
    target_norm_area: float = 0.25,
    min_landmark_fraction: Optional[float] = 0.8,
) -> np.ndarray:
    """
    Select the best person mask among SAM candidates.

    The selector assumes that candidates may come from different prompt
    strategies, typically:

    - ``strict``: bbox + pose landmarks;
    - ``complete``: bbox only.

    Before ranking candidates, the selector tries to discard masks that do not
    contain enough stable pose landmarks. This helps reject SAM masks that cover
    nearby objects, props, furniture, or the local inverse/background instead of the
    actual subject.

    This landmark filtering is intentionally conservative and non-fatal: if no
    candidate passes the landmark-consistency check, the selector falls back to the
    full candidate set.

    Selection criteria, in order
    ----------------------------
    1. Landmark-consistency filtering, when possible.
    2. Quantized SAM predicted-IoU score.
    3. Distance from the preferred normalized candidate area.
    4. Non-inverted candidates.
    5. Strict prompt candidates.

    Landmark-consistency filtering
    ------------------------------
    Only a small set of relatively stable body anchors is used, such as nose,
    shoulders, elbows, wrists, hips, and knees. Foot-related landmarks are
    intentionally ignored because they are often occluded, truncated, or confused
    with supports, props, seats, or background objects.

    A candidate is kept if it contains at least a minimum fraction of the valid
    stable landmarks. If no valid landmarks are available, or if all candidates fail
    the check, the original candidate set is used.

    Candidate area normalization
    ----------------------------
    Candidate areas are measured inside the SAM prompt bbox and normalized
    relative to the available candidate set after optional landmark filtering:

    - smallest candidate area -> 0.0
    - largest candidate area -> 1.0

    The default ``target_norm_area=0.25`` intentionally favors conservative
    person masks while still allowing candidates larger than the smallest one.
    This reduces the risk of attaching external background fragments while
    keeping a chance to recover more complete silhouettes when SAM provides a
    good intermediate candidate.

    Parameters
    ----------
    candidates : list[SamMaskCandidate]
        Candidate masks and metadata.

    bbox : tuple[int, int, int, int]
        SAM prompt bbox ``(x1, y1, x2, y2)``.

    pose_xy : np.ndarray
        MediaPipe pose landmarks in full-image coordinates, with invalid points
        encoded as ``(-1, -1)``. A stable subset of these landmarks is used to
        reject masks that do not plausibly cover the selected subject.

    target_norm_area : float, default=0.25
        Preferred normalized candidate area. ``0.0`` favors the smallest
        candidate, ``1.0`` favors the largest candidate, and intermediate values
        favor masks between the two extremes.

    min_landmark_fraction : float | None, default=0.8
        Minimum fraction of usable stable pose landmarks that must be contained
        within a candidate SAM mask for it to be considered landmark-consistent.
        A higher value makes candidate selection stricter and favors masks that
        better align with pose predictions. A lower value makes selection more
        permissive and can improve results for noisy or partially occluded poses.
        If set to 0.0, landmark guidance remains enabled, but a candidate only
        needs to contain at least one usable stable landmark when such landmarks
        are available.

        If ``None``, landmark-consistency filtering is skipped. In the standard
        ``SubjectCrop`` person path, this value is also used upstream to generate
        only bbox-only SAM candidates.

    Returns
    -------
    np.ndarray
        Selected full-frame boolean mask.

    Raises
    ------
    RuntimeError
        If no candidates are available or selection fails.

    ValueError
        If ``target_norm_area`` is outside ``[0, 1]``.
    """
    def _landmark_inside_count(
        mask: np.ndarray,
        xy: np.ndarray,
        idxs: list[int],
    ) -> tuple[int, int]:
        """
        Count how many selected landmarks fall inside a candidate mask.

        Parameters
        ----------
        mask : np.ndarray
            Full-frame boolean mask.

        xy : np.ndarray
            Landmark coordinates in full-image coordinates.

        idxs : list[int]
            Landmark indices to evaluate.

        Returns
        -------
        tuple[int, int]
            ``(inside, valid)`` where ``inside`` is the number of valid landmarks
            covered by the mask and ``valid`` is the number of usable landmarks.
        """
        mask_h, mask_w = mask.shape

        inside = 0
        valid = 0

        for idx in idxs:
            if idx < 0 or idx >= xy.shape[0]:
                continue

            px, py = xy[idx, :2]

            if px < 0 or py < 0:
                continue

            px = int(px)
            py = int(py)

            if not (0 <= px < mask_w and 0 <= py < mask_h):
                continue

            valid += 1

            if mask[py, px]:
                inside += 1

        return inside, valid

    if not candidates:
        raise RuntimeError(
            'Cannot select best person SAM mask: no candidates.'
        )

    if not (0.0 <= float(target_norm_area) <= 1.0):
        raise ValueError(
            f'target_norm_area must be in [0, 1], got {target_norm_area!r}.'
        )

    x1, y1, x2, y2 = bbox

    # Stable body anchors:
    # 0  = nose
    # 11 = left shoulder
    # 12 = right shoulder
    # 13 = left elbow
    # 14 = right elbow
    # 15 = left wrist
    # 16 = right wrist
    # 23 = left hip
    # 24 = right hip
    # 25 = left knee
    # 26 = right knee
    #
    # Wrists and knees are included because they help preserve visible arms and legs
    # without relying on more fragile extremity landmarks.
    #
    # Ankles, heels, and foot tips are intentionally excluded because they are often
    # occluded, outside the actual visible subject, or confused with supports / props.
    safe_pose_idxs = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    # Require a strong majority of usable stable pose landmarks to fall inside the
    # candidate mask.
    #
    # "Usable" means: present in pose_xy, valid, inside image bounds, and included in
    # safe_pose_idxs. If only one stable landmark is usable, that landmark is allowed
    # to be decisive: it is still better than ignoring landmark consistency entirely.
    # The configurable ``min_landmark_fraction`` controls how strict the landmark
    # consistency check is.

    measured: list[tuple[SamMaskCandidate, int]] = []
    filtered: list[tuple[SamMaskCandidate, int]] = []

    for candidate in candidates:
        area = int(candidate.mask[y1:y2, x1:x2].sum())
        measured.append((candidate, area))

        if min_landmark_fraction is not None:
            inside, valid = _landmark_inside_count(
                candidate.mask,
                pose_xy,
                safe_pose_idxs,
            )

            if valid > 0:
                min_inside = max(
                    1,
                    int(math.ceil(float(valid) * min_landmark_fraction)),
                )

                if inside >= min_inside:
                    filtered.append((candidate, area))

    # Prefer landmark-consistent masks when enabled, but do not hard-fail on
    # difficult poses.
    pool = filtered if min_landmark_fraction is not None and filtered else measured

    areas = np.asarray(
        [area for _, area in pool],
        dtype=np.float32,
    )

    min_area = float(np.min(areas))
    max_area = float(np.max(areas))
    area_span = max(1.0, max_area - min_area)

    best_candidate = None
    best_key = None

    for candidate, area in pool:
        norm_area = (float(area) - min_area) / area_span
        area_distance = abs(norm_area - float(target_norm_area))

        key = (
            _quantize_score(candidate.sam_score, bins=10),
            -float(area_distance),
            not candidate.inverted,
            candidate.source == 'strict',
        )

        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError('Failed to select best person SAM mask.')

    return best_candidate.mask


@dataclass
class SubjectCrop(NodeRef):
    """
    Subject-aware crop and inpaint-mask generator using MediaPipe, YOLO, and
    SAM-compatible segmentation.

    ``SubjectCrop`` detects subject/body-level regions of interest and produces
    either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image
    (``mode='mask'`` or ``mode='negative-mask'``).

    Supported targets are:

    - ``person``: full visible subject crop/mask;
    - ``head``: head-oriented crop/mask with hair-friendly framing;
    - ``hands``: one or more visible hand crops/masks;
    - ``left-hand``: hand appearing on the left side of the image;
    - ``right-hand``: hand appearing on the right side of the image.

    Face-detail targets such as ``face``, ``eyes``, ``left-eye``,
    ``right-eye``, ``eyebrows``, ``left-eyebrow`` and ``right-eyebrow`` are
    handled by ``FaceCrop``. Keeping these concerns separate makes
    ``SubjectCrop`` responsible only for person selection, pose-guided geometry,
    hand localization, and SAM-based subject segmentation.

    The node is intended to support workflows such as:

    - extracting subjects for compositing (e.g. with ``ImageStack``);
    - producing full-frame inpaint masks for SDXL pipelines;
    - extracting head regions for FaceID / IP-Adapter refinement;
    - extracting hand regions for localized hand repair/refinement;
    - refining a small region by cropping, processing it separately, and
      reinserting it at the original coordinates.

    Side-specific hand targets use image/viewer perspective:

    - ``left-hand`` refers to the hand on the left side of the image;
    - ``right-hand`` refers to the hand on the right side of the image.

    This is intentionally different from MediaPipe handedness labels, which follow
    anatomical subject perspective and are mapped internally when needed.

    Pipeline
    --------

    The node combines several models to robustly localize a subject region:

    - MediaPipe Pose is always executed first to obtain body landmarks.
      These landmarks provide the primary anatomical consistency signal for
      locating the selected subject.

    - YOLO (COCO class 0) proposes candidate person bounding boxes for all
      supported targets. YOLO boxes are accepted only when they are consistent
      with the valid MediaPipe pose landmarks. If YOLO fails or returns an
      inconsistent person box, a pose-derived fallback person bbox is used.

    - For ``target='head'``, a coarse head / upper-body search area is derived
      from pose landmarks. MediaPipe Face Landmarker runs inside this search area
      to obtain accurate face landmarks. A square head crop box is then derived
      from the face bbox, with an upward bias to preserve hair.

    - For hand targets, MediaPipe Hand Landmarker runs on the full image and is
      used to derive hand-local bounding boxes and landmark-based hand masks.

    - A SAM-compatible segmentation model, loaded through ``get_sam(...)``, is
      used for all supported targets. The default model is
      ``'facebook/sam-vit-large'``.

    ``target='person'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The best YOLO bbox is selected using pose-landmark alignment.
    - The selected YOLO bbox is accepted only if it contains all valid pose
      landmarks; otherwise, a fallback bbox is inferred directly from pose
      landmarks.
    - The bbox may be expanded using ``box_margin``.
    - SAM is evaluated with two prompt strategies:

    1. bbox + positive pose landmarks;
    2. bbox only.

    - The bbox+landmark strategy usually produces cleaner, more
      subject-specific masks, but may miss weak silhouette regions.
    - The bbox-only strategy may preserve a fuller silhouette, but can attach
      nearby background fragments or ambiguous objects.
    - For every raw SAM mask, the local inverse inside the prompt bbox is also
      added as a candidate to handle occasional polarity mistakes.
    - Candidate selection uses quantized SAM predicted-IoU score first, then
      prefers a conservative normalized bbox-local area.
    - The final crop region corresponds to the resolved person bbox.

    ``target='head'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - A coarse head / upper-body search area is derived from pose landmarks.
    - MediaPipe Face Landmarker runs inside this search area.
    - A landmark-derived face bbox is computed in local search-area coordinates
      and remapped to full-image coordinates.
    - A square head crop box is computed from the face bbox using
      ``expansion`` and an upward bias to preserve hair.
    - SAM segmentation is guided by the resolved person bbox for robustness,
      while the final crop region corresponds to the derived head box.
    - This intentionally separates segmentation guidance from crop geometry:
      the person bbox helps SAM find the correct subject, while the head box
      controls the output crop.

    ``target='hands'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - MediaPipe Hand Landmarker detects reliable hands in the full image.
    - Expanded hand bounding boxes are merged into a single crop box.
    - A landmark-derived hand mask is constructed for all selected hands.
    - SAM segmentation is guided by the resolved person bbox and positive
      pose landmarks.
    - The final mask is the intersection of the SAM subject mask and the
      landmark-derived hand mask.
    - The final crop region isolates one or more visible hands.

    ``target='left-hand'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - MediaPipe Hand Landmarker detects reliable hands in the full image.
    - The hand on the left side of the image is selected.
    - MediaPipe handedness labels are mapped internally because they follow
      anatomical subject perspective.
    - A hand-local bbox is computed from landmarks and optionally expanded via
    ``expansion``.
    - A landmark-derived hand mask is constructed for the selected hand.
    - SAM segmentation is guided by the resolved person bbox and positive
      pose landmarks.
    - The final mask is the intersection of the SAM subject mask and the
      landmark-derived hand mask.
    - The final crop region isolates only the left hand in image/viewer
      perspective.

    ``target='right-hand'``
    - MediaPipe Pose landmarks are computed for the full image.
    - YOLO proposes candidate person bounding boxes.
    - The selected YOLO bbox is accepted only if it is consistent with pose
      landmarks; otherwise, a pose-derived person bbox is used.
    - MediaPipe Hand Landmarker detects reliable hands in the full image.
    - The hand on the right side of the image is selected.
    - MediaPipe handedness labels are mapped internally because they follow
      anatomical subject perspective.
    - A hand-local bbox is computed from landmarks and optionally expanded via
    ``expansion``.
    - A landmark-derived hand mask is constructed for the selected hand.
    - SAM segmentation is guided by the resolved person bbox and positive
      pose landmarks.
    - The final mask is the intersection of the SAM subject mask and the
      landmark-derived hand mask.
    - The final crop region isolates only the right hand in image/viewer
      perspective.

    Hand crop semantics
    -------------------

    For hand targets, two regions are intentionally used:

    - a person-level bbox, resolved from YOLO + pose, is used as the SAM prompt
      region;
    - a hand-level bbox, resolved from MediaPipe hand landmarks, defines the final
      crop region.

    The final hand mask is obtained by intersecting the SAM subject mask with the
    landmark-derived hand mask. This keeps the crop focused on the requested hand
    region while still using SAM to reject background around the selected subject.

    For this reason, ``box_margin`` expands the person-level SAM prompt bbox for
    hand targets, not the final hand crop bbox.

    ## Head crop geometry

    Let ``face_size = max(face_w, face_h)`` from the landmark-derived face box.

    The head crop box is defined as a square centered on the face, shifted upward
    to preserve hair:

    - vertical shift: ``cy -= 0.15 * face_size``;
    - radius: ``radius = 0.75 * face_size * expansion``.

    Parameters
    ----------

    name : str, optional
        Node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default input
        (``input['default']['image']`` or ``input['default']['path']``).

    spec : dict or str or Path, optional
        Node specification (inline dict or path to a config file), resolved via
        ``resolve_spec``.

    Expected structure:

    ``model`` : dict
        ``device`` : str, optional
            Inference device (e.g. ``'cuda'``, ``'cuda:0'``, ``'cpu'``).
            Default: ``'cuda'``.

        ``dtype`` : str, optional
            Torch dtype used to load the SAM-compatible segmentation model.
            Supported values follow ``resolve_dtype`` conventions, e.g.
            ``'bf16'``, ``'float16'`` or ``'float32'``. Default: ``'bf16'``.

        ``sam_model`` : str, optional
            Hugging Face SAM-compatible model identifier used for mask
            generation. Default: ``'facebook/sam-vit-large'``.

            Typical values include:

            - ``'facebook/sam-vit-base'``;
            - ``'facebook/sam-vit-large'``;
            - ``'facebook/sam-vit-huge'``;
            - ``'syscv-community/sam-hq-vit-base'``;
            - ``'syscv-community/sam-hq-vit-large'``;
            - ``'syscv-community/sam-hq-vit-huge'``.

            Required for all supported targets because ``SubjectCrop`` uses SAM
            for ``person``, ``head`` and hand targets.

        ``yolo_model`` : str, optional
            YOLO weights used to propose person boxes. Default:
            ``'yolov8n.pt'``.

            YOLO boxes are accepted only when they are consistent with
            MediaPipe pose landmarks; otherwise the node falls back to a
            pose-derived person bbox.

        ``pose_landmarker_task`` : str
            MediaPipe PoseLandmarker ``.task`` path. Required for all targets.
            Pose landmarks are used to select the correct person bbox and to
            guide subject-level geometry.

        ``face_landmarker_task`` : str, optional
            MediaPipe FaceLandmarker ``.task`` path. Required only for
            ``target='head'``.

        ``hand_landmarker_task`` : str, optional
            MediaPipe HandLandmarker ``.task`` path. Required for
            ``target='hands'``, ``target='left-hand'`` and
            ``target='right-hand'``.

    ``params`` : dict
        ``target`` : {'person', 'head', 'hands', 'left-hand', 'right-hand'}, optional
            Region to extract. Default: ``'person'``.

        ``mode`` : {'default', 'mask', 'negative-mask'}, optional
            Output type:

            - ``'default'``:
            RGBA cutout.
            - ``'mask'``:
            full-frame 8-bit mask (white = selected region).
            - ``'negative-mask'``:
            inverted full-frame mask (white = background).

            Default: ``'default'``.

        ``crop_mode`` : {'bbox', 'bbox[w:h]', 'trim', 'full_frame'}, optional
            Applies only when ``mode='default'`` and controls the spatial
            layout of the RGBA cutout.

            Supported forms are:

            - ``'bbox'``:
              Output is the rectangular crop inside the selected crop box,
              including the original background; alpha is fully opaque
              (255 everywhere).

            - ``'bbox[w:h]'``:
              Same as ``'bbox'``, but the selected crop box is expanded toward
              the requested aspect ratio ``w:h`` while keeping the target fully
              inside the crop and staying within source-image bounds.

              The requested ratio is treated as a target, not a hard constraint.
              If the source image does not provide enough room near the borders,
              the final crop may deviate from the requested ratio.

            - ``'trim'``:
              Output is an RGBA cutout cropped to the selected bounding box,
              with alpha derived from the mask. The result is then tightly
              trimmed to the minimal box containing non-transparent pixels.

            - ``'full_frame'``:
              Same cutout as ``'trim'`` but placed back into a full-size RGBA
              canvas of the original image dimensions, preserving the original
              coordinates.

              Default: ``'trim'``.

        ``conf`` : float, optional
            YOLO confidence threshold. Typical range: 0.2-0.6.
            Default: 0.35.

        ``box_margin`` : float, optional
            Symmetric expansion ratio applied to the resolved SAM prompt bbox,
            expressed as a fraction of bbox size.

            For ``target='person'``, this also affects the default crop region
            because the SAM prompt bbox is reused as the final crop box. For
            ``target='head'`` and hand targets, the final crop region is
            target-local and remains independent from this margin.

            Typical range: 0.05-0.20. Default: 0.12.

        ``expansion`` : float, optional
            Expansion factor applied to target-local crop geometry.

            - for ``target='head'``:
             controls the derived square head crop size;
            - for hand targets:
              expands the landmark-derived hand bbox / mask;
            - for ``target='person'``:
              the main spatial padding is controlled by ``box_margin``.

              Default: ``1.0``.

        ``dilate_radius`` : int, optional
            Mask dilation radius in pixels.

            In ``mode='mask'``, dilation expands the repaintable selected
            region. In ``mode='negative-mask'``, dilation is applied before
            inversion, so it expands the protected subject region and creates a
            safety margin between the subject and the repaintable background.

            Default: 0.

        ``close_radius`` : int, optional
            Morphological closing radius in pixels.

            Closing is applied before optional inversion. It fills small holes
            and gaps in the selected subject mask. This is often useful for
            stable inpainting, but in ``mode='negative-mask'`` it also means
            that small background holes inside the subject silhouette become
            protected after inversion. Set this to 0 when those internal holes
            should remain repaintable background.

            Default: 0.

        ``smoothing_radius`` : int, optional
            Gaussian smoothing radius in pixels.

            Smoothing is applied before optional inversion. In ``mode='mask'``,
            it softens the repaintable selected region. In
            ``mode='negative-mask'``, it softens the protected subject boundary
            before inversion, producing a feathered transition between protected
            subject and repaintable background.

            Default: 0.

        ``min_landmark_fraction`` : float | None, optional
            Minimum fraction of usable stable pose landmarks that must be
            contained within a candidate SAM mask to consider it
            landmark-consistent.

            Default: 0.8.

            Typical values around 0.7 are often effective. Increasing the value
            makes pose consistency stricter, while lowering it makes candidate
            selection more permissive.

            If set to ``None``, strict landmark prompting is disabled for
            ``target='person'`` and only bbox-only SAM mask candidates are
            generated and evaluated.

            This parameter can help tune performance for difficult poses,
            occluded limbs, or noisy landmark detections.

    ``debug`` : dict
        ``save_debug`` : bool, optional
            If True, saves a debug image with the selected crop/prompt bbox
            overlay. Default: False.

    Mask post-processing
    --------------------

    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the detected subject mask is
    post-processed before it is written to disk.

    Post-processing is always applied to the positive subject mask first, before any
    optional polarity inversion:

    1. the selected subject region is assembled as a full-frame positive mask;
    2. ``close_radius``, ``dilate_radius``, and ``smoothing_radius`` are applied;
    3. if ``mode='negative-mask'``, the post-processed subject mask is inverted.

    This ordering is intentional.

    For ``mode='mask'``, the output mask directly marks the selected subject region
    as repaintable. Dilation and smoothing therefore expand and soften the subject
    region itself, which is useful when repainting or refining the selected target.

    For ``mode='negative-mask'``, the output mask marks the background as
    repaintable and protects the selected subject. Applying dilation and smoothing
    before inversion creates a protected safety band around the subject. This
    prevents the background inpaint area from bleeding into the subject boundary.

    In other words, with ``mode='negative-mask'``:

    * ``dilate_radius`` expands the protected subject area before inversion;
    * ``smoothing_radius`` feathers the transition around the protected subject;
    * ``close_radius`` closes small holes inside the protected subject area before
      inversion.

    If preserving holes inside the subject mask is important, for example gaps
    between arms, fingers, hair strands, or other background-visible openings,
    prefer setting ``close_radius=0``.

    Returns
    -------

    dict
    Output metadata dictionary (also written as a JSON sidecar) with:

    ``ok`` : bool
        Success flag.

    ``node`` : str
        Operator name.

    ``id`` : str
        Node identifier.

    ``input_image`` : str
        Source image path.

    ``mode`` : str
        Output mode.

    ``image`` : str
        Output file path (RGBA cutout or mask).

    ``params`` : dict
        Configuration parameters.

    ``model`` : dict
        Resolved model/runtime metadata, including the selected ``sam_model``,
        MediaPipe task paths, optional YOLO model, device, and dtype.

    ``crop`` : dict
        Crop metadata useful for reinsertion/compositing:

        ``anchor_xy`` : list[int]
            Center of the effective crop box in absolute source-image
            coordinates.

        ``bbox_size`` : list[int]
            Width and height of the effective crop box in pixels.

        ``bbox_xyxy`` : list[int]
            Effective end-exclusive crop box ``[x1, y1, x2, y2]`` in
            source-image coordinates.

    ``debug_bbox`` : str, optional
        Debug image path, present only when ``debug.save_debug`` is true.

    ``metadata`` : str
        JSON sidecar path.

    Notes
    -----

    - Mask outputs are always full-frame and aligned to the original image size.
    - In ``mode='default'``, ``crop_mode`` controls whether the output is a
      target-local RGBA cutout, a bbox crop, or a full-frame RGBA canvas.
    - In ``mode='mask'`` and ``mode='negative-mask'``, ``crop_mode`` is ignored
      because mask outputs are always full-frame.
    - ``dilate_radius``, ``close_radius`` and ``smoothing_radius`` affect only
      full-frame mask outputs. In ``mode='default'``, the RGBA alpha is derived
      directly from the selected mask after connected-component cleanup.
    - For ``target='head'``, SAM uses the resolved person bbox as segmentation
      prompt, while the final crop region is the derived head box.
    - For hand targets, SAM uses the resolved person bbox and positive pose
      landmarks as segmentation prompt, while the final crop region is derived from
      hand landmarks.
    - For hand targets, the final mask is obtained by intersecting the
      landmark-derived hand mask with the SAM subject mask.
    - For ``target='hands'``, multiple disconnected hand components may be
      preserved inside the same crop.
    - Side-specific hand targets use image/viewer perspective.
    - MediaPipe handedness labels follow anatomical subject perspective and are
      mapped internally for hand targets to preserve the image/viewer convention.
    - Heavy models (YOLO, SAM-compatible segmentation models, MediaPipe Tasks) are
      retrieved via the global model cache where available.
    - The segmentation backend is loaded through Hugging Face Transformers via
      ``get_sam(...)`` and cached as a ``(processor, model)`` pair.
    - This node does not require a local ``sam_checkpoint`` / ``sam_model_type``
      pair. Use ``model.sam_model`` to select the Hugging Face model id.
    - For ``target='person'``, mask polarity is resolved during candidate selection
      by evaluating both each raw SAM mask and its local inverse inside the prompt
      bbox.
    - For ``crop_mode='bbox[w:h]'``, the requested aspect ratio is treated as a
      target, not a hard guarantee. Near the image boundaries the final crop may
      deviate from the requested ratio.
    - If you change code or spec and need fresh outputs, delete the existing
      sidecar JSON to avoid reusing cached results.
  """

    # Either pass a path explicitly, or wire an upstream image into default input.
    path: Optional[Union[str, Path]] = None

    # Optional node spec (device, etc.)
    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        # Local imports to avoid hard deps if node unused
        import cv2

        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        # -----------------------------
        # Resolve input image path
        # -----------------------------
        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input
        )

        out_dir = Path(output_dir)

        # -----------------------------
        # Load image
        # -----------------------------
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(
                f"SubjectCrop node '{node_id}': cannot read image: {img_path}")

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        hand_mask: Optional[np.ndarray] = None

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        pose_xy = mp_pose_landmarks_xy(
            img_rgb=img_rgb,
            pose_landmarker=pose_landmarker,
        )

        if cfg.target == 'person':
            # YOLO proposes person bbox candidates; resolve_person_bbox_xyxy()
            # accepts a YOLO box only when it is consistent with MediaPipe pose,
            # otherwise it falls back to a pose-derived bbox.
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

        elif cfg.target == 'head':
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            head_area_rgb, a_x, a_y = crop_head_area_from_pose(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=FACE_SEARCH_AREA_EXPANSION,
            )

            face_xy = mp_face_landmarks(
                img_rgb=head_area_rgb,
                face_landmarker=landmarker,
            )
            r_x1, r_y1, r_x2, r_y2 = face_bbox_xyxy_from_landmarks(
                face_xy,
                image_shape=head_area_rgb.shape
            )

            fx1 = r_x1 + a_x
            fy1 = r_y1 + a_y
            fx2 = r_x2 + a_x
            fy2 = r_y2 + a_y

            fx1, fy1, fx2, fy2 = square_head_bbox_from_face_bbox(
                fx1, fy1, fx2, fy2, w, h,
                expansion=cfg.expansion,
            )

        elif cfg.target in ('hands', 'left-hand', 'right-hand'):
            # --------------------------------------------------
            # Resolve the subject bbox first.
            # This bbox is used only to guide SAM toward the correct person.
            # --------------------------------------------------

            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            bx1, by1, bx2, by2 = resolve_person_bbox_xyxy(
                res,
                node_id,
                pose_xy=pose_xy,
                image_shape=img_rgb.shape,
            )

            # --------------------------------------------------
            # Resolve the hand-local geometry from hand landmarks.
            #
            # - h* bbox defines the final crop region
            # - hand_mask is later intersected with the SAM subject mask
            #   so the final alpha stays hand-focused and background-free
            # --------------------------------------------------
            hand_landmarker = get_mediapipe_hand_landmarker(
                model_asset_path=cfg.hand_landmarker_task,
                device=cfg.device,
            )

            hands_res = mp_hand_landmarks_full(
                img_rgb=img_rgb,
                hand_landmarker=hand_landmarker,
            )

            hand_which = {
                'hands': 'both',
                'left-hand': 'left',
                'right-hand': 'right',
            }[cfg.target]

            hx1, hy1, hx2, hy2 = hands_bbox_xyxy_from_landmarks(
                hands_res,
                img_rgb.shape,
                which=hand_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            hand_mask = hands_mask_from_landmarks(
                hands_res,
                img_rgb.shape,
                which=hand_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        # Expand the SAM prompt bbox when applicable.
        # For hand targets this expands the person bbox used to guide SAM,
        # not the final hand crop bbox.
        if cfg.box_margin > 0:
            bx1, by1, bx2, by2 = expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.box_margin
            )

        # -----------------------------
        # Mask generation
        # -----------------------------
        sam_model_id = cfg.sam_model
        processor, sam_model = get_sam(
            model_id=sam_model_id,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        if cfg.target == 'person':
            # For full-body crops, neither strict nor complete prompting is universally
            # better:
            #
            # - strict candidates are cleaner but can miss weak silhouette regions
            # - complete candidates can preserve more silhouette but may attach artifacts
            #
            # We evaluate both families and select a conservative candidate using SAM
            # score, normalized area, and weak tie-breakers.
            candidates = _build_person_sam_candidates(
                img_rgb=img_rgb,
                bbox=(bx1, by1, bx2, by2),
                pose_xy=pose_xy,
                processor=processor,
                model=sam_model,
                device=cfg.device,
                use_landmarks=(cfg.min_landmark_fraction is not None),
            )

            mask = _select_best_person_sam_mask(
                candidates,
                bbox=(bx1, by1, bx2, by2),
                pose_xy=pose_xy,
                min_landmark_fraction=cfg.min_landmark_fraction,
            ).astype(bool)

        else:
            point_coords = None
            point_labels = None

            if cfg.target in ('hands', 'left-hand', 'right-hand'):
                # Use body pose points intentionally: SAM is asked to segment the selected
                # person inside the person bbox. The hand landmark mask is applied later to
                # restrict the result to the requested hand region.
                point_coords, point_labels = positive_points_for_sam(
                    xy=pose_xy,
                    bbox=(bx1, by1, bx2, by2),
                )

            masks, scores = predict_sam_mask(
                img_rgb=img_rgb,
                bbox=(bx1, by1, bx2, by2),
                processor=processor,
                model=sam_model,
                device=cfg.device,
                point_coords=point_coords,
                point_labels=point_labels,
            )

            if masks is None or len(masks) == 0:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': SAM returned no masks."
                )

            best_mask = None
            best_key = None

            for i in range(len(masks)):
                mi = masks[i].astype(bool)

                score_i = (
                    float(scores[i])
                    if scores is not None and i < len(scores)
                    else 0.0
                )

                if best_key is None or score_i > best_key:
                    best_key = score_i
                    best_mask = mi

            if best_mask is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': failed to select a SAM mask."
                )

            mask = best_mask.astype(bool)

        if cfg.target != 'person':
            valid = (
                (pose_xy[:, 0] >= 0) &
                (pose_xy[:, 1] >= 0)
            )

            pts = pose_xy[valid]

            if len(pts) > 0:
                inside = 0
                mask_h, mask_w = mask.shape

                for px, py in pts:
                    px = int(px)
                    py = int(py)

                    if 0 <= px < mask_w and 0 <= py < mask_h and mask[py, px]:
                        inside += 1

                min_inside = max(1, int(math.ceil(len(pts) * 0.5)))

                if inside < min_inside:
                    # SAM may occasionally return the local background instead of the prompted
                    # subject region. For non-person targets, we use pose landmark coverage as a
                    # cheap polarity sanity check: if too few pose points fall inside the mask, we
                    # invert the mask inside the prompt bbox.
                    mask = invert_mask_inside_box(
                        mask=mask,
                        box=(bx1, by1, bx2, by2)
                    )

        if cfg.target in ('hands', 'left-hand', 'right-hand'):
            if hand_mask is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': hand_mask not computed."
                )
            # Keep only the hand-local part of the SAM subject mask.
            # SAM separates subject vs background; the landmark mask constrains the
            # result to the selected hand region(s).
            mask = mask & hand_mask

        # --------------------------------------------------
        # Select crop region depending on target
        # --------------------------------------------------

        if cfg.target == 'head':
            # crop strictly around face/head bbox (full-image coordinates)
            crop_x1, crop_y1, crop_x2, crop_y2 = fx1, fy1, fx2, fy2
        elif cfg.target in ('hands', 'left-hand', 'right-hand'):
            # crop including one or both hands
            crop_x1, crop_y1, crop_x2, crop_y2 = hx1, hy1, hx2, hy2
        else:
            # Default crop region: use the target bbox already resolved above.
            crop_x1, crop_y1, crop_x2, crop_y2 = bx1, by1, bx2, by2

        if cfg.mode == 'default' and cfg.crop_mode is not None:
            if cfg.crop_mode.mode == 'bbox' and cfg.crop_mode.ratio is not None:
                crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox_toward_ratio(
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2,
                    full_w=w,
                    full_h=h,
                    ratio=cfg.crop_mode.ratio,
                )

        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_mask.size == 0:
            raise RuntimeError(
                f"SubjectCrop node '{node_id}': empty crop after bbox."
            )

        cm = crop_mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm, connectivity=8
        )

        if num > 1:
            if cfg.target in (
                'hands',
                'left-hand',
                'right-hand',
            ):
                # Keep all connected components inside already target-local crops.
                #
                # - For 'hands', this preserves multiple visible hands inside one shared crop.
                # - For 'left-hand' / 'right-hand', the crop is already constrained by the
                #   selected hand bbox and by the landmark-derived hand mask, so keeping all
                #   components is safer than selecting the component under the crop center.
                #
                # TODO:
                # If single-hand crops start carrying too many small artifacts, replace this
                # broad keep-all rule with a hand-seeded component selector, e.g. selecting the
                # connected component that contains or is nearest to a reliable hand landmark
                # such as the wrist or palm center.
                crop_mask = (labels != 0)

            else:
                # Keep a single component for other targets.
                ccx = cm.shape[1] // 2
                ccy = cm.shape[0] // 2
                target = labels[ccy, ccx]

                if target == 0:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    target = 1 + int(np.argmax(areas))

                crop_mask = (labels == target)

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        out_x1 = int(crop_x1)
        out_y1 = int(crop_y1)
        out_x2 = int(crop_x2)
        out_y2 = int(crop_y2)

        if cfg.mode == 'default':
            crop_rgb = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2, :]
            crop_h = int(crop_y2 - crop_y1)
            crop_w = int(crop_x2 - crop_x1)

            if cfg.crop_mode is None:
                raise ValueError(
                    f"{self.id}: crop_mode must be defined when mode='default'"
                )

            if cfg.crop_mode.mode == 'bbox':
                # Include the original background inside the crop; alpha is fully opaque.
                alpha = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                crop_rgba = np.dstack([crop_rgb, alpha])
                Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

            else:
                # 'trim' and 'full_frame' -> alpha of mask
                alpha = crop_mask.astype(np.uint8) * 255
                crop_rgba = np.dstack([crop_rgb, alpha])

                if cfg.crop_mode.mode == 'trim':
                    tx1, ty1, tx2, ty2 = tight_alpha_bbox(alpha)
                    crop_rgba = crop_rgba[ty1:ty2, tx1:tx2, :]

                    out_x1 = int(crop_x1 + tx1)
                    out_y1 = int(crop_y1 + ty1)
                    out_x2 = int(crop_x1 + tx2)
                    out_y2 = int(crop_y1 + ty2)

                    Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

                elif cfg.crop_mode.mode == 'full_frame':
                    full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    full_rgba[crop_y1:crop_y2, crop_x1:crop_x2, :] = crop_rgba

                    # For full-frame outputs, crop metadata must describe the output image
                    # itself in source-image coordinates.
                    out_x1 = 0
                    out_y1 = 0
                    out_x2 = w
                    out_y2 = h

                    Image.fromarray(full_rgba, mode='RGBA').save(out_path)

                else:
                    raise ValueError(
                        f'{self.id}: invalid crop_mode={cfg.crop_mode!r}'
                    )

        else:
            # build full-frame positive subject mask
            full_mask = np.zeros((h, w), dtype=np.uint8)

            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = (
                crop_mask.astype(np.uint8) * 255
            )

            # Post-process the selected subject mask before polarity conversion.
            #
            # For mode='mask', this expands/softens the repaint region itself.
            # For mode='negative-mask', this expands/softens the protected subject region
            # before inversion, creating a safety band around the subject instead of letting
            # the background mask bleed into it.
            full_mask = postprocess_mask(
                full_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                full_mask = 255 - full_mask

            Image.fromarray(full_mask, mode='L').save(out_path)

        # Optional debug bbox overlay
        dbg_path = None
        if cfg.save_debug:
            dbg = img_rgb.copy()
            dbg_x1, dbg_y1, dbg_x2, dbg_y2 = bx1, by1, bx2, by2
            if cfg.target == 'head':
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = fx1, fy1, fx2, fy2
            elif cfg.target in ('hands', 'left-hand', 'right-hand'):
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = hx1, hy1, hx2, hy2
            cv2.rectangle(
                dbg,
                (dbg_x1, dbg_y1),
                (dbg_x2, dbg_y2),
                (255, 0, 0),
                3,
            )
            dbg_path = out_path.with_name(out_path.stem + '_debug_bbox.png')
            Image.fromarray(dbg).save(dbg_path)

        b_width = int(out_x2 - out_x1)
        b_height = int(out_y2 - out_y1)
        anchor_x = int((out_x1 + out_x2) // 2)
        anchor_y = int((out_y1 + out_y2) // 2)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'model': {
                **({} if cfg.yolo_model is None else {'yolo': cfg.yolo_model}),
                **({} if sam_model_id is None else {
                    'sam_model': sam_model_id,
                }),
                **({} if cfg.face_landmarker_task is None else {
                    'face_landmarker_task': cfg.face_landmarker_task,
                }),
                **({} if cfg.pose_landmarker_task is None else {
                    'pose_landmarker_task': cfg.pose_landmarker_task,
                }),
                **({} if cfg.hand_landmarker_task is None else {
                    'hand_landmarker_task': cfg.hand_landmarker_task,
                }),
                'device': cfg.device,
                'dtype': str(cfg.dtype).replace('torch.', ''),
            },
            'params': {
                'target': cfg.target,
                'mode': cfg.mode,
                'crop_mode': None if cfg.crop_mode is None else cfg.crop_mode.raw,
                'conf': cfg.conf,
                'box_margin': cfg.box_margin,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
                'min_landmark_fraction': cfg.min_landmark_fraction,
            },
            'crop': {
                'anchor_xy': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
                'bbox_xyxy': [int(out_x1), int(out_y1), int(out_x2), int(out_y2)],
            }
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
