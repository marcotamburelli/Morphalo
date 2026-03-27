import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from PIL import Image

from morphalo.cache.models import (get_mediapipe_face_landmarker,
                                   get_mediapipe_pose_landmarker, get_sam,
                                   get_yolo)
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import postprocess_mask
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.face_region import (crop_head_area_from_pose,
                                               expand_clip_bbox,
                                               square_head_bbox_from_face_bbox)
from morphalo.nodes.vision.human import (eye_bbox_xyxy_from_landmarks,
                                         eye_mask_from_landmarks,
                                         face_bbox_xyxy_from_landmarks,
                                         mp_face_landmarks,
                                         mp_pose_landmarks_xy,
                                         person_bboxes_xyxy,
                                         select_person_bbox_xyxy)

CropMode = Literal['bbox', 'trim', 'full_frame']


def _tight_alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return the minimal end-exclusive box containing non-zero alpha pixels.

    Parameters
    ----------
    alpha : np.ndarray
        Alpha channel with shape ``(H, W)``.

    Returns
    -------
    tuple[int, int, int, int]
        Tight bounding box ``(x1, y1, x2, y2)`` in local coordinates.

    Raises
    ------
    RuntimeError
        If the alpha channel contains no non-zero pixels.
    """
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError('Trimmed crop has no non-transparent pixels.')

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


@dataclass
class Config:
    device: str
    yolo_model: Optional[str]
    sam_checkpoint: Optional[str]
    sam_model_type: Optional[str]
    mode: str
    crop_mode: Optional[CropMode]
    conf: float
    box_margin: float
    multimask: bool
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: Optional[str]
    pose_landmarker_task: str
    smoothing_radius: int


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    # sam_checkpoint = model.get('sam_checkpoint')
    device = model.get('device', 'cuda')

    # if not sam_checkpoint:
    #     raise ValueError(
    #         f"'{node_id}': Missing 'sam_checkpoint' from model configuration."
    #     )

    sam_model_type = model.get('sam_model_type')

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': Invalid mode={mode!r}")

    # It should apply only when mode='default'
    if mode == 'default':
        crop_mode = str(params.get('crop_mode', 'trim'))
        if crop_mode not in ('bbox', 'trim', 'full_frame'):
            raise ValueError(
                f"'{node_id}': invalid crop_mode={crop_mode!r} "
                "(expected 'bbox', 'trim', or 'full_frame')"
            )
    else:
        crop_mode = None

    conf = float(params.get('conf', 0.35))
    box_margin = float(params.get('box_margin', 0.12))
    multimask = bool(params.get('multimask', True))

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))

    target = str(params.get('target', 'person'))
    if target not in ('person', 'face', 'head', 'eyes', 'left-eye', 'right-eye'):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'person', 'face', 'head', 'eyes', 'left-eye', or 'right-eye')"
        )

    needs_sam = target not in ('eyes', 'left-eye', 'right-eye')

    sam_checkpoint = model.get('sam_checkpoint')
    if needs_sam and not sam_checkpoint:
        raise ValueError(
            f"'{node_id}': target={target!r} requires 'model.sam_checkpoint'."
        )

    expansion = float(params.get('expansion', 1.0))
    if expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} (expected >= 0)"
        )

    save_debug = bool(debug.get('save_debug', False))

    face_landmarker_task = None
    yolo_model = None

    pose_landmarker_task = model.get('pose_landmarker_task')
    if pose_landmarker_task is None:
        raise ValueError(
            f"'{node_id}': Missing 'model.pose_landmarker_task' (MediaPipe .task path)"
        )
    pose_landmarker_task = str(pose_landmarker_task)

    if target in ('person', 'head'):
        yolo_model = str(model.get('yolo_model', 'yolov8n.pt'))

    if target in ('face', 'head', 'eyes', 'left-eye', 'right-eye'):
        # MediaPipe is used for face/head
        face_landmarker_task = model.get('face_landmarker_task')
        if not face_landmarker_task:
            raise ValueError(
                f"'{node_id}': target={target!r} requires model.face_landmarker_task (MediaPipe .task path)"
            )
        face_landmarker_task = str(face_landmarker_task)

    return Config(
        device=device,
        yolo_model=yolo_model,
        sam_checkpoint=None if sam_checkpoint is None else str(sam_checkpoint),
        sam_model_type=None if sam_model_type is None else str(sam_model_type),
        mode=mode,
        crop_mode=crop_mode,
        conf=conf,
        box_margin=box_margin,
        multimask=multimask,
        save_debug=save_debug,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        target=target,
        expansion=expansion,
        face_landmarker_task=face_landmarker_task,
        pose_landmarker_task=pose_landmarker_task,
        smoothing_radius=smoothing_radius,
    )


def _score_sam_mask_with_landmarks(
    mask: np.ndarray,
    pose_xy: np.ndarray,
) -> tuple[int, int]:
    """
    Score a SAM mask using pose landmarks.

    Returns
    -------
    tuple[int, int]
        (num_landmarks_inside, negative_area)

        Higher is better.
    """

    h, w = mask.shape

    valid = (
        (pose_xy[:, 0] >= 0) &
        (pose_xy[:, 1] >= 0)
    )

    pts = pose_xy[valid]

    inside = 0
    for px, py in pts:
        px = int(px)
        py = int(py)

        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            inside += 1

    area = int(mask.sum())

    return (inside, -area)


def _infer_sam_model_type(ckpt: Path, sam_model_type: Optional[str]) -> str:
    if sam_model_type:
        return sam_model_type

    name = ckpt.name.lower()

    if 'vit_h' in name:
        return 'vit_h'
    if 'vit_l' in name:
        return 'vit_l'
    if 'vit_b' in name:
        return 'vit_b'

    return 'vit_h'


@dataclass
class SubjectCrop(NodeRef):
    """
    Subject-aware cutout and inpaint-mask generator using YOLO / MediaPipe / SAM.

    ``SubjectCrop`` detects a region of interest (person / face / head / eyes) and
    produces either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image (``mode='mask'`` or
    ``mode='negative-mask'``).

    The node is intended to support workflows such as:

    - extracting subjects for compositing (e.g. with ``ImageStack``),
    - producing inpaint masks for SDXL pipelines,
    - extracting head regions for FaceID / IP-Adapter refinement,
    - refining a small region by cropping, processing it separately, and reinserting
      it at the original coordinates.

    Pipeline
    --------
    The node combines several models to robustly localize a subject region:

    - MediaPipe Pose is always executed first to obtain body landmarks.
      These landmarks provide a geometric prior for locating the subject
      and estimating the head region.

    - YOLO (COCO class 0) proposes candidate person bounding boxes when
      ``target`` is ``'person'`` or ``'head'``.

    - Pose landmarks are used to select the most plausible person bounding
      box among YOLO detections.

    - A coarse *head area* is derived from the pose landmarks. This region
      approximates the subject head location and is used to improve the
      robustness of face and eye detection when the face is small relative
      to the full image.

    - MediaPipe Face Landmarker is executed inside the head area to obtain
      accurate face or eye landmarks.

    - Segment Anything (SAM) is used to produce segmentation masks when
      needed (person / face / head targets).

    ``target='person'``
        - MediaPipe Pose landmarks are computed for the full image.
        - YOLO proposes candidate person bounding boxes.
        - Pose landmarks are used to select the bbox that best matches the
        detected body.
        - The bbox may be expanded using ``box_margin``.
        - SAM segments inside the selected bbox.
        - The final crop region corresponds to the selected person bbox.

    ``target='face'``
        - MediaPipe Pose landmarks are used to estimate a coarse head area.
        - MediaPipe Face Landmarker runs inside this head area.
        - The largest detected face bbox is remapped to full-image coordinates.
        - SAM segments using the face bbox.

    ``target='head'``
        - MediaPipe Pose landmarks are computed.
        - YOLO proposes person bounding boxes.
        - Pose landmarks select the most plausible person bbox.
        - A head area is derived from pose landmarks.
        - Face landmarks are detected within the head area.
        - A square head crop box is computed from the face bbox using
        ``expansion`` and an upward bias to preserve hair.
        - SAM segmentation is still guided by the person bbox for robustness,
          while the crop region corresponds to the derived head box.

    ``target='eyes'``
        - MediaPipe Pose landmarks estimate a head area.
        - MediaPipe Face Landmarker runs inside that head area.
        - Eye landmarks are extracted and converted to a bounding box.
        - A landmark-derived mask is produced directly (SAM is skipped).

    ``target='left-eye'``
        - MediaPipe derives face landmarks.
        - A bounding box is computed from landmark indices corresponding to the
          subject's left eye (subject perspective, not viewer).
        - The box is optionally expanded via ``expansion``.
        - A landmark-derived mask is used directly; SAM is skipped.
        - The final crop region isolates only the left eye.

    ``target='right-eye'``
        - MediaPipe derives face landmarks.
        - A bounding box is computed from landmark indices corresponding to the
          subject's right eye (subject perspective, not viewer).
        - The box is optionally expanded via ``expansion``.
        - A landmark-derived mask is used directly; SAM is skipped.
        - The final crop region isolates only the right eye.

    Head crop geometry
    ------------------
    Let ``face_size = max(face_w, face_h)`` from the landmark-derived face box.

    The head crop box is defined as a square centered on the face, shifted upward
    to preserve hair:

    - vertical shift: ``cy -= 0.15 * face_size``
    - radius: ``radius = 0.75 * face_size * expansion``

    Parameters
    ----------
    id : str, optional
        Node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default input
        (``input['default']['image']`` or ``input['default']['path']``).

    spec : dict or str or Path, optional
        Node specification (inline dict or path to a config file), resolved via
        ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``sam_checkpoint`` : str
                Path to the SAM checkpoint. Required.
            ``sam_model_type`` : {'vit_h', 'vit_l', 'vit_b'}, optional
                SAM backbone type. If omitted, inferred from the checkpoint
                filename.
            ``device`` : str, optional
                Inference device (e.g. ``'cuda'``, ``'cuda:0'``, ``'cpu'``).
                Default: ``'cuda'``.
            ``yolo_model`` : str, optional
                YOLO weights. Required for ``target='person'`` and ``target='head'``.
            ``face_landmarker_task`` : str, optional
                MediaPipe FaceLandmarker ``.task`` path. Required for
                ``target='face'``, ``target='eyes'``, ``target='left-eye'``,
                ``target='right-eye'`` and ``target='head'``.
            ``pose_landmarker_task`` : str
                MediaPipe PoseLandmarker ``.task`` path.
                This model is required because pose landmarks are used to derive
                head regions and to select the correct person bbox.

        ``params`` : dict
            ``target`` : {'person', 'face', 'head', 'eyes', 'left-eye', 'right-eye'}, optional
                Region to extract. Default: ``'person'``.

            ``mode`` : {'default', 'mask', 'negative-mask'}, optional
                Output type:
                - ``'default'``: RGBA cutout.
                - ``'mask'``: full-frame 8-bit mask (white = selected region).
                - ``'negative-mask'``: inverted full-frame mask (white = background).
                Default: ``'default'``.

            ``crop_mode`` : {'bbox', 'trim', 'full_frame'}, optional
                Applies only when ``mode='default'`` and controls the spatial layout
                of the RGBA cutout:

                - ``'bbox'``:
                    Output is the rectangular crop inside the selected bounding box,
                    including the original background; alpha is fully opaque
                    (255 everywhere).

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
                YOLO confidence threshold (only used when YOLO runs).
                Typical range: 0.2-0.6. Default: 0.35.

            ``box_margin`` : float, optional
                Symmetric expansion ratio applied to the selected bounding box before
                SAM, expressed as a fraction of bbox size.
                Typical range: 0.05-0.20. Default: 0.12.

            ``multimask`` : bool, optional
                If True, SAM returns multiple candidate masks and the node selects
                one via a heuristic based on bbox-center inclusion, coverage, and
                SAM score. Default: True.

            ``expansion`` : float, optional
                Head square expansion multiplier for ``target='head'`` and box
                expansion factor for eye targets. Default: 1.0.

            ``dilate_radius`` : int, optional
                Mask dilation radius in pixels (mask modes only).
                Useful to avoid edge artifacts in inpainting. Default: 0.

            ``close_radius`` : int, optional
                Morphological closing radius in pixels (mask modes only).
                Fills small holes and gaps. Default: 0.

            ``smoothing_radius`` : int, optional
                Gaussian smoothing radius in pixels (mask modes only).
                Produces softer mask edges. Default: 0.

        ``debug`` : dict
            ``save_debug`` : bool, optional
                If True, saves a debug image with the selected SAM bbox overlay.
                Default: False.

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
        ``bbox_xyxy`` : list[int]
            Bounding box used to run SAM.
        ``params`` : dict
            Configuration parameters.
        ``crop`` : dict
            Crop metadata useful for reinsertion/compositing:

            ``anchor_xy`` : list[int]
                Center of the selected crop box in absolute source-image coordinates.
            ``bbox_size`` : list[int]
                Width and height of the crop box in pixels: ``[b_width, b_height]``.

        ``metadata`` : str
            JSON sidecar path.

    Notes
    -----
    - Mask outputs are always full-frame and aligned to the original image size.
    - For ``target='head'``, SAM uses the selected person bbox, while the final crop
      region is the derived head box.
    - For eye targets, the mask is derived directly from face landmarks and SAM is
      skipped.
    - 'left-eye' and 'right-eye' refer to the subject perspective.
      In mirrored images this may appear inverted to the viewer.
    - Heavy models (YOLO, SAM, MediaPipe Tasks) are retrieved via the global model
      cache where available.
    - The SAM predictor is created per run because it stores per-image state.
    - If you change code or spec and need fresh outputs, delete the existing sidecar
      JSON to avoid reusing cached results.
    """

    # Either pass a path explicitly, or wire an upstream image into default input.
    path: Optional[Union[str, Path]] = None

    # Optional node spec (device, etc.)
    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        # Local imports to avoid hard deps if node unused
        import cv2
        import torch
        from segment_anything import SamPredictor

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
        out_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Load image
        # -----------------------------
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(
                f"SubjectCrop node '{node_id}': cannot read image: {img_path}")

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        eye_which: Optional[str] = None
        eye_mask: Optional[np.ndarray] = None

        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        pose_xy = mp_pose_landmarks_xy(
            img_rgb=img_rgb,
            pose_landmarker=pose_landmarker,
        )

        if cfg.target == 'person':
            # -----------------------------
            # YOLO: find largest 'person' bbox (COCO class 0)
            # -----------------------------
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device
            )[0]

            person_boxes_xyxy = person_bboxes_xyxy(res, node_id)

            bx1, by1, bx2, by2 = select_person_bbox_xyxy(
                person_boxes_xyxy,
                pose_xy=pose_xy,
            )

        elif cfg.target == 'head':
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device,
            )[0]

            person_boxes_xyxy = person_bboxes_xyxy(res, node_id)

            bx1, by1, bx2, by2 = select_person_bbox_xyxy(
                person_boxes_xyxy,
                pose_xy=pose_xy,
            )

            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            head_area_rgb, a_x, a_y = crop_head_area_from_pose(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=1.6,
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

        elif cfg.target == 'face':
            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            head_area_rgb, a_x, a_y = crop_head_area_from_pose(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=1.6,
            )

            face_xy = mp_face_landmarks(
                img_rgb=head_area_rgb,
                face_landmarker=landmarker,
            )
            r_x1, r_y1, r_x2, r_y2 = face_bbox_xyxy_from_landmarks(
                face_xy,
                image_shape=head_area_rgb.shape
            )

            bx1 = r_x1 + a_x
            by1 = r_y1 + a_y
            bx2 = r_x2 + a_x
            by2 = r_y2 + a_y

        elif cfg.target in ('eyes', 'left-eye', 'right-eye'):
            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            eye_which = {
                'eyes': 'both',
                'left-eye': 'left',
                'right-eye': 'right',
            }[cfg.target]

            head_area_rgb, a_x, a_y = crop_head_area_from_pose(
                img_rgb=img_rgb,
                pose_xy=pose_xy,
                expansion=1.6,
            )

            face_xy = mp_face_landmarks(
                head_area_rgb,
                face_landmarker=landmarker,
            )
            r_x1, r_y1, r_x2, r_y2 = eye_bbox_xyxy_from_landmarks(
                face_xy,
                head_area_rgb.shape,
                which=eye_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            bx1 = r_x1 + a_x
            by1 = r_y1 + a_y
            bx2 = r_x2 + a_x
            by2 = r_y2 + a_y

            local_eye_mask = eye_mask_from_landmarks(
                face_xy,
                head_area_rgb.shape,
                which=eye_which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            eye_mask = np.zeros((h, w), dtype=bool)
            eye_h, eye_w = local_eye_mask.shape
            eye_mask[a_y:a_y + eye_h, a_x:a_x + eye_w] = local_eye_mask

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        if cfg.box_margin > 0 and cfg.target not in ('eyes', 'left-eye', 'right-eye'):
            bx1, by1, bx2, by2 = expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.box_margin
            )

        # -----------------------------
        # Mask generation
        # -----------------------------
        if cfg.target in ('eyes', 'left-eye', 'right-eye'):
            # Eye targets: use landmark-derived mask (skip SAM).
            if eye_mask is None:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': eye_mask not computed."
                )
            mask = eye_mask
            ckpt = None
            model_type = None
        else:
            # Default path: SAM box-guided segmentation
            ckpt = Path(str(cfg.sam_checkpoint)).expanduser().resolve()
            if not ckpt.exists() or not ckpt.is_file():
                raise FileNotFoundError(
                    f"SubjectCrop node '{node_id}': SAM checkpoint not found: {ckpt}")

            model_type = _infer_sam_model_type(ckpt, cfg.sam_model_type)

            sam = get_sam(
                checkpoint=str(ckpt),
                model_type=model_type,
                device=cfg.device
            )

            predictor = SamPredictor(sam)
            predictor.set_image(np.ascontiguousarray(img_rgb))

            box = np.array([bx1, by1, bx2, by2], dtype=np.float32)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=box[None, :],
                    multimask_output=bool(cfg.multimask),
                )

            if masks is None or len(masks) == 0:
                raise RuntimeError(
                    f"SubjectCrop node '{node_id}': SAM returned no masks.")

            # Pick the best mask in a robust way:
            best_i = 0
            best_key = None

            for i in range(len(masks)):

                mi = masks[i].astype(bool)

                if cfg.target == 'person':

                    key = _score_sam_mask_with_landmarks(
                        mi,
                        pose_xy,
                    )

                else:
                    frac = float(mi[by1:by2, bx1:bx2].mean())
                    score_i = float(scores[i]) if scores is not None else 0.0
                    key = (score_i, -abs(frac - 0.35))

                if best_key is None or key > best_key:
                    best_key = key
                    best_i = i

            mask = masks[int(best_i)].astype(bool)

            # ------------------------------------------------------------------
            # SAM may occasionally return the complementary (background) mask
            # when prompted with a bounding box only. In that case the subject
            # appears as a hole in the mask.
            #
            # We detect this situation using MediaPipe pose landmarks: the correct
            # subject mask should contain most valid body landmarks. If fewer than
            # half of them fall inside the predicted mask, we assume SAM returned
            # the background instead.
            #
            # When flipping the mask we must restrict the inversion to the prompt
            # bounding box. Outside that region SAM predictions are undefined and
            # inverting the whole mask would incorrectly mark the entire image as
            # foreground.
            # ------------------------------------------------------------------
            valid = (
                (pose_xy[:, 0] >= 0) &
                (pose_xy[:, 1] >= 0)
            )

            pts = pose_xy[valid]

            inside = 0
            for px, py in pts:
                px = int(px)
                py = int(py)

                if mask[py, px]:
                    inside += 1

            if cfg.target == 'person':
                min_inside = max(1, int(math.ceil(len(pts) * 0.8)))
            elif cfg.target == 'head':
                min_inside = max(1, int(math.ceil(len(pts) * 0.5)))
            elif cfg.target == 'face':
                min_inside = 1
            else:
                min_inside = max(1, int(math.ceil(len(pts) * 0.5)))

            if inside < min_inside:
                # Invert mask only inside the prompt bbox to avoid turning the entire
                # background of the image into foreground.
                inv = ~mask
                new_mask = np.zeros_like(mask, dtype=bool)
                new_mask[by1:by2, bx1:bx2] = inv[by1:by2, bx1:bx2]
                mask = new_mask

        # --------------------------------------------------
        # Select crop region depending on target
        # --------------------------------------------------

        if cfg.target == 'head':
            # crop strictly around face/head bbox (full-image coordinates)
            crop_x1, crop_y1, crop_x2, crop_y2 = fx1, fy1, fx2, fy2
        else:
            # person mode
            crop_x1, crop_y1, crop_x2, crop_y2 = bx1, by1, bx2, by2

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
            if cfg.target in ('eyes', 'left-eye', 'right-eye'):
                # Keep all components in the crop.
                # For 'eyes' this preserves both eyes (two disjoint blobs).
                # For 'left-eye'/'right-eye' the crop box is expected to isolate a single eye.
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

            if cfg.crop_mode == 'bbox':
                # Include the original background inside the crop; alpha is fully opaque.
                alpha = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                crop_rgba = np.dstack([crop_rgb, alpha])
                Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

            else:
                # 'trim' and 'full_frame' -> alpha of mask
                alpha = (crop_mask.astype(np.uint8) * 255)
                crop_rgba = np.dstack([crop_rgb, alpha])

                if cfg.crop_mode == 'trim':
                    tx1, ty1, tx2, ty2 = _tight_alpha_bbox(alpha)
                    crop_rgba = crop_rgba[ty1:ty2, tx1:tx2, :]

                    out_x1 = int(crop_x1 + tx1)
                    out_y1 = int(crop_y1 + ty1)
                    out_x2 = int(crop_x1 + tx2)
                    out_y2 = int(crop_y1 + ty2)

                    Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

                elif cfg.crop_mode == 'full_frame':
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
                        f'{self.id}: invalid crop_mode={cfg.crop_mode!r}')

        else:
            # build full-frame mask (same size as source image)
            full_mask = np.zeros((h, w), dtype=np.uint8)

            # put cropped mask back into absolute position
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask.astype(
                np.uint8) * 255

            if cfg.mode == 'negative-mask':
                full_mask = 255 - full_mask

            # post-process (expand + close + smooth)
            full_mask = postprocess_mask(
                full_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            Image.fromarray(full_mask, mode='L').save(out_path)

        # Optional debug bbox overlay
        dbg_path = None
        if cfg.save_debug:
            dbg = img_rgb.copy()
            dbg_x1, dbg_y1, dbg_x2, dbg_y2 = bx1, by1, bx2, by2
            if cfg.target == 'head':
                dbg_x1, dbg_y1, dbg_x2, dbg_y2 = fx1, fy1, fx2, fy2
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
            'selected_bbox_xyxy': [int(bx1), int(by1), int(bx2), int(by2)],
            'model': {
                **({} if cfg.yolo_model is None else {'yolo': cfg.yolo_model}),
                **({} if ckpt is None else {
                    'sam_checkpoint': str(ckpt),
                }),
                **({} if model_type is None else {
                    'sam_model_type': model_type,
                }),
                **({} if cfg.face_landmarker_task is None else {
                    'face_landmarker_task': cfg.face_landmarker_task,
                }),
                **({} if cfg.pose_landmarker_task is None else {
                    'pose_landmarker_task': cfg.pose_landmarker_task,
                }),
                'device': cfg.device,
            },
            'params': {
                'target': cfg.target,
                'mode': cfg.mode,
                'crop_mode': cfg.crop_mode,
                'conf': cfg.conf,
                'box_margin': cfg.box_margin,
                'multimask': cfg.multimask,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
            },
            'crop': {
                'anchor_xy': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
            }
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
