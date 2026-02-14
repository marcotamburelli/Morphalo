from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from stability.cache.models import (get_mediapipe_face_landmarker, get_sam,
                                    get_yolo)
from stability.core.paths import make_node_output_path
from stability.dag import NodeRef
from stability.nodes.common.config_resolve import resolve_spec
from stability.nodes.common.io import write_json_sidecar
from stability.nodes.preprocess.utils import postprocess_mask
from stability.nodes.sdxl_resolve import resolve_single_image_path

"""
Note: per far funzionare bene il crop della testa bisogna prendere il crop del soggetto e ritagliare solo un intorno sufficentemente ampio della faccia.

problema: il crop non prende l'immagine ripulita dal contorno, ma crean una immgine nel box che ha preso.
"""


@dataclass
class Config:
    device: str
    yolo_model: str
    sam_checkpoint: str
    sam_model_type: Optional[str]
    mode: str
    conf: float
    box_margin: float
    multimask: bool
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: Optional[str]
    smoothing_radius: int


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    sam_checkpoint = model.get('sam_checkpoint')
    device = model.get('device', 'cuda')

    if not sam_checkpoint:
        raise ValueError(
            f"'{node_id}': Missing 'sam_checkpoint' from model configuration."
        )

    sam_model_type = model.get('sam_model_type')

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': Invalid mode={mode!r}")

    conf = float(params.get('conf', 0.35))
    box_margin = float(params.get('box_margin', 0.12))
    multimask = bool(params.get('multimask', True))

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))

    target = str(params.get('target', 'person'))
    if target not in ('person', 'face', 'head'):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'person', 'face', or 'head')"
        )

    expansion = float(params.get('expansion', 1.0))
    if expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} (expected >= 0)"
        )

    save_debug = bool(debug.get('save_debug', False))

    face_landmarker_task = None
    yolo_model = None

    if target in ('person', 'head'):
        yolo_model = str(model.get('yolo_model', 'yolov8n.pt'))

    if target in ('face', 'head'):
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
        sam_checkpoint=str(sam_checkpoint),
        sam_model_type=None if sam_model_type is None else str(sam_model_type),
        mode=mode,
        conf=conf,
        box_margin=box_margin,
        multimask=multimask,
        save_debug=save_debug,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        target=target,
        expansion=expansion,
        face_landmarker_task=face_landmarker_task,
        smoothing_radius=smoothing_radius,
    )


def _largest_person_bbox_xyxy(res, node_id: str) -> tuple[int, int, int, int]:
    if res.boxes is None or len(res.boxes) == 0:
        raise RuntimeError(f'{node_id}: YOLO found no boxes.')

    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.detach().cpu().numpy()

    person_idx = np.where(cls == 0)[0]
    if person_idx.size == 0:
        raise RuntimeError(f"{node_id}: YOLO found no 'person' detections.")

    best_i, best_area = None, -1.0

    for i in person_idx:
        x1, y1, x2, y2 = xyxy[i]
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if area > best_area:
            best_area = area
            best_i = int(i)

    x1, y1, x2, y2 = xyxy[best_i].tolist()

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _mp_largest_face_bbox_xyxy(
    img_rgb: np.ndarray,
    face_landmarker_task,
    # face_min_score: Optional[float] = None,
) -> tuple[int, int, int, int]:
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    result = face_landmarker_task.detect(mp_image)
    face_landmarks_list = result.face_landmarks or []
    # face_presence_scores = result.face_presence_scores or []

    if not face_landmarks_list:
        raise RuntimeError('MediaPipe found no face landmarks.')

    candidates = []

    for landmarks in face_landmarks_list:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x1 = max(0, int(round(min(xs) * w)))
        y1 = max(0, int(round(min(ys) * h)))
        x2 = min(w, int(round(max(xs) * w)))
        y2 = min(h, int(round(max(ys) * h)))

        if x2 > x1 and y2 > y1:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, x1, y1, x2, y2))

    if not candidates:
        # fallback: ignore score filtering
        for landmarks in face_landmarks_list:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]

            x1 = max(0, int(round(min(xs) * w)))
            y1 = max(0, int(round(min(ys) * h)))
            x2 = min(w, int(round(max(xs) * w)))
            y2 = min(h, int(round(max(ys) * h)))

            if x2 > x1 and y2 > y1:
                area = (x2 - x1) * (y2 - y1)
                candidates.append((area, x1, y1, x2, y2))

    if not candidates:
        raise RuntimeError('Could not derive valid face bbox from landmarks.')

    # pick largest face
    _, x1, y1, x2, y2 = max(candidates, key=lambda t: t[0])

    return x1, y1, x2, y2


def _expand_clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float) -> tuple[int, int, int, int]:
    # assume x2,y2 are end-exclusive after this normalization
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bbox: {(x1, y1, x2, y2)}")

    bw, bh = (x2 - x1), (y2 - y1)
    dx = int(round(bw * margin))
    dy = int(round(bh * margin))

    bx1 = max(0, x1 - dx)
    by1 = max(0, y1 - dy)
    bx2 = min(w, x2 + dx)
    by2 = min(h, y2 + dy)

    if bx2 <= bx1 or by2 <= by1:
        raise RuntimeError(f"Invalid expanded bbox: {(bx1, by1, bx2, by2)}")

    return bx1, by1, bx2, by2


def _head_bbox_square_from_face(
    x1: int, y1: int, x2: int, y2: int,
    w: int, h: int,
    expansion: float,
) -> tuple[int, int, int, int]:

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid face bbox: {(x1, y1, x2, y2)}")

    fw = x2 - x1
    fh = y2 - y1
    face_size = max(fw, fh)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # ---- upward bias (hair priority) ----
    cy -= 0.15 * face_size

    radius = max(1.0, 0.75 * face_size * float(expansion))

    bx1 = int(math.floor(cx - radius))
    by1 = int(math.floor(cy - radius))
    bx2 = int(math.ceil(cx + radius))
    by2 = int(math.ceil(cy + radius))

    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(w, bx2)
    by2 = min(h, by2)

    if bx2 <= bx1 or by2 <= by1:
        raise RuntimeError(f"Invalid square head bbox: {(bx1, by1, bx2, by2)}")

    return bx1, by1, bx2, by2


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
    Subject-aware cropper and mask generator using YOLO, MediaPipe and
    Segment Anything (SAM).

    This node extracts either a full person, a face, or a head region from an
    input image and optionally produces a corresponding segmentation mask.

    The pipeline combines:

    - YOLO (COCO class 0) for person bounding-box detection
    - MediaPipe Face Landmarker for face localization
    - Segment Anything (SAM) for box-guided segmentation

    Depending on the selected ``target``, the workflow differs:

    * ``target="person"``
        - Detect largest person via YOLO
        - Expand bounding box
        - Segment using SAM
        - Crop to person bounding box

    * ``target="face"``
        - Detect face via MediaPipe landmarks
        - Use face bounding box directly
        - Segment using SAM inside that box
        - Crop to face bounding box

    * ``target="head"``
        - Detect person via YOLO
        - Detect face via MediaPipe
        - Build a square head bounding box centered on the face
          (with upward bias to preserve hair)
        - Segment full person via SAM
        - Crop mask and image to head bounding box

    The square head box is computed as::

        radius = 0.75 * face_size * expansion

    and vertically shifted upward to prioritize hair coverage.

    Parameters
    ----------
    id : str, optional
        Unique node identifier inside the DAG.

    path : str or Path, optional
        Path to the input image. If omitted, the image is resolved from
        upstream node input (``input["default"]["image"]`` or ``["path"]``).

    spec : dict or str or Path, optional
        Node specification dictionary or configuration file.

        Expected structure:

        ``model`` : dict
            ``sam_checkpoint`` : str
                Path to SAM checkpoint file.
            ``sam_model_type`` : str, optional
                One of ``{"vit_h", "vit_l", "vit_b"}``.
                If omitted, inferred from checkpoint filename.
            ``yolo_model`` : str, optional
                YOLO weights (required for ``"person"`` and ``"head"``).
            ``face_landmarker_task`` : str, optional
                MediaPipe FaceLandmarker ``.task`` file
                (required for ``"face"`` and ``"head"``).
            ``device`` : str
                Torch device (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).

        ``params`` : dict
            ``target`` : {"person", "face", "head"}
                Region to extract.
            ``mode`` : {"default", "mask", "negative-mask"}
                Output format.
            ``conf`` : float
                YOLO confidence threshold.
            ``box_margin`` : float
                Expansion ratio for person bounding box.
            ``expansion`` : float
                Head expansion multiplier (for ``"head"``).
                Values > 1 enlarge the square head region.
            ``multimask`` : bool
                Enable SAM multimask output.
            ``dilate_radius`` : int
                Mask dilation radius (mask modes only).
            ``close_radius`` : int
                Morphological closing radius (mask modes only).
            ``smoothing_radius`` : int
                Gaussian smoothing radius (mask modes only).

        ``debug`` : dict
            ``save_debug`` : bool
                If True, saves an image with detected bounding box overlay.

    Attributes
    ----------
    op : str
        Operator identifier derived from the concrete node class name (e.g.
        ``"img2img"``).

    Output
    -------
    dict
        Output metadata dictionary containing:

        ``ok`` : bool
            Success flag.
        ``node`` : str
            Operator name.
        ``id`` : str
            Node identifier.
        ``image`` : str
            Path to generated image or mask.
        ``bbox_xyxy`` : list[int]
            Person bounding box used for SAM.
        ``metadata`` : str
            Path to JSON sidecar.

    Output Modes
    ------------
    ``"default"``
        Saves an RGBA cutout. Alpha channel corresponds to the
        SAM-derived mask cropped to the selected region.

    ``"mask"``
        Saves an 8-bit grayscale full-frame mask aligned to the
        original image dimensions.

    ``"negative-mask"``
        Same as ``"mask"``, but inverted (white = background).

    Notes
    -----
    - SAM segmentation is always performed using the person bounding box
      (when available) to maximize robustness.
    - The head region is computed geometrically from the detected face
      and is not segmented independently.
    - Mask outputs preserve original image spatial coordinates.
    - Heavy models (YOLO, SAM) are retrieved via the global model cache.
    - The JSON sidecar acts as a persistent cache; delete it if the
      implementation or configuration changes.
    """

    # Either pass a path explicitly, or wire an upstream image into default input.
    path: Optional[Union[str, Path]] = None

    # Optional node spec (device, etc.)
    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

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

            bx1, by1, bx2, by2 = _largest_person_bbox_xyxy(res, node_id)

        elif cfg.target == 'face':
            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            bx1, by1, bx2, by2 = _mp_largest_face_bbox_xyxy(
                img_rgb=img_rgb,
                face_landmarker_task=landmarker,
                # face_min_score=cfg.face_min_score,
            )

        elif cfg.target == 'head':
            yolo = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
            res = yolo.predict(
                img_rgb,
                conf=float(cfg.conf),
                verbose=False,
                device=cfg.device
            )[0]

            landmarker = get_mediapipe_face_landmarker(
                model_asset_path=cfg.face_landmarker_task,
                device=cfg.device,
            )

            fx1, fy1, fx2, fy2 = _mp_largest_face_bbox_xyxy(
                img_rgb=img_rgb,
                face_landmarker_task=landmarker,
            )
            # Additional heuristic to capture hair/crowns:
            fx1, fy1, fx2, fy2 = _head_bbox_square_from_face(
                fx1, fy1, fx2, fy2, w, h,
                expansion=cfg.expansion,
            )

            bx1, by1, bx2, by2 = _largest_person_bbox_xyxy(res, node_id)

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        if cfg.box_margin > 0:
            bx1, by1, bx2, by2 = _expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.box_margin
            )

        # -----------------------------
        # SAM: box-guided segmentation
        # -----------------------------
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
        # SAM multimask can sometimes return a "background" mask as the top score.
        # We prefer a mask that contains the bbox center and has a reasonable coverage.
        cx = int((bx1 + bx2) // 2)
        cy = int((by1 + by2) // 2)

        best_i = 0
        best_key = None

        for i in range(len(masks)):
            mi = masks[i].astype(bool)

            # Fraction of bbox covered by the mask (0..1)
            frac = float(mi[by1:by2, bx1:bx2].mean())

            # Center of bbox should belong to the subject most of the time
            center_in = bool(mi[cy, cx])

            # Reject masks that are too tiny or almost the whole bbox
            ok_area = (0.05 <= frac <= 0.95)

            score_i = float(scores[i]) if scores is not None else 0.0

            # Priority order:
            # 1) contains center, 2) reasonable area, 3) higher score,
            # 4) prefer coverage near ~0.35 (heuristic)
            key = (center_in, ok_area, score_i, -abs(frac - 0.35))

            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        mask = masks[int(best_i)].astype(bool)  # HxW

        # Final safety: if bbox center is not inside the mask, invert it
        if not mask[cy, cx]:
            mask = ~mask

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

        if cfg.mode == 'default':
            # RGB crop aligned with selected crop region (person / face / head)
            crop_rgb = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2, :]

            # Alpha from mask (same spatial region)
            alpha = crop_mask.astype(np.uint8) * 255

            rgba = np.dstack([crop_rgb, alpha])
            Image.fromarray(rgba, mode='RGBA').save(out_path)
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
            cv2.rectangle(dbg, (bx1, by1), (bx2, by2), (255, 0, 0), 3)
            dbg_path = out_path.with_name(out_path.stem + '_debug_bbox.png')
            Image.fromarray(dbg).save(dbg_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'bbox_xyxy': [int(bx1), int(by1), int(bx2), int(by2)],
            'model': {
                **({} if cfg.yolo_model is None else {'yolo': cfg.yolo_model}),
                'sam_checkpoint': str(ckpt),
                'sam_model_type': model_type,
                **({} if cfg.face_landmarker_task is None else {
                    'face_landmarker_task': cfg.face_landmarker_task,
                }),
                'device': cfg.device,
            },
            'params': {
                'conf': float(cfg.conf),
                'box_margin': float(cfg.box_margin),
                'multimask': bool(cfg.multimask),
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'target': cfg.target,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
            },
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
