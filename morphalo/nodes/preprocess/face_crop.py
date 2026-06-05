import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import (get_mediapipe_face_landmarker,
                                   get_mediapipe_pose_landmarker, get_sam)
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
                                             offset_bbox_xyxy,
                                             offset_landmarks_xy,
                                             parse_crop_mode,
                                             positive_points_for_sam,
                                             postprocess_mask,
                                             tight_alpha_bbox)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.face_region import (
    eye_bbox_xyxy_from_landmarks, eye_mask_from_landmarks,
    eyebrow_bbox_xyxy_from_landmarks, eyebrow_mask_from_landmarks,
    face_bbox_xyxy_from_landmarks, mp_face_landmarks)
from morphalo.nodes.vision.human import (crop_head_area_from_pose,
                                         mp_pose_landmarks_xy)

FACE_BBOX_EXPANSION = 1.15
HEAD_AREA_EXPANSION = 1.6


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    sam_model: Optional[str]
    mode: str
    crop_mode: Optional[CropModeSpec]
    box_margin: float
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: str
    pose_landmarker_task: str
    smoothing_radius: int


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'bf16')))

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': Invalid mode={mode!r}")

    crop_mode = (
        parse_crop_mode(params.get('crop_mode', 'trim'), node_id=node_id)
        if mode == 'default'
        else None
    )

    target = str(params.get('target', 'face'))
    if target not in (
        'face',
        'eyes',
        'left-eye',
        'right-eye',
        'eyebrows',
        'left-eyebrow',
        'right-eyebrow',
    ):
        raise ValueError(
            f"'{node_id}': invalid target={target!r} (expected 'face', 'eyes', "
            "'left-eye', 'right-eye', 'eyebrows', 'left-eyebrow' or "
            "'right-eyebrow')"
        )

    face_landmarker_task = model.get('face_landmarker_task')
    if not face_landmarker_task:
        raise ValueError(
            f"'{node_id}': Missing 'model.face_landmarker_task' (MediaPipe .task path)"
        )
    pose_landmarker_task = model.get('pose_landmarker_task')
    if not pose_landmarker_task:
        raise ValueError(
            f"'{node_id}': Missing 'model.pose_landmarker_task' "
            '(MediaPipe .task path)'
        )

    expansion = float(params.get('expansion', 1.0))
    if expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid expansion={expansion!r} (expected >= 0)"
        )

    sam_model = None
    if target == 'face':
        sam_model = str(model.get('sam_model', 'facebook/sam-vit-large'))

    return Config(
        device=device,
        dtype=dtype,
        sam_model=sam_model,
        mode=mode,
        crop_mode=crop_mode,
        box_margin=float(params.get('box_margin', 0.12)),
        save_debug=bool(debug.get('save_debug', False)),
        dilate_radius=int(params.get('dilate_radius', 0)),
        close_radius=int(params.get('close_radius', 0)),
        target=target,
        expansion=expansion,
        face_landmarker_task=str(face_landmarker_task),
        pose_landmarker_task=str(pose_landmarker_task),
        smoothing_radius=int(params.get('smoothing_radius', 0)),
    )


@dataclass
class FaceCrop(NodeRef):
    """
    Face-aware crop and inpaint-mask generator using MediaPipe Pose,
    MediaPipe Face Landmarker, and optional SAM-compatible segmentation.

    ``FaceCrop`` detects facial regions of interest and produces either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image
      (``mode='mask'`` or ``mode='negative-mask'``).

    Supported targets are:

    - ``face``: expanded face bbox segmented with SAM;
    - ``eyes``: landmark-derived mask for both eyes;
    - ``left-eye``: landmark-derived mask for the eye on the left side of the image;
    - ``right-eye``: landmark-derived mask for the eye on the right side of the image;
    - ``eyebrows``: landmark-derived mask for both eyebrows;
    - ``left-eyebrow``: landmark-derived mask for the eyebrow on the left side of
      the image;
    - ``right-eyebrow``: landmark-derived mask for the eyebrow on the right side
      of the image.

    Side-specific targets use image/viewer perspective:

    - ``left-eye`` and ``left-eyebrow`` refer to the region on the left side of
      the image;
    - ``right-eye`` and ``right-eyebrow`` refer to the region on the right side
      of the image.

    This node intentionally focuses on facial details. Body-level targets such
    as ``person``, ``head``, ``hands``, ``left-hand`` and ``right-hand`` are
    handled by ``SubjectCrop``.

    Pipeline
    --------
    The node preserves the historical pose-guided facial detection behavior that
    was previously implemented inside ``SubjectCrop``:

    - MediaPipe Pose is executed first on the full image.
    - A coarse head / upper-body search area is derived from pose landmarks.
    - MediaPipe Face Landmarker is executed inside this pose-guided search area
      instead of on the full image.
    - Face landmarks are first obtained in head-area local coordinates.
    - Local face landmarks and local bboxes are remapped to full-image
      coordinates.
    - Downstream crop, mask, SAM prompting, and metadata logic operate in
      full-image coordinates.

    This pose-guided search area improves robustness when the face is small
    relative to the full image, while still keeping the facial geometry local
    and precise.

    ``target='face'``
        - MediaPipe Pose landmarks are computed for the full image.
        - A coarse head / upper-body search area is derived from pose landmarks.
        - MediaPipe Face Landmarker runs inside this search area.
        - A landmark-derived face bbox is computed in search-area local
          coordinates.
        - The face bbox is expanded internally by ``FACE_BBOX_EXPANSION`` to
          avoid overly tight crops around internal facial landmarks.
        - The expanded local face bbox is remapped to full-image coordinates.
        - Face landmarks are also remapped to full-image coordinates.
        - ``box_margin`` optionally expands the full-image SAM prompt bbox.
        - SAM segments the face using the expanded face bbox and a sparse subset
          of positive face-landmark points.
        - A face-landmark coverage check is used to detect occasional SAM
          polarity mistakes. If too few face landmarks fall inside the selected
          mask, the mask is inverted locally inside the prompt bbox.
        - The final crop region corresponds to the expanded full-image face bbox.

    ``target='eyes'``
        - MediaPipe Pose landmarks are computed for the full image.
        - A coarse head / upper-body search area is derived from pose landmarks.
        - MediaPipe Face Landmarker runs inside this search area.
        - Eye landmarks are selected in search-area local coordinates.
        - A landmark-derived bbox and mask are computed for both eyes.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - SAM is skipped.
        - The final crop region isolates both eyes.

    ``target='left-eye'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Landmark indices corresponding to the eye on the left side of the image
          are selected.
        - A local eye bbox and mask are computed from landmarks and optionally
          expanded via ``expansion``.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - SAM is skipped.
        - The final crop region isolates only the left eye in image/viewer
          perspective.

    ``target='right-eye'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Landmark indices corresponding to the eye on the right side of the
          image are selected.
        - A local eye bbox and mask are computed from landmarks and optionally
          expanded via ``expansion``.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - SAM is skipped.
        - The final crop region isolates only the right eye in image/viewer
          perspective.

    ``target='eyebrows'``
        - MediaPipe Pose landmarks are computed for the full image.
        - A coarse head / upper-body search area is derived from pose landmarks.
        - MediaPipe Face Landmarker runs inside this search area.
        - Eyebrow landmarks are selected in search-area local coordinates.
        - A landmark-derived bbox and mask are computed for both eyebrows.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - SAM is skipped.
        - The final crop region isolates both eyebrows.

    ``target='left-eyebrow'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Landmark indices corresponding to the eyebrow on the left side of the
          image are selected.
        - A local eyebrow bbox and mask are computed from landmarks and
          optionally expanded via ``expansion``.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - SAM is skipped.
        - The final crop region isolates only the left eyebrow in image/viewer
          perspective.

    ``target='right-eyebrow'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Landmark indices corresponding to the eyebrow on the right side of the
          image are selected.
        - A local eyebrow bbox and mask are computed from landmarks and
          optionally expanded via ``expansion``.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - SAM is skipped.
        - The final crop region isolates only the right eyebrow in image/viewer
          perspective.

    Coordinate conventions
    ----------------------
    Face landmarks are detected in the pose-guided head search area, so their
    first coordinate system is local to that crop. Bboxes and masks for eyes and
    eyebrows are also computed locally.

    Before any downstream crop or mask operation:

    - local bboxes are translated to full-image coordinates;
    - local facial landmarks are translated to full-image coordinates when
      needed;
    - local eye / eyebrow masks are pasted into full-frame boolean masks.

    This keeps the later crop, mask, connected-component, and output logic
    target-agnostic.

    Target-specific segmentation
    ----------------------------
    ``target='face'`` uses SAM because a landmark-tight face box often captures
    only internal facial structure and may not describe the visible face
    silhouette well.

    Eye and eyebrow targets skip SAM and rely directly on landmark-derived
    masks. This is intentional: these regions are small, geometrically
    well-defined, and may naturally consist of multiple disconnected components.

    For connected-component cleanup:

    - feature targets keep all non-background connected components, because both
      eyes or both eyebrows may be intentionally disjoint;
    - the full face target keeps a single connected component to avoid attaching
      unrelated fragments.

    Parameters
    ----------
    name : str, optional
        Node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default
        input (``input['default']['image']`` or ``input['default']['path']``).

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
                ``'bf16'``, ``'float16'`` or ``'float32'``.
                Default: ``'bf16'``.

            ``sam_model`` : str, optional
                Hugging Face SAM-compatible model identifier used for full-face
                mask generation. Default: ``'facebook/sam-vit-large'``.

                This parameter is used only when ``target='face'``. Eye and
                eyebrow targets use landmark-derived masks and skip SAM.

            ``pose_landmarker_task`` : str
                MediaPipe PoseLandmarker ``.task`` path. Required for all
                targets.

                Pose landmarks are used to derive the coarse head / upper-body
                search area before running MediaPipe Face Landmarker.

            ``face_landmarker_task`` : str
                MediaPipe FaceLandmarker ``.task`` path. Required for all
                targets.

        ``params`` : dict
            ``target`` : {'face', 'eyes', 'left-eye', 'right-eye',
            'eyebrows', 'left-eyebrow', 'right-eyebrow'}, optional
                Facial region to extract. Default: ``'face'``.

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
                  Same as ``'bbox'``, but the selected crop box is expanded
                  toward the requested aspect ratio ``w:h`` while keeping the
                  target fully inside the crop and staying within source-image
                  bounds.

                  The requested ratio is treated as a target, not a hard
                  constraint. If the source image does not provide enough room
                  near the borders, the final crop may deviate from the
                  requested ratio.

                - ``'trim'``:
                  Output is an RGBA cutout cropped to the selected bounding box,
                  with alpha derived from the mask. The result is then tightly
                  trimmed to the minimal box containing non-transparent pixels.

                - ``'full_frame'``:
                  Same cutout as ``'trim'`` but placed back into a full-size RGBA
                  canvas of the original image dimensions, preserving the
                  original coordinates.

                Default: ``'trim'``.

            ``box_margin`` : float, optional
                Symmetric expansion ratio applied to the full-face SAM prompt
                bbox, expressed as a fraction of bbox size.

                This parameter is used only for ``target='face'``. Eye and
                eyebrow targets are controlled by landmark geometry and
                ``expansion``.

                Typical range: 0.05-0.20. Default: 0.12.

            ``expansion`` : float, optional
                Expansion factor applied to landmark-derived feature geometry.

                - for eye targets:
                  expands the landmark-derived eye bbox / mask;
                - for eyebrow targets:
                  expands the landmark-derived eyebrow bbox / mask;
                - for ``target='face'``:
                  the landmark-derived face bbox is expanded internally by
                  ``FACE_BBOX_EXPANSION`` and can then be expanded further by
                  ``box_margin``.

                Default: ``1.0``.

            ``dilate_radius`` : int, optional
                Mask dilation radius in pixels.

                In ``mode='mask'``, dilation expands the repaintable selected
                region. In ``mode='negative-mask'``, dilation is applied before
                inversion, so it expands the protected selected region and
                creates a safety margin between the selected region and the
                repaintable background.

                Default: 0.

            ``close_radius`` : int, optional
                Morphological closing radius in pixels.

                Closing is applied before optional inversion. It fills small
                holes and gaps in the selected facial mask. This is often useful
                for stable inpainting, but in ``mode='negative-mask'`` it also
                means that small background holes inside the selected region
                become protected after inversion.

                Default: 0.

            ``smoothing_radius`` : int, optional
                Gaussian smoothing radius in pixels.

                Smoothing is applied before optional inversion. In
                ``mode='mask'``, it softens the repaintable selected region. In
                ``mode='negative-mask'``, it softens the protected selected
                boundary before inversion, producing a feathered transition
                between protected facial detail and repaintable background.

                Default: 0.

        ``debug`` : dict
            ``save_debug`` : bool, optional
                If True, saves a debug image with the selected target bbox
                overlay. Default: False.

    Mask post-processing
    --------------------
    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the selected facial mask
    is post-processed before it is written to disk.

    Post-processing is always applied to the positive selected mask first,
    before any optional polarity inversion:

    1. the selected region is assembled as a full-frame positive mask;
    2. ``close_radius``, ``dilate_radius``, and ``smoothing_radius`` are
       applied;
    3. if ``mode='negative-mask'``, the post-processed mask is inverted.

    This ordering is intentional.

    For ``mode='mask'``, the output mask directly marks the selected facial
    region as repaintable. Dilation and smoothing therefore expand and soften
    the selected region itself.

    For ``mode='negative-mask'``, the output mask marks everything outside the
    selected facial region as repaintable and protects the selected region.
    Applying dilation and smoothing before inversion creates a protected safety
    band around the selected region.

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
            Resolved model/runtime metadata, including the selected
            ``sam_model`` when SAM is used, MediaPipe task paths, device, and
            dtype.

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
    - ``target='face'`` uses SAM; eye and eyebrow targets skip SAM.
    - ``model.sam_model`` is ignored for eye and eyebrow targets.
    - For ``target='face'``, SAM prompt points use full-image face landmark
      coordinates, not local head-area coordinates.
    - For eye and eyebrow targets, the selected mask is derived directly from
      MediaPipe face landmarks.
    - Eye and eyebrow masks may contain multiple disconnected components by
      design.
    - Side-specific targets use image/viewer perspective.
    - Heavy models are retrieved via the global model cache where available.
    - The segmentation backend is loaded through Hugging Face Transformers via
      ``get_sam(...)`` and cached as a ``(processor, model)`` pair.
    - This node does not require a local ``sam_checkpoint`` / ``sam_model_type``
      pair. Use ``model.sam_model`` to select the Hugging Face model id.
    - For ``crop_mode='bbox[w:h]'``, the requested aspect ratio is treated as a
      target, not a hard guarantee. Near the image boundaries the final crop may
      deviate from the requested ratio.
    - If you change code or spec and need fresh outputs, delete the existing
      sidecar JSON to avoid reusing cached results.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        import cv2

        spec = resolve_spec(self.spec)
        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )
        out_dir = Path(output_dir)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(
                f"FaceCrop node '{node_id}': cannot read image: {img_path}"
            )

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run pose first and derive a coarse head / upper-body search area before
        # running the Face Landmarker. This preserves the historical SubjectCrop
        # behavior and improves robustness when the face is small in the full image.
        pose_landmarker = get_mediapipe_pose_landmarker(
            model_asset_path=cfg.pose_landmarker_task,
            device=cfg.device,
        )
        pose_xy = mp_pose_landmarks_xy(
            img_rgb=img_rgb,
            pose_landmarker=pose_landmarker,
        )

        landmarker = get_mediapipe_face_landmarker(
            model_asset_path=cfg.face_landmarker_task,
            device=cfg.device,
        )

        head_area_rgb, a_x, a_y = crop_head_area_from_pose(
            img_rgb=img_rgb,
            pose_xy=pose_xy,
            expansion=HEAD_AREA_EXPANSION,
        )

        face_xy_local = mp_face_landmarks(
            img_rgb=head_area_rgb,
            face_landmarker=landmarker,
        )

        face_xy_global = offset_landmarks_xy(
            face_xy_local,
            dx=a_x,
            dy=a_y,
        )

        # Feature targets such as eyes and eyebrows skip SAM and use a
        # landmark-derived full-frame mask. The full face target uses SAM instead.
        feature_mask: Optional[np.ndarray] = None

        if cfg.target == 'face':
            # Compute the face bbox in head-area local coordinates, then remap it to
            # full-image coordinates for SAM prompting and crop metadata.
            r_x1, r_y1, r_x2, r_y2 = face_bbox_xyxy_from_landmarks(
                face_xy_local,
                image_shape=head_area_rgb.shape,
                expansion=FACE_BBOX_EXPANSION,
            )

            bx1, by1, bx2, by2 = offset_bbox_xyxy(
                (r_x1, r_y1, r_x2, r_y2),
                dx=a_x,
                dy=a_y,
            )

        # Eye / eyebrow geometry is computed in the pose-guided head area, then
        # remapped to full-image coordinates. The mask is pasted into a full-frame
        # boolean mask so the downstream crop/mask logic can stay target-agnostic.
        elif cfg.target in ('eyes', 'left-eye', 'right-eye'):
            which = {
                'eyes': 'both',
                'left-eye': 'left',
                'right-eye': 'right',
            }[cfg.target]

            r_x1, r_y1, r_x2, r_y2 = eye_bbox_xyxy_from_landmarks(
                face_xy_local,
                head_area_rgb.shape,
                which=which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            bx1, by1, bx2, by2 = offset_bbox_xyxy(
                (r_x1, r_y1, r_x2, r_y2),
                dx=a_x,
                dy=a_y,
            )

            local_feature_mask = eye_mask_from_landmarks(
                face_xy_local,
                head_area_rgb.shape,
                which=which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            feature_mask = np.zeros((h, w), dtype=bool)
            mh, mw = local_feature_mask.shape
            feature_mask[a_y:a_y + mh, a_x:a_x + mw] = local_feature_mask

        elif cfg.target in ('eyebrows', 'left-eyebrow', 'right-eyebrow'):
            which = {
                'eyebrows': 'both',
                'left-eyebrow': 'left',
                'right-eyebrow': 'right',
            }[cfg.target]

            r_x1, r_y1, r_x2, r_y2 = eyebrow_bbox_xyxy_from_landmarks(
                face_xy_local,
                head_area_rgb.shape,
                which=which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            bx1, by1, bx2, by2 = offset_bbox_xyxy(
                (r_x1, r_y1, r_x2, r_y2),
                dx=a_x,
                dy=a_y,
            )

            local_feature_mask = eyebrow_mask_from_landmarks(
                face_xy_local,
                head_area_rgb.shape,
                which=which,
                expansion=max(1.0, float(cfg.expansion)),
            )

            feature_mask = np.zeros((h, w), dtype=bool)
            mh, mw = local_feature_mask.shape
            feature_mask[a_y:a_y + mh, a_x:a_x + mw] = local_feature_mask

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        # Only the full face target uses SAM, so box_margin expands only the face SAM
        # prompt bbox. Eye and eyebrow targets are controlled by landmark expansion.
        if cfg.box_margin > 0 and cfg.target == 'face':
            bx1, by1, bx2, by2 = expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.box_margin
            )

        if cfg.target == 'face':
            sam_model_id = cfg.sam_model
            if sam_model_id is None:
                raise RuntimeError(
                    f"FaceCrop node '{node_id}': SAM model is not configured."
                )

            processor, sam_model = get_sam(
                model_id=sam_model_id,
                device=cfg.device,
                dtype=cfg.dtype,
            )

            # SAM expects full-image coordinates. Use the remapped face landmarks, not the
            # local head-area landmarks.
            point_coords, point_labels = positive_points_for_sam(
                xy=face_xy_global,
                bbox=(bx1, by1, bx2, by2),
                max_points=16,
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
                    f"FaceCrop node '{node_id}': SAM returned no masks."
                )

            best_mask = None
            best_key = None

            for i in range(len(masks)):
                score_i = (
                    float(scores[i])
                    if scores is not None and i < len(scores)
                    else 0.0
                )
                if best_key is None or score_i > best_key:
                    best_key = score_i
                    best_mask = masks[i].astype(bool)

            if best_mask is None:
                raise RuntimeError(
                    f"FaceCrop node '{node_id}': failed to select a SAM mask."
                )

            mask = best_mask

            # SAM can occasionally return the local background instead of the face region.
            # Use face-landmark coverage as a polarity sanity check and invert only inside
            # the prompt bbox if too few landmarks are covered.
            valid = (
                (face_xy_global[:, 0] >= 0) &
                (face_xy_global[:, 1] >= 0)
            )
            pts = face_xy_global[valid]

            if len(pts) > 0:
                inside = 0
                mask_h, mask_w = mask.shape

                for px, py in pts:
                    px = int(px)
                    py = int(py)

                    if 0 <= px < mask_w and 0 <= py < mask_h and mask[py, px]:
                        inside += 1

                min_inside = max(1, int(math.ceil(len(pts) * 0.25)))

                if inside < min_inside:
                    mask = invert_mask_inside_box(
                        mask=mask,
                        box=(bx1, by1, bx2, by2),
                    )

        else:
            if feature_mask is None:
                raise RuntimeError(
                    f"FaceCrop node '{node_id}': feature mask not computed."
                )
            mask = feature_mask
            sam_model_id = None

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
                f"FaceCrop node '{node_id}': empty crop after bbox."
            )

        cm = crop_mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm, connectivity=8
        )

        # Eye and eyebrow masks may intentionally contain multiple disconnected
        # components. Keep all feature components, but reduce the full face mask to a
        # single component to avoid attaching unrelated fragments.
        if num > 1:
            if cfg.target != 'face':
                crop_mask = (labels != 0)
            else:
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
                alpha = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                crop_rgba = np.dstack([crop_rgb, alpha])
                Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

            else:
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
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = (
                crop_mask.astype(np.uint8) * 255
            )

            full_mask = postprocess_mask(
                full_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                full_mask = 255 - full_mask

            Image.fromarray(full_mask, mode='L').save(out_path)

        dbg_path = None
        if cfg.save_debug:
            dbg = img_rgb.copy()
            cv2.rectangle(
                dbg,
                (bx1, by1),
                (bx2, by2),
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
                **({} if sam_model_id is None else {
                    'sam_model': sam_model_id,
                }),
                'face_landmarker_task': cfg.face_landmarker_task,
                'pose_landmarker_task': cfg.pose_landmarker_task,
                'device': cfg.device,
                'dtype': str(cfg.dtype).replace('torch.', ''),
            },
            'params': {
                'target': cfg.target,
                'mode': cfg.mode,
                'crop_mode': None if cfg.crop_mode is None else cfg.crop_mode.raw,
                'box_margin': cfg.box_margin,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
            },
            'crop': {
                'anchor_xy': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
                'bbox_xyxy': [int(out_x1), int(out_y1), int(out_x2), int(out_y2)],
            },
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
