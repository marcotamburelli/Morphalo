from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import (get_mediapipe_face_landmarker,
                                   get_mediapipe_pose_landmarker)
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.geometry import (offset_bbox_xyxy,
                                            offset_landmarks_xy)
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (CropModeSpec, SizeExpr,
                                             expand_bbox_toward_ratio,
                                             parse_crop_mode,
                                             read_shape_cleanup_config,
                                             validate_size_expr)
from morphalo.nodes.preprocess.utils.geometry import (
    expand_clip_bbox_by_size_expr, tight_mask_bbox)
from morphalo.nodes.preprocess.utils.mask_ops import (
    cleanup_shape_mask, cleanup_shape_mask_by_parts, prepare_output_mask)
from morphalo.nodes.preprocess.utils.mask_selection import \
    select_image_side_mask_candidate
from morphalo.nodes.preprocess.utils.sapiens2_seg import (SAPIENS2_CLASSES,
                                                         predict_segments)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.face_region import (eye_mask_from_landmarks,
                                               eyebrow_mask_from_landmarks,
                                               face_bbox_xyxy_from_landmarks,
                                               face_side_of_jaw_mask,
                                               mp_face_landmarks)
from morphalo.nodes.vision.human import (crop_head_area_from_pose,
                                         mp_pose_landmarks_xy)

FACE_BBOX_EXPANSION = 1.15
HEAD_AREA_EXPANSION = 1.6
FACE_TARGETS = (
    'face',
    'face-neck',
    'eyes',
    'left-eye',
    'right-eye',
    'anatomical-left-eye',
    'anatomical-right-eye',
    'eyebrows',
    'left-eyebrow',
    'right-eyebrow',
    'anatomical-left-eyebrow',
    'anatomical-right-eyebrow',
)

IMAGE_RELATIVE_FACE_TARGETS = {
    'left-eye': ('eye', 'left'),
    'right-eye': ('eye', 'right'),
    'left-eyebrow': ('eyebrow', 'left'),
    'right-eyebrow': ('eyebrow', 'right'),
}

ANATOMICAL_FACE_TARGETS = {
    'anatomical-left-eye': ('eye', 'left'),
    'anatomical-right-eye': ('eye', 'right'),
    'anatomical-left-eyebrow': ('eyebrow', 'left'),
    'anatomical-right-eyebrow': ('eyebrow', 'right'),
}


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    segment_model: str
    mode: str
    crop_mode: Optional[CropModeSpec]
    box_margin: SizeExpr
    chin_margin: float
    save_debug: bool
    dilate_radius: int
    close_radius: int
    target: str
    expansion: float
    face_landmarker_task: str
    pose_landmarker_task: str
    smoothing_radius: int
    shape_cleanup: dict[str, Any]


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
    if target not in FACE_TARGETS:
        expected = ', '.join(repr(t) for t in FACE_TARGETS)
        raise ValueError(
            f"'{node_id}': invalid target={target!r} "
            f'(expected one of: {expected})'
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

    segment_model = str(
        model.get('segment_model', 'facebook/sapiens2-seg-0.4b')
    )

    box_margin = params.get('box_margin', '12%')
    validate_size_expr(box_margin)

    chin_margin = float(params.get('chin_margin', 0.03))
    if chin_margin < 0:
        raise ValueError(
            f"'{node_id}': invalid chin_margin={chin_margin!r} "
            '(expected >= 0)'
        )

    return Config(
        device=device,
        dtype=dtype,
        segment_model=segment_model,
        mode=mode,
        crop_mode=crop_mode,
        box_margin=box_margin,
        chin_margin=chin_margin,
        save_debug=bool(debug.get('save_debug', False)),
        dilate_radius=int(params.get('dilate_radius', 0)),
        close_radius=int(params.get('close_radius', 0)),
        target=target,
        expansion=expansion,
        face_landmarker_task=str(face_landmarker_task),
        pose_landmarker_task=str(pose_landmarker_task),
        smoothing_radius=int(params.get('smoothing_radius', 0)),
        shape_cleanup=read_shape_cleanup_config(
            params.get('postprocess', None),
            node_id=node_id,
        ),
    )


def _full_frame_mask(
    local_mask: np.ndarray,
    *,
    full_shape: tuple[int, int],
    offset_xy: tuple[int, int],
) -> np.ndarray:
    """
    Paste a head-area-local feature mask into full-image coordinates.

    Parameters
    ----------
    local_mask : np.ndarray
        Boolean mask in the pose-guided head-area coordinate system.

    full_shape : tuple[int, int]
        Full image shape as ``(height, width)``.

    offset_xy : tuple[int, int]
        ``(x, y)`` offset of the local head area inside the full image.

    Returns
    -------
    np.ndarray
        Boolean full-frame mask.
    """
    full_h, full_w = full_shape
    off_x, off_y = offset_xy
    local_h, local_w = local_mask.shape

    out = np.zeros((full_h, full_w), dtype=bool)
    out[off_y:off_y + local_h, off_x:off_x + local_w] = local_mask
    return out


def _face_feature_mask_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    kind: str,
    which: str,
    expansion: float,
) -> np.ndarray:
    """
    Build an eye or eyebrow feature mask from face landmarks.

    Parameters
    ----------
    face_xy : np.ndarray
        MediaPipe face landmarks in the same coordinate system as
        ``image_shape``.

    image_shape : tuple[int, ...]
        Image shape for the landmark coordinate system.

    kind : {'eye', 'eyebrow'}
        Facial feature family to mask.

    which : {'left', 'right', 'both'}
        Anatomical landmark group requested from the MediaPipe face topology.

    expansion : float
        Feature-mask expansion factor.

    Returns
    -------
    np.ndarray
        Boolean feature mask.
    """
    if kind == 'eye':
        return eye_mask_from_landmarks(
            face_xy,
            image_shape,
            which=which,
            expansion=expansion,
        )
    if kind == 'eyebrow':
        return eyebrow_mask_from_landmarks(
            face_xy,
            image_shape,
            which=which,
            expansion=expansion,
        )
    raise ValueError(f'Invalid facial feature kind={kind!r}.')


def _semantic_face_mask(
    segments: np.ndarray,
    *,
    face_bbox: tuple[int, int, int, int],
    jaw_keep_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """
    Build a facial mask from a Sapiens2 semantic label map.

    Both public targets combine the Sapiens2 ``face-neck`` class with lips,
    teeth, and tongue so that the mouth is part of the visible facial region.
    Mouth pixels are restricted to ``face_bbox`` so labels elsewhere in the
    image are not included. No connected-component or subject selection is
    performed.

    With ``jaw_keep_mask=None``, the combined mask implements ``face-neck``.
    When a jaw mask is provided, every semantic part is intersected with the
    face side of the MediaPipe jaw boundary to implement ``face``.

    Parameters
    ----------
    segments : np.ndarray
        Two-dimensional Sapiens2 integer label map in full-image coordinates.

    face_bbox : tuple[int, int, int, int]
        End-exclusive full-image face box ``(x1, y1, x2, y2)`` derived from
        MediaPipe landmarks. The box limits mouth-class selection for both
        targets. It does not restrict the Sapiens2 ``face-neck`` class.

    jaw_keep_mask : np.ndarray or None
        Boolean full-frame mask containing the facial side of the MediaPipe jaw
        curve. ``None`` returns the combined face and neck without jaw clipping.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], list[str]]
        ``(mask, part_masks, label_names)`` where ``mask`` is the combined
        boolean facial mask, ``part_masks`` contains each non-empty semantic
        part for independent structural cleanup, and ``label_names`` records
        the Sapiens2 classes considered for the target.

    Raises
    ------
    RuntimeError
        If the Sapiens2 label map contains no ``face-neck`` pixels.

    Notes
    -----
    The function assumes a facial close-up containing one relevant face. Since
    the complete ``face-neck`` label is retained, multiple people in the label
    map may contribute to the output.
    """
    face_neck = segments == int(SAPIENS2_CLASSES['face-neck'])
    if not np.any(face_neck):
        raise RuntimeError('Sapiens2 returned an empty face-neck mask.')

    x1, y1, x2, y2 = face_bbox
    face_support = np.zeros_like(face_neck, dtype=bool)
    face_support[y1:y2, x1:x2] = True

    label_names = [
        'face-neck',
        'lower-lip',
        'upper-lip',
        'lower-teeth',
        'upper-teeth',
        'tongue',
    ]
    part_masks = [face_neck]
    for label_name in label_names[1:]:
        part_masks.append(
            (segments == int(SAPIENS2_CLASSES[label_name]))
            & face_support
        )
    if jaw_keep_mask is not None:
        part_masks = [part & jaw_keep_mask for part in part_masks]
    part_masks = [part for part in part_masks if np.any(part)]
    return np.logical_or.reduce(part_masks), part_masks, label_names


@dataclass
class FaceCrop(CudaPostRunMixin, NodeRef):
    """
    Face-aware crop and inpaint-mask generator using MediaPipe Pose,
    MediaPipe Face Landmarker, and Sapiens2 semantic segmentation.

    ``FaceCrop`` detects facial regions of interest and produces either:

    - an RGBA cutout (``mode='default'``), or
    - a full-frame inpaint mask aligned to the input image
      (``mode='mask'`` or ``mode='negative-mask'``).

    Intended input
    --------------
    This preprocessor is intended for facial close-ups and portraits where one
    face occupies a substantial part of the image. It does not perform person
    detection or subject selection. In images containing multiple people,
    Sapiens2 may assign ``face-neck`` to more than one person and those regions
    may be combined in the output. Use a subject-selection or person-cropping
    stage before ``FaceCrop`` when the relevant face is not already dominant.

    Supported targets are:

    - ``face``: Sapiens2 facial semantics clipped below the MediaPipe jaw;
    - ``face-neck``: Sapiens2 face, mouth, and neck semantics without jaw clipping;
    - ``eyes``: landmark-derived mask for both eyes;
    - ``left-eye``: landmark-derived mask for the eye on the left side of the image;
    - ``right-eye``: landmark-derived mask for the eye on the right side of the image;
    - ``anatomical-left-eye``: landmark-derived mask for the anatomical left eye;
    - ``anatomical-right-eye``: landmark-derived mask for the anatomical right eye;
    - ``eyebrows``: landmark-derived mask for both eyebrows;
    - ``left-eyebrow``: landmark-derived mask for the eyebrow on the left side of
      the image;
    - ``right-eyebrow``: landmark-derived mask for the eyebrow on the right side
      of the image.
    - ``anatomical-left-eyebrow``: landmark-derived mask for the anatomical left
      eyebrow;
    - ``anatomical-right-eyebrow``: landmark-derived mask for the anatomical
      right eyebrow.

    Side-specific targets use image/viewer perspective:

    - ``left-eye`` and ``left-eyebrow`` refer to the region on the left side of
      the image;
    - ``right-eye`` and ``right-eyebrow`` refer to the region on the right side
      of the image.

    Anatomical side targets keep the underlying MediaPipe landmark-side
    convention and do not compare image position.

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
    - Downstream crop, mask, segmentation, and metadata logic operate in
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
        - Sapiens2 supplies ``face-neck`` and mouth-part semantic masks.
        - The jaw arc from landmarks 172 through 152 to 397 is connected at
          both ends to the nearest Sapiens2 mask boundary, with ``chin_margin``
          preserving a small amount of space below the chin.
        - The final crop region is derived from the selected mask and then
          optionally expanded using ``box_margin``.

    ``target='eyes'``
        - MediaPipe Pose landmarks are computed for the full image.
        - A coarse head / upper-body search area is derived from pose landmarks.
        - MediaPipe Face Landmarker runs inside this search area.
        - Eye landmarks are selected in search-area local coordinates.
        - A landmark-derived bbox and mask are computed for both eyes.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - The final crop region isolates both eyes.

    ``target='left-eye'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Anatomical left/right eye masks are computed from landmarks and
          optionally expanded via ``expansion``.
        - The mask whose bbox center appears leftmost in the image is selected.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - The final crop region isolates only the left eye in image/viewer
          perspective.

    ``target='right-eye'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Anatomical left/right eye masks are computed from landmarks and
          optionally expanded via ``expansion``.
        - The mask whose bbox center appears rightmost in the image is selected.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
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
        - The final crop region isolates both eyebrows.

    ``target='left-eyebrow'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Anatomical left/right eyebrow masks are computed from landmarks and
          optionally expanded via ``expansion``.
        - The mask whose bbox center appears leftmost in the image is selected.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
        - The final crop region isolates only the left eyebrow in image/viewer
          perspective.

    ``target='right-eyebrow'``
        - MediaPipe derives face landmarks inside the pose-guided head search
          area.
        - Anatomical left/right eyebrow masks are computed from landmarks and
          optionally expanded via ``expansion``.
        - The mask whose bbox center appears rightmost in the image is selected.
        - The bbox is remapped to full-image coordinates.
        - The local mask is pasted into a full-frame boolean mask.
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
    Both ``target='face-neck'`` and ``target='face'`` combine the Sapiens2
    ``face-neck`` class with lips, teeth, and tongue. ``target='face'`` then
    intersects every part with the face side of the MediaPipe jaw curve.

    Eye and eyebrow targets rely directly on landmark-derived
    masks. This is intentional: these regions are small, geometrically
    well-defined, and may naturally consist of multiple disconnected components.

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
                Torch dtype used to load the Sapiens2 segmentation model.
                Supported values follow ``resolve_dtype`` conventions, e.g.
                ``'bf16'``, ``'float16'`` or ``'float32'``.
                Default: ``'bf16'``.

            ``segment_model`` : str, optional
                Hugging Face Sapiens2 model used by ``face`` and ``face-neck``.
                Default: ``'facebook/sapiens2-seg-0.4b'``.

            ``pose_landmarker_task`` : str
                MediaPipe PoseLandmarker ``.task`` path. Required for all
                targets.

                Pose landmarks are used to derive the coarse head / upper-body
                search area before running MediaPipe Face Landmarker.

            ``face_landmarker_task`` : str
                MediaPipe FaceLandmarker ``.task`` path. Required for all
                targets.

        ``params`` : dict
            ``target`` : {'face', 'face-neck', 'eyes', 'left-eye', 'right-eye',
            'anatomical-left-eye', 'anatomical-right-eye', 'eyebrows',
            'left-eyebrow', 'right-eyebrow', 'anatomical-left-eyebrow',
            'anatomical-right-eyebrow'}, optional
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

            ``box_margin`` : int or str, optional
                Symmetric margin applied to the final crop bbox derived from
                the post-processed target mask. Supported forms follow the
                standard size-expression convention: integer pixels,
                ``'<n>px'`` or ``'<n>%'``. Percentages are resolved against the
                mask bbox width for left/right and mask bbox height for
                top/bottom.

                This margin is applied only to cropped default outputs
                (``crop_mode='trim'``, ``'bbox'`` or ``'bbox[w:h]'``). It is
                ignored for ``mode='mask'``, ``mode='negative-mask'`` and
                ``crop_mode='full_frame'`` because those outputs preserve the
                full source frame.

                Default: ``'12%'``.

            ``chin_margin`` : float, optional
                Fraction of the landmark forehead-to-chin distance retained
                below the jaw boundary for ``target='face'``. Default: 0.03.

            ``postprocess`` : dict, optional
                Structural cleanup applied to the selected facial silhouette
                before deriving crop geometry, RGBA alpha, or full-frame mask
                output. This block defines the canonical shape used by the node,
                so it is applied in every mode. Output-only mask refinements
                such as ``close_radius``, ``dilate_radius`` and
                ``smoothing_radius`` are applied later and only for
                ``mode='mask'`` or ``mode='negative-mask'``.

                Processing order is fixed: ``fill_holes`` ->
                ``morph_open_radius`` -> ``min_component_area``.
                For logical multi-part targets such as ``eyes`` and
                ``eyebrows``, cleanup is applied independently to each target
                part before the parts are unioned; therefore
                ``min_component_area='biggest'`` keeps the largest component per
                eye/eyebrow, not one eye/eyebrow globally.

                ``fill_holes`` : int, float, str, 'all' or None, optional
                    Fill enclosed background holes inside the selected shape
                    before removing thin details. ``0`` or ``None`` disables
                    hole filling. ``'all'`` fills every enclosed hole. Numeric
                    values are pixel areas. Percentage strings such as ``'1%'``
                    follow the shared component-area convention: the percentage
                    is measured on the image long side and squared into an area
                    threshold. Only holes with area less than or equal to the
                    resolved threshold are filled.

                ``morph_open_radius`` : int, optional
                    Radius in pixels for morphological opening, applied after
                    hole filling. Opening removes thin lines, speckles, and
                    small bridges while preserving surviving larger regions.
                    ``0`` disables this step.

                ``min_component_area`` : int, float, str, 'biggest' or None, optional
                    Remove disconnected foreground components after hole filling
                    and opening. ``0`` or ``None`` disables component filtering.
                    Numeric values are pixel areas. Percentage strings use the
                    same long-side area convention as ``fill_holes``.
                    ``'biggest'`` keeps only the largest connected component,
                    useful for a single facial target but potentially
                    destructive for legitimate multi-part targets such as both
                    eyes or both eyebrows.

                Default: all disabled.

            ``expansion`` : float, optional
                Expansion factor applied to landmark-derived feature geometry.

                - for eye targets:
                  expands the landmark-derived eye bbox / mask;
                - for eyebrow targets:
                  expands the landmark-derived eyebrow bbox / mask;
                Face and face-neck targets use the fixed internal
                ``FACE_BBOX_EXPANSION`` for local facial geometry.

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
            ``segment_model`` when Sapiens2 is used, MediaPipe task paths,
            device, and dtype.

        ``crop`` : dict
            Crop metadata useful for reinsertion/compositing:

            ``anchor_xy`` : list[int]
                Local anchor inside the output crop image.

            ``position`` : list[int]
                Source-image position where ``anchor_xy`` should be placed when
                reconstructing the original geometry.

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
    - ``target='face'`` and ``target='face-neck'`` use Sapiens2.
    - Eye and eyebrow targets do not load Sapiens2.
    - For eye and eyebrow targets, the selected mask is derived directly from
      MediaPipe face landmarks.
    - Eye and eyebrow masks may contain multiple disconnected components by
      design.
    - Side-specific targets use image/viewer perspective.
    - Heavy models are retrieved via the global model cache where available.
    - The segmentation backend is loaded through the shared Sapiens2 model
      cache.
    - For ``crop_mode='bbox[w:h]'``, the requested aspect ratio is treated as a
      target, not a hard guarantee. Near the image boundaries the final crop may
      deviate from the requested ratio.
    - If you change code or spec and need fresh outputs, delete the existing
      sidecar JSON to avoid reusing cached results.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        target = spec.get('params', {}).get('target', 'face')
        return (
            target in ('face', 'face-neck')
            and is_cuda_device(spec.get('model', {}).get('device', 'cuda'))
        )

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

        # Eye and eyebrow targets use landmark geometry directly. Face targets
        # combine the same landmarks with Sapiens2 semantic segmentation.
        feature_mask: Optional[np.ndarray] = None
        shape_part_masks: Optional[np.ndarray | list[np.ndarray]] = None
        resolved_labels: list[str] = []
        segment_model_id: Optional[str] = None
        runtime_dtype = cfg.dtype

        if cfg.target in ('face', 'face-neck'):
            # Compute the face bbox in head-area local coordinates, then remap it
            # to full-image coordinates for semantic face selection.
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
        elif cfg.target in (
            'eyes',
            'left-eye',
            'right-eye',
            'anatomical-left-eye',
            'anatomical-right-eye',
            'eyebrows',
            'left-eyebrow',
            'right-eyebrow',
            'anatomical-left-eyebrow',
            'anatomical-right-eyebrow',
        ):
            if cfg.target.startswith('eyebrow') or 'eyebrow' in cfg.target:
                feature_kind = 'eyebrow'
                both_target = 'eyebrows'
            else:
                feature_kind = 'eye'
                both_target = 'eyes'

            expansion = max(1.0, float(cfg.expansion))

            left_mask = _full_frame_mask(
                _face_feature_mask_from_landmarks(
                    face_xy_local,
                    head_area_rgb.shape,
                    kind=feature_kind,
                    which='left',
                    expansion=expansion,
                ),
                full_shape=(h, w),
                offset_xy=(a_x, a_y),
            )
            right_mask = _full_frame_mask(
                _face_feature_mask_from_landmarks(
                    face_xy_local,
                    head_area_rgb.shape,
                    kind=feature_kind,
                    which='right',
                    expansion=expansion,
                ),
                full_shape=(h, w),
                offset_xy=(a_x, a_y),
            )

            if cfg.target == both_target:
                feature_mask = left_mask | right_mask
                shape_part_masks = [left_mask, right_mask]
            elif cfg.target in ANATOMICAL_FACE_TARGETS:
                _, anatomical_side = ANATOMICAL_FACE_TARGETS[cfg.target]
                feature_mask = (
                    left_mask if anatomical_side == 'left' else right_mask
                )
                shape_part_masks = feature_mask
            else:
                _, image_side = IMAGE_RELATIVE_FACE_TARGETS[cfg.target]
                selection = select_image_side_mask_candidate(
                    [
                        (f'anatomical-left-{feature_kind}', left_mask),
                        (f'anatomical-right-{feature_kind}', right_mask),
                    ],
                    image_side=image_side,
                    node_id=node_id,
                    target_name=cfg.target,
                    error_prefix='FaceCrop',
                )
                feature_mask = (
                    left_mask
                    if selection.selected_index == 0
                    else right_mask
                )
                shape_part_masks = feature_mask

            bx1, by1, bx2, by2 = tight_mask_bbox(
                feature_mask.astype(np.uint8)
            )

        else:
            raise ValueError(f"'{node_id}': invalid target={cfg.target!r}")

        if cfg.target in ('face', 'face-neck'):
            segment_model_id = cfg.segment_model
            segments, runtime_dtype = predict_segments(
                Image.fromarray(img_rgb),
                model_id=segment_model_id,
                device=cfg.device,
                dtype=cfg.dtype,
                error_prefix='FaceCrop',
            )
            mask, semantic_parts, resolved_labels = _semantic_face_mask(
                segments,
                face_bbox=(bx1, by1, bx2, by2),
                jaw_keep_mask=None,
            )
            if cfg.target == 'face':
                jaw_keep_mask = face_side_of_jaw_mask(
                    face_xy_global,
                    img_rgb.shape,
                    support_mask=mask,
                    margin_ratio=cfg.chin_margin,
                )
                mask &= jaw_keep_mask
                semantic_parts = [
                    part & jaw_keep_mask
                    for part in semantic_parts
                    if np.any(part & jaw_keep_mask)
                ]
            shape_part_masks = semantic_parts

        else:
            if feature_mask is None:
                raise RuntimeError(
                    f"FaceCrop node '{node_id}': feature mask not computed."
                )
            mask = feature_mask
            if shape_part_masks is None:
                shape_part_masks = feature_mask

        if shape_part_masks is not None:
            shape_mask = cleanup_shape_mask_by_parts(
                mask,
                shape_part_masks,
                **cfg.shape_cleanup,
            )
        else:
            shape_mask = cleanup_shape_mask(mask, **cfg.shape_cleanup)
        shape_mask_u8 = shape_mask.astype(np.uint8) * 255

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        crop_x1 = 0
        crop_y1 = 0
        crop_x2 = w
        crop_y2 = h

        if cfg.mode == 'default' and cfg.crop_mode is not None:
            if cfg.crop_mode.mode != 'full_frame':
                crop_x1, crop_y1, crop_x2, crop_y2 = tight_mask_bbox(
                    shape_mask.astype(np.uint8)
                )
                crop_x1, crop_y1, crop_x2, crop_y2 = expand_clip_bbox_by_size_expr(
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2,
                    w,
                    h,
                    cfg.box_margin,
                )

                if (
                    cfg.crop_mode.mode == 'bbox'
                    and cfg.crop_mode.ratio is not None
                ):
                    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox_toward_ratio(
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                        full_w=w,
                        full_h=h,
                        ratio=cfg.crop_mode.ratio,
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
                alpha = shape_mask_u8[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_rgba = np.dstack([crop_rgb, alpha])

                if cfg.crop_mode.mode == 'trim':
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
            full_mask = prepare_output_mask(
                shape_mask,
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
                **({} if segment_model_id is None else {
                    'segment_model': segment_model_id,
                }),
                'face_landmarker_task': cfg.face_landmarker_task,
                'pose_landmarker_task': cfg.pose_landmarker_task,
                'device': cfg.device,
                'dtype': str(runtime_dtype).replace('torch.', ''),
            },
            'params': {
                'target': cfg.target,
                'mode': cfg.mode,
                'crop_mode': None if cfg.crop_mode is None else cfg.crop_mode.raw,
                'box_margin': cfg.box_margin,
                'chin_margin': cfg.chin_margin,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'expansion': cfg.expansion,
                'smoothing_radius': cfg.smoothing_radius,
                'postprocess': cfg.shape_cleanup,
            },
            **({} if not resolved_labels else {
                'segmentation': {'labels': resolved_labels},
            }),
            'crop': {
                'anchor_xy': [
                    int(anchor_x - out_x1),
                    int(anchor_y - out_y1),
                ],
                'position': [anchor_x, anchor_y],
                'bbox_size': [b_width, b_height],
                'bbox_xyxy': [int(out_x1), int(out_y1), int(out_x2), int(out_y2)],
            },
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
