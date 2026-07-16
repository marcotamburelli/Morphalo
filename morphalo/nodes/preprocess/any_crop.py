from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import get_grounding_dino, get_sam
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (CropModeSpec, SizeExpr,
                                             expand_bbox_toward_ratio,
                                             parse_crop_mode,
                                             read_shape_cleanup_config,
                                             validate_size_expr)
from morphalo.nodes.preprocess.utils.geometry import (
    expand_clip_bbox, expand_clip_bbox_by_size_expr, tight_mask_bbox)
from morphalo.nodes.preprocess.utils.mask_ops import (cleanup_shape_mask,
                                                      prepare_output_mask)
from morphalo.nodes.preprocess.utils.sam import predict_sam_mask
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.wiring.mixins import PromptMixin
from morphalo.nodes.wiring.prompt import PromptBundle


@dataclass(frozen=True)
class GroundedBox:
    """
    Grounding DINO detection candidate.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        End-exclusive bounding box ``(x1, y1, x2, y2)`` in full-image coordinates.
    score : float
        Detection confidence score returned by Grounding DINO.
    label : str
        Text label associated with the detected box.
    """
    bbox: tuple[int, int, int, int]
    score: float
    label: str


@dataclass
class AnyCropConfig:
    """
    Runtime configuration for AnyCrop.

    Parameters
    ----------
    device : str
        Runtime device used for Grounding DINO and SAM.
    grounding_dtype : torch.dtype
        dtype used for Grounding DINO, default ``float32``.
    sam_dtype : torch.dtype
        dtype used for SAM, default ``bf16``.
    grounding_model : str
        Hugging Face model id for Grounding DINO.
    sam_model : str
        Hugging Face SAM-compatible model id.
    mode : {'default', 'mask', 'negative-mask'}
        Output mode.
    crop_mode : CropModeSpec | None
        Crop layout configuration, used only when ``mode == 'default'``.
    box_threshold : float
        Grounding DINO object-box confidence threshold.
    text_threshold : float
        Grounding DINO text-token confidence threshold.
    box_margin : int or str
        Symmetric crop margin applied after mask cleanup.
    prompt_expansion : float
        Symmetric bbox expansion ratio applied before SAM.
    select : {'best', 'largest', 'center', 'all'}
        Strategy used to select boxes when Grounding DINO returns multiple boxes.
    save_debug : bool
        Whether to save debug overlays. Reserved for later implementation.
    dilate_radius : int
        Mask dilation radius for full-frame mask outputs.
    close_radius : int
        Morphological closing radius for full-frame mask outputs.
    smoothing_radius : int
        Gaussian smoothing radius for full-frame mask outputs.
    """
    device: str
    grounding_dtype: torch.dtype
    sam_dtype: torch.dtype
    grounding_model: str
    sam_model: str
    mode: str
    crop_mode: Optional[CropModeSpec]
    box_threshold: float
    text_threshold: float
    box_margin: SizeExpr
    prompt_expansion: float
    select: str
    save_debug: bool
    dilate_radius: int
    close_radius: int
    smoothing_radius: int
    shape_cleanup: dict[str, Any]


def _read_any_crop_cfg(spec: dict, node_id: str) -> AnyCropConfig:
    """
    Read and validate AnyCrop configuration.

    Parameters
    ----------
    spec : dict
        Resolved node specification.
    node_id : str
        Node identifier used in validation errors.

    Returns
    -------
    AnyCropConfig
        Validated runtime configuration.
    """
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    grounding_dtype = resolve_dtype(
        str(model.get('grounding_dtype', 'float32'))
    )
    sam_dtype = resolve_dtype(
        str(model.get('sam_dtype', 'bf16'))
    )

    grounding_model = str(
        model.get('grounding_model', 'IDEA-Research/grounding-dino-tiny')
    )
    sam_model = str(model.get('sam_model', 'facebook/sam-vit-large'))

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': invalid mode={mode!r}")

    if mode == 'default':
        crop_mode = parse_crop_mode(
            params.get('crop_mode', 'trim'),
            node_id=node_id,
        )
    else:
        crop_mode = None

    box_threshold = float(params.get('box_threshold', 0.35))
    text_threshold = float(params.get('text_threshold', 0.25))
    box_margin = params.get('box_margin', '8%')
    validate_size_expr(box_margin)
    prompt_expansion = float(params.get('prompt_expansion', 0.0))

    if not (0.0 <= box_threshold <= 1.0):
        raise ValueError(
            f"'{node_id}': invalid box_threshold={box_threshold!r} "
            '(expected in [0, 1])'
        )

    if not (0.0 <= text_threshold <= 1.0):
        raise ValueError(
            f"'{node_id}': invalid text_threshold={text_threshold!r} "
            '(expected in [0, 1])'
        )

    if prompt_expansion < 0:
        raise ValueError(
            f"'{node_id}': invalid prompt_expansion={prompt_expansion!r} "
            '(expected >= 0)'
        )

    select = str(params.get('select', 'best'))
    if select not in ('best', 'largest', 'center', 'all'):
        raise ValueError(
            f"'{node_id}': invalid select={select!r} "
            "(expected 'best', 'largest', 'center', or 'all')"
        )

    return AnyCropConfig(
        device=device,
        grounding_dtype=grounding_dtype,
        sam_dtype=sam_dtype,
        grounding_model=grounding_model,
        sam_model=sam_model,
        mode=mode,
        crop_mode=crop_mode,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        box_margin=box_margin,
        prompt_expansion=prompt_expansion,
        select=select,
        save_debug=bool(debug.get('save_debug', False)),
        dilate_radius=int(params.get('dilate_radius', 0)),
        close_radius=int(params.get('close_radius', 0)),
        smoothing_radius=int(params.get('smoothing_radius', 0)),
        shape_cleanup=read_shape_cleanup_config(
            params.get('postprocess', None),
            node_id=node_id,
        ),
    )


def _grounding_text_from_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        raise ValueError('AnyCrop requires a non-empty prompt.')
    return text


def _predict_grounding_dino_boxes(
    *,
    img_rgb: np.ndarray,
    text: str,
    processor: Any,
    model: Any,
    device: str,
    dtype: torch.dtype,
    box_threshold: float,
    text_threshold: float,
) -> list[GroundedBox]:
    """
    Predict Grounding DINO detection boxes for one image and one text prompt.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image with shape ``(H, W, 3)``.
    text : str
        Text prompt used for open-vocabulary localization.
    processor : Any
        Hugging Face Grounding DINO processor.
    model : Any
        Hugging Face Grounding DINO model.
    device : str
        Runtime device used for inference.
    dtype : torch.dtype
        Floating-point dtype used for processor-generated floating tensors.
    box_threshold : float
        Box confidence threshold used during post-processing.
    text_threshold : float
        Text confidence threshold used during post-processing.

    Returns
    -------
    list[GroundedBox]
        Post-processed detection candidates with integer end-exclusive boxes,
        confidence scores, and labels.
    """
    image = Image.fromarray(np.ascontiguousarray(img_rgb))

    inputs = processor(
        images=image,
        text=text,
        return_tensors='pt',
    )

    inputs = {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value.to(device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs['input_ids'],
        threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results['boxes'].detach().float().cpu().numpy()
    scores = results['scores'].detach().float().cpu().numpy()
    labels = results['labels']

    out = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        out.append(GroundedBox(
            bbox=(
                int(np.floor(x1)),
                int(np.floor(y1)),
                int(np.ceil(x2)),
                int(np.ceil(y2)),
            ),
            score=float(score),
            label=str(label),
        ))

    return out


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h

    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def _nms_grounded_boxes(
    boxes: list[GroundedBox],
    *,
    iou_threshold: float,
) -> list[GroundedBox]:
    """
    Apply class-agnostic non-maximum suppression to Grounding DINO boxes.

    Class-agnostic suppression is intentional because flexible prompts may emit
    synonymous labels for the same physical object.
    """
    remaining = sorted(boxes, key=lambda box: box.score, reverse=True)
    kept = []

    while remaining:
        selected = remaining.pop(0)
        kept.append(selected)
        remaining = [
            candidate
            for candidate in remaining
            if _bbox_iou(selected.bbox, candidate.bbox) <= iou_threshold
        ]

    return kept


def _select_grounded_boxes(
    boxes: list[GroundedBox],
    *,
    image_shape: tuple[int, int, int],
    select: str,
) -> list[GroundedBox]:
    if not boxes:
        raise RuntimeError('AnyCrop: Grounding DINO returned no boxes.')

    if select == 'best':
        return [max(boxes, key=lambda b: b.score)]

    if select == 'largest':
        return [max(
            boxes,
            key=lambda b: (
                (b.bbox[2] - b.bbox[0])
                * (b.bbox[3] - b.bbox[1])
            ),
        )]

    if select == 'center':
        h, w = image_shape[:2]
        cx0, cy0 = w * 0.5, h * 0.5

        def dist2(b: GroundedBox) -> float:
            x1, y1, x2, y2 = b.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            return (cx - cx0) ** 2 + (cy - cy0) ** 2

        return [min(boxes, key=dist2)]

    if select == 'all':
        return _nms_grounded_boxes(boxes, iou_threshold=0.7)

    raise ValueError(f'Invalid select={select!r}')


def _clip_grounded_bbox(
    bbox: tuple[int, int, int, int],
    *,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    """
    Clip and validate a Grounding DINO bbox.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Raw bbox ``(x1, y1, x2, y2)``.
    w : int
        Image width.
    h : int
        Image height.

    Returns
    -------
    tuple[int, int, int, int]
        Valid end-exclusive bbox inside image bounds.

    Raises
    ------
    RuntimeError
        If the clipped bbox is empty or invalid.
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f'Invalid Grounding DINO bbox after clipping: {(x1, y1, x2, y2)!r}')

    return x1, y1, x2, y2


def _clean_sam_mask_for_bbox(
    mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    keep_all_components: bool,
    seed_bbox: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Restrict a SAM mask to a bbox and clean connected components.

    ``seed_bbox`` identifies the SAM prompt center when the cleanup bbox was
    expanded later to satisfy a requested crop ratio.
    """
    import cv2

    x1, y1, x2, y2 = bbox
    local_mask = mask[y1:y2, x1:x2].astype(bool)

    if local_mask.size == 0:
        raise RuntimeError('AnyCrop: empty SAM mask crop after bbox.')
    if not np.any(local_mask):
        raise RuntimeError(
            'AnyCrop: selected SAM mask is empty inside its bbox.')

    cm = local_mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cm,
        connectivity=8,
    )

    if num > 1:
        if keep_all_components:
            local_mask = labels != 0
        else:
            sx1, sy1, sx2, sy2 = seed_bbox or bbox
            local_cx = int(round(0.5 * (sx1 + sx2))) - x1
            local_cy = int(round(0.5 * (sy1 + sy2))) - y1
            local_cx = max(0, min(cm.shape[1] - 1, local_cx))
            local_cy = max(0, min(cm.shape[0] - 1, local_cy))
            target = labels[local_cy, local_cx]

            if target == 0:
                areas = stats[1:, cv2.CC_STAT_AREA]
                target = 1 + int(np.argmax(areas))

            local_mask = labels == target

    clean_mask = np.zeros_like(mask, dtype=bool)
    clean_mask[y1:y2, x1:x2] = local_mask
    return clean_mask


@dataclass
class AnyCrop(CudaPostRunMixin, PromptMixin, NodeRef):
    """
    Open-vocabulary crop and inpaint-mask generator using Grounding DINO and
    SAM/SAM-HQ segmentation.

    ``AnyCrop`` is the prompt-based counterpart of ``SubjectCrop``.

    While ``SubjectCrop`` extracts a fixed semantic or anatomical target such as
    ``'person'``, ``'head'``, ``'face'``, ``'eyes'`` or ``'hands'``,
    ``AnyCrop`` localizes an arbitrary region from a text prompt. The prompt is
    resolved through ``PromptMixin`` and passed to Grounding DINO, which predicts
    one or more open-vocabulary bounding boxes. Each selected box is then used
    as a SAM prompt to obtain a pixel-level mask.

    The node produces either:

    - an RGBA cutout when ``mode='default'``;
    - a full-frame positive inpaint mask when ``mode='mask'``;
    - a full-frame negative inpaint mask when ``mode='negative-mask'``.

    The output metadata is intentionally compatible with geometry-driven
    compositing workflows. In particular, the ``crop`` block exposes
    ``anchor_xy``, ``position``, ``bbox_size`` and ``bbox_xyxy`` so that refined
    crops can be reinserted through nodes such as ``ImageStack``.

    Prompt semantics
    ----------------
    ``AnyCrop`` uses only the primary prompt field resolved by
    :class:`PromptBundle`.

    ``prompt_2``, ``negative_prompt`` and ``negative_prompt_2`` are ignored.
    This is intentional: Grounding DINO expects localization text describing
    objects or regions, not SDXL-style dual-channel generation prompts.

    Good prompts are usually short, concrete, and object-like, for example:

    - ``'a red backpack'``
    - ``'a sword'``
    - ``'the left hand holding the glass'``
    - ``'a wooden chair'``

    Multiple classes may be expressed in a Grounding-DINO-friendly form such as
    ``'a cat. a dog.'``. Set ``params.select='all'`` to combine all distinct
    detections into one mask and crop.

    Pipeline
    --------
    The node executes the following steps:

    1. Resolve the input image from ``path`` or from the upstream default input.
    2. Resolve the primary prompt through ``PromptMixin`` / ``PromptBundle``.
    3. Run Grounding DINO on the image and prompt.
    4. Select one or more detection boxes using ``params.select``.
    5. For ``select='all'``, suppress duplicate detections with class-agnostic NMS.
    6. Clip and optionally expand each selected box using
       ``params.prompt_expansion``.
    7. Run SAM once per selected box.
    8. Select each SAM candidate with the highest predicted IoU score.
    9. Clean each local mask and combine selected masks by union.
    10. Produce either an RGBA crop or a full-frame mask, depending on ``mode``.
    11. Write the output image and JSON sidecar metadata.

    Detection and mask selection
    ----------------------------
    Grounding DINO may return multiple boxes for the same prompt. ``params.select``
    controls which detection is used:

    - ``'best'``:
        Select the detection with the highest Grounding DINO score.
    - ``'largest'``:
        Select the detection with the largest bounding-box area.
    - ``'center'``:
        Select the detection whose center is closest to the image center.
    - ``'all'``:
        Select all distinct detections after class-agnostic NMS with an IoU
        threshold of ``0.7``.

    For single-box selection, the node keeps the connected component containing
    the center of the SAM prompt box, or the largest component when the center is
    background. For ``select='all'``, all foreground components inside each
    target-local SAM box are preserved before the masks are combined.

    This cleanup step reduces accidental background fragments or detached mask
    islands while preserving the selected object region.

    Crop geometry
    -------------
    For ``mode='default'``, ``crop_mode`` controls the spatial layout of the
    produced RGBA image.

    Supported values are:

    - ``'trim'``:
        The output is an RGBA cutout cropped to the selected crop box, with alpha
        derived from the cleaned SAM mask. The result is then tightly trimmed to
        the minimal box containing non-transparent pixels.

    - ``'bbox'``:
        The output is the rectangular crop inside the selected crop box, including
        the original background. Alpha is fully opaque everywhere.

    - ``'bbox[w:h]'``:
        Same as ``'bbox'``, but the selected crop box is expanded toward the
        requested aspect ratio ``w:h`` while keeping the detected region inside
        the crop and staying within the source image bounds.

        The requested ratio is a target, not a hard guarantee. Near image borders,
        the final crop may deviate from the requested ratio.

    - ``'full_frame'``:
        The output is a full-size RGBA image aligned to the original input
        coordinates. The RGB channels contain the original image and the alpha
        channel contains the cleaned selected mask.

    For ``mode='mask'`` and ``mode='negative-mask'``, ``crop_mode`` is ignored
    because mask outputs are always full-frame.

    Mask post-processing
    --------------------
    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the cleaned positive mask is
    post-processed before it is written to disk.

    Post-processing is applied in this order:

    1. morphological closing via ``close_radius``;
    2. dilation via ``dilate_radius``;
    3. Gaussian smoothing via ``smoothing_radius``;
    4. optional inversion for ``mode='negative-mask'``.

    This means that in ``mode='negative-mask'`` the selected object is protected
    first, then the result is inverted. Dilation therefore expands the protected
    object area before inversion, creating a safety band around the object.

    Parameters
    ----------
    name : str, optional
        Node identifier within the DAG.

    path : str or Path, optional
        Input image path.

        If omitted, the node resolves the upstream default input. The upstream
        payload must contain either ``'image'`` or ``'path'``.

    spec : dict or str or Path, optional
        Node specification, either as an inline dictionary or as a path to a
        configuration file resolved by ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``device`` : str, optional
                Runtime device used for Grounding DINO and SAM inference.
                Default: ``'cuda'``.

            ``grounding_dtype`` : str, optional
                Torch dtype used when loading Grounding DINO.
                Default: ``'float32'``.

                Grounding DINO is intentionally loaded in float32 by default because the
                Transformers implementation may keep parts of the text branch in float32,
                which can cause dtype mismatches if the whole model is loaded in bf16.

            ``sam_dtype`` : str, optional
                Torch dtype used when loading the SAM-compatible segmentation model.
                Default: ``'bf16'``.

            ``grounding_model`` : str, optional
                Hugging Face Grounding DINO model identifier.
                Default: ``'IDEA-Research/grounding-dino-tiny'``.

            ``sam_model`` : str, optional
                Hugging Face SAM-compatible segmentation model identifier.
                Default: ``'facebook/sam-vit-large'``.

        ``prompt`` : str | list[str] | dict, optional
            Prompt resolved by ``PromptBundle``.

            Only the primary prompt content is used for Grounding DINO
            localization. The prompt may also be supplied by wiring an upstream
            prompt node into ``prompt:default``.

        ``params`` : dict
            ``mode`` : {'default', 'mask', 'negative-mask'}, optional
                Output type.

                - ``'default'``: write an RGBA crop/cutout.
                - ``'mask'``: write a full-frame positive mask where the selected
                  region is white.
                - ``'negative-mask'``: write a full-frame inverted mask where the
                  background is white and the selected region is protected.

                Default: ``'default'``.

            ``crop_mode`` : {'trim', 'bbox', 'bbox[w:h]', 'full_frame'}, optional
                Applies only when ``mode='default'``.

                Controls whether the output is tightly trimmed, kept as a
                rectangular bbox crop, expanded toward an aspect ratio, or returned
                as a full-frame RGBA image.

                Default: ``'trim'``.

            ``box_threshold`` : float, optional
                Grounding DINO object-box confidence threshold. Detections below
                this threshold are discarded. Expected range: ``[0, 1]``.
                Default: ``0.35``.

            ``text_threshold`` : float, optional
                Grounding DINO text-token confidence threshold. Expected range:
                ``[0, 1]``. Default: ``0.25``.

            ``box_margin`` : int or str, optional
                Symmetric margin applied to the final crop bbox derived from the
                cleaned SAM mask. Supported forms follow the standard
                size-expression convention: integer pixels, ``'<n>px'`` or
                ``'<n>%'``. Percentages are resolved against the mask bbox width
                for left/right and mask bbox height for top/bottom.

                Default: ``'8%'``.

            ``prompt_expansion`` : float, optional
                Advanced SAM prompt padding ratio. This expands selected
                Grounding DINO boxes before SAM while leaving output crop
                geometry controlled by the cleaned mask and ``box_margin``.

                Default: ``0.0``.

            ``select`` : {'best', 'largest', 'center', 'all'}, optional
                Strategy used when Grounding DINO returns multiple boxes.
                ``'all'`` combines all distinct detections into one crop/mask.
                Default: ``'best'``.

            ``postprocess`` : dict, optional
                Structural cleanup applied to the selected SAM silhouette before
                deriving crop geometry, RGBA alpha, or full-frame mask output.
                This block defines the canonical shape used by the node, so it
                is applied in every mode. Output-only mask refinements such as
                ``close_radius``, ``dilate_radius`` and ``smoothing_radius`` are
                applied later and only for ``mode='mask'`` or
                ``mode='negative-mask'``.

                Processing order is fixed: ``fill_holes`` ->
                ``morph_open_radius`` -> ``min_component_area``.

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
                    useful for single-object prompts but potentially destructive
                    for prompts that intentionally select multiple objects.

                Default: all disabled.

            ``dilate_radius`` : int, optional
                Mask dilation radius in pixels, used only for ``mode='mask'`` and
                ``mode='negative-mask'``. Default: ``0``.

            ``close_radius`` : int, optional
                Morphological closing radius in pixels, used only for full-frame
                mask outputs. Default: ``0``.

            ``smoothing_radius`` : int, optional
                Gaussian smoothing radius in pixels, used only for full-frame mask
                outputs. Default: ``0``.

        ``debug`` : dict
            ``save_debug`` : bool, optional
                Reserved for debug overlays. The current implementation records
                the flag in metadata but does not yet write a debug image.
                Default: ``False``.

    Inputs
    ------
    default : dict, optional
        Upstream image payload used when ``path`` is omitted.

        The payload must contain either:

        - ``image`` : str
        - ``path`` : str

    prompt:default : dict, optional
        Optional upstream prompt bundle. If present, it overrides prompt fields
        from ``spec``. Only ``prompt`` is used by this node.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The output contains:

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
            Output image path. This may be an RGBA cutout or a full-frame mask,
            depending on ``mode``.

        ``prompt`` : str
            Prompt text passed to Grounding DINO after prompt resolution.

        ``params`` : dict
            Resolved runtime parameters, including thresholds, crop mode,
            selection strategy and mask post-processing radii.

        ``model`` : dict
            Model/runtime metadata, including Grounding DINO model id, SAM model
            id, device, grounding_dtype and sam_dtype.

        ``detections`` : dict
            Grounding DINO detection metadata.

            ``candidate_count`` : int
                Number of Grounding DINO candidates returned after thresholding.

            ``selected_count`` : int
                Number of candidates selected according to ``params.select``.

            ``candidates`` : list[dict]
                All Grounding DINO candidates. Each item contains:

                - ``idx`` : int
                  Candidate index in the Grounding DINO result list.
                - ``label`` : str
                  Text label returned by Grounding DINO.
                - ``score`` : float
                  Detection confidence score.
                - ``bbox_xyxy`` : list[int]
                  Candidate bbox in full-image coordinates.
                - ``selected`` : bool
                  Whether this candidate was selected for SAM segmentation.

        ``segments`` : list[dict]
            SAM segmentation requests derived from selected Grounding DINO candidates.

            Each item contains:

            - ``idx`` : int
              Segment index in the selected segment list.
            - ``candidate_idx`` : int | None
              Index of the source Grounding DINO candidate, when available.
            - ``label`` : str
              Label of the selected Grounding DINO candidate.
            - ``score`` : float
              Grounding DINO score of the selected candidate.
            - ``grounding_bbox_xyxy`` : list[int]
              Original Grounding DINO bbox before clipping and margin expansion.
            - ``sam_bbox_xyxy`` : list[int]
              Clipped and margin-expanded bbox used as the SAM prompt.

        ``crop`` : dict
            Crop metadata useful for reinsertion/compositing:

            ``anchor_xy`` : list[int]
                Local anchor inside the output crop image.

            ``position`` : list[int]
                Source-image position where ``anchor_xy`` should be placed when
                reconstructing the original geometry.

            ``bbox_size`` : list[int]
                Width and height of the effective output crop box.

            ``bbox_xyxy`` : list[int]
                Effective output crop box ``[x1, y1, x2, y2]`` in source-image
                coordinates.

        ``debug`` : dict
            Debug-related metadata.

        ``metadata`` : str
            JSON sidecar path.

    Notes
    -----
    - ``AnyCrop`` does not use MediaPipe, YOLO, pose landmarks, face landmarks or
      hand landmarks. It relies entirely on prompt-based Grounding DINO detection
      and SAM segmentation.
    - This node is suitable for arbitrary objects or promptable regions, but it is
      not a replacement for the specialized anatomical logic of ``SubjectCrop``.
      For robust person, head, face, eye or hand crops, ``SubjectCrop`` may still
      be preferable.
    - Mask outputs are always full-frame and aligned to the original input image.
    - In ``mode='default'``, ``crop_mode`` controls the RGBA output layout.
    - In ``mode='mask'`` and ``mode='negative-mask'``, ``crop_mode`` is ignored.
    - ``dilate_radius``, ``close_radius`` and ``smoothing_radius`` affect only
      full-frame mask outputs.
    - ``crop_mode='bbox'`` intentionally keeps the original rectangular crop
      background and writes fully opaque alpha. Use ``crop_mode='trim'`` when a
      transparent cutout is desired.
    - ``crop_mode='bbox[w:h]'`` treats the requested aspect ratio as a target, not
      as a hard guarantee.
    - When several similar objects are detected, use ``select`` to choose one
      candidate or ``select='all'`` to combine all distinct candidates.
    - Heavy models are retrieved through the global model cache where available.
    - If you change code or spec and need fresh outputs, delete the existing
      sidecar JSON to avoid reusing cached results.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        return is_cuda_device(spec.get('model', {}).get('device', 'cuda'))

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        import cv2

        spec = resolve_spec(self.spec)
        cfg = _read_any_crop_cfg(spec, node_id=self.id)

        img_path = resolve_single_image_path(
            node_id=self.id,
            path=self.path,
            input=input,
        )

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(
                f"AnyCrop node '{self.id}': cannot read image: {img_path}")

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        prompt_bundle = PromptBundle(spec=spec, input=input)
        text = _grounding_text_from_prompt(prompt_bundle.prompt)

        grounding_processor, grounding_model = get_grounding_dino(
            model_id=cfg.grounding_model,
            device=cfg.device,
            dtype=cfg.grounding_dtype,
        )

        boxes = _predict_grounding_dino_boxes(
            img_rgb=img_rgb,
            text=text,
            processor=grounding_processor,
            model=grounding_model,
            device=cfg.device,
            dtype=cfg.grounding_dtype,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
        )

        if not boxes:
            raise RuntimeError(
                'AnyCrop: Grounding DINO returned no boxes '
                f'for prompt={text!r}, '
                f'box_threshold={cfg.box_threshold}, '
                f'text_threshold={cfg.text_threshold}.'
            )

        selected_boxes = _select_grounded_boxes(
            boxes,
            image_shape=img_rgb.shape,
            select=cfg.select,
        )

        if cfg.select != 'all' and len(selected_boxes) != 1:
            raise RuntimeError(
                f'AnyCrop node {self.id!r}: select={cfg.select!r} expected one box, '
                f'got {len(selected_boxes)}.'
            )

        sam_processor, sam_model = get_sam(
            model_id=cfg.sam_model,
            device=cfg.device,
            dtype=cfg.sam_dtype,
        )

        clean_mask = np.zeros((h, w), dtype=bool)
        sam_bboxes = []
        single_mask = None

        for selected in selected_boxes:
            bx1, by1, bx2, by2 = _clip_grounded_bbox(
                selected.bbox,
                w=w,
                h=h,
            )

            if cfg.prompt_expansion > 0:
                bx1, by1, bx2, by2 = expand_clip_bbox(
                    bx1, by1, bx2, by2, w, h, cfg.prompt_expansion
                )

            sam_bbox = (bx1, by1, bx2, by2)
            sam_bboxes.append(sam_bbox)

            masks, scores = predict_sam_mask(
                img_rgb=img_rgb,
                bbox=sam_bbox,
                processor=sam_processor,
                model=sam_model,
                device=cfg.device,
            )

            if masks is None or len(masks) == 0:
                raise RuntimeError(
                    f"AnyCrop node '{self.id}': SAM returned no masks.")

            if scores is None or len(scores) == 0:
                best_idx = 0
            else:
                best_idx = int(np.argmax(scores))

            mask = masks[best_idx].astype(bool)

            if cfg.select == 'all':
                clean_mask |= _clean_sam_mask_for_bbox(
                    mask,
                    bbox=sam_bbox,
                    keep_all_components=True,
                )
            else:
                single_mask = mask

        if cfg.select != 'all':
            if single_mask is None:
                raise RuntimeError(
                    f"AnyCrop node '{self.id}': SAM mask was not computed.")
            clean_mask = _clean_sam_mask_for_bbox(
                single_mask,
                bbox=sam_bboxes[0],
                seed_bbox=sam_bboxes[0],
                keep_all_components=False,
            )

        clean_mask = cleanup_shape_mask(clean_mask, **cfg.shape_cleanup)
        clean_mask_u8 = clean_mask.astype(np.uint8) * 255

        crop_x1 = 0
        crop_y1 = 0
        crop_x2 = w
        crop_y2 = h

        if cfg.mode == 'default' and cfg.crop_mode is not None:
            if cfg.crop_mode.mode != 'full_frame':
                crop_x1, crop_y1, crop_x2, crop_y2 = tight_mask_bbox(
                    clean_mask_u8
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

        crop_mask = clean_mask[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop_mask.size == 0:
            raise RuntimeError(
                f'AnyCrop node {self.id!r}: empty crop after bbox.'
            )
        if not np.any(crop_mask):
            raise RuntimeError(
                f'AnyCrop node {self.id!r}: selected SAM mask is empty inside crop bbox.'
            )

        out_dir = Path(output_dir)

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=self.id,
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
                raise RuntimeError(
                    f'AnyCrop node {self.id!r}: crop_mode is missing in default mode.'
                )

            if cfg.crop_mode.mode == 'bbox':
                alpha = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                crop_rgba = np.dstack([crop_rgb, alpha])
                Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

            else:
                alpha = clean_mask_u8[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_rgba = np.dstack([crop_rgb, alpha])

                if cfg.crop_mode.mode == 'trim':
                    Image.fromarray(crop_rgba, mode='RGBA').save(out_path)

                elif cfg.crop_mode.mode == 'full_frame':
                    full_alpha = clean_mask.astype(np.uint8) * 255
                    full_rgba = np.dstack([img_rgb, full_alpha])

                    out_x1 = 0
                    out_y1 = 0
                    out_x2 = w
                    out_y2 = h

                    Image.fromarray(full_rgba, mode='RGBA').save(out_path)

                else:
                    raise ValueError(
                        f'AnyCrop node {self.id!r}: invalid crop_mode={cfg.crop_mode.mode!r}'
                    )

        else:
            out_mask_u8 = prepare_output_mask(
                clean_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                out_mask_u8 = 255 - out_mask_u8

            Image.fromarray(out_mask_u8, mode='L').save(out_path)

            out_x1 = 0
            out_y1 = 0
            out_x2 = w
            out_y2 = h

        bbox_w = int(out_x2 - out_x1)
        bbox_h = int(out_y2 - out_y1)
        anchor_x = int(round((out_x1 + out_x2) / 2.0))
        anchor_y = int(round((out_y1 + out_y2) / 2.0))

        selected_ids = {id(box) for box in selected_boxes}
        box_to_candidate_idx = {
            id(box): i
            for i, box in enumerate(boxes)
        }

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'prompt': text,
            'model': {
                'grounding_model': cfg.grounding_model,
                'sam_model': cfg.sam_model,
                'device': cfg.device,
                'grounding_dtype': str(cfg.grounding_dtype).replace('torch.', ''),
                'sam_dtype': str(cfg.sam_dtype).replace('torch.', ''),
            },
            'params': {
                'crop_mode': cfg.crop_mode.raw if cfg.crop_mode is not None else None,
                'box_threshold': cfg.box_threshold,
                'text_threshold': cfg.text_threshold,
                'box_margin': cfg.box_margin,
                'prompt_expansion': cfg.prompt_expansion,
                'select': cfg.select,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'smoothing_radius': cfg.smoothing_radius,
                'postprocess': cfg.shape_cleanup,
            },
            'detections': {
                'candidate_count': len(boxes),
                'selected_count': len(selected_boxes),
                'candidates': [
                    {
                        'idx': i,
                        'label': box.label,
                        'score': float(box.score),
                        'bbox_xyxy': [int(x) for x in box.bbox],
                        'selected': id(box) in selected_ids,
                    }
                    for i, box in enumerate(boxes)
                ],
            },
            'segments': [
                {
                    'idx': i,
                    'candidate_idx': box_to_candidate_idx.get(id(selected)),
                    'label': selected.label,
                    'score': float(selected.score),
                    'grounding_bbox_xyxy': [int(x) for x in selected.bbox],
                    'sam_bbox_xyxy': [int(x) for x in sam_bbox],
                }
                for i, (selected, sam_bbox) in enumerate(zip(selected_boxes, sam_bboxes))
            ],
            'crop': {
                'anchor_xy': [
                    int(anchor_x - out_x1),
                    int(anchor_y - out_y1),
                ],
                'position': [anchor_x, anchor_y],
                'bbox_size': [bbox_w, bbox_h],
                'bbox_xyxy': [int(out_x1), int(out_y1), int(out_x2), int(out_y2)],
            },
            'debug': {
                'save_debug': cfg.save_debug,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
