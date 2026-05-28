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
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.segmentation import predict_sam_mask
from morphalo.nodes.preprocess.utils import (CropModeSpec,
                                             expand_bbox_toward_ratio,
                                             parse_crop_mode, postprocess_mask,
                                             tight_alpha_bbox)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from morphalo.nodes.vision.face_region import expand_clip_bbox
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
    box_margin : float
        Symmetric bbox expansion ratio applied before SAM.
    select : {'best', 'largest', 'center'}
        Strategy used to select one box when Grounding DINO returns multiple boxes.
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
    box_margin: float
    select: str
    save_debug: bool
    dilate_radius: int
    close_radius: int
    smoothing_radius: int


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
    box_margin = float(params.get('box_margin', 0.08))

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

    if box_margin < 0:
        raise ValueError(
            f"'{node_id}': invalid box_margin={box_margin!r} "
            '(expected >= 0)'
        )

    select = str(params.get('select', 'best'))
    if select not in ('best', 'largest', 'center'):
        raise ValueError(
            f"'{node_id}': invalid select={select!r} "
            "(expected 'best', 'largest', or 'center')"
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
        select=select,
        save_debug=bool(debug.get('save_debug', False)),
        dilate_radius=int(params.get('dilate_radius', 0)),
        close_radius=int(params.get('close_radius', 0)),
        smoothing_radius=int(params.get('smoothing_radius', 0)),
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


def _select_grounded_box(
    boxes: list[GroundedBox],
    *,
    image_shape: tuple[int, int, int],
    select: str,
) -> GroundedBox:
    if not boxes:
        raise RuntimeError('AnyCrop: Grounding DINO returned no boxes.')

    if select == 'best':
        return max(boxes, key=lambda b: b.score)

    if select == 'largest':
        return max(boxes, key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]))

    if select == 'center':
        h, w = image_shape[:2]
        cx0, cy0 = w * 0.5, h * 0.5

        def dist2(b: GroundedBox) -> float:
            x1, y1, x2, y2 = b.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            return (cx - cx0) ** 2 + (cy - cy0) ** 2

        return min(boxes, key=dist2)

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


@dataclass
class AnyCrop(PromptMixin, NodeRef):
    """
    Open-vocabulary crop and inpaint-mask generator using Grounding DINO and
    SAM/SAM-HQ segmentation.

    ``AnyCrop`` is the prompt-based counterpart of ``SubjectCrop``.

    While ``SubjectCrop`` extracts a fixed semantic or anatomical target such as
    ``'person'``, ``'head'``, ``'face'``, ``'eyes'`` or ``'hands'``,
    ``AnyCrop`` localizes an arbitrary region from a text prompt. The prompt is
    resolved through ``PromptMixin`` and passed to Grounding DINO, which predicts
    one or more open-vocabulary bounding boxes. The selected box is then used as
    a SAM prompt to obtain a pixel-level mask.

    The node produces either:

    - an RGBA cutout when ``mode='default'``;
    - a full-frame positive inpaint mask when ``mode='mask'``;
    - a full-frame negative inpaint mask when ``mode='negative-mask'``.

    The output metadata is intentionally compatible with geometry-driven
    compositing workflows. In particular, the ``crop`` block exposes
    ``anchor_xy``, ``bbox_size`` and ``bbox_xyxy`` so that refined crops can be
    reinserted through nodes such as ``ImageStack``.

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
    ``'a cat. a dog.'``, but this node currently selects a single detected box
    according to ``params.select``.

    Pipeline
    --------
    The node executes the following steps:

    1. Resolve the input image from ``path`` or from the upstream default input.
    2. Resolve the primary prompt through ``PromptMixin`` / ``PromptBundle``.
    3. Run Grounding DINO on the image and prompt.
    4. Select one detection box using ``params.select``.
    5. Clip the selected box to the image bounds.
    6. Optionally expand the selected box using ``params.box_margin``.
    7. Run SAM using the expanded box as the segmentation prompt.
    8. Select the SAM candidate with the highest predicted IoU score.
    9. Restrict the mask to the crop region and keep the connected component
       associated with the selected detection.
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

    After SAM segmentation, the node keeps a single connected component from the
    selected mask. The preferred component is the one containing the center of the
    SAM prompt box, expressed in crop-local coordinates. If that point falls on
    the background, the largest foreground component is used instead.

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

            ``box_margin`` : float, optional
                Symmetric expansion ratio applied to the selected detection box
                before running SAM. The value is expressed as a fraction of the
                box width and height.

                Default: ``0.08``.

            ``select`` : {'best', 'largest', 'center'}, optional
                Strategy used when Grounding DINO returns multiple boxes.
                Default: ``'best'``.

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

        ``params`` : dict
            Resolved runtime parameters, including prompt, thresholds, crop mode,
            selection strategy and mask post-processing radii.

        ``model`` : dict
            Model/runtime metadata, including Grounding DINO model id, SAM model
            id, device, grounding_dtype and sam_dtype.

        ``detection`` : dict
            Detection metadata, including:

            ``label`` : str
                Label returned by Grounding DINO for the selected detection.

            ``score`` : float
                Grounding DINO score for the selected detection.

            ``grounding_bbox_xyxy`` : list[int]
                Raw selected Grounding DINO bbox before clipping and margin.

            ``sam_bbox_xyxy`` : list[int]
                Clipped and margin-expanded bbox used as the SAM prompt.

            ``candidates`` : list[dict]
                All Grounding DINO candidates returned after thresholding.

        ``crop`` : dict
            Crop metadata useful for reinsertion/compositing:

            ``anchor_xy`` : list[int]
                Center of the effective output crop box in absolute source-image
                coordinates.

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
    - When several similar objects are detected, use ``select`` to choose the
      preferred candidate. More specific prompts usually improve localization.
    - Heavy models are retrieved through the global model cache where available.
    - If you change code or spec and need fresh outputs, delete the existing
      sidecar JSON to avoid reusing cached results.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

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

        selected = _select_grounded_box(
            boxes,
            image_shape=img_rgb.shape,
            select=cfg.select,
        )

        bx1, by1, bx2, by2 = _clip_grounded_bbox(
            selected.bbox,
            w=w,
            h=h,
        )

        if cfg.box_margin > 0:
            bx1, by1, bx2, by2 = expand_clip_bbox(
                bx1, by1, bx2, by2, w, h, cfg.box_margin
            )

        sam_processor, sam_model = get_sam(
            model_id=cfg.sam_model,
            device=cfg.device,
            dtype=cfg.sam_dtype,
        )

        masks, scores = predict_sam_mask(
            img_rgb=img_rgb,
            bbox=(bx1, by1, bx2, by2),
            processor=sam_processor,
            model=sam_model,
            device=cfg.device,
        )

        if masks is None or len(masks) == 0:
            raise RuntimeError(
                f"AnyCrop node '{self.id}': SAM returned no masks.")

        mask = masks[int(np.argmax(scores))].astype(bool)

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
                f'AnyCrop node {self.id!r}: empty crop after bbox.'
            )
        if not np.any(crop_mask):
            raise RuntimeError(
                f'AnyCrop node {self.id!r}: selected SAM mask is empty inside crop bbox.'
            )
        cm = crop_mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cm,
            connectivity=8,
        )

        if num > 1:
            local_cx = int(round(0.5 * (bx1 + bx2))) - crop_x1
            local_cy = int(round(0.5 * (by1 + by2))) - crop_y1

            local_cx = max(0, min(cm.shape[1] - 1, local_cx))
            local_cy = max(0, min(cm.shape[0] - 1, local_cy))

            target = labels[local_cy, local_cx]

            if target == 0:
                areas = stats[1:, cv2.CC_STAT_AREA]
                target = 1 + int(np.argmax(areas))

            crop_mask = labels == target

        clean_mask = np.zeros_like(mask, dtype=bool)
        clean_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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
            out_mask_u8 = postprocess_mask(
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

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'params': {
                'prompt': text,
                'mode': cfg.mode,
                'crop_mode': cfg.crop_mode.raw if cfg.crop_mode is not None else None,
                'box_threshold': cfg.box_threshold,
                'text_threshold': cfg.text_threshold,
                'box_margin': cfg.box_margin,
                'select': cfg.select,
                'dilate_radius': cfg.dilate_radius,
                'close_radius': cfg.close_radius,
                'smoothing_radius': cfg.smoothing_radius,
            },
            'model': {
                'grounding_model': cfg.grounding_model,
                'sam_model': cfg.sam_model,
                'device': cfg.device,
                'grounding_dtype': str(cfg.grounding_dtype).replace('torch.', ''),
                'sam_dtype': str(cfg.sam_dtype).replace('torch.', ''),
            },
            'detection': {
                'label': selected.label,
                'score': float(selected.score),
                'grounding_bbox_xyxy': [int(x) for x in selected.bbox],
                'sam_bbox_xyxy': [int(bx1), int(by1), int(bx2), int(by2)],
                'candidates': [
                    {
                        'label': b.label,
                        'score': float(b.score),
                        'bbox_xyxy': [int(x) for x in b.bbox],
                    }
                    for b in boxes
                ],
            },
            'crop': {
                'anchor_xy': [anchor_x, anchor_y],
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
