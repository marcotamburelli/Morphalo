from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import get_human_segmenter
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.crop_debug import write_mask_debug_overlay
from morphalo.nodes.preprocess.utils import (CropModeSpec,
                                             expand_bbox_toward_ratio,
                                             expand_clip_bbox, parse_crop_mode,
                                             postprocess_mask,
                                             tight_alpha_bbox)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

TargetSpec = str | list[str] | tuple[str, ...]

FASHN_LABELS: dict[str, int] = {
    'arms': 12,
    'bag': 8,
    'background': 0,
    'belt': 7,
    'dress': 4,
    'face': 1,
    'feet': 15,
    'glasses': 11,
    'hair': 2,
    'hat': 9,
    'hands': 13,
    'jewelry': 17,
    'legs': 14,
    'pants': 6,
    'scarf': 10,
    'skirt': 5,
    'top': 3,
    'torso': 16,
}

COMPOSITE_TARGETS: dict[str, tuple[str, ...]] = {
    'body': ('face', 'hair', 'arms', 'hands', 'legs', 'feet', 'torso'),
    'clothes': ('top', 'dress', 'skirt', 'pants', 'belt', 'scarf'),
    'head': ('face', 'hair'),
    'person': tuple(label for label in FASHN_LABELS if label != 'background'),
    'skin': ('face', 'arms', 'hands', 'legs', 'feet', 'torso'),
}

VALID_TARGETS = tuple(FASHN_LABELS.keys()) + tuple(COMPOSITE_TARGETS.keys())


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    segment_model: str
    mode: str
    crop_mode: Optional[CropModeSpec]
    target: TargetSpec
    box_margin: float
    dilate_radius: int
    close_radius: int
    smoothing_radius: int
    save_debug: bool


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'float16')))
    segment_model = str(
        model.get('segment_model', 'fashn-ai/fashn-human-parser')
    )

    mode = str(params.get('mode', 'default'))
    if mode not in ('default', 'mask', 'negative-mask'):
        raise ValueError(f"'{node_id}': invalid mode={mode!r}")

    crop_mode = (
        parse_crop_mode(params.get('crop_mode', 'trim'), node_id=node_id)
        if mode == 'default'
        else None
    )

    target = params.get('target', 'person')
    _resolve_target_labels(target, node_id=node_id)

    box_margin = float(params.get('box_margin', 0.08))
    if box_margin < 0.0:
        raise ValueError(f"'{node_id}': box_margin must be >= 0")

    dilate_radius = int(params.get('dilate_radius', 0))
    close_radius = int(params.get('close_radius', 0))
    smoothing_radius = int(params.get('smoothing_radius', 0))
    for name, value in (
        ('dilate_radius', dilate_radius),
        ('close_radius', close_radius),
        ('smoothing_radius', smoothing_radius),
    ):
        if value < 0:
            raise ValueError(
                f"'{node_id}': {name} must be >= 0, got {value}"
            )

    return Config(
        device=device,
        dtype=dtype,
        segment_model=segment_model,
        mode=mode,
        crop_mode=crop_mode,
        target=target,
        box_margin=box_margin,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        smoothing_radius=smoothing_radius,
        save_debug=bool(debug.get('save_debug', False)),
    )


def _resolve_target_labels(
    target: TargetSpec,
    *,
    node_id: str,
) -> list[str]:
    """
    Resolve ``params.target`` into stable, de-duplicated FASHN labels.

    ``target`` may be a single label/alias string, a list of strings, or a
    tuple of strings. Atomic FASHN labels pass through unchanged, composite
    aliases expand to their member labels, and duplicates are removed while
    preserving first-seen order.

    Parameters
    ----------
    target : str | list[str] | tuple[str, ...]
        Raw target specification from node params.
    node_id : str
        Node id used in validation errors.

    Returns
    -------
    list[str]
        FASHN label names used to build the semantic mask.
    """
    expected = ', '.join(repr(t) for t in VALID_TARGETS)

    if isinstance(target, str):
        items = [target]
    elif isinstance(target, (list, tuple)):
        if not target:
            raise ValueError(f"'{node_id}': target sequence cannot be empty")
        items = list(target)
    else:
        raise ValueError(
            f"'{node_id}': invalid target={target!r} "
            f"(expected a string, list of strings, or tuple of strings; "
            f"valid values: {expected})"
        )

    labels: list[str] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, str):
            raise ValueError(
                f"'{node_id}': invalid target item={item!r} "
                f"(expected a string; valid values: {expected})"
            )

        if item in FASHN_LABELS:
            expanded = (item,)
        elif item in COMPOSITE_TARGETS:
            expanded = COMPOSITE_TARGETS[item]
        else:
            raise ValueError(
                f"'{node_id}': invalid target={item!r} "
                f"(expected one of: {expected})"
            )

        for label in expanded:
            if label not in seen:
                labels.append(label)
                seen.add(label)

    return labels


def _target_label_ids(labels: list[str]) -> list[int]:
    """
    Convert resolved FASHN label names to parser class ids.

    Parameters
    ----------
    labels : list[str]
        Label names returned by ``_resolve_target_labels``.

    Returns
    -------
    list[int]
        FASHN parser class ids in the same order.
    """
    return [FASHN_LABELS[label] for label in labels]


def _predict_segments(
    image: Image.Image,
    *,
    model_id: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, torch.dtype]:
    """
    Run the FASHN-compatible SegFormer parser and return class predictions.

    The model cache may adjust the effective runtime dtype for compatibility,
    especially on non-CUDA devices. The returned dtype is therefore the actual
    dtype observed from model parameters and is written to metadata separately
    from the requested dtype.

    Parameters
    ----------
    image : Image.Image
        RGB PIL image to segment.
    model_id : str
        Hugging Face model id using the FASHN human-parser label taxonomy.
    device : str
        Torch device for inference.
    dtype : torch.dtype
        Requested model dtype.

    Returns
    -------
    tuple[np.ndarray, torch.dtype]
        ``(segments, runtime_dtype)`` where ``segments`` is an ``H x W`` array
        of FASHN class ids.
    """
    processor, model = get_human_segmenter(
        model_id=model_id,
        device=device,
        dtype=dtype,
    )
    model_dtype = next(model.parameters()).dtype

    inputs = processor(images=image, return_tensors='pt')
    inputs = {
        key: (
            value.to(device=device, dtype=model_dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value.to(device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False,
        )
        predictions = upsampled.argmax(dim=1).squeeze(0)

    return predictions.detach().cpu().numpy().astype(np.int64), model_dtype


@dataclass
class HumanSegmentCrop(CudaPostRunMixin, NodeRef):
    """
    Human semantic segment crop and mask generator.

    ``HumanSegmentCrop`` uses a SegFormer human parser intended for
    single-person images to select semantic body, clothing, and accessory
    regions. The selected semantic region can be written as:

    - an RGBA crop/cutout when ``mode='default'``;
    - a full-frame positive inpaint mask when ``mode='mask'``;
    - a full-frame negative inpaint mask when ``mode='negative-mask'``.

    The output metadata is compatible with other preprocessing crop nodes. In
    particular, the ``crop`` block exposes ``anchor_xy``, ``position``,
    ``bbox_size`` and ``bbox_xyxy`` so that crops can be reinserted through
    nodes such as ``ImageStack``.

    Target semantics
    ----------------
    Atomic targets follow the FASHN human parser label set:

    - ``arms``
    - ``bag``
    - ``background``
    - ``belt``
    - ``dress``
    - ``face``
    - ``feet``
    - ``glasses``
    - ``hair``
    - ``hat``
    - ``hands``
    - ``jewelry``
    - ``legs``
    - ``pants``
    - ``scarf``
    - ``skirt``
    - ``top``
    - ``torso``

    Composite targets are:

    - ``body``: face, hair, arms, hands, legs, feet, torso;
    - ``clothes``: top, dress, skirt, pants, belt, scarf;
    - ``head``: face, hair;
    - ``person``: all non-background labels;
    - ``skin``: same anatomical body labels as ``body``, excluding hair.

    ``body`` and ``skin`` are parser-region composites. They exclude clothing
    labels such as ``top``, ``dress``, ``pants`` and ``skirt``. The parser does
    not expose a dedicated shoe label, so visible footwear may be included in
    ``feet``.

    ``params.target`` may be a single string, a list of strings, or a tuple of
    strings. Sequence entries may combine atomic labels and composite aliases.
    Aliases are expanded, duplicate labels are removed, and the first-seen label
    order is preserved.

    Pipeline
    --------
    The node executes the following steps:

    1. Resolve the input image from ``path`` or from the upstream default input.
    2. Run the configured SegFormer human parser through Hugging Face
       Transformers.
    3. Upsample logits to the original image size and derive a per-pixel class
       map.
    4. Build a boolean mask from ``params.target``.
    5. Derive the target bounding box from the selected mask.
    6. Produce either an RGBA crop or a full-frame mask, depending on ``mode``.
    7. Write the output image and JSON sidecar metadata.

    Crop geometry
    -------------
    For ``mode='default'``, ``crop_mode`` controls the spatial layout of the
    produced RGBA image.

    Supported values are:

    - ``'trim'``:
        The output is an RGBA cutout cropped to the mask-derived target bbox,
        optionally expanded by ``box_margin``. Alpha is derived from the
        selected semantic mask.

    - ``'bbox'``:
        The output is the rectangular crop inside the mask-derived target bbox,
        optionally expanded by ``box_margin``. The original RGB background is
        preserved inside the crop and alpha is fully opaque everywhere.

    - ``'bbox[w:h]'``:
        Same as ``'bbox'``, but the target bbox is expanded toward the requested
        aspect ratio ``w:h`` while keeping the selected region inside the crop
        and staying within source-image bounds.

        The requested ratio is a target, not a hard guarantee. Near image
        borders, the final crop may deviate from the requested ratio.

    - ``'full_frame'``:
        The output is a full-size RGBA image aligned to the original input
        coordinates. The RGB channels contain the original image and the alpha
        channel contains the selected semantic mask.

    For ``mode='mask'`` and ``mode='negative-mask'``, ``crop_mode`` is ignored
    because mask outputs are always full-frame.

    Mask post-processing
    --------------------
    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the positive selected
    mask is post-processed before it is written to disk.

    Post-processing is applied in this order:

    1. morphological closing via ``close_radius``;
    2. dilation via ``dilate_radius``;
    3. Gaussian smoothing via ``smoothing_radius``;
    4. optional inversion for ``mode='negative-mask'``.

    This means that in ``mode='negative-mask'`` the selected semantic region is
    protected first, then the result is inverted. Dilation therefore expands the
    protected area before inversion, creating a safety band around the target.

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
                Runtime device used for SegFormer inference. Default:
                ``'cuda'``.

            ``dtype`` : str, optional
                Torch dtype requested when loading the model. Default:
                ``'float16'``.

                The model cache loads non-CUDA devices in float32 for runtime
                compatibility. CUDA devices use the requested dtype.

            ``segment_model`` : str, optional
                Hugging Face model identifier. Default:
                ``'fashn-ai/fashn-human-parser'``.

                The model must use the FASHN human-parser label taxonomy.
                Alternative checkpoints are supported only when they preserve
                the same label ids and semantic meanings defined by
                ``FASHN_LABELS``.

        ``params`` : dict
            ``target`` : str | list[str] | tuple[str, ...], optional
                Atomic or composite semantic target, or a sequence combining
                atomic labels and composite aliases. Default: ``'person'``.

            ``mode`` : {'default', 'mask', 'negative-mask'}, optional
                Output type.

                - ``'default'``: write an RGBA crop/cutout.
                - ``'mask'``: write a full-frame positive mask where the
                  selected region is white.
                - ``'negative-mask'``: write a full-frame inverted mask where
                  the background is white and the selected region is protected.

                Default: ``'default'``.

            ``crop_mode`` : {'trim', 'bbox', 'bbox[w:h]', 'full_frame'}, optional
                Applies only when ``mode='default'``.

                Controls whether the output is a semantic cutout, a rectangular
                bbox crop, an aspect-ratio bbox crop, or a full-frame RGBA
                image.

                Default: ``'trim'``.

            ``box_margin`` : float, optional
                Symmetric expansion ratio applied to the mask-derived selection
                bbox, expressed as a fraction of bbox size. Applies only when
                ``mode='default'`` and ``crop_mode`` is ``'trim'``, ``'bbox'``
                or ``'bbox[w:h]'``. Ignored for ``crop_mode='full_frame'`` and
                mask outputs. For ``crop_mode='bbox[w:h]'``, the margin is
                applied before aspect-ratio expansion. Typical range:
                ``0.03``-``0.12``. Default: ``0.08``.

            ``dilate_radius`` : int, optional
                Mask dilation radius in pixels, used only for ``mode='mask'``
                and ``mode='negative-mask'``. Default: ``0``.

            ``close_radius`` : int, optional
                Morphological closing radius in pixels, used only for
                full-frame mask outputs. Default: ``0``.

            ``smoothing_radius`` : int, optional
                Gaussian smoothing radius in pixels, used only for full-frame
                mask outputs. Default: ``0``.

        ``debug`` : dict
            ``save_debug`` : bool, optional
                Save a target-mask debug overlay image. Default: ``False``.

    Inputs
    ------
    default : dict, optional
        Upstream image payload used when ``path`` is omitted.

        The payload must contain either:

        - ``image`` : str
        - ``path`` : str

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

        ``input_size`` : list[int]
            Input image size as ``[width, height]``.

        ``output_size`` : list[int]
            Output image size as ``[width, height]``.

        ``model`` : dict
            Model/runtime metadata, including the human segmenter model id,
            runtime device, requested dtype, and runtime dtype.

        ``human_segment_crop`` : dict
            Resolved semantic crop configuration, including ``target``,
            selected label names, and selected label ids.

        ``params`` : dict
            Public runtime parameters, including target, mode, crop mode,
            ``box_margin`` and mask post-processing radii.

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

        ``debug_bbox`` : str, optional
            Debug image path, present only when ``debug.save_debug`` is true.

        ``metadata`` : str
            JSON sidecar path.

    Notes
    -----
    - Mask outputs are always full-frame and aligned to the original image size.
    - ``box_margin`` is applied only to mask-derived default-output crop boxes.
    - ``dilate_radius``, ``close_radius`` and ``smoothing_radius`` affect only
      full-frame mask outputs. In ``mode='default'``, RGBA alpha is derived
      directly from the selected semantic mask.
    - The FASHN human parser returns aggregate classes such as ``hands`` and
      ``feet``; side-specific targets such as ``left-hand`` and ``right-foot``
      are not supported by this node.
    - There is no dedicated ``shoes`` class in the FASHN taxonomy. Shoes or
      socks may be classified as ``feet`` and therefore included in ``body`` and
      ``skin``.
    - The underlying parser is intended for single-person human parsing. Multi-person
      images may produce a merged semantic mask rather than a separable subject.
    - If the selected target is not present in the segmentation map, the node
      raises an error instead of writing an empty crop or mask.
    - The model and processor are retrieved through the global model cache.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        return is_cuda_device(spec.get('model', {}).get('device', 'cuda'))

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)
        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)
        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )

        with Image.open(img_path) as image:
            image_rgb = image.convert('RGB')

        w, h = image_rgb.size
        rgb = np.asarray(image_rgb, dtype=np.uint8)

        segments, runtime_dtype = _predict_segments(
            image_rgb,
            model_id=cfg.segment_model,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        resolved_labels = _resolve_target_labels(cfg.target, node_id=node_id)
        label_ids = _target_label_ids(resolved_labels)
        selected_mask = np.isin(
            segments, np.asarray(label_ids, dtype=np.int64))

        if not np.any(selected_mask):
            raise RuntimeError(
                f"HumanSegmentCrop node '{node_id}': selected mask is empty "
                f"for target={cfg.target!r}."
            )

        alpha_full = selected_mask.astype(np.uint8) * 255
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = tight_alpha_bbox(alpha_full)

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=node_id,
            ext='png',
        )

        out_x1 = 0
        out_y1 = 0
        out_x2 = w
        out_y2 = h

        if cfg.mode == 'default':
            if cfg.crop_mode is None:
                raise RuntimeError(
                    f"{self.id}: crop_mode must be defined when mode='default'"
                )

            if cfg.crop_mode.mode == 'full_frame':
                crop_rgba = np.dstack([rgb, alpha_full])
                output_image = Image.fromarray(crop_rgba, mode='RGBA')

            else:
                x1, y1, x2, y2 = bbox_x1, bbox_y1, bbox_x2, bbox_y2
                if cfg.box_margin > 0:
                    x1, y1, x2, y2 = expand_clip_bbox(
                        x1,
                        y1,
                        x2,
                        y2,
                        w,
                        h,
                        cfg.box_margin,
                    )

                if cfg.crop_mode.mode == 'bbox':
                    if cfg.crop_mode.ratio is not None:
                        x1, y1, x2, y2 = expand_bbox_toward_ratio(
                            x1,
                            y1,
                            x2,
                            y2,
                            full_w=w,
                            full_h=h,
                            ratio=cfg.crop_mode.ratio,
                        )

                    out_x1 = int(x1)
                    out_y1 = int(y1)
                    out_x2 = int(x2)
                    out_y2 = int(y2)
                    crop_rgb = rgb[out_y1:out_y2, out_x1:out_x2, :]
                    alpha = np.full(
                        (out_y2 - out_y1, out_x2 - out_x1),
                        255,
                        dtype=np.uint8,
                    )
                    output_image = Image.fromarray(
                        np.dstack([crop_rgb, alpha]),
                        mode='RGBA',
                    )

                elif cfg.crop_mode.mode == 'trim':
                    out_x1 = int(x1)
                    out_y1 = int(y1)
                    out_x2 = int(x2)
                    out_y2 = int(y2)
                    crop_rgb = rgb[out_y1:out_y2, out_x1:out_x2, :]
                    crop_alpha = alpha_full[out_y1:out_y2, out_x1:out_x2]
                    output_image = Image.fromarray(
                        np.dstack([crop_rgb, crop_alpha]),
                        mode='RGBA',
                    )

                else:
                    raise ValueError(
                        f'{self.id}: invalid crop_mode={cfg.crop_mode.mode!r}'
                    )

        else:
            out_mask_u8 = postprocess_mask(
                selected_mask,
                dilate_radius=cfg.dilate_radius,
                close_radius=cfg.close_radius,
                smoothing_radius=cfg.smoothing_radius,
            )

            if cfg.mode == 'negative-mask':
                out_mask_u8 = 255 - out_mask_u8

            output_image = Image.fromarray(out_mask_u8, mode='L')

        output_image.save(out_path)

        dbg_path = None
        if cfg.save_debug:
            dbg_path = write_mask_debug_overlay(
                img_rgb=rgb,
                mask=selected_mask,
                target_bbox=(bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                out_path=out_path,
                target=str(cfg.target),
            )

        bbox_w = int(out_x2 - out_x1)
        bbox_h = int(out_y2 - out_y1)
        anchor_x = int((out_x1 + out_x2) // 2)
        anchor_y = int((out_y1 + out_y2) // 2)

        params = {
            'target': cfg.target,
            'mode': cfg.mode,
            'crop_mode': None if cfg.crop_mode is None else cfg.crop_mode.raw,
            'box_margin': cfg.box_margin,
            'dilate_radius': cfg.dilate_radius,
            'close_radius': cfg.close_radius,
            'smoothing_radius': cfg.smoothing_radius,
        }

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'mode': cfg.mode,
            'image': str(out_path),
            'input_size': [int(w), int(h)],
            'output_size': [int(output_image.width), int(output_image.height)],
            'model': {
                'segment_model': cfg.segment_model,
                'device': cfg.device,
                'requested_dtype': str(cfg.dtype).replace('torch.', ''),
                'runtime_dtype': str(runtime_dtype).replace('torch.', ''),
            },
            'human_segment_crop': {
                **params,
                'labels': resolved_labels,
                'label_ids': [int(label_id) for label_id in label_ids],
            },
            'params': params,
            'crop': {
                'anchor_xy': [
                    int(anchor_x - out_x1),
                    int(anchor_y - out_y1),
                ],
                'position': [anchor_x, anchor_y],
                'bbox_size': [bbox_w, bbox_h],
                'bbox_xyxy': [
                    int(out_x1),
                    int(out_y1),
                    int(out_x2),
                    int(out_y2),
                ],
            },
        }
        if dbg_path is not None:
            out['debug_bbox'] = str(dbg_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
