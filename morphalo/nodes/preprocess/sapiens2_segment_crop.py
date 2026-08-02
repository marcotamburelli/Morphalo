from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.crop_debug import write_mask_debug_overlay
from morphalo.nodes.preprocess.utils import (CropModeSpec, SizeExpr,
                                             expand_bbox_toward_ratio,
                                             parse_crop_mode,
                                             read_shape_cleanup_config,
                                             validate_size_expr)
from morphalo.nodes.preprocess.utils.geometry import (
    expand_clip_bbox_by_size_expr, tight_mask_bbox)
from morphalo.nodes.preprocess.utils.mask_ops import (
    cleanup_shape_mask_by_parts, prepare_output_mask)
from morphalo.nodes.preprocess.utils.sapiens2_seg import (
    ParsedSegmentTarget, parse_target_specs, predict_segments,
    resolve_parsed_target, segment_part_masks)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path

TargetSpec = str | list[str] | tuple[str, ...]


@dataclass
class Config:
    device: str
    dtype: torch.dtype
    segment_model: str
    mode: str
    crop_mode: Optional[CropModeSpec]
    target: TargetSpec
    parsed_target: ParsedSegmentTarget
    box_margin: SizeExpr
    dilate_radius: int
    close_radius: int
    smoothing_radius: int
    save_debug: bool
    shape_cleanup: dict[str, Any]


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'bfloat16')))
    segment_model = str(
        model.get('segment_model', 'facebook/sapiens2-seg-0.4b')
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
    parsed_target = parse_target_specs(target, node_id=node_id)

    box_margin = params.get('box_margin', '8%')
    validate_size_expr(box_margin)

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
        parsed_target=parsed_target,
        box_margin=box_margin,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
        smoothing_radius=smoothing_radius,
        save_debug=bool(debug.get('save_debug', False)),
        shape_cleanup=read_shape_cleanup_config(
            params.get('postprocess', None),
            node_id=node_id,
        ),
    )


@dataclass
class Sapiens2SegmentCrop(CudaPostRunMixin, NodeRef):
    """
    Sapiens2 body-part segmentation crop and mask generator.

    ``Sapiens2SegmentCrop`` uses a Sapiens2 body-part segmentation checkpoint
    from Hugging Face Transformers to select semantic human regions. The
    selected region can be written as:

    - an RGBA crop/cutout when ``mode='default'``;
    - a full-frame positive inpaint mask when ``mode='mask'``;
    - a full-frame negative inpaint mask when ``mode='negative-mask'``.

    The output metadata follows the same crop contract as other preprocessing
    crop nodes. In particular, the ``crop`` block exposes ``anchor_xy``,
    ``position``, ``bbox_size`` and ``bbox_xyxy`` so crops can be reinserted by
    nodes such as ``ImageStack``.

    Target semantics
    ----------------
    The underlying Sapiens2 model emits anatomical body-part classes. Public
    ``left-*`` and ``right-*`` targets exposed by this node follow the project
    crop convention instead: they are image/viewer-relative. In other words,
    ``left-hand`` means the visible hand on the left side of the image, not
    necessarily the subject's anatomical left hand.

    Anatomical selection remains available through explicit
    ``anatomical-left-*`` and ``anatomical-right-*`` targets. For example:

    - ``left-hand``: choose whichever anatomical hand appears leftmost in the
      image;
    - ``right-hand``: choose whichever anatomical hand appears rightmost in the
      image;
    - ``anatomical-left-hand``: select the model's anatomical left-hand class;
    - ``anatomical-right-hand``: select the model's anatomical right-hand class.

    Non-sided atomic targets map directly to Sapiens2 classes:

    - ``background``
    - ``apparel``
    - ``eyeglass``
    - ``face-neck``
    - ``hair``
    - ``lower-clothing``
    - ``torso``
    - ``upper-clothing``
    - ``lower-lip``
    - ``upper-lip``
    - ``lower-teeth``
    - ``upper-teeth``
    - ``tongue``

    Image-relative side targets include:

    - ``left-hand`` / ``right-hand``
    - ``left-foot`` / ``right-foot``
    - ``left-arm`` / ``right-arm``
    - ``left-leg`` / ``right-leg``
    - ``left-lower-arm`` / ``right-lower-arm``
    - ``left-upper-arm`` / ``right-upper-arm``
    - ``left-lower-leg`` / ``right-lower-leg``
    - ``left-upper-leg`` / ``right-upper-leg``
    - ``left-shoe`` / ``right-shoe``
    - ``left-sock`` / ``right-sock``

    Anatomical side targets add the ``anatomical-`` prefix to the same sided
    names, such as ``anatomical-left-hand``, ``anatomical-right-foot``,
    ``anatomical-left-arm`` or ``anatomical-right-lower-leg``.

    Composite targets are:

    - ``arms``: left/right lower and upper arms;
    - ``body``: anatomical body parts, hair, and mouth parts, excluding
      clothing, socks and shoes;
    - ``clothes``: apparel, lower clothing and upper clothing;
    - ``feet``: left/right feet plus socks and shoes;
    - ``footwear``: left/right socks and shoes;
    - ``hands``: left and right hands;
    - ``head``: face/neck, hair, lips, teeth and tongue;
    - ``legs``: left/right lower and upper legs;
    - ``lower-body``: feet and legs, excluding socks and shoes;
    - ``mouth``: lips, teeth and tongue;
    - ``person``: all non-background labels;
    - ``skin``: anatomical body parts and mouth parts, excluding hair,
      clothing, socks and shoes;
    - ``upper-body``: head, hands, arms and torso.

    ``params.target`` may be a single string, a list of strings, or a tuple of
    strings. Sequence entries may combine direct, composite, image-relative and
    anatomical targets. Duplicate selected class ids are removed while
    preserving first-seen order.

    Pipeline
    --------
    The node executes the following steps:

    1. Resolve the input image from ``path`` or from the upstream default input.
    2. Run the configured Sapiens2 segmentation model through Hugging Face
       Transformers.
    3. Use the Sapiens2 image processor's semantic-segmentation post-processor
       to derive a per-pixel class map in original image coordinates.
    4. Resolve ``params.target``. Image-relative side targets compare the
       horizontal centers of their anatomical candidate masks and keep the
       leftmost/rightmost visible candidate.
    5. Build a boolean mask from the resolved class ids.
    6. Apply structural cleanup per selected semantic class, then union the
       cleaned parts. This preserves intentionally disconnected regions such as
       both hands, both feet, multiple clothing regions, or explicit target
       lists even when ``min_component_area='biggest'`` is used.
    7. Derive the target bounding box from the cleaned mask.
    8. Produce either an RGBA crop or a full-frame mask, depending on ``mode``.
    9. Write the output image and JSON sidecar metadata.

    Crop geometry
    -------------
    For ``mode='default'``, ``crop_mode`` controls the spatial layout of the
    produced RGBA image.

    Supported values are:

    - ``'trim'``:
        The output is an RGBA cutout cropped to the cleaned target bbox,
        optionally expanded by ``box_margin``. Alpha is derived from the
        cleaned semantic mask.

    - ``'bbox'``:
        The output is the rectangular crop inside the cleaned target bbox,
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
        channel contains the cleaned semantic mask.

    For ``mode='mask'`` and ``mode='negative-mask'``, ``crop_mode`` is ignored
    because mask outputs are always full-frame.

    Mask post-processing
    --------------------
    Structural cleanup from ``params.postprocess`` is applied in every output
    mode before deriving crop geometry, RGBA alpha, or full-frame mask output.
    Because cleanup is applied independently per selected semantic class, it is
    safe for intentionally disconnected composite targets.

    When ``mode`` is ``'mask'`` or ``'negative-mask'``, the cleaned positive
    mask is further post-processed before it is written to disk.

    Output-mask post-processing is applied in this order:

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
                Runtime device used for Sapiens2 inference. Default:
                ``'cuda'``.

            ``dtype`` : str, optional
                Torch dtype requested when loading the model. Default:
                ``'bfloat16'``.

                The model cache loads non-CUDA devices in float32 for runtime
                compatibility. CUDA devices use the requested dtype.

            ``segment_model`` : str, optional
                Hugging Face model identifier. Default:
                ``'facebook/sapiens2-seg-0.4b'``.

                The model must use the official 29-class Sapiens2 body-part
                segmentation taxonomy. The node validates class count at
                runtime because current Hugging Face configs expose generic
                labels such as ``LABEL_0`` rather than semantic names.

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
                image. Default: ``'trim'``.

            ``box_margin`` : int or str, optional
                Symmetric margin applied to the cleaned mask-derived selection
                bbox. Supported forms follow the standard size-expression
                convention: integer pixels, ``'<n>px'`` or ``'<n>%'``.
                Percentages are resolved against the mask bbox width for
                left/right and mask bbox height for top/bottom.

                Applies only when ``mode='default'`` and ``crop_mode`` is
                ``'trim'``, ``'bbox'`` or ``'bbox[w:h]'``. Ignored for
                ``crop_mode='full_frame'`` and mask outputs. For
                ``crop_mode='bbox[w:h]'``, the margin is applied before
                aspect-ratio expansion. Default: ``'8%'``.

            ``postprocess`` : dict, optional
                Structural cleanup applied to the selected Sapiens2 silhouette
                before deriving crop geometry, RGBA alpha, or full-frame mask
                output. This block defines the canonical shape used by the node,
                so it is applied in every mode. Output-only mask refinements
                such as ``close_radius``, ``dilate_radius`` and
                ``smoothing_radius`` are applied later and only for
                ``mode='mask'`` or ``mode='negative-mask'``.

                Processing order is fixed: ``fill_holes`` ->
                ``morph_open_radius`` -> ``min_component_area``.

                ``fill_holes`` : int, float, str, 'all' or None, optional
                    Fill enclosed background holes inside each selected semantic
                    part before removing thin details. ``0`` or ``None``
                    disables hole filling. ``'all'`` fills every enclosed hole.
                    Numeric values are pixel areas. Percentage strings such as
                    ``'1%'`` follow the shared component-area convention: the
                    percentage is measured on the image long side and squared
                    into an area threshold. Only holes with area less than or
                    equal to the resolved threshold are filled.

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
                    ``'biggest'`` keeps the largest connected component inside
                    each selected semantic class before the classes are unioned,
                    preserving multi-part targets such as both hands, both
                    feet, arms, legs, shoes, or explicit target lists.

                Default: all disabled.

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
            Model/runtime metadata, including the Sapiens2 segmenter model id,
            runtime device, requested dtype, and runtime dtype.

        ``sapiens2_segment_crop`` : dict
            Resolved semantic crop configuration, including ``target``,
            selected model label names, selected label ids, selected candidate
            names, and image-relative side-resolution metadata when applicable.

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
      directly from the cleaned semantic mask.
    - The official Sapiens2 segmentation checkpoints share the same 29-class
      taxonomy across the 0.4B, 0.8B, 1B and 5B variants.
    - Current Hugging Face configs for these checkpoints expose generic
      ``LABEL_n`` names, so this node keeps the semantic label table locally and
      validates only the number of classes against the loaded model.
    - Public side-specific targets such as ``left-hand`` and ``right-foot`` use
      image/viewer perspective. Use explicit anatomical targets such as
      ``anatomical-left-hand`` when subject-side semantics are required. Use
      composite aliases such as ``hands`` and ``feet`` when both sides should be
      selected.
    - The underlying parser is human-centric. Multi-person images may produce a
      merged semantic mask rather than a separable subject.
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

        segments, runtime_dtype = predict_segments(
            image_rgb,
            model_id=cfg.segment_model,
            device=cfg.device,
            dtype=cfg.dtype,
            error_prefix='Sapiens2SegmentCrop',
        )

        resolved = resolve_parsed_target(
            cfg.parsed_target,
            node_id=node_id,
            segments=segments,
            error_prefix='Sapiens2SegmentCrop',
        )
        resolved_labels = resolved.labels
        label_ids = resolved.label_ids
        selected_mask = np.isin(
            segments, np.asarray(label_ids, dtype=np.int64))

        if not np.any(selected_mask):
            raise RuntimeError(
                f"Sapiens2SegmentCrop node '{node_id}': selected mask is "
                f"empty for target={cfg.target!r}."
            )

        selected_mask = cleanup_shape_mask_by_parts(
            selected_mask,
            segment_part_masks(segments, label_ids),
            **cfg.shape_cleanup,
        )
        if not np.any(selected_mask):
            raise RuntimeError(
                f"Sapiens2SegmentCrop node '{node_id}': structural cleanup "
                f"removed the entire selected mask for target={cfg.target!r}."
            )

        alpha_full = selected_mask.astype(np.uint8) * 255
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = tight_mask_bbox(alpha_full)

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
                x1, y1, x2, y2 = expand_clip_bbox_by_size_expr(
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
            out_mask_u8 = prepare_output_mask(
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
            'postprocess': cfg.shape_cleanup,
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
            'sapiens2_segment_crop': {
                **params,
                'labels': resolved_labels,
                'label_ids': [int(label_id) for label_id in label_ids],
                'selected_candidates': resolved.selected_candidates,
                'side_resolutions': resolved.side_resolutions,
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
