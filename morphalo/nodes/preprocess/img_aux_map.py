import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image

from morphalo.cache.models import get_controlnet_aux_annotator
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils import (fit_to_target_rgb,
                                             resolve_min_component_area,
                                             round_up,
                                             validate_min_component_area)
from morphalo.nodes.preprocess.utils.mask_ops import remove_small_components
from morphalo.nodes.sdxl_resolve import resolve_single_image_path
from third_party.controlnet_aux.processor import MODEL_PARAMS, MODELS


def _read_postprocess_config(
    value: Any,
    *,
    node_id: str,
) -> Dict[str, Any]:
    """
    Validate optional auxiliary-map post-processing configuration.

    The post-processing block is intentionally small and edge-map oriented. It
    is disabled when omitted or empty, and is meant for sparse structural maps
    such as Canny, line art, HED, PiDiNet, and scribble outputs.

    Do not enable this block for dense or semantically encoded maps unless that
    loss of information is intentional. Depth and normal maps store meaningful
    continuous grayscale or color values; segmentation maps store category
    colors; pose maps encode small skeleton/keypoint marks. Binarizing and
    morphologically opening those maps can destroy the conditioning signal.

    Processing order is:
      1. convert the annotator output to grayscale;
      2. if ``binarize_threshold`` is provided, binarize the map;
      3. infer foreground line pixels from the map polarity;
      4. apply ``morph_open_radius`` if enabled;
      5. remove components below ``min_component_area`` if enabled;
      6. preserve original grays when no binarization threshold was provided,
         otherwise restore the detected or requested binary line polarity.

    Supported keys are:

    ``binarize_threshold``
        Optional grayscale threshold in ``[0, 255]`` used to convert the
        annotator output to a binary line/background map. When omitted or
        ``None``, grayscale line intensities are preserved and cleanup operates
        on a foreground mask inferred from polarity.

    ``min_component_area``
        Absolute pixel area or percentage string used to remove disconnected
        foreground components whose area is less than or equal to the resolved
        value. Percentage strings follow the shared preprocessing convention:
        ``'1%'`` is measured on the image long side and converted to area.

    ``morph_open_radius``
        Radius in pixels for morphological opening, i.e. erosion followed by
        dilation. This removes small local details while approximately
        preserving larger surviving contours.

    ``polarity``
        Foreground polarity for binarization. ``'bright'`` means bright lines on
        a dark background, ``'dark'`` means dark lines on a bright background,
        and ``'auto'`` infers the less-common side as foreground.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{node_id}': postprocess must be a dictionary")

    cfg = dict(value)
    if not cfg:
        return {}

    raw_threshold = cfg.get('binarize_threshold', None)
    if raw_threshold is None:
        threshold = None
    else:
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{node_id}': postprocess.binarize_threshold must be an integer or None"
            ) from exc
        if not 0 <= threshold <= 255:
            raise ValueError(
                f"'{node_id}': postprocess.binarize_threshold must be in [0, 255]"
            )

    min_component_area = cfg.get('min_component_area', 0)
    validate_min_component_area(
        min_component_area,
        node_id=node_id,
        param_name='postprocess.min_component_area',
    )

    try:
        morph_open_radius = int(cfg.get('morph_open_radius', 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{node_id}': postprocess.morph_open_radius must be an integer"
        ) from exc
    if morph_open_radius < 0:
        raise ValueError(
            f"'{node_id}': postprocess.morph_open_radius must be >= 0"
        )

    polarity = str(cfg.get('polarity', 'auto'))
    if polarity not in ('auto', 'bright', 'dark'):
        raise ValueError(
            f"'{node_id}': postprocess.polarity must be 'auto', 'bright', or 'dark'"
        )

    return {
        'binarize_threshold': threshold,
        'min_component_area': min_component_area,
        'morph_open_radius': morph_open_radius,
        'polarity': polarity,
    }


def _postprocess_aux_map(
    rgb: np.ndarray,
    *,
    config: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Simplify a sparse auxiliary edge/line map.

    The function converts ``rgb`` to grayscale, binarizes it, treats the line
    pixels as foreground, optionally removes small disconnected components, and
    optionally applies morphological opening. It returns an RGB image with the
    same line polarity as the detected or requested input style.
    """
    if not config:
        return rgb, {'enabled': False}

    threshold_value = config.get('binarize_threshold', None)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    requested_polarity = str(config.get('polarity', 'auto'))
    if requested_polarity == 'auto':
        polarity = 'dark' if float(np.mean(gray)) > 127.0 else 'bright'
    else:
        polarity = requested_polarity

    if threshold_value is None:
        foreground = gray < 255 if polarity == 'dark' else gray > 0
    else:
        threshold = int(threshold_value)
        bright = gray > threshold
        foreground = ~bright if polarity == 'dark' else bright

    morph_open_radius = int(config.get('morph_open_radius', 0))
    if morph_open_radius > 0:
        k = 2 * morph_open_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_u8 = foreground.astype(np.uint8) * 255
        foreground = cv2.morphologyEx(
            fg_u8,
            cv2.MORPH_OPEN,
            kernel,
        ) > 0

    min_component_area = resolve_min_component_area(
        config.get('min_component_area', 0),
        width=rgb.shape[1],
        height=rgb.shape[0],
    )
    foreground = remove_small_components(
        foreground,
        min_area=min_component_area,
    )

    if threshold_value is None:
        background = 255 if polarity == 'dark' else 0
        out_gray = np.full_like(gray, background, dtype=np.uint8)
        out_gray[foreground] = gray[foreground]
    else:
        out_gray = np.zeros_like(gray, dtype=np.uint8)
        if polarity == 'dark':
            out_gray[:, :] = 255
            out_gray[foreground] = 0
        else:
            out_gray[foreground] = 255

    out_rgb = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)
    resolved = {
        'enabled': True,
        'binarize_threshold': threshold_value,
        'min_component_area': config.get('min_component_area', 0),
        'resolved_min_component_area': min_component_area,
        'morph_open_radius': morph_open_radius,
        'polarity': polarity,
        'requested_polarity': requested_polarity,
    }
    return out_rgb, resolved


@dataclass
class ImgAuxMap(CudaPostRunMixin, NodeRef):
    """
    Single-image auxiliary preprocessing node producing a ControlNet-ready map.

    This node reads one input image, runs a ``controlnet-aux`` annotator on it,
    and writes the resulting auxiliary image as a PNG file. The generated image is
    intended to be used as spatial conditioning for downstream diffusion pipelines,
    for example ControlNet, T2I-Adapter, or other image-guided SDXL workflows.

    The concrete auxiliary map is selected through ``processor``. Typical examples
    include pose skeletons, Canny edges, HED edges, line art, depth maps, normal
    maps, and other structure-oriented conditioning signals.

    The node uses two separate sizing stages:

    1. Annotator inference resize
        The source image is passed to the selected ``controlnet-aux`` annotator.
        Native annotator parameters such as ``detect_resolution`` and
        ``image_resolution`` may be supplied at the top level of ``spec`` or
        inside ``params``.

    2. Final output resize
        The annotator result is resized to the final control-map canvas. This is
        controlled by ``out_width``, ``out_height``, ``pad_to_multiple_of``, and
        ``output_resize_mode``.

    These two stages are intentionally independent. ``detect_resolution`` and
    ``image_resolution`` follow the underlying ``controlnet-aux`` implementation:
    in the bundled processors they are interpreted as short-side resolutions and
    rounded to multiples of 64. They do not define the final output canvas.

    Processing flow
    ---------------
    The node performs the following steps:

    1. Resolve the input image path from the upstream default input.

    2. Load the image with OpenCV.

      The image is initially loaded in BGR format, following OpenCV conventions.

    3. Convert the image from BGR to RGB.

      ``controlnet-aux`` annotators operate on PIL/RGB images, so the image is
      converted before preprocessing.

    4. Run the selected ``controlnet-aux`` annotator on the original RGB image.

      Annotator defaults are read from ``MODEL_PARAMS[processor]`` and can be
      overridden through ``params``. The common native resize parameters
      ``detect_resolution`` and ``image_resolution`` may also be provided directly
      in ``spec``.

    5. Optionally post-process the annotator output.

      When ``postprocess`` is provided, the annotator output is simplified as a
      binary edge/line map before final resizing. This can remove weak gray
      edges, small disconnected components, and tiny local line details.

    6. Resolve the final output size.

      If ``out_width`` or ``out_height`` are provided, they define the requested
      final output canvas. Any missing dimension falls back to the corresponding
      original input-image dimension.

      The resolved size is then optionally rounded up using
      ``pad_to_multiple_of``.

    7. Resize the annotator output to the final canvas.

      The resize strategy is controlled by ``output_resize_mode``:

      - ``'stretch'`` resizes directly to the final canvas size.
      - ``'contain'`` preserves aspect ratio and pads the remaining area with
        black pixels.

    8. Save the final map as PNG.

      The final RGB map is converted back to BGR before being written with OpenCV.

    9. Write a JSON sidecar.

      The sidecar contains resolved parameters, input/output geometry, and timing
      information.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG. If not provided, an identifier is
        automatically generated by the enclosing DAG.

    spec : dict or str or pathlib.Path or sequence of (dict or str or pathlib.Path)
        Node configuration specification.

        The specification may be provided as an in-memory dictionary, as a path
        to a HOCON configuration file, or as a sequence of such elements. When a
        sequence is given, each element is resolved independently and merged from
        left to right, with later elements overriding earlier ones.

        Expected keys include:

        **Annotator selection**

        - ``processor`` : str, optional
            ``controlnet-aux`` processor identifier. Default is
            ``'openpose_full'``.

            The value must be one of the processors available in
            ``third_party.controlnet_aux.processor.MODELS``.

        - ``device`` : str, optional
            Device used for checkpoint-backed annotators. Default is ``'cuda'``.

            Lightweight or stateless processors may ignore this value.

        **Annotator-native sizing**

        - ``detect_resolution`` : int, optional
            Native ``controlnet-aux`` inference resolution.

            For the bundled processors this is interpreted as a short-side
            resolution and rounded internally to a multiple of 64.

        - ``image_resolution`` : int, optional
            Native ``controlnet-aux`` map resolution after detection/rendering.

            For the bundled processors this is also interpreted as a short-side
            resolution and rounded internally to a multiple of 64.

        **Final output sizing**

        - ``out_width`` : int or None, optional
            Requested final output width.

            If omitted, the original input image width is used. The resolved value
            may still be rounded up by ``pad_to_multiple_of``.

        - ``out_height`` : int or None, optional
            Requested final output height.

            If omitted, the original input image height is used. The resolved value
            may still be rounded up by ``pad_to_multiple_of``.

        - ``pad_to_multiple_of`` : int, optional
            Output-size alignment multiple. Default is ``64``.

            If greater than 1, the final output width and height are rounded up to
            the nearest multiple of this value. This is useful for diffusion
            pipelines that expect or prefer dimensions aligned to a fixed multiple.

            Use ``1`` or ``0`` to disable output-size alignment.

        - ``output_resize_mode`` : {'stretch', 'contain'}, optional
            Strategy used to adapt the annotator output to the final canvas.
            Default is ``'stretch'``.

            ``'stretch'``
                Resize the annotator output directly to the final width and height.

                This may slightly distort the auxiliary map if the target aspect
                ratio differs from the annotator output aspect ratio. However, it
                guarantees that the control image fills the whole target canvas and
                matches the downstream generation size exactly.

                This is usually the preferred mode when the control map is meant to
                align pixel-for-pixel with a diffusion canvas.

            ``'contain'``
                Preserve the annotator output aspect ratio and fit it inside the
                final canvas.

                Empty areas are padded with black pixels. This avoids geometric
                distortion, but the padding becomes part of the conditioning signal.

            ``'cover'`` is intentionally not supported because cropping a control map
            would discard spatial conditioning information.

        **Annotator parameters**

        - ``params`` : dict, optional
            Annotator-specific parameter overrides.

            The node starts from ``MODEL_PARAMS[processor]`` and applies this
            dictionary on top. Supported keys depend on the selected processor.
            Top-level ``detect_resolution`` and ``image_resolution`` override the
            same keys inside ``params`` when provided.

        **Auxiliary-map post-processing**

        - ``postprocess`` : dict, optional
            Optional edge/line-map simplification applied after the annotator and
            before final output resizing. It is disabled when omitted.

            This block is intended for sparse structural maps whose useful
            signal is "where are the lines?", for example ``canny``,
            ``lineart*``, ``scribble_*``, ``softedge_hed``, and
            ``softedge_pidinet``. It is useful when the annotator produces too
            many weak edges, texture marks, or disconnected specks and you want a
            simpler ControlNet/T2I-Adapter guide.

            It should normally be left disabled for dense or semantically encoded
            maps. In particular, avoid it for depth maps such as ``depth_midas``,
            normal maps such as ``normalbae``, segmentation maps, and pose maps
            such as ``openpose_full``. These maps carry information in continuous
            values, category colors, or tiny keypoint/skeleton geometry; the
            postprocess step may binarize the image and can therefore destroy
            the conditioning signal when a threshold is enabled.

            When enabled, the operation order is:
            grayscale conversion, optional ``binarize_threshold``, foreground
            mask inference from polarity, ``morph_open_radius``,
            ``min_component_area``, then binary polarity restoration or grayscale
            preservation.

            Supported keys:

            ``binarize_threshold`` : int or None, optional
                Grayscale threshold in ``[0, 255]`` used to convert the map to
                binary foreground/background. Lower values preserve weaker
                lines; higher values keep only stronger lines. If omitted or
                ``None``, the postprocess keeps the original grayscale
                intensities for surviving line pixels and only uses polarity to
                build a cleanup mask. Default: ``None``.

            ``min_component_area`` : int, float, str or None, optional
                Remove disconnected foreground components whose area is less than
                or equal to this threshold. Numeric values are pixel areas.
                Percentage strings such as ``'1%'`` follow the shared
                preprocessing convention: the percentage is measured linearly on
                the long side and converted to area. Default: ``0``.

            ``morph_open_radius`` : int, optional
                Radius in pixels for morphological opening, i.e. erosion followed
                by dilation. This removes small local details while approximately
                preserving larger contours, but can erase very thin Canny,
                lineart, or scribble strokes. Default: ``0``.

            ``polarity`` : {'auto', 'bright', 'dark'}, optional
                Foreground polarity for line pixels. ``'bright'`` means bright
                lines on a dark background; ``'dark'`` means dark lines on a
                bright background. ``'auto'`` treats the less-common side after
                thresholding as foreground and preserves that output style.
                Default: ``'auto'``.

    path : str or pathlib.Path, optional
        Input image path.

        If provided, this path is used directly and the node does not require an
        upstream default input. If omitted, the input image is resolved from the
        upstream ``default`` input.

    Inputs
    ------
    default : dict, optional
        Required only when ``path`` is not provided. Upstream output dictionary
        containing the source image path. The node expects either ``"image"`` or
        ``"path"`` to point to the image file.

    Outputs
    -------
    dict
        Primary output dictionary.

        The output contains at least:

        - ``ok`` : bool
          Success flag.

        - ``node`` : str
          Operator name derived from the concrete node class.

        - ``id`` : str
          Node identifier.

        - ``input_image`` : str
          Resolved input image path.

        - ``image`` : str
          Path to the generated auxiliary map PNG.

        - ``input`` : dict
          Input image metadata, including original width and height.

        - ``output`` : dict
          Output image metadata, including final width and height.

        - ``params`` : dict
          Resolved preprocessing and annotator parameters.

        - ``timing`` : dict
          Runtime timing information.

        - ``metadata`` : str
          Path to the JSON sidecar.

    Notes
    -----
    - The final output image is written as PNG to avoid lossy compression artifacts.
      This is important for thin edges, pose skeletons, line art, and depth
      gradients.

    - The final control map should usually have the same size as the image that will
      be generated or transformed downstream.

    - ``output_resize_mode='stretch'`` is generally the safest default for
      ControlNet-style workflows where the conditioning image must match the target
      generation canvas exactly.

    - ``output_resize_mode='contain'`` is useful when preserving the annotator map
      aspect ratio is more important than filling the whole output canvas.

    - Black padding introduced by ``'contain'`` is still part of the conditioning
      image and may influence generation.

    - This node is spatial and stateless. It does not perform temporal smoothing,
      tracking, or cross-frame consistency.

    - Some ``controlnet-aux`` processors may require additional checkpoints or
      third-party backends and may not be available in every environment.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        processor = spec.get('processor', 'openpose_full')
        model = MODELS.get(processor)
        return (
            bool(model and model['checkpoint'])
            and is_cuda_device(spec.get('device', 'cuda'))
        )

    def _build_annotator(self, processor: str, device: str):
        if processor not in MODELS:
            raise ValueError(
                f'Unknown processor={processor}. Allowed: {list(MODELS.keys())}')

        cls = MODELS[processor]['class']
        is_ckpt = bool(MODELS[processor]['checkpoint'])

        # NOTE: dwpose in controlnet-aux is not hub-loadable in your env (no .from_pretrained)
        if processor == 'dwpose':
            raise ValueError(
                'dwpose is not available out-of-the-box in controlnet-aux (no from_pretrained). '
                'Use openpose_* for now, or install a dedicated DWPose backend later.'
            )

        if is_ckpt:
            return get_controlnet_aux_annotator(processor=processor, cls=cls, device=device)
        else:
            return cls()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        in_path = resolve_single_image_path(
            node_id=self.id,
            path=self.path,
            input=input,
        )
        spec = resolve_spec(self.spec)

        processor = spec.get('processor', 'openpose_full')
        device = spec.get('device', 'cuda')

        out_w = spec.get('out_width', None)
        out_h = spec.get('out_height', None)

        pad_to_multiple_of = int(spec.get('pad_to_multiple_of', 64))
        if pad_to_multiple_of < 0:
            raise ValueError(
                f"'{self.id}': invalid pad_to_multiple_of={pad_to_multiple_of!r}; "
                'expected a positive integer.'
            )

        output_resize_mode = str(spec.get('output_resize_mode', 'stretch'))
        if output_resize_mode not in ('stretch', 'contain'):
            raise ValueError(
                f"'{self.id}': invalid output_resize_mode={output_resize_mode!r} "
                "expected 'stretch' or 'contain'."
            )
        keep_aspect = output_resize_mode == 'contain'

        # params: defaults + overrides + top-level native annotator sizing
        params = dict(MODEL_PARAMS.get(processor, {}))
        params.update(spec.get('params', {}) or {})
        postprocess_config = _read_postprocess_config(
            spec.get('postprocess', None),
            node_id=self.id,
        )

        if 'detect_resolution' in spec:
            params['detect_resolution'] = spec['detect_resolution']
        elif 'detect_resolution' not in params:
            params['detect_resolution'] = 512

        if 'image_resolution' in spec:
            params['image_resolution'] = spec['image_resolution']
        elif 'image_resolution' not in params:
            params['image_resolution'] = 512

        for key in ('detect_resolution', 'image_resolution'):
            if key not in params:
                continue

            value = int(params[key])
            if value <= 0:
                raise ValueError(
                    f"'{self.id}': invalid {key}={params[key]!r}; "
                    'expected a positive integer.'
                )
            params[key] = value

        frame_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f'Unable to read image: {in_path}')

        in_h, in_w = frame_bgr.shape[:2]

        # decide output size
        target_w = int(out_w) if out_w is not None else in_w
        target_h = int(out_h) if out_h is not None else in_h

        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                f"'{self.id}': invalid output size {target_w}x{target_h}; "
                'expected positive dimensions.'
            )

        if pad_to_multiple_of and pad_to_multiple_of > 1:
            target_w = round_up(target_w, pad_to_multiple_of)
            target_h = round_up(target_h, pad_to_multiple_of)

        annotator = self._build_annotator(processor, device=device)

        # controlnet-aux detectors take PIL/RGB and perform their native resizing.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb)
        pil_out = annotator(pil_in, **params).convert('RGB')
        out_rgb = np.array(pil_out, dtype=np.uint8)
        out_rgb, resolved_postprocess = _postprocess_aux_map(
            out_rgb,
            config=postprocess_config,
        )

        # --- output fit/pad ---
        out_rgb = fit_to_target_rgb(
            out_rgb,
            out_w=target_w,
            out_h=target_h,
            keep_aspect=keep_aspect,
            pad_to_multiple_of=1,  # target dimensions are already aligned above
        )

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        out_dir = Path(output_dir)

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=self.id,
            ext='png',
        )
        cv2.imwrite(str(out_path), out_bgr)

        dt = time.perf_counter() - t0

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(in_path),
            'image': str(out_path),
            'input': {
                'width': in_w,
                'height': in_h,
            },
            'params': {
                'processor': processor,
                'detect_resolution': params.get('detect_resolution'),
                'image_resolution': params.get('image_resolution'),
                'out_width': out_w,
                'out_height': out_h,
                'pad_to_multiple_of': pad_to_multiple_of,
                'output_resize_mode': output_resize_mode,
                'annotator_params': params,
                'postprocess': resolved_postprocess,
            },
            'output': {
                'width': target_w,
                'height': target_h,
            },
            'timing': {'seconds': round(dt, 3)},
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
