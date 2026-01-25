import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image

from stability.cache.models import get_controlnet_aux_annotator
from stability.dag import NodeRef
from stability.nodes.preprocess.utils import *
from stability.nodes.utils import *
from third_party.controlnet_aux.processor import MODEL_PARAMS, MODELS


@dataclass
class ImgAuxMap(NodeRef):
    """
    Single-image auxiliary preprocessing node producing a packed control image
    via ControlNet auxiliary annotators.

    This node reads an input image (via the default DAG connection) and produces a
    single packed control image suitable for conditioning diffusion pipelines
    (e.g. ControlNet, T2I-Adapter, pose/edge/depth-guided pipelines).

    The control image is generated using one of the auxiliary annotators provided
    by ``controlnet-aux`` (e.g. OpenPose, Canny, HED, MiDaS, LineArt, Zoe, etc.),
    selected through the ``processor`` parameter. All annotators share a common execution
    and resizing pipeline, ensuring consistent spatial semantics across different
    control modalities.

    The output control image is a standard BGR image (OpenCV convention) written
    to disk as PNG, together with a JSON sidecar containing resolved parameters,
    input/output metadata, and timing information.

    Processing overview
    -------------------
    1) Load the input image in BGR format (OpenCV).
    2) Convert to RGB and resize for inference so that the long side equals
    ``detect_long_side`` (aspect ratio preserved).
    3) Run the selected controlnet-aux annotator on the resized image.
    4) Convert the annotator output back to a NumPy array.
    5) Resize the output control image to the target size:
    - If ``out_width``/``out_height`` are provided, fit and optionally pad.
    - Otherwise, preserve the original input image size.
    6) Write the output control image to disk (PNG) and emit a JSON sidecar.

    Parameters
    ----------
    id : str, optional
        Unique node identifier within the DAG. If not provided, it is auto-generated
        by the enclosing DAG/NodeRef implementation.
    spec : dict | str | pathlib.Path, optional
        Node configuration. The node resolves configuration using ``resolve_spec``.
        Parameters are read at the top level of the resolved spec.

    Inputs
    ------
    default : dict
        Required upstream output mapping that must contain the input image path.
        The node expects either:

        - ``image`` : str
            Filesystem path to the input image.
        - ``path`` : str
            Alternative key for the filesystem path.

    Outputs
    -------
    dict
        JSON-serializable dictionary containing at least:

        - ``ok`` : bool
        - ``node`` : str
            Operator name (derived from the class name by ``NodeRef``).
        - ``id`` : str
            Node id.
        - ``input_image`` : str
            Input image path.
        - ``image`` : str
            Filesystem path to the generated control image (PNG).
        - ``input`` : dict
            Input image resolution metadata.
        - ``output`` : dict
            Output image resolution metadata.
        - ``params`` : dict
            Resolved preprocessing and annotator parameters.
        - ``metadata`` : str
            Filesystem path to the JSON sidecar.
        - ``timing`` : dict
            Elapsed time for the run.

    Configuration keys (spec)
    -------------------------
    Annotator selection
    ~~~~~~~~~~~~~~~~~~~
    processor : str, default "openpose_full"
        Control modality / annotator identifier. Must be one of the keys supported
        by ``controlnet-aux`` (e.g. ``openpose_full``, ``canny``, ``depth_midas``,
        ``lineart_realistic``, ``normal_bae``, etc.).

    Inference resizing
    ~~~~~~~~~~~~~~~~~~
    detect_long_side : int, default 512
        Long-side resolution used for annotator inference. The input image is resized
        before running the annotator to reduce computation cost while preserving
        aspect ratio. Typical values are 512 (quality) or 384 (fast mode).

    Output resizing
    ~~~~~~~~~~~~~~~
    out_width : int | None, default None
        Target output width. If not provided, the original input image width is used.
    out_height : int | None, default None
        Target output height. If not provided, the original input image height is used.
    keep_aspect : bool, default True
        If True, the output control image is resized using letterbox fit (aspect ratio
        preserved) and padded as needed. If False, the image is stretched to exactly
        match the target size.
    pad_to_multiple_of : int, default 64
        If > 1, output dimensions are rounded up to the nearest multiple of this value
        (commonly required by diffusion backbones such as SDXL).

    Annotator parameters
    ~~~~~~~~~~~~~~~~~~~~
    params : dict, optional
        Dictionary of annotator-specific parameters that override the defaults defined
        by ``MODEL_PARAMS[processor]``. The exact keys depend on the selected annotator.

    Notes
    -----
    - All outputs are written as PNG to avoid lossy compression artifacts that could
    degrade thin edges, pose skeletons, or depth gradients.
    - This node is purely spatial and stateless; no temporal smoothing or tracking
    is applied. For temporally consistent preprocessing, use the corresponding
    video node (``VideoAuxMap``).
    - Some annotators (e.g. DWPose) may require additional backends and are not
    available out-of-the-box in all environments.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _build_annotator(self, processor: str, device: str):
        if processor not in MODELS:
            raise ValueError(
                f"Unknown processor={processor}. Allowed: {list(MODELS.keys())}")

        cls = MODELS[processor]["class"]
        is_ckpt = bool(MODELS[processor]["checkpoint"])

        # NOTE: dwpose in controlnet-aux is not hub-loadable in your env (no .from_pretrained)
        if processor == "dwpose":
            raise ValueError(
                "dwpose is not available out-of-the-box in controlnet-aux (no from_pretrained). "
                "Use openpose_* for now, or install a dedicated DWPose backend later."
            )

        if is_ckpt:
            return get_controlnet_aux_annotator(processor=processor, cls=cls, device=device)
        else:
            return cls()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        input = input or {}
        upstream = input.get("default")
        if upstream is None:
            raise ValueError(
                "ImgAuxMap requires an input image wired to the default input.")

        in_path = upstream.get("image") or upstream.get("path")
        if not in_path:
            raise ValueError(
                "Upstream output does not contain an image path (image/path).")

        spec = resolve_spec(self.spec)

        processor = spec.get("processor", "openpose_full")
        device = spec.get('device', 'cuda')

        detect_long_side = int(spec.get("detect_long_side", 512))
        out_w = spec.get("out_width", None)
        out_h = spec.get("out_height", None)
        pad_to_multiple_of = int(spec.get("pad_to_multiple_of", 64))
        keep_aspect = bool(spec.get("keep_aspect", True))

        # params: defaults + overrides
        params = dict(MODEL_PARAMS.get(processor, {}))
        params.update(spec.get("params", {}) or {})

        frame_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {in_path}")

        in_h, in_w = frame_bgr.shape[:2]

        # decide output size
        target_w = int(out_w) if out_w else in_w
        target_h = int(out_h) if out_h else in_h
        if pad_to_multiple_of and pad_to_multiple_of > 1:
            target_w = round_up(target_w, pad_to_multiple_of)
            target_h = round_up(target_h, pad_to_multiple_of)

        annotator = self._build_annotator(processor, device=device)

        # --- inference resize on RGB ---
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_small = resize_long_side_rgb(rgb, detect_long_side)

        pil_in = Image.fromarray(rgb_small)
        # controlnet-aux detectors take PIL
        pil_out = annotator(pil_in, **params)
        out_rgb = np.array(pil_out, dtype=np.uint8)

        # --- output fit/pad ---
        out_rgb = fit_to_target_rgb(
            out_rgb,
            out_w=target_w,
            out_h=target_h,
            keep_aspect=keep_aspect,
            pad_to_multiple_of=1,  # already padded target dims above
        )

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = make_node_output_path(
            out_dir=out_dir, node_id=self.id, ext="png")
        cv2.imwrite(str(out_path), out_bgr)

        dt = time.perf_counter() - t0

        meta = {
            "ok": True,
            "node": self.op,
            "id": self.id,
            "input_image": str(in_path),
            "image": str(out_path),
            "input": {
                "width": in_w,
                "height": in_h,
            },
            "params": {
                "processor": processor,
                "detect_long_side": detect_long_side,
                "out_width": out_w,
                "out_height": out_h,
                "pad_to_multiple_of": pad_to_multiple_of,
                "keep_aspect": keep_aspect,
                "annotator_params": params,
            },
            "output": {
                "width": target_w,
                "height": target_h,
            },
            "timing": {"seconds": round(dt, 3)},
        }

        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(json.dumps(
            meta, indent=2, ensure_ascii=False), encoding="utf-8")
        meta["metadata"] = str(meta_path)

        return meta
