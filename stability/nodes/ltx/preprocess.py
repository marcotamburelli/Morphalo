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
from stability.nodes import make_node_output_path, resolve_spec
from stability.nodes.ltx import read_video_info
from stability.nodes.preprocess.utils import *
from third_party.controlnet_aux.processor import MODEL_PARAMS, MODELS


@dataclass
class VideoAuxMap(NodeRef):
    """
    Video auxiliary preprocessing node producing a packed control video
    via ControlNet auxiliary annotators.

    This node reads an input video (via the default DAG connection) and produces a
    control video suitable for video-conditioned diffusion pipelines (e.g. video
    ControlNet, pose-/edge-/depth-guided video generation).

    Each frame of the input video is processed independently using one of the
    auxiliary annotators provided by ``controlnet-aux`` (e.g. OpenPose, Canny, HED,
    MiDaS, LineArt, Zoe). The same preprocessing logic, resizing rules, and annotator
    parameters are applied consistently across all frames.

    The output is a standard MP4 video (OpenCV convention, BGR frames) plus a JSON
    sidecar containing resolved parameters, input/output metadata, frame counts,
    and timing information.

    Processing overview
    -------------------
    1) Open the input video and read basic metadata (FPS, resolution, frame count).
    2) For each frame:
    a) Convert from BGR to RGB.
    b) Resize for inference so that the long side equals ``detect_long_side``
        (aspect ratio preserved).
    c) Run the selected controlnet-aux annotator.
    d) Resize the resulting control frame to the target output resolution
        (fit+pad or stretch).
    3) Write processed frames to the output video using a constant FPS.
    4) Emit a JSON sidecar summarizing parameters, statistics, and timing.

    Temporal semantics
    ------------------
    Each frame is processed independently. No temporal smoothing, tracking, or
    cross-frame identity association is performed. Temporal coherence is delegated
    to the downstream video diffusion model.

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
        Required upstream output mapping that must contain the input video path.
        The node expects either:

        - ``video`` : str
            Filesystem path to the input video.
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
        - ``video`` : str
            Filesystem path to the generated control video (MP4).
        - ``input`` : dict
            Input video metadata (FPS, resolution, frame count).
        - ``params`` : dict
            Resolved preprocessing and annotator parameters.
        - ``stats`` : dict
            Frame processing statistics.
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
        by ``controlnet-aux``.

    Inference resizing
    ~~~~~~~~~~~~~~~~~~
    detect_long_side : int, default 512
        Long-side resolution used for annotator inference. Each frame is resized
        before running the annotator to reduce computation cost while preserving
        aspect ratio.

    Output resizing
    ~~~~~~~~~~~~~~~
    out_width : int | None, default None
        Target output frame width. If not provided, the input video width is used.
    out_height : int | None, default None
        Target output frame height. If not provided, the input video height is used.
    keep_aspect : bool, default True
        If True, control frames are resized using letterbox fit (aspect ratio
        preserved) and padded as needed. If False, frames are stretched.
    pad_to_multiple_of : int, default 64
        If > 1, output frame dimensions are rounded up to the nearest multiple of
        this value.

    Annotator parameters
    ~~~~~~~~~~~~~~~~~~~~
    params : dict, optional
        Dictionary of annotator-specific parameters overriding
        ``MODEL_PARAMS[processor]``.

    Video processing
    ~~~~~~~~~~~~~~~~
    fps : float | None, default None
        Optional output FPS override. If not provided, the input video FPS is used.
    max_frames : int, default -1
        If > 0, process only the first ``max_frames`` frames of the input video.

    Notes
    -----
    - Output is encoded as MP4 using a fixed codec and constant FPS.
    - All frames are processed independently; this node does not perform temporal
    smoothing or tracking.
    - For consistent conditioning, it is recommended to use the same
    ``detect_long_side`` and output sizing across all auxiliary maps in a pipeline.
    - Some annotators may be CPU-bound and can be computationally expensive on long
    videos.
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
                "VideoAuxMap requires an input video wired to the default input.")

        in_path = upstream.get("video") or upstream.get("path")
        if not in_path:
            raise ValueError(
                "Upstream output does not contain a video path (video/path).")

        spec = resolve_spec(self.spec)

        processor = spec.get("processor", "openpose_full")
        device = spec.get('device', 'cuda')

        detect_long_side = int(spec.get("detect_long_side", 512))
        out_w = spec.get("out_width", None)
        out_h = spec.get("out_height", None)
        pad_to_multiple_of = int(spec.get("pad_to_multiple_of", 64))
        keep_aspect = bool(spec.get("keep_aspect", True))
        max_frames = int(spec.get("max_frames", -1))
        fps_override = spec.get("fps", None)

        # params: defaults + overrides
        params = dict(MODEL_PARAMS.get(processor, {}))
        params.update(spec.get("params", {}) or {})

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_video_path = make_node_output_path(
            out_dir=out_dir, node_id=self.id, ext="mp4")

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {in_path}")

        fps_in, in_w, in_h, in_n = read_video_info(cap)
        fps_out = float(fps_override) if fps_override else (
            fps_in if fps_in > 0 else 24.0
        )

        # decide output size
        target_w = int(out_w) if out_w else in_w
        target_h = int(out_h) if out_h else in_h
        if pad_to_multiple_of and pad_to_multiple_of > 1:
            target_w = round_up(target_w, pad_to_multiple_of)
            target_h = round_up(target_h, pad_to_multiple_of)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_video_path),
            fourcc,
            fps_out,
            (target_w, target_h)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open VideoWriter: {out_video_path}")

        annotator = self._build_annotator(processor, device=device)

        # rewind
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames_total = 0
        frames_written = 0

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames_total += 1
                if max_frames > 0 and frames_total > max_frames:
                    break

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
                writer.write(out_bgr)
                frames_written += 1

        finally:
            cap.release()
            writer.release()

        dt = time.perf_counter() - t0

        meta = {
            "ok": True,
            "node": self.op,
            "id": self.id,
            "video": str(out_video_path),
            "input": {
                "video": str(in_path),
                "fps": fps_in,
                "width": in_w,
                "height": in_h,
                "num_frames": in_n,
            },
            "params": {
                "processor": processor,
                "detect_long_side": detect_long_side,
                "out_width": out_w,
                "out_height": out_h,
                "pad_to_multiple_of": pad_to_multiple_of,
                "keep_aspect": keep_aspect,
                "fps_out": fps_out,
                "max_frames": max_frames,
                "annotator_params": params,
                "fourcc": "mp4v",
            },
            "stats": {
                "frames_total": frames_total,
                "frames_written": frames_written,
            },
            "timing": {"seconds": round(dt, 3)},
        }

        meta_path = out_video_path.with_suffix(".json")
        meta_path.write_text(json.dumps(
            meta, indent=2, ensure_ascii=False), encoding="utf-8")
        meta["metadata"] = str(meta_path)

        return meta
