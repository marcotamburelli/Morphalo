import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.ltx.preprocess.utils.canny_extractor import CannyExtractor
from morphalo.nodes.ltx.video_utils import read_video_info


@dataclass
class VideoCannyMap(NodeRef):
    """
    Video Canny preprocessing node producing a single packed control video.

    This node reads an input video (via the default DAG connection) and generates a
    single packed control video suitable for downstream edge-conditioned adapters
    (e.g. ControlNet/Canny-like conditioning or video-conditioned pipelines). The
    produced control video is a standard BGR video (OpenCV convention) composed of
    Canny edges rendered on either a black background (default) or the original
    frames (debug).

    Processing overview
    -------------------
    1) Decode frames from the input video.
    2) Convert each frame to grayscale and optionally apply Gaussian blur.
    3) Run Canny edge detection per frame with user-provided thresholds.
    4) Optionally post-process edges (e.g. dilation and/or inversion).
    5) Render edges into a single control frame on black background (default) or
       over the original frame (debug).
    6) Optionally downscale the rendered control frame by setting ``long_side``.
    7) Encode and write the packed control video to disk (mp4), plus a JSON sidecar.

    Output semantics
    ----------------
    Each output frame contains the Canny edge map derived from the corresponding
    input frame. The node does not attempt temporal smoothing, tracking, or
    edge-consistency enforcement across frames. Any temporal stability depends on
    the input video quality and on preprocessing choices (blur/thresholds/dilation).

    Outputs
    -------
    The node returns a JSON-serializable dict containing at least:
    - ``video``: filesystem path to the packed control video (mp4)
    - ``metadata``: filesystem path to a JSON sidecar describing the run

    Parameters
    ----------
    id : str, optional
        Unique node identifier within the DAG. If not provided, it is auto-generated
        by the enclosing DAG/NodeRef implementation.
    spec : dict or str or pathlib.Path or sequence of (dict or str or pathlib.Path)
        Node configuration specification.

        The specification may be provided as an in-memory dictionary, as a path
        to a HOCON configuration file, or as a sequence of such elements. When a
        sequence is given, each element is resolved independently and merged from
        left to right, with later elements overriding earlier ones.

    Configuration keys (spec)
    -------------------------
    Preprocessing / rendering
    ~~~~~~~~~~~~~~~~~~~~~~~~
    preserve_bg : bool, default False
        If True, render edges on top of the original frame (debug).
        If False, render on a black background (recommended for control inputs).
    long_side : int | None, default None
        If set, downscale the rendered control frame so that its long side equals
        ``long_side`` (aspect ratio preserved). Resize is applied after edge
        rendering.
    max_frames : int, default -1
        If > 0, process only the first ``max_frames`` frames (useful for tests).

    Canny configuration
    ~~~~~~~~~~~~~~~~~~
    low_threshold : int, default 80
        Lower hysteresis threshold for ``cv2.Canny``.
    high_threshold : int, default 160
        Upper hysteresis threshold for ``cv2.Canny``.
    aperture_size : int, default 3
        Aperture size for the Sobel operator used internally by Canny
        (typically 3, 5, or 7).
    l2_gradient : bool, default False
        If True, use a more precise L2 norm for gradient magnitude in Canny.
    blur_ksize : int, default 5
        Gaussian blur kernel size applied before Canny to reduce noise.
        Use 0 or 1 to disable. Even values are rounded up to the next odd value.
    blur_sigma : float, default 0.0
        Gaussian sigma for pre-blur. A value of 0 lets OpenCV choose automatically.
    dilate : int, default 0
        If > 0, apply morphological dilation after Canny to thicken edges.
    dilate_iter : int, default 1
        Number of dilation iterations if ``dilate`` is enabled.
    invert : bool, default False
        If True, invert the final edge map (e.g. black edges on white).

    Output encoding
    ~~~~~~~~~~~~~~~
    fps : float | None, default None
        If set, overrides the output FPS. If None, the input video's FPS is used
        (or a fallback default if unavailable).

    Attributes
    ----------
    op : str
        Operator identifier derived from the concrete class name (NodeRef contract).
    extractor : CannyExtractor
        Stateless extractor used to compute and render edges for each frame.

    Notes
    -----
    - Input contract
      This node expects an upstream output wired into the default input containing
      a filesystem path to the input video. The path is typically provided under
      the key ``video`` (or ``path``), depending on the upstream node.

    - The packed control video is written using OpenCV's VideoWriter with
      ``fourcc = cv2.VideoWriter_fourcc(*"mp4v")``. This implies lossy compression.
      For thin edges, compression may introduce minor artifacts. For high-fidelity
      datasets or training use cases, exporting individual PNG frames or re-encoding
      the output with a higher-quality codec is recommended.

    - If the input video is heavily compressed, noisy, or low resolution, Canny
      outputs may flicker. Increasing pre-blur, adjusting thresholds, or rendering
      at higher inference resolution (then downscaling) can improve stability.
    """

    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        input = input or {}
        init_up = input.get('default')
        if init_up is None:
            raise ValueError(
                'VideoCannyMap requires an init video wired into the default input (src >> video_canny).')

        in_path = init_up.get('video') or init_up.get('path')
        if not in_path:
            raise ValueError(
                "Init image upstream output must contain 'video' (path).")

        spec = resolve_spec(self.spec)

        # --- resolve spec (same pattern as your nodes) ---
        long_side = spec.get('long_side', 512)
        preserve_bg = spec.get('preserve_bg', False)

        # NEW: two-stage sizing (controlnet_aux-like)
        detect_resolution = spec.get('detect_resolution', 512)
        image_resolution = spec.get('image_resolution', 512)

        low_threshold = spec.get('low_threshold', 80)
        high_threshold = spec.get('high_threshold', 160)
        blur_ksize = spec.get('blur_ksize', 5)
        blur_sigma = spec.get('blur_sigma', 0.0)
        aperture_size = spec.get('aperture_size', 3)
        l2_gradient = spec.get('l2_gradient', False)
        dilate = spec.get('dilate', 0)
        dilate_iter = spec.get('dilate_iter', 1)
        invert = spec.get('invert', False)

        out_dir = Path(output_dir)

        out_video_path = make_node_output_path(
            out_dir=out_dir,
            node_id=self.id,
            ext='mp4'
        )

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f'Unable to open video: {in_path}')

        fps_in, in_w, in_h, in_frames = read_video_info(cap)
        fps_out = float(spec.get('fps', fps_in if fps_in > 0 else 24.0))

        extractor = CannyExtractor()

        # read first frame to determine output size (must match extractor output)
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError('Empty video or failed first frame read')

        out0 = extractor.render_frame(
            frame0,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
            aperture_size=aperture_size,
            l2_gradient=l2_gradient,
            dilate=dilate,
            dilate_iter=dilate_iter,
            preserve_bg=preserve_bg,
            invert=invert,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            long_side=long_side,
        )

        out_h, out_w = out0.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Unable to open VideoWriter: {out_video_path}')

        # rewind to start, and process all frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames_total = 0
        frames_written = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1

            out = extractor.render_frame(
                frame,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                blur_ksize=blur_ksize,
                blur_sigma=blur_sigma,
                aperture_size=aperture_size,
                l2_gradient=l2_gradient,
                dilate=dilate,
                dilate_iter=dilate_iter,
                preserve_bg=preserve_bg,
                invert=invert,
                detect_resolution=detect_resolution,
                image_resolution=image_resolution,
                long_side=long_side,
            )

            # safety: enforce consistent size (VideoWriter requires fixed WxH)
            if out.shape[0] != out_h or out.shape[1] != out_w:
                # in pratica non dovrebbe succedere, ma meglio essere robusti
                out = cv2.resize(out, (out_w, out_h),
                                 interpolation=cv2.INTER_AREA)

            writer.write(out)
            frames_written += 1

        cap.release()
        writer.release()

        dt_s = time.perf_counter() - t0

        meta = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'video': str(out_video_path),

            'input': {
                'video': str(in_path),
                'fps': fps_in,
                'width': in_w,
                'height': in_h,
                'num_frames': in_frames,
            },

            'params': {
                'detect_resolution': detect_resolution,
                'image_resolution': image_resolution,
                'long_side': long_side,
                'preserve_bg': preserve_bg,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                'blur_ksize': blur_ksize,
                'blur_sigma': blur_sigma,
                'aperture_size': aperture_size,
                'l2_gradient': l2_gradient,
                'dilate': dilate,
                'dilate_iter': dilate_iter,
                'invert': invert,
                'fps_out': fps_out,
                'fourcc': 'mp4v',
            },

            'stats': {
                'frames_total': frames_total,
                'frames_written': frames_written,
            },

            'timing': {'seconds': round(dt_s, 3)},
        }

        meta_path = out_video_path.with_suffix('.json')
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        return meta
