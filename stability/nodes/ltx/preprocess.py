import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch

from stability.dag import NodeRef
from stability.nodes import make_node_output_path, resolve_spec
from stability.nodes.ltx import read_video_info
from stability.nodes.utils import (CannyExtractor, DepthVideoExtractor,
                                   ModelPaths, SkeletonExtractor,
                                   compute_long_side_resize)


# -----------------------------
# Small utils
# -----------------------------
def resize_long_side(bgr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = float(long_side) / float(max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return bgr


# -----------------------------
# The node: VideoPoseMap -> single RGB control video
# -----------------------------
@dataclass
class VideoPoseMap(NodeRef):
    """
    Video pose preprocessing node producing a single packed control video.

    This node reads an input video (via the default DAG connection) and generates a
    single packed control video suitable for downstream pose-conditioned adapters
    (e.g. IC-LoRA pose / video-conditioned pipelines). The produced control video is
    a standard BGR video (OpenCV convention) composed of rendered landmarks on either
    a black background (default) or the original frames (debug).

    Processing overview
    -------------------
    1) Decode frames from the input video.
    2) Run per-frame landmark inference using MediaPipe Tasks via ``SkeletonExtractor``
    in ``RunningMode.VIDEO`` (timestamped inference).
    3) Render pose / hands / face landmarks into a single control frame.
    4) Optionally downscale the rendered control frame by setting ``long_side``.
    5) Encode and write the packed control video to disk (mp4), plus a JSON sidecar.

    Multi-person output semantics
    -----------------------------
    The extractor may return multiple poses/faces/hands per frame depending on
    ``max_poses``, ``max_faces``, and ``max_hands``. The output control frame renders
    all detections using fixed colors. No per-person identity is encoded.

    Important: this node currently does NOT perform explicit multi-person tracking
    nor association of hands/face detections to specific people. Any tracking-like
    stability comes only from MediaPipe's internal temporal behavior in VIDEO mode.

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
    spec : dict | str | pathlib.Path
        Node configuration. The node resolves configuration using
        ``resolve_spec(self.spec)`` and reads parameters at the top level.

    Configuration keys (spec)
    -------------------------
    Preprocessing / rendering
    ~~~~~~~~~~~~~~~~~~~~~~~~
    preserve_bg : bool, default False
        If True, render landmarks on top of the original frame (debug).
        If False, render on a black background (recommended for control inputs).
    thickness : int, default 2
        Line thickness for edges.
    point_radius : int, default 2
        Point radius for face landmark drawing helpers (if used by the renderer).
    max_frames : int, default -1
        If > 0, process only the first ``max_frames`` frames (useful for tests).
    long_side : int | None, default None
        If set, downscale the rendered control frame so that its long side equals
        ``long_side`` (aspect ratio preserved). Resize is applied after inference
        and rendering.

    Extractor configuration
    ~~~~~~~~~~~~~~~~~~~~~~
    conf_min_pose : float, default 0.3
    conf_min_hand : float, default 0.3
    conf_min_face : float, default 0.3
    max_poses : int, default 4
    max_faces : int, default 4
    max_hands : int, default 8

    Model configuration
    ~~~~~~~~~~~~~~~~~~
    models : dict, required
        Dictionary used to build ``ModelPaths`` via ``ModelPaths(**spec['models'])``.
        Expected keys:
        - ``pose_task``: path to the MediaPipe pose landmarker .task file
        - ``hand_task``: path to the MediaPipe hand landmarker .task file
        - ``face_task``: path to the MediaPipe face landmarker .task file

    Attributes
    ----------
    op : str
        Operator identifier derived from the concrete class name (NodeRef contract).
    extractor : SkeletonExtractor
        Lazily created in ``run()`` and reused for the duration of the call.

    Notes
    -----
    - This node expects an upstream output wired into the default input containing
    a filesystem path to the input video. The path is typically provided under
    the key ``video`` (or ``path``), depending on the upstream node.

    - The packed control video is currently written using OpenCV's VideoWriter
    with ``fourcc = cv2.VideoWriter_fourcc(*"mp4v")``. This implies lossy
    compression. While suitable for preview and inference-time control inputs,
    this may introduce minor edge artifacts on thin lines. For high-fidelity
    datasets or training use cases, exporting individual PNG frames or
    re-encoding the output with a higher-quality codec is recommended.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        input = input or {}
        init_up = input.get('default')
        if init_up is None:
            raise ValueError(
                'VideoPoseMap requires an init video wired into the default input (src >> video_pose).')

        in_path = init_up.get('video') or init_up.get('path')
        if not in_path:
            raise ValueError(
                "Init image upstream output must contain 'video' (path).")

        spec = resolve_spec(self.spec)

        # preprocessing
        preserve_bg = spec.get('preserve_bg', False)
        thickness = spec.get('thickness', 2)
        point_radius = spec.get('point_radius', 2)
        max_frames = spec.get('max_frames', -1)
        long_side = spec.get('long_side', None)

        # Extractor params
        conf_min_pose = spec.get('conf_min_pose', 0.3)
        conf_min_hand = spec.get('conf_min_hand', 0.3)
        conf_min_face = spec.get('conf_min_face', 0.3)
        max_poses = spec.get('max_poses', 4)
        max_faces = spec.get('max_faces', 4)
        max_hands = spec.get('max_hands', 8)

        models = ModelPaths(**spec.get('models', {}))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_video_path = make_node_output_path(
            out_dir=out_dir,
            node_id=self.id,
            ext='mp4'
        )

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise FileNotFoundError(
                f'Impossible to open video: {in_path}')

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # fallback

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if long_side is not None:
            out_h, out_w, _ = compute_long_side_resize(h, w, long_side)
        else:
            out_h = h
            out_w = w

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))

        if not writer.isOpened():
            raise RuntimeError(
                f'Impossible to create writer: {out_video_path}')

        extractor = SkeletonExtractor(
            models=models,
            conf_min_pose=conf_min_pose,
            conf_min_hand=conf_min_hand,
            conf_min_face=conf_min_face,
            max_poses=max_poses,
            max_faces=max_faces,
            max_hands=max_hands,
            running_mode="VIDEO"
        )

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_ms = frame_idx * \
                    int(1000 // fps) if fps >= 1 else frame_idx * 33

                out = extractor.render_frame(
                    frame_bgr=frame,
                    thickness=thickness,
                    point_radius=point_radius,
                    preserve_bg=preserve_bg,
                    timestamp_ms=timestamp_ms,
                    long_side=long_side
                )
                writer.write(out)

                frame_idx += 1
                if max_frames > 0 and frame_idx >= max_frames:
                    break
        finally:
            extractor.close()
            cap.release()
            writer.release()

        dt = time.perf_counter() - t0

        if frame_idx == 0:
            raise ValueError('No frames decoded')

        out = {
            'ok': True,
            'node': 'VideoPoseMap',
            'input_video': in_path,
            'video': str(out_video_path),
            'model': {
                'pose_task':  models.pose_task,
                'hand_task':  models.hand_task,
                'face_task':  models.face_task
            },
            'params': {
                'long_side': long_side,
                'preserve_bg': preserve_bg,
                'thickness': thickness,
                'point_radius': point_radius,
                'fourcc': 'mp4v',
                'conf_min_pose': conf_min_pose,
                'conf_min_hand': conf_min_hand,
                'conf_min_face': conf_min_face,
            },
            'timing': {'seconds': round(dt, 3)},
        }

        meta_path = out_video_path.with_suffix('.json')
        meta_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        out['metadata'] = str(meta_path)

        return out


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
    spec : dict | str | pathlib.Path
        Node configuration. The node resolves configuration using
        ``resolve_spec(self.spec)`` and reads parameters at the top level.

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

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

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
        long_side = spec.get('long_side', 384)
        preserve_bg = spec.get('preserve_bg', False)

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
        out_dir.mkdir(parents=True, exist_ok=True)

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

        # compute output size based on first frame (after optional resize)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError('Empty video or failed first frame read')

        if long_side is not None:
            frame0 = resize_long_side(frame, long_side)
        else:
            frame0 = frame

        out_h, out_w = frame0.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Unable to open VideoWriter: {out_video_path}')

        extractor = CannyExtractor()

        # rewind to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # stats
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
                long_side=long_side,
            )

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


_model_id: str = "Intel/dpt-hybrid-midas"
_device: str = "cuda"
_autocast: bool = True


@dataclass
class VideoDepthMap(NodeRef):
    """
    Video depth preprocessing node producing a single packed control video.

    See DepthMap for the single-image analogue. This node processes an input video
    and writes a packed depth control video (mp4) plus a JSON sidecar.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        input = input or {}
        upstream = input.get("default")
        if upstream is None:
            raise ValueError(
                "VideoDepthMap requires an input video wired to the default input.")

        in_path = upstream.get("video") or upstream.get("path")
        if not in_path:
            raise ValueError("Upstream output does not contain a video path.")

        spec = resolve_spec(self.spec)

        preserve_bg = bool(
            spec.get("preserve_bg", False)
        )  # usually False for depth
        max_frames = int(spec.get("max_frames", -1))

        # output resize (control resolution)
        long_side = spec.get("long_side", 384)
        # inference resize (can be higher than output)
        long_side_infer = spec.get("long_side_infer", None)

        normalize = spec.get("normalize", "per_frame")  # per_frame|global
        invert = bool(spec.get("invert_depth", False))
        clip_p_low = float(spec.get("clip_p_low", 2.0))
        clip_p_high = float(spec.get("clip_p_high", 98.0))

        fps_override = spec.get("fps", None)

        model = spec.get('model', {
            "id": _model_id,
            "device": _device,
            "autocast": _autocast,
        })

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {in_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        in_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fps_out = float(fps_override) if fps_override else (
            fps_in if fps_in > 0 else 24.0)

        # build extractor (model cached internally via ModelCache)
        extractor = DepthVideoExtractor(
            model_id=model['id'],
            device=model['device'],
            autocast=model['autocast'],
        )

        # probe first frame to define output size
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Empty video or failed first frame read.")

        # compute output frame (depth) size
        depth0 = extractor.render_frame(
            frame0,
            long_side_infer=long_side_infer,
            normalize="per_frame",
            invert=invert,
            clip_p_low=clip_p_low,
            clip_p_high=clip_p_high,
        )

        # If you want preserve_bg, you can overlay depth on original. Typically false.
        if preserve_bg:
            # simple overlay: use depth as luminance mask (debug)
            out0 = cv2.addWeighted(frame0, 0.6, resize_long_side(depth0, max(
                frame0.shape[:2])) if long_side_infer else depth0, 0.4, 0.0)
        else:
            out0 = depth0

        if long_side is not None:
            out0 = resize_long_side(out0, int(long_side))

        out_h, out_w = out0.shape[:2]

        out_video_path = make_node_output_path(
            out_dir=out_dir,
            node_id=self.id,
            ext='mp4'
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open VideoWriter: {out_video_path}")

        # rewind
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # optional global normalization pass (two-pass)
        # NOTE: global requires computing min/max over frames. Keeping it simple:
        global_minmax = None
        if normalize == "global":
            # Pass 1: compute a global robust range from per-frame quantiles on raw depth.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ql = float(clip_p_low) / 100.0
            qh = float(clip_p_high) / 100.0

            global_lo = None
            global_hi = None
            idx = 0

            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                idx += 1
                if max_frames > 0 and idx > max_frames:
                    break

                d = extractor.predict_depth_tensor(fr, long_side_infer=long_side_infer)
                lo = torch.quantile(d, ql).item()
                hi = torch.quantile(d, qh).item()

                global_lo = lo if global_lo is None else min(global_lo, lo)
                global_hi = hi if global_hi is None else max(global_hi, hi)

            if global_lo is None or global_hi is None or global_hi <= global_lo:
                # Safety fallback
                normalize = "per_frame"
                global_minmax = None
            else:
                global_minmax = (float(global_lo), float(global_hi))

            # Rewind for Pass 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # main pass
        frames_total = 0
        frames_written = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1
            if max_frames > 0 and frames_total > max_frames:
                break

            depth_bgr = extractor.render_frame(
                frame,
                long_side_infer=long_side_infer,
                normalize=normalize,
                invert=invert,
                clip_p_low=clip_p_low,
                clip_p_high=clip_p_high,
                global_minmax=global_minmax,
            )

            out = depth_bgr if not preserve_bg else cv2.addWeighted(
                frame, 0.6, resize_long_side(depth_bgr, max(frame.shape[:2])), 0.4, 0.0)

            if long_side is not None:
                out = resize_long_side(out, int(long_side))

            writer.write(out)
            frames_written += 1

        cap.release()
        writer.release()

        dt_s = time.perf_counter() - t0

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
                "preserve_bg": preserve_bg,
                "max_frames": max_frames,
                "long_side": long_side,
                "long_side_infer": long_side_infer,
                "normalize": normalize,
                "invert_depth": invert,
                "clip_p_low": clip_p_low,
                "clip_p_high": clip_p_high,
                "fps_out": fps_out,
                "fourcc": "mp4v",
            },
            "stats": {
                "frames_total": frames_total,
                "frames_written": frames_written,
            },
            'model': model,
            "timing": {"seconds": round(dt_s, 3)}
        }

        meta_path = out_video_path.with_suffix('.json')
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        return meta
