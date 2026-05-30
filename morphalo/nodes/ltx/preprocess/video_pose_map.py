import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.ltx.preprocess.utils.drawing_utils import \
    compute_long_side_resize
from morphalo.nodes.ltx.preprocess.utils.skeleton_extrtactor import (
    ModelPaths, SkeletonExtractor)


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

    spec: SpecInput = field(default_factory=dict)

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
        max_frames = spec.get('max_frames', -1)

        # --- sizes (ControlNet-style naming, with backward compat) ---
        # detect_resolution: resize BEFORE inference (speed/quality tradeoff)
        detect_resolution = spec.get('detect_resolution', None)
        # image_resolution: resize AFTER rendering (final control video size)
        image_resolution = spec.get(
            'image_resolution',
            spec.get('long_side', 512)
        )

        # Extractor params
        conf_min_pose = spec.get('conf_min_pose', 0.3)
        conf_min_hand = spec.get('conf_min_hand', 0.3)
        conf_min_face = spec.get('conf_min_face', 0.3)
        max_poses = spec.get('max_poses', 4)
        max_faces = spec.get('max_faces', 4)
        max_hands = spec.get('max_hands', 8)

        models = ModelPaths(**spec.get('models', {}))

        out_dir = Path(output_dir)

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

        # Compute output size:
        # 1) optional pre-infer resize (detect_resolution)
        tmp_h, tmp_w = h, w
        if detect_resolution is not None:
            tmp_h, tmp_w, _ = compute_long_side_resize(
                tmp_h, tmp_w, int(detect_resolution))
        # 2) optional post-render resize (image_resolution)
        out_h, out_w = tmp_h, tmp_w
        if image_resolution is not None:
            out_h, out_w, _ = compute_long_side_resize(
                out_h, out_w, int(image_resolution))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, float(fps), (int(out_w), int(out_h)))

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

                # VIDEO mode requires monotonically increasing timestamps.
                # Use float math to support non-integer fps.
                timestamp_ms = int(
                    round((frame_idx * 1000.0) / float(fps))
                ) if fps and fps > 0 else frame_idx * 33

                # Optional pre-infer resize for speed/quality tradeoff.
                # (Must happen before creating mp.Image inside SkeletonExtractor.)
                if detect_resolution is not None:
                    nh, nw, _ = compute_long_side_resize(
                        frame.shape[0],
                        frame.shape[1],
                        int(detect_resolution)
                    )
                    if (nh, nw) != frame.shape[:2]:
                        frame = cv2.resize(
                            frame, (int(nw), int(nh)), interpolation=cv2.INTER_AREA)

                out = extractor.render_frame(
                    frame_bgr=frame,
                    preserve_bg=preserve_bg,
                    timestamp_ms=timestamp_ms,
                    long_side=image_resolution
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
            'node': self.op,
            'id': self.id,
            'input_video': in_path,
            'video': str(out_video_path),
            'model': {
                'pose_task':  models.pose_task,
                'hand_task':  models.hand_task,
                'face_task':  models.face_task
            },
            'params': {
                'detect_resolution': detect_resolution,
                'image_resolution': image_resolution,
                'preserve_bg': preserve_bg,
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
