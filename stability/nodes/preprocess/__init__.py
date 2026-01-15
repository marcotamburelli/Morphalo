import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2

from stability.dag import NodeRef
from stability.nodes import make_node_output_path, resolve_spec
from stability.nodes.preprocess.utils import *


def _read_bgr(path: str) -> cv2.typing.MatLike:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Unable to read image: {path}')
    return img


@dataclass
class PoseMap(NodeRef):
    """
    Single-image pose preprocessing node producing a packed control image.

    This node reads an input image (via the default DAG connection) and renders a
    packed pose map suitable for pose-conditioned adapters (e.g. pose ControlNet,
    pose-guided pipelines, or IC-LoRA pose inputs). The pose map is produced by
    running MediaPipe Tasks landmark detection through ``SkeletonExtractor`` in
    IMAGE mode and rendering pose, hands, and face landmarks into a single control
    image.

    The output control image is a standard BGR image (OpenCV convention) containing
    rendered landmarks either on a black background (default) or on the original
    image (debug).

    Processing overview
    -------------------
    1) Load the input image in BGR format (OpenCV).
    2) Run landmark inference (pose/hands/face) using MediaPipe Tasks (IMAGE mode).
    3) Render all detected landmarks into a single packed control frame.
    4) Optionally downscale the rendered output via ``long_side``.
    5) Write the output control image to disk (PNG) plus a JSON sidecar.

    Multi-person output semantics
    -----------------------------
    The extractor may return multiple poses/faces/hands for a single image depending
    on ``max_poses``, ``max_faces``, and ``max_hands``. The output control image
    renders all detections using fixed colors. No explicit per-person tracking or
    identity association is performed (not applicable for single-image mode).

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
            Filesystem path to the generated pose control image (PNG).
        - ``model`` : dict
            Resolved MediaPipe task paths (pose/hand/face).
        - ``params`` : dict
            Resolved preprocessing and extractor parameters.
        - ``metadata`` : str
            Filesystem path to the JSON sidecar.
        - ``timing`` : dict
            Elapsed time for the run.

    Configuration keys (spec)
    -------------------------
    Rendering / resizing
    ~~~~~~~~~~~~~~~~~~~
    preserve_bg : bool, default False
        If True, render landmarks on top of the original image (debug).
        If False, render on a black background (recommended for control inputs).
    thickness : int, default 2
        Line thickness used when rendering landmark edges.
    point_radius : int, default 2
        Point radius used when rendering landmark points (where applicable).
    long_side : int | None, default None
        If set, downscale the output control image so that its long side equals
        ``long_side`` (aspect ratio preserved). Resize is applied after inference
        and rendering.

    Extractor configuration
    ~~~~~~~~~~~~~~~~~~~~~~
    conf_min_pose : float, default 0.3
        Minimum confidence threshold for pose landmark detections.
    conf_min_hand : float, default 0.3
        Minimum confidence threshold for hand landmark detections.
    conf_min_face : float, default 0.3
        Minimum confidence threshold for face landmark detections.
    max_poses : int, default 4
        Maximum number of pose instances to detect and render.
    max_faces : int, default 4
        Maximum number of face instances to detect and render.
    max_hands : int, default 8
        Maximum number of hands to detect and render.

    Model configuration
    ~~~~~~~~~~~~~~~~~~
    models : dict, required
        Dictionary used to construct ``ModelPaths`` via ``ModelPaths(**spec['models'])``.
        Expected keys:

        - ``pose_task`` : str
            Path to the MediaPipe Pose Landmarker .task file.
        - ``hand_task`` : str
            Path to the MediaPipe Hand Landmarker .task file.
        - ``face_task`` : str
            Path to the MediaPipe Face Landmarker .task file.

    Notes
    -----
    - The output control image is written as PNG to avoid lossy compression artifacts
      that could degrade thin landmark lines and points.
    - This node operates in MediaPipe IMAGE mode, so it does not use temporal smoothing
      or tracking. For temporal stability on videos, use the corresponding video node
      (e.g. ``VideoPoseMap``) in VIDEO mode.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        input = input or {}

        up = input.get('default')
        if up is None:
            raise ValueError(
                'PoseMap requires an input image wired into the default input (src >> pose_map).')

        in_path = up.get('image') or up.get('path')
        if not in_path:
            raise ValueError("Upstream output must contain 'image' (path).")

        spec = resolve_spec(self.spec)

        preserve_bg = bool(spec.get('preserve_bg', False))
        thickness = int(spec.get('thickness', 2))
        point_radius = int(spec.get('point_radius', 2))
        long_side = spec.get('long_side', None)

        conf_min_pose = float(spec.get('conf_min_pose', 0.3))
        conf_min_hand = float(spec.get('conf_min_hand', 0.3))
        conf_min_face = float(spec.get('conf_min_face', 0.3))
        max_poses = int(spec.get('max_poses', 4))
        max_faces = int(spec.get('max_faces', 4))
        max_hands = int(spec.get('max_hands', 8))

        models = ModelPaths(**spec.get('models', {}))

        frame = _read_bgr(in_path)

        extractor = SkeletonExtractor(
            models=models,
            conf_min_pose=conf_min_pose,
            conf_min_hand=conf_min_hand,
            conf_min_face=conf_min_face,
            max_poses=max_poses,
            max_faces=max_faces,
            max_hands=max_hands,
            running_mode='IMAGE',
        )

        try:
            out_img = extractor.render_frame(
                frame_bgr=frame,
                thickness=thickness,
                point_radius=point_radius,
                preserve_bg=preserve_bg,
                timestamp_ms=0,
                long_side=long_side,
            )
        finally:
            extractor.close()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = make_node_output_path(
            out_dir=out_dir, node_id=self.id, ext='png')
        cv2.imwrite(str(out_path), out_img)

        dt_s = time.perf_counter() - t0

        meta = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(in_path),
            'image': str(out_path),
            'model': {
                'pose_task': models.pose_task,
                'hand_task': models.hand_task,
                'face_task': models.face_task,
            },
            'params': {
                'preserve_bg': preserve_bg,
                'thickness': thickness,
                'point_radius': point_radius,
                'long_side': long_side,
                'conf_min_pose': conf_min_pose,
                'conf_min_hand': conf_min_hand,
                'conf_min_face': conf_min_face,
                'max_poses': max_poses,
                'max_faces': max_faces,
                'max_hands': max_hands,
            },
            'timing': {'seconds': round(dt_s, 3)},
        }

        meta_path = out_path.with_suffix('.json')
        meta_path.write_text(json.dumps(
            meta, indent=2, ensure_ascii=False), encoding='utf-8')
        meta['metadata'] = str(meta_path)
        return meta


@dataclass
class CannyMap(NodeRef):
    """
    Single-image Canny preprocessing node producing a packed control image.

    This node reads an input image (via the default DAG connection) and generates a
    single packed control image suitable for edge-conditioned adapters (e.g. ControlNet
    Canny). The produced control image is a standard BGR image (OpenCV convention)
    containing the Canny edge map rendered either on a black background (default) or
    over the original image (debug).

    Processing overview
    -------------------
    1) Load the input image in BGR format (OpenCV).
    2) Convert to grayscale and optionally apply Gaussian blur.
    3) Run Canny edge detection with the configured thresholds.
    4) Optionally post-process edges (dilation and/or inversion).
    5) Render the final edge map on black background or on the original image.
    6) Optionally downscale the rendered output via ``long_side``.
    7) Write the output control image to disk (PNG) plus a JSON sidecar.

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
        - ``metadata`` : str
            Filesystem path to the JSON sidecar.

        The JSON sidecar includes all resolved parameters and timing information.

    Configuration keys (spec)
    -------------------------
    Rendering / resizing
    ~~~~~~~~~~~~~~~~~~~
    preserve_bg : bool, default False
        If True, render edges on top of the original image (debug).
        If False, render on a black background (recommended for control inputs).
    long_side : int | None, default None
        If set, downscale the output control image so that its long side equals
        ``long_side`` (aspect ratio preserved). Resize is applied after edge rendering.

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

    Notes
    -----
    - The output control image is written as PNG to avoid compression artifacts that
      could degrade thin edges.
    - If the input image is noisy or heavily compressed, Canny outputs may contain
      spurious edges. Increasing pre-blur, adjusting thresholds, or applying dilation
      can improve robustness depending on the target adapter.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        input = input or {}

        up = input.get('default')
        if up is None:
            raise ValueError(
                'CannyMap requires an input image wired into the default input (src >> canny_map).'
            )

        in_path = up.get('image') or up.get('path')
        if not in_path:
            raise ValueError("Upstream output must contain 'image' (path).")

        spec = resolve_spec(self.spec)

        preserve_bg = bool(spec.get('preserve_bg', False))
        long_side = spec.get('long_side', None)

        low_threshold = int(spec.get('low_threshold', 80))
        high_threshold = int(spec.get('high_threshold', 160))
        blur_ksize = int(spec.get('blur_ksize', 5))
        blur_sigma = float(spec.get('blur_sigma', 0.0))
        aperture_size = int(spec.get('aperture_size', 3))
        l2_gradient = bool(spec.get('l2_gradient', False))
        dilate = int(spec.get('dilate', 0))
        dilate_iter = int(spec.get('dilate_iter', 1))
        invert = bool(spec.get('invert', False))

        frame = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f'Unable to read image: {in_path}')

        extractor = CannyExtractor()
        out_img = extractor.render_frame(
            frame_bgr=frame,
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

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = make_node_output_path(
            out_dir=out_dir, node_id=self.id, ext='png')
        cv2.imwrite(str(out_path), out_img)

        dt_s = time.perf_counter() - t0

        meta = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(in_path),
            'image': str(out_path),
            'params': {
                'preserve_bg': preserve_bg,
                'long_side': long_side,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                'blur_ksize': blur_ksize,
                'blur_sigma': blur_sigma,
                'aperture_size': aperture_size,
                'l2_gradient': l2_gradient,
                'dilate': dilate,
                'dilate_iter': dilate_iter,
                'invert': invert,
            },
            'timing': {'seconds': round(dt_s, 3)},
        }

        meta_path = out_path.with_suffix('.json')
        meta_path.write_text(json.dumps(
            meta, indent=2, ensure_ascii=False), encoding='utf-8')
        meta['metadata'] = str(meta_path)
        return meta


@dataclass
class DepthMap(NodeRef):
    """
    Single-image depth preprocessing node producing a packed control image.

    This node reads an input image (via the default DAG connection) and produces a
    packed depth control image suitable for depth-conditioned adapters (e.g. ControlNet
    Depth). Depth is estimated using a DPT/MiDaS-family model via
    ``DepthVideoExtractor`` (despite the name, the extractor operates per-frame).

    The output is a 3-channel BGR visualization (OpenCV convention) representing
    normalized depth (typically as grayscale replicated to 3 channels), optionally
    inverted, and written as a PNG plus a JSON sidecar.

    Processing overview
    -------------------
    1) Load the input image in BGR format (OpenCV).
    2) Run depth prediction (optionally at a separate inference resolution).
    3) Normalize depth to an 8-bit visualization using robust clipping percentiles.
    4) Optionally invert the depth visualization (near/far swap).
    5) Optionally downscale the output visualization via ``long_side``.
    6) Write the output control image to disk (PNG) plus a JSON sidecar.

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
            Filesystem path to the generated depth control image (PNG).
        - ``metadata`` : str
            Filesystem path to the JSON sidecar.

        The JSON sidecar includes resolved parameters, model configuration, and timing.

    Configuration keys (spec)
    -------------------------
    Output resizing
    ~~~~~~~~~~~~~~
    long_side : int | None, default 384
        If set, downscale the final depth visualization so that its long side equals
        ``long_side`` (aspect ratio preserved). Resize is applied after normalization.

    Inference resizing
    ~~~~~~~~~~~~~~~~~
    long_side_infer : int | None, default None
        Optional long-side value used for the *inference input* to the depth model.
        This can be higher than ``long_side`` to obtain a cleaner depth estimate, then
        downscale for control usage. If None, inference runs at the input resolution.

    Normalization / visualization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    normalize : str, default "per_frame"
        Normalization strategy. For single-image use, ``"per_frame"`` is the effective
        mode (global normalization requires multiple frames and is not applicable here).
    invert_depth : bool, default False
        If True, invert the depth visualization (swap near/far).
    clip_p_low : float, default 2.0
        Lower percentile for robust clipping before normalization.
    clip_p_high : float, default 98.0
        Upper percentile for robust clipping before normalization.

    Model configuration
    ~~~~~~~~~~~~~~~~~~
    model : dict, optional
        Model configuration used to initialize the underlying extractor.
        Typical keys:

        - ``id`` : str
            HuggingFace model id (default ``"Intel/dpt-hybrid-midas"``).
        - ``device`` : str
            Execution device string (e.g. ``"cuda"`` or ``"cpu"``).
        - ``autocast`` : bool
            Whether to use autocast for inference on compatible devices.

    Notes
    -----
    - The output is written as PNG to avoid compression artifacts that could degrade
      depth edges and gradients.
    - Robust clipping (``clip_p_low``/``clip_p_high``) helps stabilize visualization
      across images with different depth ranges, but extreme settings may wash out
      near/far contrast.
    - For single-image processing, global normalization is not meaningful; if requested,
      implementations should either ignore it or raise an error. This node effectively
      uses per-image normalization.
    """

    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        input = input or {}

        up = input.get('default')
        if up is None:
            raise ValueError(
                'DepthMap requires an input image wired into the default input (src >> depth_map).')

        in_path = up.get('image') or up.get('path')
        if not in_path:
            raise ValueError("Upstream output must contain 'image' (path).")

        spec = resolve_spec(self.spec)

        long_side = spec.get('long_side', 384)
        long_side_infer = spec.get('long_side_infer', None)

        # keep same key as video
        normalize = spec.get('normalize', 'per_frame')
        invert = bool(spec.get('invert_depth', False))
        clip_p_low = float(spec.get('clip_p_low', 2.0))
        clip_p_high = float(spec.get('clip_p_high', 98.0))

        model = spec.get('model', {
            'id': 'Intel/dpt-hybrid-midas',
            'device': 'cuda',
            'autocast': True,
        })

        frame = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f'Unable to read image: {in_path}')

        extractor = DepthVideoExtractor(
            model_id=model['id'],
            device=model['device'],
            autocast=bool(model.get('autocast', True)),
        )

        depth_bgr = extractor.render_frame(
            frame_bgr=frame,
            long_side_infer=long_side_infer,
            # global makes no sense for single image
            normalize='per_frame' if normalize == 'per_frame' else 'per_frame',
            invert=invert,
            clip_p_low=clip_p_low,
            clip_p_high=clip_p_high,
            global_minmax=None,
        )

        if long_side is not None:
            # depth_bgr is already BGR uint8; reuse your resize helper if you want,
            # but cv2.resize is fine here:
            from stability.nodes.preprocess.utils import \
                resize_long_side as \
                _resize_long_side  # :contentReference[oaicite:6]{index=6}
            depth_bgr = _resize_long_side(depth_bgr, int(long_side))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = make_node_output_path(
            out_dir=out_dir, node_id=self.id, ext='png')
        cv2.imwrite(str(out_path), depth_bgr)

        dt_s = time.perf_counter() - t0

        meta = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(in_path),
            'image': str(out_path),
            'model': model,
            'params': {
                'long_side': long_side,
                'long_side_infer': long_side_infer,
                'normalize': 'per_frame',
                'invert_depth': invert,
                'clip_p_low': clip_p_low,
                'clip_p_high': clip_p_high,
            },
            'timing': {'seconds': round(dt_s, 3)},
        }

        meta_path = out_path.with_suffix('.json')
        meta_path.write_text(json.dumps(
            meta, indent=2, ensure_ascii=False), encoding='utf-8')
        meta['metadata'] = str(meta_path)
        return meta
