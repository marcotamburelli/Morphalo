"""Render a posed MHR mannequin with optional source body proportions."""

from dataclasses import dataclass, field
from functools import partial
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from morphalo.cache.models import (
    get_instanthmr_model,
    get_mediapipe_pose_landmarker,
    get_yolo,
)
from morphalo.core.cuda_mem import CudaPostRunMixin
from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import load_init_image, write_json_sidecar
from morphalo.nodes.img import resolve_long_side_size
from morphalo.nodes.preprocess.utils.body_collision import refine_terminal_collisions
from morphalo.nodes.preprocess.utils.body_pose_debug import (
    draw_body_pose_overlay,
    save_pose_crop_debug,
)
from morphalo.nodes.preprocess.utils.body_pose_fit import (
    BodyPoseCrop,
    detect_body_landmarks,
    project_body_joints,
    refine_body_pose,
)
from morphalo.nodes.preprocess.utils.body_render import fit_body_canvas, render_body
from morphalo.nodes.preprocess.utils.body_shape import (
    adjust_body_volumes,
    rest_body_axes,
    rest_neck_axis,
)
from morphalo.nodes.preprocess.utils.color_ops import parse_color
from morphalo.nodes.sdxl_resolve import (
    resolve_image_paths,
    resolve_single_image_path,
)
from morphalo.nodes.vision.human import (
    person_bboxes_xyxy,
    select_person_bbox_xyxy,
)
from third_party.instanthmr.parameters import BodyProportions


CHECKPOINT = {
    'repo_id': 'jtydhr88/ComfyUI-InstantHMR',
    'filename': 'instanthmr_medium_commercial.safetensors',
    'revision': '964effb58c3358ca8a0e35da386f49de1480ac9e',
}
BODY_SHAPE_PRESETS = {
    'estimated': (None, None, None),
    'masculine': (0.0, 0.0, 1.0),
    'feminine': (0.65, None, 0.90),
}
GENDER_ALIASES = {'m': 'masculine', 'f': 'feminine'}


@dataclass(frozen=True)
class Config:
    """Resolved model assets, inference settings, and render parameters."""

    checkpoint: Path | dict[str, str]
    rig_path: Path
    pose_landmarker_task: Path | None
    yolo_model: str
    device: str
    body_color: tuple[int, int, int]
    background_color: tuple[int, int, int] | None
    chest_volume: float | None
    hip_volume: float | None
    waist_size: float | None
    body_volume: float
    neck_thickness: float
    long_side: int | None
    fit_canvas: bool
    split_pose_detection: bool
    fit_head: bool
    save_debug: bool


def _shape_value(
    value: object,
    name: str,
    node_id: str,
    *,
    bounds: tuple[float, float | None] | None = None,
) -> float:
    """Validate a finite body-shape value and any inclusive lower and upper bounds."""
    try:
        valid = isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
    except OverflowError:
        valid = False

    if not valid:
        raise ValueError(f"'{node_id}': params.{name} must be a finite number")

    number = float(value)

    if bounds is not None:
        minimum, maximum = bounds

        if maximum is None and number < minimum:
            raise ValueError(f"'{node_id}': params.{name} must be at least {minimum}")

        if maximum is not None and not minimum <= number <= maximum:
            raise ValueError(f"'{node_id}': params.{name} must be between {minimum} and {maximum}")

    return number


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Resolve model, render, and body-shape settings.

    Parameters
    ----------
    spec : dict
        Resolved specification with ``model``, optional ``params``, and optional
        ``debug`` blocks.
    node_id : str
        Node identifier included in validation errors.

    Returns
    -------
    Config
        Configuration with parsed colors, effective volume adjustments and
        absolute local asset paths. A checkpoint mapping describes a file
        on Hugging Face.

    Raises
    ------
    ValueError
        If a configuration block, checkpoint, model path, color, or shape
        setting is invalid.
    """
    model = spec.get('model', {})
    params = spec.get('params', {})
    debug = spec.get('debug', {})

    if not isinstance(model, dict) or not isinstance(params, dict):
        raise ValueError(f"'{node_id}': model and params must be objects")

    if not isinstance(debug, dict):
        raise ValueError(f"'{node_id}': debug must be an object")

    rig_path = model.get('mhr_model_pt')

    if not isinstance(rig_path, (str, Path)) or rig_path == '':
        raise ValueError(
            f"'{node_id}': model.mhr_model_pt required (local TorchScript file path)"
        )

    pose_task = model.get('pose_landmarker_task')

    if pose_task is not None:
        if not isinstance(pose_task, (str, Path)) or not str(pose_task).strip():
            raise ValueError(
                f"'{node_id}': model.pose_landmarker_task must be a local .task path"
            )

        pose_task = Path(pose_task).expanduser().resolve()

    checkpoint = model.get('checkpoint', CHECKPOINT)

    if isinstance(checkpoint, dict):
        unexpected = checkpoint.keys() - {'repo_id', 'filename', 'revision'}

        if unexpected:
            raise ValueError(
                f"'{node_id}': unknown model.checkpoint fields: "
                f'{sorted(unexpected, key=str)}'
            )

        for key in ('repo_id', 'filename'):
            value = checkpoint.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"'{node_id}': model.checkpoint.{key} must be a non-empty string"
                )

        if 'revision' in checkpoint:
            revision = checkpoint['revision']
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError(
                    f"'{node_id}': model.checkpoint.revision must be a non-empty string"
                )

        checkpoint = dict(checkpoint)

    elif isinstance(checkpoint, (str, Path)) and checkpoint != '':
        checkpoint = Path(checkpoint).expanduser().resolve()

    else:
        raise ValueError(
            f"'{node_id}': model.checkpoint must be a file path or a Hugging Face mapping"
        )

    body_color = parse_color(
        params.get('body_color', '#b8b8b8'),
        node_id=f'{node_id}: params.body_color',
    )
    background_color = params.get('background_color')

    if background_color is not None:
        background_color = parse_color(
            background_color,
            node_id=f'{node_id}: params.background_color',
        )

    gender = params.get('gender', 'estimated')

    if isinstance(gender, str):
        gender = gender.strip().lower()
        gender = GENDER_ALIASES.get(gender, gender)

    if not isinstance(gender, str) or gender not in BODY_SHAPE_PRESETS:
        raise ValueError(
            f"'{node_id}': params.gender must be estimated, masculine (m), or feminine (f)"
        )

    chest_preset, hip_preset, waist_preset = BODY_SHAPE_PRESETS[gender]
    chest_volume = params.get('chest_volume', chest_preset)
    hip_volume = params.get('hip_volume', hip_preset)
    waist_size = params.get('waist_size', waist_preset)

    if chest_volume is not None:
        chest_volume = _shape_value(chest_volume, 'chest_volume', node_id, bounds=(0.0, None))

    if hip_volume is not None:
        hip_volume = _shape_value(hip_volume, 'hip_volume', node_id, bounds=(0.0, None))

    if waist_size is not None:
        waist_size = _shape_value(waist_size, 'waist_size', node_id, bounds=(0.8, 1.2))

    body_volume = _shape_value(
        params.get('body_volume', 1.0), 'body_volume', node_id, bounds=(0.75, 1.25),
    )
    neck_thickness = _shape_value(
        params.get('neck_thickness', 1.0), 'neck_thickness', node_id, bounds=(0.75, 1.25),
    )

    long_side = params.get('long_side')

    if long_side is not None:
        if isinstance(long_side, bool) or not isinstance(long_side, Integral) or long_side <= 0:
            raise ValueError(f"'{node_id}': params.long_side must be a positive integer or None")

        long_side = int(long_side)

    fit_canvas = params.get('fit_canvas', True)

    if not isinstance(fit_canvas, bool):
        raise ValueError(f"'{node_id}': params.fit_canvas must be a boolean")

    split_pose_detection = params.get('split_pose_detection', False)

    if not isinstance(split_pose_detection, bool):
        raise ValueError(f"'{node_id}': params.split_pose_detection must be a boolean")

    fit_head = params.get('fit_head', False)

    if not isinstance(fit_head, bool):
        raise ValueError(f"'{node_id}': params.fit_head must be a boolean")

    return Config(
        checkpoint=checkpoint,
        rig_path=Path(rig_path).expanduser().resolve(),
        pose_landmarker_task=pose_task,
        yolo_model=str(model.get('yolo_model', 'yolov8n.pt')),
        device=str(model.get('device', 'cuda')),
        body_color=body_color,
        background_color=background_color,
        chest_volume=chest_volume,
        hip_volume=hip_volume,
        waist_size=waist_size,
        body_volume=body_volume,
        neck_thickness=neck_thickness,
        long_side=long_side,
        fit_canvas=fit_canvas,
        split_pose_detection=split_pose_detection,
        fit_head=fit_head,
        save_debug=bool(debug.get('save_debug', False)),
    )


def _person_bbox(
    detector,
    rgb: np.ndarray,
    device: str,
    path: Path,
) -> tuple[int, int, int, int]:
    """
    Select the largest detected person box for InstantHMR.

    Parameters
    ----------
    detector : ultralytics.YOLO
        Cached detector with a person class.
    rgb : numpy.ndarray
        Full-frame RGB image with shape ``(H, W, 3)``.
    device : str
        Inference device used by YOLO.
    path : pathlib.Path
        Input path included in detection errors.

    Returns
    -------
    tuple[int, int, int, int]
        Person bounding box with the largest area, in full-frame
        ``(x1, y1, x2, y2)`` pixels.

    Raises
    ------
    ValueError
        If there are no usable person detections.
    """
    result = detector.predict(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        classes=[0],
        device=device,
        verbose=False,
    )[0]

    try:
        boxes = person_bboxes_xyxy(result, str(path))
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    return select_person_bbox_xyxy(boxes, pose_xy=None)


@dataclass
class BodyGeometryMap(CudaPostRunMixin, NodeRef):
    """
    Render a full-frame MHR mannequin for compositing or image conditioning.

    The main image determines joint angles, global orientation, and camera
    placement. It also supplies body proportions when used on its own.
    Connect :meth:`source` to render another person's proportions in the
    main image's pose. Changes in limb length or torso width can move
    projected joints and change the silhouette.

    Processing flow
    ---------------
    1. Select the largest detected person box in the main image. Estimate
       the pose, camera, and body proportions with InstantHMR.
    2. Estimate proportions for each usable reference image and average the
       shape, skeletal lengths and widths, and bone-scale coefficients.
    3. If configured, fit body segment directions against MediaPipe in the
       image plane and in relative depth. Use the hip midpoint to align the
       pelvis in the target frame.
    4. Decode the MHR mesh with the chosen proportions and the target pose.
       Project it into the target frame and use depth ordering to resolve
       surfaces that overlap.
    5. Save one shaded RGBA image and its DAG metadata.

    Parameters
    ----------
    name : str, optional
        Node identifier within the DAG. The enclosing DAG assigns it when
        omitted.
    spec : dict or str or pathlib.Path or sequence of these
        Configuration resolved by ``resolve_spec``. A sequence can combine
        dictionaries and HOCON paths, merged from left to right.

        ``model`` : dict
            ``checkpoint`` : str or pathlib.Path or dict, optional
                Local safetensors path or a Hugging Face mapping with
                ``repo_id``, ``filename``, and optional ``revision``.
                Omitting ``revision`` in a custom mapping uses the repository's
                default branch. The built-in mapping pins
                ``instanthmr_medium_commercial.safetensors`` from
                ``jtydhr88/ComfyUI-InstantHMR`` to the verified commit.
                Downloads are reused from the Hugging Face cache.
            ``mhr_model_pt`` : str or pathlib.Path
                Required path to the local MHR v1.0.1 LOD 1 TorchScript rig.
                The installer places it at
                ``~/models/mhr/mhr_model.pt``.
            ``pose_landmarker_task`` : str or pathlib.Path, optional
                Local MediaPipe Pose Landmarker task. A path enables 3D pose
                refinement using shoulders, elbows, wrists, hands, hips, knees,
                ankles, and forefeet. The fitting accepts usable observations
                outside the image and uses no visibility or presence weighting.
                Omit the path to render the direct InstantHMR pose.
                ``./fetch_media_pipe.sh`` installs the task at
                ``~/models/mediapipe/pose_landmarker_heavy.task``.
            ``yolo_model`` : str, optional
                Person detector checkpoint name. Defaults to ``'yolov8n.pt'``.
                The loader checks ``~/models/yolo`` first. Ultralytics
                downloads missing official weights into that directory.
            ``device`` : str, optional
                Device for YOLO and InstantHMR, defaulting to ``'cuda'``.
                TorchScript mesh decoding runs on CPU.

        ``params`` : dict, optional
            ``split_pose_detection`` : bool, optional
                With a MediaPipe pose task configured, redetect the upper
                and lower body in overlapping crops before fitting. Use a
                horizontal cut below 45 degrees from the image vertical and
                a vertical cut at 45 degrees or more.
                Missing regional landmarks fall back to the full-person
                detection. Defaults to ``False``.
            ``fit_head`` : bool, optional
                Refine the neck and head after body fitting using MediaPipe
                nose, eye, and ear landmarks. Defaults to ``False``.
            ``long_side`` : int or None, optional
                Positive length in pixels of the longer render canvas side
                before padding. The shorter side follows the EXIF-oriented
                target's aspect ratio, rounded to at least one pixel.
                Defaults to ``None`` to use the target dimensions. Detection
                and fitting use the original target resolution.
            ``fit_canvas`` : bool, optional
                Expand the render canvas on sides where the mannequin
                overflows, adding a margin of 3% of its projected height.
                Runs after ``long_side`` is applied. Defaults to ``True``.
            ``gender`` : str, optional
                Default surface targets on the source mesh, or on the target
                mesh when no source is connected. ``'estimated'`` (default)
                leaves all three regions without a local target.
                ``'masculine'`` (``'m'``) uses neutral chest and hips and a
                neutral waist size of 1.0. ``'feminine'`` (``'f'``) uses a
                chest target of 0.65 and a waist size of 0.90, leaving the
                hips without a local target. Explicit values replace the
                preset for their own region. Pass ``None`` to disable a
                region's local target even when the preset defines one.
            ``body_volume`` : float, optional
                Radial thickness multiplier for the torso and limbs, from
                0.75 to 1.25. Defaults to 1.0. For example, 1.10 increases
                thickness around the bones by up to 10%. The effect tapers
                into the head, hands and feet. Applied before local targets.
                The multiplier controls thickness, and the rig's bone
                lengths and joint centers remain fixed.
            ``neck_thickness`` : float, optional
                Radial thickness multiplier for the neck, from 0.75 to
                1.25. Defaults to 1.0 for the estimated neck thickness,
                independently of ``gender``. For example, 0.90 reduces
                thickness by up to 10%, with a smooth taper where the
                neck meets the head and shoulders. The neck length and
                the head and neck pose remain fixed.
            ``chest_volume`` : float or None, optional
                Unitless target relative to the rib-cage profile estimated
                from the shaped sternum and ribs. Accepts finite values
                greater than or equal to zero. Use 0 to bring the chest
                toward that profile and positive values to add forward
                prominence. Start between 0 and 0.5 for modest changes.
                At full influence, values of 0.5, 1, and 2 target
                about 1.85, 3.05, and 3.86 cm from neutral, respectively.
                The target approaches 4 cm via ``tanh``, so values above
                2 add little effect.
            ``hip_volume`` : float or None, optional
                Unitless target relative to the side contour interpolated
                between the upper thigh and pelvis. Accepts finite values
                greater than or equal to zero. Use 0 to bring the hip sides
                toward that contour and positive values to widen it.
                Start between 0 and 0.5 for modest changes. At full influence, values
                of 0.5, 1, and 2 target about 1.62, 2.67, and 3.37 cm per
                side from neutral, respectively. The target approaches
                3.5 cm per side via ``tanh``, so values above 2 add little effect.
            ``waist_size`` : float or None, optional
                Target waist-width multiplier from 0.8 to 1.2, relative
                to the neutral contour between the pelvis and lower ribs.
                1.0 follows that contour and 0.90 targets 90% of its width.
            ``body_color`` : str or sequence of numbers, optional
                Mannequin base color before shading. Defaults to ``'#b8b8b8'``.
            ``background_color`` : str or sequence of numbers or None, optional
                Fill color. Defaults to ``None`` for a transparent background.
                Both colors accept CSS names, hex strings, or RGB triplets.

        ``debug`` : dict, optional
            ``save_debug`` : bool, optional
                Save an overlay of the rendered MHR skeleton and any
                MediaPipe pose observations in a separate color. Also save
                pre-merge overlays for the MediaPipe crops and the original
                input image of each failed crop.
                Defaults to ``False``.

    Inputs
    ------
    default : dict
        One target image path under ``image`` or ``path``.
    source : dict, optional
        One or more reference paths under ``image``, ``images``, or ``path``.
        Reference views should depict the same person. Unreadable images and
        failed estimates are recorded, and at least one view must succeed
        when the attachment is connected.

    Outputs
    -------
    dict
        ``image`` : str
            PNG path. The base canvas uses the EXIF-oriented target size
            or the size requested by ``long_side``, and may expand around
            the projection. Visible mannequin pixels are opaque.
        ``canvas_padding`` : dict[str, int]
            Added render pixels on the left, top, right, and bottom. The
            left and top values locate the base canvas within the PNG.
        ``source_images`` : list[str]
            Reference paths that contributed to the arithmetic mean.
        ``rejected_source_images`` : list[dict]
            Rejected reference paths and their failure reasons.
        ``pose_fit`` : dict, optional
            Status and available diagnostics when a pose task is configured.
            These include the selected MediaPipe world X/Y signs and
            ``world_depth_scale``, the positive Z-to-XY calibration ratio.
            The output also reports the
            ``root_retargeted``, ``spine_retargeted``, ``limbs_retargeted``,
            ``torso_width_refined``, and ``feet_retargeted`` stage flags, and
            errors before and after fitting. Direction errors and torso
            twist angles are in degrees. MediaPipe twist diagnostics use the
            full-person detection.
            Joint reprojection RMS, pelvis offset and projected shoulder and
            hip widths are in pixels. The distance between the torso
            midpoints is in centimetres.
            ``terminal_collisions`` reports hand and foot surface samples
            detected inside the mesh before and after local correction.
            ``head_fit`` reports the optional head fit status and landmark
            reprojection errors in pixels.
        ``debug`` : dict, optional
            Diagnostic directory and ``pose_overlay`` PNG path when
            ``debug.save_debug`` is true. ``mediapipe_crops`` records each
            attempted crop's full-frame bounds and either the MediaPipe-only
            overlay or the detection error. Failed crops also include the
            input image path, and rotated crops include their four corners
            in target pixels.
        ``metadata`` : str
            Path to the JSON sidecar used by the DAG.

    Notes
    -----
    Run ``./fetch_mhr.sh`` to install the local MHR TorchScript rig. The
    default InstantHMR checkpoint permits commercial use under the SAM
    License terms. The publisher lists the Nano, Tiny, Small, and standard
    Medium checkpoints as non-commercial.

    Each input uses the person detection with the largest bounding-box area.
    A target with no usable detections fails. If a different detection has
    the largest box, crop the input around the intended subject.

    A partial image, such as a half-body portrait, can be used as the target
    or as a source view if the subject is detected and estimation succeeds.
    InstantHMR estimates a full body from the available pixels. Pose and
    proportions for regions outside the frame or hidden by occlusion can
    therefore be less reliable. Every usable source view contributes equally
    to the proportion average, including its estimates for hidden regions.
    For a partial target, the full mesh is projected at the target scale
    and can extend the canvas when ``fit_canvas`` is enabled.

    Pose refinement matches landmarks by anatomical side. It selects global
    X/Y signs for the MediaPipe world coordinates by comparing their segment
    directions with the image observations. It calibrates relative Z scale
    by projecting the MediaPipe world joints with the InstantHMR focal length
    and the observed hip midpoint. A positive scale preserves depth order.
    With insufficient observations, the fitter uses a scale of 1.

    The fitting first orients the rig at the pelvis using the hip axis and
    the torso centerline. A lateral camera shift aligns the hip midpoint.
    It then adjusts the spine to align the shoulder axis and centerline.
    With split pose detection, image landmarks are merged in target pixels.
    Complete shoulder and hip axes are translated vertically to match the
    heights of their full-person midpoints. Their horizontal positions and
    the other regional landmarks are retained. Local 3D directions come from
    the crop containing both joints when available. The 3D torso centerline
    comes from the full-person detection.
    A separate pass refines the projected shoulder and hip widths, allowing
    up to three degrees of change in torso directions and signed twist.
    With the torso fixed, elbows, wrists, hands, knees, and ankles are fitted to
    their image positions. The sign of MediaPipe depth differences guides
    which end of each segment is nearer. Stages with no limb pixels use 3D
    directions when available. A final pass refines ankle flexion.
    Priors favor the InstantHMR angles for axial twists and spine curvature.
    A final collision pass can adjust hand or foot articulation when the
    skinned surface penetrates another region. It keeps the projected
    limb joints close to the fitted positions.

    The chosen proportions set the bone lengths. The distance between the
    virtual shoulder and hip midpoints can vary as the spine bends. Focal
    length and camera depth are fixed throughout the fit. Local clavicle,
    head, wrist, finger, and toe rotations come from InstantHMR.

    Volume controls deform the transferred or estimated surface before
    skinning. The selected skeletal proportions determine joint placement.
    Body thickness is scaled first. Local targets then move the surface
    toward profiles measured from the surrounding anatomy, with smooth
    transitions at the edges. The targets share their meaning across
    presets. For example, ``gender='feminine'`` and ``chest_volume=0``
    target the neutral rib cage, while ``chest_volume=None`` retains the
    chest from the selected shape after the general thickness adjustment.

    The PNG can be composited over a background or passed to an RGB
    annotator with ``background_color`` set.

    Examples
    --------
    Wire a single pose image to the main input and a ``FileImage`` containing
    one or more body references to the optional attachment::

        target >> body
        references >> body.source()
    """

    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        """Indicate whether person detection and coefficient inference use CUDA."""
        cfg = _read_cfg(resolve_spec(self.spec), self.id)

        return is_cuda_device(cfg.device)

    def source(self) -> AttachmentSink:
        """
        Declare the optional attachment for transferring body proportions.

        Use it when the main image is a pose reference and the mannequin
        should have the proportions of a different person. One reference is
        enough. Multiple photos of the same person, such as a frontal and
        a side view, give the node several estimates to average. The main
        image still supplies the joint angles and framing. Leave the
        attachment disconnected to reconstruct the main subject directly.

        Returns
        -------
        AttachmentSink
            Sink for one image payload. A ``FileImage`` with a list or directory
            of paths can supply several reference views through one wire.
        """

        return AttachmentSink(
            name=f'source:{self.id}',
            target=self,
            input_id='source',
        )

    def run(
        self,
        output_dir: str | Path,
        input: dict | None = None,
    ) -> dict:
        cfg = _read_cfg(resolve_spec(self.spec), self.id)

        target_path = resolve_single_image_path(
            node_id=self.id,
            path=None,
            input=input,
        )

        source_paths = []

        if input is not None and 'source' in input:
            source_paths = resolve_image_paths(
                node_id=self.id,
                path=None,
                input=input,
                input_key='source',
            )

        if not cfg.rig_path.is_file():
            raise FileNotFoundError(
                f"'{self.id}': missing MHR rig {cfg.rig_path}. Run ./fetch_mhr.sh."
            )

        if isinstance(cfg.checkpoint, Path) and not cfg.checkpoint.is_file():
            raise FileNotFoundError(
                f"'{self.id}': missing InstantHMR checkpoint {cfg.checkpoint}."
            )

        if (
            cfg.pose_landmarker_task is not None
            and not cfg.pose_landmarker_task.is_file()
        ):
            raise FileNotFoundError(
                f"'{self.id}': missing MediaPipe task {cfg.pose_landmarker_task}. "
                'Run ./fetch_media_pipe.sh.'
            )

        # The target estimate defines the output frame, camera, and joint pose.
        detector = get_yolo(model_name=cfg.yolo_model, device=cfg.device)
        target_image = load_init_image(target_path, apply_exif_orientation=True)
        render_width, render_height = target_image.size
        render_scale = 1.0

        if cfg.long_side is not None:
            render_scale = cfg.long_side / max(target_image.size)
            render_width, render_height = resolve_long_side_size(
                image=target_image, long_side=cfg.long_side, multiple=1,
            )
            render_width = max(1, render_width)
            render_height = max(1, render_height)

        target_rgb = np.array(target_image)
        target_bbox = _person_bbox(
            detector, target_rgb, cfg.device, target_path
        )
        runtime = get_instanthmr_model(
            checkpoint=cfg.checkpoint,
            rig_path=cfg.rig_path,
            device=cfg.device,
        )
        target = runtime.estimate(target_rgb, target_bbox)

        # Collect proportions in the shared MHR basis. Reuse the target estimate
        # when that same image also appears among the references.
        views, sources, rejected = [], [], []

        for path in source_paths:
            try:
                if path == target_path:
                    estimate = target
                else:
                    rgb = np.array(
                        load_init_image(path, apply_exif_orientation=True)
                    )
                    bbox = _person_bbox(detector, rgb, cfg.device, path)
                    estimate = runtime.estimate(rgb, bbox)

            except (OSError, ValueError) as exc:
                rejected.append({'image': str(path), 'reason': str(exc)})

                continue

            views.append(estimate.proportions)
            sources.append(str(path))

        # An explicit reference branch must contribute at least one estimate.
        if source_paths and not views:
            reasons = '; '.join(view['reason'] for view in rejected)
            raise ValueError(
                f"'{self.id}': no usable source images. {reasons}"
            )

        proportions = BodyProportions.mean(views) \
            if views else target.proportions

        # Fit body joints using the final proportions so their lengths agree
        # with the rendered skeleton.
        height, width = target_rgb.shape[:2]
        pose_fit = None
        landmarks = None
        crop_debug: dict[str, BodyPoseCrop] | None = {} if cfg.save_debug else None

        if cfg.pose_landmarker_task is not None:
            pose_landmarker = get_mediapipe_pose_landmarker(
                model_asset_path=str(cfg.pose_landmarker_task),
                device='cpu',
            )
            try:
                landmarks = detect_body_landmarks(
                    rgb=target_rgb,
                    bbox=target_bbox,
                    pose_landmarker=pose_landmarker,
                    split_pose_detection=cfg.split_pose_detection,
                    crop_debug=crop_debug,
                )
            except RuntimeError as exc:
                pose_fit = {'status': 'skipped', 'reason': str(exc)}
            else:
                target, pose_fit = refine_body_pose(
                    runtime=runtime,
                    target=target,
                    proportions=proportions,
                    landmarks=landmarks,
                    image_size=(width, height),
                    fit_head=cfg.fit_head,
                )

        mesh = partial(runtime.mesh, proportions=proportions)

        local_targets = (cfg.chest_volume, cfg.hip_volume, cfg.waist_size)

        if (
            cfg.body_volume != 1.0
            or cfg.neck_thickness != 1.0
            or any(value is not None for value in local_targets)
        ):
            body_axes = rest_body_axes(runtime.rig) if cfg.body_volume != 1.0 else None
            neck_axis = rest_neck_axis(runtime.rig) if cfg.neck_thickness != 1.0 else None
            mesh = partial(
                mesh,
                rest_deformation=partial(
                    adjust_body_volumes,
                    chest_volume=cfg.chest_volume,
                    hip_volume=cfg.hip_volume,
                    waist_size=cfg.waist_size,
                    body_volume=cfg.body_volume,
                    body_axes=body_axes,
                    neck_thickness=cfg.neck_thickness,
                    neck_axis=neck_axis,
                ),
            )

        vertices = mesh(target)

        if pose_fit is not None and pose_fit['status'] == 'fitted':
            target, vertices, collisions = refine_terminal_collisions(
                runtime=runtime,
                target=target,
                proportions=proportions,
                image_size=(width, height),
                vertices=vertices,
                mesh=mesh,
            )
            pose_fit['terminal_collisions'] = collisions

        render_focal = target.focal_length * render_scale
        padding = (0, 0, 0, 0)

        if cfg.fit_canvas:
            padding = fit_body_canvas(
                vertices,
                runtime.faces,
                width=render_width,
                height=render_height,
                focal=render_focal,
            )

        image = render_body(
            vertices,
            runtime.faces,
            width=render_width,
            height=render_height,
            focal=render_focal,
            body_color=cfg.body_color,
            background_color=cfg.background_color,
            padding=padding,
        )

        # Persist the PNG and the source diagnostics for downstream DAG runs.
        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='png',
        )
        image.save(out_path)
        output = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'image': str(out_path),
            'canvas_padding': dict(zip(('left', 'top', 'right', 'bottom'), padding)),
            'source_images': sources,
            'rejected_source_images': rejected,
        }

        if pose_fit is not None:
            output['pose_fit'] = pose_fit

        if cfg.save_debug:
            debug_dir = out_path.with_name(out_path.stem + '_debug')
            debug_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = debug_dir / 'pose_overlay.png'
            rig_xy = project_body_joints(
                runtime=runtime,
                target=target,
                proportions=proportions,
                image_size=(width, height),
            )
            # Account for integer rounding of the shorter side at the principal point.
            offset = (
                padding[0] + (render_width - width * render_scale) / 2,
                padding[1] + (render_height - height * render_scale) / 2,
            )
            draw_body_pose_overlay(
                image,
                rig_xy,
                landmarks,
                scale=render_scale,
                offset=offset,
            ).save(overlay_path)
            output['debug'] = {
                'debug_directory': str(debug_dir),
                'pose_overlay': str(overlay_path),
            }

            if crop_debug:
                output['debug']['mediapipe_crops'] = save_pose_crop_debug(
                    crop_debug, debug_dir,
                )

        output['metadata'] = str(write_json_sidecar(out_path, output))

        return output
