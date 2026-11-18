"""Render a posed MHR mannequin with optional source body proportions."""

from dataclasses import dataclass, field
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
from morphalo.nodes.preprocess.utils.body_pose_fit import refine_body_pose
from morphalo.nodes.preprocess.utils.body_render import render_body
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


@dataclass(frozen=True)
class Config:
    """Resolved model assets, inference settings, and render colors."""

    checkpoint: Path | dict[str, str]
    rig_path: Path
    pose_landmarker_task: Path | None
    yolo_model: str
    device: str
    body_color: tuple[int, int, int]
    background_color: tuple[int, int, int] | None


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Resolve model settings and validate the render colors.

    Parameters
    ----------
    spec : dict
        Resolved specification with a ``model`` block and optional ``params``.
    node_id : str
        Node identifier included in validation errors.

    Returns
    -------
    Config
        Configuration with parsed RGB colors and absolute paths for local
        assets. A checkpoint mapping describes a file on Hugging Face.

    Raises
    ------
    ValueError
        If a configuration block, checkpoint, model path, or color is invalid.
    """
    model = spec.get('model', {})
    params = spec.get('params', {})

    if not isinstance(model, dict) or not isinstance(params, dict):
        raise ValueError(f"'{node_id}': model and params must be objects")

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

    return Config(
        checkpoint=checkpoint,
        rig_path=Path(rig_path).expanduser().resolve(),
        pose_landmarker_task=pose_task,
        yolo_model=str(model.get('yolo_model', 'yolov8n.pt')),
        device=str(model.get('device', 'cuda')),
        body_color=body_color,
        background_color=background_color,
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
    3. If configured, refine the pose against visible MediaPipe joints
       using the chosen proportions and the original camera.
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
                Local MediaPipe Pose Landmarker task. A path enables
                refinement of body articulation using visible shoulders,
                elbows, wrists, hips, knees, and ankles. Omit it to render
                the direct InstantHMR pose. The task installed by
                ``./fetch_media_pipe.sh`` is
                ``~/models/mediapipe/pose_landmarker_heavy.task``.
            ``yolo_model`` : str, optional
                Person detector checkpoint name. Defaults to ``'yolov8n.pt'``.
                The loader checks ``~/models/yolo`` first. Ultralytics
                downloads missing official weights into that directory.
            ``device`` : str, optional
                Device for YOLO and InstantHMR, defaulting to ``'cuda'``.
                TorchScript mesh decoding runs on CPU.

        ``params`` : dict, optional
            ``body_color`` : str or sequence of numbers, optional
                Mannequin base color before shading. Defaults to ``'#b8b8b8'``.
            ``background_color`` : str or sequence of numbers or None, optional
                Fill color. Defaults to ``None`` for a transparent background.
                Both colors accept CSS names, hex strings, or RGB triplets.

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
            PNG path. The image matches the EXIF-oriented target dimensions.
            Visible mannequin pixels are opaque.
        ``source_images`` : list[str]
            Reference paths that contributed to the arithmetic mean.
        ``rejected_source_images`` : list[dict]
            Rejected reference paths and their failure reasons.
        ``pose_fit`` : dict, optional
            Status, visible joint count, and weighted pixel reprojection
            error when available. Present when a pose task is configured.
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
    For a partial target, the full mesh is projected into the target frame
    and any geometry outside the image bounds is clipped.

    Pose refinement keeps the inferred camera and the selected body
    proportions fixed. It adjusts torso and limb angles while retaining
    the local clavicle, head, wrist, and finger rotations estimated by
    InstantHMR. Keeping the clavicles fixed preserves the 3D distance
    between the two shoulder joints. The projected width can still change
    with torso rotation. A fit with too few visible joints or no
    reprojection improvement uses the initial pose.

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
        target_rgb = np.array(
            load_init_image(target_path, apply_exif_orientation=True)
        )
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
        pose_fit = None

        if cfg.pose_landmarker_task is not None:
            pose_landmarker = get_mediapipe_pose_landmarker(
                model_asset_path=str(cfg.pose_landmarker_task),
                device='cpu',
            )
            target, pose_fit = refine_body_pose(
                runtime=runtime,
                target=target,
                proportions=proportions,
                rgb=target_rgb,
                bbox=target_bbox,
                pose_landmarker=pose_landmarker,
            )

        # Render the resulting mesh in the target camera frame.
        vertices = runtime.mesh(target, proportions)
        height, width = target_rgb.shape[:2]
        image = render_body(
            vertices,
            runtime.faces,
            width=width,
            height=height,
            focal=target.focal_length,
            body_color=cfg.body_color,
            background_color=cfg.background_color,
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
            'source_images': sources,
            'rejected_source_images': rejected,
        }

        if pose_fit is not None:
            output['pose_fit'] = pose_fit

        output['metadata'] = str(write_json_sidecar(out_path, output))

        return output
