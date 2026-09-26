from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from morphalo.cache.models import get_mediapipe_face_landmarker
from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils.aux_annotator import build_aux_annotator
from morphalo.nodes.preprocess.utils.face_luminance_compositing import (
    compare_face_masks, composite_face_luminance, gray_to_rgb,
    sample_landmark_luminance, sharpen_face_luminance,
    warp_face_luminance_background)
from morphalo.nodes.sdxl_resolve import (resolve_image_paths,
                                         resolve_single_image_path)
from morphalo.nodes.vision.face_geometry import (_draw_landmarks,
                                                 _estimate_landmark_similarity,
                                                 _fuse_source_landmarks,
                                                 _transform_landmarks,
                                                 _yaw_degrees_from_transform,
                                                 draw_geometry_alignment,
                                                 draw_luminance_map,
                                                 face_mesh_mask)
from third_party.controlnet_aux.processor import MODEL_PARAMS, MODELS


@dataclass(frozen=True)
class Config:
    """Resolved FaceGeometryMap configuration."""

    device: str
    face_landmarker_task: str
    processor: str
    processor_device: str
    processor_params: Dict[str, Any]
    detail_gain: float


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Resolve and validate the landmark-transfer configuration.

    Parameters
    ----------
    spec : dict
        Resolved node specification.
    node_id : str
        Node identifier included in validation errors.

    Returns
    -------
    Config
        Validated MediaPipe configuration.

    Raises
    ------
    ValueError
        If the model block or Face Landmarker path is missing.
    """
    model = spec.get('model', {})

    if not isinstance(model, dict):
        raise ValueError(f"'{node_id}': model must be an object")

    face_landmarker_task = model.get('face_landmarker_task')

    if not face_landmarker_task:
        raise ValueError(
            f"'{node_id}': model.face_landmarker_task required "
            '(MediaPipe .task path)'
        )

    params = spec.get('params', {})

    if not isinstance(params, dict):
        raise ValueError(f"'{node_id}': params must be an object")

    processor = params.get('processor')
    processor_params = params.get('processor_params', {})

    try:
        detail_gain = float(params.get('detail_gain', 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{node_id}': params.detail_gain must be a non-negative number"
        ) from exc

    if not np.isfinite(detail_gain) or detail_gain < 0.0:
        raise ValueError(
            f"'{node_id}': params.detail_gain must be a non-negative number"
        )

    if not isinstance(processor_params, dict):
        raise ValueError(
            f"'{node_id}': params.processor_params must be an object")

    if processor not in MODELS:
        raise ValueError(
            f"'{node_id}': params.processor must be one of "
            f'{sorted(MODELS)}'
        )

    if processor == 'dwpose':
        raise ValueError(
            f"'{node_id}': params.processor='dwpose' is unavailable because "
            'the bundled backend has no pretrained loader'
        )

    return Config(
        device=str(model.get('device', 'cpu')),
        face_landmarker_task=str(face_landmarker_task),
        processor=str(processor),
        processor_device=str(params.get(
            'processor_device',
            model.get('device', 'cpu'),
        )),
        processor_params=dict(processor_params),
        detail_gain=detail_gain,
    )


def _load_image_rgb(path: Path) -> np.ndarray:
    """
    Load an image as an 8-bit RGB array.

    Parameters
    ----------
    path : pathlib.Path
        Path to the input image.

    Returns
    -------
    numpy.ndarray
        Image with shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))

    return rgb


def _run_processor(image_rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Run the configured annotator and restore target image geometry."""
    import cv2

    annotator = build_aux_annotator(
        cfg.processor,
        device=cfg.processor_device,
    )
    annotator_params = dict(MODEL_PARAMS.get(cfg.processor, {}))
    annotator_params.update(cfg.processor_params)
    result = annotator(
        Image.fromarray(image_rgb),
        **annotator_params,
    ).convert('RGB')
    output_rgb = np.asarray(result, dtype=np.uint8)
    height, width = image_rgb.shape[:2]

    if output_rgb.shape[:2] != (height, width):
        output_rgb = cv2.resize(
            output_rgb,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

    return output_rgb


@dataclass
class FaceGeometryMap(NodeRef):
    """
    Build a facial conditioning map from source geometry and a target image.

    ``FaceGeometryMap`` is a preprocessing node for face-guided image
    generation. It estimates the facial geometry shown in one or more source
    images, places that geometry over the face in a target image, constructs a
    coherent grayscale image around the transferred face, and runs a selected
    ``controlnet-aux`` processor on the result. The primary output is therefore
    a conditioning map, not a composited or face-swapped photograph.

    The source images determine facial proportions and approximate depth. The
    target determines the destination position, scale, and 3D orientation, as
    well as the low-frequency illumination used to render the synthetic face.
    Fine identity texture is deliberately not copied. In a complete generation
    workflow, the resulting map is normally combined with img2img and an
    identity conditioner such as IP-Adapter FaceID.

    Typical uses include:

    - generating a depth map that constrains the broad 3D structure of a new
      face while preserving the target composition;
    - generating Canny, line-art, HED, or similar structural maps from the same
      transferred geometry;
    - comparing several annotators without repeating face detection, multiview
      fusion, and target alignment outside the node;
    - using several views of the same person to improve the estimated source
      depth, especially around geometry that is weak in a frontal view.

    Examples
    --------
    Wire the target image through the normal DAG edge and one or more views of
    the source face through the dedicated attachment returned by
    :meth:`source`::

        target_image >> face_geometry_map
        source_images >> face_geometry_map.source()

    A typical depth configuration is::

        FaceGeometryMap(
            name='face_depth',
            spec={
                'model': {
                    'face_landmarker_task': '/models/face_landmarker.task',
                    'device': 'cuda',
                },
                'params': {
                    'processor': 'depth_midas',
                    'processor_device': 'cuda',
                    'detail_gain': 0.0,
                },
            },
        )

    Processing flow
    ---------------
    The node performs the following steps:

    1. Detect MediaPipe XYZ face landmarks in the target and in every usable
       source image.
    2. Select the source view with the smallest absolute yaw as the frontal
       reference and normalize all other source landmarks into its coordinate
       frame.
    3. Preserve the reference X/Y geometry and fuse source Z coordinates with
       yaw-derived weights when multiple views are available.
    4. Fit a 3D similarity transform from the fused source landmarks to the
       target landmarks. This transfers target placement, scale, and pose while
       retaining the source facial proportions represented by MediaPipe.
    5. Sample the target's low-frequency luminance at its facial landmarks and
       rasterize those values over the transformed source mesh. This produces a
       synthetic grayscale face with target-compatible lighting.
    6. Build a local coordinate deformation that contracts the surrounding
       target image toward the synthetic face boundary. The deformation is
       strongest beside the replacement and decays smoothly to zero away from
       it, preserving real surrounding luminance instead of inventing a fill.
    7. Run the selected ``controlnet-aux`` processor on the completed image and
       save its result as the node's primary conditioning map.
    8. Save landmark overlays and luminance-compositing intermediates in a
       separate debug directory, together with a JSON sidecar.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG. If omitted, an identifier is
        assigned by the enclosing DAG.

    spec : dict or str or pathlib.Path or sequence of (dict or str or pathlib.Path)
        Node configuration specification resolved by ``resolve_spec``.

        The specification may be an in-memory dictionary, a HOCON file path,
        or a sequence of specifications merged from left to right.

        Expected structure:

        ``model`` : dict
            Face-landmark model configuration.

            ``face_landmarker_task`` : str
                Required path to the MediaPipe Face Landmarker ``.task`` file.

            ``device`` : str, optional
                Logical device used by the shared MediaPipe model cache.
                Default: ``'cpu'``.

        ``params`` : dict
            Geometry-map and annotator configuration.

            ``processor`` : str
                Required ``controlnet-aux`` processor identifier. It must be
                present in ``third_party.controlnet_aux.processor.MODELS``.
                Useful choices include ``'depth_midas'``, ``'canny'``, and the
                available line-art processors. ``'dwpose'`` is not supported by
                this node because the bundled backend has no pretrained loader.

            ``processor_device`` : str, optional
                Device used by checkpoint-backed annotators. Defaults to
                ``model.device``. Stateless processors may ignore this value.

            ``processor_params`` : dict, optional
                Keyword arguments forwarded to the selected annotator after its
                defaults from ``MODEL_PARAMS`` have been loaded. Supported keys
                depend on the processor.

            ``detail_gain`` : float, optional
                Strength of a mask-aware unsharp filter applied to the complete
                rasterized face before it is blended into the grayscale
                intermediate. Default: ``0.0``.

                Keep ``0.0`` for depth processors, where smooth continuous
                luminance is generally preferable. Values around ``1.0`` to
                ``1.5`` are useful starting points for line-art processors, and
                ``1.5`` to ``2.0`` for Canny. Higher values make eyes, lips,
                nose, and eyebrows easier for edge-oriented processors to
                detect, but may introduce halos or preserve unwanted target
                detail.

    Inputs
    ------
    default : dict
        Target payload containing an image path under ``image`` or ``path``.
        The detected face provides the destination pose and placement.

    source : dict
        Source payload containing one or more paths under ``image``, ``images``,
        or ``path``. All usable images should depict the same person. Images in
        which no face is detected are reported and skipped; execution fails if
        no source face can be detected.

    Outputs
    -------
    dict
        Primary output dictionary, also persisted as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the conditioning map generated by the selected processor.

        ``source_images`` : list[str]
            Source images that contributed usable landmarks.

        ``rejected_source_images`` : list[dict]
            Source images skipped because a face could not be detected.

        ``target_image`` : str
            Resolved target image path.

        ``debug_directory`` : str
            Directory containing source and target landmark overlays, normalized
            source views, the fused source geometry, and luminance-compositing
            intermediates.

        ``transform`` : list[list[float]]
            Fitted 3D similarity transform from fused source geometry to target
            geometry.

        ``source_views`` : list[dict]
            Per-source yaw, depth weight, base-view status, and debug paths.

        ``metadata`` : str
            Path to the JSON sidecar.

    Notes
    -----
    - MediaPipe depth is relative rather than metric and may regularize subtle
      identity geometry, particularly the shape and asymmetry of the nose.
    - Source X/Y coordinates come from the view with minimum absolute yaw. With
      multiple sources, Z is a yaw-weighted mean that favors oblique views up
      to 60 degrees and rejects views at or beyond 80 degrees.
    - The node transfers the expression already represented by the fused source
      landmarks. It does not separately estimate or retarget target expression.
    - The grayscale intermediate is an implementation surface used to obtain a
      coherent processor input. It is not intended to be the final generated
      image.
    - Face selection is currently limited to the face returned by the shared
      landmark helper; multi-face selection is not exposed as configuration.
    - Identity texture, skin detail, and photorealistic reconstruction remain
      the responsibility of downstream image generation and identity
      conditioning nodes.
    """

    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        """Return whether the configured processor uses a CUDA checkpoint."""
        cfg = _read_cfg(resolve_spec(self.spec), node_id=self.id)
        model = MODELS.get(cfg.processor)

        return (
            bool(model and model['checkpoint'])
            and is_cuda_device(cfg.processor_device)
        )

    def source(self) -> AttachmentSink:
        """
        Declare the input that supplies the source facial geometry.

        Use this sink to connect one or more images of the person whose facial
        proportions should be transferred. It is separate from the default DAG
        input: the default input supplies the target image, pose, placement, and
        illumination, while ``source`` supplies the geometry used to construct
        the replacement face map.

        A single connected image is sufficient. Multiple source images may be
        connected to improve the depth estimate by showing the same face from
        different yaw angles. All source images should depict the same person;
        they are fused as observations of one facial geometry rather than
        treated as independent identities.

        Examples
        --------
        Connect the target through the normal edge and the source reference
        through this attachment::

            target_image >> face_geometry_map
            source_image >> face_geometry_map.source()

        Several views may feed the same attachment::

            source_front >> face_geometry_map.source()
            source_left >> face_geometry_map.source()
            source_right >> face_geometry_map.source()

        Returns
        -------
        AttachmentSink
            Attachment sink bound to the node's ``source`` input channel. The
            connected payloads are resolved as source image paths when the node
            executes.
        """

        return AttachmentSink(
            name=f'source:{self.id}',
            target=self,
            input_id='source',
        )

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Detect, align, render, and persist source face landmarks.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Base directory where node artifacts are written.
        input : dict[str, dict] or None, optional
            Upstream payloads keyed by input channel.

        Returns
        -------
        dict[str, Any]
            Output payload containing landmark and debug artifacts.
        """
        from morphalo.nodes.vision.face_landmarks import (
            mp_face_landmarks_xyz, mp_face_landmarks_xyz_with_transform)

        # 1. Resolve the target placement image and the source geometry image.
        cfg = _read_cfg(resolve_spec(self.spec), node_id=self.id)
        target_path = resolve_single_image_path(
            node_id=self.id,
            path=None,
            input=input,
            input_key='default',
        )
        source_paths = resolve_image_paths(
            node_id=self.id,
            path=None,
            input=input,
            input_key='source',
        )

        # 2. Detect corresponding MediaPipe landmarks with one cached model.
        target_rgb = _load_image_rgb(target_path)
        landmarker = get_mediapipe_face_landmarker(
            model_asset_path=cfg.face_landmarker_task,
            device=cfg.device,
            output_facial_transformation_matrixes=True,
        )

        source_landmarks = []
        source_yaws = []
        usable_source_paths = []
        usable_source_rgbs = []
        rejected_sources = []
        for source_path in source_paths:
            source_rgb = _load_image_rgb(source_path)

            try:
                source_xyz, source_transform = (
                    mp_face_landmarks_xyz_with_transform(
                        img_rgb=source_rgb,
                        face_landmarker=landmarker,
                    )
                )
            except RuntimeError as exc:
                rejected_sources.append({
                    'image': str(source_path),
                    'reason': str(exc),
                })
                continue

            usable_source_paths.append(source_path)
            usable_source_rgbs.append(source_rgb)
            source_landmarks.append(source_xyz)
            source_yaws.append(
                _yaw_degrees_from_transform(source_transform)
            )

        if not source_landmarks:
            raise RuntimeError(
                f"FaceGeometryMap node '{self.id}': no source face "
                'could be detected'
            )

        target_xyz = mp_face_landmarks_xyz(
            img_rgb=target_rgb,
            face_landmarker=landmarker,
        )

        # 3. Normalize every source into the most frontal source frame. Keep
        # that base X/Y and fuse Z using the confidence derived from source yaw.
        fused_source_xyz, normalized_sources, base_index, depth_weights = (
            _fuse_source_landmarks(source_landmarks, source_yaws)
        )

        # 4. Fit one 3D similarity from the fused source geometry to the target.
        # Its rotation transfers target pose without changing source proportions.
        matrix = _estimate_landmark_similarity(fused_source_xyz, target_xyz)
        transformed_xyz = _transform_landmarks(fused_source_xyz, matrix)
        transformed_xy = transformed_xyz[:, :2]

        # 5. Render exactly one primary conditioning map for the DAG output.
        # For processor output, first construct one coherent grayscale image.
        # Processors are sensitive to edits in their output space, whereas an
        # ordinary luminance image tolerates geometric warping and blending.
        import cv2

        target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
        target_mask = face_mesh_mask(
            target_rgb.shape,
            target_xyz[:, :2],
        )
        face_span = max(
            float(np.ptp(target_xyz[:, 0])),
            float(np.ptp(target_xyz[:, 1])),
            float(np.ptp(transformed_xyz[:, 0])),
            float(np.ptp(transformed_xyz[:, 1])),
        )

        # Transfer only the target's low-frequency illumination field. The
        # transformed source landmarks provide shape, while IP-Adapter and
        # img2img remain responsible for identity texture and fine detail.
        blur_sigma = max(1.0, face_span * 0.015)
        vertex_luminance = sample_landmark_luminance(
            target_gray,
            target_xyz[:, :2],
            blur_sigma=blur_sigma,
        )

        synthetic_face, insert_mask = draw_luminance_map(
            target_rgb.shape,
            transformed_xyz,
            vertex_luminance,
        )
        synthetic_face = sharpen_face_luminance(
            synthetic_face,
            insert_mask,
            detail_gain=cfg.detail_gain,
        )
        geometry_comparison = compare_face_masks(target_mask, insert_mask)

        # Remove a meaningful margin around the target mesh so residual skin
        # does not survive beneath the new face.
        erase_radius = max(2, int(round(face_span * 0.05)))
        erase_size = erase_radius * 2 + 1
        erase_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erase_size, erase_size),
        )
        erase_mask = cv2.dilate(
            target_mask.astype(np.uint8),
            erase_kernel,
        ) > 0
        feather_radius = max(2, int(round(face_span * 0.025)))
        warped_gray, warp_displacement, warp_params = (
            warp_face_luminance_background(
                target_gray,
                erase_mask,
                insert_mask,
                face_span=face_span,
                feather_radius=feather_radius,
            )
        )
        composed_gray, blend_alpha = composite_face_luminance(
            warped_gray,
            synthetic_face,
            insert_mask,
            feather_radius=feather_radius,
        )
        composed_rgb = gray_to_rgb(composed_gray)
        output_map = _run_processor(composed_rgb, cfg)
        luminance_debug = {
            'target_gray': gray_to_rgb(target_gray),
            'synthetic_face': gray_to_rgb(synthetic_face),
            'warped_background': gray_to_rgb(warped_gray),
            'composite': composed_rgb,
            'processor': output_map,
            'erase_mask': gray_to_rgb(
                erase_mask.astype(np.uint8) * 255
            ),
            'target_mesh_mask': gray_to_rgb(
                target_mask.astype(np.uint8) * 255
            ),
            'insert_mask': gray_to_rgb(
                insert_mask.astype(np.uint8) * 255
            ),
            'blend_alpha': gray_to_rgb(
                np.rint(blend_alpha * 255.0).astype(np.uint8)
            ),
            'warp_displacement': gray_to_rgb(
                np.rint(
                    warp_displacement
                    / max(float(np.max(warp_displacement)), 1e-6)
                    * 255.0
                ).astype(np.uint8)
            ),
            'erase_radius': erase_radius,
            'feather_radius': feather_radius,
            'blur_sigma': blur_sigma,
            'detail_gain': cfg.detail_gain,
            'warp_params': warp_params,
            'geometry_comparison': geometry_comparison,
        }

        target_debug = _draw_landmarks(
            target_rgb,
            target_xyz[:, :2],
            color=(0, 255, 255),
        )
        geometry_alignment = draw_geometry_alignment(
            target_rgb,
            target_xyz[:, :2],
            transformed_xy,
            target_mask,
            insert_mask,
        )

        # 6. Persist the primary output and collect every diagnostic image in a
        # timestamped directory next to it.
        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='png',
        )
        debug_dir = out_path.with_name(out_path.stem + '_debug')
        debug_dir.mkdir(parents=True, exist_ok=True)
        target_debug_path = debug_dir / 'target.png'
        geometry_alignment_path = debug_dir / 'geometry_alignment.png'
        source_result_path = debug_dir / 'source_result.png'
        base_canvas = np.zeros_like(usable_source_rgbs[base_index])
        source_result = _draw_landmarks(
            base_canvas,
            fused_source_xyz[:, :2],
            color=(255, 255, 255),
        )

        Image.fromarray(output_map).save(out_path)
        Image.fromarray(target_debug).save(target_debug_path)
        Image.fromarray(geometry_alignment).save(geometry_alignment_path)
        Image.fromarray(source_result).save(source_result_path)

        luminance_debug_paths = {}
        if luminance_debug is not None:
            debug_names = (
                'target_gray',
                'synthetic_face',
                'warped_background',
                'composite',
                'processor',
                'erase_mask',
                'target_mesh_mask',
                'insert_mask',
                'blend_alpha',
                'warp_displacement',
            )

            for name in debug_names:
                debug_path = debug_dir / f'luminance_{name}.png'
                Image.fromarray(luminance_debug[name]).save(debug_path)
                luminance_debug_paths[name] = str(debug_path)

        source_views = []
        for index, (path, rgb, original_xyz, normalized_xyz, yaw, weight) in (
            enumerate(zip(
                usable_source_paths,
                usable_source_rgbs,
                source_landmarks,
                normalized_sources,
                source_yaws,
                depth_weights,
            ))
        ):
            source_number = index + 1
            source_debug_path = debug_dir / f'source_{source_number}.png'
            normalized_debug_path = (
                debug_dir / f'source_{source_number}_norm.png'
            )
            source_debug = _draw_landmarks(
                rgb,
                original_xyz[:, :2],
                color=(255, 255, 0),
            )
            normalized_debug = _draw_landmarks(
                base_canvas,
                normalized_xyz[:, :2],
                color=(255, 255, 255),
            )

            Image.fromarray(source_debug).save(source_debug_path)
            Image.fromarray(normalized_debug).save(normalized_debug_path)

            source_views.append({
                'image': str(path),
                'yaw_degrees': yaw,
                'depth_weight': weight,
                'is_base': index == base_index,
                'debug_image': str(source_debug_path),
                'debug_normalized': str(normalized_debug_path),
            })

        base_debug_path = source_views[base_index]['debug_image']

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'image': str(out_path),
            'source_images': [str(path) for path in usable_source_paths],
            'rejected_source_images': rejected_sources,
            'target_image': str(target_path),
            'debug_directory': str(debug_dir),
            'debug_source': base_debug_path,
            'debug_source_result': str(source_result_path),
            'debug_target': str(target_debug_path),
            'debug_geometry_alignment': str(geometry_alignment_path),
            'geometry_alignment_legend': {
                'target_landmarks': 'cyan',
                'synthetic_landmarks': 'magenta',
                'target_contour': 'green',
                'synthetic_contour': 'red',
            },
            'transform': matrix.tolist(),
            'source_views': source_views,
            'model': {
                'face_landmarker_task': cfg.face_landmarker_task,
                'device': cfg.device,
            },
            'params': {
                'processor': cfg.processor,
                'processor_device': cfg.processor_device,
                'processor_params': cfg.processor_params,
                'detail_gain': cfg.detail_gain,
            },
        }

        if luminance_debug is not None:
            out['luminance_compositing'] = {
                'erase_radius': luminance_debug['erase_radius'],
                'feather_radius': luminance_debug['feather_radius'],
                'blur_sigma': luminance_debug['blur_sigma'],
                'detail_gain': luminance_debug['detail_gain'],
                'warp_params': luminance_debug['warp_params'],
                'geometry_comparison': (
                    luminance_debug['geometry_comparison']
                ),
                'debug': luminance_debug_paths,
            }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
