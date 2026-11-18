from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from morphalo.cache.models import (
    get_face_reconstruction_model, get_mediapipe_face_landmarker,
)
from morphalo.core.cuda_mem import CudaPostRunMixin
from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.cuda_stat import cuda_mem_stats, cuda_prerun, cuda_sync
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils.face_reconstruction_geometry import (
    AlignedFace, align_reconstruction_face, compose_reconstructed_face,
    reconstruction_to_image, render_reconstruction,
    transfer_reconstruction_parameters,
)
from morphalo.nodes.sdxl_resolve import resolve_image_paths, resolve_single_image_path
from morphalo.nodes.vision.face_landmarks import mp_face_landmarks_xyz
from third_party.tddfa_v3.runtime import COEFFICIENT_SLICES


ASSET_REPOSITORY = 'Zidu-Wang/3DDFA-V3'


@dataclass(frozen=True)
class Config:
    """Resolved model, source-transfer, and diagnostic configuration."""

    device: str
    face_landmarker_task: str
    repo_id: str
    revision: str
    preserve: tuple[str, ...]
    color_transfer: float
    occlusion_threshold: float
    save_debug: bool


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Validate reconstruction settings and mandatory parameter provenance.

    Parameters
    ----------
    spec : dict
        Resolved node specification.
    node_id : str
        Identifier included in configuration errors.

    Returns
    -------
    Config
        Source identity is implicit; only expression and albedo can be selected.

    Raises
    ------
    ValueError
        If configuration blocks or parameter selection are invalid.
    """
    model, params = spec.get('model', {}), spec.get('params', {})

    if not isinstance(model, dict) or not isinstance(params, dict):
        raise ValueError(f"'{node_id}': model and params must be objects")

    task = model.get('face_landmarker_task')

    if not isinstance(task, str) or not task:
        raise ValueError(f"'{node_id}': model.face_landmarker_task required")

    preserve = params.get('preserve', [])
    allowed = {'expression', 'albedo'}

    if (not isinstance(preserve, (list, tuple))
            or not all(isinstance(value, str) for value in preserve)
            or set(preserve) - allowed):
        raise ValueError(
            f"'{node_id}': params.preserve accepts only expression and albedo. "
            'Identity always comes from source and is not configurable. '
            'Pose, translation, and illumination always come from target.'
        )

    try:
        color_transfer = float(params.get('color_transfer', 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{node_id}': params.color_transfer must be a number in [0, 1]"
        ) from exc
    if not np.isfinite(color_transfer) or not 0.0 <= color_transfer <= 1.0:
        raise ValueError(
            f"'{node_id}': params.color_transfer must be finite and in [0, 1]"
        )
    if color_transfer > 0.0 and 'albedo' in preserve:
        raise ValueError(
            f"'{node_id}': params.color_transfer > 0 is incompatible with "
            "preserve=['albedo']; color matching requires target albedo"
        )

    try:
        occlusion_threshold = float(params.get('occlusion_threshold', 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{node_id}': params.occlusion_threshold must be a number in [0, 1]"
        ) from exc
    if not np.isfinite(occlusion_threshold) or not 0.0 <= occlusion_threshold <= 1.0:
        raise ValueError(
            f"'{node_id}': params.occlusion_threshold must be finite and in [0, 1]"
        )

    debug = spec.get('debug', {})

    if not isinstance(debug, dict):
        raise ValueError(f"'{node_id}': debug must be an object")

    device = str(model.get('device', 'cuda'))

    return Config(
        device=device,
        face_landmarker_task=task,
        repo_id=str(model.get('id', ASSET_REPOSITORY)),
        revision=str(model.get('revision', 'main')),
        preserve=tuple(dict.fromkeys(preserve)),
        color_transfer=color_transfer,
        occlusion_threshold=occlusion_threshold,
        save_debug=bool(debug.get('save_debug', False)),
    )


def _load_rgb(path: Path) -> np.ndarray:
    """Load a filesystem image as an independent uint8 RGB array."""
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def _load_occlusion(path: Path, shape: tuple) -> np.ndarray:
    """
    Load a target-space occlusion mask using white-as-protected semantics.

    Parameters
    ----------
    path : pathlib.Path
        Grayscale or RGB mask image. White protects fully; black permits changes.
    shape : tuple[int, ...]
        Target shape beginning with height and width. Differing mask dimensions
        are resized by nearest neighbor, preserving the supplied grayscale levels.

    Returns
    -------
    numpy.ndarray
        Float32 ``(H, W)`` protection alpha in [0, 1], including soft edges.
    """
    with Image.open(path) as image:
        mask = image.convert('L')
        size = (shape[1], shape[0])

        if mask.size != size:
            mask = mask.resize(size, resample=Image.Resampling.NEAREST)

        return np.asarray(mask, dtype=np.float32) / 255.0


def _save_reconstruction(
    debug_dir: Path,
    prefix: str,
    rgb: np.ndarray,
    alignment: AlignedFace,
    coefficients: np.ndarray,
    decoded: dict,
    triangles: np.ndarray,
    raster: dict,
    mapped: dict,
) -> dict:
    """
    Save one reconstruction's visual and numeric diagnostics without rerendering.

    Crop-space images show the network input, predicted-light color, and neutral
    studio shading. The overlay and mask use the full original image frame so
    alignment can be inspected. The NPZ archive retains coefficients, topology,
    vertex attributes, and coordinate transforms for numerical investigation.
    The caller invokes this helper only when debug output is enabled.

    Parameters
    ----------
    debug_dir : pathlib.Path
        Existing diagnostic directory.
    prefix : str
        Unique source/target/mixed artifact prefix.
    rgb : numpy.ndarray
        Original RGB image onto which the rendered crop is mapped.
    alignment : AlignedFace
        Input crop and original-image transforms.
    coefficients : numpy.ndarray
        Reconstruction parameters with shape ``(257,)``.
    decoded : dict
        Mesh and color data from the cached decoder.
    triangles : numpy.ndarray
        Shared mesh connectivity.
    raster : dict
        Color, neutral-shape, and mask renders in crop coordinates.
    mapped : dict
        Same renders mapped to the original image frame.

    Returns
    -------
    dict
        Diagnostic paths, alignment transforms, and coefficient groups.
    """
    overlay = rgb.copy()
    overlay[mapped['mask']] = np.rint(
        0.3 * rgb[mapped['mask']] + 0.7 * mapped['shape'][mapped['mask']],
    ).astype(np.uint8)
    images = {
        'crop': alignment.rgb,
        'render': raster['shape'],
        'color': raster['color'],
        'overlay': overlay,
        'mask': mapped['mask'].astype(np.uint8) * 255,
    }

    paths = {}

    for name, image in images.items():
        path = debug_dir / f'{prefix}_{name}.png'
        Image.fromarray(image).save(path)
        paths[name] = str(path)

    geometry = debug_dir / f'{prefix}_geometry.npz'
    # Projected vertices use the model's 224-pixel crop coordinates, regardless
    # of raster size. Add homogeneous coordinates to map them back to the image.
    screen = np.column_stack((
        decoded['screen_vertices'],
        np.ones(len(decoded['screen_vertices'])),
    ))
    original_xy = (screen @ alignment.crop_to_image.T)[:, :2]

    np.savez_compressed(
        geometry,
        coefficients=coefficients,
        triangles=triangles,
        canonical_vertices=decoded['canonical_vertices'],
        camera_vertices=decoded['camera_vertices'],
        screen_vertices=decoded['screen_vertices'],
        original_xy=original_xy,
        albedo=decoded['albedo'],
        colors=decoded['colors'],
        image_to_crop=alignment.image_to_crop,
        crop_to_image=alignment.crop_to_image,
    )

    paths['geometry'] = str(geometry)

    return {
        'debug': paths,
        'coefficients': {
            name: coefficients[indices].tolist()
            for name, indices in COEFFICIENT_SLICES.items()
        },
        'alignment': {
            'image_to_crop': alignment.image_to_crop.tolist(),
            'crop_to_image': alignment.crop_to_image.tolist(),
            'landmarks_five': alignment.landmarks.tolist(),
        },
    }


@dataclass
class FaceReconstruction(CudaPostRunMixin, NodeRef):
    """
    Reconstruct and transfer source facial identity with 3DDFA-V3.

    The default input supplies the target image. The ``source`` attachment
    supplies one or more views of the same identity. MediaPipe supplies five
    landmarks to the original 3DDFA-V3 alignment routine; a ResNet-50 regresses
    the morphable-model parameters from its 224-by-224 RGB crop.

    Identity geometry always comes from the arithmetic mean of source identity
    coefficients. Rotation, translation, and spherical-harmonic illumination
    always come from target. Expression and albedo default to target and may
    optionally be selected from source. Diagnostic neutral renders use fixed
    studio lighting; the transferred color render uses target illumination.

    The output is an RGB reconstruction composite. Conditioning maps can be
    produced by subsequent nodes. Skin microtexture remains the responsibility
    of downstream identity-conditioned image generation.

    Processing flow
    ---------------
    1. Align and reconstruct the target to obtain destination pose, placement,
       illumination, and the original face silhouette.
    2. Infer coefficients for each usable source view and average them in the
       shared model basis. Transfer identity and optional source groups into
       the target vector.
    3. Render the mixed face in the target frame and optionally match its colors
       to reliable original-target face pixels. Radially deform the original
       silhouette beneath the eroded replacement silhouette and feather the
       RGB render onto the warped background.
    4. Save the composite and JSON card. When debug is enabled, also save
       separate source, target, and mixed reconstructions with numeric geometry.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG. Assigned by the enclosing DAG
        when omitted.
    spec : dict or str or pathlib.Path or sequence of these, optional
        Node configuration resolved by ``resolve_spec``. Dictionaries and HOCON
        paths may be supplied together in a sequence, merged from left to right.

        Expected structure:

        ``model`` : dict
            ``face_landmarker_task`` : str
                Required MediaPipe Face Landmarker task path.

            ``device`` : str, optional
                Torch device for coefficient inference and mesh decoding.
                Default: ``'cuda'``. Set ``'cpu'`` for CPU inference.
                MediaPipe landmark detection runs on CPU.

            ``id`` : str, optional
                Asset dataset repository. Default: ``'Zidu-Wang/3DDFA-V3'``.
                Alternatives must provide compatible files under ``assets/``.

            ``revision`` : str, optional
                Dataset revision. Default: ``'main'``.

            External assets use the standard Hugging Face cache, normally
            ``~/.cache/huggingface/hub``, as other Morphalo weights do.

        ``params`` : dict, optional
            ``preserve`` : list[str], optional
                Additional source groups to transfer:

                - ``'expression'``: facial deformation.
                - ``'albedo'``: surface color before lighting.

                Default: ``[]``. Identity is always transferred implicitly and
                must not be listed. Pose, placement, and illumination always
                remain those of the target.

            ``color_transfer`` : float, optional
                Strength of Lab-space statistical matching to the original
                target face. Range: [0, 1]. Default: 0 disables matching.
                Only compatible with target albedo: a positive value together
                with ``preserve=['albedo']`` raises ValueError.

                After rendering and before feathering, statistics are estimated
                from the common interior of original and reconstructed masks,
                excluding occlusion and a two-pixel boundary margin. Correction
                is applied to the full visible render. No surviving samples
                leaves the render unchanged. Raw mixed debug renders and decoded
                vertex colors retain the model prediction for comparison.

            ``occlusion_threshold`` : float, optional
                Threshold for blocking radial warp directions. Range: [0, 1].
                Default: 0. A ray is blocked only if the mask value at its
                eroded-new-face endpoint is strictly greater than this value.
                Zero blocks any nonblack sample, including soft edges; 0.5
                blocks values above half intensity; 1 blocks no directions.
                Raise this value if mask smoothing blocks too many directions.
                This never thresholds the final protection alpha.

        ``debug`` : dict, optional
            ``save_debug`` : bool, optional
                Compute and save source renders and diagnostic images, geometry
                archives, and coefficients. Default: ``False``. Target-mask and
                transferred-color rendering are required in both modes.

    Inputs
    ------
    default : dict
        One target path under ``image`` or ``path``.
    source : dict
        Source paths under ``image``, ``images``, or ``path``. Undetected faces
        are reported and skipped; execution fails if no source remains.
    occlusion : dict, optional
        Target-space occlusion mask path under ``image`` or ``path``, connected
        through :meth:`occlusion`. White protects original pixels and black
        allows modification; gray values provide partial protection. A different
        mask size is resized to the target by nearest-neighbor sampling.

    Outputs
    -------
    dict
        Primary payload, also written to a JSON sidecar. Operational fields are
        present independently of debug mode:

        ``image``, ``composite`` : str
            Two references to the same saved RGB composite.
        ``source_images`` : list[str]
            Source images whose detected faces contributed to the source mean.
        ``source_views`` : list[dict]
            Operational source records containing only each contributing image
            path. Numeric and visual source diagnostics live under ``debug``.
        ``rejected_source_images`` : list[dict]
            Undetected source faces, with their image paths and failure reasons.
        ``target_image`` : str
            Resolved target image path.
        ``occlusion_image`` : str, optional
            Resolved input mask path, present only when an occlusion is wired.
        ``parameter_origins`` : dict[str, str]
            Whether each coefficient group comes from source mean or target.
        ``model``, ``params`` : dict
            Resolved model/assets, device, revision, and transfer configuration.
        ``composition`` : dict
            Applied destination-erosion and feathering radii in pixels. Enabled
            color transfer also reports sample count and whether it was applied.
        ``timing`` : dict
            ``seconds`` measures processing after model loading, including all
            optional diagnostic renders and file writes. Writing the final JSON
            sidecar itself is excluded.
        ``cuda_mem`` : dict, optional
            CUDA memory statistics in GB, using the same helpers as img2img.
            Omitted on CPU.
        ``metadata`` : str
            Path to the JSON sidecar, added to the returned payload after writing.
        ``debug`` : dict, optional
            Present only with ``debug.save_debug=True``. Contains:

            ``debug_directory`` : str
                Directory containing the diagnostic PNG and NPZ files.
            ``source_views``, ``target``, ``mixed`` : list[dict] or dict
                Source/target/transferred reconstruction records, including
                image paths where applicable, coefficients by group, alignment
                transforms, and artifact paths in each record's ``debug`` field.
                Neutral renders use fixed studio lighting; color renders use
                each reconstruction's inferred lighting, with target lighting
                for the mixed face.
            ``mean_source_coefficients`` : dict[str, list[float]]
                Averaged source groups, including pose/light groups retained
                solely for diagnosis rather than transferred to the target.
            ``warp`` : dict
                Shared radial center, ray count, transition width, maximum
                displacement, and minimum radial Jacobian; empty for an identity
                warp. Includes blocked-ray count when an active occlusion exists.
            ``occlusion_mask`` : str, optional
                PNG of the resized grayscale protection mask, retaining soft edges.
                Present only when an occlusion mask is connected.

    Notes
    -----
    All source views must depict the same person. Source coefficients are
    averaged equally; target pose, translation, and lighting are never averaged
    into that identity. Without debug, source meshes are neither decoded nor
    rendered; the target mask and mixed color render remain necessary.

    Composition requires the mask-union center inside both silhouettes,
    including the eroded new face. Each ray uses the furthest original-boundary
    hit and the nearest eroded-destination hit to contain the old silhouette
    inside the replacement. Multiple intersections from concavities are allowed;
    invalid placement or missing intersections still raise an error.

    An optional occlusion mask is applied after contour sampling and destination
    erosion, leaving the full mesh masks available for geometric alignment. A
    ray is blocked only when its selected eroded-destination point is protected;
    adjacent valid directions taper toward identity. An occluded original
    contour point does not block deformation toward a free destination. The
    full face is pasted, then the unwarped original is layered over it using
    the grayscale protection mask as alpha, exactly once. This preserves
    occluders such as clothes or shoulders without changing inferred
    geometry. It does not prevent inverse sampling of occluder colors elsewhere.

    Examples
    --------
    Connect images using the standard target edge and source attachment::

        target >> reconstruction
        source_views >> reconstruction.source()

    Optionally protect foreground occluders using a white-on-black mask::

        occlusion_mask >> reconstruction.occlusion()

    A minimal specification is::

        {
            'model': {
                'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
            },
            'params': {'preserve': ['albedo']},
        }
    """

    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        """Return whether reconstruction uses CUDA."""
        cfg = _read_cfg(resolve_spec(self.spec), self.id)
        return is_cuda_device(cfg.device)

    def source(self) -> AttachmentSink:
        """
        Declare the source-identity attachment, separate from the target input.

        Connect one image or several views of the same person. Each detected
        face supplies coefficients in the shared 3DDFA-V3 basis; their mean
        determines identity and any additional groups selected in ``preserve``.
        The default input supplies target pose, placement, and illumination.
        Images with no detected source face are reported and skipped; at least
        one usable source is required.

        Examples
        --------
        Supply the target through the normal edge and source views through
        this dedicated attachment::

            target >> reconstruction
            source_front >> reconstruction.source()
            source_oblique >> reconstruction.source()

        A FileImage node producing several image paths can supply all views::

            source_images >> reconstruction.source()

        Returns
        -------
        AttachmentSink
            Source sink accepting one or more views of the same person.
        """
        return AttachmentSink(
            name=f'source:{self.id}',
            target=self,
            input_id='source',
        )

    def occlusion(self) -> AttachmentSink:
        """
        Declare an optional target-space mask of foreground occluders.

        Use a segmentation of the actual target image to protect regions such
        as clothes, shoulders, hands, or hair. White preserves original target
        pixels fully; black permits modification; gray provides proportional
        protection. Supply soft edges using smoothing in an upstream mask node.
        The mask must share the target framing. Differing sizes are resized by
        nearest neighbor, retaining the supplied grayscale levels.

        The mask describes protected pixels, not an instruction to create an
        occlusion. For example, an upper-body segmentation behaves as follows:

        - If the body covers the chin in the target, its selected pixels remain
          in front of the replacement face. The reconstructed chin cannot be
          pasted over them: original body pixels are restored after blending.
        - If the selected body region does not overlap the replacement face,
          the face remains visible there. The body pixels are still protected;
          the mask does not force the chin behind a body that is elsewhere.
        - If a larger reconstructed face extends into the selected body region,
          the body also covers that new overlap, according to the supplied mask.

        Protection also matters near the face, outside the render's support.
        The background warp may reach a nearby shoulder or clothing even when
        it does not overlap the inserted face. Selected pixels remain unchanged
        rather than being carried along by that deformation.

        Geometric face masks remain complete for center and contour sampling.
        After sampling, a ray uses identity deformation only if its selected
        eroded-destination mask value exceeds ``params.occlusion_threshold``.
        The default 0 blocks every nonblack sample; raising the threshold ignores
        fainter portions of soft edges for ray blocking only. Neighboring valid
        directions taper toward blocked ones. An original contour point in
        the occlusion does not block the ray: the old face must still deform
        toward a free new contour, even when hair touches the original boundary.
        A protected region away from the destination endpoint does not by itself
        block the entire ray, but its own pixels are still preserved.

        Composition uses three layers in order: the warped target, the complete
        replacement face with its ordinary edge feathering, and the unwarped
        original target with this mask as alpha. The face is not cut by the
        occlusion mask. Protection alpha is applied only once, after pasting,
        independently of the ray-blocking threshold. Fully white regions are
        restored exactly; gray regions provide partial protection. Full geometry
        is still inferred; the mask controls visibility and warp directions.
        Incorrect segmentation remains the responsibility of the upstream mask.

        Notes
        -----
        This node uses the supplied mask directly; it does not classify selected
        pixels as body versus face or infer depth ordering. Protection applies to
        this node's composite, not automatically to downstream image generation.
        Restoring an occluder does not prevent inverse sampling from carrying
        some of its colors into other, unprotected pixels.

        Examples
        --------
        Connect a FileImage or another mask-producing node::

            foreground_mask >> reconstruction.occlusion()

        Returns
        -------
        AttachmentSink
            Sink bound to ``input_id='occlusion'``. Omit this connection to use
            the normal unoccluded composition path.
        """
        return AttachmentSink(
            name=f'occlusion:{self.id}',
            target=self,
            input_id='occlusion',
        )

    def run(self, output_dir: str | Path, input: Optional[dict] = None) -> dict:
        cfg = _read_cfg(resolve_spec(self.spec), self.id)

        target_path = resolve_single_image_path(
            node_id=self.id,
            path=None,
            input=input,
        )
        sources = resolve_image_paths(
            node_id=self.id,
            path=None,
            input=input,
            input_key='source',
        )

        occlusion_path = None
        if input is not None and input.get('occlusion') is not None:
            occlusion_path = resolve_single_image_path(
                node_id=self.id, path=None, input=input, input_key='occlusion',
            )

        detector = get_mediapipe_face_landmarker(
            model_asset_path=cfg.face_landmarker_task, device='cpu',
        )
        runtime = get_face_reconstruction_model(
            device=cfg.device, repo_id=cfg.repo_id, revision=cfg.revision,
        )

        cuda_prerun(cfg.device)
        t0 = time.perf_counter()

        target_rgb = _load_rgb(target_path)
        occlusion = (
            _load_occlusion(occlusion_path, target_rgb.shape)
            if occlusion_path is not None else None
        )
        target_landmarks = mp_face_landmarks_xyz(
            img_rgb=target_rgb, face_landmarker=detector,
        )

        out_path = make_node_output_path(
            out_dir=Path(output_dir), node_id=self.id, ext='png',
        )
        debug_dir = out_path.with_name(out_path.stem + '_debug')

        if cfg.save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        # Only target visibility and transferred color are needed to compose.
        # Neutral shading and all source geometry are diagnostic work.
        target_fields = ('color', 'shape') if cfg.save_debug else ()

        # The target provides pose, placement, lighting, and groups not selected
        # from source. Its rendered silhouette also tells composition which
        # original-face region must fit beneath the transferred face. Keep the
        # crop transforms to map both meshes into this same full-image frame.
        target_alignment = align_reconstruction_face(
            target_rgb, target_landmarks, runtime.reference,
        )
        target_coefficients = runtime.infer(target_alignment.rgb)
        target_decoded = runtime.decode(
            target_coefficients, render_fields=target_fields,
        )
        target_raster = render_reconstruction(
            target_decoded, runtime.triangles, render_fields=target_fields,
        )
        target_render = reconstruction_to_image(
            target_raster, target_alignment, target_rgb.shape,
        )
        target_view = {'image': str(target_path)}

        if cfg.save_debug:
            target_view.update(_save_reconstruction(
                debug_dir, 'target', target_rgb, target_alignment,
                target_coefficients, target_decoded, runtime.triangles,
                target_raster, target_render,
            ))

        coefficients, source_views, rejected = [], [], []
        source_debug_views = []

        # Every usable source contributes one coefficient vector to the mean.
        # Detection failures are recorded without discarding the other views.
        # Source geometry is unnecessary for transfer: decode/render/save it
        # only for visual comparison when debug is enabled.
        for path in sources:
            rgb = _load_rgb(path)

            try:
                landmarks = mp_face_landmarks_xyz(
                    img_rgb=rgb, face_landmarker=detector,
                )
            except RuntimeError as exc:
                rejected.append({'image': str(path), 'reason': str(exc)})
                continue

            alignment = align_reconstruction_face(
                rgb, landmarks, runtime.reference,
            )

            values = runtime.infer(alignment.rgb)
            view = {'image': str(path)}

            if cfg.save_debug:
                decoded = runtime.decode(values)
                raster = render_reconstruction(decoded, runtime.triangles)
                mapped = reconstruction_to_image(raster, alignment, rgb.shape)
                source_debug_views.append({
                    **view,
                    **_save_reconstruction(
                        debug_dir, f'source_{len(source_views) + 1}', rgb, alignment,
                        values, decoded, runtime.triangles, raster, mapped,
                    ),
                })
            source_views.append(view)
            coefficients.append(values)

        if not coefficients:
            raise RuntimeError(
                f"FaceReconstruction node '{self.id}': no source face could be detected"
            )

        # Transfer groups in coefficient space, preserving target pose and
        # lighting. Render the mixed vector using the target crop's camera frame.
        mean, mixed = transfer_reconstruction_parameters(
            coefficients, target_coefficients, cfg.preserve,
        )

        mixed_fields = ('color', 'shape') if cfg.save_debug else ('color',)
        mixed_decoded = runtime.decode(mixed, render_fields=mixed_fields)

        mixed_raster = render_reconstruction(
            mixed_decoded, runtime.triangles, render_fields=mixed_fields,
        )
        mixed_render = reconstruction_to_image(
            mixed_raster, target_alignment, target_rgb.shape,
        )
        composite, composition = compose_reconstructed_face(
            target_rgb, target_render['mask'], mixed_render,
            occlusion_mask=occlusion,
            color_transfer=cfg.color_transfer,
            occlusion_threshold=cfg.occlusion_threshold,
            collect_diagnostics=cfg.save_debug,
        )

        Image.fromarray(composite).save(out_path)

        # Warp statistics are diagnostic only; composition keeps its operational
        # radii in both modes. All optional additions belong under one debug key.
        warp_diagnostics = composition.pop('warp')
        debug = {}
        if cfg.save_debug:
            om = {}
            if occlusion is not None:
                occlusion_debug_path = debug_dir / 'occlusion_mask.png'
                Image.fromarray(
                    np.rint(occlusion * 255).astype(np.uint8)
                ).save(occlusion_debug_path)
                om = {'occlusion_mask': str(occlusion_debug_path)}

            debug = {
                'debug': {
                    'debug_directory': str(debug_dir),
                    'source_views': source_debug_views,
                    'target': target_view,
                    'mixed': _save_reconstruction(
                        debug_dir, 'mixed', target_rgb, target_alignment, mixed,
                        mixed_decoded, runtime.triangles, mixed_raster, mixed_render,
                    ),
                    'warp': warp_diagnostics,
                    'mean_source_coefficients': {
                        name: mean[indices].tolist()
                        for name, indices in COEFFICIENT_SLICES.items()
                    },
                    **om,
                }
            }

        # Stop after every diagnostic write, including the mixed NPZ/render.
        # Synchronization makes elapsed time include pending CUDA work.
        cuda_sync(cfg.device)
        dt_s = time.perf_counter() - t0
        mem = cuda_mem_stats(cfg.device)
        cuda_mem = {}
        if mem:
            cuda_mem = {'cuda_mem': mem}

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'image': str(out_path),
            'output_kind': 'reconstruction_composite',
            'composite': str(out_path),
            'target_image': str(target_path),
            'source_images': [view['image'] for view in source_views],
            'source_views': source_views,
            'rejected_source_images': rejected,
            'parameter_origins': {
                name: (
                    'source_mean'
                    if name == 'identity' or name in cfg.preserve else 'target'
                )
                for name in COEFFICIENT_SLICES
            },
            'model': {
                'id': cfg.repo_id,
                'revision': cfg.revision,
                'device': cfg.device,
                'face_landmarker_task': cfg.face_landmarker_task,
                'assets': {name: str(path) for name, path in runtime.assets.items()},
            },
            'params': {
                'preserve': list(cfg.preserve),
                'color_transfer': cfg.color_transfer,
                'occlusion_threshold': cfg.occlusion_threshold,
            },
            'composition': composition,
            **({'occlusion_image': str(occlusion_path)} if occlusion_path is not None else {}),
            'timing': {'seconds': round(dt_s, 3)},
            **cuda_mem,
            **debug,
        }
        out['metadata'] = str(write_json_sidecar(out_path, out))

        return out
