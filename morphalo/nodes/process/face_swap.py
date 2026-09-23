from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from morphalo.cache.models import (
    FaceSwapperRuntime,
    get_face_swapper,
    get_insightface,
)
from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_image_paths, resolve_single_image_path


@dataclass(frozen=True)
class Config:
    """Resolved FaceSwap model configuration."""

    device: str
    model_name: str
    det_size: Tuple[int, int]
    swapper: str
    save_debug: bool


def _read_cfg(spec: dict, node_id: str) -> Config:
    """
    Resolve and validate the FaceSwap model configuration.

    Parameters
    ----------
    spec : dict
        Resolved node specification.
    node_id : str
        Node identifier included in validation errors.

    Returns
    -------
    Config
        Validated model configuration.

    Raises
    ------
    ValueError
        If the model or debug blocks, detection size, or swapper name is
        invalid.
    """
    model = spec.get('model', {})
    debug = spec.get('debug', {})

    if not isinstance(model, dict):
        raise ValueError(f"'{node_id}': model must be an object")

    if not isinstance(debug, dict):
        raise ValueError(f"'{node_id}': debug must be an object")

    det_size = model.get('det_size', (640, 640))

    if not isinstance(det_size, (list, tuple)) or len(det_size) != 2:
        raise ValueError(
            f"'{node_id}': invalid det_size={det_size!r} "
            '(expected [w, h] or (w, h))'
        )

    det_size = (int(det_size[0]), int(det_size[1]))

    if det_size[0] <= 0 or det_size[1] <= 0:
        raise ValueError(f"'{node_id}': det_size values must be positive")

    swapper = str(model.get('swapper', 'simswap_256'))
    supported = ('simswap_256', 'simswap_unofficial_512')

    if swapper not in supported:
        raise ValueError(
            f"'{node_id}': unsupported swapper={swapper!r} "
            f'(expected one of {supported!r})'
        )

    return Config(
        device=str(model.get('device', 'cpu')),
        model_name=str(model.get('model_name', 'buffalo_l')),
        det_size=det_size,
        swapper=swapper,
        save_debug=bool(debug.get('save_debug', False)),
    )


def _load_image_bgr(path: Path) -> np.ndarray:
    """
    Load an image as an 8-bit BGR array.

    Parameters
    ----------
    path : pathlib.Path
        Path to the input image.

    Returns
    -------
    numpy.ndarray
        Image with shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    import cv2

    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _select_largest_face(faces: Sequence[Any]) -> Any:
    """
    Select the most prominent face by detector bounding-box area.

    Parameters
    ----------
    faces : sequence[Any]
        Detected InsightFace face objects exposing ``bbox`` in XYXY format.

    Returns
    -------
    Any
        Face with the largest non-negative bounding-box area.

    Raises
    ------
    ValueError
        If no faces are provided.
    """
    if not faces:
        raise ValueError('Cannot select a face from an empty sequence')

    return max(
        faces,
        key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0]))
        * max(0.0, float(face.bbox[3] - face.bbox[1])),
    )


def _mean_embedding(faces: Sequence[Any]) -> np.ndarray:
    """
    Average native ArcFace embeddings across source faces.

    Parameters
    ----------
    faces : sequence[Any]
        Source face objects exposing compatible ``embedding`` vectors.

    Returns
    -------
    numpy.ndarray
        Mean ArcFace embedding as a one-dimensional ``float32`` array.

    Raises
    ------
    ValueError
        If the sequence is empty or contains non-finite or inconsistent
        embeddings.
    """
    # Keep the native ArcFace embeddings here. CrossFace performs the conversion
    # to SimSwap's identity space after the identities have been averaged.
    embeddings = []
    for face in faces:
        embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(embedding)):
            raise ValueError('Source face embedding contains non-finite values')

        embeddings.append(embedding)

    if not embeddings:
        raise ValueError('No source face embeddings were provided')

    if len({embedding.shape for embedding in embeddings}) != 1:
        raise ValueError('Source face embeddings have inconsistent dimensions')

    return np.mean(np.stack(embeddings), axis=0, dtype=np.float32)


def _run_crossface_simswap(session: Any, embedding: np.ndarray) -> np.ndarray:
    """
    Convert an ArcFace embedding into SimSwap identity space.

    Parameters
    ----------
    session : Any
        ONNX Runtime session for ``crossface_simswap.onnx``.
    embedding : numpy.ndarray
        ArcFace identity embedding with shape ``(D,)``.

    Returns
    -------
    numpy.ndarray
        Unit-normalized SimSwap identity tensor with shape ``(1, D_out)``.

    Raises
    ------
    RuntimeError
        If the converter contract is invalid or inference produces an invalid
        identity vector.
    """
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            f'CrossFace model must expose one input, found {len(inputs)}'
        )

    tensor = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    output = np.asarray(
        session.run(None, {inputs[0].name: tensor})[0],
        dtype=np.float32,
    ).reshape(1, -1)
    norm = np.linalg.norm(output, axis=1, keepdims=True)

    if np.any(norm <= 1e-8) or not np.all(np.isfinite(norm)):
        raise RuntimeError('CrossFace produced an invalid identity embedding')

    # SimSwap expects a unit-length identity vector. Normalize only after the
    # averaged ArcFace identity has been converted into SimSwap's space.

    return output / norm


_ARCFACE_112_V1 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _align_face_crop(
    image_bgr: np.ndarray,
    target_face: Any,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a target face to the canonical ArcFace geometry.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Full target image in BGR format.
    target_face : Any
        InsightFace face object exposing five landmarks through ``kps``.
    crop_size : int
        Width and height required by the selected swapper.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Aligned square BGR crop and the ``2 x 3`` affine matrix mapping the
        original image into crop coordinates.

    Raises
    ------
    ValueError
        If the target face does not provide five finite landmarks.
    RuntimeError
        If the affine transform cannot be estimated.
    """
    import cv2

    keypoints = np.asarray(getattr(target_face, 'kps', None), dtype=np.float32)

    if keypoints.shape != (5, 2) or not np.all(np.isfinite(keypoints)):
        raise ValueError('Target face must provide five finite alignment keypoints')

    # The canonical landmarks are defined in ArcFace's 112x112 coordinate
    # system. Scale them to the input resolution required by the swapper.
    reference = _ARCFACE_112_V1 * (float(crop_size) / 112.0)

    # This matrix maps coordinates from the original target image into the
    # aligned crop. Paste-back must therefore use its inverse.
    matrix, _ = cv2.estimateAffinePartial2D(
        keypoints,
        reference,
        method=cv2.LMEDS,
    )

    if matrix is None or matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise RuntimeError('Could not estimate target face alignment transform')

    crop = cv2.warpAffine(
        image_bgr,
        matrix,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return crop, matrix.astype(np.float32)


def _preprocess_target_crop(
    crop_bgr: np.ndarray,
    runtime: FaceSwapperRuntime,
) -> np.ndarray:
    """
    Convert an aligned BGR crop into a SimSwap input tensor.

    Parameters
    ----------
    crop_bgr : numpy.ndarray
        Aligned face crop in BGR format.
    runtime : FaceSwapperRuntime
        Runtime contract defining input size and channel normalization.

    Returns
    -------
    numpy.ndarray
        Contiguous ``float32`` RGB tensor with shape ``(1, 3, H, W)``.

    Raises
    ------
    RuntimeError
        If the runtime contains an invalid normalization standard deviation.
    """
    import cv2

    if crop_bgr.shape[:2] != (runtime.size, runtime.size):
        crop_bgr = cv2.resize(
            crop_bgr,
            (runtime.size, runtime.size),
            interpolation=cv2.INTER_LINEAR,
        )

    # OpenCV and InsightFace operate in BGR, while the SimSwap models consume
    # normalized RGB tensors in NCHW layout.
    rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.asarray(runtime.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(runtime.std, dtype=np.float32).reshape(1, 1, 3)

    if np.any(std <= 0):
        raise RuntimeError('Face swapper normalization std must be positive')

    tensor = ((rgb - mean) / std).transpose(2, 0, 1)[None]

    return np.ascontiguousarray(tensor, dtype=np.float32)


def _run_simswap(
    session: Any,
    target_tensor: np.ndarray,
    source_embedding: np.ndarray,
) -> np.ndarray:
    """
    Run the SimSwap generator for one aligned target face.

    Parameters
    ----------
    session : Any
        ONNX Runtime session exposing ``source`` and ``target`` inputs.
    target_tensor : numpy.ndarray
        Preprocessed target face with shape ``(1, 3, H, W)``.
    source_embedding : numpy.ndarray
        Converted source identity tensor with shape ``(1, D)``.

    Returns
    -------
    numpy.ndarray
        Generated face crop in BGR format with dtype ``uint8``.

    Raises
    ------
    RuntimeError
        If the model input contract or generated output is invalid.
    """
    # Bind inputs by semantic name rather than by ONNX declaration order. The
    # published SimSwap variants do not need to expose them in the same order.
    inputs = session.get_inputs()
    by_name = {item.name.lower(): item.name for item in inputs}
    source_name = by_name.get('source')
    target_name = by_name.get('target')

    if source_name is None or target_name is None or len(inputs) != 2:
        names = [item.name for item in inputs]
        raise RuntimeError(
            "SimSwap model must expose exactly 'source' and 'target' inputs; "
            f'found {names!r}'
        )

    output = np.asarray(
        session.run(
            None,
            {
                source_name: np.ascontiguousarray(source_embedding, dtype=np.float32),
                target_name: np.ascontiguousarray(target_tensor, dtype=np.float32),
            },
        )[0]
    )

    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]

    if output.ndim != 3 or output.shape[0] != 3:
        raise RuntimeError(f'Unexpected SimSwap output shape: {output.shape!r}')

    if not np.all(np.isfinite(output)):
        raise RuntimeError('SimSwap produced non-finite pixels')

    # ONNX returns an RGB CHW float tensor. Convert it back to the BGR uint8
    # representation used by the geometry and compositing stages.
    rgb = np.clip(output.transpose(1, 2, 0), 0.0, 1.0)

    return np.rint(rgb[:, :, ::-1] * 255.0).astype(np.uint8)


def _create_crop_mask(crop_size: int) -> np.ndarray:
    """
    Create a feathered square mask for paste-back compositing.

    Parameters
    ----------
    crop_size : int
        Width and height of the aligned face crop.

    Returns
    -------
    numpy.ndarray
        Floating-point alpha mask with values in ``[0, 1]``.
    """
    import cv2

    # Exclude the unreliable crop border, then feather the transition so that
    # the swapped face blends into the target skin and surrounding context.
    margin = max(4, int(round(crop_size * 0.08)))
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    mask[margin:crop_size - margin, margin:crop_size - margin] = 1.0
    sigma = max(1.0, crop_size * 0.02)

    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _paste_back_face(
    original_bgr: np.ndarray,
    swapped_crop_bgr: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """
    Warp a generated face back and blend it into the target image.

    Parameters
    ----------
    original_bgr : numpy.ndarray
        Original target image in BGR format.
    swapped_crop_bgr : numpy.ndarray
        Generated square face crop in aligned coordinates.
    affine_matrix : numpy.ndarray
        Forward affine transform from original image to aligned crop.

    Returns
    -------
    numpy.ndarray
        Composited full-resolution BGR image with dtype ``uint8``.

    Raises
    ------
    ValueError
        If the generated crop is not a square three-channel image.
    """
    import cv2

    height, width = original_bgr.shape[:2]
    crop_size = swapped_crop_bgr.shape[0]

    if swapped_crop_bgr.shape != (crop_size, crop_size, 3):
        raise ValueError('Swapped face crop must be a square HxWx3 image')

    # Alignment mapped original -> crop; inverse warping returns both the
    # generated face and its mask to the original image coordinate system.
    inverse = cv2.invertAffineTransform(np.asarray(affine_matrix, dtype=np.float32))
    warped_face = cv2.warpAffine(
        swapped_crop_bgr,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_mask = cv2.warpAffine(
        _create_crop_mask(crop_size),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    alpha = np.clip(warped_mask, 0.0, 1.0)[:, :, None]
    blended = (
        warped_face.astype(np.float32) * alpha
        + original_bgr.astype(np.float32) * (1.0 - alpha)
    )

    return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)


def _make_paste_back_debug(
    image_bgr: np.ndarray,
    affine_matrix: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    """
    Draw the inverse-warped crop boundary on a copy of the output image.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Composited output image in BGR format.
    affine_matrix : numpy.ndarray
        Forward affine transform from original image to aligned crop.
    crop_size : int
        Width and height of the aligned crop.

    Returns
    -------
    numpy.ndarray
        Copy of the output image with the crop quadrilateral highlighted.
    """
    import cv2

    inverse = cv2.invertAffineTransform(
        np.asarray(affine_matrix, dtype=np.float32)
    )
    crop_corners = np.array(
        [
            [0.0, 0.0],
            [float(crop_size - 1), 0.0],
            [float(crop_size - 1), float(crop_size - 1)],
            [0.0, float(crop_size - 1)],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    image_corners = cv2.transform(crop_corners, inverse)[0]
    polygon = np.rint(image_corners).astype(np.int32).reshape(-1, 1, 2)

    debug_bgr = image_bgr.copy()
    line_width = max(2, int(round(min(image_bgr.shape[:2]) * 0.004)))
    cv2.polylines(
        debug_bgr,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )

    return debug_bgr


@dataclass
class FaceSwap(NodeRef):
    """
    Replace the face in a target image with an identity from a source image.

    ``FaceSwap`` modifies the face of the image received through the default
    input channel. The target image supplies the pose, expression, lighting,
    body, and surrounding scene. A separate source image supplies the identity
    that is transferred onto the target face.

    Wire the node producing the target image directly into ``FaceSwap``. Wire
    the node producing the identity reference into the attachment sink returned
    by :meth:`source`. The node detects the largest face in both images, extracts
    the source identity, generates a replacement aligned to the target face, and
    blends it back into the original target image.

    Multiple images may be connected to :meth:`source`. In that case the node
    selects the largest face from each image and averages their identity
    embeddings before performing the swap. Images without a detectable source
    face are ignored; execution fails if none of the source images contains a
    usable face.

    Examples
    --------
    Wire the target image through the normal DAG edge and the identity image
    through the dedicated source attachment::

        target_image >> face_swap
        identity_image >> face_swap.source()

    Inputs
    ------
    default : dict
        Target payload containing ``image`` or ``path``.
    source : dict
        One or more source payloads containing ``image`` or ``path``.

    Configuration
    -------------
    model.model_name : str, default='buffalo_l'
        InsightFace model pack used for detection and ArcFace embeddings.
    model.device : str, default='cpu'
        Device used by InsightFace and ONNX Runtime.
    model.det_size : tuple[int, int], default=(640, 640)
        Detection canvas size passed to InsightFace.
    model.swapper : {'simswap_256', 'simswap_unofficial_512'}, default='simswap_256'
        Face swap model loaded through the shared model cache.
    debug.save_debug : bool, default=False
        Save a separate image highlighting the inverse-warped crop boundary.

    Outputs
    -------
    dict
        Output image path, source and target paths, resolved model metadata,
        and JSON sidecar path. ``debug_paste_box`` is included when
        ``debug.save_debug`` is enabled.

    Notes
    -----
    The internal pipeline is:

    1. Load the target and source images in BGR format and analyse them with a
       shared, cached InsightFace instance.
    2. Select the largest detected face in the target and in each usable source
       image. Only one target face is modified.
    3. Average the native ArcFace embeddings from the selected source faces,
       then convert the averaged embedding into SimSwap identity space with the
       CrossFace ONNX model. The converted identity vector is L2-normalized.
    4. Estimate a similarity transform from the target's five facial landmarks
       to the canonical ArcFace landmark template scaled to the swapper input
       resolution. This produces the aligned target crop and the forward affine
       matrix used later for compositing.
    5. Convert the aligned BGR crop to RGB, apply the normalization associated
       with the selected SimSwap backend, and run the swapper with the converted
       source identity.
    6. Convert the generated crop back to BGR, inverse-warp it into the original
       target coordinates, and blend it with a feathered mask that excludes the
       unreliable crop border.

    Model assets and ONNX sessions are resolved through the shared model cache.
    Their paths, tensor normalization, and input resolution are backend details
    and are not exposed as node parameters.

    """

    spec: SpecInput = field(default_factory=dict)

    def source(self) -> AttachmentSink:
        """
        Create the source-identity attachment sink.

        Returns
        -------
        AttachmentSink
            Sink bound to the node's ``source`` input channel.
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
        Execute the face swap pipeline.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Base directory where node artifacts are written.
        input : dict[str, dict] or None, optional
            Upstream payloads keyed by input channel.

        Returns
        -------
        dict[str, Any]
            Output payload containing the generated image and metadata paths.

        Raises
        ------
        ValueError
            If required images are missing or no usable face is detected.
        RuntimeError
            If model loading, inference, alignment, or compositing fails.
        """
        import cv2

        # 1. Resolve the target image and one or more references for the source
        # identity. Model file paths remain an implementation detail of the cache.
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

        if not source_paths:
            raise ValueError(f"FaceSwap node '{self.id}': no source images provided")

        # 2. Use one cached InsightFace analyser for target detection and source
        # embeddings, ensuring that all identity vectors share the same space.
        target_bgr = _load_image_bgr(target_path)
        analyser = get_insightface(
            model_name=cfg.model_name,
            det_size=cfg.det_size,
            device=cfg.device,
        )
        target_faces = analyser.get(target_bgr)

        if not target_faces:
            raise ValueError(
                f"FaceSwap node '{self.id}': no target face detected in {target_path}"
            )

        # Face selection is deliberately fixed to the largest detected face for
        # the initial node contract. It is explicit here but not user-configurable.
        target_face = _select_largest_face(target_faces)
        source_faces = []
        for source_path in source_paths:
            faces = analyser.get(_load_image_bgr(source_path))
            if faces:
                source_faces.append(_select_largest_face(faces))

        if not source_faces:
            raise ValueError(f"FaceSwap node '{self.id}': no source face detected")

        # 3. Average the selected source identities, then convert the resulting
        # ArcFace vector into the identity representation expected by SimSwap.
        runtime = get_face_swapper(model_name=cfg.swapper, device=cfg.device)
        identity = _run_crossface_simswap(
            runtime.converter,
            _mean_embedding(source_faces),
        )

        # 4. Align the target face to the model's canonical geometry, preprocess
        # it according to the selected backend, and run the swap network.
        aligned, matrix = _align_face_crop(target_bgr, target_face, runtime.size)
        target_tensor = _preprocess_target_crop(aligned, runtime)
        swapped = _run_simswap(runtime.swapper, target_tensor, identity)

        # 5. Return the generated crop to the target image using the inverse
        # alignment transform and a feathered mask.
        output_bgr = _paste_back_face(target_bgr, swapped, matrix)

        # 6. Persist the image and a reproducibility sidecar. Selection strategy
        # is intentionally omitted because it is not part of the public config.
        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='png',
        )
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(output_rgb).save(out_path)

        debug_path = None
        if cfg.save_debug:
            debug_bgr = _make_paste_back_debug(
                output_bgr,
                matrix,
                runtime.size,
            )
            debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
            debug_path = out_path.with_name(
                out_path.stem + '_debug_paste_box.png'
            )
            Image.fromarray(debug_rgb).save(debug_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'image': str(out_path),
            'source_images': [str(path) for path in source_paths],
            'target_image': str(target_path),
            'model': {
                'model_name': cfg.model_name,
                'device': cfg.device,
                'det_size': list(cfg.det_size),
                'swapper': cfg.swapper,
            },
        }

        if debug_path is not None:
            out['debug_paste_box'] = str(debug_path)

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
