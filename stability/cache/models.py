
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from diffusers import (AutoencoderKL, ControlNetModel, DiffusionPipeline,
                       OmniGenPipeline, QwenImageEditPipeline,
                       StableDiffusionXLPipeline, T2IAdapter)
from insightface.app import FaceAnalysis
from transformers import (CLIPVisionModelWithProjection, DPTForDepthEstimation,
                          DPTImageProcessor, pipeline)
from ultralytics import YOLO as YOLOModel

from stability.cache import CacheKey, ModelCache
from stability.nodes.sdxl_resolve import ResolvedModelRef


def dtype_key(dtype: torch.dtype) -> str:
    # stable cache key string
    return str(dtype).replace('torch.', '')


def get_sdxl_base_pipe(
    *,
    model_ref: ResolvedModelRef,
    device: str,
    dtype: torch.dtype,
    vae_id: Optional[str] = None,
) -> StableDiffusionXLPipeline:

    # include vae_id in the cache key to avoid mismatches
    extra = f'vae={vae_id}' if vae_id else 'vae=<default>'
    key = CacheKey(
        kind=f'sdxl_base_pipe:{model_ref.source}',
        ref=model_ref.ref,
        device=device,
        dtype=dtype_key(dtype),
        extra=extra
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    vae = get_vae(
        vae_id=vae_id,
        device=device,
        dtype=dtype
    ) if vae_id else None

    pipe_kwargs = {'torch_dtype': dtype}
    if vae is not None:
        pipe_kwargs['vae'] = vae

    if model_ref.source == 'single_file':
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_ref.ref,
            **pipe_kwargs,
        ).to(device)
    elif model_ref.source == 'pretrained_id':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_ref.ref,
            **pipe_kwargs,
        ).to(device)
    else:
        raise ValueError(
            f"Invalid 'model_ref.source': {model_ref.source}"
        )

    return ModelCache.put(key, pipe)


def get_controlnet(*, model_id: str, device: str, dtype: torch.dtype) -> ControlNetModel:
    key = CacheKey(
        kind='controlnet',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    cn = ControlNetModel.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, cn)


def get_vae(*, vae_id: str, device: str, dtype: torch.dtype) -> AutoencoderKL:
    key = CacheKey(
        kind='vae',
        ref=vae_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    vae = AutoencoderKL.from_pretrained(
        vae_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, vae)


def get_depth_estimator(*, model_id: str, device: str) -> Tuple[DPTImageProcessor, DPTForDepthEstimation]:
    proc_key = CacheKey(
        kind='depth_processor',
        ref=model_id,
        device='cpu',
        dtype='na'
    )
    mod_key = CacheKey(
        kind='depth_model',
        ref=model_id,
        device=device,
        dtype='na'
    )

    processor = ModelCache.get(proc_key)
    if processor is None:
        processor = ModelCache.put(
            proc_key, DPTImageProcessor.from_pretrained(model_id)
        )

    model = ModelCache.get(mod_key)
    if model is None:
        model = DPTForDepthEstimation.from_pretrained(model_id).to(device)
        model.eval()
        ModelCache.put(mod_key, model)

    return processor, model


def get_ip_image_encoder(
    *,
    repo_id: str,
    subfolder: str,
    device: str,
    dtype: torch.dtype
) -> CLIPVisionModelWithProjection:
    key = CacheKey(
        kind='ip_image_encoder',
        ref=f'{repo_id}:{subfolder}',
        device=device,
        dtype=dtype_key(dtype)
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    enc = CLIPVisionModelWithProjection.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, enc)


def get_translator(
    *,
    model_id: str,
    source_lang: str,
    target_lang: str,
    device: str
) -> pipeline:
    key = CacheKey(
        kind='translator',
        ref=f'{model_id}:{source_lang}:{target_lang}',
        device=device,
        dtype='-'
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    translator = pipeline(
        task='translation',
        model=model_id,
        src_lang=source_lang,
        tgt_lang=target_lang,
        device=0 if device == 'cuda' else -1
    )

    return ModelCache.put(key, translator)


def get_t2i_adapter(*, model_id: str, device: str, dtype: torch.dtype) -> T2IAdapter:
    key = CacheKey(
        kind='t2i_adapter',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    adapter = T2IAdapter.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, adapter)


def get_controlnet_aux_annotator(
        *,
        processor: str,
        cls: Any, device: str,
        repo_id: str = 'lllyasviel/Annotators'
):
    """
    Cache wrapper for controlnet-aux annotators that support .from_pretrained(repo_id).

    Note: This only covers "checkpoint=True" annotators (HED, Midas, Openpose, etc.).
    """
    key = CacheKey(
        kind='controlnet_aux_annotator',
        ref=f'{processor}:{repo_id}',
        device=device,
        dtype='na'
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    proc = cls.from_pretrained(repo_id).to(device)
    return ModelCache.put(key, proc)


def get_omnigen(*, model_id: str, device: str, dtype: torch.dtype) -> OmniGenPipeline:
    key = CacheKey(
        kind='omnigen',
        ref=model_id,
        device=device,
        dtype=dtype_key(dtype)
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    omg = OmniGenPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    return ModelCache.put(key, omg)


def get_qwen_image(
    *,
    model_id: str,
    dtype: torch.dtype,
    device_map: str = 'balanced',
) -> DiffusionPipeline:
    """
    Load and cache a Qwen-Image Diffusers pipeline.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. "Qwen/Qwen-Image").
    dtype : torch.dtype
        Torch dtype for loading weights.
    device_map : str
        Accelerate device_map for dispatching modules. Typical values:
        "balanced", "auto", "cuda", "cpu".

    Returns
    -------
    DiffusionPipeline
        Cached pipeline instance.
    """
    key = CacheKey(
        kind='qwen_image',
        ref=model_id,
        device=str(device_map),
        dtype=dtype_key(dtype),
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    return ModelCache.put(key, pipe)


def get_qwen_image_edit_pipe(
    *,
    model_id: str,
    dtype: torch.dtype,
    device_map: str = 'balanced',
) -> QwenImageEditPipeline:
    """
    Load and cache a Qwen-Image-Edit Diffusers pipeline.
    """
    key = CacheKey(
        kind='qwen_image_edit',
        ref=model_id,
        device=str(device_map),
        dtype=dtype_key(dtype),
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    pipe = QwenImageEditPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    return ModelCache.put(key, pipe)


def get_yolo(*, model_name: str, device: str) -> YOLOModel:

    key = CacheKey(
        kind='yolo',
        ref=model_name,
        device=device,
        dtype='na'
    )

    cached = ModelCache.get(key)

    if cached is not None:
        return cached

    models_home = Path.home() / 'models'
    model_path = models_home / 'yolo' / model_name

    m = YOLOModel(model_path)

    return ModelCache.put(key, m)


def get_sam(
    *,
    checkpoint: str,
    model_type: str,
    device: str,
) -> torch.nn.Module:
    from segment_anything import sam_model_registry

    key = CacheKey(
        kind='sam',
        ref=f'{model_type}:{checkpoint}',
        device=device,
        dtype='na'
    )
    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()

    return ModelCache.put(key, sam)


def get_insightface(
    *,
    model_name: str,
    det_size: Tuple[int, int],
    device: str,
) -> FaceAnalysis:
    key = CacheKey(
        kind='insightface',
        ref=f'{model_name}:{int(det_size[0])}x{int(det_size[1])}',
        device=device,
        dtype='na'
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    if isinstance(device, str) and device.startswith('cuda'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # ctx_id: -1 for CPU, 0..N for GPU index
        ctx_id = 0
        if ':' in device:
            try:
                ctx_id = int(device.split(':', 1)[1])
            except Exception:
                ctx_id = 0
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1

    app = FaceAnalysis(
        name=model_name,
        providers=providers
    )
    app.prepare(
        ctx_id=ctx_id,
        det_size=(int(det_size[0]), int(det_size[1]))
    )

    return ModelCache.put(key, app)


def get_mediapipe_face_landmarker(
    *,
    model_asset_path: str,
    device: str,
):
    """
    Cached MediaPipe FaceLandmarker (Tasks API).

    Note:
    - Despite the historical function name, this loads a FaceLandmarker model
      (e.g. face_landmarker.task). It's used to get face landmarks (and derive a
      bbox from them), not FaceDetector.
    - `device` is kept for cache-key consistency with the rest of the project.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(str(model_asset_path)).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    key = CacheKey(
        kind="mediapipe_face_landmarker",
        ref=str(model_path),
        device=str(device),
        dtype="na",
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        # Keep these off unless you explicitly need them:
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return ModelCache.put(key, landmarker)


def get_mediapipe_pose_landmarker(
    *,
    model_asset_path: str,
    device: str,
):
    """
    Cached MediaPipe PoseLandmarker (Tasks API).

    Parameters
    ----------
    model_asset_path : str
        Path to the MediaPipe PoseLandmarker `.task` model file.
    device : str
        Logical device string kept for cache-key consistency with the rest
        of the project. MediaPipe Tasks does not currently use it here as
        an execution selector, but it is still part of the cache identity.

    Returns
    -------
    Any
        A cached MediaPipe `PoseLandmarker` instance configured for
        single-image inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.

    Notes
    -----
    This mirrors the caching strategy already used for
    `get_mediapipe_face_landmarker`.

    The landmarker is configured for image mode because `SubjectCrop`
    performs per-image analysis rather than video or live-stream inference.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(str(model_asset_path)).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f'MediaPipe model not found: {model_path}')

    key = CacheKey(
        kind='mediapipe_pose_landmarker',
        ref=str(model_path),
        device=str(device),
        dtype='na',
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False,
    )

    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return ModelCache.put(key, landmarker)


def get_mediapipe_hand_landmarker(
    *,
    model_asset_path: str,
    device: str,
):
    """
    Cached MediaPipe HandLandmarker (Tasks API).

    Parameters
    ----------
    model_asset_path : str
        Path to the MediaPipe HandLandmarker `.task` model file.
    device : str
        Logical device string kept for cache-key consistency with the rest
        of the project. MediaPipe Tasks does not currently use it here as
        an execution selector, but it is still part of the cache identity.

    Returns
    -------
    Any
        A cached MediaPipe `HandLandmarker` instance configured for
        single-image inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.

    Notes
    -----
    The landmarker is configured for image mode because the current scoring
    pipeline performs per-image analysis rather than video or live-stream
    inference.

    The configuration allows detecting up to two hands, which is the most
    natural choice for character images and portraits.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(str(model_asset_path)).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f'MediaPipe model not found: {model_path}')

    key = CacheKey(
        kind='mediapipe_hand_landmarker',
        ref=str(model_path),
        device=str(device),
        dtype='na',
    )

    cached = ModelCache.get(key)
    if cached is not None:
        return cached

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    return ModelCache.put(key, landmarker)
