import torch
from diffusers.utils import *

from stability.cache.ltx_models import *
from stability.core.config import *
from stability.lightricks.pipeline_ltx_condition_control import LTXVideoCondition
from stability.nodes import *
from stability.nodes.ltx import *

_default_model = 'Lightricks/LTX-Video-0.9.7-dev'
_default_upscaler = 'Lightricks/ltxv-spatial-upscaler-0.9.7'


def resolve_ltx_common(source_spec: Union[Dict[str, Any], str, Path]):
    spec = resolve_spec(source_spec)

    # model settings
    model = spec.get('model', {})
    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))
    model_id = model.get('id', _default_model)
    upscaler_id = model.get('upscaler', _default_upscaler)

    # params
    params = spec.get('params', {})
    guidance_scale = float(params.get('guidance_scale', 3))
    guidance_rescale = float(params.get('guidance_rescale', 0.7))

    height = int(params.get('height', 480))
    width = int(params.get('width', 832))
    fps = int(params.get('fps', 24))
    num_frames = int(params.get('num_frames', 96))

    steps_main = int(params.get('steps_main', 30))
    steps_refine = int(params.get('steps_refine', 10))
    denoise_strength = float(params.get('denoise_strength', 0.4))
    downscale_factor = float(params.get('downscale_factor', 2 / 3))
    upscale_factor = float(params.get('upscale_latent_factor', 2))

    decode_timestep = float(params.get('decode_timestep', 0.05))
    decode_noise_scale = float(params.get('decode_noise_scale', 0.025))
    image_cond_noise_scale = float(params.get('image_cond_noise_scale', 0.025))

    max_sequence_length = int(params.get('max_sequence_length', 512))

    # seed
    seed = resolve_seed(spec.get('seed', 'random'))
    gen = torch.Generator(device=device).manual_seed(seed)

    return spec, device, dtype, model_id, upscaler_id, guidance_scale, \
        guidance_rescale, height, width, fps, num_frames, steps_main, \
        steps_refine, denoise_strength, downscale_factor, upscale_factor, \
        decode_timestep, decode_noise_scale, image_cond_noise_scale, \
        max_sequence_length, seed, gen


def finalize_video_output(
    *,
    node_kind: str,
    node_id: str,
    video_path: Path,
    seed: int,
    params: dict,
    dt_s: float,
    cuda_mem: dict,
    input_image: Optional[str] = None,
    input_video: Optional[str] = None,
    ic_lora_specs: Optional["ICLoRaSpec"] = None,
    model_info: Optional[dict] = None,
) -> dict:
    out = {
        'ok': True,
        'node': node_kind,
        'id': node_id,
        'video': str(video_path),
        'seed': seed,
        'params': params,
        'timing': {'seconds': round(dt_s, 3)},
        'cuda_mem': cuda_mem,
    }

    if input_image is not None:
        out['input_image'] = input_image

    if input_video is not None:
        out['input_video'] = input_video

    if ic_lora_specs is not None:
        out['ic_lora'] = {
            'model_id': ic_lora_specs.model_id,
            'weight_name': ic_lora_specs.weight_name,
            'adapter_name': ic_lora_specs.adapter_name,
            'adapter_weight': ic_lora_specs.adapter_weight,
        }

    if model_info is not None:
        out['model'] = model_info

    meta_path = write_json_sidecar(video_path, out)
    out['metadata'] = str(meta_path)
    return out


def downscale_size(height: int, width: int, factor: float, pipe: LTXConditionPipeline):
    down_h = int(height * factor)
    down_w = int(width * factor)

    return round_to_vae(down_h, down_w, pipe)


def upscale_size(height: int, width: int, factor):
    return int(height * factor), int(width * factor)


class IcLoRaMixin:
    def __post_init__(self):
        super().__post_init__()
        self.ic_lora = ICLoRaRegistry(self)

    def build_ic_lora_bundle(self, input):
        return ICLoRaBundle(
            ic_lora=self.ic_lora.spec,
            input=input
        )


def apply_ic_lora(ic_lora_bundle: ICLoRaBundle, pipe: LTXConditionPipeline, device: str) -> Optional[torch.Tensor]:
    if ic_lora_bundle.has_has_ic_lora:
        pipe.load_lora_weights(
            ic_lora_bundle.model_id,
            weight_name=ic_lora_bundle.weight_name,
            adapter_name=ic_lora_bundle.adapter_name
        )

        pipe.set_adapters(
            [ic_lora_bundle.adapter_name],
            [ic_lora_bundle.adapter_weight]
        )

        control_frames = load_video(ic_lora_bundle.video_path)
        return read_video_tensor(control_frames, device=device)
    else:
        pipe.unload_lora_weights()
        return None


def build_image_condition(*, image_path: str, frame_index: int) -> LTXVideoCondition:
    """
    Build an LTX one-frame video condition from a single image.

    The current LTX conditioning API expects a video tensor/sequence. A practical
    workaround is to serialize a one-frame video and reload it as a video object.
    This mirrors your existing approach in the legacy Img2Video node.

    Parameters
    ----------
    image_path : str
        Path to the input keyframe image.
    frame_index : int
        Index of the frame in the condition sequence to anchor the conditioning.

    Returns
    -------
    LTXVideoCondition
        Condition object compatible with ``LTXConditionPipeline``.
    """
    image = load_image(image_path)
    video_1frame = load_video(export_to_video([image]))

    return LTXVideoCondition(video=video_1frame, frame_index=int(frame_index))
