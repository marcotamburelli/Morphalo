import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from stability.core.config import *
import os


# Hugging Face environment configuration
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
os.environ['HF_HUB_DISABLE_TELEMETRY'] = HF_HUB_DISABLE_TELEMETRY


# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
# transformer.enable_layerwise_casting(
#     storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
# )

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)

pipeline.vae.enable_tiling()
pipeline.vae.enable_slicing()


# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(
    onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device,
                       offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(
    pipeline.vae, onload_device=onload_device, offload_type="leaf_level")

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California,
slow dolly-in, gentle ripples, film grain, shallow depth of field
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=20,
).frames[0]
export_to_video(video, "outputs/xxx.mp4", fps=24)
