import os
import secrets
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    StableDiffusionXLPipeline
)

from common.config import HF_HOME, HF_HUB_CACHE, HF_HUB_DISABLE_TELEMETRY

# Hugging Face env (must be set before loading)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_HUB_DISABLE_TELEMETRY"] = HF_HUB_DISABLE_TELEMETRY

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

dtype = torch.bfloat16

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=dtype,
).to("cuda")

pipe_t2i = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=dtype,
).to("cuda")

# AutoPipeline per img2img, ma con componenti IDENTICI alla t2i
pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to("cuda")

pipe_t2i.enable_vae_tiling()
pipe_i2i.enable_vae_tiling()

prompt = "cinematic portrait, soft rim light, ultra-detailed"
neg = "lowres, blurry, artifacts, extra fingers"

seed_t2i = secrets.randbelow(2**31)
seed_i2i = secrets.randbelow(2**31)

gen = torch.Generator("cuda").manual_seed(seed_t2i)

img = pipe_t2i(
    prompt=prompt,
    negative_prompt=neg,
    num_inference_steps=30,
    guidance_scale=6.0,
    generator=gen,
).images[0]

gen = torch.Generator("cuda").manual_seed(seed_i2i)
img2 = pipe_i2i(
    prompt="Elven princess in plate armor. Lord of the Rings style. Determined look, sharper details, more realistic skin texture",
    negative_prompt=neg,
    image=img,
    strength=0.70,
    num_inference_steps=30,
    guidance_scale=9.0,
    generator=gen,
).images[0]


path = out_dir / "t2i.png"
img.save(path)

path = out_dir / "img2img.png"
img2.save(path)


print(f"Saved: {path.resolve()}")
print(
    f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
