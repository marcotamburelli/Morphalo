import secrets
import os
from pathlib import Path
import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    AutoPipelineForImage2Image,
)
from common.config import HF_HOME, HF_HUB_CACHE, HF_HUB_DISABLE_TELEMETRY

# Hugging Face env (must be set before loading)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_HUB_DISABLE_TELEMETRY"] = HF_HUB_DISABLE_TELEMETRY

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)


device = "cuda"
dtype = torch.bfloat16  # su 5090 ok; se vedi problemi, prova torch.float16

model_path = os.path.expanduser(
    "~/models/juggernaut/juggernautXL_ragnarokBy.safetensors"
)

# VAE fix (opzionale ma consigliato in fp16/bf16)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=dtype,
).to(device)


pipe_t2i = StableDiffusionXLPipeline.from_single_file(
    model_path,
    vae=vae,
    torch_dtype=dtype,
).to(device)

pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to(device)

pipe_t2i.enable_vae_tiling()
pipe_i2i.enable_vae_tiling()

# --- prompts ---
prompt_t2i = """
half-body fashion runway photograph of a high-end model,
confident elegant posture, symmetrical face, sharp jawline,
professional catwalk pose, minimalist modern runway,
soft studio lighting, high fashion editorial style,
85mm lens look, shallow depth of field, ultra-detailed, cinematic
"""
neg_t2i = """
lowres, blurry, artifacts, extra fingers, extra limbs,
deformed face, bad anatomy, overexposed, harsh shadows
"""

prompt_i2i = """
elven warrior princess in full plate armor,
regal and powerful presence, same face and body proportions,
ornate elven plate armor, engraved steel, polished metal,
long flowing hair, noble expression,
epic fantasy cinematic lighting, dramatic sky,
high detail, ultra-detailed, cinematic
"""
neg_i2i = """
lowres, blurry, artifacts, extra fingers, extra limbs,
deformed face, bad anatomy, modern clothing, runway, fashion show
"""
# --- seeds (diversi, come vuoi tu) ---
seed_t2i = secrets.randbelow(2**31)
seed_i2i = secrets.randbelow(2**31)

gen_t2i = torch.Generator(device=device).manual_seed(seed_t2i)
gen_i2i = torch.Generator(device=device).manual_seed(seed_i2i)

# --- T2I (settaggi normali per Juggernaut v9) ---
img = pipe_t2i(
    prompt=prompt_t2i,
    negative_prompt=neg_t2i,
    num_inference_steps=30,   # 25–40 tipico
    guidance_scale=6.0,       # 4–7 tipico
    generator=gen_t2i,
).images[0]

# --- I2I (metamorfosi controllata) ---
img2 = pipe_i2i(
    prompt=prompt_i2i,
    negative_prompt=neg_i2i,
    image=img,
    strength=0.7,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=gen_i2i,
).images[0]


# path = out_dir / "t2i.png"
# img.save(path)

# path = out_dir / "img2img.png"
# img2.save(path)


p1 = out_dir / f"t2i_juggv9_seed{seed_t2i}.png"
p2 = out_dir / f"i2i_juggv9_seed{seed_i2i}_from{seed_t2i}_s022.png"
img.save(p1)
img2.save(p2)


print(
    f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
)


print("allocated:", torch.cuda.memory_allocated()/1024**3)
print("reserved :", torch.cuda.memory_reserved()/1024**3)
