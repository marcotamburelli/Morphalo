import os
from pathlib import Path
import torch

from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from common.config import HF_HOME, HF_HUB_CACHE, HF_HUB_DISABLE_TELEMETRY

# Hugging Face env (must be set before loading)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_HUB_DISABLE_TELEMETRY"] = HF_HUB_DISABLE_TELEMETRY

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# quant_config = PipelineQuantizationConfig(
#     quant_backend="bitsandbytes_4bit",
#     quant_kwargs={
#         "load_in_4bit": True,
#         "bnb_4bit_quant_type": "nf4",
#         "bnb_4bit_compute_dtype": torch.bfloat16,
#     },
#     # components_to_quantize=["transformer", "text_encoder"],
#     components_to_quantize=["text_encoder"],
# )

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    # quantization_config=quant_config,
    device_map="balanced",
    # device_map="cuda"
)

# Reduce VRAM peaks further
# pipeline.enable_model_cpu_offload()

prompt = (
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, "
    "highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
)

image = pipeline(prompt, num_inference_steps=25).images[0]
path = out_dir / "cat_margarita.png"
image.save(path)

print(f"Saved: {path.resolve()}")
print(f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
