import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from pyhocon import ConfigFactory

from stability.core.config import *

# da controllare:
# https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx_video.md


def round_to_vae(height: int, width: int, pipe: LTXConditionPipeline) -> Tuple[int, int]:
    # LTX VAE likes sizes multiple of its spatial compression ratio
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


def _as_str_prompt(x: Any, joiner: str = "\n") -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return joiner.join(str(i) for i in x).strip()
    return str(x).strip()


def load_hocon(path: str | Path) -> Dict[str, Any]:
    cfg = ConfigFactory.parse_file(str(Path(path).expanduser()))
    # Plain ordered dict (JSON-serializable-ish)
    return cfg.as_plain_ordered_dict()


def main():
    os.environ['HF_HOME'] = HF_HOME
    os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = HF_HUB_DISABLE_TELEMETRY

    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ltx_t2v_hocon.py path/to/spec.conf")

    spec_path = Path(sys.argv[1]).expanduser().resolve()
    spec = load_hocon(spec_path)

    model = spec.get("model", {})
    params = spec.get("params", {})

    # --- config with defaults (mirrors your argparse defaults) ---
    model_id = model.get("id", "Lightricks/LTX-Video-0.9.7-dev")
    upscaler_id = model.get(
        "upscaler", "Lightricks/ltxv-spatial-upscaler-0.9.7")

    prompt = _as_str_prompt(spec.get("prompt"))
    if not prompt:
        raise ValueError("spec.prompt is required (string or list of strings)")

    negative = _as_str_prompt(
        spec.get("negative_prompt",
                 "worst quality, inconsistent motion, blurry, jittery, distorted"),
        joiner=", ",
    )

    out = spec.get("out", "output_t2v.mp4")
    height = int(params.get("height", 512))
    width = int(params.get("width", 704))
    fps = int(params.get("fps", 24))
    num_frames = int(params.get("num_frames", 121))
    seed = int(spec.get("seed", 0))
    steps_main = int(params.get("steps_main", 30))
    steps_refine = int(params.get("steps_refine", 10))
    denoise_strength = float(params.get("denoise_strength", 0.4))
    downscale_factor = float(params.get("downscale_factor", 2 / 3))
    upscale_factor = float(params.get("upscale_latent_factor", 2))

    # --- device / dtype ---
    device = model.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: model.device is CUDA but torch.cuda.is_available() is False; falling back to CPU.")
        device = "cpu"

    dtype_s = str(model.get("dtype", "")).lower().strip()
    if dtype_s in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif dtype_s in ("fp16", "float16"):
        dtype = torch.float16
    elif dtype_s in ("fp32", "float32", ""):
        # default: bf16 on cuda, fp32 on cpu (same spirit as your script)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        raise ValueError(f"Unsupported model.dtype: {dtype_s}")

    # --- load pipelines ---
    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )

    pipe.enable_attention_slicing("max")

    # VAE: riduce il picco nel decode (conv3d)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        upscaler_id,
        vae=pipe.vae,
        torch_dtype=dtype,
    ).to(device)

    onload_device = torch.device(device)
    offload_device = torch.device("cpu")

    # Transformer (di solito è il pezzo più grosso)
    pipe.transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=True,
    )

    # Text encoder: spesso conviene a blocchi
    apply_group_offloading(
        pipe.text_encoder,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=2,
    )

    # VAE: così non resta piantato in GPU quando non serve
    apply_group_offloading(
        pipe.vae,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
    )

    expected_h, expected_w = height, width

    # Part 1: generate at smaller res -> latent video
    down_h = int(expected_h * downscale_factor)
    down_w = int(expected_w * downscale_factor)
    down_h, down_w = round_to_vae(down_h, down_w, pipe)

    gen = torch.Generator(device=device).manual_seed(seed)

    # --- measure time + memory ---
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    latents = pipe(
        conditions=None,
        prompt=prompt,
        negative_prompt=negative,
        width=down_w,
        height=down_h,
        num_frames=num_frames,
        num_inference_steps=steps_main,
        generator=gen,
        output_type="latent",
    ).frames

    # Part 2: latent upsample (2x)
    up_h, up_w = int(down_h * upscale_factor), int(down_w * upscale_factor)
    up_latents = pipe_upsample(latents=latents, output_type="latent").frames

    # Part 3: short denoise refine (optional but recommended)
    video = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=up_w,
        height=up_h,
        num_frames=num_frames,
        denoise_strength=denoise_strength,
        num_inference_steps=steps_refine,
        latents=up_latents,
        # recommended for timestep-aware VAE variants (0.9.1+)
        decode_timestep=float(params.get("decode_timestep", 0.05)),
        image_cond_noise_scale=float(
            params.get("image_cond_noise_scale", 0.025)
        ),
        generator=gen,
        output_type="pil",
    ).frames[0]

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    dt_s = time.perf_counter() - t0

    # Part 4: resize down to target
    video = [f.resize((expected_w, expected_h)) for f in video]

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(out_path), fps=fps)

    # Optional: write sidecar for debugging
    # --- memory stats ---
    mem = {}
    if device.startswith('cuda'):
        mem = {
            'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 3),
            'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 3),
            'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

    meta = {
        "ok": True,
        "node": "ltx_t2v_standalone",
        "spec": str(spec_path),
        "out": str(out_path),
        "model": {"id": model_id, "upscaler": upscaler_id, "device": device, "dtype": str(dtype).replace("torch.", "")},
        "params": {
            "width": expected_w,
            "height": expected_h,
            "fps": fps,
            "num_frames": num_frames,
            "seed": seed,
            "steps_main": steps_main,
            "steps_refine": steps_refine,
            "denoise_strength": denoise_strength,
            "downscale_factor": downscale_factor,
        },
        'timing': {'seconds': round(dt_s, 3)},
        'cuda_mem': mem,
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(
        meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved:", out_path)
    print("Meta :", meta_path)


if __name__ == "__main__":
    main()
