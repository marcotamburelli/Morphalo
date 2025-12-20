import json
import os
import time

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

from stability.core.config import *
from stability.nodes import *


def run_t2i(spec: dict, out_dir: Path) -> dict:
    """
    Text-to-image node.
    Returns a JSON-serializable dict (intended to be printed to stdout).
    """
    # Hugging Face env (must be set before loading)
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = HF_HUB_DISABLE_TELEMETRY

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model settings ---
    model = spec.get('model', {})
    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))

    model_path = model.get('path')
    if not model_path:
        raise ValueError('spec.model.path is required')
    model_path = os.path.expanduser(model_path)

    # --- prompts ---
    prompt = norm_prompt(spec.get('prompt'))
    negative = norm_prompt(spec.get('negative_prompt'), joiner=', ')

    # --- params ---
    params = spec.get('params', {})
    steps = int(params.get('steps', 30))
    cfg = float(params.get('cfg', params.get('guidance_scale', 6.0)))
    width = int(params.get('width', 1024))
    height = int(params.get('height', 1024))

    # --- seed / generator ---
    seed = resolve_seed(spec.get('seed', 'random'))
    gen = torch.Generator(device=device).manual_seed(seed)

    # --- optional VAE fix ---
    vae_id = spec.get('vae', {}).get('id') if isinstance(
        spec.get('vae'),
        dict
    ) else spec.get('vae')
    # If you want to always use the fix by default, set it in HOCON:
    # vae = 'madebyollin/sdxl-vae-fp16-fix'
    vae = None
    if vae_id:
        vae = AutoencoderKL.from_pretrained(
            vae_id, torch_dtype=dtype).to(device)

    # --- load pipeline ---
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        **({'vae': vae} if vae is not None else {})
    ).to(device)

    # memory-friendly
    # pipe.enable_vae_tiling()

    # --- measure time + memory ---
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
        generator=gen,
    )

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    dt_s = time.perf_counter() - t0

    img = result.images[0]

    # --- save image ---
    tag = time.strftime('%Y-%m-%d_%H%M%S')
    img_path = out_dir / f'{tag}_t2i_seed{seed}.png'
    img.save(img_path)

    # --- memory stats ---
    mem = {}
    if device.startswith('cuda'):
        mem = {
            'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 3),
            'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 3),
            'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

    # --- output JSON dict ---
    out = {
        'ok': True,
        'node': 't2i',
        'image': str(img_path),
        'seed': seed,
        'model': {
            'path': model_path,
            'dtype': str(dtype).replace('torch.', ''),
            'device': device,
        },
        'params': {
            'steps': steps,
            'guidance_scale': cfg,
            'width': width,
            'height': height,
        },
        'timing': {
            'seconds': round(dt_s, 3),
        },
        'cuda_mem': mem,
    }

    # Optionally save a sidecar metadata file (handy for debugging)
    meta_path = img_path.with_suffix('.json')
    meta_path.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False
        ),
        encoding='utf-8'
    )
    out['metadata'] = str(meta_path)

    return out


# Example 'CLI-like' usage:
if __name__ == '__main__':
    import sys

    # suppose spec is already loaded from HOCON into dict:
    # spec = load_hocon_spec(sys.argv[1])
    # out = run_t2i(spec, Path('outputs'))
    # print(json.dumps(out, ensure_ascii=False))
    pass
