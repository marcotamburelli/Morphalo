import json
import os
import time

import torch
from diffusers import (AutoencoderKL, StableDiffusionXLImg2ImgPipeline)

from stability.core.config import *
from stability.nodes import *


def run_i2i(spec: dict, image: Union[str, Path, Image.Image], out_dir: Path) -> dict:
    """
    Image-to-image node.
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
    cfg = float(params.get('cfg', params.get('guidance_scale', 5.5)))
    strength = float(params.get('strength', 0.22))

    # Optional: keep resolution aligned to init image by default
    # (SDXL i2i usually works fine without forcing width/height)
    # If you want to force it, set these in spec.params
    width = params.get('width', None)
    height = params.get('height', None)
    if width is not None:
        width = int(width)
    if height is not None:
        height = int(height)

    # --- seed / generator ---
    seed = resolve_seed(spec.get('seed', 'random'))
    gen = torch.Generator(device=device).manual_seed(seed)

    # --- optional VAE fix ---
    vae_id = spec.get('vae', {}).get('id') if isinstance(
        spec.get('vae'), dict) else spec.get('vae')
    vae = None
    if vae_id:
        vae = AutoencoderKL.from_pretrained(
            vae_id, torch_dtype=dtype).to(device)

    # --- load pipeline ---
    pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        **({'vae': vae} if vae is not None else {})
    ).to(device)

    # pipe_i2i.enable_vae_tiling()

    init_img = load_init_image(image)

    # --- measure time + memory ---
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        image=init_img,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=gen,
    )
    # only pass width/height if explicitly set (otherwise let pipeline handle)
    if width is not None:
        call_kwargs['width'] = width
    if height is not None:
        call_kwargs['height'] = height

    result = pipe_i2i(**call_kwargs)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    dt_s = time.perf_counter() - t0

    img_out = result.images[0]

    # --- save image ---
    tag = time.strftime('%Y-%m-%d_%H%M%S')
    src_seed = None
    if isinstance(image, (str, Path)):
        src = str(Path(str(image)).expanduser())
    else:
        src = '<PIL.Image>'

    img_path = out_dir / f'{tag}_i2i_seed{seed}_s{int(strength*1000):03d}.png'
    img_out.save(img_path)

    # --- memory stats ---
    mem = {}
    if device.startswith('cuda'):
        mem = {
            'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 3),
            'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 3),
            'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

    out = {
        'ok': True,
        'node': 'i2i',
        'input_image': src,
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
            'strength': strength,
            **({} if width is None else {'width': width}),
            **({} if height is None else {'height': height}),
        },
        'timing': {'seconds': round(dt_s, 3)},
        'cuda_mem': mem,
    }

    meta_path = img_path.with_suffix('.json')
    meta_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    out['metadata'] = str(meta_path)

    return out
