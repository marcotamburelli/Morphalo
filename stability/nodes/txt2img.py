import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLPipeline)

from stability.core.config import *
from stability.dag import AttachmentSink, NodeRef
from stability.nodes import *


class _ControlNetSpec(TypedDict):
    key: str
    model_id: str
    conditioning_scale: float


@dataclass
class Txt2Img(NodeRef):
    """
    Text-to-image node with optional ControlNet conditioning.

    This node generates an image from a text prompt using an SDXL-based
    text-to-image pipeline. One or more ControlNet inputs may be attached
    to guide the generation process.

    The node produces a single primary output dictionary containing at least:
    - ``image``: filesystem path to the generated image
    - ``metadata``: filesystem path to a JSON sidecar with generation details

    Parameters
    ----------
    id : str
        Unique node identifier within the DAG.
    spec : dict
        Configuration dictionary describing the model, prompts, and
        generation parameters. Typical keys include ``model``, ``prompt``,
        ``negative_prompt``, ``params``, ``seed``, and optional ``vae``.

    Attributes
    ----------
    op : str
        Operator identifier automatically derived from the concrete class
        name (e.g. ``"txt2img"``).
    """

    spec: Dict[str, Any] = field(default_factory=dict)

    # Internal list of ControlNets requested via createControlNet()
    _controlnets: List[_ControlNetSpec] = field(default_factory=list, init=False, repr=False)
    _cn_counter: int = field(default=0, init=False, repr=False)

    def createControlNet(
        self,
        model_id: str,
        conditioning_scale: float = 1.0,
        key: Optional[str] = None,
    ) -> AttachmentSink:
        """
        Declares a ControlNet input for this node and returns an AttachmentSink.

        Usage:
            cn = out.createControlNet("diffusers/controlnet-canny-sdxl-1.0", conditioning_scale=0.8)
            conditioning_img_node >> cn

        Parameters
        ----------
        model_id : str
            HuggingFace repo id or local path for the ControlNet weights.
        conditioning_scale : float, optional
            Strength of this ControlNet.
        key : str, optional
            Optional stable key. If not provided, an incremental key is generated.

        Returns
        -------
        AttachmentSink
            A sink that can be wired with ``src_node >> sink``. The upstream node
            output (typically containing an image path) will be provided to this
            node under an ``input_id`` like ``controlnet:<key>``.
        """
        if not model_id:
            raise ValueError('model_id is required')

        if key is None:
            self._cn_counter += 1
            key = f'cn{self._cn_counter}'

        # register ControlNet config internally
        self._controlnets.append({
            'key': key,
            'model_id': model_id,
            'conditioning_scale': float(conditioning_scale),
        })

        # sink: wires an image-producing node into this specific controlnet slot
        return AttachmentSink(
            id=f'controlnet:{key}',
            target=self,
            input_id=f'controlnet:{key}',
        )

    # alias più corto se vuoi
    addControlNet = createControlNet

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Executes the T2I node. If ControlNets were declared, uses an SDXL ControlNet pipeline.
        """
        # --- Hugging Face env (must be set before loading) ---
        os.environ['HF_HOME'] = HF_HOME
        os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = HF_HUB_DISABLE_TELEMETRY

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = self.spec or {}

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
        vae_id = spec.get('vae', {}).get('id') if isinstance(spec.get('vae'), dict) else spec.get('vae')
        vae = None
        if vae_id:
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype).to(device)

        # --- resolve ControlNet inputs from DAG wiring ---
        # input is a mapping input_id -> upstream output dict
        input = input or {}
        cn_images: List[Any] = []
        cn_scales: List[float] = []
        cn_models: List[ControlNetModel] = []

        if self._controlnets:
            # For each declared controlnet, expect an incoming edge feeding input_id "controlnet:<key>"
            for cn in self._controlnets:
                in_id = f"controlnet:{cn['key']}"
                upstream = input.get(in_id)
                if upstream is None:
                    raise ValueError(f"Missing ControlNet input for '{in_id}'. Did you wire an image into it?")
                # we assume upstream output contains either:
                # - {'image': '/path/to/img.png'}  (your stable nodes will do this)
                # - or directly {'path': ...}
                img_path = upstream.get('image') or upstream.get('path')
                if not img_path:
                    raise ValueError(f"Upstream output for '{in_id}' does not contain an image path")

                cn_images.append(img_path)  # diffusers can accept PIL.Image too; here pass path and load later if needed
                cn_scales.append(float(cn["conditioning_scale"]))

                cn_models.append(
                    ControlNetModel.from_pretrained(cn['model_id'], torch_dtype=dtype).to(device)
                )

        # --- load pipeline (with or without controlnet) ---
        # Path A: try direct ControlNet pipeline from single file (if supported in your diffusers version)
        pipe = None
        if cn_models:
            # try:
            pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                model_path,
                controlnet=cn_models if len(cn_models) > 1 else cn_models[0],
                torch_dtype=dtype,
                **({'vae': vae} if vae is not None else {}),
            ).to(device)
        # except Exception:
            #     # Path B: build base pipeline then wrap into ControlNet pipeline via components
            #     base = StableDiffusionXLPipeline.from_single_file(
            #         model_path,
            #         torch_dtype=dtype,
            #         **({'vae': vae} if vae is not None else {}),
            #     ).to(device)

            #     # Construct ControlNet pipeline from base components
            #     pipe = StableDiffusionXLControlNetPipeline(
            #         **base.components,
            #         controlnet=cn_models if len(cn_models) > 1 else cn_models[0],
            #     ).to(device)
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
                **({'vae': vae} if vae is not None else {}),
            ).to(device)

        # --- measure time + memory ---
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # --- call pipeline ---
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=gen,
        )

        if cn_models:
            # NOTE: depending on diffusers version, arg name is often control_image / controlnet_conditioning_image
            # StableDiffusionXLControlNetPipeline uses `control_image` in many versions.
            call_kwargs['image'] = cn_images if len(cn_images) > 1 else cn_images[0]
            call_kwargs['controlnet_conditioning_scale'] = cn_scales if len(cn_scales) > 1 else cn_scales[0]

        result = pipe(**call_kwargs)

        if device.startswith('cuda'):
            torch.cuda.synchronize()

        dt_s = time.perf_counter() - t0
        img = result.images[0]

        # --- save image ---
        tag = time.strftime('%Y-%m-%d_%H%M%S')
        suffix = '_t2i_cn' if cn_models else '_t2i'
        img_path = out_dir / f'{tag}{suffix}_seed{seed}.png'
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

        out = {
            'ok': True,
            'node': 't2i',
            'id': self.id,
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
            'controlnet': [
                {
                    'key': cn['key'],
                    'model_id': cn['model_id'],
                    'conditioning_scale': cn['conditioning_scale'],
                    'input_id': f'controlnet:{cn['key']}',
                }
                for cn in self._controlnets
            ] if self._controlnets else [],
            'timing': {'seconds': round(dt_s, 3)},
            'cuda_mem': mem,
        }

        meta_path = img_path.with_suffix('.json')
        meta_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
        out['metadata'] = str(meta_path)

        return out
