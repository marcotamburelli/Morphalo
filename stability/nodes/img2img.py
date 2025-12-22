import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionXLControlNetImg2ImgPipeline,
                       StableDiffusionXLImg2ImgPipeline)
from PIL import Image

from stability.core.config import *
from stability.dag import AttachmentSink, NodeRef
from stability.nodes import *


class _ControlNetSpec(TypedDict):
    key: str
    model_id: str
    conditioning_scale: float


@dataclass
class Img2Img(NodeRef):
    """
    Image-to-image node with optional ControlNet conditioning.

    This node transforms an input image into a new image guided by a text
    prompt. The initial image is provided via the default DAG connection
    (``src >> img2img``), while one or more ControlNet inputs may be attached
    to further constrain the generation (e.g. depth, canny, lineart).

    The node produces a single primary output dictionary containing at least:
    - ``image``: filesystem path to the generated image
    - ``metadata``: filesystem path to a JSON sidecar with generation details

    Parameters
    ----------
    id : str
        Unique node identifier within the DAG.
    spec : dict
        Configuration dictionary describing the model, prompts, and
        image-to-image parameters. Typical keys include ``model``, ``prompt``,
        ``negative_prompt``, ``params`` (e.g. ``steps``, ``strength``,
        ``guidance_scale``), ``seed``, and optional ``vae``.

    Attributes
    ----------
    op : str
        Operator identifier automatically derived from the concrete class
        name (e.g. ``"img2img"``).

    Notes
    -----
    - The default input edge provides the *init image* for the image-to-image
      process.
    - ControlNet inputs are attached via dedicated sinks and are passed to
      the underlying SDXL ControlNet image-to-image pipeline as conditioning
      images.
    - If no ControlNet is configured, this node behaves as a standard
      image-to-image transformation.
    """

    spec: Dict[str, Any] = field(default_factory=dict)

    _controlnets: List[_ControlNetSpec] = field(default_factory=list, init=False, repr=False)
    _cn_counter: int = field(default=0, init=False, repr=False)

    def addControlNet(
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
        if key is None:
            self._cn_counter += 1
            key = f'cn{self._cn_counter}'

        self._controlnets.append({
            'key': key,
            'model_id': model_id,
            'conditioning_scale': float(conditioning_scale),
        })

        return AttachmentSink(
            id=f'controlnet:{key}',
            target=self,
            input_id=f'controlnet:{key}',
        )

    createControlNet = addControlNet  # alias

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        # HF env
        os.environ['HF_HOME'] = HF_HOME
        os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = HF_HUB_DISABLE_TELEMETRY

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = self.spec or {}

        # model settings
        model = spec.get('model', {})
        device = model.get('device', 'cuda')
        dtype = resolve_dtype(model.get('dtype', 'bf16'))

        model_path = model.get('path')
        if not model_path:
            raise ValueError('spec.model.path is required')
        model_path = os.path.expanduser(model_path)

        # prompts
        prompt = norm_prompt(spec.get('prompt'))
        negative = norm_prompt(spec.get('negative_prompt'), joiner=', ')

        # params
        params = spec.get('params', {})
        steps = int(params.get('steps', 30))
        cfg = float(params.get('cfg', params.get('guidance_scale', 6.0)))
        strength = float(params.get('strength', 0.7))  # IMG2IMG specific :contentReference[oaicite:2]{index=2}
        width = int(params.get('width', 1024))
        height = int(params.get('height', 1024))

        # seed
        seed = resolve_seed(spec.get('seed', 'random'))
        gen = torch.Generator(device=device).manual_seed(seed)

        # VAE optional
        vae_id = spec.get('vae', {}).get('id') if isinstance(spec.get('vae'), dict) else spec.get('vae')
        vae = None
        if vae_id:
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype).to(device)

        # inputs from DAG
        input = input or {}

        # 1) init image comes from default input_id
        init_up = input.get('default')
        if init_up is None:
            raise ValueError('Img2Img requires an init image wired into the default input (src >> img2img).')

        init_path = init_up.get('image') or init_up.get('path')
        if not init_path:
            raise ValueError("Init image upstream output must contain 'image' (path).")

        init_image = Image.open(init_path).convert('RGB').resize((width, height))

        # 2) control images (optional but typically present)
        cn_models: List[ControlNetModel] = []
        cn_images: List[Image.Image] = []
        cn_scales: List[float] = []

        for cn in self._controlnets:
            in_id = f'controlnet:{cn['key']}'
            up = input.get(in_id)
            if up is None:
                raise ValueError(f"Missing ControlNet input for '{in_id}'.")

            p = up.get('image') or up.get('path')
            if not p:
                raise ValueError(f"Upstream output for '{in_id}' must contain 'image' (path).")

            cn_images.append(Image.open(p).convert('RGB').resize((width, height)))
            cn_scales.append(float(cn['conditioning_scale']))
            cn_models.append(ControlNetModel.from_pretrained(cn['model_id'], torch_dtype=dtype).to(device))

        # pipeline load (two-path: direct or via base components)
        if cn_models:
            # try:
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                model_path,
                controlnet=cn_models if len(cn_models) > 1 else cn_models[0],
                torch_dtype=dtype,
                **({'vae': vae} if vae is not None else {}),
            ).to(device)
            # except Exception:
            #     base = StableDiffusionXLPipeline.from_single_file(
            #         model_path,
            #         torch_dtype=dtype,
            #         **({'vae': vae} if vae is not None else {}),
            #     ).to(device)

            #     pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            #         **base.components,
            #         controlnet=cn_models if len(cn_models) > 1 else cn_models[0],
            #     ).to(device)
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
                **({'vae': vae} if vae is not None else {}),
            ).to(device)

        # run
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # IMPORTANT: Img2Img + ControlNet uses image=init and control_image=control :contentReference[oaicite:3]{index=3}
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init_image,  # init image :contentReference[oaicite:4]{index=4}
            control_image=cn_images if len(cn_images) > 1 else cn_images[0],  # control image(s) :contentReference[oaicite:5]{index=5}
            controlnet_conditioning_scale=cn_scales if len(cn_scales) > 1 else cn_scales[0],  # :contentReference[oaicite:6]{index=6}
            strength=strength,  # :contentReference[oaicite:7]{index=7}
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
            width=width,
            height=height,
        )

        if device.startswith('cuda'):
            torch.cuda.synchronize()

        dt_s = time.perf_counter() - t0
        img = result.images[0]

        tag = time.strftime('%Y-%m-%d_%H%M%S')
        img_path = out_dir / f'{tag}_i2i_cn_seed{seed}.png'
        img.save(img_path)

        out = {
            'ok': True,
            'node': 'i2i',
            'id': self.id,
            'image': str(img_path),
            'seed': seed,
            'params': {
                'steps': steps,
                'guidance_scale': cfg,
                'strength': strength,
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
            ],
            'timing': {'seconds': round(dt_s, 3)},
        }

        meta_path = img_path.with_suffix('.json')
        meta_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
        out['metadata'] = str(meta_path)

        return out
