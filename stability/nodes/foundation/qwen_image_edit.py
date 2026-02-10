import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from stability.cache.models import get_qwen_image_edit_pipe
from stability.core.paths import ensure_out_dir
from stability.dag import NodeRef
from stability.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                   resolve_seed, resolve_spec)
from stability.nodes.common.cuda_stat import (cuda_mem_stats, cuda_prerun,
                                              cuda_sync)
from stability.nodes.common.env import setup_env
from stability.nodes.io import finalize_image_output, save_image
from stability.nodes.wiring.mixins import PromptMixin
from stability.nodes.wiring.prompt import PromptBundle


@dataclass
class QwenImageEdit(PromptMixin, NodeRef):
    """
    Image editing node based on Qwen-Image-Edit.

    This node applies a natural-language editing instruction to an input image
    using the Qwen-Image-Edit Diffusers pipeline. Unlike traditional img2img
    diffusion, the input image is treated as a semantic reference rather than
    a strict geometric constraint: the model may change pose, viewpoint, or
    structure when explicitly or implicitly requested by the prompt.

    The node expects a single input image wired into the default input slot,
    following the same conventions as Img2Img nodes.

    Notes
    -----
    - Editing is instruction-driven rather than noise-driven; parameters such
      as `strength` are not supported.
    - The primary control parameters are the textual prompt and `true_cfg_scale`.
    - If `width` and `height` are not provided, the output resolution defaults
      to the input image resolution.
    - The model may regenerate parts of the image beyond local pixel edits
      (e.g. changing pose or camera angle) when prompted.

    Parameters
    ----------
    spec : SpecInput, optional
        Node specification dictionary. Expected fields include:

        model : dict, optional
            Model configuration.

            - id : str, optional
                Hugging Face model identifier. Defaults to
                ``'Qwen/Qwen-Image-Edit'``.
            - dtype : str, optional
                Torch dtype used to load model weights (e.g. ``'bf16'``,
                ``'fp16'``). Defaults to ``'bf16'``.
            - device_map : str, optional
                Accelerate device map used for model dispatch. Typical values
                are ``'balanced'``, ``'auto'``, ``'cuda'``. Defaults to
                ``'balanced'``.

        params : dict, optional
            Inference parameters.

            - steps : int, optional
                Number of inference steps. Defaults to 25.
            - true_cfg_scale : float, optional
                True classifier-free guidance scale controlling how strongly
                the textual instruction overrides the input image.
            - width : int, optional
                Output image width. If omitted, the input image width is used.
            - height : int, optional
                Output image height. If omitted, the input image height is used.

        seed : int or str, optional
            Random seed used for generation. Can be an integer or ``'random'``.

    Input
    -----
    default : dict
        Upstream image payload. Must contain either:

        - 'image' : str
            Path to the input image file.
        - 'path' : str
            Alternative key for the image path.

    Returns
    -------
    dict
        Standard image node output dictionary containing:

        - image : str
            Path to the generated image.
        - seed : int
            Seed used for generation.
        - params : dict
            Effective inference parameters.
        - model_info : dict
            Information about the model used.
        - timing and CUDA memory statistics.

    See Also
    --------
    Img2Img :
        Traditional diffusion-based image-to-image node.
    QwenImage :
        Text-to-image node based on the Qwen-Image model.
    """

    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        setup_env()

        input = input or {}

        init_up = input.get('default')
        if init_up is None:
            raise ValueError(
                'QwenImageEdit requires an init image wired into the default input (src >> qwen_edit).'
            )

        init_path = init_up.get('image') or init_up.get('path')
        if not init_path:
            raise ValueError(
                "Init image upstream output must contain 'image' (path).")

        init_image = Image.open(init_path).convert('RGB')

        spec = resolve_spec(self.spec)

        # prompt
        pb = PromptBundle(spec=spec, input=input)
        prompt = pb.prompt
        negative = pb.negative_prompt

        # model
        model = spec.get('model', {}) if isinstance(spec, dict) else {}
        model_id = model.get('id', 'Qwen/Qwen-Image-Edit')
        dtype = resolve_dtype(model.get('dtype', 'bf16'))
        device_map = model.get('device_map', 'balanced')

        # params
        params = spec.get('params', {}) if isinstance(spec, dict) else {}
        steps = int(params.get('steps', 25))

        true_cfg_scale = params.get('true_cfg_scale', None)
        if true_cfg_scale is not None:
            true_cfg_scale = float(true_cfg_scale)

        height = params.get('height', None)
        width = params.get('width', None)
        if height is not None:
            height = int(height)
        if width is not None:
            width = int(width)

        # seed / generator
        seed = resolve_seed(spec.get('seed', 'random'))
        gen = torch.Generator(device='cpu').manual_seed(seed)

        # pipeline
        pipe = get_qwen_image_edit_pipe(
            model_id=model_id,
            dtype=dtype,
            device_map=device_map,
        )

        # --- run + stats ---
        # cuda_prerun expects a string device; with device_map it’s still useful if CUDA is involved.
        # We'll treat 'balanced'/'auto' as 'cuda' for stats, since modules may be on GPU.
        stats_device = 'cuda' if str(device_map) in (
            'balanced', 'auto', 'cuda'
        ) else 'cpu'
        cuda_prerun(stats_device)
        t0 = time.perf_counter()

        call_kwargs = {
            'image': init_image,
            'prompt': prompt,
            'negative_prompt': negative or ' ',
            'num_inference_steps': steps,
            'generator': gen,
        }

        if true_cfg_scale is not None:
            call_kwargs['true_cfg_scale'] = true_cfg_scale
        if width is not None:
            call_kwargs['width'] = width
        if height is not None:
            call_kwargs['height'] = height

        result = pipe(**call_kwargs)

        cuda_sync(stats_device)
        dt_s = time.perf_counter() - t0

        img = result.images[0]

        out_dir = ensure_out_dir(output_dir)
        img_path = save_image(out_dir, node_id=self.id, seed=seed, img=img)

        mem = cuda_mem_stats(stats_device)

        out = finalize_image_output(
            node_kind=str(self.op),
            node_id=self.id,
            img_path=img_path,
            seed=seed,
            params={
                'steps': steps,
                **({} if true_cfg_scale is None else {'true_cfg_scale': true_cfg_scale}),
                **({} if height is None else {'height': height}),
                **({} if width is None else {'width': width}),
            },
            dt_s=dt_s,
            cuda_mem=mem,
            model_info={
                'id': model_id,
                'device_map': device_map,
                'dtype': str(dtype).replace('torch.', ''),
            }
        )

        return out
