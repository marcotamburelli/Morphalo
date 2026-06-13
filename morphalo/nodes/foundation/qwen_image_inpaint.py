from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from morphalo.cache.models import (evict_qwen_image_inpaint_controlnet,
                                   evict_qwen_image_inpaint_pipe,
                                   get_qwen_image_inpaint_pipe)
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.cuda_mem import (cleanup_torch_cuda,
                                            synchronize_torch_cuda)
from morphalo.nodes.common.cuda_stat import (cuda_mem_stats, cuda_prerun,
                                             cuda_sync)
from morphalo.nodes.common.env import setup_env
from morphalo.nodes.common.io import save_image
from morphalo.nodes.foundation.qwen_controlnet_wiring import \
    QwenImageInpaintMixin
from morphalo.nodes.foundation.qwen_utils import (qwen_cpu_generator,
                                                  qwen_stats_device,
                                                  resolve_qwen_model_config,
                                                  resolve_qwen_seed)
from morphalo.nodes.io import finalize_image_output
from morphalo.nodes.wiring.mixins import PromptMixin
from morphalo.nodes.wiring.prompt import PromptBundle

DEFAULT_QWEN_IMAGE_INPAINT_CONTROLNET = \
    'InstantX/Qwen-Image-ControlNet-Inpainting'


@dataclass
class QwenImageInpaint(QwenImageInpaintMixin, PromptMixin, NodeRef):
    """
    Qwen-Image inpainting node with a dedicated Qwen ControlNet backend.

    This node edits a masked region of an input image using
    ``QwenImageControlNetInpaintPipeline``. It is conceptually an inpainting
    node, not a text-to-image node with optional ControlNet attachments.

    The node uses a single Qwen inpainting ControlNet backend configured by
    ``model.controlnet_id``. The ControlNet is part of the node backend rather
    than a DAG-composable list of ControlNet attachments.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.
    spec : SpecInput, optional
        Node specification.

        ``model`` keys:

        - ``id`` : str, optional
            Base Qwen-Image model identifier. Defaults to ``'Qwen/Qwen-Image'``.
        - ``controlnet_id`` : str, optional
            Qwen inpainting ControlNet identifier. Defaults to
            ``'InstantX/Qwen-Image-ControlNet-Inpainting'``.
        - ``dtype`` : str, optional
            Torch dtype used to load the model. Defaults to ``'bf16'``.
        - ``device_map`` : {'balanced'}, optional
            Accelerate device map used to dispatch the model. Defaults to
            ``'balanced'``. The base pipeline is loaded through ``device_map``.
            The Qwen ControlNet component is loaded without ``device_map``
            because ``QwenImageControlNetModel.from_pretrained`` does not
            support it, then moved to a concrete runtime device.

        ``params`` keys:

        - ``steps`` : int, optional
            Number of inference steps. Defaults to ``30``.
        - ``true_cfg_scale`` : float, optional
            True classifier-free guidance scale. Defaults to ``4.0``.
        - ``controlnet_conditioning_scale`` : float, optional
            Strength of the inpainting ControlNet. Defaults to ``1.0``.
        - ``width`` : int, optional
            Output width. If omitted, the control image width is used.
        - ``height`` : int, optional
            Output height. If omitted, the control image height is used.

        ``seed`` : int or str, optional
            Seed value or ``'random'``. Defaults to ``'random'``.
    evict_after_run : bool, default=False
        If True, evict cached Qwen inpainting pipeline and ControlNet after a
        successful execution.

    Inputs
    ------
    default : dict
        Required base image payload containing ``'image'`` or ``'path'``. Internally
        passed to Diffusers as ``control_image``.
    mask : dict
        Required mask payload wired via :meth:`mask`. White areas are repainted.
    prompt:default : dict, optional
        Optional upstream prompt bundle overriding prompt fields from ``spec``.

    Outputs
    -------
    dict
        Primary output dictionary containing the generated image path, seed,
        metadata path, timing, model information, and resolved input metadata.

    Notes
    -----
    - The node currently supports a single inpainting ControlNet backend.
    - ``model.controlnet_id`` configures that backend; it is not declared via a
      multi-slot registry like SDXL ControlNet.
    - The prompt should describe the whole desired image, not only the masked
      area, because Qwen inpainting models are sensitive to global prompt
      context.
    """

    spec: SpecInput = field(default_factory=dict)
    evict_after_run: bool = False

    @property
    def uses_cuda(self) -> bool:
        return True

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        setup_env()

        input = input or {}
        spec = resolve_spec(self.spec)

        bundle = self.build_qwen_image_inpaint_bundle(input)

        pb = PromptBundle(spec=spec, input=input)
        prompt = pb.prompt
        negative = pb.negative_prompt
        if not prompt:
            raise ValueError('QwenImageInpaint requires a non-empty prompt.')

        model_cfg = resolve_qwen_model_config(
            spec,
            default_model_id='Qwen/Qwen-Image',
            node_name='QwenImageInpaint',
            supported_by='QwenImageInpaint',
        )

        model = spec.get('model', {}) if isinstance(spec, dict) else {}
        controlnet_model_id = model.get(
            'controlnet_id',
            DEFAULT_QWEN_IMAGE_INPAINT_CONTROLNET,
        )

        params = spec.get('params', {}) if isinstance(spec, dict) else {}
        steps = int(params.get('steps', 30))
        true_cfg_scale = float(params.get('true_cfg_scale', 4.0))
        conditioning_scale = float(
            params.get('controlnet_conditioning_scale', 1.0)
        )

        width = params.get('width', None)
        height = params.get('height', None)
        if width is None:
            width = bundle.control_image_arg.size[0]
        else:
            width = int(width)
        if height is None:
            height = bundle.control_image_arg.size[1]
        else:
            height = int(height)

        seed = resolve_qwen_seed(spec)
        gen = qwen_cpu_generator(seed)

        pipe = get_qwen_image_inpaint_pipe(
            model_id=model_cfg.model_id,
            controlnet_model_id=controlnet_model_id,
            dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
        )

        stats_device = qwen_stats_device()
        cuda_prerun(stats_device)
        t0 = time.perf_counter()

        call_kwargs: Dict[str, Any] = {
            'prompt': prompt,
            'negative_prompt': negative or ' ',
            'control_image': bundle.control_image_arg,
            'control_mask': bundle.mask_arg,
            'controlnet_conditioning_scale': conditioning_scale,
            'width': width,
            'height': height,
            'num_inference_steps': steps,
            'true_cfg_scale': true_cfg_scale,
            'generator': gen,
        }

        result = pipe(**call_kwargs)

        cuda_sync(stats_device)
        dt_s = time.perf_counter() - t0

        img = result.images[0]
        img_path = save_image(
            output_dir,
            node_id=self.id,
            seed=seed,
            img=img,
        )
        mem = cuda_mem_stats(stats_device)

        out = finalize_image_output(
            node_kind=str(self.op),
            node_id=self.id,
            img_path=img_path,
            seed=seed,
            params={
                'steps': steps,
                'true_cfg_scale': true_cfg_scale,
                'controlnet_conditioning_scale': conditioning_scale,
                'width': width,
                'height': height,
                'inputs': bundle.metadata,
            },
            dt_s=dt_s,
            cuda_mem=mem,
            model_info={
                'id': model_cfg.model_id,
                'controlnet_id': controlnet_model_id,
                'device_map': model_cfg.device_map,
                'dtype': str(model_cfg.dtype).replace('torch.', ''),
                'pipeline': 'QwenImageControlNetInpaintPipeline',
            },
        )

        return out

    def post_run(self) -> None:
        """
        Optional post-run cleanup hook.

        Notes
        -----
        When ``evict_after_run`` is enabled, this hook removes the cached Qwen
        image inpainting pipeline and ControlNet corresponding to the resolved
        runtime configuration, then triggers best-effort Python/CUDA cleanup.
        """
        if self.uses_cuda:
            synchronize_torch_cuda()

        if not self.evict_after_run:
            return

        try:
            spec = resolve_spec(self.spec)

            model_cfg = resolve_qwen_model_config(
                spec,
                default_model_id='Qwen/Qwen-Image',
                node_name='QwenImageInpaint',
                supported_by='QwenImageInpaint',
            )

            model = spec.get('model', {}) if isinstance(spec, dict) else {}
            controlnet_model_id = model.get(
                'controlnet_id',
                DEFAULT_QWEN_IMAGE_INPAINT_CONTROLNET,
            )

            pipe = evict_qwen_image_inpaint_pipe(
                model_id=model_cfg.model_id,
                controlnet_model_id=controlnet_model_id,
                dtype=model_cfg.dtype,
                device_map=model_cfg.device_map,
            )
            controlnet = evict_qwen_image_inpaint_controlnet(
                model_id=controlnet_model_id,
                dtype=model_cfg.dtype,
            )

            if pipe is None and controlnet is None:
                raise RuntimeError(
                    'QwenImageInpaint cache objects not found for eviction: '
                    f'model_id={model_cfg.model_id!r}, '
                    f'controlnet_id={controlnet_model_id!r}, '
                    f'device_map={model_cfg.device_map!r}, '
                    f'dtype={model_cfg.dtype!r}'
                )

            del pipe
            del controlnet
        finally:
            cleanup_torch_cuda()
