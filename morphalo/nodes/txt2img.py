import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image

from morphalo.cache.models import get_sdxl_base_pipe
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.cuda_stat import (cuda_mem_stats, cuda_prerun,
                                             cuda_sync)
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import save_image
from morphalo.nodes.img import resolve_long_side_size
from morphalo.nodes.image_output import ImageOutputMixin
from morphalo.nodes.io import finalize_image_output
from morphalo.nodes.sdxl_pipe_builder import build_pipe_kwargs
from morphalo.nodes.sdxl_resolve import resolve_common
from morphalo.nodes.wiring.conditioning import (apply_ip_adapter, apply_lora,
                                                cleanup_adapters)
from morphalo.nodes.wiring.mixins import (ControlNetMixin, LoraMixin, PromptMixin,
                                          T2IAdapterMixin)
from morphalo.nodes.wiring.prompt import PromptBundle


def _first_image(
    image_or_images: Union[Image.Image, list[Image.Image], None],
) -> Optional[Image.Image]:
    """
    Return the first PIL image from a single image or image list.

    Parameters
    ----------
    image_or_images : PIL.Image.Image or list[PIL.Image.Image] or None
        Conditioning image payload.

    Returns
    -------
    PIL.Image.Image or None
        First image, or ``None`` when no image is available.
    """
    if image_or_images is None:
        return None

    if isinstance(image_or_images, list):
        return image_or_images[0] if image_or_images else None

    return image_or_images


def _first_conditioning_image(
    *,
    cn_bundle: Any,
    t2i_bundle: Any,
) -> Optional[Image.Image]:
    """
    Return the first available conditioning image.

    ControlNet has priority over T2I-Adapter because Txt2Img already enforces
    that they cannot be active together.

    Parameters
    ----------
    cn_bundle : Any
        ControlNet bundle exposing ``has_controlnet`` and ``control_image_arg``.

    t2i_bundle : Any
        T2I-Adapter bundle exposing ``has_t2i_adapter`` and ``adapter_image_arg``.

    Returns
    -------
    PIL.Image.Image or None
        First conditioning image, or ``None`` if no conditioning is active.
    """
    if cn_bundle.has_controlnet:
        return _first_image(cn_bundle.control_image_arg)

    if t2i_bundle.has_t2i_adapter:
        return _first_image(t2i_bundle.adapter_image_arg)

    return None


def _resolve_txt2img_size(
    *,
    width: int,
    height: int,
    long_side: Optional[int],
    cn_bundle: Any,
    t2i_bundle: Any,
    multiple: int = 8,
) -> Tuple[int, int]:
    """
    Resolve effective Txt2Img generation size.

    If ``long_side`` is provided, the aspect ratio is inferred from the first
    available ControlNet or T2I-Adapter conditioning image.

    Otherwise, the already-resolved ``width`` and ``height`` values are used.

    Parameters
    ----------
    width : int
        Resolved width from the node configuration.

    height : int
        Resolved height from the node configuration.

    long_side : int or None
        Optional target long side. When provided, overrides ``width`` and
        ``height`` using the first conditioning image aspect ratio.

    cn_bundle : Any
        ControlNet bundle.

    t2i_bundle : Any
        T2I-Adapter bundle.

    multiple : int, default=8
        Alignment multiple for generated dimensions.

    Returns
    -------
    tuple[int, int]
        Effective ``(width, height)``.

    Raises
    ------
    ValueError
        If ``long_side`` is provided but no ControlNet or T2I-Adapter image is
        available.
    """
    if long_side is None:
        return int(width), int(height)

    conditioning_image = _first_conditioning_image(
        cn_bundle=cn_bundle,
        t2i_bundle=t2i_bundle,
    )

    if conditioning_image is None:
        raise ValueError(
            "Txt2Img params 'long_side' requires at least one ControlNet or "
            'T2I-Adapter conditioning image.'
        )

    return resolve_long_side_size(
        image=conditioning_image,
        long_side=int(long_side),
        multiple=multiple,
    )


@dataclass
class Txt2Img(
    ImageOutputMixin,
    CudaPostRunMixin,
    T2IAdapterMixin,
    LoraMixin,
    ControlNetMixin,
    PromptMixin,
    NodeRef,
):
    """
    SDXL text-to-image generation node with optional ControlNet, T2I-Adapter, and IP-Adapter/FaceID.

    This node generates an image from text prompts using Stable Diffusion XL. Prompts
    are resolved from the node configuration and may optionally be overridden by an
    upstream prompt bundle attached through the prompt channel (see Notes).

    Depending on the configured conditioning mechanisms, the node selects one of the
    following pipelines:

    - If one or more ControlNet specifications are configured, the node uses
    ``StableDiffusionXLControlNetPipeline`` and conditions the generation on the
    attached ControlNet images (e.g. canny, depth, lineart).
    - Else, if one or more T2I-Adapter specifications are configured, the node uses
    ``StableDiffusionXLAdapterPipeline`` and conditions the generation on the
    attached adapter images.
    - Otherwise, the node falls back to the standard ``StableDiffusionXLPipeline``.

    IP-Adapter / FaceID conditioning (image-based reference guidance) is orthogonal
    to the above choices and may be enabled or disabled independently, based on the
    resolved bundles. IP-Adapter and FaceID are mutually exclusive.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG. If not provided, an identifier is
        automatically generated by the enclosing DAG.
    spec : dict or str or pathlib.Path or sequence of (dict or str or pathlib.Path)
        Node configuration specification.

        The specification may be provided as an in-memory dictionary, as a path to a
        HOCON configuration file, or as a sequence of such elements. When a sequence
        is given, each element is resolved independently and merged from left to
        right, with later elements overriding earlier ones.

        The node uses :func:`resolve_common` to resolve the shared SDXL configuration.
        Expected keys include:

        **Model / runtime**
        Exactly one of the following keys must be provided under ``spec.model``:

        - ``model.id`` : str
            Identifier of a pretrained model (e.g. a Hugging Face repository ID).
            The model will be loaded via the Diffusers "from_pretrained" mechanism.
            If not already cached locally, it will be downloaded on first use.

        - ``model.path`` : str
            Path to a local model file or directory.
            This is typically used for:
            - Single-file checkpoints (e.g. ``.safetensors``)
            - Fully downloaded model directories
            - Custom or fine-tuned local models

            The path is expanded via ``os.path.expanduser`` and must exist locally.

        Providing both ``model.id`` and ``model.path`` is an error.
        Providing neither is also an error.

        Additional runtime options:

        - ``model.device`` : str, optional
            Device to run on (default: ``"cuda"``).
        - ``model.dtype`` : str, optional
            DType name resolved by ``resolve_dtype`` (default: ``"bf16"``).

        **VAE**
        - ``vae`` : str or dict, optional
            Either a VAE id string, or a dict with ``vae.id``.

        **Generation parameters** (under ``params``)
        - ``params.steps`` : int, optional
            Number of denoising steps (default: 30).
        - ``params.cfg`` or ``params.guidance_scale`` : float, optional
            Classifier-free guidance scale (default: 6.0). ``cfg`` takes precedence.
        - ``params.width`` : int, optional
            Output width (default: 1024).
        - ``params.height`` : int, optional
            Output height (default: 1024).
        - ``params.long_side`` : int, optional
            If provided, overrides ``params.width`` and ``params.height`` by
            deriving the aspect ratio from the first available ControlNet or
            T2I-Adapter conditioning image and scaling the longest side to this
            value.

            If both ControlNet and T2I-Adapter are absent, using
            ``params.long_side`` raises an error.

        **Batch and randomness**

        - ``batch`` : int, optional  
            Number of images to generate.

            - Must be greater than 0.
            - If ``seed`` is a list, defaults to ``len(seed)``.
            - If ``seed`` is a single integer, must be ``1``.
            - If ``seed`` is ``"rand"``, controls how many independent random seeds
            are generated.

        - ``seed`` : int or list[int] or str, optional  
            Seed specification controlling stochastic sampling.

            Supported forms:

            - ``int``  
            Single deterministic seed. Implies ``batch = 1``.

            - ``list[int]``  
            Explicit list of seeds.

            - ``"rand"`` or ``"random"``  
            Generate one or more random seeds (controlled by ``batch``).

    Attributes
    ----------
    op : str
        Operator identifier derived from the concrete node class name (e.g.
        ``"txt2img"``).
    controlnet : ControlNetRegistry
        Registry used to declare ControlNet conditioning inputs (via sinks),
        provided by ``ControlNetMixin``.
    t2i_adapter : T2IAdapterRegistry
        Registry used to declare T2I-Adapter conditioning inputs (via sinks),
        provided by ``T2IAdapterMixin``.
    ip_adapter : IpAdapterRegistry
        Registry used to declare IP-Adapter reference inputs (via sinks),
        provided by ``ControlNetMixin``.
    face_id : FaceIdRegistry
        Registry used to declare FaceID reference inputs (via sinks),
        provided by ``ControlNetMixin``.
    prompt : PromptRegistry
        Registry used to attach an upstream prompt bundle, provided by
        ``PromptMixin``.

    Inputs
    ------
    prompt:default : dict, optional
        Optional upstream prompt bundle. If present, it overrides prompt fields from
        ``spec``. Expected keys include ``prompt``, ``prompt_2``,
        ``negative_prompt``, and ``negative_prompt_2``.

    Additional typed inputs may be attached via ControlNet, T2I-Adapter, IP-Adapter,
    or FaceID sinks declared on this node (see Notes).

    Outputs
    -------
    dict
        Primary output dictionary.

        The node may produce either a single image or a batch of images,
        depending on the resolved ``batch`` size.

        **Single image output (batch = 1)**

        - ``ok`` : bool
        - ``node`` : str (e.g. ``"txt2img"``)
        - ``id`` : str (node id)
        - ``image`` : str  
        Path to the generated image.
        - ``seed`` : int  
        Seed used to generate the image.
        - ``metadata`` : str  
        Path to the JSON sidecar.

        **Batch output (batch > 1)**

        - ``ok`` : bool
        - ``node`` : str
        - ``id`` : str
        - ``images`` : list[str]  
        Paths to generated images.
        - ``seeds`` : list[int]  
        Seeds aligned with ``images`` (same order).
        - ``batch`` : int  
        Number of generated images.
        - ``metadata`` : str  
        Path to the JSON sidecar (shared across the batch).

        The metadata sidecar includes resolved parameters, model information,
        timing information, and (when running on CUDA) memory statistics.

        For batch outputs, a single sidecar file is written per node execution
        and anchored to one of the generated images (deterministically chosen).

    Notes
    -----
    - Hugging Face cache and environment variables are configured by the DAG
    runner prior to loading any pipeline components.
    - Prompt resolution is performed by :class:`PromptBundle`: if a prompt bundle is
    wired into ``prompt:default``, it takes precedence over the local ``spec``;
    otherwise the local ``spec`` is used.
    - ControlNet and T2I-Adapter inputs are collected from DAG wiring through their
    respective registries and resolved into runtime bundles before pipeline
    construction.
    - ControlNet and T2I-Adapter are mutually exclusive within the same node.
    Attempting to enable both results in an error.
    - For ``StableDiffusionXLControlNetPipeline`` (txt2img), ControlNet conditioning
    images are passed via the pipeline ``image`` argument together with
    ``controlnet_conditioning_scale`` (via :func:`build_pipe_kwargs`).
    - For ``StableDiffusionXLAdapterPipeline`` (txt2img), adapter conditioning images
    are passed via the pipeline ``image`` argument together with
    ``adapter_conditioning_scale`` (via :func:`build_pipe_kwargs`).
    - IP-Adapter conditioning uses one or more reference images passed through
    ``ip_adapter_image``; FaceID uses precomputed embeddings passed through
    ``ip_adapter_image_embeds``. IP-Adapter and FaceID are mutually exclusive.
    - Outputs are written as an image file plus a JSON sidecar containing run metadata
    (parameters, model info, timing, and optional CUDA memory stats).
    """

    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        return is_cuda_device(spec.get('model', {}).get('device', 'cuda'))

    def run(self, output_dir: str | Path, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        from diffusers import (StableDiffusionXLAdapterPipeline,
                               StableDiffusionXLControlNetPipeline,
                               StableDiffusionXLPipeline)

        ctx = resolve_common(self.spec)

        # --- resolve inputs from DAG wiring ---
        cn_bundle, ip_bundle, face_bundle = self.build_control_bundles(
            input=input,
            device=ctx.model.device,
            dtype=ctx.model.dtype,
        )
        lora_bundle = self.build_lora_bundle()
        t2i_bundle = self.build_t2i_adapter_bundle(
            input=input,
            device=ctx.model.device,
            dtype=ctx.model.dtype,
        )

        base = get_sdxl_base_pipe(
            model_ref=ctx.model.model_ref,
            device=ctx.model.device,
            dtype=ctx.model.dtype,
            vae_id=ctx.model.vae_id,
        )

        if ip_bundle.has_ip_adapter and face_bundle.has_face_id:
            raise ValueError(
                'IP_Adapter and IP-Adapter-FaceID are mutually exclusive (chose one).'
            )

        if cn_bundle.has_controlnet and t2i_bundle.has_t2i_adapter:
            raise ValueError(
                'ControlNet and T2I-Adapter are mutually exclusive in Txt2Img (choose one).'
            )

        if cn_bundle.has_controlnet:
            pipe = StableDiffusionXLControlNetPipeline(
                **base.components,
                controlnet=cn_bundle.controlnet_arg
            )
        elif t2i_bundle.has_t2i_adapter:
            pipe = StableDiffusionXLAdapterPipeline(
                **base.components,
                adapter=t2i_bundle.adapter_model_arg
            )
        else:
            pipe = StableDiffusionXLPipeline(**base.components)

        width, height = _resolve_txt2img_size(
            width=ctx.width,
            height=ctx.height,
            long_side=ctx.long_side,
            cn_bundle=cn_bundle,
            t2i_bundle=t2i_bundle,
        )

        prompt_bundle = PromptBundle(spec=ctx.spec, input=input)

        cuda_prerun(ctx.model.device)
        t0 = time.perf_counter()

        cleanup_adapters(pipe)
        apply_ip_adapter(
            ip_bundle=ip_bundle,
            face_bundle=face_bundle,
            pipe=pipe,
            batch=ctx.batch,
            device=ctx.model.device,
            dtype=ctx.model.dtype,
        )
        apply_lora(lora_bundle=lora_bundle, pipe=pipe)

        pipe_kwargs = build_pipe_kwargs(
            cn_bundle=cn_bundle,
            t2i_bundle=t2i_bundle,
            ip_bundle=ip_bundle,
            face_bundle=face_bundle,
            lora_bundle=lora_bundle,
            height=height,
            width=width,
            device=ctx.model.device,
            dtype=ctx.model.dtype,
        )

        if 'ip_adapter_masks' in pipe_kwargs.get('cross_attention_kwargs', {}):
            l = len(ip_bundle.weight_names_arg) + \
                len(face_bundle.weight_names_arg)
            masks = pipe_kwargs['cross_attention_kwargs']['ip_adapter_masks']
            assert len(masks) == l

        result = pipe(
            prompt=prompt_bundle.prompt,
            prompt_2=prompt_bundle.prompt_2,
            negative_prompt=prompt_bundle.negative_prompt,
            negative_prompt_2=prompt_bundle.negative_prompt_2,
            num_inference_steps=ctx.steps,
            guidance_scale=ctx.cfg,
            generator=ctx.rng.generators,
            width=width,
            height=height,
            num_images_per_prompt=ctx.batch,
            ** pipe_kwargs,
        )

        cuda_sync(ctx.model.device)
        dt_s = time.perf_counter() - t0

        images = result.images
        if len(images) != len(ctx.rng.seeds):
            raise ValueError(
                f'Expected {len(ctx.rng.seeds)} output images, got {len(images)}'
            )

        img_paths = [save_image(
            output_dir,
            node_id=self.id,
            seed=seed,
            img=img,
        ) for img, seed in zip(images, ctx.rng.seeds)]

        mem = cuda_mem_stats(ctx.model.device)

        out = finalize_image_output(
            node_kind=str(self.op),
            node_id=self.id,
            img_path=img_paths,
            seed=ctx.rng.seeds,
            params={
                'steps': ctx.steps,
                'guidance_scale': ctx.cfg,
                'width': ctx.width,
                'height': ctx.height,
            },
            dt_s=dt_s,
            cuda_mem=mem,
            controlnet_specs=self.controlnet.specs,
            t2i_adapter_specs=self.t2i_adapter.specs,
            ip_adapter_specs=self.ip_adapter.specs,
            face_id_specs=self.face_id.specs,
            lora_specs=self.lora.specs,
            model_info={
                'source': ctx.model.model_ref.source,
                'ref': ctx.model.model_ref.ref,
                **({'vae_id': ctx.model.vae_id} if ctx.model.vae_id is not None else {}),
                'dtype': str(ctx.model.dtype).replace('torch.', ''),
                'device': ctx.model.device,
            }
        )

        return out
