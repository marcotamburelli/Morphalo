from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from morphalo.cache.models import get_sdxl_base_pipe
from morphalo.core.paths import make_node_output_path
from morphalo.dag.core import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_seed, resolve_spec)
from morphalo.nodes.common.io import write_json
from morphalo.nodes.evaluate.helper import clamp01
from morphalo.nodes.evaluate.img_wiring_mixin import ImgBundle, ImgWiringMixin
from morphalo.nodes.sdxl_resolve import ResolvedModelRef, resolve_model_ref
from morphalo.nodes.wiring.mixins import PromptMixin
from morphalo.nodes.wiring.prompt import PromptBundle


@dataclass
class ScorerConfig:
    device: str
    dtype: str
    model_ref: ResolvedModelRef
    vae_id: Optional[str]
    seed: int
    score_weight: float


def _read_cfg(spec: dict, node_id: str) -> ScorerConfig:
    model = spec.get('model', {})
    params = spec.get('params', {})

    device = model.get('device', 'cuda')
    dtype = resolve_dtype(model.get('dtype', 'bf16'))

    model_ref = resolve_model_ref(model)
    vae_id = spec.get('vae', {}).get('id') if isinstance(
        spec.get('vae'),
        dict
    ) else spec.get('vae')

    seed = resolve_seed(spec.get('seed', 'random'))

    score_weight = float(params.get('score_weight', 1.0))

    if score_weight <= 0:
        raise ValueError(f"'{node_id}': params.score_weight must be > 0.")

    return ScorerConfig(
        device=device,
        dtype=dtype,
        model_ref=model_ref,
        vae_id=vae_id,
        seed=seed,
        score_weight=score_weight,
    )


def _score_from_mse_pair(
    *,
    mse_uncond: float,
    mse_cond: float,
) -> dict[str, Any]:
    """
    Convert unconditional and conditional MSE values into a prompt score.

    The score combines:
    - relative gain: how much conditioning improves noise prediction
    - absolute fit: how well the conditional prediction matches the true noise

    Parameters
    ----------
    mse_uncond : float
        Mean squared error of the unconditional noise prediction.
    mse_cond : float
        Mean squared error of the conditional noise prediction.

    Returns
    -------
    dict[str, Any]
        Dictionary containing score, breakdown, features, and flags.
    """
    if mse_uncond < 0 or mse_cond < 0:
        raise ValueError(
            f'Invalid MSE values: mse_uncond={mse_uncond}, mse_cond={mse_cond}.'
        )

    flags: list[str] = []

    rel_gain = (mse_uncond - mse_cond) / max(mse_uncond, 1e-8)
    rel_gain_clamped = clamp01(rel_gain)

    fit = 1.0 / (1.0 + mse_cond)
    fit = clamp01(fit)

    if mse_cond >= mse_uncond:
        flags.append('prompt_did_not_improve_noise_prediction')

    score = 0.5 * rel_gain_clamped + 0.5 * fit

    return {
        'score': clamp01(score),
        'observable': True,
        'breakdown': {
            'rel_gain': float(rel_gain_clamped),
            'fit': float(fit),
        },
        'features': {
            'mse_uncond': float(mse_uncond),
            'mse_cond': float(mse_cond),
            'raw_rel_gain': float(rel_gain),
        },
        'flags': flags,
    }


def _aggregate_prompt_score(
    *,
    prompt_score_res: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate prompt-alignment score into the node-local final score.

    Parameters
    ----------
    prompt_score_res : dict[str, Any]
        Prompt alignment score result.

    Returns
    -------
    dict[str, Any]
        Aggregated score result containing score, weights, and flags.
    """
    flags = list(prompt_score_res.get('flags', []))

    if not prompt_score_res.get('observable', False):
        return {
            'score': 0.0,
            'weights': {
                'prompt_alignment': 0.0,
            },
            'flags': flags,
        }

    return {
        'score': float(prompt_score_res['score']),
        'weights': {
            'prompt_alignment': 1.0,
        },
        'flags': flags,
    }


def _prompt_alignment_from_mse_stats(
    *,
    mse_uncond_list: list[float],
    mse_cond_list: list[float],
) -> dict[str, Any]:
    """
    Convert per-step unconditional and conditional MSE values into a prompt-alignment score.

    Parameters
    ----------
    mse_uncond_list : list[float]
        Per-step unconditional MSE values.
    mse_cond_list : list[float]
        Per-step conditional MSE values.

    Returns
    -------
    dict[str, Any]
        Dictionary containing score, observability, breakdown, features, and flags.
    """
    if len(mse_uncond_list) != len(mse_cond_list):
        raise ValueError('MSE lists must have the same length.')
    if not mse_uncond_list:
        raise ValueError('At least one MSE pair is required.')

    step_results = [
        _score_from_mse_pair(
            mse_uncond=float(mu),
            mse_cond=float(mc),
        )
        for mu, mc in zip(mse_uncond_list, mse_cond_list)
    ]

    scores = [float(res['score']) for res in step_results]
    rel_gains = [float(res['breakdown']['rel_gain']) for res in step_results]
    fits = [float(res['breakdown']['fit']) for res in step_results]

    flags: list[str] = []
    for res in step_results:
        flags.extend(res.get('flags', []))

    observable = all(bool(res.get('observable', False))
                     for res in step_results)

    return {
        'score': float(sum(scores) / len(scores)),
        'observable': observable,
        'breakdown': {
            'rel_gain': float(sum(rel_gains) / len(rel_gains)),
            'fit': float(sum(fits) / len(fits)),
        },
        'features': {
            'mse_uncond_mean': float(sum(mse_uncond_list) / len(mse_uncond_list)),
            'mse_cond_mean': float(sum(mse_cond_list) / len(mse_cond_list)),
            'mse_uncond_list': [float(x) for x in mse_uncond_list],
            'mse_cond_list': [float(x) for x in mse_cond_list],
            'step_scores': scores,
            'num_steps': len(step_results),
        },
        'flags': flags,
    }


def _image_to_sdxl_latents(
    *,
    img_rgb,
    vae,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Encode an RGB image into SDXL latent space.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image with shape ``(H, W, 3)`` and dtype convertible to float32.
    vae : Any
        Diffusers AutoencoderKL.
    device : str
        Target device.
    dtype : torch.dtype
        Target dtype for the diffusion-side latents.

    Returns
    -------
    torch.Tensor
        Latents with shape ``(1, 4, H/8, W/8)`` scaled for SDXL UNet input.
    """
    x = torch.from_numpy(img_rgb).to(device=device, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x * 2.0 - 1.0  # [0,1] -> [-1,1]

    vae_dtype = next(vae.parameters()).dtype
    x = x.to(dtype=vae_dtype)

    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample()

    latents = latents * vae.config.scaling_factor
    return latents.to(dtype=dtype)


def _sdxl_encode_prompt_pair(
    *,
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    prompt_2: Optional[str],
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Encode SDXL prompt conditioning for both conditional and unconditional branches.

    Parameters
    ----------
    pipe : Any
        Prepared StableDiffusionXLPipeline-like object exposing ``encode_prompt``.
    prompt : str
        Primary prompt.
    prompt_2 : str or None
        Secondary prompt.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - prompt_embeds
        - pooled_prompt_embeds
        - negative_prompt_embeds
        - negative_pooled_prompt_embeds
    """
    cond = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt='',
        negative_prompt_2='',
    )

    # Diffusers SDXL encode_prompt returns a tuple whose exact unpacking can vary
    # slightly across versions, but in practice for SDXL the load-bearing values are:
    # prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
    # negative_pooled_prompt_embeds.
    prompt_embeds = cond[0]
    negative_prompt_embeds = cond[1]
    pooled_prompt_embeds = cond[2]
    negative_pooled_prompt_embeds = cond[3]

    return {
        'prompt_embeds': prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
    }


def _sdxl_noise_mse_pair(
    *,
    pipe: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    prompt: str,
    prompt_2: Optional[str],
    timestep: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[float, float]:
    """
    Compute unconditional and conditional noise-prediction MSE for one timestep.

    Parameters
    ----------
    pipe : Any
        Prepared SDXL pipeline exposing ``unet``, ``scheduler``, and
        ``_get_add_time_ids`` or equivalent helper.
    latents : torch.Tensor
        Clean SDXL latents with shape ``(1, 4, H, W)``.
    prompt : str
        Primary prompt.
    prompt_2 : str or None
        Secondary prompt.
    timestep : int
        Diffusion timestep to evaluate.
    generator : torch.Generator, optional
        Generator for deterministic noise sampling.

    Returns
    -------
    tuple[float, float]
        ``(mse_uncond, mse_cond)``
    """
    device = latents.device
    dtype = latents.dtype

    t = torch.tensor([timestep], device=device, dtype=torch.long)
    noise = torch.randn(latents.shape, generator=generator,
                        device=device, dtype=dtype)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    enc = _sdxl_encode_prompt_pair(
        pipe=pipe,
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
    )

    # Scale input as expected by the scheduler / UNet.
    model_input = pipe.scheduler.scale_model_input(noisy_latents, t)

    # SDXL UNet also expects added conditioning via pooled text embeds + size ids.
    h, w = latents.shape[-2] * 8, latents.shape[-1] * 8
    text_encoder_projection_dim = enc['pooled_prompt_embeds'].shape[-1]

    add_time_ids = pipe._get_add_time_ids(
        original_size=(h, w),
        crops_coords_top_left=(0, 0),
        target_size=(h, w),
        dtype=enc['prompt_embeds'].dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)

    added_cond_kwargs_cond = {
        'text_embeds': enc['pooled_prompt_embeds'],
        'time_ids': add_time_ids,
    }
    added_cond_kwargs_uncond = {
        'text_embeds': enc['negative_pooled_prompt_embeds'],
        'time_ids': add_time_ids,
    }

    with torch.no_grad():
        pred_uncond = pipe.unet(
            model_input,
            t,
            encoder_hidden_states=enc['negative_prompt_embeds'],
            added_cond_kwargs=added_cond_kwargs_uncond,
            return_dict=False,
        )[0]

        pred_cond = pipe.unet(
            model_input,
            t,
            encoder_hidden_states=enc['prompt_embeds'],
            added_cond_kwargs=added_cond_kwargs_cond,
            return_dict=False,
        )[0]

    mse_uncond = float(F.mse_loss(pred_uncond.float(), noise.float()).item())
    mse_cond = float(F.mse_loss(pred_cond.float(), noise.float()).item())

    return mse_uncond, mse_cond


def _prompt_alignment_from_timesteps(
    *,
    pipe,
    latents: torch.Tensor,
    prompt: str,
    prompt_2: Optional[str],
    timesteps: list[int],
    generator: Optional[torch.Generator] = None,
) -> dict[str, Any]:
    mse_uncond_list: list[float] = []
    mse_cond_list: list[float] = []

    for t in timesteps:
        mse_u, mse_c = _sdxl_noise_mse_pair(
            pipe=pipe,
            latents=latents,
            prompt=prompt,
            prompt_2=prompt_2,
            timestep=t,
            generator=generator,
        )
        mse_uncond_list.append(mse_u)
        mse_cond_list.append(mse_c)

    return _prompt_alignment_from_mse_stats(
        mse_uncond_list=mse_uncond_list,
        mse_cond_list=mse_cond_list,
    )


@dataclass
class PromptScorer(ImgWiringMixin, PromptMixin, NodeRef):
    """
    Evaluate and rank images based on alignment with a given text prompt using
    a diffusion-based consistency signal.

    ``PromptScorer`` estimates how well each candidate image matches a textual
    description by measuring how strongly the prompt improves the model’s ability
    to predict diffusion noise compared to an unconditional baseline.

    The node can be used both as a standalone selector over multiple candidate
    images and as a stage in a chain of scorer nodes contributing to a global
    ranking.

    Input wiring
    ------------
    Candidate images are attached explicitly via repeated calls to :meth:`image`.

    Each call to ``image()`` creates a new dynamic input sink:

    - first call  -> ``image:0``
    - second call -> ``image:1``
    - third call  -> ``image:2``
    - ...

    Example:

    >>> scorer = PromptScorer(id='best_prompt', spec=...)
    >>> node_a >> scorer.image()
    >>> node_b >> scorer.image()

    Each upstream input must provide either:

    - ``'image'`` : path to an image file
    - or ``'path'`` : path to an image file

    Prompt input and wiring
    -----------------------
    The scorer obtains its textual conditioning through :class:`PromptMixin`,
    which exposes a single prompt input channel (``prompt:default``).

    A prompt can be provided in two ways:

    1. Upstream wiring
        A prompt-producing node can be connected to the scorer:

        >>> prompt_node >> scorer.prompt()

        where ``prompt_node`` is typically a node of type ``Prompt``.

    2. Node specification
        A prompt can be defined directly in the node ``spec`` under the key
        ``prompt``.

    If both sources are present, upstream values take precedence over the
    node ``spec``.

    The scorer operates on:

    - ``prompt``
        Primary textual conditioning (required).

    - ``prompt_2``
        Optional secondary conditioning, typically used for style or global
        rendering hints.

    Negative prompts may be present in the input or spec but are currently
    ignored by this scorer.

    If no valid primary ``prompt`` is provided, the node raises an error.

    Chained scorers
    ---------------
    If upstream nodes already produced a ``rankings`` structure, this node will:

    - reuse the candidate image set from upstream
    - append its own score contribution to each image
    - recompute the final ranking using all accumulated scores

    In this mode, explicit ``image()`` wiring is not required.

    Scoring principle
    -----------------
    The score is based on a diffusion consistency test:

    1. The input image is encoded into SDXL latent space using the VAE.

    2. For a set of diffusion timesteps, random noise is added to the latent.

    3. The model predicts the noise in two conditions:

        - unconditional (no prompt)
        - conditional (with prompt)

    4. The mean squared error (MSE) between predicted and true noise is computed
       for both cases.

    5. A per-step score is derived from:

        - relative gain:
            how much conditioning improves noise prediction

        - absolute fit:
            how well the conditional prediction matches the true noise

    6. Final score is obtained by averaging across timesteps.

    Intuition
    ---------
    If an image is well aligned with the prompt:

    - the conditional model predicts noise more accurately than the unconditional one
    - the relative improvement is consistent across timesteps

    If the image is unrelated to the prompt:

    - conditioning provides little or no improvement
    - MSE difference collapses

    Aggregation
    -----------
    The node produces a single prompt-alignment score per image.

    The final node-local score is:

        score = prompt_alignment_score

    The final node-local score directly corresponds to the prompt alignment signal,
    without additional weighting or sub-components.

    No additional sub-components are used at this stage.

    Cross-node aggregation (when chaining multiple scorers) combines scores using
    a weighted mean, where each node contributes with its ``score_weight``.

    Outputs
    -------
    dict
        Dictionary containing:

        ``image`` : str
            Path of the best-ranked image.

        ``best_score`` : float
            Final aggregated score of the best image.

        ``rankings`` : dict[str, list[dict]]
            Mapping from node id to score contributions. Each entry is a list of:

                - ``image`` : str
                - ``score`` : float
                - ``score_weight`` : float
                - ``flags`` : list[str], optional
                - ``error`` : str, optional

        ``results`` : list[dict]
            Per-candidate results for this node. Each entry contains:

                - ``input_image`` : str
                - ``score`` : float
                - ``prompt_alignment`` : dict
                - ``aggregation`` : dict
                - ``flags`` : list[str]

            The ``prompt_alignment`` block includes:

                - ``score`` : float
                - ``observable`` : bool
                - ``breakdown`` :
                    - ``rel_gain``
                    - ``fit``
                - ``features`` :
                    - per-step MSE values
                    - per-step scores
                - ``flags`` : list[str]

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG. If not provided, an identifier is
        automatically generated by the enclosing DAG.
    spec : dict or str or Path, optional
        Node configuration resolved via ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``id`` or ``path`` : str
                SDXL model identifier or local checkpoint.

            ``device`` : str, optional
                Device used for inference (default: ``'cuda'``).

            ``dtype`` : str, optional
                Floating-point dtype (e.g., ``'bf16'``).

        ``vae`` : str or dict, optional
            Optional VAE override.

        ``seed`` : int or 'random', optional
            Seed used for deterministic noise sampling.

        ``params`` : dict
            ``score_weight`` : float, optional
            Weight of this node in cross-node ranking aggregation.

    Notes
    -----
    - This node does not generate images; it evaluates existing candidates.
    - Negative prompts are not used in scoring to avoid over-constraining the signal.
    - The scoring signal is heuristic and intended for relative ranking, not
    absolute semantic correctness.
    - Results may depend on the chosen timesteps and random seed.
    - The same noise seed is reused across candidates to improve comparability.

    Design rationale
    ----------------
    ``PromptScorer`` measures prompt alignment using the internal mechanics of
    the diffusion model itself, rather than external embedding similarity.

    Compared to CLIP-based approaches, this method:

    - leverages the exact model used for generation
    - captures both semantic and stylistic alignment
    - remains consistent with SDXL’s dual-prompt conditioning

    The node is intended as a complementary signal alongside structural and
    face-based scorers, enabling a more complete and robust selection pipeline.
    """

    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = ImgBundle(
            node_id=node_id,
            img_count=self.image.img_count,
            input=input,
        )

        prompt_bundle = PromptBundle(spec=spec, input=input)

        prompt = prompt_bundle.prompt
        prompt_2 = prompt_bundle.prompt_2

        if not prompt:
            raise ValueError(f"'{node_id}': prompt is required.")

        candidates: list[dict[str, Any]] = []

        pipe = get_sdxl_base_pipe(
            model_ref=cfg.model_ref,
            device=cfg.device,
            dtype=cfg.dtype,
            vae_id=cfg.vae_id,
        )

        timesteps = [300, 500, 700]

        for path in bundle.paths:
            try:
                with Image.open(path) as im:
                    img_rgb = np.array(im.convert('RGB'), copy=True)

                latents = _image_to_sdxl_latents(
                    img_rgb=img_rgb,
                    vae=pipe.vae,
                    device=cfg.device,
                    dtype=cfg.dtype,
                )

                gen = torch.Generator(
                    device=cfg.device).manual_seed(cfg.seed)

                prompt_score_res = _prompt_alignment_from_timesteps(
                    pipe=pipe,
                    latents=latents,
                    prompt=prompt,
                    prompt_2=prompt_2,
                    timesteps=timesteps,
                    generator=gen,
                )

                agg = _aggregate_prompt_score(
                    prompt_score_res=prompt_score_res,
                )

                candidate = {
                    'input_image': path,
                    'score': float(agg['score']),
                    'prompt_alignment': prompt_score_res,
                    'aggregation': agg,
                    'flags': agg['flags'],
                }

            except RuntimeError as e:
                candidate = {
                    'input_image': path,
                    'score': 0.0,
                    'prompt_alignment': {
                        'score': 0.0,
                        'observable': False,
                        'breakdown': {},
                        'features': {},
                        'flags': [f'prompt_runtime_error: {e}'],
                    },
                    'aggregation': {
                        'score': 0.0,
                        'weights': {
                            'prompt_alignment': 0.0,
                        },
                        'flags': [f'prompt_runtime_error: {e}'],
                    },
                    'flags': [f'prompt_runtime_error: {e}'],
                    'error': str(e),
                }

            candidates.append(candidate)

        rankings = bundle.build_rankings(
            candidates=candidates,
            score_weight=cfg.score_weight,
        )

        best_image, best_score = bundle.calculate_best_image(rankings)

        if best_score <= 0.0 or best_image is None:
            raise RuntimeError(
                f'{self.id}: all candidate images failed scoring.'
            )

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'image': best_image,
            'best_score': best_score,
            'model': {
                'source': cfg.model_ref.source,
                'ref': cfg.model_ref.ref,
                **({'vae_id': cfg.vae_id} if cfg.vae_id is not None else {}),
                'dtype': str(cfg.dtype),
                'device': cfg.device,
            },
            'seed': cfg.seed,
            'params': {
                'score_weight': cfg.score_weight,
                'timesteps': timesteps,
            },
            'rankings': rankings,
            'results': candidates,
        }

        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='json',
        )
        meta_path = write_json(out_path, out)
        out['metadata'] = str(meta_path)

        return out
