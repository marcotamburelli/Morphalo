from __future__ import annotations

import math
from typing import Callable, Optional


def _lightx2v_lightning_scheduler():
    """
    Build the scheduler used by LightX2V's Qwen Lightning Diffusers script.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config({
        'base_image_seq_len': 256,
        'base_shift': math.log(3),
        'invert_sigmas': False,
        'max_image_seq_len': 8192,
        'max_shift': math.log(3),
        'num_train_timesteps': 1000,
        'shift': 1.0,
        'shift_terminal': None,
        'stochastic_sampling': False,
        'time_shift_type': 'exponential',
        'use_beta_sigmas': False,
        'use_dynamic_shifting': True,
        'use_exponential_sigmas': False,
        'use_karras_sigmas': False,
    })


SAMPLING_SCHEDULERS: dict[str, Callable[[], object]] = {
    'lightx2v-lightning': _lightx2v_lightning_scheduler,
}


def resolve_qwen_sampling_scheduler(scheduler_name: Optional[str]):
    """
    Resolve an optional Qwen sampling scheduler by profile name.

    Parameters
    ----------
    scheduler_name : str or None
        Name of the scheduler profile to instantiate. Supported values are
        defined by ``SAMPLING_SCHEDULERS``. When ``None``, no scheduler override
        is requested and this function returns ``None``.

    Returns
    -------
    object or None
        A freshly constructed Diffusers scheduler instance, or ``None`` when no
        sampling scheduler override is requested.

    Raises
    ------
    ValueError
        If ``scheduler_name`` names an unknown scheduler profile.

    Notes
    -----
    This helper does not mutate a pipeline directly. Cache builders should pass
    the returned scheduler into ``from_pretrained`` when constructing scheduler-
    specific pipeline instances.
    """
    if scheduler_name is None:
        return None

    try:
        scheduler_factory = SAMPLING_SCHEDULERS[scheduler_name]
    except KeyError as exc:
        known = ', '.join(sorted(SAMPLING_SCHEDULERS))
        raise ValueError(
            f'Unknown Qwen sampling scheduler {scheduler_name!r}. '
            f'Known schedulers: {known}'
        ) from exc

    return scheduler_factory()
