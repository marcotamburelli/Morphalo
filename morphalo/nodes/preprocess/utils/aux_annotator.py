from typing import Any

from morphalo.cache.models import get_controlnet_aux_annotator
from third_party.controlnet_aux.processor import MODELS


def build_aux_annotator(processor: str, device: str) -> Any:
    """
    Build or retrieve a ``controlnet-aux`` annotator.

    Parameters
    ----------
    processor : str
        Processor identifier registered by ``controlnet-aux``.
    device : str
        Device used for checkpoint-backed annotators.

    Returns
    -------
    Any
        Cached checkpoint-backed annotator or a new stateless processor.

    Raises
    ------
    ValueError
        If the processor is unknown or unsupported by the bundled backend.
    """
    if processor not in MODELS:
        raise ValueError(
            f'Unknown processor={processor}. Allowed: {list(MODELS.keys())}'
        )

    model = MODELS[processor]

    # The bundled DWPose class has no from_pretrained implementation and
    # therefore cannot use the shared checkpoint-backed annotator cache.
    if processor == 'dwpose':
        raise ValueError(
            'dwpose is not available out-of-the-box in controlnet-aux '
            '(no from_pretrained). Use openpose_* for now, or install a '
            'dedicated DWPose backend later.'
        )

    cls = model['class']

    if model['checkpoint']:
        return get_controlnet_aux_annotator(
            processor=processor,
            cls=cls,
            device=device,
        )

    return cls()
