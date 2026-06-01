from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from morphalo.dag import AttachmentSink, NodeRef


@dataclass(frozen=True)
class QwenImageInpaintInputs:
    """
    Runtime image inputs for Qwen image inpainting.

    Parameters
    ----------
    control_image : PIL.Image.Image
        Base image that provides the editing context.
    mask : PIL.Image.Image
        Inpainting mask in ``L`` mode. White pixels indicate the editable area.
    metadata : dict[str, Any]
        JSON-serializable metadata describing the resolved input paths and
        image sizes.
    """
    control_image: Image.Image
    mask: Image.Image
    metadata: Dict[str, Any]


def _payload_image_path(payload: Dict[str, Any], *, input_id: str) -> str:
    """
    Resolve a single image path from a Morphalo upstream payload.

    Parameters
    ----------
    payload : dict[str, Any]
        Upstream node output payload.
    input_id : str
        Input identifier used for diagnostics.

    Returns
    -------
    str
        Resolved filesystem path as a string.

    Raises
    ------
    ValueError
        If the payload does not contain a usable image path, or if a list of
        images is provided where a single image is required.
    TypeError
        If the payload path value has an unsupported type.
    """
    value = payload.get('image') or payload.get(
        'path') or payload.get('images')
    if not value:
        raise ValueError(
            f'Upstream output for {input_id!r} must contain '
            "'image', 'path', or 'images'."
        )

    if isinstance(value, (str, Path)):
        return str(value)

    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f'Input {input_id!r} expects exactly one image, got '
                f'{len(value)} images.'
            )
        return str(value[0])

    raise TypeError(
        f'Unsupported image path payload for {input_id!r}: '
        f'{type(value).__name__}'
    )


class QwenImageInpaintRegistry:
    """
    Single-slot registry for Qwen image inpainting inputs.

    This registry exposes the non-default image input required by
    ``QwenImageInpaint``:

    - ``mask``: required inpainting mask.

    The base image is intentionally not declared here. It is received through
    the node's standard ``default`` input so the DAG reads naturally::

        source >> qwen_inpaint
        mask >> qwen_inpaint.mask()

    Internally, the base image is passed to
    ``QwenImageControlNetInpaintPipeline`` as ``control_image`` because that is
    the name used by the Diffusers backend.
    """

    MASK_INPUT_ID = 'mask'

    def __init__(self, owner: NodeRef):
        self._owner = owner

    def mask(self) -> AttachmentSink:
        """
        Declare the required inpainting mask input.

        Returns
        -------
        AttachmentSink
            Sink targeting input id ``'mask'``. The upstream payload must
            provide a single image path through ``'image'`` or ``'path'``.

        Notes
        -----
        The mask follows Diffusers inpainting convention: white pixels mark
        the area to repaint, black pixels mark the area to preserve.
        """
        return AttachmentSink(
            name=f'qwen_image_inpaint_mask:{self._owner.id}',
            target=self._owner,
            input_id=self.MASK_INPUT_ID,
        )


class QwenImageInpaintBundle:
    """
    Runtime resolver for Qwen image inpainting inputs.

    The bundle resolves the node's incoming DAG payloads into PIL images ready
    for ``QwenImageControlNetInpaintPipeline``.

    Input contract
    --------------
    ``default``
        Required base image. Must provide ``'image'`` or ``'path'``.
        Internally passed to Diffusers as ``control_image``.
    ``mask``
        Required mask image. Must provide ``'image'`` or ``'path'``.
    """

    def __init__(self, *, input: Optional[Dict[str, Dict]] = None):
        self._inputs = self._build(input=input or {})

    def _build(self, *, input: Dict[str, Dict]) -> QwenImageInpaintInputs:
        default_up = input.get('default')
        if default_up is None:
            raise ValueError(
                'QwenImageInpaint requires an input image wired to default.'
            )

        mask_up = input.get(QwenImageInpaintRegistry.MASK_INPUT_ID)
        if mask_up is None:
            raise ValueError(
                'QwenImageInpaint requires a mask image wired to mask().'
            )

        image_path = _payload_image_path(default_up, input_id='default')
        mask_path = _payload_image_path(
            mask_up,
            input_id=QwenImageInpaintRegistry.MASK_INPUT_ID,
        )

        control_image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        metadata = {
            'control_image': {
                'input_id': 'default',
                'path': str(Path(image_path).expanduser()),
                'width': control_image.size[0],
                'height': control_image.size[1],
            },
            'mask': {
                'input_id': QwenImageInpaintRegistry.MASK_INPUT_ID,
                'path': str(Path(mask_path).expanduser()),
                'width': mask.size[0],
                'height': mask.size[1],
            },
        }

        return QwenImageInpaintInputs(
            control_image=control_image,
            mask=mask,
            metadata=metadata,
        )

    @property
    def control_image_arg(self) -> Image.Image:
        return self._inputs.control_image

    @property
    def mask_arg(self) -> Image.Image:
        return self._inputs.mask

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._inputs.metadata


class QwenImageInpaintMixin:
    """
    Adds single-slot inpainting wiring helpers to a node.
    """

    inpaint: QwenImageInpaintRegistry

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inpaint = QwenImageInpaintRegistry(owner=self)

    def mask(self) -> AttachmentSink:
        """
        Declare the inpainting mask input.

        Returns
        -------
        AttachmentSink
            Sink targeting input id ``'mask'``.
        """
        return self.inpaint.mask()

    def build_qwen_image_inpaint_bundle(
        self,
        input: Optional[Dict[str, Dict]],
    ) -> QwenImageInpaintBundle:
        """
        Resolve the node's inpainting image inputs.

        Parameters
        ----------
        input : dict[str, dict] or None
            Runtime DAG input mapping.

        Returns
        -------
        QwenImageInpaintBundle
            Resolved image/mask/control bundle.
        """
        return QwenImageInpaintBundle(input=input or {})
