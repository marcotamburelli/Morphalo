from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils.mask_ops import prepare_output_mask
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


@dataclass
class AlphaMask(NodeRef):
    """
    Extract an inpaint mask from the transparency of an input image.

    ``AlphaMask`` writes an 8-bit grayscale PNG in the input image's coordinate
    space. Fully transparent pixels map to black (preserve), fully opaque
    pixels map to white (repaint), and partially transparent pixels map to
    intermediate values before optional mask post-processing.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.

    path : str or Path, optional
        Path to a single input image with alpha or transparency data. If
        omitted, the node resolves the upstream default input.

    spec : dict or str or Path or sequence, optional
        Node specification, resolved via ``resolve_spec``.

        ``params`` : dict
            ``close_radius`` : int, optional
                Non-negative morphological closing radius in pixels. Closing
                fills small holes and gaps in the mask. Default is ``0``.

            ``dilate_radius`` : int, optional
                Signed radius in pixels. Positive values expand the white
                region and negative values erode it. Default is ``0``.

            ``smoothing_radius`` : int, optional
                Non-negative Gaussian smoothing radius in pixels. Default is
                ``0``.

    Inputs
    ------
    default : dict, optional
        Upstream image payload, required when ``path`` is omitted. The
        ``image``, ``images`` or ``path`` field must resolve to exactly one
        image.

    Outputs
    -------
    dict
        Output metadata dictionary with the standard ``ok``, ``node`` and
        ``id`` fields. The image and its JSON sidecar are saved separately.

        ``image`` : str
            Path to the grayscale mask PNG.

        ``input_image`` : str
            Path to the source image.

        ``input_size``, ``output_size`` : list[int]
            Dimensions as ``[width, height]``. The mask has the same size as
            the input image.

        ``params`` : dict
            Resolved mask post-processing radii.

        ``metadata`` : str
            Path to the JSON sidecar.

    Notes
    -----
    - Mask post-processing applies closing, dilation or erosion, then Gaussian
      smoothing. A radius of ``0`` disables the corresponding step.
    - Images without alpha or transparency data raise ``ValueError``. An image
      with an alpha channel whose pixels are all opaque produces a white mask.

    Examples
    --------
    Use a semantic crop as both the inpaint image and the source of its mask::

        from morphalo.nodes.preprocess import AlphaMask, Sapiens2SegmentCrop

        crop = Sapiens2SegmentCrop(
            name='hand_crop',
            spec={'params': {'target': 'left-hand', 'crop_mode': 'trim'}},
        )
        mask = AlphaMask(
            name='hand_mask',
            spec={'params': {'dilate_radius': 2}},
        )
        source >> crop
        crop >> inpaint
        crop >> mask >> inpaint.mask()
    """

    path: str | Path | None = None
    spec: SpecInput = field(default_factory=dict)

    def run(
        self,
        output_dir: str | Path,
        input: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        spec_params = resolve_spec(self.spec).get('params', {})
        params = {
            name: int(spec_params.get(name, 0))
            for name in ('close_radius', 'dilate_radius', 'smoothing_radius')
        }
        for name in ('close_radius', 'smoothing_radius'):
            if params[name] < 0:
                raise ValueError(f"'{self.id}': {name} must be >= 0, got {params[name]}")

        source_path = resolve_single_image_path(
            node_id=self.id,
            path=self.path,
            input=input,
        )
        with Image.open(source_path) as source:
            if not source.has_transparency_data:
                raise ValueError(f"'{self.id}': input image must have alpha or transparency data")
            last_band = source.getbands()[-1]
            if last_band in ('A', 'a'):
                alpha = np.asarray(source.getchannel(last_band))
            else:
                alpha = np.asarray(source.convert('RGBA').getchannel('A'))

        mask = Image.fromarray(prepare_output_mask(alpha, **params))
        out_path = make_node_output_path(out_dir=output_dir, node_id=self.id)
        mask.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(source_path),
            'image': str(out_path),
            'input_size': list(mask.size),
            'output_size': list(mask.size),
            'params': params,
        }
        out['metadata'] = str(write_json_sidecar(out_path, out))
        return out
