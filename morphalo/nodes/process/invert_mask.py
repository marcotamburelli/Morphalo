from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image, ImageOps

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


@dataclass
class InvertMask(NodeRef):
    """
    Invert an existing mask without repeating segmentation or cropping.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.
    path : str or Path, optional
        Mask image path. If omitted, read the upstream default input's
        ``image`` or ``path`` field, as provided by crop nodes in ``mode='mask'``.

    Outputs
    -------
    dict
        ``image`` is the inverted mask PNG path. ``input_image`` records the
        source path, and ``input_size`` / ``output_size`` contain
        ``[width, height]``. ``metadata`` points to the JSON sidecar.

    Notes
    -----
    The source is converted to 8-bit grayscale and inverted as ``255 - value``.
    White becomes black, black becomes white, and soft mask edges retain their
    intermediate values. Dimensions are preserved. For color images, luminance
    is used; the alpha channel is not interpreted as a mask.

    Examples
    --------
    Reuse one crop's mask in both polarities::

        from morphalo.nodes.preprocess import SubjectCrop
        from morphalo.nodes.process import InvertMask

        mask = SubjectCrop(name='mask', spec={'params': {'mode': 'mask'}})
        negative = InvertMask(name='negative_mask')
        source >> mask >> negative
    """

    path: Optional[Union[str, Path]] = None

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        mask_path = resolve_single_image_path(
            node_id=self.id,
            path=self.path,
            input=input,
        )

        with Image.open(mask_path) as source:
            inverted = ImageOps.invert(source.convert('L'))

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='png',
        )
        inverted.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'input_image': str(mask_path),
            'image': str(out_path),
            'input_size': list(inverted.size),
            'output_size': list(inverted.size),
        }
        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)
        return out
