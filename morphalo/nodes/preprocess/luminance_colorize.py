from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.preprocess.utils.color_ops import (colorize_by_luminance,
                                                       read_color_op_config)
from morphalo.nodes.sdxl_resolve import resolve_single_image_path


@dataclass
class LuminanceColorize(NodeRef):
    """
    Colorize an input image by mapping luminance to a target RGB color.

    ``LuminanceColorize`` is a deterministic preprocessing node. It reads a
    single upstream image, computes each pixel's luma from its RGB channels,
    multiplies that scalar brightness by a target color, blends the result with
    the original RGB pixels according to ``strength``, and writes the result as
    a PNG.

    At full strength, black pixels map to black, white pixels map to the chosen
    color, and intermediate luminance values map to darker versions of that
    color. The alpha channel is preserved unchanged.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.

    path : str or Path, optional
        Input image path. If omitted, the node resolves the upstream default input
        using ``input['default']['image']`` or ``input['default']['path']``.

    spec : dict or str or Path, optional
        Node specification, resolved via ``resolve_spec``.

        Expected structure:

        ``params`` : dict
            ``color`` : str or list[int], optional
                Target RGB color as ``'#rrggbb'`` or ``[r, g, b]`` with channel
                values in ``[0, 255]``. The legacy key ``target_color`` is also
                accepted. Default is ``'#ffffff'``.

            ``strength`` : float, optional
                Blend amount in ``[0, 1]`` between the original RGB pixels and
                the luminance colorized pixels. Default is ``1``.

    Inputs
    ------
    default : dict, optional
        Upstream image payload. Required only when ``path`` is omitted. The
        image path is resolved from ``image`` or ``path``.

    Outputs
    -------
    dict
        Output metadata dictionary, also written as a JSON sidecar.

        The most important fields are:

        ``image`` : str
            Path to the colorized PNG image.

        ``input_size`` / ``output_size`` : list[int]
            Image size as ``[width, height]``. This transform preserves size.

        ``colorize`` : dict
            Resolved colorization metadata, including ``color``, ``hex_color``,
            ``strength``, ``mode`` and alpha handling.

        ``params`` : dict
            Normalized parameter values used for the run.

    Notes
    -----
    - RGB luma is computed with Rec.601 weights:
      ``0.299 * R + 0.587 * G + 0.114 * B``.
    - ``strength = 0`` returns the original RGB pixels.
    - ``strength = 1`` returns the fully luminance-colorized RGB pixels.
    - This node does not run model inference.
    """

    path: Optional[Union[str, Path]] = None
    spec: SpecInput = field(default_factory=dict)

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = read_color_op_config(spec, node_id=node_id)

        img_path = resolve_single_image_path(
            node_id=node_id,
            path=self.path,
            input=input,
        )

        img = Image.open(img_path)
        input_width, input_height = img.size
        colorized = colorize_by_luminance(
            img,
            color=cfg.color,
            strength=cfg.strength,
        )
        out_width, out_height = colorized.size

        out_dir = Path(output_dir)
        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )

        colorized.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'input_image': str(img_path),
            'image': str(out_path),
            'input_size': [int(input_width), int(input_height)],
            'output_size': [int(out_width), int(out_height)],
            'colorize': {
                'color': list(cfg.color),
                'hex_color': '#%02x%02x%02x' % cfg.color,
                'strength': cfg.strength,
                'mode': 'luminance',
                'alpha': 'preserve',
            },
            'params': {
                'color': list(cfg.color),
                'strength': cfg.strength,
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
