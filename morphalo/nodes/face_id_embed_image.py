from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import get_insightface
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.cuda_mem import CudaPostRunMixin
from morphalo.nodes.common.device import is_cuda_device
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import resolve_image_paths


@dataclass
class Config:
    device: str
    model_name: str
    det_size: Tuple[int, int]
    agg: str
    paired: bool
    output_dtype: torch.dtype


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})

    # Model
    model_name = str(model.get('model_name', 'buffalo_l'))
    device = str(model.get('device', 'cpu'))

    det_size = model.get('det_size', (640, 640))
    if isinstance(det_size, (list, tuple)) and len(det_size) == 2:
        det_size = (int(det_size[0]), int(det_size[1]))
    else:
        raise ValueError(
            f"'{node_id}': invalid det_size={det_size!r} (expected [w,h] or (w,h))")

    # Params
    agg = str(params.get('agg', 'mean'))
    if agg not in ('mean', 'first'):
        raise ValueError(
            f"'{node_id}': invalid agg={agg!r} (expected 'mean' or 'first')")

    paired = bool(params.get('paired', True))

    output_dtype = resolve_dtype(params.get('output_dtype') or 'float16')

    return Config(
        device=device,
        model_name=model_name,
        det_size=det_size,
        agg=agg,
        paired=paired,
        output_dtype=output_dtype,
    )


@dataclass
class FaceIdEmbedImage(CudaPostRunMixin, NodeRef):
    """
    Extract and serialize FaceID identity embeddings from one or more reference images.

    This preprocessing node resolves one or more input image paths (either from the
    explicit ``path`` attribute or from the upstream ``input['default']`` payload),
    detects faces using InsightFace, extracts normalized identity embeddings
    (``normed_embedding``), optionally aggregates them, and saves the resulting
    tensor to disk in a format compatible with Diffusers IP-Adapter FaceID pipelines.

    The saved tensor provides the identity embedding component used by Diffusers
    FaceID pipelines. For FaceID Plus / PlusV2, the CLIP-based image component is
    handled separately downstream.

    Input resolution
    ----------------
    Image paths are resolved in the following order:

    1) If ``path`` is provided:
    - it may be a single path (str/Path) or a list of paths.

    2) Otherwise, the node reads the upstream default input:
    - ``input['default']['image']``, `input['default']['images']`` or
      ``input['default']['path']``.
    - the upstream value may be a single path or a list of paths.

    Only filesystem paths are supported here (the node does not accept in-memory
    arrays/tensors as inputs).

    Face selection
    --------------
    If multiple faces are detected in an image, the node selects the largest face
    (by bounding-box area) before computing the embedding.

    Aggregation
    -----------
    When multiple reference images are provided, embeddings are aggregated according
    to ``params.agg``:

    - ``'mean'``: average embeddings across images (recommended).
    - ``'first'``: use the first image only.

    Paired output
    -------------
    If ``params.paired`` is true, the output tensor contains a zero "negative"
    embedding followed by the positive embedding, matching common Diffusers / FaceID
    usage patterns:

    - paired:  ``(2, 1, D)``  where index 0 is negative, index 1 is positive
    - unpaired: ``(1, 1, D)``

    Configuration (spec)
    --------------------
    This node is configured exclusively via ``spec`` (inline dict or resolved spec
    file). Expected keys:

    - ``model`` (dict):
        - ``model_name`` (str): InsightFace model name (default ``'buffalo_l'``).
        - ``det_size`` (list[int,int] | tuple[int,int]): detection resolution
          passed to InsightFace ``prepare`` (default ``(640, 640)``).
        - ``device`` (str):
          Device selector used to configure InsightFace providers (e.g. ``'cpu'``,
          ``'cuda'``, ``'cuda:0'``). Actual provider initialization is handled by
          the global model cache.

    - ``params`` (dict):
        - ``agg`` (str): ``'mean'`` or ``'first'`` (default ``'mean'``).
        - ``paired`` (bool): whether to emit paired embeddings (default ``True``).
        - ``output_dtype`` (str): dtype string resolved by ``resolve_dtype``
        (e.g. ``'float16'``, ``'float32'``, ``'bf16'``). Default ``'float16'``.

    Caching
    -------
    InsightFace ``FaceAnalysis`` is obtained via ``get_insightface(...)`` and is
    cached by (model_name, device, det_size). Only the per-image detection call is
    performed per run.

    Returns
    -------
    dict
        Output dictionary containing at least:

        - ``ok`` (bool): True if execution succeeds.
        - ``node`` (str): operator name.
        - ``id`` (str): node identifier.
        - ``model`` (dict): resolved model settings
        (``model_name``, ``device``, ``det_size``).
        - ``embeds`` (str): path to the saved ``.pt`` tensor.
        - ``n_images`` (int): number of processed images.
        - ``agg`` (str): aggregation strategy used.
        - ``paired`` (bool): paired output flag.
        - ``shape`` (list[int]): saved tensor shape.
        - ``dtype`` (str): saved tensor dtype
        (torch dtype name without ``'torch.'``).
        - ``metadata`` (str): path to the JSON sidecar.

        The node may expose either a single input image or multiple input images,
        depending on how many filesystem paths were resolved.

        **Single image output**

        - ``image`` (str): absolute path to the input image.

        **Multiple images output**

        - ``images`` (list[str]): absolute paths to the input images.

        The order of ``images`` is stable and corresponds to the resolved input
        order.

    Notes
    -----
    - The visual (CLIP-based) component used by FaceID Plus / PlusV2 is not computed
      here; image paths are forwarded for downstream CLIP embedding injection.
    - ``det_size`` controls the resolution used during face detection (not the
      embedding dimensionality). Larger values may improve detection robustness but
      cost performance.
    - This node saves the computed identity embeddings to disk and also exposes
      the original image path(s) in the output payload for downstream nodes that
      may need to reuse the same references.
    """

    path: Union[str, Path, List[Union[str, Path]]] = None
    spec: SpecInput = field(default_factory=dict)

    @property
    def uses_cuda(self) -> bool:
        spec = resolve_spec(self.spec)
        return is_cuda_device(spec.get('model', {}).get('device', 'cpu'))

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        # Local imports to avoid hard dependency if node unused
        import cv2

        spec = resolve_spec(self.spec)

        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        paths = resolve_image_paths(
            node_id=self.id,
            path=self.path,
            input=input
        )

        app = get_insightface(
            model_name=cfg.model_name,
            det_size=cfg.det_size,
            device=cfg.device,
        )

        embs: List[torch.Tensor] = []

        for p in paths:
            with Image.open(p) as pil:
                rgb = np.asarray(pil.convert('RGB'))

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            faces = app.get(bgr)
            if not faces:
                raise ValueError(
                    f"FaceIdEmbedImage node '{self.id}': no face detected in {p}"
                )

            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            e = torch.from_numpy(face.normed_embedding).to(torch.float32)
            embs.append(e)

        # Aggregate into a single positive embedding vector (D,)
        if cfg.agg == 'first' or len(embs) == 1:
            pos_vec = embs[0]
        else:
            eps = 1e-8

            pos_vec = torch.stack(embs, dim=0).mean(dim=0)
            pos_vec = pos_vec / (pos_vec.norm(p=2) + eps)

        # Normalize to 3D tensor: (1, N, D) where N=1 after aggregation
        # (batch_like_dim=1, n_refs=1, embed_dim=D)
        pos = pos_vec.view(1, 1, -1)  # (1, 1, D)

        if cfg.paired:
            neg = torch.zeros_like(pos)        # (1, 1, D)
            id_embeds = torch.cat([neg, pos], dim=0)  # (2, 1, D)
        else:
            id_embeds = pos  # (1, 1, D)

        id_embeds = id_embeds.to(cfg.output_dtype)

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='pt',
        )

        torch.save(id_embeds.cpu(), out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'model': {
                'model_name': cfg.model_name,
                'device': cfg.device,
                'det_size': list(cfg.det_size),
            },
            'embeds': str(out_path),
            'n_images': len(paths),
            'agg': cfg.agg,
            'paired': cfg.paired,
            'shape': list(id_embeds.shape),
            'dtype': str(id_embeds.dtype).replace('torch.', ''),
        }

        if len(paths) > 1:
            out['images'] = [str(p) for p in paths]
        else:
            out['image'] = str(paths[0])

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
