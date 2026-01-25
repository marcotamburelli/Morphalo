from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from stability.dag import NodeRef
from stability.nodes.utils import resolve_spec


@dataclass
class FaceIdEmbedImage(NodeRef):
    """
    Extract and serialize FaceID embeddings from one or more reference images.

    This preprocessing node loads one or more images, detects a face using
    InsightFace, extracts normalized identity embeddings, optionally aggregates
    them, and serializes the result in a tensor format compatible with
    Diffusers IP-Adapter FaceID pipelines.

    The produced tensor can be directly consumed as
    ``ip_adapter_image_embeds`` for FaceID, FaceID Plus, and FaceID PlusV2
    adapters.

    Notes
    -----
    - Only the first detected face per image is used.
    - Face embeddings are extracted using InsightFace ``normed_embedding``,
      which encodes identity information and is largely invariant to pose
      and lighting.
    - When ``paired=True``, the output tensor includes a zero "negative"
      embedding followed by the positive embedding, matching the canonical
      Hugging Face FaceID examples.
    - The visual (CLIP-based) component required by FaceID Plus / PlusV2 is
      *not* computed here; original image paths are forwarded for downstream
      CLIP embedding injection.

    Parameters
    ----------
    path : str or pathlib.Path or list of (str or pathlib.Path)
        Path(s) to one or more reference images containing a face.
        If multiple images are provided, they are aggregated according
        to ``agg``.

    spec : dict or str or pathlib.Path, optional
        Optional node specification. Only the ``device`` key is used here
        (e.g. ``"cpu"`` or ``"cuda"``) to configure InsightFace providers.

    model_name : str, default="buffalo_l"
        InsightFace model identifier used for face detection and embedding
        extraction.

    det_size : tuple of int, default=(640, 640)
        Detection resolution passed to InsightFace. Larger values may improve
        detection robustness at the cost of performance.

    agg : {"mean", "first"}, default="mean"
        Aggregation strategy when multiple reference images are provided:

        - ``"mean"``: average all extracted embeddings (recommended).
        - ``"first"``: use only the first image.

    paired : bool, default=True
        Whether to produce paired embeddings in the format
        ``[negative, positive]``. When enabled, the output tensor has
        shape ``(2, 1, D)`` and is suitable for FaceID adapters expecting
        explicit negative conditioning.

    output_dtype : {"float16", "float32"}, default="float16"
        Data type used when saving the embedding tensor to disk.

    Attributes
    ----------
    None

    Returns
    -------
    dict
        Output dictionary with the following keys:

        - ``ok`` : bool  
          Always ``True`` if execution succeeds.

        - ``node`` : str  
          Node operator name.

        - ``id`` : str  
          Node identifier.

        - ``embeds`` : str  
          Path to the saved ``.pt`` file containing the FaceID embeddings.

        - ``image`` : str or list of str  
          Original image path(s), forwarded for downstream CLIP-based
          FaceID Plus / PlusV2 processing.

        - ``n_images`` : int  
          Number of input images processed.

        - ``agg`` : str  
          Aggregation strategy used.

        - ``paired`` : bool  
          Whether paired embeddings were produced.

        - ``model`` : str  
          InsightFace model name.

        - ``shape`` : list of int  
          Shape of the saved tensor, typically:

              - ``(2, 1, D)`` if ``paired=True``
              - ``(1, 1, D)`` if ``paired=False``

        - ``dtype`` : str  
          Torch dtype of the saved tensor.

    Tensor Shape Conventions
    ------------------------
    The saved embedding tensor follows the convention expected by Diffusers:

    - Paired embeddings:
        ``(2, N, D)``
        where index 0 is the negative embedding and index 1 is the positive one.

    - Unpaired embeddings:
        ``(1, N, D)``

    In this node, after aggregation, ``N`` is typically ``1``.

    See Also
    --------
    diffusers.loaders.IPAdapterMixin
    insightface.app.FaceAnalysis
    """

    path: Union[str, Path, List[Union[str, Path]]]
    spec: Union[Dict[str, Any], str, Path] = field(default_factory=dict)

    # insightface settings
    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = (640, 640)

    # aggregation across multiple images: "mean" or "first"
    agg: str = "mean"

    # create paired embeds (neg + pos) as in HF examples
    paired: bool = True

    # output dtype for saved tensor
    output_dtype: str = "float16"  # "float16" | "float32"

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)
        device = spec.get("device", "cpu")

        # Local imports to avoid hard dependency if node unused
        import cv2
        from insightface.app import FaceAnalysis

        # normalize paths
        if not isinstance(self.path, list):
            paths = [self.path]
        else:
            paths = self.path

        paths = [Path(str(p)).expanduser().resolve() for p in paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"FaceIdEmbedImage node '{self.id}': file not found: {p}")
            if not p.is_file():
                raise FileNotFoundError(f"FaceIdEmbedImage node '{self.id}': not a file: {p}")

        if isinstance(device, str) and device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # setup insightface
        app = FaceAnalysis(name=self.model_name, providers=providers)
        # For CPU providers, ctx_id should be -1; for CUDA it's typically 0.
        ctx_id = 0 if ("CUDAExecutionProvider" in providers) else -1
        app.prepare(ctx_id=ctx_id, det_size=self.det_size)

        embs: List[torch.Tensor] = []

        for p in paths:
            pil = Image.open(p).convert("RGB")
            rgb = np.asarray(pil)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            faces = app.get(bgr)
            if not faces:
                raise ValueError(f"FaceIdEmbedImage node '{self.id}': no face detected in {p}")

            # normed_embedding: np.ndarray shape (D,)
            e = torch.from_numpy(faces[0].normed_embedding).to(torch.float32)
            embs.append(e)

        if self.agg not in ("mean", "first"):
            raise ValueError(
                f"FaceIdEmbedImage node '{self.id}': invalid agg={self.agg!r} (use 'mean' or 'first')"
            )

        # Aggregate into a single positive embedding vector (D,)
        if self.agg == "first" or len(embs) == 1:
            pos_vec = embs[0]
        else:
            pos_vec = torch.stack(embs, dim=0).mean(dim=0)

        # Normalize to 3D tensor: (1, N, D) where N=1 after aggregation
        # (batch_like_dim=1, n_refs=1, embed_dim=D)
        pos = pos_vec.view(1, 1, -1)  # (1, 1, D)

        if self.paired:
            neg = torch.zeros_like(pos)        # (1, 1, D)
            id_embeds = torch.cat([neg, pos], dim=0)  # (2, 1, D)
        else:
            id_embeds = pos  # (1, 1, D)

        # dtype
        if self.output_dtype == "float16":
            id_embeds = id_embeds.to(torch.float16)
        elif self.output_dtype == "float32":
            id_embeds = id_embeds.to(torch.float32)
        else:
            raise ValueError(
                f"FaceIdEmbedImage node '{self.id}': invalid output_dtype={self.output_dtype!r}"
            )

        out_dir = Path(str(output_dir)).expanduser().resolve() / "faceid"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"faceid_{self.id}.pt"

        torch.save(id_embeds.cpu(), out_path)

        return {
            "ok": True,
            "node": self.op,
            "id": self.id,
            "embeds": str(out_path),
            "image": [str(p) for p in paths] if len(paths) > 1 else str(paths[0]),
            "n_images": len(paths),
            "agg": self.agg,
            "paired": self.paired,
            "model": self.model_name,
            "shape": list(id_embeds.shape),
            "dtype": str(id_embeds.dtype).replace("torch.", ""),
        }
