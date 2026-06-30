from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from morphalo.cache.models import get_dinov2_encoder
from morphalo.core.paths import make_node_output_path
from morphalo.dag import AttachmentSink, NodeRef
from morphalo.nodes.common.config_resolve import (SpecInput, resolve_dtype,
                                                  resolve_spec)
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.sdxl_resolve import (resolve_image_paths,
                                         resolve_single_image_path)

_ROTATIONS = (-60, -45, -30, 0, 30, 45, 60)
_FLIPS = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)


@dataclass(frozen=True)
class Config:
    encoder: str
    device: str
    dtype: torch.dtype
    batch_size: int


@dataclass(frozen=True)
class Variant:
    source_image: str
    candidate_index: int
    rotation: int
    flip_horizontal: bool
    flip_vertical: bool
    image: Image.Image


def _read_cfg(spec: dict, node_id: str) -> Config:
    model = spec.get('model', {})
    params = spec.get('params', {})

    encoder = str(model.get('encoder', 'facebook/dinov2-base'))
    device = str(model.get('device', 'cuda'))
    dtype = resolve_dtype(str(model.get('dtype', 'float32')))

    batch_size = int(params.get('batch_size', 32))
    if batch_size <= 0:
        raise ValueError(
            f"'{node_id}': params.batch_size must be > 0, got {batch_size!r}"
        )

    return Config(
        encoder=encoder,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )


def _batched(items: Iterable[Variant], batch_size: int) -> Iterator[list[Variant]]:
    batch: list[Variant] = []

    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def _fill_transparent_rotation_padding(img: Image.Image) -> Image.Image:
    """
    Fill transparent rotation padding by smoothly extending nearby pixels.

    The selector rotates references with an alpha channel so the newly exposed
    pixels are explicitly marked as unknown. Missing pixels are first filled
    from the nearest valid source pixel, then only the filled region is smoothed.
    Real source pixels remain untouched.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt

    rgba = img.convert('RGBA')
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    missing = (alpha == 0)

    if not np.any(missing):
        return rgba.convert('RGB')

    rgb = arr[:, :, :3]
    _, nearest = distance_transform_edt(
        missing,
        return_distances=True,
        return_indices=True,
    )

    filled = rgb.copy()
    missing_y, missing_x = np.nonzero(missing)
    nearest_y = nearest[0, missing_y, missing_x]
    nearest_x = nearest[1, missing_y, missing_x]
    filled[missing_y, missing_x] = rgb[nearest_y, nearest_x]

    filled = np.ascontiguousarray(filled)
    for _ in range(2):
        blurred = cv2.GaussianBlur(filled, (0, 0), sigmaX=2.0)
        filled[missing] = blurred[missing]

    return Image.fromarray(filled, mode='RGB')


def _transform_image(
    img: Image.Image,
    *,
    flip_horizontal: bool,
    flip_vertical: bool,
    rotation: int,
) -> Image.Image:
    out = img

    if flip_horizontal:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertical:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if rotation != 0:
        out = out.convert('RGBA').rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
        out = _fill_transparent_rotation_padding(out)

    return out


def _iter_variants(candidate_paths: list[str]) -> Iterator[Variant]:
    for candidate_index, source_path in enumerate(candidate_paths):
        source = _open_rgb(source_path)

        for flip_horizontal, flip_vertical in _FLIPS:
            for rotation in _ROTATIONS:
                transformed = _transform_image(
                    source,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    rotation=rotation,
                )
                yield Variant(
                    source_image=source_path,
                    candidate_index=candidate_index,
                    rotation=rotation,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    image=transformed,
                )


def _embed_images(
    images: list[Image.Image],
    *,
    processor,
    model,
    device: str,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors='pt')
    try:
        model_dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        model_dtype = torch.float32

    moved = {}
    for k, v in inputs.items():
        if not hasattr(v, 'to'):
            continue
        if torch.is_floating_point(v):
            moved[k] = v.to(device=device, dtype=model_dtype)
        else:
            moved[k] = v.to(device=device)

    with torch.inference_mode():
        outputs = model(**moved)

    emb = outputs.last_hidden_state[:, 0, :].float()
    return F.normalize(emb, p=2, dim=-1)


@dataclass
class VisualReferenceSelector(NodeRef):
    """
    Select the candidate reference image most similar to a query image.

    ``VisualReferenceSelector`` is a visual reference retrieval node. It is
    designed for DAG workflows where an upstream node provides a small dataset
    of possible reference images, while a lateral input provides the image that
    should be matched against that dataset.

    The default upstream input is interpreted as the reference dataset. Each
    provided reference image is automatically expanded into a fixed set of
    variants by applying horizontal flips, vertical flips, combined flips, and
    coarse rotations. Rotation padding is filled by smoothly extending nearby
    valid pixels into the exposed empty regions, avoiding artificial black or
    flat-color borders. This lets the selector compare the query against
    references that may appear in a different orientation without requiring the
    user to prepare those variants manually.

    The lateral input declared by :meth:`query` provides the query image: a
    single crop or sample image whose visual embedding is used as the search
    target. The node embeds the query and every transformed reference variant
    with DINOv2, compares normalized embeddings with cosine similarity, and
    writes the transformed reference variant with the highest similarity score.

    Downstream nodes receive the selected transformed image via the standard
    ``image`` / ``path`` fields. In other words, downstream consumers do not need
    to know that retrieval, flipping, or rotation happened; they simply receive
    the best matching reference image ready for use.

    This node is intended for workflows where a downstream IP-Adapter or other
    image-conditioned step needs the closest visual reference from a small
    library, for example choosing a foot, hand, weapon, garment, or prop
    reference that best matches a localized crop.

    Parameters
    ----------
    name : str, optional
        Unique node identifier within the DAG.
    spec : dict or str or Path, optional
        Node specification, resolved via ``resolve_spec``.

        Expected structure:

        ``model`` : dict
            ``encoder`` : str, optional
                DINOv2 model id. Default is ``'facebook/dinov2-base'``.
            ``device`` : str, optional
                Model execution device. Default is ``'cuda'``.
            ``dtype`` : str, optional
                Model dtype resolved through ``resolve_dtype``. Default is
                ``'float32'``.

        ``params`` : dict
            ``batch_size`` : int, optional
                Number of candidate variants embedded per forward pass. This
                affects performance only. Default is ``32``.

    Inputs
    ------
    Default input
        Reference dataset. The upstream output must provide ``image``,
        ``images``, or ``path``. Lists are accepted and preserve order. Each
        source image is used as the base for the internally generated
        flip/rotation variants.

    ``query``
        Single query image. The upstream output must provide ``image`` or
        ``path`` and must resolve to exactly one image. This image is not
        augmented; it is embedded as the search target.

    Outputs
    -------
    dict
        Metadata dictionary, also written as a JSON sidecar. The important
        fields are:

        ``image`` / ``path``
            Path to the selected transformed candidate image.

        ``source_image``
            Original candidate image path before flip/rotation.

        ``score``
            Cosine similarity between the query embedding and selected variant
            embedding.

        ``selection``
            Details of the selected transform.
    """

    spec: SpecInput = field(default_factory=dict)

    def query(self) -> AttachmentSink:
        """
        Declare the lateral query-image input for this selector.

        The query image is the sample that the reference dataset should be
        matched against. Wire a node that produces a single image into this sink,
        for example a localized crop from ``SubjectCrop`` or ``AnyCrop``::

            refs >> selector
            foot_crop >> selector.query()

        The upstream query payload must provide one of:

        - ``image`` : str
        - ``path`` : str

        Unlike the default reference-dataset input, the query input must resolve
        to exactly one image. Lists are rejected because the selector produces a
        single best match for one query at a time.

        Returns
        -------
        AttachmentSink
            Sink bound to this node with ``input_id='query'``. Connect a node
            producing the query image to this sink.
        """
        return AttachmentSink(
            name=f'visual_reference_query:{self.id}',
            target=self,
            input_id='query',
        )

    def run(
        self,
        output_dir: str | Path,
        input: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        spec = resolve_spec(self.spec)
        node_id = self.id
        cfg = _read_cfg(spec, node_id=node_id)

        input = input or {}
        candidate_paths = [
            str(p)
            for p in resolve_image_paths(
                node_id=node_id,
                path=None,
                input=input,
                input_key='default',
            )
        ]
        query_path = str(resolve_single_image_path(
            node_id=node_id,
            path=None,
            input=input,
            input_key='query',
        ))

        if not candidate_paths:
            raise ValueError(
                f"VisualReferenceSelector node '{node_id}': no candidate images found."
            )

        processor, model = get_dinov2_encoder(
            model_id=cfg.encoder,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        query_img = _open_rgb(query_path)
        query_emb = _embed_images(
            [query_img],
            processor=processor,
            model=model,
            device=cfg.device,
        )[0]

        best_variant: Optional[Variant] = None
        best_score: Optional[float] = None
        total_variants = 0

        for batch in _batched(_iter_variants(candidate_paths), cfg.batch_size):
            batch_emb = _embed_images(
                [v.image for v in batch],
                processor=processor,
                model=model,
                device=cfg.device,
            )
            scores = torch.matmul(batch_emb, query_emb)

            for variant, score_tensor in zip(batch, scores):
                total_variants += 1
                score = float(score_tensor.detach().cpu().item())
                if best_score is None or score > best_score:
                    best_score = score
                    best_variant = variant

        if best_variant is None or best_score is None:
            raise RuntimeError(
                f"VisualReferenceSelector node '{node_id}': no variants were scored."
            )

        out_dir = Path(output_dir)
        out_path = make_node_output_path(
            out_dir=out_dir,
            node_id=node_id,
            ext='png',
        )
        best_variant.image.save(out_path)

        out = {
            'ok': True,
            'node': self.op,
            'id': node_id,
            'image': str(out_path),
            'path': str(out_path),
            'query_image': query_path,
            'source_image': best_variant.source_image,
            'score': best_score,
            'model': {
                'encoder': cfg.encoder,
                'device': cfg.device,
                'dtype': str(cfg.dtype).replace('torch.', ''),
            },
            'params': {
                'batch_size': cfg.batch_size,
                'rotations': list(_ROTATIONS),
                'flips': [
                    {
                        'horizontal': bool(h),
                        'vertical': bool(v),
                    }
                    for h, v in _FLIPS
                ],
                'background': 'edge_inpaint',
            },
            'selection': {
                'candidate_index': int(best_variant.candidate_index),
                'source_image': best_variant.source_image,
                'score': best_score,
                'rotation': int(best_variant.rotation),
                'flip_horizontal': bool(best_variant.flip_horizontal),
                'flip_vertical': bool(best_variant.flip_vertical),
                'output_size': [
                    int(best_variant.image.size[0]),
                    int(best_variant.image.size[1]),
                ],
            },
            'stats': {
                'candidate_count': len(candidate_paths),
                'variant_count': int(total_variants),
            },
        }

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
