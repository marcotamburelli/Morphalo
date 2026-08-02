from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from PIL import Image

from morphalo.cache.models import get_sapiens2_segmenter
from morphalo.nodes.preprocess.utils.mask_selection import \
    select_image_side_mask_candidate

TargetSpec = str | list[str] | tuple[str, ...]
SideMode = Literal['none', 'anatomical', 'image-relative']
ImageSide = Literal['left', 'right']


@dataclass(frozen=True)
class SegmentCandidate:
    """
    Candidate group of one or more Sapiens2 model classes.
    """
    name: str
    label_names: tuple[str, ...]
    label_ids: tuple[int, ...]


@dataclass(frozen=True)
class SegmentTarget:
    """
    Public Sapiens2 target definition.
    """
    candidates: tuple[SegmentCandidate, ...]
    side_mode: SideMode = 'none'
    image_side: Optional[ImageSide] = None


@dataclass(frozen=True)
class ResolvedSegmentTarget:
    """
    Fully resolved target after optional image-relative side selection.
    """
    labels: list[str]
    label_ids: list[int]
    selected_candidates: list[str]
    side_resolutions: list[dict[str, Any]]


@dataclass(frozen=True)
class ParsedTargetItem:
    """
    Validated public target entry from ``params.target``.
    """
    name: str
    spec: SegmentTarget


@dataclass(frozen=True)
class ParsedSegmentTarget:
    """
    Normalized ``params.target`` representation stored in node config.
    """
    raw: TargetSpec
    items: tuple[ParsedTargetItem, ...]


SAPIENS2_CLASSES: dict[str, int] = {
    'background': 0,
    'apparel': 1,
    'eyeglass': 2,
    'face-neck': 3,
    'hair': 4,
    'left-foot': 5,
    'left-hand': 6,
    'left-lower-arm': 7,
    'left-lower-leg': 8,
    'left-shoe': 9,
    'left-sock': 10,
    'left-upper-arm': 11,
    'left-upper-leg': 12,
    'lower-clothing': 13,
    'right-foot': 14,
    'right-hand': 15,
    'right-lower-arm': 16,
    'right-lower-leg': 17,
    'right-shoe': 18,
    'right-sock': 19,
    'right-upper-arm': 20,
    'right-upper-leg': 21,
    'torso': 22,
    'upper-clothing': 23,
    'lower-lip': 24,
    'upper-lip': 25,
    'lower-teeth': 26,
    'upper-teeth': 27,
    'tongue': 28,
}


def _candidate(name: str, *label_names: str) -> SegmentCandidate:
    return SegmentCandidate(
        name=name,
        label_names=tuple(label_names),
        label_ids=tuple(SAPIENS2_CLASSES[label] for label in label_names),
    )


def _target(name: str, *label_names: str) -> SegmentTarget:
    return SegmentTarget(candidates=(_candidate(name, *label_names),))


def _anatomical_target(name: str, *label_names: str) -> SegmentTarget:
    return SegmentTarget(
        candidates=(_candidate(name, *label_names),),
        side_mode='anatomical',
    )


def _image_side_target(
    *,
    image_side: ImageSide,
    left_name: str,
    left_labels: tuple[str, ...],
    right_name: str,
    right_labels: tuple[str, ...],
) -> SegmentTarget:
    return SegmentTarget(
        candidates=(
            _candidate(left_name, *left_labels),
            _candidate(right_name, *right_labels),
        ),
        side_mode='image-relative',
        image_side=image_side,
    )


SAPIENS2_TARGETS: dict[str, SegmentTarget] = {
    'background': _target('background', 'background'),
    'apparel': _target('apparel', 'apparel'),
    'eyeglass': _target('eyeglass', 'eyeglass'),
    'face-neck': _target('face-neck', 'face-neck'),
    'hair': _target('hair', 'hair'),
    'lower-clothing': _target('lower-clothing', 'lower-clothing'),
    'torso': _target('torso', 'torso'),
    'upper-clothing': _target('upper-clothing', 'upper-clothing'),
    'lower-lip': _target('lower-lip', 'lower-lip'),
    'upper-lip': _target('upper-lip', 'upper-lip'),
    'lower-teeth': _target('lower-teeth', 'lower-teeth'),
    'upper-teeth': _target('upper-teeth', 'upper-teeth'),
    'tongue': _target('tongue', 'tongue'),
    'anatomical-left-foot': _anatomical_target(
        'anatomical-left-foot', 'left-foot'),
    'anatomical-right-foot': _anatomical_target(
        'anatomical-right-foot', 'right-foot'),
    'anatomical-left-hand': _anatomical_target(
        'anatomical-left-hand', 'left-hand'),
    'anatomical-right-hand': _anatomical_target(
        'anatomical-right-hand', 'right-hand'),
    'anatomical-left-lower-arm': _anatomical_target(
        'anatomical-left-lower-arm', 'left-lower-arm'),
    'anatomical-right-lower-arm': _anatomical_target(
        'anatomical-right-lower-arm', 'right-lower-arm'),
    'anatomical-left-upper-arm': _anatomical_target(
        'anatomical-left-upper-arm', 'left-upper-arm'),
    'anatomical-right-upper-arm': _anatomical_target(
        'anatomical-right-upper-arm', 'right-upper-arm'),
    'anatomical-left-lower-leg': _anatomical_target(
        'anatomical-left-lower-leg', 'left-lower-leg'),
    'anatomical-right-lower-leg': _anatomical_target(
        'anatomical-right-lower-leg', 'right-lower-leg'),
    'anatomical-left-upper-leg': _anatomical_target(
        'anatomical-left-upper-leg', 'left-upper-leg'),
    'anatomical-right-upper-leg': _anatomical_target(
        'anatomical-right-upper-leg', 'right-upper-leg'),
    'anatomical-left-shoe': _anatomical_target(
        'anatomical-left-shoe', 'left-shoe'),
    'anatomical-right-shoe': _anatomical_target(
        'anatomical-right-shoe', 'right-shoe'),
    'anatomical-left-sock': _anatomical_target(
        'anatomical-left-sock', 'left-sock'),
    'anatomical-right-sock': _anatomical_target(
        'anatomical-right-sock', 'right-sock'),
    'anatomical-left-arm': _anatomical_target(
        'anatomical-left-arm',
        'left-lower-arm',
        'left-upper-arm',
    ),
    'anatomical-right-arm': _anatomical_target(
        'anatomical-right-arm',
        'right-lower-arm',
        'right-upper-arm',
    ),
    'anatomical-left-leg': _anatomical_target(
        'anatomical-left-leg',
        'left-lower-leg',
        'left-upper-leg',
    ),
    'anatomical-right-leg': _anatomical_target(
        'anatomical-right-leg',
        'right-lower-leg',
        'right-upper-leg',
    ),
    'left-foot': _image_side_target(
        image_side='left',
        left_name='anatomical-left-foot',
        left_labels=('left-foot',),
        right_name='anatomical-right-foot',
        right_labels=('right-foot',),
    ),
    'right-foot': _image_side_target(
        image_side='right',
        left_name='anatomical-left-foot',
        left_labels=('left-foot',),
        right_name='anatomical-right-foot',
        right_labels=('right-foot',),
    ),
    'left-hand': _image_side_target(
        image_side='left',
        left_name='anatomical-left-hand',
        left_labels=('left-hand',),
        right_name='anatomical-right-hand',
        right_labels=('right-hand',),
    ),
    'right-hand': _image_side_target(
        image_side='right',
        left_name='anatomical-left-hand',
        left_labels=('left-hand',),
        right_name='anatomical-right-hand',
        right_labels=('right-hand',),
    ),
    'left-lower-arm': _image_side_target(
        image_side='left',
        left_name='anatomical-left-lower-arm',
        left_labels=('left-lower-arm',),
        right_name='anatomical-right-lower-arm',
        right_labels=('right-lower-arm',),
    ),
    'right-lower-arm': _image_side_target(
        image_side='right',
        left_name='anatomical-left-lower-arm',
        left_labels=('left-lower-arm',),
        right_name='anatomical-right-lower-arm',
        right_labels=('right-lower-arm',),
    ),
    'left-upper-arm': _image_side_target(
        image_side='left',
        left_name='anatomical-left-upper-arm',
        left_labels=('left-upper-arm',),
        right_name='anatomical-right-upper-arm',
        right_labels=('right-upper-arm',),
    ),
    'right-upper-arm': _image_side_target(
        image_side='right',
        left_name='anatomical-left-upper-arm',
        left_labels=('left-upper-arm',),
        right_name='anatomical-right-upper-arm',
        right_labels=('right-upper-arm',),
    ),
    'left-lower-leg': _image_side_target(
        image_side='left',
        left_name='anatomical-left-lower-leg',
        left_labels=('left-lower-leg',),
        right_name='anatomical-right-lower-leg',
        right_labels=('right-lower-leg',),
    ),
    'right-lower-leg': _image_side_target(
        image_side='right',
        left_name='anatomical-left-lower-leg',
        left_labels=('left-lower-leg',),
        right_name='anatomical-right-lower-leg',
        right_labels=('right-lower-leg',),
    ),
    'left-upper-leg': _image_side_target(
        image_side='left',
        left_name='anatomical-left-upper-leg',
        left_labels=('left-upper-leg',),
        right_name='anatomical-right-upper-leg',
        right_labels=('right-upper-leg',),
    ),
    'right-upper-leg': _image_side_target(
        image_side='right',
        left_name='anatomical-left-upper-leg',
        left_labels=('left-upper-leg',),
        right_name='anatomical-right-upper-leg',
        right_labels=('right-upper-leg',),
    ),
    'left-shoe': _image_side_target(
        image_side='left',
        left_name='anatomical-left-shoe',
        left_labels=('left-shoe',),
        right_name='anatomical-right-shoe',
        right_labels=('right-shoe',),
    ),
    'right-shoe': _image_side_target(
        image_side='right',
        left_name='anatomical-left-shoe',
        left_labels=('left-shoe',),
        right_name='anatomical-right-shoe',
        right_labels=('right-shoe',),
    ),
    'left-sock': _image_side_target(
        image_side='left',
        left_name='anatomical-left-sock',
        left_labels=('left-sock',),
        right_name='anatomical-right-sock',
        right_labels=('right-sock',),
    ),
    'right-sock': _image_side_target(
        image_side='right',
        left_name='anatomical-left-sock',
        left_labels=('left-sock',),
        right_name='anatomical-right-sock',
        right_labels=('right-sock',),
    ),
    'left-arm': _image_side_target(
        image_side='left',
        left_name='anatomical-left-arm',
        left_labels=('left-lower-arm', 'left-upper-arm'),
        right_name='anatomical-right-arm',
        right_labels=('right-lower-arm', 'right-upper-arm'),
    ),
    'right-arm': _image_side_target(
        image_side='right',
        left_name='anatomical-left-arm',
        left_labels=('left-lower-arm', 'left-upper-arm'),
        right_name='anatomical-right-arm',
        right_labels=('right-lower-arm', 'right-upper-arm'),
    ),
    'left-leg': _image_side_target(
        image_side='left',
        left_name='anatomical-left-leg',
        left_labels=('left-lower-leg', 'left-upper-leg'),
        right_name='anatomical-right-leg',
        right_labels=('right-lower-leg', 'right-upper-leg'),
    ),
    'right-leg': _image_side_target(
        image_side='right',
        left_name='anatomical-left-leg',
        left_labels=('left-lower-leg', 'left-upper-leg'),
        right_name='anatomical-right-leg',
        right_labels=('right-lower-leg', 'right-upper-leg'),
    ),
    'arms': _target(
        'arms',
        'left-lower-arm',
        'left-upper-arm',
        'right-lower-arm',
        'right-upper-arm',
    ),
    'body': _target(
        'body',
        'face-neck',
        'hair',
        'left-foot',
        'left-hand',
        'left-lower-arm',
        'left-lower-leg',
        'left-upper-arm',
        'left-upper-leg',
        'right-foot',
        'right-hand',
        'right-lower-arm',
        'right-lower-leg',
        'right-upper-arm',
        'right-upper-leg',
        'torso',
        'lower-lip',
        'upper-lip',
        'lower-teeth',
        'upper-teeth',
        'tongue',
    ),
    'clothes': _target('clothes', 'apparel', 'lower-clothing', 'upper-clothing'),
    'feet': _target(
        'feet',
        'left-foot',
        'left-shoe',
        'left-sock',
        'right-foot',
        'right-shoe',
        'right-sock',
    ),
    'footwear': _target(
        'footwear', 'left-shoe', 'left-sock', 'right-shoe', 'right-sock'),
    'hands': _target('hands', 'left-hand', 'right-hand'),
    'head': _target(
        'head',
        'face-neck',
        'hair',
        'lower-lip',
        'upper-lip',
        'lower-teeth',
        'upper-teeth',
        'tongue',
    ),
    'legs': _target(
        'legs',
        'left-lower-leg',
        'left-upper-leg',
        'right-lower-leg',
        'right-upper-leg',
    ),
    'lower-body': _target(
        'lower-body',
        'left-foot',
        'left-lower-leg',
        'left-upper-leg',
        'right-foot',
        'right-lower-leg',
        'right-upper-leg',
    ),
    'mouth': _target(
        'mouth', 'lower-lip', 'upper-lip', 'lower-teeth', 'upper-teeth', 'tongue'),
    'person': _target(
        'person',
        *(label for label in SAPIENS2_CLASSES if label != 'background'),
    ),
    'skin': _target(
        'skin',
        'face-neck',
        'left-foot',
        'left-hand',
        'left-lower-arm',
        'left-lower-leg',
        'left-upper-arm',
        'left-upper-leg',
        'right-foot',
        'right-hand',
        'right-lower-arm',
        'right-lower-leg',
        'right-upper-arm',
        'right-upper-leg',
        'torso',
        'lower-lip',
        'upper-lip',
        'lower-teeth',
        'upper-teeth',
        'tongue',
    ),
    'upper-body': _target(
        'upper-body',
        'face-neck',
        'hair',
        'left-hand',
        'left-lower-arm',
        'left-upper-arm',
        'right-hand',
        'right-lower-arm',
        'right-upper-arm',
        'torso',
    ),
}


def parse_target_specs(
    target: TargetSpec,
    *,
    node_id: str,
) -> ParsedSegmentTarget:
    """
    Parse and validate a public Sapiens2 target specification.
    """
    expected = ', '.join(repr(t) for t in SAPIENS2_TARGETS)

    if isinstance(target, str):
        items = [target]
    elif isinstance(target, (list, tuple)):
        if not target:
            raise ValueError(f"'{node_id}': target sequence cannot be empty")
        items = list(target)
    else:
        raise ValueError(
            f"'{node_id}': invalid target={target!r} "
            f"(expected a string, list of strings, or tuple of strings; "
            f"valid Sapiens2 values: {expected})"
        )

    parsed_items: list[ParsedTargetItem] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(
                f"'{node_id}': invalid target item={item!r} "
                f"(expected a string; valid Sapiens2 values: {expected})"
            )
        if item not in SAPIENS2_TARGETS:
            raise ValueError(
                f"'{node_id}': invalid target={item!r} "
                f"(expected one of: {expected})"
            )
        parsed_items.append(
            ParsedTargetItem(name=item, spec=SAPIENS2_TARGETS[item])
        )

    return ParsedSegmentTarget(raw=target, items=tuple(parsed_items))


def resolve_parsed_target(
    parsed_target: ParsedSegmentTarget,
    *,
    node_id: str,
    segments: np.ndarray,
    error_prefix: str = 'Sapiens2',
) -> ResolvedSegmentTarget:
    """
    Resolve parsed target specs against a concrete segmentation map.
    """
    label_ids: list[int] = []
    labels: list[str] = []
    selected_candidates: list[str] = []
    side_resolutions: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _add_candidate(candidate: SegmentCandidate) -> None:
        selected_candidates.append(candidate.name)
        for label_name, label_id in zip(candidate.label_names, candidate.label_ids):
            if label_id not in seen:
                labels.append(label_name)
                label_ids.append(int(label_id))
                seen.add(int(label_id))

    for item in parsed_target.items:
        spec = item.spec
        if spec.side_mode == 'image-relative':
            selected, side_meta = select_image_side_candidate(
                item.name,
                spec,
                segments,
                node_id=node_id,
                error_prefix=error_prefix,
            )
            _add_candidate(selected)
            side_resolutions.append(side_meta)
        else:
            for candidate in spec.candidates:
                _add_candidate(candidate)

    return ResolvedSegmentTarget(
        labels=labels,
        label_ids=label_ids,
        selected_candidates=selected_candidates,
        side_resolutions=side_resolutions,
    )


def candidate_mask(
    segments: np.ndarray,
    candidate: SegmentCandidate,
) -> np.ndarray:
    return np.isin(
        segments,
        np.asarray(candidate.label_ids, dtype=np.int64),
    )


def select_image_side_candidate(
    requested_target: str,
    spec: SegmentTarget,
    segments: np.ndarray,
    *,
    node_id: str,
    error_prefix: str = 'Sapiens2',
) -> tuple[SegmentCandidate, dict[str, Any]]:
    """
    Select the leftmost or rightmost non-empty anatomical candidate.
    """
    if spec.image_side not in ('left', 'right'):
        raise ValueError(
            f"'{node_id}': image-relative target={requested_target!r} "
            'is missing image_side'
        )

    selection = select_image_side_mask_candidate(
        [
            (candidate.name, candidate_mask(segments, candidate))
            for candidate in spec.candidates
        ],
        image_side=spec.image_side,
        node_id=node_id,
        target_name=requested_target,
        error_prefix=error_prefix,
    )
    selected = spec.candidates[selection.selected_index]

    return selected, {
        'target': requested_target,
        'mode': spec.side_mode,
        'image_side': spec.image_side,
        'selected_candidate': selected.name,
        'selected_label_ids': [int(label_id) for label_id in selected.label_ids],
        'candidate_centers_x': selection.candidate_centers_x,
    }


def validate_model_num_labels(
    model: Any,
    *,
    model_id: str,
    error_prefix: str = 'Sapiens2',
) -> None:
    """
    Validate that a loaded model matches the Sapiens2 segmentation taxonomy.
    """
    config = getattr(model, 'config', None)
    num_labels = getattr(config, 'num_labels', None)
    if num_labels is None:
        id2label = getattr(config, 'id2label', None)
        num_labels = len(id2label) if id2label else None
    expected = len(SAPIENS2_CLASSES)
    if num_labels != expected:
        raise ValueError(
            f"{error_prefix}: model {model_id!r} exposes "
            f"num_labels={num_labels!r}, expected {expected} for the "
            'Sapiens2 body-part segmentation taxonomy.'
        )


def segment_part_masks(
    segments: np.ndarray,
    label_ids: list[int],
) -> list[np.ndarray]:
    """
    Build per-label masks for part-wise structural cleanup.
    """
    return [
        segments == int(label_id)
        for label_id in label_ids
        if np.any(segments == int(label_id))
    ]


def move_inputs_to_device(
    inputs: Dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Move processor outputs to the runtime device and model dtype.
    """
    return {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value.to(device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }


def predict_segments(
    image: Image.Image,
    *,
    model_id: str,
    device: str,
    dtype: torch.dtype,
    error_prefix: str = 'Sapiens2',
) -> tuple[np.ndarray, torch.dtype]:
    """
    Run Sapiens2 semantic segmentation and return class predictions.
    """
    processor, model = get_sapiens2_segmenter(
        model_id=model_id,
        device=device,
        dtype=dtype,
    )
    model_dtype = next(model.parameters()).dtype
    validate_model_num_labels(
        model,
        model_id=model_id,
        error_prefix=error_prefix,
    )

    inputs = processor(images=image, return_tensors='pt')
    inputs = move_inputs_to_device(
        dict(inputs),
        device=device,
        dtype=model_dtype,
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        processed = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
        )[0]
        if hasattr(processed, 'segmentation'):
            processed = processed.segmentation

    return processed.detach().cpu().numpy().astype(np.int64), model_dtype
