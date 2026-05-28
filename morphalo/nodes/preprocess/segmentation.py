from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


def predict_sam_mask(
    *,
    img_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    processor: Any,
    model: Any,
    device: str,
    point_coords: Optional[list[list[float]]] = None,
    point_labels: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict SAM mask candidates from one bbox and optional positive points.

    This helper targets the Hugging Face SAM / SAM-HQ processor API, where mask
    post-processing requires both ``original_sizes`` and ``reshaped_input_sizes``.
    SAM2-style processors are intentionally not handled here.
    """
    image = Image.fromarray(np.ascontiguousarray(img_rgb))

    x1, y1, x2, y2 = bbox

    kwargs: dict[str, Any] = {
        'images': image,
        'input_boxes': [[[float(x1), float(y1), float(x2), float(y2)]]],
        'return_tensors': 'pt',
    }

    if point_coords is not None and point_labels is not None:
        kwargs['input_points'] = [[point_coords]]
        kwargs['input_labels'] = [[point_labels]]

    inputs = processor(**kwargs)

    model_dtype = next(model.parameters()).dtype

    inputs = {
        key: (
            value.to(device=device, dtype=model_dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value.to(device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    masks_t = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().float().cpu(),
        inputs['original_sizes'].detach().cpu(),
        inputs['reshaped_input_sizes'].detach().cpu(),
    )[0]

    masks_np = masks_t.numpy()

    if masks_np.ndim == 4:
        masks_np = masks_np[0]
    elif masks_np.ndim != 3:
        raise RuntimeError(f'Unexpected SAM mask shape: {masks_np.shape!r}')

    scores_np = (
        outputs.iou_scores
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    return masks_np.astype(bool), scores_np.astype(np.float32)
