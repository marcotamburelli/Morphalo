from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageColor
from skimage.color import lab2rgb, rgb2lab


@dataclass
class ColorConfig:
    color: tuple[int, int, int]
    strength: float


def read_color_op_config(spec: dict, node_id: str) -> ColorConfig:
    """
    Read shared color and strength parameters from a node spec.

    Parameters
    ----------
    spec : dict
        Resolved node specification. Parameters are read from ``spec['params']``.
    node_id : str
        Node identifier used to make validation errors actionable.

    Returns
    -------
    ColorConfig
        Parsed target color and validated strength.

    Raises
    ------
    ValueError
        If ``color`` or ``strength`` is invalid.
    """
    params = spec.get('params', {})

    color = parse_color(
        params.get('color', params.get('target_color', '#ffffff')),
        node_id=node_id,
    )

    try:
        strength = float(params.get('strength', 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{node_id}': params.strength must be a number in [0, 1]"
        ) from exc
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"'{node_id}': params.strength must be in [0, 1], got {strength}"
        )

    return ColorConfig(color=color, strength=strength)


def _is_rgb_triplet(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(channel, (int, float)) for channel in value)
    )


def parse_color(value: Any, *, node_id: str) -> tuple[int, int, int]:
    """
    Parse one user-provided color into an RGB integer triplet.

    Parameters
    ----------
    value : Any
        Color name, hex string, or 3-item RGB sequence with channel values in
        ``[0, 255]``.
    node_id : str
        Node identifier used to make validation errors actionable.

    Returns
    -------
    tuple[int, int, int]
        Parsed ``(red, green, blue)`` color components.

    Raises
    ------
    ValueError
        If ``value`` is not a supported color format.
    """
    if isinstance(value, str):
        try:
            rgb = ImageColor.getrgb(value)
        except ValueError as exc:
            raise ValueError(
                f"'{node_id}': invalid color {value!r}"
            ) from exc

        if len(rgb) == 4:
            rgb = rgb[:3]
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    if _is_rgb_triplet(value):
        rgb = tuple(int(channel) for channel in value)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(
                f"'{node_id}': RGB values must be in [0, 255]"
            )
        return rgb

    raise ValueError(
        f"'{node_id}': colors entries must be color names, hex "
        'strings, or RGB triplets'
    )


def parse_colors(
    value: Any,
    *,
    node_id: str,
) -> tuple[tuple[int, int, int], ...] | None:
    """
    Parse optional user-provided colors into RGB integer triplets.

    Parameters
    ----------
    value : Any
        ``None``, one color, or a non-empty sequence of colors. Each color may
        be a color name, hex string, or 3-item RGB sequence.
    node_id : str
        Node identifier used to make validation errors actionable.

    Returns
    -------
    tuple[tuple[int, int, int], ...] or None
        Parsed RGB colors, or ``None`` when no colors were provided.

    Raises
    ------
    ValueError
        If ``value`` is not ``None``, one supported color, or a non-empty
        sequence of supported colors.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return (parse_color(value, node_id=node_id),)

    if _is_rgb_triplet(value):
        return (parse_color(value, node_id=node_id),)

    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(
            f"'{node_id}': colors must be a color or a non-empty "
            'list of colors'
        )

    return tuple(parse_color(item, node_id=node_id) for item in value)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """
    Compute Rec.601 luma from an RGB array.

    Parameters
    ----------
    rgb : np.ndarray
        RGB array with the channel dimension last and values in ``[0, 255]``.

    Returns
    -------
    np.ndarray
        Single-channel luminance array with values in ``[0, 1]``.
    """
    # Rec.601 luma weights: green contributes most to perceived brightness,
    # then red, then blue.
    return (
        rgb[..., 0:1] * 0.299
        + rgb[..., 1:2] * 0.587
        + rgb[..., 2:3] * 0.114
    ) / 255.0


def colorize_by_luminance(
    img: Image.Image,
    *,
    color: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    """
    Blend an image with a luminance-scaled target color.

    Parameters
    ----------
    img : Image.Image
        Source image. It is converted to ``RGBA`` before processing.
    color : tuple[int, int, int]
        Target ``(red, green, blue)`` color used for full-brightness pixels.
    strength : float
        Blend amount in ``[0, 1]`` between the original RGB pixels and the
        luminance-colorized RGB pixels.

    Returns
    -------
    Image.Image
        Colorized ``RGBA`` image with the original alpha channel preserved.
    """
    src = img.convert('RGBA')
    arr = np.asarray(src, dtype=np.float32)

    rgb = arr[..., :3]
    alpha = arr[..., 3:4]

    luminance = _luminance(rgb)
    target = np.asarray(color, dtype=np.float32).reshape(1, 1, 3) * luminance
    out_rgb = rgb * (1.0 - strength) + target * strength
    out = np.concatenate([out_rgb, alpha], axis=2)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode='RGBA')


def tint_by_luminance(
    img: Image.Image,
    *,
    color: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    """
    Blend each pixel toward a target color using luminance-scaled strength.

    Parameters
    ----------
    img : Image.Image
        Source image. It is converted to ``RGBA`` before processing.
    color : tuple[int, int, int]
        Target ``(red, green, blue)`` color for the tint.
    strength : float
        Maximum blend amount in ``[0, 1]``. The per-pixel amount is
        ``luminance * strength``.

    Returns
    -------
    Image.Image
        Tinted ``RGBA`` image with the original alpha channel preserved.
    """
    src = img.convert('RGBA')
    arr = np.asarray(src, dtype=np.float32)

    rgb = arr[..., :3]
    alpha = arr[..., 3:4]

    amount = _luminance(rgb) * strength
    target = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out_rgb = rgb * (1.0 - amount) + target * amount
    out = np.concatenate([out_rgb, alpha], axis=2)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode='RGBA')


def match_lab_color_statistics(
    refined: Image.Image,
    target: Image.Image,
    *,
    strength: float,
    eps: float = 1e-6,
) -> Image.Image:
    """
    Match refined image color statistics to a target image in Lab space.

    Parameters
    ----------
    refined : Image.Image
        Refined source image. It is converted to ``RGBA`` before processing.
    target : Image.Image
        Target reference image with the same size as ``refined``. It is
        converted to ``RGBA`` before processing.
    strength : float
        Blend amount in ``[0, 1]`` between original and statistically matched
        Lab colors.
    eps : float, optional
        Small denominator guard used when a refined Lab channel has near-zero
        standard deviation.

    Returns
    -------
    Image.Image
        Corrected ``RGBA`` image with the refined alpha channel preserved.

    Raises
    ------
    ValueError
        If image sizes differ or ``strength`` is outside ``[0, 1]``.
    """
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f'strength must be in [0, 1], got {strength!r}')

    src = refined.convert('RGBA')
    ref = target.convert('RGBA')
    if src.size != ref.size:
        raise ValueError(
            f'refined and target must have the same size, got {src.size} '
            f'and {ref.size}'
        )

    if strength == 0.0:
        return src

    src_arr = np.asarray(src, dtype=np.float32)
    ref_arr = np.asarray(ref, dtype=np.float32)

    src_alpha = src_arr[..., 3:4]
    valid = src_alpha[..., 0] > 0.0
    valid &= ref_arr[..., 3] > 0.0
    if not np.any(valid):
        return src

    src_lab = rgb2lab(src_arr[..., :3] / 255.0).astype(np.float32)
    ref_lab = rgb2lab(ref_arr[..., :3] / 255.0).astype(np.float32)

    src_samples = src_lab[valid]
    ref_samples = ref_lab[valid]
    src_mean = src_samples.mean(axis=0)
    ref_mean = ref_samples.mean(axis=0)
    src_std = src_samples.std(axis=0)
    ref_std = ref_samples.std(axis=0)

    matched_lab = (src_lab - src_mean) * (ref_std / (src_std + eps)) + ref_mean
    out_lab = src_lab.copy()
    out_lab[valid] = (
        src_lab[valid] * (1.0 - strength)
        + matched_lab[valid] * strength
    )

    out_rgb = np.clip(lab2rgb(out_lab), 0.0, 1.0) * 255.0
    out = np.concatenate([out_rgb, src_alpha], axis=2)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)

    return Image.fromarray(out, mode='RGBA')
