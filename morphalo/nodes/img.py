from typing import Tuple

from PIL import Image


def resolve_long_side_size(
    *,
    image: Image.Image,
    long_side: int,
    multiple: int = 8,
) -> Tuple[int, int]:
    """
    Resolve a proportional output size from an input image.

    The longest side is scaled toward ``long_side`` and then aligned to the
    specified ``multiple`` and the shorter side is scaled proportionally
    from the input image aspect ratio.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    long_side : int
        Target size for the longest side.
    multiple : int, default=8
        Round each resolved dimension down to a multiple of this value.

    Returns
    -------
    tuple[int, int]
        Resolved ``(width, height)``.
    """
    if long_side <= 0:
        raise ValueError(f"'long_side' must be > 0, got {long_side}")

    src_w, src_h = image.size

    if src_w <= 0 or src_h <= 0:
        raise ValueError(f'invalid input image size {src_w}x{src_h}')

    if src_w >= src_h:
        width = int(long_side)
        height = int(round(src_h * (width / src_w)))
    else:
        height = int(long_side)
        width = int(round(src_w * (height / src_h)))

    if multiple > 1:
        width = max(multiple, (width // multiple) * multiple)
        height = max(multiple, (height // multiple) * multiple)

    return width, height
