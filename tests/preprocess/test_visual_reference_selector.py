from PIL import Image

from morphalo.nodes.preprocess.visual_reference_selector import (
    _transform_image,
)


def test_rotation_extends_border_without_black_padding():
    img = Image.new('RGB', (9, 9), color=(10, 20, 30))
    pixels = img.load()

    for y in range(1, 8):
        for x in range(1, 8):
            pixels[x, y] = (220, 180, 140)

    rotated = _transform_image(
        img,
        flip_horizontal=False,
        flip_vertical=False,
        rotation=45,
    )

    corner = rotated.getpixel((0, 0))
    border = (10, 20, 30)
    interior = (220, 180, 140)
    border_distance = sum((a - b) ** 2 for a, b in zip(corner, border))
    interior_distance = sum((a - b) ** 2 for a, b in zip(corner, interior))

    assert corner != (0, 0, 0)
    assert border_distance < interior_distance
