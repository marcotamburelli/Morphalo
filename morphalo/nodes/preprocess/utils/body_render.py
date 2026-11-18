"""Rasterize a camera-space body mesh into an RGBA conditioning image."""

import numpy as np
from PIL import Image


def _vertex_shading(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute a shading multiplier for each vertex in camera space.

    The cross product of each triangle's edges gives a normal whose magnitude
    is proportional to its area. Sum the incident face normals at each vertex
    and normalize the result to obtain a smooth vertex normal. The diffuse
    contribution is the clamped dot product with a light above and to the
    left of the camera. A vertex with a zero accumulated normal receives
    only the ambient contribution.

    Parameters
    ----------
    vertices : numpy.ndarray
        Floating-point mesh vertices of shape ``(V, 3)`` in camera space.
        X points right, Y points down, and Z increases away from the camera.
    faces : numpy.ndarray
        Zero-based triangle indices of shape ``(F, 3)``, with consistent
        outward-facing winding.

    Returns
    -------
    numpy.ndarray
        Shading multipliers of shape ``(V,)`` in the range [0.36, 1.0].
        The renderer interpolates these values across each visible triangle
        before multiplying the base RGB color.
    """
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.zeros_like(vertices)

    for corner in faces.T:
        np.add.at(normals, corner, face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-12)

    # A broad key light above and to the left of the camera plus ambient light.
    light = np.array((-0.35, -0.45, -0.82), dtype=np.float32)
    diffuse = np.clip(normals @ light, 0, 1)

    return 0.36 + 0.64 * diffuse


def render_body(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int,
    height: int,
    focal: float,
    body_color: tuple[int, int, int],
    background_color: tuple[int, int, int] | None,
) -> Image.Image:
    """
    Project and shade a mesh with a perspective camera.

    The camera's principal point is at ``(width / 2, height / 2)`` in pixels.
    An inverse-depth buffer selects the nearest surface at each pixel.
    The renderer skips triangles with any vertex at a depth below 0.1 mm.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices of shape ``(V, 3)`` in camera-space metres, with X
        rightward, Y downward, and Z increasing away from the camera.
    faces : numpy.ndarray
        Zero-based triangle indices of shape ``(F, 3)``.
    width, height : int
        Output dimensions in pixels.
    focal : float
        Target focal length in pixels.
    body_color : tuple[int, int, int]
        Base RGB color before applying shading. Components are in [0, 255].
    background_color : tuple[int, int, int] or None
        Opaque RGB fill, or ``None`` for a transparent background.

    Returns
    -------
    PIL.Image.Image
        RGBA image with alpha 255 at visible mesh pixels.

    Raises
    ------
    ValueError
        If the projected mesh covers no pixel in the output frame.
    """
    near = 1e-4
    faces_in_front = faces[np.all(vertices[faces, 2] >= near, axis=1)]

    inverse_z = 1.0 / np.maximum(vertices[:, 2], near)
    projected = vertices[:, :2] * (focal * inverse_z[:, None])
    projected += np.array((width / 2, height / 2), dtype=projected.dtype)
    shading = _vertex_shading(vertices, faces)

    # One grid for all triangles. Each triangle only evaluates its bounding box.
    pixel_x = np.arange(width, dtype=np.float32)[None, :]
    pixel_y = np.arange(height, dtype=np.float32)[:, None]
    z_buffer = np.zeros((height, width), dtype=np.float32)
    illumination = np.zeros_like(z_buffer)

    for indices in faces_in_front:
        xy = projected[indices]
        x0 = max(0, int(np.ceil(xy[:, 0].min())))
        x1 = min(width, int(np.floor(xy[:, 0].max())) + 1)
        y0 = max(0, int(np.ceil(xy[:, 1].min())))
        y1 = min(height, int(np.floor(xy[:, 1].max())) + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        edge_b, edge_c = xy[1:] - xy[0]
        area = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]

        if abs(area) < 1e-8:
            continue

        x = pixel_x[:, x0:x1] - xy[0, 0]
        y = pixel_y[y0:y1] - xy[0, 1]
        b = (x * edge_c[1] - y * edge_c[0]) / area
        c = (edge_b[0] * y - edge_b[1] * x) / area
        a = 1 - b - c

        za, zb, zc = inverse_z[indices]
        depth = a * za + b * zb + c * zc
        region = np.s_[y0:y1, x0:x1]
        nearer = (
            (a >= -1e-6)
            & (b >= -1e-6)
            & (c >= -1e-6)
            & (depth > z_buffer[region])
        )

        if not nearer.any():
            continue

        z_buffer[region][nearer] = depth[nearer]

        # Perspective-correct interpolation across the winning triangle.
        shade_a, shade_b, shade_c = shading[indices]
        lit = (
            a[nearer] * za * shade_a
            + b[nearer] * zb * shade_b
            + c[nearer] * zc * shade_c
        )
        illumination[region][nearer] = lit / depth[nearer]

    visible = z_buffer > 0

    if not visible.any():
        raise ValueError(
            'MHR mesh does not cover any pixels in the target frame'
        )

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    if background_color is not None:
        rgba[:] = (*background_color, 255)

    rgba[visible, :3] = np.rint(
        illumination[visible, None] * np.asarray(body_color),
    ).clip(0, 255).astype(np.uint8)
    rgba[visible, 3] = 255

    return Image.fromarray(rgba)
