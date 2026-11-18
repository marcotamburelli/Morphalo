"""Image crops, InstantHMR inference, and the MHR TorchScript mesh."""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open

from .model import IMAGE_SIZE, InstantHMR
from .parameters import (
    TABLE_NAMES,
    BodyEstimate,
    BodyProportions,
    decode_body,
    pack_body,
)

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def prepare_crop(rgb: np.ndarray, bbox: tuple[int, int, int, int]):
    """
    Prepare the person crop and its full-frame camera conditioning.

    Center a square crop on the detection, with a side 1.2 times the longer
    box edge. Areas outside the image are padded with black before RGB
    normalization.

    Parameters
    ----------
    rgb : numpy.ndarray
        Full-frame unsigned 8-bit RGB image with shape ``(H, W, 3)``.
    bbox : tuple[int, int, int, int]
        Person bounding box in full-frame ``(x1, y1, x2, y2)`` pixels.

    Returns
    -------
    crop : numpy.ndarray
        ImageNet-normalized float32 crop of shape ``(3, 288, 288)``.
    cliff : numpy.ndarray
        Float32 vector of shape ``(3,)`` containing the two angular offsets
        of the box center and the ratio of box size to focal length.
    focal : float
        Focal length in full-frame pixels, estimated from the image diagonal.
    """
    height, width = rgb.shape[:2]
    focal = 1.05 * math.hypot(height, width)

    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    box_size = max(x2 - x1, y2 - y1)
    cliff = np.array(
        (
            math.atan((center_x - width / 2) / focal),
            math.atan((center_y - height / 2) / focal),
            box_size / focal,
        ),
        dtype=np.float32,
    )

    half = 0.6 * box_size
    left, top = math.floor(center_x - half), math.floor(center_y - half)
    right, bottom = math.ceil(center_x + half), math.ceil(center_y + half)
    sx, sy = (right - left) / IMAGE_SIZE, (bottom - top) / IMAGE_SIZE
    warp = np.array((
        (sx, 0, left + (sx - 1) / 2),
        (0, sy, top + (sy - 1) / 2),
    ))
    crop = cv2.warpAffine(
        rgb,
        warp,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    normalized = (crop.astype(np.float32) / 255 - MEAN) / STD

    return np.ascontiguousarray(normalized.transpose(2, 0, 1)), cliff, focal


class InstantHMRRuntime:
    """
    Estimate body parameters and generate the compatible MHR LOD 1 mesh.

    Parameters
    ----------
    checkpoint : pathlib.Path
        InstantHMR Medium-commercial safetensors containing network weights,
        decoder tables, and triangle indices.
    rig_path : pathlib.Path
        MHR v1.0.1 LOD 1 TorchScript file.
    device : str
        Torch device used for network inference. Parameter packing and mesh
        decoding run on CPU.

    Attributes
    ----------
    faces : numpy.ndarray
        Zero-based triangle indices of shape ``(36874, 3)``, verified against
        the network checkpoint's vertex ordering.

    Raises
    ------
    ValueError
        If the rig has an unexpected mesh size or triangle connectivity.

    Notes
    -----
    The rig uses neutral facial expression and disables pose correctives to
    match the mesh used by InstantHMR.
    """

    def __init__(self, checkpoint: Path, rig_path: Path, device: str):
        self.device = torch.device(device)
        network = InstantHMR()

        with safe_open(checkpoint, framework='pt') as weights:
            state = {
                key: weights.get_tensor(key)
                for key in network.state_dict()
            }
            self.tables = {
                name: weights.get_tensor(f'head_pose.{name}')
                for name in TABLE_NAMES
            }
            expected_faces = weights.get_tensor('head_pose.faces').long()

        network.load_state_dict(state)
        self.network = network.to(self.device).eval()

        # The release contains the large pose-corrective model. Keep the rig
        # on the CPU and disable its correctives to match InstantHMR training.
        self.rig = torch.jit.load(str(rig_path), map_location='cpu').eval()
        mesh = self.rig.character_torch.mesh
        faces = mesh.faces.long()

        if mesh.rest_vertices.shape != (18439, 3) or faces.shape != (36874, 3):
            raise ValueError('Expected the MHR v1.0.1 LOD 1 TorchScript mesh')

        if not torch.equal(faces, expected_faces):
            raise ValueError(
                'MHR TorchScript and InstantHMR mesh topology differ'
            )

        self.faces = faces.numpy()

    @torch.inference_mode()
    def estimate(
        self,
        rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> BodyEstimate:
        """
        Regress body proportions, joint pose, and camera for one detection.

        Parameters
        ----------
        rgb : numpy.ndarray
            Full-frame unsigned 8-bit RGB image with shape ``(H, W, 3)``.
        bbox : tuple[int, int, int, int]
            Person box in full-frame ``(x1, y1, x2, y2)`` pixels.

        Returns
        -------
        BodyEstimate
            Single-image batched CPU tensors and a focal length in pixels.

        Raises
        ------
        ValueError
            If the network predicts non-finite parameters.
        """
        crop, cliff, focal = prepare_crop(rgb, bbox)
        coefficients, translation = self.network(
            torch.from_numpy(crop).unsqueeze(0).to(self.device),
            torch.from_numpy(cliff).unsqueeze(0).to(self.device),
        )
        prediction = torch.cat((coefficients, translation), dim=-1).cpu()

        if not torch.isfinite(prediction).all():
            raise ValueError('InstantHMR returned non-finite parameters')

        coefficients, translation = prediction.split((447, 3), dim=-1)
        pose, proportions = decode_body(coefficients, self.tables)

        return BodyEstimate(pose, proportions, translation, focal)

    @torch.inference_mode()
    def mesh(
        self,
        target: BodyEstimate,
        proportions: BodyProportions,
    ) -> np.ndarray:
        """
        Decode a body with the requested proportions in the target camera.

        Parameters
        ----------
        target : BodyEstimate
            Estimate supplying joint pose and camera translation.
        proportions : BodyProportions
            Single-image CPU coefficients from the target or source views.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(18439, 3)`` with vertices in camera-space metres.
            X points right, Y points down, and positive Z points away from
            the camera.

        Raises
        ------
        ValueError
            If the rig produces non-finite vertices.
        """
        params = pack_body(target.pose, proportions, self.tables)
        expression = params.new_zeros((params.shape[0], 72))
        vertices, _ = self.rig(
            proportions.shape,
            params,
            expression,
            False,
        )

        # MHR gives centimetres, Y up and Z towards the viewer. InstantHMR's
        # perspective camera expects metres, Y down and positive Z away.
        camera_axes = vertices.new_tensor((0.01, -0.01, -0.01))
        vertices = (
            vertices * camera_axes + target.camera_translation[:, None, :]
        )

        if not torch.isfinite(vertices).all():
            raise ValueError('MHR returned non-finite vertices')

        return vertices[0].numpy()
