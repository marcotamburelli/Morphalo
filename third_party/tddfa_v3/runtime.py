"""Morphalo's inference assembly for the adapted 3DDFA-V3 components.

This module is our implementation of the inference path needed by Morphalo,
not a file copied verbatim from upstream. It assembles the adapted network,
morphable-model decoder, and alignment reference while preserving their input,
camera, and coefficient conventions. See README.md and NOTICE.txt for provenance.

The caller supplies local asset paths and a Torch device. Asset downloads and
application-level model caching remain outside this package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from .model import MorphableFaceModel
from .networks import define_net_recon


# Local asset contract: model means/bases and topology, regressor checkpoint,
# and the canonical 68-point reference used to derive alignment's five points.
# Download locations and application-level caching belong to the caller.
ASSET_FILENAMES = (
    'face_model.npy',
    'net_recon.pth',
    'similarity_Lm3D_all.mat',
)
# Layout shared by the official regressor heads and model.split_alpha:
# 80 identity + 64 expression + 80 albedo + 3 rotation + 27 light + 3 translation.
# Lighting stores nine spherical-harmonic coefficients for each RGB channel.
# These offsets describe the checkpoint format, not tunable node settings.
COEFFICIENT_SLICES = {
    'identity': slice(0, 80),
    'expression': slice(80, 144),
    'albedo': slice(144, 224),
    'rotation': slice(224, 227),
    'illumination': slice(227, 254),
    'translation': slice(254, 257),
}


class FaceReconstructionRuntime:
    """
    Load local 3DDFA-V3 assets and expose coefficient inference and decoding.

    The ResNet-50 regressor maps an already aligned 224-by-224 RGB image to
    257 coefficients. The morphable model reconstructs vertices from identity
    and expression, then applies pose, camera projection, and optional lighting.
    Alignment and rasterization are handled by the caller; this runtime returns
    coefficient vectors and vertex arrays rather than rendered images.

    Source/target parameter selection also belongs to the caller. Decoding a
    mixed vector uses exactly the identity, expression, pose, albedo, lighting,
    and translation supplied in that vector.

    Parameters
    ----------
    assets : dict[str, pathlib.Path]
        Paths keyed by all filenames in ``ASSET_FILENAMES``:

        - ``face_model.npy``: NumPy dictionary containing mean geometry/color,
          identity/expression/albedo bases, triangles, and vertex adjacency.
        - ``net_recon.pth``: Torch checkpoint with the ``net_recon`` state dict.
        - ``similarity_Lm3D_all.mat``: canonical 68-point XYZ reference under ``lm``.

        Files must already exist locally and be compatible with this decoder.
    device : str, optional
        Torch inference device. Default: ``'cpu'``.

    Attributes
    ----------
    network : torch.nn.Module
        ResNet-50 coefficient regressor in evaluation mode.
    model : MorphableFaceModel
        Decoder using the original camera and lighting conventions.
    reference : numpy.ndarray
        Canonical five-point XYZ alignment reference with shape ``(5, 3)``,
        ordered as two eye centers, nose, and two mouth corners.
    triangles : numpy.ndarray
        Zero-based mesh connectivity with shape ``(T, 3)``.
    assets : dict[str, pathlib.Path]
        Local asset paths retained for provenance.
    device : str
        Torch inference device.

    Raises
    ------
    KeyError
        If an asset mapping, model dictionary, reference file, or checkpoint
        lacks a required key.
    OSError
        If a required local file is missing or cannot be read.
    ValueError
        If checked topology indices, basis dimensions, or alignment-reference
        values are incompatible with the expected model layout.
    RuntimeError
        If the checkpoint does not match the network under strict loading, or
        Torch cannot place the network/model on the requested device.

    Notes
    -----
    Geometry and translation use the morphable model's native units, not a
    calibrated metric scale. The original fixed perspective camera is retained.
    All returned NumPy arrays are independent CPU copies with the single-image
    batch dimension removed. Studio-gray shading is for geometric inspection;
    predicted-light colors use the illumination encoded in the supplied vector.
    """

    def __init__(self, assets: dict[str, Path], device: str = 'cpu'):
        self.assets = {name: Path(assets[name]) for name in ASSET_FILENAMES}
        self.device = str(device)

        # Upstream serializes a dictionary inside a scalar object array, hence
        # allow_pickle and item(). The flattened mean contains XYZ per vertex;
        # each linear basis must have 3N rows and its corresponding head width.
        data = np.load(self.assets['face_model.npy'], allow_pickle=True).item()
        vertices = np.asarray(data['u']).size // 3
        triangles = np.asarray(data['tri'], dtype=np.int64)
        if (triangles.ndim != 2 or triangles.shape[1] != 3
                or triangles.min() < 0 or triangles.max() >= vertices):
            raise ValueError('3DDFA-V3 face_model.npy has invalid triangle indices')

        for name, count in (('id', 80), ('exp', 64), ('alb', 80)):
            if np.asarray(data[name]).shape != (vertices * 3, count):
                raise ValueError(f'3DDFA-V3 face_model.npy has an invalid {name} basis')

        # This reference is geometry for crop fitting, not landmarks detected
        # in the current image. Preserve upstream's 68-to-five-point convention.
        reference68 = np.asarray(
            loadmat(self.assets['similarity_Lm3D_all.mat'])['lm'],
        )
        if reference68.shape != (68, 3) or not np.isfinite(reference68).all():
            raise ValueError('Invalid 3DDFA-V3 five-point alignment reference')

        # The upstream landmark numbers are one-based: nose, two corners per
        # eye, and two mouth corners. Average each eye pair, then reorder from
        # nose/eyes/mouth to eyes/nose/mouth to match alignment's image points.
        indices = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        reference = np.stack((
            reference68[indices[0]], reference68[indices[[1, 2]]].mean(axis=0),
            reference68[indices[[3, 4]]].mean(axis=0), reference68[indices[5]],
            reference68[indices[6]],
        ))[[1, 2, 0, 3, 4]]

        # use_last_fc=False selects the seven coefficient heads expected by the
        # official checkpoint, rather than a single classifier. Load the whole
        # state strictly on CPU, then switch to evaluation mode on the device.
        # The morphable bases and camera tensors are placed on that device too.
        network = define_net_recon('resnet50', use_last_fc=False, init_path=None)
        checkpoint = torch.load(
            self.assets['net_recon.pth'],
            map_location='cpu',
            weights_only=True,
        )
        network.load_state_dict(checkpoint['net_recon'], strict=True)
        network = network.eval().to(device)
        self.network = network
        self.model = MorphableFaceModel(data, device=device)
        self.reference = reference
        self.triangles = triangles

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """
        Regress one coefficient vector from an already aligned RGB crop.

        The caller must perform five-point alignment and cropping beforehand.
        Preserve RGB channel order, scale intensities to ``[0, 1]``, and convert
        HWC pixels to a single-image NCHW tensor. This reproduces the upstream
        inference input; no ImageNet mean/std normalization is applied.

        Evaluation-mode inference predicts all groups in ``COEFFICIENT_SLICES``.
        This method does not select source groups, fuse views, or decode a mesh.

        Parameters
        ----------
        rgb : numpy.ndarray
            Aligned image with shape ``(224, 224, 3)`` and dtype ``uint8``.
            X increases rightward and Y downward, as in a standard image.

        Returns
        -------
        numpy.ndarray
            Independent float32 CPU array with shape ``(257,)``, containing
            finite identity, expression, albedo, rotation, light, and translation
            values in the checkpoint's order.

        Raises
        ------
        ValueError
            If image dimensions or dtype do not match the aligned input contract.
        RuntimeError
            If the regressor returns an unexpected coefficient count or any
            nonfinite value.
        """
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError('3DDFA-V3 expects an aligned uint8 RGB 224x224 crop')

        # Own the NumPy storage before creating the tensor, preserving caller
        # pixels. HWC -> CHW -> NCHW adds the batch axis expected by the network.
        tensor = torch.from_numpy(np.array(rgb, copy=True)).to(
            device=self.device, dtype=torch.float32,
        ).permute(2, 0, 1).unsqueeze(0) / 255.0

        # eval() fixes layer behavior; inference_mode additionally disables
        # autograd bookkeeping. The returned vector belongs to the caller.
        with torch.inference_mode():
            coefficients = self.network(tensor)[0].float().cpu().numpy()
        if coefficients.shape != (257,) or not np.isfinite(coefficients).all():
            raise RuntimeError('3DDFA-V3 returned invalid coefficients')
        return coefficients.copy()

    def decode(
        self,
        coefficients: np.ndarray,
        *,
        render_fields: tuple[str, ...] = ('color', 'shape'),
    ) -> dict[str, np.ndarray]:
        """
        Decode a coefficient vector into geometry and optional vertex colors.

        Build identity-plus-expression geometry in the canonical model frame,
        apply rotation and translation, convert to the upstream camera frame,
        and project into the aligned crop. These stages preserve the model's
        coordinate conventions; mapping crop pixels to the full image and
        resolving triangle visibility are the caller's responsibility.

        Appearance is optional. Predicted-light color combines estimated albedo
        with spherical-harmonic illumination from the coefficient vector.
        Neutral shape diagnostics instead use uniform gray reflectance and
        fixed studio lights, so appearance does not obscure geometric relief.
        Geometry-only decoding skips normals, albedo, and lighting calculations.

        Parameters
        ----------
        coefficients : numpy.ndarray
            Finite vector with shape ``(257,)`` in ``COEFFICIENT_SLICES`` order.
            Rotation angles are in radians; translation uses native model units.
            The vector may be inferred from one image or mixed by the caller.
        render_fields : tuple[str, ...], optional
            Requested appearance outputs:

            - ``'color'``: estimated albedo and illuminated RGB vertex colors.
            - ``'shape'``: uniform-gray vertex colors under fixed studio lights.

            Default: both. An empty tuple computes only geometry, sufficient
            for the caller to rasterize a visibility mask.

        Returns
        -------
        dict[str, numpy.ndarray]
            Independent float32 CPU arrays, without a batch dimension. Let N be
            the number of vertices in the external face model:

            ``canonical_vertices`` : numpy.ndarray
                Shape ``(N, 3)``. Identity-plus-expression XYZ vertices before
                pose and translation, in native nonmetric model coordinates.
            ``camera_vertices`` : numpy.ndarray
                Shape ``(N, 3)``. Posed/transformed XYZ in the fixed camera frame.
                Z must be positive; smaller Z is nearer the camera.
            ``screen_vertices`` : numpy.ndarray
                Shape ``(N, 2)``. Projected X/Y in the 224-pixel crop coordinate
                frame, with Y downward. Positions may fall outside the crop.
            ``albedo`` : numpy.ndarray, optional
                Shape ``(N, 3)``. RGB surface color before lighting, returned for
                ``'color'``. Values are normalized by 255 and are not clipped;
                the linear appearance basis can produce values outside [0, 1].
            ``colors`` : numpy.ndarray, optional
                Shape ``(N, 3)``. Albedo illuminated by the vector's estimated
                light, clipped to [0, 1]. Returned for ``'color'``.
            ``gray_colors`` : numpy.ndarray, optional
                Shape ``(N, 3)``. Studio-lit gray vertex colors, clipped to [0, 1].
                Returned for ``'shape'`` and used only for geometric diagnostics.

        Raises
        ------
        ValueError
            If the coefficient vector has the wrong shape or nonfinite values.
        RuntimeError
            If decoded camera coordinates are nonfinite or any vertex has Z <= 0.
            This checks projection depth, not whether projected X/Y lie in-frame.
        """
        values = np.asarray(coefficients, dtype=np.float32)
        if values.shape != (257,) or not np.isfinite(values).all():
            raise ValueError('Expected 257 finite reconstruction coefficients')

        with torch.inference_mode():
            # Add the decoder's batch axis, then separate the checkpoint groups.
            # Shape uses identity and expression; albedo has a separate basis.
            parts = self.model.split_alpha(
                torch.as_tensor(values, device=self.device)[None],
            )
            shape = self.model.compute_shape(parts['id'], parts['exp'])

            # Vertices are row vectors: transform applies shape @ rotation plus
            # translation. to_camera changes Z in place to camera_distance - Z;
            # clone the posed tensor so that conversion has no side effect on it.
            rotation = self.model.compute_rotation(parts['angle'])
            posed = self.model.transform(shape, rotation, parts['trans'])
            camera = self.model.to_camera(posed.clone())
            if not torch.isfinite(camera).all() or torch.any(camera[..., 2] <= 0):
                raise RuntimeError('Reconstructed face is outside the camera frustum')

            # Project with the upstream fixed camera. Flip Y from the model's
            # upward-positive projection to image coordinates: 223 = 224 - 1,
            # the last pixel index in the aligned crop. This is not clipping.
            screen = self.model.to_image(camera)
            screen[..., 1] = 223.0 - screen[..., 1]
            tensors = {
                'canonical_vertices': shape,
                'camera_vertices': camera,
                'screen_vertices': screen,
            }

            if render_fields:
                # Lighting needs normals in the posed orientation. Rotate the
                # canonical normals like the vertices, without translation.
                normals = self.model.compute_norm(shape) @ rotation
                if 'color' in render_fields:
                    # Estimate reflectance separately, then apply the supplied
                    # spherical-harmonic light and bound displayed colors.
                    albedo = self.model.compute_albedo(parts['alb'])
                    tensors['albedo'] = albedo
                    tensors['colors'] = self.model.compute_texture(
                        albedo, normals, parts['sh'],
                    ).clamp(0, 1)
                if 'shape' in render_fields:
                    # 0.78 is a fixed neutral reflectance for inspection, not a
                    # predicted skin color. Studio light directions/intensities
                    # come from the adapted decoder and ignore estimated light.
                    tensors['gray_colors'] = self.model.compute_gray_shading_with_directionlight(
                        torch.full_like(shape, 0.78), normals,
                    ).clamp(0, 1)

        # Strip the single-image batch axis and detach ownership from Torch
        # storage. Rasterization/serialization can consume these arrays on CPU.
        return {
            name: tensor[0].float().cpu().numpy().copy()
            for name, tensor in tensors.items()
        }
