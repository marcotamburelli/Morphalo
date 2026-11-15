# 3DDFA-V3 inference subset

Adapted from [wang-zidu/3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3),
commit **e15385837dc1e051a6cf376b3827f2e279537b29** (2024-11-10).

Paper: Wang et al., *3D Face Reconstruction with the Geometric Guidance of
Facial Part Segmentation*, CVPR 2024, pp. 1672–1682.

## Copied components

- `networks.py`: inference subset of upstream `model/networks.py`: the ResNet-50
  backbone and seven regression heads. Training, recognition, other backbones,
  and MobileNet helpers were omitted. Parameter names and shapes are retained
  for strict loading of the official checkpoint.
- `model.py`: morphable-model decoding, normals, rotations, perspective camera,
  spherical-harmonic color, and studio-gray shading from `model/recon.py`.
  The constructor accepts external model data rather than hardcoded paths.
  Network setup, renderer setup, UV extraction, segmentation, and the demo's
  combined `forward` method were omitted. Morphalo calls the individual
  operations under Torch inference mode. The upstream coefficient docstring
  says 256; the actual network has **257** coefficients, including XYZ translation.
- `preprocess.py`: `POS`, five-point extraction, resize/crop, and alignment
  from `util/preprocess.py`. Assets are supplied by the caller; least-squares
  explicitly uses `rcond=None`, modern Pillow resampling is selected, and
  one-element coordinate arrays use `.item()` for NumPy 2.x compatibility.

The upstream source already credits Deep3DFaceRecon and torchvision for parts
of these implementations. `LICENSE` preserves the 3DDFA-V3 MIT notice;
`LICENSE.Deep3DFaceRecon` and `LICENSE.torchvision` retain those projects' notices.
Upstream comments and formatting are mostly preserved in the copied routines.

## Morphalo integration

`runtime.py` is Morphalo's own inference assembly, not a verbatim upstream file.
`FaceReconstructionRuntime(assets, device)` loads the local checkpoint, model
bases, and alignment reference, then exposes coefficient inference and decoding.
It also defines the required asset filenames and coefficient layout. It has no
Morphalo cache or Hugging Face dependency; the caller supplies resolved paths.

Morphalo uses MediaPipe only to obtain the five image landmarks. Eye centers
are averages of the inner/outer corners, followed by nose and mouth corners.
Y is converted from image-down to the upstream alignment's image-up convention.
The original 224x224 crop construction and RGB `[0, 1]` network input remain.
This detector substitution is an experimental compatibility assumption; it
does not guarantee equivalence to RetinaFace plus the upstream 106-point model.

`morphalo.cache.models` resolves assets and caches the constructed runtime.
Rendering is implemented separately in Morphalo with a CPU Z-buffer and
perspective-correct attribute interpolation in
`morphalo.nodes.preprocess.utils.face_reconstruction_geometry`. The
`FaceReconstruction` node orchestrates this workflow. No TensorFlow/MTCNN, RetinaFace,
nvdiffrast, or compiled upstream renderer is required.

Identity coefficients always come from the source mean. Expression and albedo
may also come from that mean via `params.preserve` (default `[]`; accepted
values: `expression`, `albedo`). Identity is implicit and is not listed as an
option. Rotation, translation, and illumination always
come from target. Neutral debug renders use the original studio lights and
are distinct from the target-lit color render used for composition.

With target albedo, `params.color_transfer` can optionally match rendered Lab
color statistics to the original target before feathering (range [0, 1], default
0). Sampling uses the common mask interior, excludes occlusion and a two-pixel
boundary margin, and reads the unwarped target. The resulting partial correction
applies to all visible reconstructed-face pixels. Empty sample regions leave
the render unchanged. Positive transfer with `preserve: ['albedo']` is rejected.
Raw debug renders and decoded coefficients remain unmodified model outputs.

The node always returns an RGB composite. Conditioning maps belong to subsequent
nodes. Diagnostics are disabled by default; `debug.save_debug: true` enables
source/target/mixed renders, overlays, coefficients, and compressed geometry
archives. Without diagnostics, sources are only regressed into coefficient
vectors, the target is rendered only for visibility, and the transferred mesh
is rendered in color without studio shading. Timing and CUDA memory statistics
use the same helpers and metadata fields as img2img; processing time excludes
model loading and includes every optional diagnostic file write. Diagnostic
records (source/target/mixed renders, coefficient groups, geometry paths, and
warp statistics) are grouped under the JSON card's optional `debug` key.
Operational source records and composition radii remain present in both modes.

An optional `foreground_mask >> reconstruction.occlusion()` attachment protects
foreground objects in target coordinates. White pixels are protected (threshold
128); differing mask dimensions use nearest-neighbor resizing. Full geometric
masks and erosion are computed before occlusion filtering. Only rays whose
sampled eroded-destination point is occluded use identity deformation, with a
local angular transition in neighboring valid directions. Occlusion at the
original endpoint does not prevent contraction toward a free new contour. The
render is blended outside occlusion and protected original pixels are restored
after blending. This does not change
inference or prevent occluder colors from being sampled elsewhere by the warp.

## External assets

Assets are downloaded separately from the authors' dataset:
[Zidu-Wang/3DDFA-V3/assets](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/tree/main/assets).

Required files:

- `net_recon.pth`: official ResNet-50 regressor.
- `face_model.npy`: morphable face bases, topology, and attributes.
- `similarity_Lm3D_all.mat`: canonical alignment reference.

Download into the standard Hugging Face cache:

```bash
venv/bin/python bin/fetch_face_reconstruction.py
```

Both the helper and the node use the same Hugging Face defaults as the other
Morphalo weights, normally `~/.cache/huggingface/hub`. No separate asset/cache
directory is configured on the node. `model.revision` can pin the dataset
revision. The node defines `Zidu-Wang/3DDFA-V3` as the default `model.id`;
a compatible alternative dataset can be selected through `model.id` or the
helper's `--id` option. It must provide the same filenames under `assets/` and
compatible model bases and checkpoint architecture. The repository and revision
are passed explicitly to the cache helpers. Resolved snapshot paths are recorded
in node metadata.

**No weights or model data are included here.** The software licenses above do
not relicense the external assets. The authors describe `face_model.npy` as
derived from BFM, Exp_Pca, and Deep3D; their applicable asset terms remain separate.

## Validation

Unit tests cover mandatory parameter origins, mean fusion, alignment coordinate
round trips, triangle visibility, and output diagnostics. Real inference uses
the external assets; performance and identity fidelity must be assessed from
the source/target/mixed debug renders. The model describes facial shape within
its learned basis and does not reconstruct arbitrary skin microdetail.
