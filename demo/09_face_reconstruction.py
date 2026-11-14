"""
3DDFA-V3 face reconstruction and depth-guided img2img demo.

This example transfers source facial identity to a target image, extracts depth
from the resulting RGB composite, and regenerates it at low denoise to blend
the reconstructed face with its surroundings.

Conceptual graph
----------------
Target image -----------------------> FaceReconstruction -> RGB composite
Source views -----------------------> FaceReconstruction.source()    |
                                                                     v
RGB composite ---------------------> ImgAuxMap(depth_midas) -> ControlNet
RGB composite -----------------------------------------------------> Img2Img
Prompt ------------------------------------------------------------> Img2Img

How to run
----------
From the repository root:

    ./bin/run_dag.sh demo.09_face_reconstruction --dag face_reconstruction

Inputs (local paths)
-------------------
- Target image: ``~/images/generated_2.png``.
- Source identity views: ``~/images/face_x``.
- MediaPipe task: ``~/models/mediapipe/face_landmarker.task``.
- SDXL checkpoint: the same local Juggernaut model used in demo 08.

Notes
-----
- Img2Img uses the RGB composite as its initial image, with ``strength=0.3``.
  Its ControlNet receives the MiDaS depth map extracted from that composite.
- The prompt describes a young woman walking peacefully in a park.
- Reconstruction diagnostics are enabled with ``debug.save_debug``: separate
  source/target/mixed renders, overlays, coefficients, and geometry archives.
- External reconstruction, MiDaS, and ControlNet weights are downloaded on first
  use if absent from their caches.
"""
from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Img2Img, Prompt
from morphalo.nodes.preprocess import FaceReconstruction, ImgAuxMap


ROOT = Path(__file__).resolve().parents[1]

with DAG(
    name='face_reconstruction',
    out_dir=ROOT / 'outputs' / 'face_reconstruction',
):
    target = FileImage(name='target', path='~/images/generated_2.png')
    sources = FileImage(name='sources', path='~/images/face_x')

    reconstruction = FaceReconstruction(
        name='reconstruction',
        spec={
            'model': {
                'device': 'cuda',
                'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
            },
            'params': {
                # Identity is automatic. Optionally preserve expression/albedo.
                'preserve': [],
                'color_transfer': 1.0,
            },
            'debug': {'save_debug': True},
        },
    )
    target >> reconstruction
    sources >> reconstruction.source()

    depth = ImgAuxMap(
        name='depth',
        spec={
            'processor': 'depth_zoe',
            'device': 'cuda',
            'detect_resolution': 1024,
        },
    )
    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': {
                'content': ['A young woman walking peacefully in a park.'],
            },
        },
    )

    out = Img2Img(
        name='out',
        spec={
            'model': {
                'path': (
                    '~/models/juggernaut/'
                    'juggernautXL_ragnarokBy.safetensors'
                ),
                'dtype': 'bf16',
                'device': 'cuda',
            },
            'params': {
                'steps': 30,
                'cfg': 5.0,
                'strength': 0.3,
                'long_side': 1024,
            },
        },
    )
    prompt >> out.prompt()
    sources >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin',
        # one value per reference image (matches STYLE_REFS length)
        scale=0.7,
        key='style',
    )
    # The composite supplies both the init image and its depth conditioning.
    reconstruction >> depth >> out.controlnet.add(
        'diffusers/controlnet-depth-sdxl-1.0',
        conditioning_scale=1.0,
        key='depth',
    )
    reconstruction >> out
