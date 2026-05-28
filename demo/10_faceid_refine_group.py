'''
Advanced FaceID demo: using a NodeGroup macro (refine_face_group).

This demo shows how to compose a reusable subgraph (NodeGroup) produced by a
macro function, and how to wire it using named ports.

Pipeline (high level)
---------------------
Prompt  - - - - - - - - - - - - - - - - - - - - - - ┐
                                                    v
Txt2Img (base image) ---------------------------> refine_face_group -> Output
Face reference image -> FaceIdEmbedImage (inside group) - - - - - -^

What refine_face_group does internally (summary)
------------------------------------------------
Defined in: faces.py

Inside the group:
1) Tap 'in_image' provides the base image.
2) SubjectCrop crops the head region from the base image.
3) Img2Img refines the cropped head.
4) FaceIdEmbedImage computes identity embeddings and injects them into Img2Img.
5) ImageStack overlays the refined head back onto the original image using crop
   metadata (anchor/bbox/transform) so placement is consistent.

Why a group?
------------
- You can reuse the same "refine head/face" pattern across many DAGs.
- The group is injected into the parent DAG only when you wire to it.
- Ports make the external wiring explicit and readable.

How to run
----------
    ./bin/run_dag.sh demo.10_faceid_refine_group --dag faceid_refine_group
'''

from pathlib import Path

# Import the macro that builds the NodeGroup.
# Adjust the import path to match your repo layout.
from demo.macro.refine import refine_face_group
from morphalo.dag import DAG
from morphalo.nodes.file_image import FileImage
from morphalo.nodes.prompt import Prompt
from morphalo.nodes.txt2img import Txt2Img

ROOT = Path(__file__).resolve().parents[1]

FACE_REF = '~/images/identity_face.png'


with DAG(
    name='faceid_refine_group',
    out_dir=ROOT / 'outputs' / 'faceid_refine_group',
):

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------
    #
    # Face reference image used by the group to compute identity embeddings.
    #
    face_img = FileImage(
        name='face_img',
        path=FACE_REF,
    )

    # Keep the prompt consistent across demos for comparability.
    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': {
                'content': [
                    'A cinematic picture of a silver-haired mage with violet eyes.',
                    'Natural skin texture, cinematic lighting, sharp focus.',
                ],
                'style': [
                    '35mm film look, shallow depth of field, subtle grain.',
                ],
            },
            'negative_prompt': [
                'blurry',
                'low quality',
                'bad anatomy',
                'extra fingers',
            ],
        },
    )

    # -------------------------------------------------------------------------
    # Stage 1: base generation (no identity constraint yet)
    # -------------------------------------------------------------------------
    #
    # This stage produces a good "base" image. The next stage will refine only
    # the head region and stabilize identity via FaceID.
    #
    base = Txt2Img(
        name='base',
        spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {
                'steps': 30,
                'cfg': 7.0,
                'width': 1024,
                'height': 1024,
            },
            'seed': 1234,
        },
    )

    prompt >> base.prompt()

    # -------------------------------------------------------------------------
    # Stage 2: refine head/face using a NodeGroup macro
    # -------------------------------------------------------------------------
    #
    # The macro returns a NodeGroup with named ports:
    # - 'in_image'      (Tap)             : base image to refine
    # - 'in_face_image' (FaceIdEmbedImage): reference face image
    # - 'in_prompt'     (Tap)             : prompt payload
    #
    # The group output is the internal ImageStack ('out'), so you can wire:
    #     group >> downstream
    #
    refine = refine_face_group(
        'refine_face',
        refine_spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {
                'steps': 40,
                'cfg': 3.0,
                'strength': 0.4,
            },
            'seed': 1234,
        },
        # Optional ImageStack spec (canvas, out_mode, etc.)
        # stack_spec={...},
        face_id_scale=1.5,
        # face_id_clip_strength=0.5,
        layer_feather=200,
        layer_corner_radius=150,
    )

    # Wire parent DAG nodes into the group's ports.
    base >> refine('in_image')
    face_img >> refine('in_face_image')
    prompt >> refine('in_prompt')

    # The group itself behaves like a node for downstream wiring:
    # refine >> something_else
