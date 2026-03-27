"""
FaceID (identity) demo.

This example introduces FaceID as a *lateral identity constraint*.

Key idea
--------
Unlike IP-Adapter "style" (which consumes reference images directly),
FaceID consumes *identity embeddings* (saved as a `.pt` tensor), typically
extracted via InsightFace.

In this repo, `FaceIdEmbedImage` performs that extraction step:
- reads one (or many) face images,
- extracts embeddings,
- aggregates them (optional),
- saves a `.pt` tensor to disk,
- returns a payload containing:
  - `embeds`: path to `.pt`
  - `image`: original reference image path(s)
This output can then be wired into `node.face_id.add(...)`.

Conceptual graph
----------------
Prompt  - - - - - - - - - - - - - - - - ┐
                                        v
FaceIdEmbedImage (embeds + ref imgs) -> Txt2Img -> Output

How to run
----------
    ./bin/run_dag.sh demo.04_faceid_identity --dag faceid_identity_txt2img

Notes
-----
- FaceID and IP-Adapter are intended to be mutually exclusive in your runner
  (avoid wiring both into the same generation node unless you explicitly support it).
- FaceID "Plus/PlusV2" weights can optionally use CLIP injection; the runner uses
  the `image` field from the wired upstream output for that side-channel.
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FaceIdEmbedImage, Prompt, Txt2Img

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User input (local path, typically gitignored)
# -----------------------------------------------------------------------------
#
# A face reference image used to extract identity embeddings.
# For best results:
# - front/3-4 view, decent resolution, good lighting
# - minimal occlusions (hands, hair covering face, strong motion blur)
#
FACE_REF = '~/images/identity_face.png'


with DAG(
    'faceid_identity_txt2img',
    out_dir=ROOT / 'outputs' / 'faceid_identity_txt2img',
) as dag:

    # 1) Prompt
    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': {
                'content': [
                    'A cinematic portrait of a silver-haired mage with violet eyes.',
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

    # 2) Extract identity embeddings from a face image (InsightFace under the hood)
    #
    # This node writes a `.pt` tensor to disk and returns its path in `embeds`.
    # The returned payload also includes `image`, which is used by FaceID Plus/PlusV2
    # variants for optional CLIP side-channel injection.
    face_emb = FaceIdEmbedImage(
        name='face_emb',
        path=FACE_REF,
        # spec can be omitted; defaults are fine for a demo.
        # If you want to show knobs later, you can add:
        # spec={'model_name': 'buffalo_l', 'det_size': [640, 640], 'device': 'cuda', 'paired': True, 'agg': 'mean'}
    )

    # 3) Generator (Txt2Img)
    out = Txt2Img(
        name='out',
        spec={
            'model': {
                'id': 'stabilityai/stable-diffusion-xl-base-1.0',
            },
            'params': {
                'steps': 25,
                'cfg': 7.0,
                'width': 1024,
                'height': 1024,
            },
            'seed': 'random',
        },
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    # Prompt lateral wiring
    prompt >> out.prompt()

    # FaceID lateral wiring
    #
    # `face_id.add(...)` declares an identity slot and returns a sink.
    # The sink expects upstream output to contain:
    # - `embeds` (path to `.pt` tensor)
    # - `image` (ref image path(s), required by Plus/PlusV2 for CLIP injection)
    #
    # The Plus/PlusV2 variants combine InsightFace identity embeddings with
    # additional CLIP image features, which generally improves identity
    # preservation and structural consistency compared to the basic FaceID model.
    face_emb >> out.face_id.add(
        model_id='h94/IP-Adapter-FaceID',
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=1.5,
        # clip_strength=0.5,
        key='id',
    )