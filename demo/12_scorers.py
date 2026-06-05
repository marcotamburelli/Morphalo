"""
Scorer demos: standalone and chained ranking of candidate images.

This module illustrates the intended usage of the scorer nodes currently
available in the project:

- ``PersonScorer``
- ``FaceScorer``
- ``PromptScorer``

It defines four DAGs:

1. ``person_score``
    Rank candidate images using body pose, hands, and feet plausibility.

2. ``face_score``
    Rank candidate images using face plausibility heuristics, optionally
    enriched by identity similarity against a reference face embedding.

3. ``prompt_score``
    Rank candidate images by alignment with a textual prompt using a
    diffusion-based consistency signal.

4. ``score_chain``
    Combine all scorers in sequence so that each node contributes one score
    component to the same final ranking.

Why this demo?
--------------
Generative workflows often produce several candidate images for the same scene
or subject. In practice, the "best" image is often not the one that looks best
under a single criterion.

For example:

- a candidate may have good overall body structure but a weak face
- another may have a good face but poor prompt alignment
- another may match the prompt well but contain malformed hands

These scorer nodes make image selection explicit, inspectable, and composable.

Scoring model
-------------
Each scorer evaluates one or more candidate images and returns:

- the best-ranked image
- a final score
- detailed per-candidate results
- a ``rankings`` structure

When scorers are chained, downstream scorers reuse the upstream candidate set
and append their own score contribution to the existing ranking instead of
requiring the images to be wired again.

Conceptual overview
-------------------
Standalone scorer:

Candidate 1 ----\
Candidate 2 -----+--> Scorer --> best image + rankings
Candidate 3 ----/

Chained scorers:

Candidate images --> PersonScorer --> FaceScorer --> PromptScorer
                                        ^                ^
                                        |                |
                                   identity ref        prompt

How to run
----------
Run one DAG at a time, for example:

    ./bin/run_dag.sh demo.12_scorers --dag person_score
    ./bin/run_dag.sh demo.12_scorers --dag face_score
    ./bin/run_dag.sh demo.12_scorers --dag prompt_score
    ./bin/run_dag.sh demo.12_scorers --dag score_chain

Notes
-----
- ``PersonScorer`` uses repeated ``image()`` calls to declare candidate slots.
- ``FaceScorer`` also supports repeated ``image()`` wiring and can optionally
  accept identity embeddings via ``identity()``.
- ``PromptScorer`` requires a prompt and accepts it through ``prompt()``.
- In the chained DAG, only the first scorer needs explicit candidate image
  wiring; downstream scorers reuse the accumulated ranking state.
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FaceIdEmbedImage, FileImage, Prompt
from morphalo.nodes.evaluate import FaceScorer, PersonScorer, PromptScorer

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User inputs (local assets, typically gitignored)
# -----------------------------------------------------------------------------
#
# Candidate images to be ranked.
# In a real workflow these would often come from upstream generation nodes,
# but for a compact demo we load them directly from disk.
#
IMG_1 = '~/images/img_1.png'
IMG_2 = '~/images/img_2.jpg'
IMG_3 = '~/images/img_3.jpg'

# Optional identity reference used by FaceScorer.
# This image should contain a clear face for robust embedding extraction.
#
FACE_REF = '~/images/identity_face.png'


# -----------------------------------------------------------------------------
# Shared scorer assets
# -----------------------------------------------------------------------------
#
# PersonScorer requires pose and hand landmark models.
#
PERSON_MODEL_SPEC = {
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
    'hand_landmarker_task': '~/models/mediapipe/hand_landmarker.task',
}

# FaceScorer requires face and pose landmark models.
# Identity similarity is optional and is enabled in this demo via FaceIdEmbedImage.
#
FACE_MODEL_SPEC = {
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

# PromptScorer uses an SDXL model.
#
PROMPT_MODEL_SPEC = {
    'id': 'stabilityai/stable-diffusion-xl-base-1.0',

    # Optional runtime settings.
    #
    # 'device': 'cuda',
    # 'dtype': 'bf16',
}

# Shared prompt used by PromptScorer standalone and in the scorer chain.
#
PROMPT_SPEC = {
    'prompt': {
        'content': [
            'Girl sitting on the floor.',
            'Legs stretched out.',
        ],
        'style': [
            'cinematic lighting, 35mm film, shallow depth of field.',
            'photorealistic.',
        ],
    },
}


# -----------------------------------------------------------------------------
# DAG 1: PersonScorer standalone
# -----------------------------------------------------------------------------
#
# Demonstrates dynamic candidate-image wiring via repeated `image()` calls.
# PersonScorer ranks images using pose, hand, and foot plausibility.
#
with DAG(
    name='person_score',
    out_dir=ROOT / 'outputs' / 'person_score',
):
    img_1 = FileImage(name='img_1', path=IMG_1)
    img_2 = FileImage(name='img_2', path=IMG_2)
    img_3 = FileImage(name='img_3', path=IMG_3)

    person_scorer = PersonScorer(
        name='person_scorer',
        spec={
            'model': PERSON_MODEL_SPEC,
            # 'params': {
            #     'w_pose': 0.45,
            #     'w_hands': 0.40,
            #     'w_feet': 0.15,
            #     'score_weight': 1.0,
            # },
        },
    )

    # Each call to `image()` declares one candidate slot:
    # - first call  -> image:0
    # - second call -> image:1
    # - third call  -> image:2
    #
    img_1 >> person_scorer.image()
    img_2 >> person_scorer.image()
    img_3 >> person_scorer.image()


# -----------------------------------------------------------------------------
# DAG 2: FaceScorer standalone
# -----------------------------------------------------------------------------
#
# Demonstrates face-centered scoring over the candidate set.
# FaceScorer can combine:
#
# - face landmark plausibility
# - eye plausibility
# - face size / framing heuristics
# - optional identity similarity against a reference embedding
#
with DAG(
    name='face_score',
    out_dir=ROOT / 'outputs' / 'face_score',
):
    img_1 = FileImage(name='img_1', path=IMG_1)
    img_2 = FileImage(name='img_2', path=IMG_2)
    img_3 = FileImage(name='img_3', path=IMG_3)

    # Face reference image used to extract identity embeddings.
    #
    # The embedding is computed using an InsightFace backend (default model:
    # 'buffalo_l'), enabling identity-aware scoring in addition to purely
    # geometric or landmark-based facial plausibility.
    #
    face_emb = FaceIdEmbedImage(
        name='face_emb',
        path=FACE_REF,
    )

    face_scorer = FaceScorer(
        name='face_scorer',
        spec={
            'model': FACE_MODEL_SPEC,
            # 'params': {
            #     'face_region_expansion': 1.6,
            #     'w_landmarks': 1.0,
            #     'w_eyes': 0.75,
            #     'w_size': 0.25,
            #     'w_identity': 1.0,
            #     'score_weight': 1.0,
            # },
        },
    )

    # Candidate images to evaluate.
    #
    img_1 >> face_scorer.image()
    img_2 >> face_scorer.image()
    img_3 >> face_scorer.image()

    # Optional identity wiring.
    #
    # The reference face is converted to an identity embedding and attached
    # through the dedicated `identity()` sink.
    #
    face_emb >> face_scorer.identity()


# -----------------------------------------------------------------------------
# DAG 3: PromptScorer standalone
# -----------------------------------------------------------------------------
#
# Demonstrates text-based ranking of candidate images.
# PromptScorer measures how strongly the prompt improves diffusion noise
# prediction relative to an unconditional baseline.
#
with DAG(
    name='prompt_score',
    out_dir=ROOT / 'outputs' / 'prompt_score',
):
    img_1 = FileImage(name='img_1', path=IMG_1)
    img_2 = FileImage(name='img_2', path=IMG_2)
    img_3 = FileImage(name='img_3', path=IMG_3)

    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    prompt_scorer = PromptScorer(
        name='prompt_scorer',
        spec={
            'model': PROMPT_MODEL_SPEC,
            # 'params': {
            #     'score_weight': 1.0,
            # },
        },
    )

    # PromptScorer receives textual conditioning through PromptMixin.
    #
    prompt >> prompt_scorer.prompt()

    # Standalone mode: wire all candidate images explicitly.
    #
    img_1 >> prompt_scorer.image()
    img_2 >> prompt_scorer.image()
    img_3 >> prompt_scorer.image()


# -----------------------------------------------------------------------------
# DAG 4: chained scorer pipeline
# -----------------------------------------------------------------------------
#
# This is the most representative workflow:
#
# 1. PersonScorer filters by overall human/body plausibility
# 2. FaceScorer adds a face-centered contribution
# 3. PromptScorer adds text-alignment contribution
#
# The first scorer receives the explicit candidate image set.
# Downstream scorers reuse the accumulated rankings and candidate paths from the
# upstream scorer output, so additional `image()` wiring is not required.
#
with DAG(
    name='score_chain',
    out_dir=ROOT / 'outputs' / 'score_chain',
):
    img_1 = FileImage(name='img_1', path=IMG_1)
    img_2 = FileImage(name='img_2', path=IMG_2)
    img_3 = FileImage(name='img_3', path=IMG_3)

    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    # Face reference image used to extract identity embeddings.
    #
    # The embedding is computed using an InsightFace backend (default model:
    # 'buffalo_l'), enabling identity-aware scoring in addition to purely
    # geometric or landmark-based facial plausibility.
    #
    face_emb = FaceIdEmbedImage(
        name='face_emb',
        path=FACE_REF,
    )

    person_scorer = PersonScorer(
        name='person_scorer',
        spec={
            'model': PERSON_MODEL_SPEC,
            # 'params': {
            #     'w_pose': 0.45,
            #     'w_hands': 0.40,
            #     'w_feet': 0.15,
            #     'score_weight': 1.0,
            # },
        },
    )

    face_scorer = FaceScorer(
        name='face_scorer',
        spec={
            'model': FACE_MODEL_SPEC,
            # 'params': {
            #     'face_region_expansion': 1.6,
            #     'w_landmarks': 1.0,
            #     'w_eyes': 0.75,
            #     'w_size': 0.25,
            #     'w_identity': 1.0,
            #     'score_weight': 1.0,
            # },
        },
    )

    prompt_scorer = PromptScorer(
        name='prompt_scorer',
        spec={
            'model': PROMPT_MODEL_SPEC,
            # 'params': {
            #     'score_weight': 1.0,
            # },
        },
    )

    # Prompt input is only needed by PromptScorer.
    #
    prompt >> prompt_scorer.prompt()

    # Identity input is only needed by FaceScorer.
    #
    face_emb >> face_scorer.identity()

    # Candidate set enters the chain only once, at the first scorer.
    #
    img_1 >> person_scorer.image()
    img_2 >> person_scorer.image()
    img_3 >> person_scorer.image()

    # Chained scorer aggregation:
    # downstream scorers reuse the upstream ranking state and append their own
    # contributions instead of requiring candidate images to be rewired.
    #
    person_scorer >> face_scorer >> prompt_scorer
