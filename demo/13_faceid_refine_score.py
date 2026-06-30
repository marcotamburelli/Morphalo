"""
FaceID refine + scoring demo.

This demo shows a multi-stage image selection and refinement workflow built as a
DAG with one internal NodeGroup.

Pipeline overview
-----------------
1. Start from one init image.
2. Generate a batch of candidate variants with Img2Img, using:
   - prompt conditioning
   - style IP-Adapter references
   - optional regional style mask
   - depth conditioning
3. Rank the batch inside a NodeGroup using:
   - PersonScorer
   - PromptScorer
4. Refine the best candidate with a second Img2Img pass using FaceID.
5. Apply FaceScorer only at the end, after FaceID refinement, so face quality
   and identity consistency are judged on the final output rather than on the
   pre-refinement candidates.

Why this structure?
-------------------
The internal group performs early selection based on:

- global human plausibility
- prompt alignment

This keeps the first ranking focused on coarse structural and semantic quality.

Face scoring is intentionally delayed until after FaceID refinement, because the
second Img2Img pass is the stage that most directly affects facial identity and
facial plausibility.

Conceptual graph
----------------
init image -------------------------------------> img_0 (batch)
prompt -----------------------------------------> img_0.prompt()
style refs / style mask / depth ----------------> img_0 adapters

img_0 batch --> score_group(PersonScorer -> PromptScorer) --> img_1 (FaceID refine)
prompt ---------------------------------------------------> img_1.prompt()
face ref -------------------------------------------------> img_1.face_id()

img_1 final ---------------------------------------------> FaceScorer
face ref embedding --------------------------------------> FaceScorer.identity()

How to run
----------
    ./bin/run_dag.sh demo.13_faceid_refine_score --dag faceid_refine_score

Notes
-----
- The NodeGroup exposes only the ports needed for early ranking:
  candidate images.
- Prompt is kept internal to the group because PromptScorer is part of the
  group implementation.
- FaceScorer is placed after the final Img2Img pass so it evaluates the actual
  identity-refined result.
"""

from pathlib import Path

from morphalo.dag import DAG, NodeGroup
from morphalo.nodes import FaceIdEmbedImage, FileImage, Img2Img, Prompt, Tap
from morphalo.nodes.evaluate import FaceScorer, PersonScorer, PromptScorer
from morphalo.nodes.preprocess import ImgAuxMap

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# User inputs (local assets, typically gitignored)
# -----------------------------------------------------------------------------
#
# Init image: the base image to be transformed.
#
INIT_IMG = '~/images/init_img.png'

# Style references used by IP-Adapter.
#
STYLE_REFS = [
    '~/images/picture_8.jpg',
    '~/images/style_ref_02.png',
]

# Additional style reference for a second IP-Adapter slot.
#
STYLE_REF_A = '~/images/style_ref_03.jpg'

# Optional regional mask for style application.
# Convention: white = apply, black = ignore.
#
STYLE_MASK = '~/images/style_mask.png'

# Face reference images used both for FaceID conditioning and final identity-aware
# face scoring.
#
FACE_REF = [
    '~/images/identity_face_0.png',
    '~/images/identity_face_1.png',
]


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
#
FACE_MODEL_SPEC = {
    'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
    'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
}

# PromptScorer uses an SDXL model.
#
PROMPT_SCORER_MODEL_SPEC = {
    'path': '~/models/juggernaut/juggernautXL_ragnarokBy.safetensors',

    # Optional runtime settings.
    #
    # 'device': 'cuda',
    # 'dtype': 'bf16',
}


with DAG(
    name='faceid_refine_score',
    out_dir=ROOT / 'outputs' / 'faceid_refine_score',
):
    # -------------------------------------------------------------------------
    # Internal group: early candidate scoring
    # -------------------------------------------------------------------------
    #
    # This group ranks the first Img2Img batch using:
    # - PersonScorer for body / hand / foot plausibility
    # - PromptScorer for text alignment
    #
    # FaceScorer is intentionally NOT included here, because face identity and
    # fine facial plausibility are more meaningful after the dedicated FaceID
    # refinement pass.
    #
    with NodeGroup('score_group') as g:
        # -------------------
        # Ports
        # -------------------
        #
        # Candidate images produced by the upstream batch generator.
        #
        in_images = Tap(name='in_images')

        # -------------------
        # Internal prompt for prompt-based ranking
        # -------------------
        #
        # This prompt is intentionally simple and semantic. It is used only for
        # ranking the intermediate batch with PromptScorer.
        #
        score_prompt = Prompt(
            name='score_prompt',
            spec={
                'prompt': {
                    'content': [
                        'A female mage sitting on the ground.',
                        'Full body visible.',
                        'Legs extended forward.',
                        'Facing the camera.',
                    ],
                    'style': [
                        'epic fantasy.',
                        'photorealistic.',
                        'cinematic lighting.',
                    ],
                },
            },)

        # -------------------
        # Internal scorers
        # -------------------
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

        prompt_scorer = PromptScorer(
            name='prompt_scorer',
            spec={
                'model': PROMPT_SCORER_MODEL_SPEC,
                # 'params': {
                #     'score_weight': 1.0,
                # },
            },
        )

        # Internal prompt wiring for PromptScorer.
        #
        score_prompt >> prompt_scorer.prompt()

        # Candidate images enter the group once and are ranked in sequence.
        #
        in_images >> person_scorer >> prompt_scorer

        # Expose the entry port so the group can be wired from the outer DAG.
        #
        g.register_ports(in_images)

    # -------------------------------------------------------------------------
    # Outer prompt
    # -------------------------------------------------------------------------
    #
    # Main creative prompt used by both Img2Img passes.
    #
    prompt = Prompt(
        name='prompt',
        spec={
            'lang': 'eng_Latn',
            'prompt': {
                'content': [
                    'A silver-haired female mage with violet eyes sitting on the ground.',
                    'Full body visible.',
                    'Legs extended forward.',
                    'Facing the camera.',
                ],
                'style': [
                    'epic fantasy.',
                    'photorealistic.',
                    'cinematic lighting.',
                    'shallow depth of field.',
                ],
            },
            'negative_prompt': [
                'blurry',
                'extra fingers',
                'low quality',
            ],
        },
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------
    init_img = FileImage(
        name='init_img',
        path=INIT_IMG,
    )

    style_refs = FileImage(
        name='style_refs',
        path=STYLE_REFS,
    )

    style_ref_a = FileImage(
        name='style_ref_a',
        path=STYLE_REF_A,
    )

    style_mask = FileImage(
        name='style_mask',
        path=STYLE_MASK,
    )

    # Face reference image(s) used to extract identity embeddings.
    #
    # The embedding is computed using an InsightFace backend (default model:
    # 'buffalo_l'), enabling both FaceID conditioning and identity-aware
    # face scoring.
    #
    face_emb = FaceIdEmbedImage(
        name='face_emb',
        path=FACE_REF,
    )

    # -------------------------------------------------------------------------
    # Stage 1: batch generation
    # -------------------------------------------------------------------------
    #
    # Img2Img generates a batch of candidate variants from the init image.
    # This stage mixes:
    # - prompt guidance
    # - style references via IP-Adapter
    # - optional masked style localization
    # - depth structural conditioning
    #
    img_0 = Img2Img(
        name='img_0',
        spec={
            'model': {
                'path': '~/models/juggernaut/juggernautXL_ragnarokBy.safetensors',
                'dtype': 'bf16',
                'device': 'cuda',
            },
            'params': {
                'steps': 30,
                'cfg': 3,
                'strength': 0.6,
                'width': 1024,
                'height': 1024,
            },
            'seed': 'random',
            'batch': 5,
        },
    )

    depth = ImgAuxMap(
        name='depth',
        spec={
            'processor': 'depth_midas',
            'detect_resolution': 1024,
        },
    )

    # Main image flow.
    #
    init_img >> img_0

    # Prompt into first Img2Img pass.
    #
    prompt >> img_0.prompt()

    # Depth conditioning from the init image.
    #
    init_img >> depth >> img_0.controlnet.add(
        'diffusers/controlnet-depth-sdxl-1.0',
        conditioning_scale=0.7,
        key='depth',
    )

    # Style IP-Adapter slot 1.
    #
    ip_style = img_0.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin',
        scale=0.5,
        key='style',
    )

    style_refs >> ip_style
    style_mask >> ip_style.mask()

    # Style IP-Adapter slot 2.
    #
    style_ref_a >> img_0.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin',
        scale=0.7,
        key='style_2',
    )

    # -------------------------------------------------------------------------
    # Stage 2: FaceID refinement of the best early-ranked candidate
    # -------------------------------------------------------------------------
    #
    # The score_group receives the full candidate batch from img_0 and selects
    # the best candidate according to person plausibility and prompt alignment.
    # That selected image is then refined by a second Img2Img pass with FaceID.
    #
    img_1 = Img2Img(
        name='img_1',
        spec={
            'model': {
                'path': '~/models/juggernaut/juggernautXL_ragnarokBy.safetensors',
                'dtype': 'bf16',
                'device': 'cuda',
            },
            'params': {
                'steps': 30,
                'cfg': 3,
                'strength': 0.45,
                'width': 1024,
                'height': 1024,
            },
            'seed': 'random',
            'batch': 5,
        },
    )

    # Prompt into second Img2Img pass.
    #
    prompt >> img_1.prompt()

    # FaceID conditioning for identity anchoring.
    #
    face_emb >> img_1.face_id.add(
        model_id='h94/IP-Adapter-FaceID',
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=1.0,
        # clip_strength=1.0,
        key='face_id',
    )

    # Best intermediate candidate -> final FaceID refinement.
    #
    img_0 >> g('in_images') >> img_1

    # -------------------------------------------------------------------------
    # Stage 3: final face scoring
    # -------------------------------------------------------------------------
    #
    # FaceScorer is applied only at the end, after FaceID refinement, so the
    # final selected output is judged on the actual refined face rather than on
    # the pre-refinement batch.
    #
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

    # Identity reference for identity-aware face scoring.
    #
    face_emb >> face_scorer.identity()

    # Final refined image -> final face scoring.
    #
    img_1 >> face_scorer.image()
