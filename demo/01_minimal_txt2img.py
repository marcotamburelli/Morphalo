'''
Minimal txt2img DAG example.

This example demonstrates the smallest possible Stability workflow.

Concepts introduced
-------------------

1. DAG declaration
2. Prompt node
3. Txt2Img node
4. Lateral wiring (Prompt → generator)

Pipeline structure
------------------

Prompt  - - - ->  Txt2Img
                     |
                     v
                  Output image

How to run
----------
    ./bin/run_dag.sh demo.01_minimal_txt2img --dag minimal_txt2img
'''

from pathlib import Path

from stability.dag import DAG
from stability.nodes import Prompt, Txt2Img

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# DAG definition
# -----------------------------------------------------------------------------
#
# Workflows are declared using the DAG context manager.
# Nodes must be instantiated *inside* the DAG context in order to be registered.
#
with DAG(
    name='minimal_txt2img',
    out_dir=ROOT / 'outputs' / 'minimal_txt2img',
):

    # -------------------------------------------------------------------------
    # Prompt node
    # -------------------------------------------------------------------------
    #
    # Prompt nodes prepare textual prompts for downstream generation nodes.
    #
    # Supported prompt formats:
    #
    #   str
    #   list[str]
    #   dict(content=..., style=...)
    #
    # If the `lang` field is specified and differs from `eng_Latn`,
    # the prompt is automatically translated to English before being
    # passed to downstream generation nodes.
    #
    # Language codes follow the FLORES-200 convention:
    #
    #   https://github.com/facebookresearch/flores/blob/main/flores200/README.md
    #
    prompt = Prompt(
        name='prompt',
        spec={
            # Structured prompt example
            #
            # Here we intentionally use Italian to demonstrate automatic translation.
            #
            'prompt': {
                'content': [
                    "Un mago dai lunghi capelli d'argento",
                    'ritratto cinematografico realistico'
                ],
                'style': [
                    'illuminazione cinematografica',
                    'profondità di campo ridotta'
                ]
            },

            'negative_prompt': [
                'sfocato',
                'bassa qualità',
                'anatomia errata'
            ],

            # Source language of the prompt
            #
            # The prompt will be translated to English (eng_Latn)
            # before being used by the diffusion pipeline.
            #
            'lang': 'ita_Latn',
        },
    )

    # -------------------------------------------------------------------------
    # Image generation node
    # -------------------------------------------------------------------------
    #
    # Txt2Img performs image generation using Stable Diffusion XL.
    #
    # The node resolves its configuration through the shared image generation
    # resolver used by all generator nodes (txt2img, img2img, inpaint, etc.).
    #
    # The most important configuration sections are:
    #
    #   model   → model loading and runtime configuration
    #   vae     → optional VAE override
    #   params  → generation parameters
    #   seed    → RNG configuration
    #
    txt2img = Txt2Img(
        name='generate',
        spec={
            # -----------------------------------------------------------------
            # Model configuration
            # -----------------------------------------------------------------
            #
            # Exactly one of the following must be provided:
            #
            #   model.id   → pretrained model identifier
            #   model.path → local model checkpoint/directory
            #
            # model.id uses Diffusers `from_pretrained` and downloads weights
            # automatically on first use.
            #
            'model': {
                'id': 'stabilityai/stable-diffusion-xl-base-1.0',

                # Optional runtime settings
                #
                # 'device': 'cuda',
                # 'dtype': 'bf16',
            },

            # -----------------------------------------------------------------
            # Optional VAE override
            # -----------------------------------------------------------------
            #
            # Some SDXL workflows benefit from using a custom VAE.
            #
            # This can be specified either as:
            #
            #   'vae': 'vae-model-id'
            #
            # or
            #
            #   'vae': {'id': 'vae-model-id'}
            #
            # Example:
            #
            # 'vae': 'madebyollin/sdxl-vae-fp16-fix'
            #
            # If omitted, the default VAE bundled with the base model is used.
            #
            # 'vae': 'madebyollin/sdxl-vae-fp16-fix',

            # -----------------------------------------------------------------
            # Generation parameters
            # -----------------------------------------------------------------
            #
            # These parameters control the diffusion process.
            #
            'params': {
                # Number of denoising steps
                #
                # Higher → better quality but slower
                #
                'steps': 25,

                # Classifier-free guidance
                #
                # `cfg` is equivalent to `guidance_scale`.
                #
                # Typical SDXL range:
                #
                #   4.0 – 7.0
                #
                'cfg': 5.0,

                # Image resolution
                #
                # For SDXL the base resolution is typically 1024.
                #
                'width': 1024,
                'height': 1024,

                # Strength is used by img2img/inpaint.
                # It is ignored by txt2img but accepted by the common resolver.
                #
                # 'strength': 0.7,
            },

            # -----------------------------------------------------------------
            # RNG / seed
            # -----------------------------------------------------------------
            #
            # The seed controls reproducibility.
            #
            # Fixed seed → deterministic image
            # 'random'   → new seed every run
            #
            'seed': 1234,
        }
    )

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------
    #
    # Lateral wiring:
    # The prompt node does not sit in the main image flow.
    # Instead it injects text prompts into the generator node.
    #
    prompt >> txt2img.prompt()