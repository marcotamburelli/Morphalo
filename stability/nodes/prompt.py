from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from stability.cache.models import get_translator
from stability.const import ENG
from stability.core.paths import load_json_dict, make_node_output_path
from stability.dag import NodeRef
from stability.nodes.common.config_resolve import SpecInput, resolve_spec
from stability.nodes.common.io import write_json_sidecar
from stability.nodes.wiring.prompt import norm_prompt_pair

DEFAULT_MODEL = 'facebook/nllb-200-distilled-600M'


def _translate(tr: Any, text: str) -> str:
    if not text:
        return text

    out = tr(text)

    return out[0]['translation_text']


@dataclass
class Prompt(NodeRef):
    """
    Prepare and normalize text prompts from a configuration specification.

    This node extracts, normalizes, and optionally translates textual prompts
    from a configuration dictionary (`spec`). It supports both simple and
    structured prompt definitions and produces a canonical prompt bundle
    suitable for downstream image/video generation nodes.

    Supported input formats
    -----------------------
    The following forms are supported for both ``prompt`` and
    ``negative_prompt`` entries in ``spec``:

    1. Simple form:
       - ``str``
       - ``list[str]``

       In this case, the value is treated as the main prompt content.

    2. Structured form:
       - ``dict`` with optional keys:
         - ``content`` : ``str`` or ``list[str]``
         - ``style``   : ``str`` or ``list[str]``

       The ``content`` field is interpreted as the primary prompt, while
       ``style`` is interpreted as a secondary prompt (e.g. stylistic or
       compositional guidance).

    Normalization rules
    -------------------
    - Prompt content and style fields are normalized using ``norm_prompt_pair``.
    - Descriptive prompts (``prompt``) are joined using newline separators.
    - Negative prompts (``negative_prompt``) are joined using comma separators.

    Language handling and translation
    ---------------------------------
    If ``spec["lang"]`` is defined and differs from ``eng_Latn``, all extracted
    prompt fields are translated into English (``eng_Latn``) using a cached
    Hugging Face translation pipeline (NLLB by default).
    See https://github.com/facebookresearch/flores/blob/main/flores200/README.md

    The target language is always English, as downstream diffusion models are
    assumed to operate optimally with English prompts.

    Attributes
    ----------
    spec : dict or str or pathlib.Path or sequence of (dict or str or pathlib.Path)
        Configuration dictionary containing prompt definitions and optional
        language metadata. Relevant keys include:

        The specification may be provided as an in-memory dictionary, as a path
        to a HOCON configuration file, or as a sequence of such elements. When a
        sequence is given, each element is resolved independently and merged from
        left to right, with later elements overriding earlier ones.

        - ``prompt``
        - ``negative_prompt``
        - ``lang`` (source language code, e.g. ``ita_Latn``); for language
          codes see
          https://github.com/facebookresearch/flores/blob/main/flores200/README.md 
        - ``translate_model`` (optional Hugging Face model id)
        - ``device`` (default 'cuda')

    Outputs
    -------
    dict
        A dictionary containing normalized (and possibly translated) prompts.

    Notes
    -----
    - The JSON sidecar produced by this node is used as a persistent cache.
      If the node implementation, its configuration, or specification changes,
      any previously generated sidecar files must be removed manually in order
      to avoid reusing stale cached outputs.
    - Translation is deterministic and cached per (model, source language,
      device) combination.
    - This node does not perform prompt optimization or rewriting beyond
      normalization and translation.
    - The output format is intentionally minimal and stable to allow easy
      consumption by downstream generator nodes.
    """
    spec: SpecInput = field(default_factory=dict)

    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict:
        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='json',
            tag=self.op,
        )
        if out_path.exists():
            return load_json_dict(out_path)

        spec = resolve_spec(self.spec)

        src_lang = spec.get('lang', ENG)
        model_id = spec.get('translate_model', DEFAULT_MODEL)
        device = spec.get('device', 'cuda')

        prompt, prompt_2 = norm_prompt_pair(spec.get('prompt'))
        negative_prompt, negative_prompt_2 = norm_prompt_pair(
            spec.get('negative_prompt'),
            joiner=', '
        )

        translated = False

        if src_lang and src_lang != ENG:
            tr = get_translator(
                model_id=model_id,
                source_lang=src_lang,
                target_lang=ENG,
                device=device
            )
            prompt = _translate(tr, prompt)
            prompt_2 = _translate(tr, prompt_2)
            negative_prompt = _translate(tr, negative_prompt)
            negative_prompt_2 = _translate(tr, negative_prompt_2)
            translated = True

        out: Dict[str, Any] = {
            'ok': True,
            'node': 'prompt',
            'lang': {'src': src_lang, 'tgt': ENG},
            'translated': translated,
            'model': {'translate_model': model_id},
            'prompt': prompt,
            'negative_prompt': negative_prompt,
        }

        if prompt_2:
            out['prompt_2'] = prompt_2
        if negative_prompt_2:
            out['negative_prompt_2'] = negative_prompt_2

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
