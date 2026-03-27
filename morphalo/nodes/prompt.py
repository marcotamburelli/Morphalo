from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from morphalo.cache.models import get_translator
from morphalo.const import ENG
from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.common.io import write_json_sidecar
from morphalo.nodes.wiring.prompt import PromptValue, norm_prompt_pair

DEFAULT_MODEL = 'facebook/nllb-200-distilled-600M'


def _translate(
        tr: Any,
        batch: List[str],
        *,
        max_length: Optional[int] = None,
        batch_size: int = 16
) -> List[str]:
    if not batch:
        return []

    kwargs = {"batch_size": batch_size}
    if max_length is not None:
        kwargs["max_length"] = max_length

    outs = tr(batch, **kwargs)
    return [o["translation_text"] for o in outs]


def translate_prompt_value(
    tr: Any,
    value: PromptValue,
    *,
    max_length: Optional[int] = None,
    batch_size: int = 16
) -> PromptValue:
    """
    Translate a prompt value preserving its original structure.

    Supported shapes:
      - str
      - list[str | None]
      - dict with keys 'content'/'style' (each str | list[str] | None)

    Empty or blank values are pruned.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return _translate(
            tr,
            [value],
            max_length=max_length,
            batch_size=batch_size
        )[0]

    if isinstance(value, list):
        clean_list = [
            str(x).strip()
            for x in value
            if x is not None and str(x).strip()
        ]

        if not clean_list:
            return []

        return _translate(
            tr,
            clean_list,
            max_length=max_length,
            batch_size=batch_size
        )

    if isinstance(value, dict):
        content = value.get('content', [])
        style = value.get('style', [])

        if isinstance(content, str):
            content = [content]
        if isinstance(style, str):
            style = [style]

        if not isinstance(content, list) or not isinstance(style, list):
            raise TypeError(
                "In case of dict, 'content' and 'style' must be str or list[str]. "
                f"Got {type(content).__name__} and {type(style).__name__}."
            )

        content_clean = [
            str(x).strip()
            for x in content
            if x is not None and str(x).strip()
        ]
        style_clean = [
            str(x).strip()
            for x in style
            if x is not None and str(x).strip()
        ]

        batch = content_clean + style_clean
        if not batch:
            return {}

        out_batch = _translate(
            tr,
            batch=batch,
            max_length=max_length,
            batch_size=batch_size
        )

        n = len(content_clean)
        out_dict: Dict[str, Any] = {}

        if content_clean:
            out_dict["content"] = out_batch[:n]
        if style_clean:
            out_dict["style"] = out_batch[n:]

        return out_dict

    raise TypeError(
        f'Prompt must be str, list[str], or dict, got {type(value).__name__}'
    )


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
        - ``translate_max_length`` (optional)
          Maximum output length passed to the translation pipeline when translating
          prompts to English. This parameter is forwarded at call time to avoid
          truncation of long prompts and suppress related warnings. If not specified,
          the default ``max_length`` of the translation model is used.

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
        spec = resolve_spec(self.spec)

        src_lang = spec.get('lang', ENG)
        model_id = spec.get('translate_model', DEFAULT_MODEL)
        device = spec.get('device', 'cuda')
        max_length = spec.get('translate_max_length')
        if max_length is not None:
            max_length = int(max_length)

        # Read raw structured values first (may be str | list | dict)
        raw_prompt = spec.get('prompt')
        raw_negative = spec.get('negative_prompt')

        translated = False

        if src_lang and src_lang != ENG:
            tr = get_translator(
                model_id=model_id,
                source_lang=src_lang,
                target_lang=ENG,
                device=device
            )
            raw_prompt = translate_prompt_value(
                tr,
                value=raw_prompt,
                max_length=max_length
            )
            raw_negative = translate_prompt_value(
                tr,
                value=raw_negative,
                max_length=max_length
            )

            translated = True

        # Normalize only after (optional) translation
        prompt, prompt_2 = norm_prompt_pair(raw_prompt)
        negative, negative_2 = norm_prompt_pair(raw_negative, joiner=', ')

        out: Dict[str, Any] = {
            'ok': True,
            'node': 'prompt',
            'lang': {'src': src_lang, 'tgt': ENG},
            'translated': translated,
            'model': {'translate_model': model_id},
            'prompt': prompt,
            'negative_prompt': negative,
            **({} if max_length is None else {'translate_max_length': max_length}),
        }

        if prompt_2:
            out['prompt_2'] = prompt_2
        if negative_2:
            out['negative_prompt_2'] = negative_2

        out_path = make_node_output_path(
            out_dir=Path(output_dir),
            node_id=self.id,
            ext='json',
        )

        meta_path = write_json_sidecar(out_path, out)
        out['metadata'] = str(meta_path)

        return out
