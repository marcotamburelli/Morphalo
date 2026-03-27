from typing import (Any, Dict, List, Optional, Tuple, TypeAlias, TypedDict,
                    Union)

from morphalo.dag import AttachmentSink, NodeRef


class PromptDict(TypedDict, total=False):
    content: Union[str, List[str]]
    style: Union[str, List[str]]


PromptValue: TypeAlias = Union[
    str,
    List[str],
    PromptDict,
    None,
]


def norm_prompt(value: Any, *, joiner: str = '\n') -> str:
    """
    Normalize a prompt that can be:
      - str
      - list[str]
    """

    if value is None:
        return ''

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        # filter Nones / non-str defensively
        parts = [str(x).strip()
                 for x in value if x is not None and str(x).strip()]
        return joiner.join(parts).strip()

    # be strict: better to fail early than silently stringify weird objects
    raise TypeError(
        f'Prompt must be str or list[str], got {type(value).__name__}')


def norm_prompt_pair(value: PromptValue, *, joiner: str = '\n') -> Tuple[str, str]:
    """
    Normalize a prompt that can be:
      - str
      - list[str]
      - dict with keys: content/style (each str or list[str])

    Returns (content, style).
    """

    if value is None:
        return '', ''

    # legacy / simple form: treat as content
    if isinstance(value, (str, list)):
        return norm_prompt(value, joiner=joiner), ''

    # structured form
    if isinstance(value, dict):
        content = norm_prompt(value.get('content'), joiner=joiner)
        style = norm_prompt(value.get('style'), joiner=joiner)
        return content, style

    raise TypeError(
        f'Prompt must be str, list[str], or dict, got {type(value).__name__}')


class PromptRegistry:
    """
    Single-slot prompt attachment for a node.

    This registry intentionally exposes a single prompt channel. Wiring more
    than one upstream prompt into the same node will fail DAG validation due
    to duplicate input_id.
    """

    INPUT_ID = 'prompt:default'

    def __init__(self, owner: NodeRef):
        """
        Initialize a prompt registry for a node.

        Parameters
        ----------
        owner : NodeRef
            The node that owns this registry. All attachment sinks created
            by this registry will target this node.
        """

        self._owner = owner
        self._counter = 0

    def __call__(
        self,
        sink_id: Optional[str] = None
    ):
        """
        Create an attachment sink for the prompt input channel.

        This method returns an ``AttachmentSink`` bound to the owning node and
        targeting the fixed prompt input channel (``prompt:default``). The sink
        represents a single wiring point in the DAG DSL.

        Parameters
        ----------
        sink_id : str, optional
            Optional unique identifier for the sink instance. This identifier
            represents the identity of the attachment sink itself (for debugging,
            logging, or future extensions), and is independent of the semantic
            input channel.

            If not provided, a unique sink identifier is automatically generated
            by the registry. Automatically generated identifiers are guaranteed
            to be unique within the lifetime of the registry instance.

        Returns
        -------
        AttachmentSink
            An attachment sink targeting the owning node on the fixed prompt
            input channel (``prompt:default``).

        Notes
        -----
        - The prompt channel is intentionally single-slot: wiring more than one
        upstream prompt into the same node will fail DAG validation due to
        duplicate ``input_id``.
        - The ``sink_id`` identifies the sink instance only and does not affect
        input routing semantics, which are entirely determined by ``input_id``.
        """
        if sink_id is None:
            self._counter += 1
            # unique sink identity (NOT the input channel)
            sink_id = f'sink:prompt:{self._owner.id}:{self._counter}'

        return AttachmentSink(
            name=sink_id,
            target=self._owner,
            input_id=PromptRegistry.INPUT_ID
        )


class PromptBundle:
    def __init__(self, *, spec: Dict[str, Any], input: Optional[Dict[str, Dict]]):
        input = input or {}
        upstream = input.get(PromptRegistry.INPUT_ID) or {}

        # prompts in spec has lower priority
        prompt, prompt_2 = norm_prompt_pair(spec.get('prompt'))
        negative_prompt, negative_prompt_2 = norm_prompt_pair(
            spec.get('negative_prompt'),
            joiner=', '
        )

        self._prompt = upstream.get('prompt') or prompt or ''
        self._prompt_2 = upstream.get('prompt_2') or prompt_2 or None
        self._negative_prompt = upstream.get('negative_prompt') \
            or negative_prompt or ''
        self._negative_prompt_2 = upstream.get('negative_prompt_2') \
            or negative_prompt_2 or None

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def prompt_2(self) -> Optional[str]:
        return self._prompt_2

    @property
    def negative_prompt(self) -> str:
        return self._negative_prompt

    @property
    def negative_prompt_2(self) -> Optional[str]:
        return self._negative_prompt_2
