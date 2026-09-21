import hashlib
import re
from typing import Tuple

from morphalo.dag import NodeRef, add_to_scope_cache, get_from_scope_cache


_SAFE_ID_RE = re.compile(r'[^A-Za-z0-9_]+')


def _source_nodes(node: NodeRef) -> Tuple[NodeRef, ...]:
    """
    Return the concrete source nodes represented by a prompt expression.

    Plain prompt-capable nodes represent themselves. ``PromptMerge`` instances
    carry their flattened sources so chained expressions such as ``a + b + c``
    can build one ordered merge instead of nesting merge nodes.
    """
    sources = getattr(node, '_prompt_merge_sources', None)
    if sources is None:
        return (node,)
    return tuple(sources)


def _source_name(source: NodeRef) -> str:
    """
    Return the stable local name used to identify a merge source.

    Explicit node names are preferred. Auto-generated nodes fall back to the
    final component of their scope-qualified id.
    """
    return source.name or source.id.split('.')[-1]


def _merge_name(source_names: tuple[str, ...]) -> str:
    """
    Build a readable, deterministic node name for a prompt merge.

    Each source contributes a short readable prefix. A hash of the full ordered
    source names preserves uniqueness when prefixes collide.
    """
    digest = hashlib.sha1('\0'.join(source_names).encode('utf-8')).hexdigest()[:10]
    readable = '_'.join(source_name[:10].strip('_') for source_name in source_names)
    readable = _SAFE_ID_RE.sub('_', readable).strip('_') or 'prompts'
    return f'{readable}_{digest}'


class PromptOutputMixin:
    """
    Mixin for nodes whose output can be treated as prompt payloads.

    The mixin adds ``+`` as prompt-bundle merge syntax for source,
    pass-through, and prompt-composition nodes. The actual payload validation
    and field concatenation are deferred to ``PromptMerge`` at runtime, so the
    DAG core and runner remain unaware of prompt-specific fields such as
    ``prompt``, ``prompt_2``, ``negative_prompt`` and ``negative_prompt_2``.
    """

    def _merge_with(self, other):
        from morphalo.nodes.prompt_merge import PromptMerge

        sources = _source_nodes(self) + _source_nodes(other)
        source_names = tuple(_source_name(source) for source in sources)

        existing = get_from_scope_cache('prompt_merge', source_names)
        if existing is not None:
            return existing

        merge = PromptMerge(
            name=_merge_name(source_names),
            source_names=source_names,
        )
        merge._prompt_merge_sources = sources

        for idx, source in enumerate(sources):
            source >> merge.prompt(idx)

        return add_to_scope_cache('prompt_merge', source_names, merge)

    def __add__(self, other):
        """
        Return a ``PromptMerge`` node representing ``self + other``.

        The merge is graph-local and order-sensitive: ``a + b`` is distinct
        from ``b + a``. Repeating the same ordered expression inside one active
        DAG or node group reuses the existing merge node from the scope cache
        instead of declaring a duplicate node id.

        Chained expressions are flattened. For example, ``a + b + c`` creates
        one merge whose runtime inputs are ``prompt:0``, ``prompt:1`` and
        ``prompt:2`` in that order.

        Parameters
        ----------
        other : PromptOutputMixin
            Another prompt-capable node or merge expression.

        Returns
        -------
        PromptMerge
            A pass-through node that emits one concatenated prompt bundle.
        """
        if not isinstance(other, PromptOutputMixin):
            return NotImplemented
        return self._merge_with(other)

    def __radd__(self, other):
        """
        Return a ``PromptMerge`` node for reflected ``+`` operations.

        This allows prompt-capable nodes to participate in mixed multiple
        inheritance cases where another mixin's ``__add__`` declines the
        operation first.
        """
        if not isinstance(other, PromptOutputMixin):
            return NotImplemented
        return other._merge_with(self)
