import hashlib
import re
from typing import Tuple

from morphalo.dag import NodeRef, add_to_scope_cache, get_from_scope_cache


_SAFE_ID_RE = re.compile(r'[^A-Za-z0-9_]+')


def _source_nodes(node: NodeRef) -> Tuple[NodeRef, ...]:
    """
    Return the concrete source nodes represented by an image expression.

    Plain image-capable nodes represent themselves. ``ImageMerge`` instances
    carry their flattened sources so chained expressions such as ``a + b + c``
    can build one ordered merge instead of nesting merge nodes.
    """
    sources = getattr(node, '_image_merge_sources', None)
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
    Build a readable, deterministic node name for an image merge.

    Each source contributes a short readable prefix. A hash of the full ordered
    source names preserves uniqueness when prefixes collide.
    """
    digest = hashlib.sha1('\0'.join(source_names).encode('utf-8')).hexdigest()[:10]
    readable = '_'.join(source_name[:10].strip('_') for source_name in source_names)
    readable = _SAFE_ID_RE.sub('_', readable).strip('_') or 'images'
    return f'{readable}_{digest}'


class ImageOutputMixin:
    """
    Mixin for nodes whose output can be treated as image payloads.

    The mixin adds ``+`` as image-list merge syntax for source, pass-through,
    and generative nodes. The actual payload validation is deferred to
    ``ImageMerge`` at runtime, so the DAG core and runner remain unaware of
    image-specific fields such as ``image`` and ``images``.
    """

    def __add__(self, other):
        """
        Return an ``ImageMerge`` node representing ``self + other``.

        The merge is graph-local and order-sensitive: ``a + b`` is distinct
        from ``b + a``. Repeating the same ordered expression inside one active
        DAG or node group reuses the existing merge node from the scope cache
        instead of declaring a duplicate node id.

        Chained expressions are flattened. For example, ``a + b + c`` creates
        one merge whose runtime inputs are ``image:0``, ``image:1`` and
        ``image:2`` in that order.

        Parameters
        ----------
        other : ImageOutputMixin
            Another image-capable node or merge expression.

        Returns
        -------
        ImageMerge
            A pass-through node that emits a combined ``images`` payload.
        """
        if not isinstance(other, ImageOutputMixin):
            return NotImplemented

        from morphalo.nodes.image_merge import ImageMerge

        sources = _source_nodes(self) + _source_nodes(other)
        source_names = tuple(_source_name(source) for source in sources)

        existing = get_from_scope_cache('image_merge', source_names)
        if existing is not None:
            return existing

        merge = ImageMerge(
            name=_merge_name(source_names),
            source_names=source_names,
        )
        merge._image_merge_sources = sources

        for idx, source in enumerate(sources):
            source >> merge.image(idx)

        return add_to_scope_cache('image_merge', source_names, merge)
