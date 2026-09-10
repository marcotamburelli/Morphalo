from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Tuple, Union

from morphalo.core.paths import load_latest_output
from morphalo.dag.core import _DagContext


class _NodeLike(Protocol):
    id: str


NodeRefOrId = Union[_NodeLike, str]


class Runtime:
    """
    Read-only helpers for inspecting latest sidecar outputs in the active DAG.

    ``Runtime`` resolves the current root DAG output directory from the active
    DAG scope and reads persisted JSON sidecars produced by previous node
    executions. It does not execute nodes, force upstream work, or keep an
    in-memory cache.

    Notes
    -----
    These helpers are intended for declaration-time code that needs metadata
    from already materialized outputs in the current DAG. All lookups are scoped
    to the active root ``DAG``.
    """

    @staticmethod
    def get_out(node: NodeRefOrId) -> dict[str, Any]:
        """
        Return the latest materialized node output as a dictionary.

        The output is loaded from the most recent JSON sidecar written for the
        node in the active DAG output directory.

        Parameters
        ----------
        node : NodeRef or str
            Node object exposing an ``id`` attribute, or a fully-qualified node
            id string.

        Returns
        -------
        dict
            Parsed JSON sidecar for the latest materialized output of ``node``.

        Raises
        ------
        RuntimeError
            If called outside an active ``DAG`` scope.
        FileNotFoundError
            If no sidecar exists for ``node`` in the current DAG output
            directory.
        """
        node_id = Runtime._node_id(node)
        out = Runtime.maybe_get_out(node_id)
        if out is None:
            raise FileNotFoundError(
                f"No cached output found for node {node_id!r} in "
                f"{Runtime._out_dir()!s}."
            )
        return out

    @staticmethod
    def maybe_get_out(node: NodeRefOrId) -> Optional[dict[str, Any]]:
        """
        Return the latest materialized node output, if present.

        The output is loaded from the most recent JSON sidecar written for the
        node in the active DAG output directory.

        Parameters
        ----------
        node : NodeRef or str
            Node object exposing an ``id`` attribute, or a fully-qualified node
            id string.

        Returns
        -------
        dict or None
            Parsed JSON sidecar for the latest materialized output of ``node``,
            or ``None`` when no sidecar is available.

        Raises
        ------
        RuntimeError
            If called outside an active ``DAG`` scope.
        """
        return load_latest_output(
            out_dir=Runtime._out_dir(),
            node_id=Runtime._node_id(node),
        )

    @staticmethod
    def get_output_size(node: NodeRefOrId) -> Tuple[int, int]:
        """
        Resolve the latest output image size for a node.

        Parameters
        ----------
        node : NodeRef or str
            Node object exposing an ``id`` attribute, or a fully-qualified node
            id string.

        Returns
        -------
        tuple[int, int]
            Output size as ``(width, height)``.

        Raises
        ------
        RuntimeError
            If called outside an active ``DAG`` scope.
        FileNotFoundError
            If no sidecar exists for ``node``.
        KeyError
            If the latest sidecar does not expose an output size through one of
            the supported metadata shapes.

        Notes
        -----
        Size resolution is attempted in this order:

        1. ``params.width`` and ``params.height``
        2. ``output_size``
        3. ``resize.resolved_size``
        """
        node_id = Runtime._node_id(node)
        out = Runtime.get_out(node_id)

        size = Runtime._size_from_params(out)
        if size is not None:
            return size

        size = Runtime._size_from_key(out, 'output_size')
        if size is not None:
            return size

        resize = out.get('resize')
        if isinstance(resize, Mapping):
            size = Runtime._size_from_key(resize, 'resolved_size')
            if size is not None:
                return size

        raise KeyError(
            f"Could not resolve output size for node {node_id!r}. Expected "
            "'params.width'/'params.height', 'output_size', or "
            "'resize.resolved_size' in latest sidecar."
        )

    @staticmethod
    def get_input_size(node: NodeRefOrId) -> Tuple[int, int]:
        """
        Resolve the latest input image size for a node.

        Parameters
        ----------
        node : NodeRef or str
            Node object exposing an ``id`` attribute, or a fully-qualified node
            id string.

        Returns
        -------
        tuple[int, int]
            Input size as ``(width, height)``.

        Raises
        ------
        RuntimeError
            If called outside an active ``DAG`` scope.
        FileNotFoundError
            If no sidecar exists for ``node``.
        KeyError
            If the latest sidecar does not expose ``input_size``.
        """
        node_id = Runtime._node_id(node)
        out = Runtime.get_out(node_id)

        size = Runtime._size_from_key(out, 'input_size')
        if size is not None:
            return size

        raise KeyError(
            f"Could not resolve input size for node {node_id!r}. Expected "
            "'input_size' in latest sidecar."
        )

    @staticmethod
    def _node_id(node: NodeRefOrId) -> str:
        return node if isinstance(node, str) else node.id

    @staticmethod
    def _out_dir():
        if not _DagContext.has_current():
            raise RuntimeError(
                'Runtime requires an active DAG context. Use '
                '`with DAG(...) as dag:`.'
            )

        root = _DagContext._stack[0]
        out_dir = getattr(root, 'out_dir', None)
        if out_dir is None:
            raise RuntimeError(
                'Runtime requires an active root DAG with an `out_dir`.'
            )
        return out_dir

    @staticmethod
    def _size_from_params(out: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        params = out.get('params')
        if not isinstance(params, Mapping):
            return None
        if 'width' not in params or 'height' not in params:
            return None
        return int(params['width']), int(params['height'])

    @staticmethod
    def _size_from_key(
        out: Mapping[str, Any],
        key: str,
    ) -> Optional[Tuple[int, int]]:
        value = out.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        return int(value[0]), int(value[1])
