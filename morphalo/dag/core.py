from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple, Union

from morphalo.core.ids import validate_dotted_identifier

Dest = Union['NodeRef', 'AttachmentSink']
DestList = List[Dest]
DestTuple = Tuple[Dest, ...]
DestSeq = Union[DestList, DestTuple]


def get_entry_nodes(nodes: List['NodeRef'], edges: List['Edge']) -> List['NodeRef']:
    """
    Return nodes with no incoming edges.

    Parameters
    ----------
    nodes : list[NodeRef]
        Nodes of the graph.
    edges : list[Edge]
        Edges of the graph.

    Returns
    -------
    list[NodeRef]
        Nodes whose id does not appear as `node_to` in any edge.
    """
    has_incoming = {e.node_to for e in edges}
    return [n for n in nodes if n.id not in has_incoming]


def get_out_nodes(nodes: List["NodeRef"], edges: List["Edge"]) -> List["NodeRef"]:
    """
    Return nodes with no outgoing edges.

    Parameters
    ----------
    nodes : list[NodeRef]
        Nodes of the graph.
    edges : list[Edge]
        Edges of the graph.

    Returns
    -------
    list[NodeRef]
        Nodes whose id does not appear as `node_from` in any edge.
    """
    has_outgoing = {e.node_from for e in edges}
    return [n for n in nodes if n.id not in has_outgoing]


class DagRegistry:
    """
    In-memory registry of DAG objects created during module import.

    This registry enables an Airflow-like pattern where DAG files only
    *declare* DAGs at import time, while a separate runner/CLI later
    discovers and executes them.

    Notes
    -----
    - The registry is process-local (not shared across processes).
    - It is intended for discovery; execution is handled by a DAGRunner.
    - Call ``clear()`` before importing a module to avoid stale DAGs.
    """

    _dags: List['DAG'] = []

    @classmethod
    def add(cls, dag: 'DAG') -> None:
        """Register a DAG instance."""
        cls._dags.append(dag)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered DAGs."""
        cls._dags.clear()

    @classmethod
    def all(cls) -> List['DAG']:
        """Return all registered DAGs in definition order."""
        return list(cls._dags)

    @classmethod
    def get(cls, name: str) -> Optional['DAG']:
        """Return the first DAG with the given name, if any."""
        for d in cls._dags:
            if getattr(d, 'name', None) == name:
                return d
        return None


@dataclass
class Edge:
    node_from: str
    node_to: str
    input_id: str


class GraphScope:
    """
    Declarative graph scope used to collect nodes and edges during DAG construction.

    ``GraphScope`` represents a structural context in which ``NodeRef`` instances
    and ``Edge`` dependencies are declared. It is the common base class for both:

    - ``DAG``: a root, executable workflow with an associated output directory.
    - ``NodeGroup``: a reusable subgraph that can be nested inside a DAG.

    A ``GraphScope`` is typically entered using a ``with`` statement. While the
    scope is active, newly created nodes automatically register themselves into
    the current scope via ``_DagContext``, and wiring operations (``>>``) add
    edges to this scope.

    Nested scopes are supported through a global context stack. The active
    scope stack is used to derive a hierarchical prefix (via
    ``_DagContext.prefix()``), which is automatically prepended to node
    identifiers created within nested scopes. This ensures that node ids remain
    unique when groups are composed or reused.

    Naming
    ------
    Scope names must be valid dotted identifiers. Dots are allowed and express
    logical hierarchy; whitespace and path separators are not allowed.

    Valid examples include ``"refine"``, ``"img_0"``, and
    ``"macro.refine"``. Invalid examples include ``"my..group"``,
    ``".group"``, ``"group."``, ``"my/group"``, and ``"my group"``.

    Notes
    -----
    - ``GraphScope`` is purely declarative. It does not execute any computation.
    - Only root ``DAG`` instances are registered in ``DagRegistry`` and are
      considered executable workflows.
    - ``NodeGroup`` instances act as composable subgraphs and are not registered.
    - Node identifiers generated inside a scope are automatically qualified by
      the active scope path to prevent collisions.

    Attributes
    ----------
    node_groups : dict[str, NodeGroup]
        Injected child groups keyed by their scope-qualified identifiers.
    cache : dict[str, Any]
        Declaration-time scratch cache for DSL helpers that need to intern
        graph-local helper objects. The cache is scoped to this graph only and
        is not used by the runner.
    """

    def __init__(self, name: str):
        validate_dotted_identifier(name, kind='Graph scope name')
        self.name = name
        self.nodes: List['NodeRef'] = []
        self.edges: List['Edge'] = []
        self.node_groups: Dict[str, 'NodeGroup'] = {}
        self.cache: Dict[str, Any] = {}
        self._id_counter = 0

    def next_id(self, prefix: str) -> str:
        """
        Generate a local, scope-specific identifier.

        This counter is local to the current scope. The returned identifier
        may later be qualified with a hierarchical prefix derived from the
        active scope stack.
        """
        self._id_counter += 1
        return f'{prefix}_{self._id_counter}'

    def __enter__(self) -> Self:
        """
        Enter this graph scope.

        The scope is pushed onto the global ``_DagContext`` stack, making it
        the active destination for newly created nodes and edges.
        """
        _DagContext.push(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit this graph scope.

        The scope is removed from the global ``_DagContext`` stack. Subclasses
        (e.g. ``DAG``) may extend this behavior to perform registration or
        validation.
        """
        _DagContext.pop()

    def add_node(self, node: 'NodeRef') -> None:
        """
        Register a node inside this scope.

        Parameters
        ----------
        node : NodeRef
            The node to register.
        """
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """
        Register a dependency edge inside this scope.

        Parameters
        ----------
        edge : Edge
            The directed edge to register.
        """
        self.edges.append(edge)

    def add_node_group(self, group: 'NodeGroup') -> None:
        """
        Register an injected node group by its scope-qualified identifier.
        """
        existing = self.node_groups.get(group.id)
        if existing is not None and existing is not group:
            raise RuntimeError(
                f'Node group id collision {group.id!r} in parent scope'
            )
        self.node_groups[group.id] = group


class DAG(GraphScope):
    """
    Root directed acyclic graph (DAG) describing an executable workflow.

    A ``DAG`` represents a static computation graph composed of ``NodeRef``
    instances (nodes) and ``Edge`` objects (directed dependencies). It is the
    root, executable graph unit consumed by a ``DAGRunner``.

    The DAG is typically constructed inside a ``with DAG(...):`` context.
    While the context is active:

    - Newly created nodes automatically register themselves into the DAG.
    - Wiring operations (``>>``) add dependency edges between nodes.
    - Nested ``NodeGroup`` scopes may be declared to structure and reuse
      subgraphs.

    ``DAG`` instances are declarative: they do not execute computation
    themselves. Execution is delegated to a ``DAGRunner``, which evaluates
    nodes in topological order based on the declared edges.

    Only root ``DAG`` instances are registered in ``DagRegistry`` upon
    successful exit of their context manager.

    Naming
    ------
    DAG names must be valid dotted identifiers. The root DAG name is used as a
    registry key and for workflow selection; it is not included in node artifact
    paths.

    Node names declared inside the DAG, when explicitly provided via
    ``NodeRef(name=...)`` or a concrete node constructor, must follow the same
    dotted identifier convention. Dots in node names are meaningful:
    ``name="stage.depth"`` produces a node id ``"stage.depth"`` and artifacts
    under ``out_dir/stage/depth``.

    A valid name has one or more non-empty components separated by dots and no
    whitespace or path separators. For example, ``"standing_refine"``,
    ``"stage.depth"``, and ``"img_0.out"`` are valid; ``"stage..depth"``,
    ``".depth"``, ``"depth."``, ``"stage/depth"``, and ``"stage depth"`` are
    invalid.

    Notes
    -----
    - Node identifiers created inside nested scopes are automatically
      qualified using the active scope path (via ``_DagContext``) to ensure
      global uniqueness within the DAG.
    - ``NodeGroup`` instances are structural subgraphs and are not registered
      as executable DAGs.

    Attributes
    ----------
    name : str
        Human-readable identifier for the DAG.
    out_dir : str or Path
        Base output directory associated with this DAG. This path is passed
        to nodes during execution and may be used to store generated artifacts.
    nodes : list[NodeRef]
        All nodes belonging to the DAG (including those declared inside nested
        groups).
    edges : list[Edge]
        All directed edges defining data dependencies between nodes.
    """

    def __init__(self, name: str, out_dir: str | Path):
        """
        Initialize a new DAG.

        Parameters
        ----------
        name : str
            Human-readable identifier for the DAG. Must be a valid dotted
            identifier. The DAG name is not included in node artifact paths.
        out_dir : str
            Base output directory associated with this DAG. The directory
            is not created automatically; it is passed to nodes during
            execution and may be created by the runner or individual nodes.
        """
        super().__init__(name)
        self.out_dir = out_dir

    def __exit__(self, exc_type, exc, tb) -> None:
        super().__exit__(exc_type, exc, tb)
        if exc_type is None:
            DagRegistry.add(self)


@dataclass(frozen=True)
class PortRef:
    """
    A group port bound to a specific internal entry node.

    A PortRef is meant to be used as a wiring target:

        src >> group('some_entry')

    The port wires `src` into the referenced entry node and returns the
    owning group to allow chaining.

    Notes
    -----
    This is intentionally minimal: a port is just an entry-node alias.
    """
    node: 'NodeRef'
    node_group: 'NodeGroup'

    def __rrshift__(self, src: 'NodeRef') -> 'NodeGroup':
        """
        Wire an upstream node into this port and return the group for chaining.
        """
        self.node_group._ensure_injected()
        src >> self.node
        return self.node_group

    def __rshift__(self, other: Union['Dest', 'DestSeq']) -> Union['Dest', 'DestSeq']:
        """
        Wire the group output node to a downstream destination.

        Notes
        -----
        This enables fluent chaining:

            src >> group('port') >> downstream
        """
        self.node_group._ensure_injected()
        out = self.node_group.get_out_node()
        return out >> other


class NodeGroup(GraphScope):
    """
    Reusable subgraph scope (group) that can be nested inside a root DAG.

    A ``NodeGroup`` is a declarative container used to define a portion of a
    workflow as a self-contained subgraph.

    Unlike a root ``DAG``, a ``NodeGroup`` is not executable on its own:
    it has no ``out_dir`` and it is not registered in ``DagRegistry``.

    Naming
    ------
    Group names must be valid dotted identifiers and become part of the
    scope-qualified node ids for nodes declared inside the group. For example:

        with NodeGroup("img_0"):
            ImgAuxMap(name="depth_midas")

    creates the node id ``"img_0.depth_midas"`` and stores artifacts under
    ``out_dir/img_0/depth_midas``.

    Node names inside groups, when explicitly provided via ``NodeRef(name=...)``
    or a concrete node constructor, use the same dotted identifier rules as
    top-level node names. Dots are allowed to create additional artifact
    hierarchy, so ``NodeGroup("img_0")`` with ``name="maps.depth"`` produces
    ``"img_0.maps.depth"`` and artifacts under ``out_dir/img_0/maps/depth``.

    Valid examples include ``"img_0"``, ``"macro.refine"``, and
    ``"maps.depth"``. Invalid examples include ``"img..0"``, ``".img"``,
    ``"img."``, ``"img/0"``, and ``"img 0"``.

    Entry and output nodes
    ----------------------
    - ``get_entry_nodes()`` returns nodes with no incoming edges within the group.
    - ``get_out_node()`` returns the single terminal node of the group.

    Ports
    -----
    This implementation supports an optional "port" mechanism that simply
    aliases one or more *entry nodes* by their local names.

    - By default, ``src >> group`` broadcasts ``src`` into all entry nodes.
    - If ports are registered, you can wire to a specific entry via:
          ``src >> group('entry_name')``

    Notes
    -----
    Ports are intentionally thin and do not introduce a separate abstraction
    layer. They are just named references to entry nodes.

    Attributes
    ----------
    id : str
        Scope-qualified group identifier, such as ``outer.inner``.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.id = _DagContext.prefix() + name
        validate_dotted_identifier(self.id, kind='Node group id')
        self._ports: Dict[str, PortRef] = {}
        self._injected: bool = False
        self._injected_into: Optional[int] = None

    def get_entry_nodes(self) -> List['NodeRef']:
        return get_entry_nodes(nodes=self.nodes, edges=self.edges)

    def get_out_node(self) -> 'NodeRef':
        ends = get_out_nodes(nodes=self.nodes, edges=self.edges)
        if not ends:
            raise RuntimeError(
                f'{self.name!r} has no end node (graph may be cyclic or empty)'
            )
        if len(ends) != 1:
            ids = ', '.join(n.id for n in ends)
            raise RuntimeError(
                f'{self.name!r} must have a single output node, found {len(ends)}: {ids}'
            )
        return ends[0]

    def register_ports(self, *nodes: 'NodeRef') -> None:
        """
        Register entry nodes as selectable ports.

        Parameters
        ----------
        *nodes : NodeRef
            Nodes to be exposed as ports. Each node name becomes the port key.

        Raises
        ------
        RuntimeError
            If a provided node is not an entry node within the group.
        ValueError
            If a duplicate port name is registered.

        Notes
        -----
        Call this after internal edges have been defined, otherwise entry
        detection may be incorrect.
        """
        entry_names = {n.name for n in self.get_entry_nodes()}

        for node in nodes:
            port_name = node.name

            if port_name not in entry_names:
                raise RuntimeError(
                    f'Node {port_name!r} is not an entry node in group {self.name!r}. '
                    'Register ports only after internal edges are defined.'
                )

            if port_name in self._ports:
                raise ValueError(
                    f'Duplicate port {port_name!r} in group {self.name!r}'
                )

            self._ports[port_name] = PortRef(node=node, node_group=self)

    def port(self, name: str) -> PortRef:
        """
        Return a registered port by name.

        Raises
        ------
        KeyError
            If the port name is not registered.
        """
        if name not in self._ports:
            raise KeyError(
                f'Unknown port {name!r} for group {self.name!r}. '
                f'Known ports: {sorted(self._ports)}'
            )
        return self._ports[name]

    def _ensure_injected(self) -> None:
        if self._injected:
            return

        parent = _DagContext.current()

        # Prevent injecting a group into itself (would only happen with misuse)
        if parent is self:
            raise RuntimeError('Cannot inject a NodeGroup into itself')

        # Optional but recommended: fail fast on id collisions in the parent
        parent_ids = {n.id for n in parent.nodes}
        for n in self.nodes:
            if n.id in parent_ids:
                raise RuntimeError(
                    f'Cannot inject group {self.name!r}: node id collision {n.id!r} in parent scope'
                )

        for n in self.nodes:
            parent.add_node(n)

        for e in self.edges:
            parent.add_edge(e)

        parent.add_node_group(self)
        for group in self.node_groups.values():
            parent.add_node_group(group)

        self._injected = True
        self._injected_into = id(parent)

    def __rrshift__(self, src: 'NodeRef') -> 'NodeGroup':
        """
        Wire an upstream node into this group and return the group for chaining.

        Policy: broadcast to all entry nodes.
        """
        self._ensure_injected()

        entries = self.get_entry_nodes()
        if not entries:
            raise RuntimeError(
                f'{self.name!r} has no entry nodes (group is empty?)'
            )

        # Policy: either require single entry OR broadcast.
        # You chose to accept multiple, so this fans out (src -> each entry).
        src >> entries

        return self

    def __rshift__(self, other: Union[Dest, DestSeq]) -> Union[Dest, DestSeq]:
        """
        Wire this group's single output node to a downstream destination.
        """
        self._ensure_injected()
        out = self.get_out_node()
        return out >> other

    def __call__(self, target: str) -> PortRef:
        """
        Return a port reference (entry node alias) by name.

        This enables:
            src >> group('entry_name') >> downstream

        Notes
        -----
        Ports must be registered via :meth:`register_ports`.
        """
        self._ensure_injected()
        return self.port(target)


class _DagContext:
    """
    Global graph scope context stack.

    ``_DagContext`` maintains a process-local stack of active ``GraphScope``
    instances (e.g. ``DAG`` and ``NodeGroup``). The top of the stack
    represents the currently active scope where newly created nodes and edges
    are registered.

    The active scope path is also used to derive a hierarchical prefix
    (via ``prefix()``), which is automatically prepended to node identifiers
    created inside nested scopes. This ensures that node ids remain globally
    unique within a root DAG when groups are nested or reused.

    Responsibilities
    ----------------
    - Track the currently active ``GraphScope``.
    - Provide access to the active scope via ``current()``.
    - Maintain the nesting stack for ``with``-based scope management.
    - Generate hierarchical name prefixes based on the active scope path.

    Notes
    -----
    - This class is purely declarative and does not execute any graph logic.
    - Only the outermost ``DAG`` scope is considered an executable workflow.
    - Duplicate scope names on the active stack are rejected to prevent
      ambiguous hierarchical prefixes.
    - The context is process-local and not thread-safe.
    """

    _stack: list[GraphScope] = []

    @classmethod
    def current(cls) -> GraphScope:
        if not cls._stack:
            raise RuntimeError(
                'No active DAG context. Use `with DAG(...) as dag:`.'
            )
        return cls._stack[-1]

    @classmethod
    def push(cls, dag: GraphScope) -> None:
        # Disallow duplicate names on the active stack to avoid ambiguous prefixes.
        name = getattr(dag, 'name', None)
        if name is None:
            raise ValueError(
                'DAG must have a non-empty `name` to be used in context'
            )

        for d in cls._stack:
            if getattr(d, 'name', None) == name:
                raise RuntimeError(
                    f'Nested DAG name collision: a DAG named {name!r} is already active'
                )

        cls._stack.append(dag)

    @classmethod
    def pop(cls) -> GraphScope:
        if not cls._stack:
            raise RuntimeError('No active DAG context to pop')
        return cls._stack.pop()

    @classmethod
    def has_current(cls) -> bool:
        return bool(cls._stack)

    @classmethod
    def depth(cls) -> int:
        return len(cls._stack)

    @classmethod
    def path(cls, *, include_root: bool = True) -> list[str]:
        """
        Return the active DAG name path as a list of segments.

        Parameters
        ----------
        include_root : bool
            If True, include the root DAG name as the first segment.
            If False, return only nested segments (empty at root scope).
        """
        if not cls._stack:
            return []
        names = [getattr(d, 'name') for d in cls._stack]
        return names if include_root else names[1:]

    @classmethod
    def prefix(cls, *, include_root: bool = False, sep: str = '.') -> str:
        """
        Return a dotted prefix representing the current nested DAG path.

        Default behavior excludes the root DAG name, so at top-level this returns
        an empty string. Nested scopes return e.g. "subdagA.subdagB".

        Parameters
        ----------
        include_root : bool
            If True, include the root DAG name in the prefix.
        sep : str
            Separator used to join path segments.
        """
        parts = cls.path(include_root=include_root)
        return sep.join(parts) + '.' if parts else ''


_MISSING = object()


def get_from_scope_cache(
    namespace: str,
    key: Any,
    default: Any = None,
) -> Any:
    """
    Return a value from the active graph scope cache.

    The cache is scoped to the currently active DAG or node group. It is meant
    for declaration-time DSL helpers that need graph-local interning without
    exposing the graph scope itself.
    """
    scope = _DagContext.current()
    scoped_cache = scope.cache.get(namespace)
    if scoped_cache is None:
        return default
    return scoped_cache.get(key, default)


def add_to_scope_cache(
    namespace: str,
    key: Any,
    value: Any,
    *,
    replace: bool = False,
) -> Any:
    """
    Store a value in the active graph scope cache and return it.

    By default, adding a duplicate key is an error. Pass ``replace=True`` when
    overwriting is intentional.
    """
    scope = _DagContext.current()
    scoped_cache = scope.cache.setdefault(namespace, {})
    existing = scoped_cache.get(key, _MISSING)
    if existing is not _MISSING and not replace:
        raise KeyError(
            f'Scope cache entry already exists for namespace={namespace!r}, '
            f'key={key!r}'
        )
    scoped_cache[key] = value
    return value


@dataclass(kw_only=True)
class NodeRef:
    """
    Base reference for a computation node inside a graph scope.

    A ``NodeRef`` represents a runnable computation unit (an operator) declared
    within an active ``GraphScope`` (typically a root ``DAG`` or a nested
    ``NodeGroup``).

    Nodes are usually instantiated inside a ``with DAG(...):`` or
    ``with NodeGroup(...):`` context. During construction, each node:

    - Derives its operator name (``op``) from the concrete class.
    - Generates a unique identifier (``id``), optionally based on the provided
    ``name``.
    - Automatically registers itself into the currently active graph scope
    via ``_DagContext``.

    Identifier semantics
    --------------------
    The final ``id`` of a node is automatically qualified using the active
    scope path provided by ``_DagContext``. This means that nodes created
    inside nested groups receive a hierarchical prefix (e.g.
    ``groupA.groupB.node_1``), ensuring uniqueness within the effective root DAG.

    Naming
    ------
    Node names must be valid dotted identifiers. Dots are allowed and represent
    logical artifact hierarchy, even outside a ``NodeGroup``. For example,
    ``name="stage.depth"`` creates node id ``"stage.depth"`` at the DAG root and
    stores artifacts under ``out_dir/stage/depth``. Inside
    ``NodeGroup("img_0")``, the same name creates
    ``"img_0.stage.depth"`` and stores artifacts under
    ``out_dir/img_0/stage/depth``.

    Valid examples include ``"out"``, ``"depth_midas"``, ``"stage.depth"``,
    and ``"my.sub.node"``. Invalid examples include ``"my..node"``,
    ``".node"``, ``"node."``, ``"../node"``, ``"my/node"``, ``"my node"``,
    and names with leading or trailing whitespace.

    Dependency wiring DSL
    ----------------------
    ``NodeRef`` implements a small DSL using the right-shift operator:

    - ``a >> b`` creates a default edge from node ``a`` to node ``b`` using
    ``input_id="default"``.
    - ``a >> b.some_sink(...)`` delegates edge creation to an ``AttachmentSink``
    for typed or auxiliary inputs (e.g. ControlNet, IP-Adapter).

    Fan-out wiring is supported via lists or tuples:
    - ``a >> [b, c]`` connects ``a`` to multiple destinations.

    Execution contract
    ------------------
    Nodes are declarative graph elements. They do not execute themselves.
    Execution is performed by a ``DAGRunner``, which:

    - Resolves dependencies via edges.
    - Calls ``run(output_dir, input=...)`` in topological order.
    - Propagates each node's returned dictionary to downstream nodes.

    Parameters
    ----------
    name : str, optional
        Optional local identifier for the node within the current scope.
        If provided, it must be a valid dotted identifier. If not provided, an
        identifier is generated using the scope's internal counter.

    Attributes
    ----------
    id : str
        Globally unique identifier within the effective root DAG,
        including any hierarchical scope prefix.
    op : str
        Operator identifier derived from the concrete node class name.

    Notes
    -----
    This base class defines the structural and execution contract for nodes
    but does not implement any operator logic. Concrete subclasses must
    implement ``run()``.
    """

    name: Optional[str] = None
    op: str = field(init=False)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Finalize node initialization and register it in the active graph scope.

        This method is automatically invoked by ``dataclasses`` after object
        initialization. It performs three responsibilities:

        1. Derives the operator identifier (``op``) from the concrete class name.
        2. Generates a unique, scope-qualified node identifier (``id``).
        3. Registers the node in the currently active ``GraphScope``.

        Identifier semantics
        --------------------
        The final ``id`` is constructed by combining:

        - The hierarchical prefix derived from the active scope stack
        (via ``_DagContext.prefix()``).
        - Either the explicitly provided ``name`` or a scope-local
        auto-generated identifier (via ``next_id()``).

        This ensures that nodes created inside nested ``NodeGroup`` scopes
        receive a globally unique identifier within the effective root DAG.

        The scope-qualified node id is validated after it is assembled.

        Raises
        ------
        RuntimeError
            If no active graph scope exists (i.e. the node is created outside
            a ``with DAG(...)`` or ``with NodeGroup(...)`` block).
        """
        # operator name derived from concrete class
        self.op = self.__class__.__name__.lower()

        scope = _DagContext.current()

        # build scope-qualified identifier
        self.id = _DagContext.prefix() + (self.name or scope.next_id(self.op))
        validate_dotted_identifier(self.id, kind='Node id')

        # register node in current graph scope
        scope.add_node(self)

    # node >> other
    def __rshift__(self, other: Union[Dest, DestSeq]) -> Union[Dest, DestSeq]:
        """
        Wire this node to another destination using the ``>>`` operator.

        This method defines the default wiring semantics for ``NodeRef`` instances.

        Supported cases
        ---------------
        1) ``self >> other_node``:
        Creates a default edge from ``self`` to ``other_node`` with
        ``input_id="default"``. The meaning of this default input is defined
        by the destination node/operator (e.g. for an Img2Img node this may
        correspond to the init image).

        2) ``self >> attachment_sink``:
        Delegates edge creation to ``AttachmentSink.__rrshift__``. This is used
        for special or typed inputs (e.g. IP-Adapter, ControlNet, masks).

        3) ``self >> [node_or_sink, ...]``:
        Fans out the connection to multiple destinations by applying ``>>``
        to each element in the sequence. Only ``list`` and ``tuple`` are accepted.

        Chaining
        --------
        The method returns the destination object to enable fluent chaining:

            a >> b >> c

        For fan-out cases, the original sequence is returned.

        Interoperability with other graph elements
        ------------------------------------------
        If ``other`` is not a supported destination type handled directly by this
        method, ``NotImplemented`` is returned. This allows Python to invoke
        ``other.__rrshift__(self)`` when available (e.g. for ``NodeGroup``), enabling
        extended wiring semantics such as group injection.

        Parameters
        ----------
        other : NodeRef | AttachmentSink | Sequence[NodeRef | AttachmentSink]
            The destination of the connection, or a list/tuple of destinations.

        Returns
        -------
        NodeRef | AttachmentSink | Sequence[NodeRef | AttachmentSink] | NotImplemented
            The destination object for chaining, the original sequence in fan-out
            cases, or ``NotImplemented`` to allow right-hand operand handling.

        Raises
        ------
        RuntimeError
            If no active graph scope exists.
        """

        if isinstance(other, (list, tuple)):
            for dst in other:
                self >> dst
            return other

        dag = _DagContext.current()

        if isinstance(other, NodeRef):
            # default wiring: primary image -> default input
            dag.add_edge(Edge(
                node_from=self.id,
                node_to=other.id,
                input_id='default'
            ))
            return other

        if isinstance(other, AttachmentSink):
            # Let the sink decide the exact role (ip_adapter/controlnet/etc.)
            # This way `img >> out.ip_adapter(...)` works even if the sink wants to handle indexes/appends.
            return other.__rrshift__(self)

        return NotImplemented

    # Optional: node >>= other
    def __irshift__(self, other: Union['NodeRef', 'AttachmentSink']) -> 'NodeRef':
        """
        In-place right shift operator (``>>=``).

        This is functionally equivalent to ``self >> other`` but returns ``self``.
        It is provided mainly for completeness; most DAG wiring will use ``>>``.

        Parameters
        ----------
        other : NodeRef or AttachmentSink
            The destination of the connection.

        Returns
        -------
        NodeRef
            Returns ``self``.
        """
        res = self.__rshift__(other)
        if res is NotImplemented:
            raise TypeError(
                f"Cannot wire {type(self).__name__} >>= {type(other).__name__}")
        return self

    @property
    def uses_cuda(self) -> bool:
        """Return whether this node executes work on a CUDA device."""
        return False

    @abstractmethod
    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict:
        """
        Executes this node and returns its primary output.

        Implementations should treat ``input`` as a mapping from ``input_id`` to
        upstream outputs (dictionaries). For source nodes (nodes with no incoming
        edges), the runner may call ``run()`` with an empty mapping or ``None``
        depending on runner policy.

        Parameters
        ----------
        output_dir : str or Path-like
            Base directory where the node may write output artifacts.
        input : dict[str, dict], optional
            Mapping from input identifier to upstream node output dictionaries.
            The runner builds this mapping from incoming edges.

        Returns
        -------
        dict
            The node output. This dictionary will be propagated to all downstream
            edges as the value carried by the node.

        Notes
        -----
        - The exact expected keys of the returned dictionary are defined by the
          node/operator contract (e.g. it may include an ``"image"`` path).
        - Exceptions raised by this method are expected to propagate and fail the run.
        """
        pass

    def post_run(self) -> None:
        """Optional hook executed after run()."""
        pass


@dataclass
class AttachmentSink:
    """
    A declarative input sink representing a specific input channel of a target node.

    ``AttachmentSink`` models non-default or typed inputs of a ``NodeRef``.
    Typical examples include:

    - IP-Adapter reference images
    - ControlNet conditioning images
    - Mask inputs
    - Any auxiliary attachment that conceptually feeds into a node

    A sink is usually created by a target node (or a helper method on it) and
    participates in wiring via the ``>>`` operator:

        src_node >> sink

    This registers an edge from ``src_node`` to ``sink.target`` using
    ``sink.input_id`` to identify the input channel being populated.

    Unlike ``NodeRef``, an ``AttachmentSink``:

    - Is not a computation node.
    - Is not added to ``GraphScope.nodes``.
    - Does not participate in execution.
    - Exists purely to enrich wiring semantics.

    Identifier semantics
    --------------------
    Each sink receives a scope-qualified ``id`` at initialization time,
    derived from the active graph scope (via ``_DagContext.prefix()``).
    This identifier is primarily useful for debugging or higher-level
    deduplication logic.

    Parameters
    ----------
    name : str
        Logical identifier of the sink instance.
    target : NodeRef
        The node that will receive the attached input.
    input_id : str
        Input channel identifier used to construct the target node's
        ``input`` mapping during execution.

    Attributes
    ----------
    id : str
        Scope-qualified identifier of this sink.

    Notes
    -----
    ``AttachmentSink`` is purely declarative and does not execute any logic
    by itself. It only participates in graph wiring.
    """

    name: str
    target: NodeRef
    input_id: str

    id: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Finalize sink initialization and assign a scope-qualified identifier.

        This method is automatically invoked by ``dataclasses`` after initialization.
        It derives the sink's unique identifier using the active graph scope.

        Raises
        ------
        RuntimeError
            If no active graph scope exists (i.e. the sink is created outside
            a ``with DAG(...)`` or ``with NodeGroup(...)`` block).
        """
        dag = _DagContext.current()
        self.id = _DagContext.prefix() + (self.name or dag.next_id('sink'))

    def __rrshift__(self, src: NodeRef) -> 'AttachmentSink':
        """
        Wire a source node to this sink using the ``>>`` operator.

        This method is invoked when Python evaluates ``src >> sink`` and the
        sink appears on the right-hand side. It registers an edge in the active
        graph scope from ``src.id`` to ``self.target.id`` using ``self.input_id``.

        Returns ``self`` to allow chaining.
        """
        dag = _DagContext.current()
        dag.add_edge(Edge(
            node_from=src.id,
            node_to=self.target.id,
            input_id=self.input_id
        ))

        return self
