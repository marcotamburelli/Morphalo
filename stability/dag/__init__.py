from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Edge:
    node_from: str
    node_to: str
    input_id: str


@dataclass
class DAG:
    """
    Directed Acyclic Graph (DAG) describing a computational workflow.

    A ``DAG`` represents a static computation graph composed of nodes
    (``NodeRef`` instances) and directed edges (``Edge`` instances) that
    define data dependencies between nodes.

    The DAG is typically constructed inside a ``with DAG(...):`` context.
    While the context is active, newly created nodes automatically register
    themselves into the DAG, and wiring operations (via ``>>``) add edges
    between nodes.

    The DAG itself is a *declarative* structure: it does not execute any
    computation. Execution is delegated to a ``DAGRunner``, which consumes
    the DAG definition and evaluates nodes in dependency order.

    Attributes
    ----------
    name : str
        Human-readable identifier for the DAG. This value is intended for
        logging, debugging, and user-facing interfaces.
    out_dir : str
        Base output directory associated with this DAG. This path is passed
        to nodes during execution and may be used to store generated artifacts.
    nodes : list of NodeRef
        List of all nodes belonging to the DAG. Nodes are added automatically
        when instantiated inside the DAG context.
    edges : list of Edge
        List of directed edges defining dependencies between nodes. Each edge
        connects a source node to a destination node and specifies an
        ``input_id`` used to assemble inputs during execution.

    Notes
    -----
    - The DAG is assumed to be immutable once execution starts.
    - The DAG enforces no semantic constraints on nodes; it only encodes
      structure. Semantic validation (e.g. required inputs) must be handled
      separately.
    - The DAG must be acyclic. Cycles are considered structural errors and
      must be detected during validation.
    """

    name: str
    out_dir: str
    nodes: List['NodeRef'] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def __enter__(self) -> 'DAG':
        if _DagContext.has_current():
            raise RuntimeError("Nested DAGs are not supported")
        _DagContext.push(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _DagContext.pop()

    def add_node(self, node: 'NodeRef') -> None:
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)


class _DagContext:
    _dag: Optional[DAG] = None

    @classmethod
    def current(cls) -> DAG:
        if cls._dag is None:
            raise RuntimeError(
                'No active DAG context. Use `with DAG(...) as dag:`.')
        return cls._dag

    @classmethod
    def push(cls, dag: DAG) -> None:
        cls._dag = dag

    @classmethod
    def pop(cls) -> None:
        cls._dag = None

    @classmethod
    def has_current(cls) -> bool:
        return cls._dag is not None


@dataclass
class NodeRef:
    """
    Base reference for a node in the DAG.

    A ``NodeRef`` represents a runnable computation unit (an operator) inside a DAG.
    Nodes are typically instantiated within a ``with DAG(...):`` context; during
    construction they automatically register themselves into the currently active
    DAG via ``__post_init__``.

    The class also implements a small DSL for wiring dependencies using the
    right-shift operator:

    - ``a >> b`` creates a default edge from node ``a`` to node ``b``.
    - ``a >> b.some_sink(...)`` delegates edge creation to an ``AttachmentSink``
      (e.g. for ControlNet / IP-Adapter style inputs).

    The runner expects each node to produce a single primary output (a dictionary).
    That output is propagated to downstream nodes via edges and assembled into the
    ``input`` mapping passed to ``run()``.

    Parameters
    ----------
    id : str
        Unique node identifier within a DAG.

    Attributes
    ----------
    op : str
        Operator identifier automatically derived from the concrete node class
        name (lowercase or snake_case).

    Notes
    -----
    This base class defines the execution contract via ``run()`` but does not
    implement any actual operator logic. Concrete nodes must subclass ``NodeRef``
    and implement ``run()``.
    """

    id: str
    op: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Registers this node into the currently active DAG context.

        This method is invoked automatically by ``dataclasses`` after initialization.

        Raises
        ------
        RuntimeError
            If no DAG context is active (i.e. the node is created outside a
            ``with DAG(...):`` block).
        """
        # operator name derived from concrete class
        self.op = self.__class__.__name__.lower()

        # register node in current DAG context
        _DagContext.current().add_node(self)

    # node >> other
    def __rshift__(self, other: Union['NodeRef', 'AttachmentSink']) -> Union['NodeRef', 'AttachmentSink']:
        """
        Wires this node to another node or to an attachment sink using ``>>``.

        There are two supported cases:

        1) ``self >> other_node``:
           Creates a *default* edge from ``self`` to ``other_node`` with
           ``input_id="default"``. The meaning of this default input is defined
           by the destination node/operator (e.g. for an Img2Img node this could
           correspond to the init image).

        2) ``self >> attachment_sink``:
           Delegates edge creation to ``AttachmentSink.__rrshift__``. This is used
           for special/typed inputs such as IP-Adapter reference images or
           ControlNet conditioning images.

        Parameters
        ----------
        other : NodeRef or AttachmentSink
            The destination of the connection.

        Returns
        -------
        NodeRef or AttachmentSink
            Returns ``other`` to enable chaining (e.g. ``a >> b >> c``) or returns
            the sink for sink-based connections.

        Notes
        -----
        This method assumes a DAG context is active.

        Raises
        ------
        RuntimeError
            If no DAG context is active.
        """
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

        return NotImplemented  # type: ignore[return-value]

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
        self.__rshift__(other)
        return self

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


@dataclass
class AttachmentSink:
    """
    A sink representing a specific input channel of a target node.

    ``AttachmentSink`` is used to model "special" inputs that are not the default
    input of a node. Typical examples include:

    - IP-Adapter reference images (potentially multiple images per adapter)
    - ControlNet conditioning images (potentially multiple control nets)
    - Any additional conditioning sources that conceptually attach to a node

    The sink is created by the target node (or a helper method on it) and can be
    wired using the ``>>`` operator:

    - ``src_node >> sink``

    This adds an edge from ``src_node`` to the sink's ``target`` node, using the
    sink's ``input_id`` to identify which input channel is being populated.

    Attributes
    ----------
    id : str
        Identifier of this sink instance. This can be used by higher-level code
        to de-duplicate or reference sinks (e.g. keyed by adapter model path).
    target : NodeRef
        The node that will receive the attached input.
    input_id : str
        Input identifier used by the runner to build the target node's input
        mapping (``input_id -> upstream output``).

    Notes
    -----
    ``AttachmentSink`` does not execute anything by itself. It only participates
    in DAG wiring.
    """

    id: str
    target: NodeRef
    input_id: str

    def __rrshift__(self, src: NodeRef) -> 'AttachmentSink':
        """
        Wires a source node to this sink using the ``>>`` operator.

        This method is invoked when Python evaluates ``src >> sink`` and the sink
        is on the right-hand side. It registers an edge in the current DAG from
        ``src.id`` to ``target.id`` using ``input_id`` to specify the input channel.

        Parameters
        ----------
        src : NodeRef
            The upstream node providing the attached input.

        Returns
        -------
        AttachmentSink
            Returns ``self`` to allow fluent chaining if desired.

        Raises
        ------
        RuntimeError
            If no DAG context is active.
        """
        dag = _DagContext.current()
        dag.add_edge(Edge(
            node_from=src.id,
            node_to=self.target.id,
            input_id=self.input_id
        ))

        return self
