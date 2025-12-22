from typing import Dict, List, Optional

from stability.dag import *
from stability.dag.validation import validate_dag


@dataclass
class Execution:
    edge: Edge
    output: Optional[Dict] = None


class DAGRunner:
    """
    Executes a DAG by resolving node dependencies and propagating outputs
    along edges until all nodes have been evaluated.

    The runner operates on a static DAG definition (nodes + edges) and
    performs the following steps:

    1. Validates the DAG structure (acyclic, valid edges).
    2. Tracks edge executions, where each edge carries the output produced
       by its source node.
    3. Executes nodes only when all their incoming edges have produced outputs.
    4. Propagates each node's output to all downstream edges.

    Execution model assumptions:
    - Each node produces a single primary output (a dict).
    - Outputs are immutable once produced.
    - Input for a node is built as a mapping:
          input_id -> upstream node output
    - Nodes with no incoming edges are considered source nodes and are
      executed with an empty input dictionary.

    The runner is deterministic and single-threaded.
    Parallelism, retries, and partial re-execution are intentionally
    out of scope.
    """

    _executions: List[Execution] = []

    def __init__(self, dag: DAG):
        """
        Initializes a DAGRunner for a given DAG.

        This constructor does not execute the DAG. It only prepares
        the internal execution state by creating one execution record
        for each edge in the DAG.

        Each execution record tracks:
        - the edge itself
        - the output produced by the source node of that edge (initially None)

        Parameters
        ----------
        dag : DAG
            The directed acyclic graph to be executed. The DAG is assumed
            to be immutable during execution.
        """

        self.__dag = dag
        self._executions = [Execution(edge=edge)
                            for edge in self.__dag.edges]

    def run(self):
        """
        Executes the DAG.

        Nodes are executed in dependency order: a node is executed only
        when all its incoming edges have produced an output. Source nodes
        (nodes with no incoming edges) are executed first with an empty
        input dictionary.

        For each executed node:
        - `node.run(output_dir, input=...)` is called.
        - The returned output is propagated to all outgoing edges.
        - Downstream nodes may become eligible for execution.

        If the execution reaches a state where no remaining node can be
        executed (i.e. missing inputs or cyclic dependencies), the method
        raises a RuntimeError.

        Raises
        ------
        RuntimeError
            If the DAG contains a cycle, unresolved dependencies, or if
            execution cannot make further progress.

        Notes
        -----
        - This method assumes the DAG has already been validated.
        - The runner does not catch exceptions raised by node execution;
          failures propagate immediately.
        - Execution is strictly sequential and deterministic.

        Returns
        -------
        None
            The runner does not return a value. Side effects are produced
            by node execution and propagated through the DAG.
        """

        # Validating the DAG
        validate_dag(self.__dag)

        # Detect initial nodes
        node_to_execute = {
            node.id: node for node in self.__dag.nodes if
            all(edge.node_to != node.id for edge in self.__dag.edges)
        }

        while node_to_execute:
            pending: Dict[str, NodeRef] = {}
            next_to_execute: Dict[str, NodeRef] = {}

            for node in node_to_execute.values():
                source_executions = [
                    execution for execution in self._executions if execution.edge.node_to == node.id]

                if any(e.output is None for e in source_executions):
                    pending = {**pending, node.id: node}
                else:
                    input = {
                        execution.edge.input_id: execution.output for execution in source_executions
                    }
                    out = node.run(self.__dag.out_dir, input=input)

                    for execution in self._executions:
                        if execution.edge.node_from == node.id:
                            execution.output = out
                            next_to_execute = {
                                **next_to_execute,
                                **{n.id: n for n in self.__dag.nodes if n.id == execution.edge.node_to}
                            }

            node_to_execute = {**pending, **next_to_execute}
