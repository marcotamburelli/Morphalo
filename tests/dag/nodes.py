from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from morphalo.dag import AttachmentSink, NodeRef


@dataclass
class TrackedNode(NodeRef):
    """
    Base test node that tracks executions.

    This class is designed exclusively for unit tests targeting the DAG
    execution engine (e.g. ``_execute``). It captures call count, last received
    input map, and a history of outputs produced.

    Attributes
    ----------
    calls : int
        Number of times ``run`` has been invoked.
    last_input : dict[str, dict] or None
        The last input mapping received by ``run``.
    outputs : list[dict]
        All outputs produced by ``run``, in call order.

    Notes
    -----
    - The runner passes ``input`` as a mapping: ``input_id -> upstream_output``.
    - For entry/source nodes, ``input`` may be ``None`` or an empty dict,
      depending on runner policy.
    """

    calls: int = field(default=0, init=False)
    last_input: Optional[Dict[str, Dict[str, Any]]
                         ] = field(default=None, init=False)
    outputs: list[Dict[str, Any]] = field(default_factory=list, init=False)

    def _track(
        self,
        input: Optional[Dict[str, Dict[str, Any]]],
        out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Track the call and store output.

        Parameters
        ----------
        input : dict[str, dict] or None
            Input map received by ``run``.
        out : dict
            Output produced by ``run``.

        Returns
        -------
        dict
            The same output dict (for convenience).
        """
        self.calls += 1
        self.last_input = input
        self.outputs.append(out)
        return out


@dataclass
class SourceNode(TrackedNode):
    """
    Source node: returns a constant payload and ignores inputs.
    """

    value: Any = None

    def run(self, output_dir, input: Dict[str, Dict] = None) -> Dict[str, Any]:
        out = {'value': self.value, 'from': self.id}
        return self._track(input, out)


@dataclass
class PassNode(TrackedNode):
    """
    Pass-through node: takes the first input and re-emits its ``value``.

    The node also includes ``via`` to record which input_id was consumed.
    This is useful to ensure the runner is wiring input_ids correctly.
    """

    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        k, v = next(iter(input.items()))
        out = {'value': v['value'], 'from': self.id, 'via': k}
        return self._track(input, out)


@dataclass
class MergeNode(TrackedNode):
    """
    Merge node: collects all inputs and returns a dict keyed by input_id.

    Output format
    -------------
    {
        'merged': {input_id: value, ...},
        'from': '<node_id>'
    }
    """

    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        merged = {k: v.get('value') for k, v in input.items()}
        out = {'merged': merged, 'from': self.id}
        return self._track(input, out)

    def sink(self, *, name: str, input_id: str) -> AttachmentSink:
        """
        Create an AttachmentSink for this node.

        Parameters
        ----------
        name : str
            Logical sink name (debug identifier).
        input_id : str
            Input channel id used by the runner when building the input mapping.

        Returns
        -------
        AttachmentSink
            A sink that can be wired via ``src >> merge.sink(...)``.
        """
        return AttachmentSink(name=name, target=self, input_id=input_id)


@dataclass
class NoneNode(TrackedNode):
    """
    Node that returns None to trigger DagValidationError inside ``_execute``.
    """

    def run(self, output_dir, input: Dict[str, Dict] = None):
        # Intentionally do not record an output (there is none)
        self.calls += 1
        self.last_input = input
        return None


@dataclass
class ExtractNode(TrackedNode):
    """
    Adapter node for tests: converts a MergeNode-like payload into a 'value' payload.

    This is useful to keep runner tests focused on scheduling/propagation instead
    of failing due to mismatched mock output schemas.

    Expected upstream format
    ------------------------
    input['default'] contains a dict with key 'merged', where 'merged' is a dict.

    Output format
    -------------
    {'value': <selected>, 'from': self.id}
    """

    key: str = 'default'

    def run(self, output_dir, input: Dict[str, Dict]) -> Dict[str, Any]:
        upstream = input['default']
        merged = upstream['merged']
        out = {'value': merged[self.key], 'from': self.id}
        return self._track(input, out)
