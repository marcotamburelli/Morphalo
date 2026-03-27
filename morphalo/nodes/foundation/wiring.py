import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from morphalo.dag import AttachmentSink, NodeRef


# -----------------------------
# Declaration-time spec
# -----------------------------
@dataclass(frozen=True)
class OmniGenImageSpec:
    key: str


# -----------------------------
# Registry (DAG construction time)
# -----------------------------
class OmniGenImageRegistry:
    """
    Registry for declaring and wiring image inputs for an OmniGen node.

    This class is responsible for *declaring* named image slots that can be
    populated by upstream DAG nodes via ``AttachmentSink`` connections.
    Each slot is identified by a stable string ``key`` and corresponds to
    exactly one DAG edge targeting the owning node.

    The registry operates at **DAG construction time** only. It does not
    resolve image paths or load images. Runtime resolution is delegated to
    :class:`OmniGenImageBundle`.

    Image slots are wired using the ``>>`` operator, similarly to ControlNet
    wiring::

        refs >> omnigen.image.add(key="ref")
        style >> omnigen.image.add(key="style")

    Each declared slot produces an ``AttachmentSink`` with an ``input_id``
    following the convention::

        omnigen:image:<key>

    Notes
    -----
    - Due to DAG validation rules, each ``input_id`` must be unique per
      destination node. As a consequence, at most **one upstream node**
      may be wired for each ``key``.
    - The upstream node may emit either a single image or a list of images.
      In both cases, the slot is treated as a *group* of images under the
      same key.
    - If multiple independent sources are required for the same conceptual
      role, distinct keys must be used (e.g. ``ref_a``, ``ref_b``), or the
      ``input_id`` scheme must be extended with explicit indexing.

    Parameters
    ----------
    owner : NodeRef
        The DAG node that owns this registry and will receive the image
        inputs at execution time.
    """

    def __init__(self, owner: NodeRef):
        self._owner = owner
        self._counter = 0
        self._specs: List[OmniGenImageSpec] = []

    def add(self, *, key: Optional[str] = None) -> AttachmentSink:
        """
        Declare a new OmniGen image slot and return an attachment sink.

        This method registers a named image slot associated with the owning
        node and returns an ``AttachmentSink`` that can be used to wire an
        upstream image-producing node into this slot.

        The returned sink is intended to be used with the ``>>`` operator::

            image_node >> omnigen.image.add(key="ref")

        If no ``key`` is provided, a unique key is automatically generated
        using an incremental counter (``img1``, ``img2``, ...). Automatically
        generated keys are deterministic within a single registry instance
        and preserve insertion order.

        Parameters
        ----------
        key : str, optional
            Stable identifier for the image slot. This key is later used
            to reference the corresponding images inside the prompt via
            ``{{img:<key>}}`` or ``{{img:<key>[idx]}}``.

        Returns
        -------
        AttachmentSink
            An attachment sink bound to the owning node and configured with
            an ``input_id`` of the form ``omnigen:image:<key>``. Wiring an
            upstream node into this sink registers the corresponding DAG edge.

        Notes
        -----
        - Each key may be declared only once per node. Declaring multiple
          slots with the same key will lead to ambiguous wiring and should
          be avoided.
        - Due to DAG validation constraints, only one upstream edge may
          target a given key. However, that upstream edge may carry multiple
          images as a list.
        """
        if key is None:
            self._counter += 1
            key = f'img{self._counter}'

        self._specs.append(OmniGenImageSpec(key=key))

        return AttachmentSink(
            name=f'omnigen:image:{key}',
            target=self._owner,
            input_id=f'omnigen:image:{key}',
        )

    @property
    def specs(self) -> List[OmniGenImageSpec]:
        return self._specs


# -----------------------------
# Runtime bundle
# -----------------------------
@dataclass(frozen=True)
class KeySpan:
    """
    A contiguous span within flat_images:
    - start: 0-based index into flat_images
    - length: number of images for this key
    """
    start: int
    length: int


class OmniGenImageBundle:
    """
    Resolves OmniGen image inputs and provides:
      - flat_paths: List[str]
      - images_by_key: Dict[str, List[str]]
      - span_by_key: Dict[str, KeySpan]
      - token resolution for {{img:key}} and {{img:key[idx]}}

    Upstream contract (from wired nodes):
      upstream['image'] can be a string path OR a list of paths.
      upstream['path'] is accepted as fallback.
    """

    IMG_PLACEHOLDER_FMT = '<img><|image_{n}|></img>'  # n is 1-based

    def __init__(
        self,
        specs: List[OmniGenImageSpec],
        *,
        input: Optional[Dict[str, Dict]] = None,
    ):
        self._flat_paths: List[str] = []
        self._images_by_key: Dict[str, List[str]] = {}
        self._span_by_key: Dict[str, KeySpan] = {}

        if specs:
            self._build(specs, input=input or {})

    def _build(self, specs: List[OmniGenImageSpec], *, input: Dict[str, Dict]) -> None:
        cursor = 0

        for spec in specs:
            in_id = f'omnigen:image:{spec.key}'
            upstream = input.get(in_id)
            if upstream is None:
                raise ValueError(
                    f"Missing OmniGen image input for '{in_id}'. Did you wire an image node into it?"
                )

            payload = upstream.get('image') or upstream.get('path')
            if not payload:
                raise ValueError(
                    f"Upstream output for '{in_id}' does not contain 'image' (or 'path')."
                )

            if isinstance(payload, list):
                paths = [str(p) for p in payload]
            else:
                paths = [str(payload)]

            if len(paths) == 0:
                raise ValueError(f"Empty image list for '{in_id}'.")

            self._images_by_key[spec.key] = paths
            self._span_by_key[spec.key] = KeySpan(
                start=cursor,
                length=len(paths)
            )

            self._flat_paths.extend(paths)
            cursor += len(paths)

    # ---- public views ----
    @property
    def flat_paths(self) -> List[str]:
        return self._flat_paths

    @property
    def images_by_key(self) -> Dict[str, List[str]]:
        return self._images_by_key

    @property
    def span_by_key(self) -> Dict[str, KeySpan]:
        return self._span_by_key

    # ---- token helpers ----
    def token_for(self, key: str, idx: Optional[int] = None) -> str:
        """
        Returns the OmniGen prompt placeholder token for:
          - {{img:key}}       -> first image for that key
          - {{img:key[idx]}}  -> idx-th image for that key (0-based idx)

        Raises:
          KeyError / IndexError with good messages.
        """
        if key not in self._span_by_key:
            known = ', '.join(sorted(self._span_by_key.keys()))
            raise KeyError(f"Unknown img key '{key}'. Known keys: [{known}]")

        span = self._span_by_key[key]

        if idx is None:
            idx = 0

        if idx < 0 or idx >= span.length:
            raise IndexError(
                f"img key '{key}' index {idx} out of range (0..{span.length-1})."
            )

        # convert to 1-based for OmniGen token numbering
        n = span.start + idx + 1
        return self.IMG_PLACEHOLDER_FMT.format(n=n)

    def render_prompt(self, prompt: str) -> str:
        """
        Replaces:
        {{img:key}} or {{img:key[idx]}}
        with:
        <img><|image_N|></img>

        `idx` is interpreted as 0-based.

        Raises:
            ValueError if images are wired but none are referenced in the prompt.
        """
        # {{img:foo}} or {{img:foo[2]}}
        pattern = re.compile(r'\{\{img:([a-zA-Z0-9_\-]+)(?:\[(\d+)\])?\}\}')

        used = False

        def _repl(m: re.Match) -> str:
            nonlocal used
            used = True
            key = m.group(1)
            idx_s = m.group(2)
            idx = int(idx_s) if idx_s is not None else None
            return self.token_for(key, idx)

        rendered = pattern.sub(_repl, prompt or "")

        # Images were provided but never referenced
        if self._flat_paths and not used:
            raise ValueError(
                "OmniGen images are wired, but the prompt does not reference them. "
                "Use {{img:<key>}} or {{img:<key>[idx]}} in the prompt."
            )

        return rendered
