import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

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

            payload = upstream.get('image') or \
                upstream.get('images') or \
                upstream.get('path')
            if not payload:
                raise ValueError(
                    f"Upstream output for '{in_id}' does not contain 'image', 'images' (or 'path')."
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


def _payload_image_paths(payload: Dict[str, Any], *, input_id: str) -> List[str]:
    """
    Resolve one or more image paths from a Morphalo upstream payload.
    """
    value = payload.get('images') or payload.get(
        'image') or payload.get('path')
    if not value:
        raise ValueError(
            f"Upstream output for {input_id!r} must contain "
            "'image', 'images', or 'path'."
        )

    if isinstance(value, (str, Path)):
        return [str(value)]

    if isinstance(value, list):
        if not value:
            raise ValueError(f'Upstream image list for {input_id!r} is empty.')
        return [str(x) for x in value]

    raise TypeError(
        f'Unsupported image path payload for {input_id!r}: {type(value).__name__}'
    )


class ImageSequenceRegistry:
    """
    Registry for declaring indexed image inputs on a DAG node.

    If present, the node's ``default`` input is always the first part of the
    runtime image sequence. Additional images are declared here using integer
    indexes and are appended after the default image(s), ordered from the
    smallest index to the largest. If no default input is wired, the ordered
    registry images form the whole image sequence. If neither default nor
    indexed images are provided, the resolved sequence is empty; nodes decide
    whether that is valid for their model semantics.
    """

    INPUT_PREFIX = 'image'

    def __init__(self, owner: NodeRef):
        self._owner = owner
        self._idxs: List[int] = []

    def __call__(self, idx: int) -> AttachmentSink:
        """
        Shorthand for ``add(idx=idx)``.
        """
        return self.add(idx)

    def add(self, idx: int) -> AttachmentSink:
        """
        Declare an indexed image input and return its attachment sink.

        ``idx`` is an integer ordering index, not a semantic label. At runtime,
        the owning node receives a single ordered image sequence:

        1. all image(s) from the optional ``default`` input, in their payload
           order;
        2. all images wired through this registry, sorted by ascending ``idx``.

        This ordering is important because the model is prompted by referring to
        positions in the image list, for example "the first image", "the second
        image", and so on. Use smaller indexes for images that should appear
        earlier after the optional default image.

        Parameters
        ----------
        idx : int
            Ordering index for this image slot. Indexes must be unique within
            the registry. The numeric value determines the slot order among
            registry-declared images.

        Returns
        -------
        AttachmentSink
            Sink that can be wired from an upstream image-producing node.
        """
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError(
                'Image sequence idx must be an integer.'
            )
        if idx in self._idxs:
            raise ValueError(f'Duplicate image sequence idx: {idx!r}')

        self._idxs.append(idx)
        return AttachmentSink(
            name=f'image_sequence:image:{idx}',
            target=self._owner,
            input_id=f'{self.INPUT_PREFIX}:{idx}',
        )

    @property
    def specs(self) -> List[int]:
        return self._idxs


class ImageSequenceBundle:
    """
    Runtime resolver for indexed image sequence inputs.

    The bundle intentionally permits an empty sequence. Some foundation models
    can run either as text-to-image or as image-conditioned editors. Nodes that
    require at least one image should validate ``bundle.images`` themselves.
    """

    def __init__(
        self,
        specs: List[int],
        *,
        input: Optional[Dict[str, Dict]] = None,
    ):
        self._images: List[Image.Image] = []
        self._metadata: List[Dict[str, Any]] = []
        self._build(specs, input=input or {})

    def _build(
        self,
        specs: List[int],
        *,
        input: Dict[str, Dict],
    ) -> None:
        default_up = input.get('default')

        entries: List[Tuple[str, str]] = []
        if default_up is not None:
            for path in _payload_image_paths(default_up, input_id='default'):
                entries.append(('default', path))

        for idx in sorted(specs):
            input_id = f'{ImageSequenceRegistry.INPUT_PREFIX}:{idx}'
            upstream = input.get(input_id)
            if upstream is None:
                raise ValueError(
                    f'Missing image sequence input for {input_id!r}. '
                    f'Did you wire an image into node.image.add(idx={idx!r})?'
                )
            for path in _payload_image_paths(upstream, input_id=input_id):
                entries.append((input_id, path))

        self._images = [Image.open(path).convert('RGB') for _, path in entries]
        self._metadata = [
            {
                'input_id': input_id,
                'path': str(Path(path).expanduser()),
                'width': image.size[0],
                'height': image.size[1],
            }
            for (input_id, path), image in zip(entries, self._images)
        ]

    @property
    def images(self) -> List[Image.Image]:
        return self._images

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self._metadata


class ImageSequenceMixin:
    """
    Adds ordered image-sequence wiring helpers to a node.
    """

    image: ImageSequenceRegistry

    def __post_init__(self) -> None:
        super().__post_init__()
        self.image = ImageSequenceRegistry(owner=self)

    def build_image_sequence_bundle(
        self,
        input: Optional[Dict[str, Dict]],
    ) -> ImageSequenceBundle:
        return ImageSequenceBundle(
            self.image.specs,
            input=input or {},
        )
