from typing import Any, Dict, Optional

from morphalo.dag.core import AttachmentSink, NodeRef


class ImgWiringMixin:
    """
    Mixin that adds dynamic image input wiring to a node.

    After initialization, the node exposes an ``image`` registry that allows
    attaching multiple candidate images via repeated calls.

    Example
    -------
    >>> scorer = PersonScorer(...)
    >>> node_a >> scorer.image()
    >>> node_b >> scorer.image()

    Each call to ``image()`` registers a new input slot (``image:{idx}``).
    """

    def __post_init__(self):
        super().__post_init__()
        self.image = ImgRegistry(owner=self)


class ImgRegistry:
    """
    Registry for dynamically declaring image input slots.

    Each call to the registry creates a new ``AttachmentSink`` associated
    with a unique ``image:{idx}`` input id on the owning node.

    This allows nodes to accept a variable number of candidate images
    without predefining input ports.

    Notes
    -----
    - The order of calls defines the expected input indices.
    - The number of expected images is given by ``img_count``.
    """

    def __init__(self, owner: NodeRef):
        self._owner = owner
        self._idx = 0

    def __call__(self) -> AttachmentSink:
        """
        Create a new candidate-image input sink.

        Each call registers one additional expected input image and returns an
        ``AttachmentSink`` wired to a unique attachment id of the form
        ``image:{idx}``.

        Returns
        -------
        AttachmentSink
            Sink to be connected to an upstream node producing an image path.

        Notes
        -----
        Call this method once for each candidate image you want the scorer to
        evaluate. The total number of expected candidates is determined by the
        number of times this method is called.
        """
        idx = self._idx
        self._idx += 1

        return AttachmentSink(
            name=f'score:{self._owner.id}:{idx}',
            target=self._owner,
            input_id=f'image:{idx}',
        )

    @property
    def img_count(self) -> int:
        return self._idx


class ImgBundle:
    """
    Helper for collecting candidate images and propagating ranking state.
    The bundle accepts both singular and plural image payload conventions
    (``image`` / ``images``), as well as the generic ``path`` field. All inputs
    are normalized to a flat list of unique image paths, preserving order.

    ``ImgBundle`` centralizes the logic for:

    - extracting image paths from node inputs
    - merging candidates coming from explicit wiring and upstream scorers
    - carrying forward accumulated ranking contributions across nodes

    It acts as the bridge between:

    - the DAG input layer (wiring, default inputs)
    - the scoring layer (per-node evaluation)
    - the ranking layer (cross-node score aggregation)

    The resulting bundle provides:
    - a normalized list of candidate image paths
    - an existing ranking structure (if any) to be extended by the current node
    """

    def __init__(
        self,
        *,
        node_id: str,
        img_count: int,
        input: Optional[Dict[str, Dict]] = None
    ):
        """
        Build the candidate image set and inherit upstream ranking data.

        The initialization collects candidate image paths from multiple sources,
        in the following order:

        1. Explicit wiring (``image:{idx}``)
            For each declared image input, the corresponding upstream node must
            provide one of:

                - ``'image'`` : path to a single image
                - ``'images'`` : list of image paths
                - ``'path'`` : path to a single image or list of paths

            Missing inputs or invalid payloads raise an error.

        2. Upstream rankings (scorer chaining)
            If the default input contains a ``rankings`` structure, all images
            referenced in it are added as candidates. This enables chaining of
            scorer nodes without re-wiring images explicitly.

        3. Fallback to 'default' input
            If no explicit images are provided, the bundle inspects the ``default``
            input payload.

            The following keys are supported (in priority order):

            - ``'rankings'`` : dict
                If present, image paths are extracted from ranking outputs produced
                by scorer nodes.

            - ``'image'`` : str
                Path to a single image.

            - ``'images'`` : list[str]
                List of image paths.

            - ``'path'`` : str or list[str]
                Path or list of paths to image files.

            The first available key among these is used. If none are found, an error
            is raised.

            Values may be either a single path or a sequence of paths. All values are
            normalized to a flat list of paths.

        The final candidate list is the concatenation of all collected sources.

        Parameters
        ----------
        node_id : str
            Identifier of the current node (used for error reporting and as key
            in the ranking structure).

        img_count : int
            Number of explicitly declared image inputs (via ``image()`` calls).

        input : dict, optional
            Upstream input dictionary provided by the DAG runtime.

        Raises
        ------
        ValueError
            If:
            - a declared image input is missing
            - an upstream payload does not contain a valid image path
            - no candidate images can be collected
        """
        if input is None:
            input = {}

        self._node_id = node_id
        self._paths: list[str] = []

        # extracting wired images as potential source
        for idx in range(img_count):
            up = input.get(f'image:{idx}')

            if up is None:
                raise ValueError(
                    f"{node_id}: missing upstream for image idx={idx} "
                    f"(expected wiring into input_id='image:{idx}')"
                )

            path = up.get('image')\
                or up.get('images')\
                or up.get('path')
            if not path:
                raise ValueError(
                    f"{node_id}: upstream for idx={idx} must contain one or more image paths"
                )

            if isinstance(path, list):
                self._paths += [str(p) for p in path]
            else:
                self._paths.append(str(path))

        init_up = input.get('default', {})

        # Checking images from ranking of previous scoring
        rankings: dict = init_up.get('rankings', None)
        init_path = init_up.get('image') \
            or init_up.get('images')\
            or init_up.get('path')

        if rankings is not None:
            # If a ranking is found from a previous scoring node
            # images are collected from ranking
            for info_list in rankings.values():
                for info in info_list:
                    self._paths.append(str(info['image']))

        elif init_path is not None:
            # Otherwise image or path input fields are checked as alternative source.
            if not isinstance(init_path, list):
                init_path = [init_path]

            self._paths += [str(p) for p in init_path]

        if not self._paths:
            raise ValueError(
                f"{node_id}: no upstream image found."
            )

        seen = set()
        unique_paths = []
        for p in self._paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        self._paths = unique_paths

        self._rankings = rankings or {}

    def build_rankings(
        self,
        *,
        candidates: list[dict[str, Any]],
        score_weight: float
    ) -> Dict[str, list[Any]]:
        ranking = sorted(
            candidates,
            key=lambda x: float(x['score']),
            reverse=True,
        )

        if not ranking:
            raise RuntimeError(f"{self._node_id}: no candidates were scored.")

        return {
            self._node_id: [
                {
                    'image': item['input_image'],
                    'score': float(item['score']),
                    'flags': item.get('flags', []),
                    'score_weight': score_weight,
                    **({} if 'error' not in item else {'error': item['error']}),
                }
                for item in ranking
            ],
            **self._rankings,
        }

    @property
    def paths(self) -> list[str]:
        return self._paths

    @staticmethod
    def calculate_best_image(rankings: Dict[str, list[Any]]) -> tuple[None | str, float]:
        """
        Select the best image based on aggregated weighted scores.

        The input ``rankings`` is a mapping from node id to a list of score
        contributions. Each contribution provides:

            - image
            - score
            - score_weight

        For each image, scores are combined as a weighted mean across all
        contributing nodes.

        Returns
        -------
        tuple[str or None, float]
            Path of the best image and its aggregated score.
            Returns (None, -1.0) if no valid candidates are found.
        """
        weights_sum: dict[str, float] = {}
        scores_wsum: dict[str, float] = {}

        for info_list in rankings.values():
            for info in info_list:
                image = str(info['image'])
                score = float(info['score'])
                weight = float(info['score_weight'])

                weights_sum[image] = weights_sum.get(image, 0) + weight
                scores_wsum[image] = scores_wsum.get(image, 0) + score * weight

        best_image: Optional[str] = None
        best_score: float = -1.0

        for image in weights_sum.keys():
            score = scores_wsum[image] / weights_sum[image]

            if score > best_score:
                best_image = image
                best_score = score

        return best_image, best_score
