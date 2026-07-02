"""The ``linear_probe`` regime: frozen backbone + a lightweight fitted head.

``linear_probe`` freezes the backbone and trains only a small head on the features it
produces via :meth:`Model.embed` (never :meth:`Model.forward`). Concretely the adapter:

1. streams the ``train`` split, embedding each batch with ``model.embed`` (backbone
   frozen -- we never call ``forward`` and never touch its weights);
2. fits an injectable :class:`Head` on ``(embedding, label)`` pairs;
3. at predict time embeds the eval batch and asks the head for class predictions.

The **default head is a pure-stdlib nearest-class-centroid classifier**
(:class:`NearestCentroidHead`): the per-class mean embedding is computed on ``train`` and
a sample is assigned the class of the nearest centroid (squared Euclidean). It is
deterministic and dependency-free, which is exactly why it is the placeholder head -- real
linear / logistic-regression heads (numpy / torch / scikit-learn) arrive in M3/M6 behind
optional extras, and plug in through the same :class:`Head` protocol via the ``head``
argument of :class:`LinearProbeAdapter`.

Pure stdlib -- no ``torch``/``numpy``/``sklearn`` import.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.types import Batch, Tensor
from rfbench.regimes.base import FittedState, RegimeAdapter, TrainSplit

#: A single embedding vector as the pure-stdlib head sees it: a sequence of floats.
EmbeddingVector = Sequence[float]

#: Mini-batch size for streaming embeddings during probe fit (keeps a foundation backbone's
#: activations within GPU memory when the train split is large, e.g. RadioML's 176k samples).
_EMBED_BATCH = 256


@runtime_checkable
class Head(Protocol):
    """A fittable classification head over frozen embeddings.

    Injected into :class:`LinearProbeAdapter` so the placeholder centroid head can be
    swapped for a real linear / logreg head (numpy/torch, M3/M6) without touching the
    adapter. Implementations MUST be deterministic given deterministic inputs.
    """

    def fit(self, embeddings: Sequence[EmbeddingVector], labels: Sequence[int]) -> None:
        """Fit the head on paired ``(embedding, label)`` training data."""
        ...

    def predict(self, embeddings: Sequence[EmbeddingVector]) -> list[int]:
        """Predict a class id per embedding (order-preserving)."""
        ...


class NearestCentroidHead:
    """Pure-stdlib nearest-class-centroid classifier (the placeholder probe head).

    Stores one centroid per class -- the coordinate-wise mean of that class's training
    embeddings -- and predicts the class whose centroid is closest under squared Euclidean
    distance. Ties are broken by the smaller class id (via a stable ``min`` over sorted
    centroids), keeping predictions deterministic. No numpy: sums and squared distances are
    plain Python over floats.
    """

    def __init__(self) -> None:
        #: class id -> centroid vector, populated by :meth:`fit`. Sorted-key iteration at
        #: predict time gives deterministic tie-breaking (smallest class id wins).
        self._centroids: dict[int, tuple[float, ...]] = {}

    def fit(self, embeddings: Sequence[EmbeddingVector], labels: Sequence[int]) -> None:
        """Compute per-class mean embeddings from the training pairs."""
        if len(embeddings) != len(labels):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and labels ({len(labels)}) length mismatch"
            )
        if not embeddings:
            raise ValueError("cannot fit NearestCentroidHead on an empty training set")

        sums: dict[int, list[float]] = {}
        counts: dict[int, int] = {}
        dim: int | None = None
        for vector, label in zip(embeddings, labels, strict=True):
            row = [float(v) for v in vector]
            if dim is None:
                dim = len(row)
            elif len(row) != dim:
                raise ValueError(
                    f"inconsistent embedding dim: expected {dim}, got {len(row)} for a sample"
                )
            key = int(label)
            if key not in sums:
                sums[key] = [0.0] * len(row)
                counts[key] = 0
            acc = sums[key]
            for i, value in enumerate(row):
                acc[i] += value
            counts[key] += 1

        self._centroids = {
            label: tuple(total / counts[label] for total in acc) for label, acc in sums.items()
        }

    def predict(self, embeddings: Sequence[EmbeddingVector]) -> list[int]:
        """Assign each embedding the class of its nearest centroid."""
        if not self._centroids:
            raise RuntimeError("NearestCentroidHead.predict called before fit")
        ordered = sorted(self._centroids.items())  # stable, class-id ascending
        preds: list[int] = []
        for vector in embeddings:
            row = [float(v) for v in vector]
            best_label = min(ordered, key=lambda item: _sq_distance(row, item[1]))[0]
            preds.append(best_label)
        return preds


def _sq_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Squared Euclidean distance between two equal-length vectors (pure Python)."""
    return sum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


class LinearProbeAdapter(RegimeAdapter):
    """The ``linear_probe`` regime: freeze the backbone, fit a head on ``embed`` features.

    The head is injectable (default :class:`NearestCentroidHead`, the pure-stdlib
    placeholder) so a real linear/logreg head can be dropped in behind extras (M3/M6)
    without changing the adapter. This concrete adapter accepts only the ``linear_probe``
    regime (a ``few_shot`` spec is refused); the ``few_shot`` regime subclasses this adapter
    and adds the ``k``-per-class subsampling on top.
    """

    #: The regime this concrete adapter is allowed to carry. The base is ``linear_probe``;
    #: :class:`~rfbench.regimes.few_shot.FewShotAdapter` overrides it to ``few_shot``. A
    #: spec whose name differs is rejected, so a ``few_shot`` spec can never be handed to a
    #: plain ``LinearProbeAdapter`` (or vice-versa) and mislabel the row.
    _expected_regime: Regime = Regime.LINEAR_PROBE

    def __init__(
        self,
        head: Head | None = None,
        *,
        label_field: str = "label",
        regime: RegimeSpec | None = None,
    ) -> None:
        resolved = regime if regime is not None else RegimeSpec(self._expected_regime)
        if resolved.name is not self._expected_regime:
            raise ValueError(
                f"{type(self).__name__} only accepts the '{self._expected_regime.value}' "
                f"regime, got '{resolved.name.value}'"
            )
        self.regime = resolved
        #: The frozen-backbone head; defaults to the pure-stdlib centroid placeholder.
        self._head: Head = head if head is not None else NearestCentroidHead()
        #: Batch field holding the supervision label used to fit the head.
        self._label_field = label_field

    def fit(self, model: Model, train_split: TrainSplit) -> FittedState:
        """Freeze the backbone and fit the head on per-sample ``model.embed`` features.

        The backbone is used only through :meth:`Model.embed` (never ``forward``), so its
        weights are untouched -- this is what "frozen backbone" means at the harness level.
        Subclasses (few-shot) override :meth:`_select_train_samples` to reduce the split
        before fitting; the base uses every sample.
        """
        samples = list(self._select_train_samples(train_split))
        if not samples:
            raise ValueError(f"{self.regime.name.value}: empty train split, nothing to fit")

        labels = [int(sample[self._label_field]) for sample in samples]
        embeddings = self._embed_samples(model, samples)
        self._head.fit(embeddings, labels)

        return FittedState(
            head=self._head,
            info={
                "regime": self.regime.name.value,
                "n_train_samples": len(samples),
                "n_classes": len(set(labels)),
            },
        )

    def predict(self, model: Model, inputs: Batch, state: FittedState) -> Tensor:
        """Embed ``inputs`` with the frozen backbone and classify with the fitted head."""
        head = state.head
        if head is None:
            raise RuntimeError(
                f"{self.regime.name.value}.predict called before fit (no head in state)"
            )
        embeddings = _as_vectors(model.embed(inputs))
        return head.predict(embeddings)

    # -- overridable selection hook -----------------------------------------------------
    def _select_train_samples(self, train_split: TrainSplit) -> Iterable[Batch]:
        """Return the training samples to fit on. Base: the full split (few-shot subsets)."""
        return train_split

    def _embed_samples(self, model: Model, samples: Sequence[Batch]) -> list[EmbeddingVector]:
        """Collate ``samples`` and embed them in MINI-BATCHES, one vector per sample.

        ``model.embed`` is called on each collated chunk (same field-agnostic contract as
        ``forward``); outputs are normalised to per-sample float vectors so any head sees a
        uniform ``Sequence[EmbeddingVector]``. Chunking is essential for real data: embedding
        a whole train split (e.g. 176k RadioML samples) through a foundation backbone in one
        call OOMs the GPU -- we stream it in ``_EMBED_BATCH``-sized chunks instead.
        """
        vectors: list[EmbeddingVector] = []
        for start in range(0, len(samples), _EMBED_BATCH):
            chunk = samples[start : start + _EMBED_BATCH]
            chunk_vectors = _as_vectors(model.embed(_collate(chunk)))
            if len(chunk_vectors) != len(chunk):
                raise ValueError(
                    f"model.embed returned {len(chunk_vectors)} vectors for {len(chunk)} samples"
                )
            vectors.extend(chunk_vectors)
        return vectors


# --- collation / embedding normalisation (mirrors evaluate._collate, kept local) -------
def _collate(samples: Sequence[Batch]) -> Batch:
    """Collate per-sample dicts into a dict of field -> list (no torch collate)."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {key: [sample[key] for sample in samples] for key in keys}


def _as_vectors(embedded: Tensor) -> list[EmbeddingVector]:
    """Normalise an ``embed`` output into a list of per-sample float vectors.

    Accepts the pure-Python shapes the harness produces without a tensor framework: a list
    of per-sample vectors, or a single vector treated as one sample. Each element is coerced
    to a ``list[float]`` so the stdlib head never depends on the concrete container type.
    """
    if isinstance(embedded, Mapping):
        raise TypeError("model.embed must return per-sample vectors, not a mapping")
    as_list = list(embedded)
    if as_list and not isinstance(as_list[0], (list, tuple)):
        # A single flat vector -> one sample.
        return [[float(v) for v in as_list]]
    return [[float(v) for v in row] for row in as_list]


__all__ = ["Head", "NearestCentroidHead", "LinearProbeAdapter", "EmbeddingVector"]
