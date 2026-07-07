"""The normative ``linear_probe`` / ``few_shot`` head: scikit-learn logistic regression.

:class:`LogisticRegressionHead` implements the :class:`~rfbench.regimes.probe.Head`
protocol (``fit``/``predict``) with ``sklearn.linear_model.LogisticRegression`` and is the
head the real board runs (as opposed to :class:`~rfbench.regimes.probe.NearestCentroidHead`,
the pure-stdlib placeholder). It lives in its own module -- not in
:mod:`rfbench.regimes.probe` -- so that module keeps its documented "no
torch/numpy/sklearn import" contract; ``sklearn`` is imported lazily inside :meth:`__init__`
(never at module top), mirroring the ``torch`` lazy-import pattern used elsewhere in the
harness (e.g. :func:`rfbench.core.evaluate._environment_fingerprint`).

Requires the ``tasks`` extra (``scikit-learn>=1.3``, already declared in
``pyproject.toml``) -- no new dependency is added.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rfbench.regimes.probe import EmbeddingVector

#: Deterministic given the same ``(embeddings, labels)`` -- required by the ``Head`` protocol.
_RANDOM_STATE = 42


class LogisticRegressionHead:
    """A frozen-embedding classification head backed by ``sklearn`` logistic regression.

    Wraps ``sklearn.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000,
    random_state=42)`` behind the :class:`~rfbench.regimes.probe.Head` protocol so it drops
    into :class:`~rfbench.regimes.probe.LinearProbeAdapter` / ``FewShotAdapter`` unchanged.
    ``multi_class`` is left at sklearn's own default (auto-selects multinomial for >2
    classes on recent versions) rather than pinned, so the head tracks sklearn's own
    guidance as the pinned floor (``>=1.3``) advances.
    """

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415 - lazy by design

        #: The underlying fitted estimator; unfit until :meth:`fit` is called.
        self._model: Any = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=_RANDOM_STATE
        )
        self._fitted = False

    def fit(self, embeddings: Sequence[EmbeddingVector], labels: Sequence[int]) -> None:
        """Fit the logistic regression on paired ``(embedding, label)`` training data."""
        self._model.fit(list(embeddings), list(labels))
        self._fitted = True

    def predict(self, embeddings: Sequence[EmbeddingVector]) -> list[int]:
        """Predict a class id per embedding (order-preserving)."""
        if not self._fitted:
            raise RuntimeError("LogisticRegressionHead.predict called before fit")
        return [int(label) for label in self._model.predict(list(embeddings))]


__all__ = ["LogisticRegressionHead"]
