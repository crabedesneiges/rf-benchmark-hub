"""HOC+LR AMC baseline (J2) -- higher-order-cumulant features + logistic regression.

The classical, *learning-free-feature* modulation classifier of the AMC literature
(Swami & Sadler, "Hierarchical Digital Modulation Classification Using Cumulants",
IEEE Trans. Commun. 48(3), 2000; Dobre et al., "Survey of automatic modulation
classification techniques", IET Commun. 1(2), 2007). Each IQ window ``(2, L)`` is read as a
complex baseband signal ``z = I + jQ`` and summarised by a handful of *higher-order cumulants*
(HOC) -- statistics that are (near-)invariant to carrier phase/scale yet separate the analog
(AM/FM/…​) and digital (PSK/QAM) constellations by their fourth- and sixth-order structure.
A ``sklearn`` :class:`~sklearn.linear_model.LogisticRegression` (the SAME lbfgs / max_iter=1000
/ seed-42 estimator as the normative :class:`~rfbench.regimes.heads.LogisticRegressionHead`)
maps the cumulant vector to one of the modulation classes. It seeds the AMC board as the
strong *DSP* reference the deep baselines (MCLDNN, ResNet, CLDNN) must beat.

Cumulant definitions (over one window; expectations are the sample means ``E[.] = mean(.)``
of the length-``L`` complex sequence ``z``; ``z*`` is the complex conjugate). The features are
the magnitudes ``|C_pq|`` of the standard AMC cumulant set:

* ``C20 = E[z^2]``
* ``C21 = E[|z|^2]``                                     (the signal power)
* ``C40 = E[z^4]      - 3 * C20^2``
* ``C41 = E[z^3 z*]   - 3 * C20 * C21``
* ``C42 = E[|z|^4]    - |C20|^2 - 2 * C21^2``
* ``C60 = E[z^6]      - 15 * C20 * C40 - 30 * C20^3``
* ``C63 = E[|z|^4 z^2]- 6 * C20 * C42 - 9 * C21 * C40``
         ``- 18 * C20 * C21^2 - 9 * |C20|^2 * C20``

These are the textbook cumulant-to-moment relations for a zero-mean complex process (Swami &
Sadler 2000, Table I; Mendel 1991, "Tutorial on higher-order statistics"). ``C40``/``C42`` are
the most discriminative between PSK and QAM, ``C60``/``C63`` add separation on the denser
(RML2018 24-way) constellations. The window mean is subtracted first so the "zero-mean"
assumption holds even for the residual DC of a short window; the two magnitudes ``|C21|`` and
``|C42|`` are then used to power-normalise the higher orders (``C4x/C21^2``, ``C6x/C21^3``) so a
per-window amplitude scale carries no information -- exactly the invariance the DSP classifier
relies on.

Contract bridge (read ``rfbench/core/model.py``). Unlike the torch baselines this model is
*not* an ``nn.Module``: it is a pure-numpy feature extractor + a fitted sklearn head, so it does
NOT go through ``rfbench.training.train_baseline`` (which requires an ``nn.Module``). Instead it
exposes an out-of-contract :meth:`fit` that a dedicated CPU script (``slurm/train_hoc_amc.sh``)
calls on the train split before handing the fitted model to
:func:`rfbench.core.evaluate.evaluate`. :meth:`forward` returns ``list[list[float]]`` -- one
per-class score vector per sample -- which the AMC metrics' pure-stdlib ``argmax`` decoder
(``rfbench.tasks.amc.metrics._as_class_index``) consumes with no torch. :meth:`embed` returns the
raw HOC feature vectors so a ``linear_probe`` regime can re-fit its own head on them.

HARD CONSTRAINT (mirrors the torch baselines): ``import rfbench`` stays dependency-free. This
module imports **neither numpy nor sklearn at module top** -- both are pulled in lazily *inside*
the methods (the same pattern as the ``torch`` imports of the sibling baselines and the sklearn
import of :class:`~rfbench.regimes.heads.LogisticRegressionHead`), so the module stays importable
on the numpy-less x86 frontend. The ``@register_model("hoc_lr")`` entry in
:data:`rfbench.core.registry.MODELS` is created only on an explicit
``import rfbench.models.baselines.hoc_amc``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor

if TYPE_CHECKING:  # numpy is a lazy runtime dep; annotate without importing it at module top.
    import numpy as np

#: One IQ window as handed to the feature extractor: a ``(2, L)`` array-like (numpy array on the
#: cluster, nested Python lists in a synthetic fixture). Typed ``object`` -- like ``mcldnn``'s
#: ``_iq_to_tensor(iq_batch: object)`` -- because it stays fully dynamic until ``np.asarray``.
Window = object

#: The leaderboard / registry name written into ``result.json.model.name``.
MODEL_NAME = "hoc_lr"
#: Number of HOC features per window (|C20|, |C21|, |C40|, |C41|, |C42|, |C60|, |C63| after the
#: power-normalisation of the >=4-order cumulants). Fixed so a mis-shaped feature batch fails loud.
N_HOC_FEATURES = 7
#: Regularisation floor added to the window power before the higher-order cumulants are
#: power-normalised, so a (numerically) zero-power window does not divide by zero.
_POWER_EPS = 1e-12
#: The seed shared with the normative LogisticRegressionHead (lbfgs / max_iter=1000 / seed 42).
_RANDOM_STATE = 42


def _to_complex(window: Window) -> np.ndarray:
    """Read one ``(2, L)`` IQ window as a length-``L`` zero-mean complex vector ``z = I + jQ``.

    ``window`` is one element of the collated ``batch["iq"]`` list -- a ``(2, L)`` array-like
    (numpy on the cluster, nested lists in a synthetic fixture); ``np.asarray`` handles both. The
    window mean is subtracted so the "zero-mean process" assumption the cumulant-to-moment
    relations rest on holds even for a short window's residual DC. Raises ``ValueError`` on a
    non-``(2, L)`` payload so a mis-shaped batch fails loudly.
    """
    import numpy as np  # noqa: PLC0415 - lazy by design (numpy absent on the frontend)

    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"expected an IQ window of shape (2, L); got {arr.shape}")
    z = arr[0] + 1j * arr[1]
    centered: np.ndarray = z - z.mean()
    return centered


def _hoc_features(window: Window) -> list[float]:
    """Return the length-:data:`N_HOC_FEATURES` HOC feature vector for one IQ window.

    Computes the standard AMC cumulant set (see the module docstring for the exact
    cumulant-to-moment formulas) from the sample moments of ``z = I + jQ``, then reports the
    scale-invariant magnitudes: ``|C20|`` and ``|C21|`` normalised by the window power ``C21``
    (so ``|C20|/C21`` in [0, 1]), and the fourth/sixth-order cumulants power-normalised
    (``|C4x|/C21^2``, ``|C6x|/C21^3``). Pure numpy -- no scipy -- so it runs anywhere numpy does.
    """
    import numpy as np  # noqa: PLC0415 - lazy by design

    z = _to_complex(window)
    zc = np.conjugate(z)
    abs2 = np.abs(z) ** 2  # |z|^2

    # Raw moments (sample means) needed by the cumulant-to-moment relations.
    m20 = np.mean(z * z)  # E[z^2]
    m21 = np.mean(abs2)  # E[|z|^2] == C21 (real, the window power)
    m40 = np.mean(z**4)  # E[z^4]
    m41 = np.mean(z**3 * zc)  # E[z^3 z*]
    m42 = np.mean(abs2**2)  # E[|z|^4]
    m60 = np.mean(z**6)  # E[z^6]
    m63 = np.mean(abs2**2 * z * z)  # E[|z|^4 z^2]

    # Cumulants (Swami & Sadler 2000, Table I; zero-mean complex process).
    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * m20 * m20
    c41 = m41 - 3.0 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2.0 * m21**2
    c60 = m60 - 15.0 * m20 * c40 - 30.0 * m20**3
    c63 = (
        m63
        - 6.0 * m20 * c42
        - 9.0 * m21 * c40
        - 18.0 * (m20 * m21**2)
        - 9.0 * (np.abs(m20) ** 2) * m20
    )

    power = float(c21.real) + _POWER_EPS
    return [
        float(np.abs(c20)) / power,  # |C20| / C21   (phase/scale-robust)
        float(c21.real) / power,  # |C21| / C21 == 1 after norm; kept so the raw power is dropped
        float(np.abs(c40)) / power**2,  # |C40| / C21^2
        float(np.abs(c41)) / power**2,  # |C41| / C21^2
        float(np.abs(c42)) / power**2,  # |C42| / C21^2
        float(np.abs(c60)) / power**3,  # |C60| / C21^3
        float(np.abs(c63)) / power**3,  # |C63| / C21^3
    ]


def _features_for_batch(iq_batch: object) -> list[list[float]]:
    """Map a collated ``batch["iq"]`` (list of ``(2, L)`` windows) to a list of HOC vectors.

    Accepts a single unbatched ``(2, L)`` window (promoted to a batch of one) so the model plugs
    into both the ``evaluate`` batch path and a direct single-sample call.
    """
    import numpy as np  # noqa: PLC0415 - lazy by design

    # Coerce once, so a nested-list fixture and a real numpy batch take the same path; a 2-D array
    # is one unbatched (2, L) window, a 3-D array is (B, 2, L). Iterating a numpy array yields its
    # first-axis slices, which are exactly the per-sample (2, L) windows _hoc_features consumes.
    arr = np.asarray(iq_batch, dtype=np.float64)
    windows: Iterable[Window] = [arr] if arr.ndim == 2 else arr
    return [_hoc_features(window) for window in windows]


@register_model(MODEL_NAME)
class HocLogisticRegression(Model):
    """HOC-feature + logistic-regression AMC baseline (registered ``"hoc_lr"``).

    A :class:`~rfbench.core.model.Model` in the ``"baseline"`` family whose forward pass is a
    numpy HOC feature extractor followed by a fitted ``sklearn`` logistic regression. It is *not*
    an ``nn.Module`` and therefore trains through its own :meth:`fit` (an out-of-``Model``-contract
    method the dedicated CPU script calls) rather than through ``rfbench.training.train_baseline``:

    * :meth:`fit` computes the HOC features of every train sample and fits the logistic regression.
    * :meth:`forward` returns ``list[list[float]]`` (one per-class probability vector per sample),
      decoded by the AMC metrics' pure-stdlib ``argmax`` -- no torch anywhere on the eval path.
    * :meth:`embed` returns the raw HOC feature vectors so a ``linear_probe`` regime can re-fit.
    * :attr:`n_params` reports the fitted classifier's coefficient count (``0`` until fit).
    * :attr:`regime` is ``from_scratch``: the features are hand-designed and the head is fit from
      scratch on the task's own train split (no pretraining), so this is the regime ``evaluate``
      writes into ``result.json.regime``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(self, *, name: str = MODEL_NAME) -> None:
        """Construct the (unfitted) baseline; the sklearn estimator is built lazily in :meth:`fit`.

        Kept argument-free-friendly (only an optional ``name``) so the registry path
        ``MODELS.get("hoc_lr")()`` builds it with no arguments, exactly like the torch baselines.
        """
        if not name:
            raise ValueError("HocLogisticRegression needs a non-empty name")
        self.name = name
        #: Declared regime (D5): hand-designed features + a head fit from scratch on the train.
        self.regime = RegimeSpec(Regime.FROM_SCRATCH)
        #: The fitted sklearn estimator; ``None`` until :meth:`fit` runs.
        self._clf: Any = None
        #: Class ids in the fitted classifier's column order (for the score-vector layout).
        self._classes: list[int] = []

    def fit(self, samples: Iterable[Batch]) -> HocLogisticRegression:
        """Fit the logistic regression on the HOC features of a train split (out-of-contract).

        ``samples`` is any iterable of per-sample dicts ``{"iq": (2, L), "label": int, ...}`` --
        e.g. the object ``AmcDataset.load("train")`` returns. Extracts the HOC feature vector of
        every sample, then fits ``LogisticRegression(solver="lbfgs", max_iter=1000,
        random_state=42)`` -- the SAME estimator as the normative
        :class:`~rfbench.regimes.heads.LogisticRegressionHead`, so the DSP row is directly
        comparable to the FM linear-probe rows. Returns ``self`` for call chaining. Deterministic
        given the same data (seeded, no shuffle), so a single seed-42 run is the canonical row.
        """
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415 - lazy by design

        features: list[list[float]] = []
        labels: list[int] = []
        for sample in samples:
            features.append(_hoc_features(sample["iq"]))
            labels.append(int(sample["label"]))
        if not features:
            raise ValueError("HocLogisticRegression.fit received an empty train split")
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=_RANDOM_STATE)
        clf.fit(features, labels)
        self._clf = clf
        self._classes = [int(c) for c in clf.classes_]
        return self

    def forward(self, x: Batch) -> Tensor:
        """Return per-class probability vectors ``list[list[float]]`` for the collated AMC batch.

        Each row is the class-probability vector (``predict_proba``) in the class-id order the
        classifier learned; the AMC metrics take its ``argmax`` per sample. Raises
        ``RuntimeError`` if called before :meth:`fit`.
        """
        if self._clf is None:
            raise RuntimeError("HocLogisticRegression.forward called before fit")
        features = _features_for_batch(x["iq"])
        proba = self._clf.predict_proba(features)
        return [[float(p) for p in row] for row in proba]

    def embed(self, x: Batch) -> Tensor:
        """Return the raw HOC feature vectors ``list[list[float]]`` for a ``linear_probe`` regime.

        Unlike the trivial planchers, HOC features are a genuine frozen representation, so this
        baseline supports ``linear_probe`` (a head re-fit on these vectors). No fit is required.
        """
        return _features_for_batch(x["iq"])

    @property
    def n_params(self) -> int:
        """Fitted-classifier coefficient count (``0`` before fit); written to model.n_params."""
        if self._clf is None:
            return 0
        return int(self._clf.coef_.size + self._clf.intercept_.size)


__all__ = [
    "HocLogisticRegression",
    "MODEL_NAME",
    "N_HOC_FEATURES",
]
