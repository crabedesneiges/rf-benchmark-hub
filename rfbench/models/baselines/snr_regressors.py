"""SNR-estimation regression baselines (J4): ``mean_snr`` floor + ``snr_moment_ridge``.

The two seed baselines that anchor the ``snr_estimation`` board (raw-IQ -> SNR in dB
regression), mirroring the AMC ``trivial_amc`` / ``hoc_amc`` pattern but for a **continuous**
target scored by error metrics (lower is better):

* ``mean_snr`` -- the regression "zero-rule" floor: always predicts the **mean SNR of the
  train split** (a constant). Its test RMSE is the standard deviation of the SNR distribution,
  the honest "predict the average" reference every real estimator must beat. Pure stdlib -- no
  numpy -- like the AMC ``majority_class`` plancher.
* ``snr_moment_ridge`` -- the DSP reference: a handful of **scale-invariant envelope/moment
  features** (kurtosis, PAPR, moment ratios, an M20/C42-style term) fed to a ridge regression
  (standardised features), fit from scratch on the train split. It mirrors ``hoc_lr`` (hand-
  designed features + a linear head from ``sklearn``) and is deterministic at seed 42, so a
  single run is the canonical row. numpy + scikit-learn are imported lazily inside the methods,
  so ``import rfbench`` stays dependency-free; the ``@register_model`` entries are created only
  on an explicit ``import rfbench.models.baselines.snr_regressors``.

Both are true :class:`~rfbench.core.model.Model` s in the ``"baseline"`` family and train via
their own out-of-contract :meth:`fit` (called by the dedicated CPU script before
:func:`rfbench.core.evaluate.evaluate`): :meth:`forward` returns ``list[float]`` -- one predicted
SNR (dB) per sample -- which the regression metrics (``Rmse`` / ``Mae`` in
:mod:`rfbench.tasks.snr_estimation.metrics`) consume directly. :attr:`regime` is ``from_scratch``
(features/priors learned only from the task's own train split).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor

#: One IQ window handed to the feature extractor: a ``(2, L)`` array-like (numpy on the cluster,
#: nested Python lists in a synthetic fixture). Stays dynamic until ``np.asarray``.
Window = object

#: Registry / leaderboard names written into ``result.json.model.name``.
MEAN_MODEL_NAME = "mean_snr"
RIDGE_MODEL_NAME = "snr_moment_ridge"
#: Number of scale-invariant SNR features per window (kurtosis, 6th-moment ratio, PAPR, envelope
#: coefficient-of-variation, |C20|/M2, |C42|/M2^2). Fixed so a mis-shaped batch fails loud.
N_SNR_FEATURES = 6
#: Ridge L2 strength (default scikit-learn); the estimator is deterministic, so seed 42 is nominal.
_RIDGE_ALPHA = 1.0
_RANDOM_STATE = 42
#: Power floor so a (numerically) zero-power window never divides by zero.
_EPS = 1e-12


def _batch_size(iq_batch: object) -> int:
    """Return the number of samples in a collated ``batch["iq"]`` (list or numpy ``(B, 2, L)``).

    Uses ``len`` for both a Python list of windows and a batched numpy array (whose first axis is
    the batch). A bare ``(2, L)`` single window is promoted to a batch of one. No numpy import, so
    ``mean_snr`` stays dependency-free.
    """
    shape = getattr(iq_batch, "shape", None)
    if shape is not None:  # numpy-like: (B, 2, L) -> B ; (2, L) -> 1 unbatched window
        return int(shape[0]) if len(shape) == 3 else 1
    return len(iq_batch)  # a plain list of per-sample windows


def _snr_features(window: Window) -> list[float]:
    """Return the length-:data:`N_SNR_FEATURES` scale-invariant SNR feature vector for one window.

    Reads ``z = I + jQ`` from a ``(2, L)`` window and summarises its envelope ``|z|`` by features
    that track SNR while being invariant to overall scale (ratios only): the envelope-power
    kurtosis ``M4/M2^2`` (2 for complex-Gaussian noise, structured for a clean signal), the 6th
    moment ratio ``M6/M2^3``, the peak-to-average power ratio, the envelope coefficient of
    variation, and two constellation-coherence cumulant terms (``|C20|/M2``, ``|C42|/M2^2``).
    Pure numpy -- no scipy.
    """
    import numpy as np  # noqa: PLC0415 - lazy by design (numpy absent on the frontend)

    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"expected an IQ window of shape (2, L); got {arr.shape}")
    z = arr[0] + 1j * arr[1]
    envelope = np.abs(z)
    power = envelope**2  # |z|^2

    m2 = float(np.mean(power)) + _EPS
    m4 = float(np.mean(power**2))
    m6 = float(np.mean(power**3))
    c20 = complex(np.mean(z * z))  # E[z^2]
    c42 = m4 - abs(c20) ** 2 - 2.0 * m2**2  # C42 cumulant (real for a proper process)

    kurtosis = m4 / m2**2
    sixth_ratio = m6 / m2**3
    papr = float(np.max(power)) / m2
    env_cov = float(np.std(envelope)) / (float(np.mean(envelope)) + _EPS)
    c20_norm = abs(c20) / m2
    c42_norm = abs(c42) / m2**2
    return [kurtosis, sixth_ratio, papr, env_cov, c20_norm, c42_norm]


def _features_for_batch(iq_batch: object) -> list[list[float]]:
    """Map a collated ``batch["iq"]`` (list of ``(2, L)`` windows) to a list of SNR feature rows.

    Accepts a single unbatched ``(2, L)`` window (promoted to a batch of one), so the model plugs
    into both the ``evaluate`` batch path and a direct single-sample call (mirrors ``hoc_amc``).
    """
    import numpy as np  # noqa: PLC0415 - lazy by design

    arr = np.asarray(iq_batch, dtype=np.float64)
    windows: Iterable[Window] = [arr] if arr.ndim == 2 else arr
    return [_snr_features(window) for window in windows]


@register_model(MEAN_MODEL_NAME)
class MeanSnr(Model):
    """Constant-mean SNR floor (registered ``"mean_snr"``): predicts the train-split mean SNR.

    The regression "zero-rule" -- the SNR analogue of the AMC ``majority_class`` plancher. Its
    test RMSE equals the standard deviation of the test SNR distribution, so it is the honest
    "predict the average" floor every real estimator must beat. Pure stdlib (no numpy/torch):
    :meth:`fit` learns one float (the train mean) and :meth:`forward` broadcasts it.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(self, *, name: str = MEAN_MODEL_NAME) -> None:
        """Construct the (unfitted) floor; :meth:`fit` learns the constant from the train split."""
        if not name:
            raise ValueError("MeanSnr needs a non-empty name")
        self.name = name
        #: Declared regime (D5): the constant is learned only from the task's own train split.
        self.regime = RegimeSpec(Regime.FROM_SCRATCH)
        #: The learned constant SNR (dB); ``None`` until :meth:`fit` runs.
        self._mean: float | None = None

    def fit(self, samples: Iterable[Batch]) -> MeanSnr:
        """Learn the mean SNR (dB) of the train split (out-of-contract). Returns ``self``."""
        values = [float(sample["snr_db"]) for sample in samples]
        if not values:
            raise ValueError("MeanSnr.fit received an empty train split")
        self._mean = sum(values) / len(values)
        return self

    def forward(self, x: Batch) -> Tensor:
        """Return the constant train-mean SNR for every sample in the batch (``list[float]``)."""
        if self._mean is None:
            raise RuntimeError("MeanSnr.forward called before fit")
        return [self._mean] * _batch_size(x["iq"])

    def embed(self, x: Batch) -> Tensor:  # pragma: no cover - a constant predictor has no rep
        """A constant predictor has no representation to probe."""
        raise NotImplementedError("mean_snr is a constant floor; it exposes no embedding")

    @property
    def n_params(self) -> int:
        """The single learned constant (``0`` before fit)."""
        return 0 if self._mean is None else 1


@register_model(RIDGE_MODEL_NAME)
class SnrMomentRidge(Model):
    """Moment-feature + ridge SNR regressor (registered ``"snr_moment_ridge"``), the DSP reference.

    Not an ``nn.Module``: it trains through its own :meth:`fit` (called by the dedicated CPU
    script before :func:`rfbench.core.evaluate.evaluate`), computing :data:`N_SNR_FEATURES`
    scale-invariant envelope/moment features per window and fitting a standardised
    ``sklearn`` ridge regression -- the SNR analogue of ``hoc_lr``. Deterministic at seed 42, so a
    single run is the canonical row. :meth:`embed` returns the raw feature rows (a genuine frozen
    representation, so a ``linear_probe`` regime can re-fit); :attr:`n_params` is the fitted
    coefficient count.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(self, *, name: str = RIDGE_MODEL_NAME) -> None:
        """Construct the (unfitted) regressor; the sklearn pipeline is built lazily in ``fit``."""
        if not name:
            raise ValueError("SnrMomentRidge needs a non-empty name")
        self.name = name
        #: Declared regime (D5): hand-designed features + a head fit from scratch on the train.
        self.regime = RegimeSpec(Regime.FROM_SCRATCH)
        #: The fitted ``StandardScaler -> Ridge`` pipeline; ``None`` until :meth:`fit` runs.
        self._model: Any = None

    def fit(self, samples: Iterable[Batch]) -> SnrMomentRidge:
        """Fit ``StandardScaler -> Ridge`` on the SNR features of a train split (out-of-contract).

        ``samples`` is any iterable of per-sample dicts ``{"iq": (2, L), "snr_db": float, ...}`` --
        e.g. the object ``SnrDataset.load("train")`` returns. Returns ``self`` for call chaining;
        deterministic given the same data (ridge has a closed-form solution), so a single seed-42
        run is the canonical row.
        """
        from sklearn.linear_model import Ridge  # noqa: PLC0415 - lazy by design
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        features: list[list[float]] = []
        targets: list[float] = []
        for sample in samples:
            features.append(_snr_features(sample["iq"]))
            targets.append(float(sample["snr_db"]))
        if not features:
            raise ValueError("SnrMomentRidge.fit received an empty train split")
        model = make_pipeline(
            StandardScaler(), Ridge(alpha=_RIDGE_ALPHA, random_state=_RANDOM_STATE)
        )
        model.fit(features, targets)
        self._model = model
        return self

    def forward(self, x: Batch) -> Tensor:
        """Return the predicted SNR (dB) for every sample (``list[float]``). Raises before fit."""
        if self._model is None:
            raise RuntimeError("SnrMomentRidge.forward called before fit")
        features = _features_for_batch(x["iq"])
        return [float(prediction) for prediction in self._model.predict(features)]

    def embed(self, x: Batch) -> Tensor:
        """Return the raw SNR feature rows ``list[list[float]]`` for a ``linear_probe`` regime."""
        return _features_for_batch(x["iq"])

    @property
    def n_params(self) -> int:
        """Fitted-regressor coefficient count (``0`` before fit); written to ``model.n_params``."""
        if self._model is None:
            return 0
        ridge = self._model.steps[-1][1]
        return int(ridge.coef_.size + 1)  # coefficients + intercept


__all__ = [
    "MeanSnr",
    "SnrMomentRidge",
    "MEAN_MODEL_NAME",
    "RIDGE_MODEL_NAME",
    "N_SNR_FEATURES",
]
