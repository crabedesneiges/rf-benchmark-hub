"""J2 acceptance tests for the HOC+LR AMC baseline.

The HOC feature extractor needs numpy and the classifier needs sklearn, so the whole module is
guarded with ``pytest.importorskip``: it SKIPS in the dependency-free lint/CI venv (no numpy/
sklearn) and RUNS on the ARM venv where ``rfbench[tasks]`` is installed. The cumulant formulas are
checked against toy signals with hand-derived moments (constant, balanced BPSK), and the Model
contract (forward -> per-class scores, embed -> HOC features, fit + n_params) is exercised on a
synthetic AMC split.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from rfbench.core.model import Model, Regime  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.hoc_amc import (  # noqa: E402
    MODEL_NAME,
    N_HOC_FEATURES,
    HocLogisticRegression,
    _hoc_features,
    _to_complex,
)

_WINDOW = 128


# --------------------------------------------------------------------------------------------------
# Cumulant / feature formulas on toy signals with known moments
# --------------------------------------------------------------------------------------------------
def test_to_complex_is_zero_mean() -> None:
    """_to_complex subtracts the window mean so the zero-mean assumption holds."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((2, _WINDOW))
    z = _to_complex(window)
    assert abs(complex(z.mean())) < 1e-12
    assert z.shape == (_WINDOW,)


def test_to_complex_rejects_bad_shape() -> None:
    """A non-(2, L) payload fails loudly rather than mis-computing features."""
    with pytest.raises(ValueError, match=r"shape \(2, L\)"):
        _to_complex(np.zeros((3, _WINDOW)))


def test_features_have_fixed_length() -> None:
    """Every window maps to exactly N_HOC_FEATURES features."""
    rng = np.random.default_rng(1)
    feats = _hoc_features(rng.standard_normal((2, _WINDOW)))
    assert len(feats) == N_HOC_FEATURES == 7
    assert all(isinstance(f, float) for f in feats)


def test_constant_window_yields_zero_higher_order_features() -> None:
    """A constant window is all-DC: after mean subtraction z==0, every cumulant magnitude is 0.

    With ``z = 0`` the power ``C21 = 0`` and only the ``_POWER_EPS`` floor avoids a divide-by-zero;
    all numerators (``|C20|``, ``|C40|``, …​) are exactly 0, so every reported feature is 0.
    """
    window = np.ones((2, _WINDOW)) * 3.7  # any constant -> DC-only
    feats = _hoc_features(window)
    assert all(abs(f) < 1e-9 for f in feats)


def test_bpsk_cumulants_match_hand_derivation() -> None:
    """Balanced real BPSK (z in {+1,-1}, equal counts) has hand-derivable cumulants.

    For z real, mean 0, |z|=1 everywhere: E[z^2]=E[z^4]=E[|z|^2]=E[|z|^4]=1, so
      C20 = 1, C21 = 1, C40 = 1 - 3*1^2 = -2, C42 = 1 - |1|^2 - 2*1^2 = -2.
    After the power normalisation (power == C21 == 1): |C20|/C21 = 1, the C21 feature = 1,
    |C40|/C21^2 = 2 and |C42|/C21^2 = 2. C41/C60/C63 are real here; only their magnitudes matter.
    """
    half = _WINDOW // 2
    i = np.concatenate([np.ones(half), -np.ones(half)])  # balanced +/-1, mean 0
    q = np.zeros(_WINDOW)  # real signal -> Q == 0
    feats = _hoc_features(np.stack([i, q]))
    # feats = [|C20|/C21, C21/C21, |C40|/C21^2, |C41|/C21^2, |C42|/C21^2, |C60|/C21^3, |C63|/C21^3]
    assert feats[0] == pytest.approx(1.0, abs=1e-9)  # |C20|/C21
    assert feats[1] == pytest.approx(1.0, abs=1e-9)  # C21/C21
    assert feats[2] == pytest.approx(2.0, abs=1e-9)  # |C40|/C21^2
    assert feats[4] == pytest.approx(2.0, abs=1e-9)  # |C42|/C21^2


def test_features_are_scale_invariant() -> None:
    """Scaling the whole window by a constant leaves the (power-normalised) features unchanged."""
    rng = np.random.default_rng(2)
    window = rng.standard_normal((2, _WINDOW))
    base = _hoc_features(window)
    scaled = _hoc_features(window * 17.0)
    assert base == pytest.approx(scaled, abs=1e-6)


# --------------------------------------------------------------------------------------------------
# Registration + Model contract
# --------------------------------------------------------------------------------------------------
def test_hoc_is_registered() -> None:
    """Importing the module registers it under 'hoc_lr' -> the class (registry path)."""
    assert MODELS.get(MODEL_NAME) is HocLogisticRegression


def test_hoc_implements_model_contract() -> None:
    """HOC baseline is a baseline-family Model with a non-empty name and from_scratch regime."""
    model = HocLogisticRegression()
    assert isinstance(model, Model)
    assert model.name == MODEL_NAME
    assert model.family == "baseline"
    assert model.regime.name is Regime.FROM_SCRATCH
    assert model.n_params == 0  # unfitted


def _synthetic_split(n_per_class: int = 30, n_classes: int = 3, seed: int = 42) -> list[dict]:
    """A separable synthetic AMC split: each class is a distinct constellation (real/imag mix)."""
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    for cls in range(n_classes):
        for _ in range(n_per_class):
            i = rng.standard_normal(_WINDOW)
            # Vary the I/Q correlation by class so HOC features separate the classes.
            q = (cls / max(n_classes - 1, 1)) * i + rng.standard_normal(_WINDOW) * 0.1
            samples.append({"iq": np.stack([i, q]), "label": cls, "snr_db": 0})
    rng.shuffle(samples)
    return samples


def test_fit_then_forward_returns_per_class_scores() -> None:
    """After fit, forward returns one probability vector per sample (width == n_classes)."""
    split = _synthetic_split()
    model = HocLogisticRegression().fit(split)
    batch = {"iq": [s["iq"] for s in split[:5]]}
    scores = model.forward(batch)
    assert len(scores) == 5
    assert all(len(row) == 3 for row in scores)
    # probabilities: each row sums to ~1 and argmax is a valid class.
    for row in scores:
        assert sum(row) == pytest.approx(1.0, abs=1e-6)
        assert 0 <= max(range(len(row)), key=row.__getitem__) < 3


def test_forward_before_fit_raises() -> None:
    """forward before fit is a loud error rather than a silent wrong answer."""
    with pytest.raises(RuntimeError, match="before fit"):
        HocLogisticRegression().forward({"iq": [np.zeros((2, _WINDOW))]})


def test_embed_returns_hoc_features() -> None:
    """embed returns the raw HOC feature vectors (usable for a linear_probe), no fit required."""
    model = HocLogisticRegression()
    window = np.random.default_rng(3).standard_normal((2, _WINDOW))
    feats = model.embed({"iq": [window]})
    assert len(feats) == 1
    assert len(feats[0]) == N_HOC_FEATURES


def test_n_params_positive_after_fit() -> None:
    """After fit, n_params reports the classifier coefficient + intercept count."""
    model = HocLogisticRegression().fit(_synthetic_split())
    n = model.n_params
    assert isinstance(n, int)
    # 3 classes * 7 features coefs + 3 intercepts == 24.
    assert n == 3 * N_HOC_FEATURES + 3


def test_fit_is_deterministic() -> None:
    """Two fits on the same seeded data give identical predictions (seeded LR, no shuffle)."""
    split = _synthetic_split()
    batch = {"iq": [s["iq"] for s in split[:6]]}
    a = HocLogisticRegression().fit(split).forward(batch)
    b = HocLogisticRegression().fit(split).forward(batch)
    assert a == pytest.approx(b)


def test_unbatched_window_is_accepted() -> None:
    """A single unbatched (2, L) window is promoted to a batch of one on both forward and embed."""
    model = HocLogisticRegression().fit(_synthetic_split())
    single = np.random.default_rng(4).standard_normal((2, _WINDOW))
    assert len(model.forward({"iq": single})) == 1
    assert len(model.embed({"iq": single})) == 1


def test_empty_train_split_raises() -> None:
    """Fitting on an empty split fails loudly."""
    with pytest.raises(ValueError, match="empty train split"):
        HocLogisticRegression().fit([])
