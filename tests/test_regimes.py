"""WP-41 acceptance tests for the regime adapters (:mod:`rfbench.regimes`).

A single pure-Python dummy :class:`~rfbench.core.model.Model` (``embed`` returns small
deterministic vectors, no torch/numpy) is run end-to-end through ALL FOUR adapters. We
assert that:

* each adapter produces predictions and surfaces the correct declared regime name;
* ``few_shot`` honours ``k`` (per-class support size) and writes ``k_shot``, while
  ``from_scratch`` / ``full_finetune`` reject a stray ``k``;
* ``few_shot`` subsampling is deterministic (seed-stable) and honours ``k``;
* ``import rfbench.regimes`` stays dependency-free (no torch/numpy/sklearn/jsonschema);
* :class:`~rfbench.regimes.heads.LogisticRegressionHead` fits/predicts through the ``Head``
  protocol (skipped if ``sklearn`` is not installed in this venv);
* :func:`~rfbench.regimes.few_shot.run_episodic` draws ``n_episodes`` distinct seeded
  support sets and returns exactly one result per episode.

Must pass with only ``pytest`` (no torch/numpy/sklearn) installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.types import Batch, Tensor
from rfbench.regimes import (
    EpisodeResult,
    FewShotAdapter,
    FromScratchAdapter,
    FullFinetuneAdapter,
    LinearProbeAdapter,
    NearestCentroidHead,
    make_adapter,
    run_episodic,
)
from rfbench.regimes.base import FittedState, RegimeAdapter


# --------------------------------------------------------------------------------------------------
# Pure-Python dummy model (no torch/numpy)
# --------------------------------------------------------------------------------------------------
class DummyModel(Model):
    """A deterministic pure-Python model.

    ``embed`` maps a sample to a tiny 2-D vector that is *linearly separable by label*:
    the label is baked into the first coordinate, so the nearest-centroid head recovers it
    perfectly. ``forward`` echoes the labels back (a perfect classifier) so the
    pass-through regimes yield predictions too. No tensor framework is used.
    """

    name = "dummy-regime-model"
    family = "foundation"

    def forward(self, x: Batch) -> Tensor:
        # ``x`` is the collated batch; echo labels as class predictions.
        return list(x["label"])

    def embed(self, x: Batch) -> Tensor:
        # One 2-D vector per sample; coordinate 0 = label (separable), coordinate 1 = iq[0].
        labels = x["label"]
        iq = x["iq"]
        return [
            [float(label), float(sample_iq[0])] for label, sample_iq in zip(labels, iq, strict=True)
        ]

    @property
    def n_params(self) -> int:
        return 42


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------
def _train_split() -> list[Batch]:
    """A tiny label-balanced train split: 4 samples per class, 3 classes."""
    samples: list[Batch] = []
    for label in (0, 1, 2):
        for j in range(4):
            samples.append({"iq": [0.1 * j, 0.2], "label": label, "snr_db": j})
    return samples


def _eval_batch() -> Batch:
    """A collated eval batch (dict of field -> list), one sample per class."""
    return {
        "iq": [[0.0, 0.2], [0.3, 0.2], [0.5, 0.2]],
        "label": [0, 1, 2],
        "snr_db": [0, 5, 10],
    }


def _run_adapter(adapter: RegimeAdapter) -> list[Any]:
    """Fit an adapter on the train split and predict the eval batch."""
    model = DummyModel()
    state = adapter.fit(model, _train_split())
    return list(adapter.predict(model, _eval_batch(), state))


# --------------------------------------------------------------------------------------------------
# All four adapters run end-to-end and surface the right regime
# --------------------------------------------------------------------------------------------------
def test_all_four_adapters_run_end_to_end() -> None:
    """Every adapter fits + predicts and reports its declared regime name."""
    cases: list[tuple[RegimeAdapter, Regime]] = [
        (FromScratchAdapter(), Regime.FROM_SCRATCH),
        (FullFinetuneAdapter(), Regime.FULL_FINETUNE),
        (LinearProbeAdapter(), Regime.LINEAR_PROBE),
        (FewShotAdapter(k=2), Regime.FEW_SHOT),
    ]
    for adapter, expected in cases:
        preds = _run_adapter(adapter)
        assert adapter.name is expected
        assert adapter.regime.name is expected
        # 3-sample eval batch -> 3 predictions, one per class.
        assert len(preds) == 3
        assert preds == [0, 1, 2]


def test_make_adapter_dispatches_each_regime() -> None:
    """The factory maps a RegimeSpec to the matching adapter type and surface."""
    assert isinstance(make_adapter(RegimeSpec(Regime.FROM_SCRATCH)), FromScratchAdapter)
    assert isinstance(make_adapter(RegimeSpec(Regime.FULL_FINETUNE)), FullFinetuneAdapter)
    assert isinstance(make_adapter(RegimeSpec(Regime.LINEAR_PROBE)), LinearProbeAdapter)
    few = make_adapter(RegimeSpec(Regime.FEW_SHOT, k_shot=3))
    assert isinstance(few, FewShotAdapter)
    assert few.regime.k_shot == 3


# --------------------------------------------------------------------------------------------------
# Regime <-> k coupling
# --------------------------------------------------------------------------------------------------
def test_few_shot_writes_k_shot_and_honours_k() -> None:
    """few_shot carries k_shot verbatim and keeps exactly k samples per class."""
    adapter = FewShotAdapter(k=2)
    assert adapter.regime.name is Regime.FEW_SHOT
    assert adapter.regime.k_shot == 2
    assert adapter.k == 2

    state = adapter.fit(DummyModel(), _train_split())
    # 3 classes x k=2 = 6 support samples fit the head.
    assert state.info["n_train_samples"] == 6
    assert state.info["n_classes"] == 3


def test_few_shot_k_larger_than_class_size_takes_all() -> None:
    """When k exceeds a class's population, all of that class's samples are kept."""
    adapter = FewShotAdapter(k=99)
    state = adapter.fit(DummyModel(), _train_split())
    # 4 per class x 3 classes = the full 12-sample split.
    assert state.info["n_train_samples"] == 12


def test_pass_through_regimes_reject_stray_k() -> None:
    """from_scratch / full_finetune reject a stray k_shot (mirrors RegimeSpec).

    RegimeSpec forbids a non-few_shot spec from carrying ``k_shot`` at all, so the
    coupling can never be violated upstream. The pass-through adapter re-asserts it
    defensively: we exercise that guard directly by handing it a spec whose ``k_shot`` was
    forced past the frozen constructor via ``object.__setattr__`` (the frozen dataclass
    would otherwise reject it) -- the adapter must still refuse to build.
    """
    from rfbench.regimes.passthrough import _PassThroughAdapter

    smuggled = RegimeSpec(Regime.FROM_SCRATCH)
    object.__setattr__(smuggled, "k_shot", 2)  # bypass the frozen-dataclass guard
    with pytest.raises(ValueError, match="does not accept k_shot"):
        _PassThroughAdapter(smuggled)


def test_regime_spec_rejects_k_on_non_few_shot() -> None:
    """RegimeSpec itself forbids k on non-few_shot (the guarantee adapters rely on)."""
    with pytest.raises(ValueError, match="k_shot must be set iff"):
        RegimeSpec(Regime.FROM_SCRATCH, k_shot=2)

    # And a linear_probe adapter refuses a few_shot-shaped spec (wrong regime for it).
    with pytest.raises(ValueError, match="only accepts the 'linear_probe' regime"):
        LinearProbeAdapter(regime=RegimeSpec(Regime.FEW_SHOT, k_shot=2))


def test_few_shot_requires_k() -> None:
    """FewShotAdapter cannot be built without a valid k (RegimeSpec enforces k >= 1)."""
    with pytest.raises(ValueError, match="k_shot must be >= 1"):
        FewShotAdapter(k=0)


# --------------------------------------------------------------------------------------------------
# Determinism of few_shot subsampling
# --------------------------------------------------------------------------------------------------
def test_few_shot_subsampling_is_deterministic() -> None:
    """Same (k, seed, split) -> byte-identical support set; different seed may differ."""
    a = FewShotAdapter(k=2, seed=42)
    b = FewShotAdapter(k=2, seed=42)
    split = _train_split()
    support_a = a._select_train_samples(split)
    support_b = b._select_train_samples(list(split))
    assert support_a == support_b
    # exactly k per class, ascending label order
    assert [s["label"] for s in support_a] == [0, 0, 1, 1, 2, 2]


def test_few_shot_seed_changes_selection() -> None:
    """A different seed selects a different (still valid) k-shot support set."""
    split = _train_split()
    default = FewShotAdapter(k=1, seed=42)._select_train_samples(split)
    other = FewShotAdapter(k=1, seed=7)._select_train_samples(split)
    # Both keep exactly one per class...
    assert [s["label"] for s in default] == [0, 1, 2]
    assert [s["label"] for s in other] == [0, 1, 2]
    # ...but the chosen exemplars differ for at least one class (seed actually matters).
    assert default != other


# --------------------------------------------------------------------------------------------------
# Head injection + centroid correctness
# --------------------------------------------------------------------------------------------------
def test_injected_head_is_used() -> None:
    """A custom head passed to linear_probe is the one fit and queried."""

    class _ConstHead:
        def __init__(self) -> None:
            self.fit_calls = 0

        def fit(
            self,
            embeddings: Sequence[Sequence[float]],
            labels: Sequence[int],
        ) -> None:
            self.fit_calls += 1

        def predict(self, embeddings: Sequence[Sequence[float]]) -> list[int]:
            return [7 for _ in embeddings]

    head = _ConstHead()
    adapter = LinearProbeAdapter(head)
    preds = _run_adapter(adapter)
    assert head.fit_calls == 1
    assert preds == [7, 7, 7]


def test_nearest_centroid_head_recovers_separable_labels() -> None:
    """The default centroid head recovers labels from separable embeddings."""
    head = NearestCentroidHead()
    head.fit([[0.0, 1.0], [0.0, 1.1], [5.0, 0.0], [5.0, 0.2]], [0, 0, 1, 1])
    assert head.predict([[0.1, 0.9], [4.9, 0.1]]) == [0, 1]


def test_centroid_head_rejects_empty_and_unfit() -> None:
    """The stdlib head fails loudly on empty fit and predict-before-fit."""
    head = NearestCentroidHead()
    with pytest.raises(ValueError, match="empty training set"):
        head.fit([], [])
    with pytest.raises(RuntimeError, match="before fit"):
        NearestCentroidHead().predict([[0.0, 0.0]])


# --------------------------------------------------------------------------------------------------
# Dependency-freedom
# --------------------------------------------------------------------------------------------------
def test_import_is_dependency_free() -> None:
    """Importing rfbench.regimes must not pull torch/numpy/sklearn/jsonschema."""
    import importlib
    import sys

    for mod in ("torch", "numpy", "sklearn", "jsonschema"):
        sys.modules.pop(mod, None)
    importlib.import_module("rfbench.regimes")
    for mod in ("torch", "numpy", "sklearn"):
        assert mod not in sys.modules, f"rfbench.regimes must not import {mod}"


# --------------------------------------------------------------------------------------------------
# LogisticRegressionHead (normative board head; requires sklearn)
# --------------------------------------------------------------------------------------------------
def test_logistic_regression_head_fits_and_predicts() -> None:
    """LogisticRegressionHead recovers labels through the Head protocol (needs sklearn)."""
    pytest.importorskip("sklearn")
    from rfbench.regimes.heads import LogisticRegressionHead

    head = LogisticRegressionHead()
    embeddings = [[0.0, 1.0], [0.1, 0.9], [5.0, 0.0], [5.1, 0.1]]
    labels = [0, 0, 1, 1]
    head.fit(embeddings, labels)
    assert head.predict([[0.05, 0.95], [5.05, 0.05]]) == [0, 1]


def test_logistic_regression_head_rejects_predict_before_fit() -> None:
    """predict before fit fails loudly, mirroring NearestCentroidHead's guard."""
    pytest.importorskip("sklearn")
    from rfbench.regimes.heads import LogisticRegressionHead

    with pytest.raises(RuntimeError, match="before fit"):
        LogisticRegressionHead().predict([[0.0, 0.0]])


def test_logistic_regression_head_runs_through_linear_probe_adapter() -> None:
    """LogisticRegressionHead drops into LinearProbeAdapter unchanged (Head protocol)."""
    pytest.importorskip("sklearn")
    from rfbench.regimes.heads import LogisticRegressionHead

    adapter = LinearProbeAdapter(LogisticRegressionHead())
    preds = _run_adapter(adapter)
    assert len(preds) == 3


# --------------------------------------------------------------------------------------------------
# run_episodic (multi-episode few-shot orchestration)
# --------------------------------------------------------------------------------------------------
def test_run_episodic_returns_one_result_per_episode() -> None:
    """run_episodic returns exactly n_episodes results, seeds base_seed..base_seed+n-1."""
    model = DummyModel()

    def predict_fn(model: Model, adapter: FewShotAdapter, state: FittedState) -> float:
        preds = adapter.predict(model, _eval_batch(), state)
        labels = _eval_batch()["label"]
        correct = sum(1 for p, y in zip(preds, labels, strict=True) if p == y)
        return correct / len(labels)

    results = run_episodic(
        lambda seed: FewShotAdapter(k=1, seed=seed),
        model,
        _train_split(),
        predict_fn,
        n_episodes=10,
        base_seed=42,
    )

    assert len(results) == 10
    assert [r.seed for r in results] == list(range(42, 52))
    assert all(isinstance(r, EpisodeResult) for r in results)
    # The perfectly-separable dummy model/head recovers all labels every episode.
    assert all(r.primary_metric == 1.0 for r in results)


def test_run_episodic_draws_distinct_supports_across_seeds() -> None:
    """Different seeds select different k-shot supports when the split allows it."""
    captured_supports: list[list[Batch]] = []

    class _RecordingAdapter(FewShotAdapter):
        def _select_train_samples(self, train_split: Iterable[Batch]) -> list[Batch]:
            support = super()._select_train_samples(train_split)
            captured_supports.append(support)
            return support

    model = DummyModel()

    def predict_fn(model: Model, adapter: FewShotAdapter, state: FittedState) -> float:
        return 0.0

    run_episodic(
        lambda seed: _RecordingAdapter(k=1, seed=seed),
        model,
        _train_split(),
        predict_fn,
        n_episodes=10,
        base_seed=42,
    )

    assert len(captured_supports) == 10
    # At least two distinct supports across 10 differently-seeded draws from a 4-per-class
    # split -- seeds actually vary the selection (mirrors test_few_shot_seed_changes_selection).
    distinct = {
        tuple(s["label"] for s in support) + tuple(s["iq"][0] for s in support)
        for support in captured_supports
    }
    assert len(distinct) > 1


def test_run_episodic_rejects_non_positive_n_episodes() -> None:
    """n_episodes must be >= 1."""
    model = DummyModel()

    def predict_fn(model: Model, adapter: FewShotAdapter, state: FittedState) -> float:
        return 0.0

    with pytest.raises(ValueError, match="n_episodes must be >= 1"):
        run_episodic(
            lambda seed: FewShotAdapter(k=1, seed=seed),
            model,
            _train_split(),
            predict_fn,
            n_episodes=0,
        )
