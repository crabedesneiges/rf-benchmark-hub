"""J2 acceptance tests for the trivial AMC floor baselines (``majority_class`` / ``chance``).

These two models are PURE STDLIB -- no numpy, no torch, no sklearn -- so this whole module runs in
the dependency-free lint/CI venv with only ``pytest`` + ``jsonschema`` installed (no importorskip
needed). Everything is exercised on hand-built synthetic AMC samples shaped exactly like
:class:`~rfbench.tasks.amc.dataset.AmcDataset` yields, including an end-to-end
:func:`rfbench.core.evaluate.evaluate` pass that validates against ``result.schema.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Model, Regime
from rfbench.core.registry import MODELS
from rfbench.core.types import Batch
from rfbench.models.baselines.trivial_amc import (
    CHANCE_MODEL_NAME,
    MAJORITY_MODEL_NAME,
    MajorityClass,
    UniformChance,
)
from rfbench.tasks.amc import AmcDataset, AmcTask


def _load_schema() -> dict[str, Any]:
    """Load and parse ``result.schema.json`` for the end-to-end validators."""
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return schema


_CHECKSUM = "sha256:" + "cd" * 32


def _train_samples(labels: list[int]) -> list[Batch]:
    """Build a train split: per-sample dicts with a (2, 2) IQ stub and the given labels."""
    return [{"iq": [[0.1, 0.2], [0.3, 0.4]], "label": lbl, "snr_db": 0} for lbl in labels]


def _collated_batch(batch_size: int) -> dict[str, Any]:
    """A collated eval batch: ``iq`` a list of ``batch_size`` (2, 2) windows (labels irrelevant)."""
    return {"iq": [[[0.1, 0.2], [0.3, 0.4]] for _ in range(batch_size)]}


# --------------------------------------------------------------------------------------------------
# Registration + Model contract
# --------------------------------------------------------------------------------------------------
def test_floors_are_registered() -> None:
    """Importing the module registers both floors under their lowercase names."""
    assert MODELS.get(MAJORITY_MODEL_NAME) is MajorityClass
    assert MODELS.get(CHANCE_MODEL_NAME) is UniformChance


def test_floors_implement_model_contract() -> None:
    """Both floors are baseline-family Models with a non-empty name and n_params == 0."""
    for model in (MajorityClass(), UniformChance()):
        assert isinstance(model, Model)
        assert model.family == "baseline"
        assert model.name
        assert model.n_params == 0


def test_floors_declare_from_scratch_regime() -> None:
    """Both floors declare the ``from_scratch`` regime (fit from scratch on the train prior)."""
    assert MajorityClass().regime.name is Regime.FROM_SCRATCH
    assert UniformChance().regime.name is Regime.FROM_SCRATCH


def test_embed_raises_not_implemented() -> None:
    """A constant/uniform predictor has no representation -- embed raises, per the contract."""
    train = _train_samples([0, 1])
    for model in (MajorityClass().fit(train), UniformChance().fit(train)):
        with pytest.raises(NotImplementedError):
            model.embed(_collated_batch(1))


# --------------------------------------------------------------------------------------------------
# MajorityClass semantics
# --------------------------------------------------------------------------------------------------
def test_majority_predicts_modal_class() -> None:
    """majority_class learns the most frequent train label and predicts it for every sample."""
    # class 2 appears 4x, classes 0/1 fewer -> modal class is 2.
    model = MajorityClass().fit(_train_samples([0, 1, 2, 2, 2, 2, 1]))
    scores = model.forward(_collated_batch(3))
    assert len(scores) == 3
    # one-hot at class 2, and argmax(row) == 2 for every sample.
    assert all(max(range(len(row)), key=row.__getitem__) == 2 for row in scores)
    assert all(row[2] == 1.0 for row in scores)


def test_majority_is_deterministic() -> None:
    """Two fits on the same data give identical predictions (no RNG anywhere)."""
    labels = [3, 3, 1, 0, 3, 2]
    a = MajorityClass().fit(_train_samples(labels)).forward(_collated_batch(2))
    b = MajorityClass().fit(_train_samples(labels)).forward(_collated_batch(2))
    assert a == b


def test_majority_ties_break_to_lowest_id() -> None:
    """A frequency tie resolves to the lowest class id (deterministic, RNG-free)."""
    # classes 0 and 1 each appear twice -> tie -> pick 0.
    model = MajorityClass().fit(_train_samples([0, 0, 1, 1]))
    scores = model.forward(_collated_batch(1))
    assert max(range(len(scores[0])), key=scores[0].__getitem__) == 0


def test_majority_forward_before_fit_raises() -> None:
    """forward before fit is a loud error rather than a silent wrong answer."""
    with pytest.raises(RuntimeError, match="before fit"):
        MajorityClass().forward(_collated_batch(1))


# --------------------------------------------------------------------------------------------------
# UniformChance semantics
# --------------------------------------------------------------------------------------------------
def test_chance_predictions_are_valid_classes() -> None:
    """chance draws valid class ids in [0, C) and returns one-hot vectors of width C."""
    model = UniformChance().fit(_train_samples([0, 1, 2, 3, 4]))  # C = 5
    scores = model.forward(_collated_batch(20))
    assert len(scores) == 20
    for row in scores:
        assert len(row) == 5
        assert sum(row) == 1.0  # exactly one class selected
        drawn = max(range(len(row)), key=row.__getitem__)
        assert 0 <= drawn < 5


def test_chance_is_seed_deterministic() -> None:
    """Same seed + same data -> identical draws (reproducible chance row)."""
    labels = [0, 1, 2, 3]
    a = UniformChance(seed=42).fit(_train_samples(labels)).forward(_collated_batch(30))
    b = UniformChance(seed=42).fit(_train_samples(labels)).forward(_collated_batch(30))
    assert a == b


def test_chance_different_seed_differs() -> None:
    """A different seed produces a different draw sequence (the RNG is actually used)."""
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    a = UniformChance(seed=42).fit(_train_samples(labels)).forward(_collated_batch(40))
    b = UniformChance(seed=7).fit(_train_samples(labels)).forward(_collated_batch(40))
    assert a != b


def test_chance_forward_before_fit_raises() -> None:
    """forward before fit is a loud error (the class count is unknown)."""
    with pytest.raises(RuntimeError, match="before fit"):
        UniformChance().forward(_collated_batch(1))


def test_empty_train_split_raises() -> None:
    """Fitting on an empty split fails loudly for both floors."""
    with pytest.raises(ValueError, match="empty train split"):
        MajorityClass().fit([])
    with pytest.raises(ValueError, match="empty train split"):
        UniformChance().fit([])


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() through the full harness (schema-valid, no numpy/torch)
# --------------------------------------------------------------------------------------------------
def _amc_task(labels: list[int]) -> AmcTask:
    """A synthetic AMC task whose single dataset replays the given labels as the test split."""
    samples = [
        {"iq": [[0.1, 0.2], [0.3, 0.4]], "label": lbl, "snr_db": (0 if i % 2 else 10)}
        for i, lbl in enumerate(labels)
    ]
    ds = AmcDataset("radioml_2016_10a", samples=samples, checksum=_CHECKSUM)
    return AmcTask(datasets=[ds])


def test_majority_end_to_end_scores_the_prior() -> None:
    """majority_class through evaluate() scores exactly the modal-class frequency, schema-valid."""
    labels = [0, 0, 0, 1, 2]  # modal class 0 appears 3/5 of the time
    task = _amc_task(labels)
    model = MajorityClass().fit(task.datasets()[0].load("train"))
    result = evaluate(model, task, "test", model.regime, batch_size=2, compute_bootstrap_ci=False)
    Draft202012Validator(_load_schema()).validate(result)
    assert result["regime"]["name"] == "from_scratch"
    # always predicts class 0 -> accuracy == fraction of class-0 samples == 3/5.
    assert result["metrics"]["values"]["accuracy_overall"] == pytest.approx(3 / 5)


def test_chance_end_to_end_runs_and_validates() -> None:
    """chance through evaluate() produces a schema-valid row with a plausible floor accuracy."""
    labels = [i % 4 for i in range(40)]  # 4 balanced classes
    task = _amc_task(labels)
    model = UniformChance(seed=42).fit(task.datasets()[0].load("train"))
    result = evaluate(model, task, "test", model.regime, batch_size=8, compute_bootstrap_ci=False)
    Draft202012Validator(_load_schema()).validate(result)
    acc = result["metrics"]["values"]["accuracy_overall"]
    # expected 1/4; realised fluctuates but must stay in a sane floor band under a real model.
    assert 0.0 <= acc <= 0.6
