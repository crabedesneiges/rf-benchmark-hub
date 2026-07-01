"""Contract tests for ``rfbench.core`` (WP-02 acceptance).

These tests assert the frozen-contract guarantees without any third-party dependency
beyond ``pytest``:

* ``import rfbench`` and ``import rfbench.core`` both succeed (and stay dependency-free);
* each ABC is non-instantiable (direct instantiation raises ``TypeError``);
* every ABC declares the expected ``@abstractmethod`` set;
* the ``Regime`` enum and ``RegimeSpec`` coupling match the JSON schema;
* the registries register/get/raise as specified.
"""

from __future__ import annotations

import abc
from typing import Any

import pytest

from rfbench.core.types import Tensor


def test_import_rfbench() -> None:
    """``import rfbench`` succeeds."""
    import rfbench  # noqa: F401


def test_import_rfbench_core() -> None:
    """``import rfbench.core`` succeeds."""
    import rfbench.core  # noqa: F401


def test_core_import_is_dependency_free() -> None:
    """Importing the core must not pull in torch or jsonschema."""
    import importlib
    import sys

    for heavy in ("torch", "jsonschema"):
        sys.modules.pop(heavy, None)
    importlib.reload(importlib.import_module("rfbench.core"))
    assert "torch" not in sys.modules
    assert "jsonschema" not in sys.modules


# --- ABCs are non-instantiable ------------------------------------------------------

ABC_NAMES = ["Task", "Dataset", "Metric", "Model"]


@pytest.mark.parametrize("abc_name", ABC_NAMES)
def test_abc_is_abstract_base(abc_name: str) -> None:
    """Each contract subclasses ``abc.ABC`` and carries abstract methods."""
    import rfbench.core as core

    cls = getattr(core, abc_name)
    assert issubclass(cls, abc.ABC)
    assert getattr(cls, "__abstractmethods__", frozenset())


@pytest.mark.parametrize("abc_name", ABC_NAMES)
def test_abc_direct_instantiation_raises_typeerror(abc_name: str) -> None:
    """Instantiating an ABC directly raises ``TypeError``."""
    import rfbench.core as core

    cls = getattr(core, abc_name)
    with pytest.raises(TypeError):
        cls()


# --- @abstractmethod coverage -------------------------------------------------------

EXPECTED_ABSTRACTS = {
    "Task": {"datasets", "metrics", "default_split", "tracks", "build_targets"},
    "Dataset": {"download", "prepare", "load"},
    "Metric": {"reset", "update", "compute"},
    "Model": {"forward", "embed", "n_params"},
}


@pytest.mark.parametrize("abc_name, expected", sorted(EXPECTED_ABSTRACTS.items()))
def test_abstractmethod_coverage(abc_name: str, expected: set[str]) -> None:
    """Each ABC declares exactly the expected set of abstract methods."""
    import rfbench.core as core

    cls = getattr(core, abc_name)
    assert set(cls.__abstractmethods__) == expected


def test_partial_subclass_still_abstract() -> None:
    """A subclass leaving one abstract method unimplemented stays non-instantiable."""
    import rfbench.core as core

    class HalfMetric(core.Metric):
        name = "half"
        primary_key = "x"

        def reset(self) -> None:  # pragma: no cover - body irrelevant
            pass

        def update(self, pred: Tensor, target: Tensor, meta: dict[str, Any] | None = None) -> None:
            pass

        # compute() deliberately not implemented

    with pytest.raises(TypeError):
        HalfMetric()  # type: ignore[abstract]


def test_full_subclass_is_instantiable() -> None:
    """Implementing every abstract method makes a subclass instantiable."""
    import rfbench.core as core

    class DummyMetric(core.Metric):
        name = "dummy"
        primary_key = "accuracy_overall"

        def reset(self) -> None:
            pass

        def update(self, pred: Tensor, target: Tensor, meta: dict[str, Any] | None = None) -> None:
            pass

        def compute(self) -> dict[str, float | list[dict[str, float]]]:
            return {"accuracy_overall": 1.0}

    metric = DummyMetric()
    assert metric.primary_key in metric.compute()


# --- Regime enum + RegimeSpec coupling (mirrors result.schema.json) ------------------


def test_regime_enum_values_match_schema() -> None:
    """The ``Regime`` enum values match the ``regime.name`` schema enum verbatim."""
    from rfbench.core import Regime

    assert {r.value for r in Regime} == {
        "from_scratch",
        "full_finetune",
        "linear_probe",
        "few_shot",
    }
    # str-Enum: the value serialises straight into result.json.
    assert Regime.FEW_SHOT.value == "few_shot"
    assert isinstance(Regime.FEW_SHOT, str)


def test_regime_spec_few_shot_requires_k_shot() -> None:
    """``k_shot`` must be set iff the regime is ``few_shot`` (schema allOf)."""
    from rfbench.core import Regime, RegimeSpec

    RegimeSpec(Regime.FEW_SHOT, k_shot=5)  # ok
    RegimeSpec(Regime.LINEAR_PROBE)  # ok

    with pytest.raises(ValueError):
        RegimeSpec(Regime.FEW_SHOT)  # missing k_shot
    with pytest.raises(ValueError):
        RegimeSpec(Regime.LINEAR_PROBE, k_shot=1)  # forbidden k_shot
    with pytest.raises(ValueError):
        RegimeSpec(Regime.FEW_SHOT, k_shot=0)  # k_shot must be >= 1


def test_regime_spec_is_frozen() -> None:
    """``RegimeSpec`` is an immutable value object."""
    import dataclasses

    from rfbench.core import Regime, RegimeSpec

    spec = RegimeSpec(Regime.LINEAR_PROBE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = Regime.FROM_SCRATCH  # type: ignore[misc]


# --- Registries ---------------------------------------------------------------------


def test_registry_register_get_names() -> None:
    """A registry registers a class, returns it unchanged, and looks it up by name."""
    from rfbench.core import Registry

    reg = Registry("thing")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
    assert reg.names() == ["foo"]


def test_registry_unknown_name_raises_keyerror() -> None:
    """Looking up an unregistered name raises ``KeyError`` with the available names."""
    from rfbench.core import Registry

    reg = Registry("thing")
    with pytest.raises(KeyError):
        reg.get("missing")


def test_registry_duplicate_registration_raises() -> None:
    """Re-registering the same name raises to catch accidental collisions."""
    from rfbench.core import Registry

    reg = Registry("thing")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(KeyError):

        @reg.register("dup")
        class B:
            pass


def test_module_registries_exist() -> None:
    """The three canonical registries exist and are distinct instances."""
    from rfbench.core import METRICS, MODELS, TASKS, Registry

    for reg in (TASKS, MODELS, METRICS):
        assert isinstance(reg, Registry)
    assert len({id(TASKS), id(MODELS), id(METRICS)}) == 3
