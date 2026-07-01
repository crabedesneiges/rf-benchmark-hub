"""Name -> class registries for tasks, models and metrics.

One :class:`Registry` instance exists per kind (``TASKS``, ``MODELS``, ``METRICS``).
Registration is decorator-based; the registries are expected to be populated at import
time (M0) and treated as frozen thereafter. :meth:`Registry.get` returns the
registered *class*; the :func:`get_task` helper additionally instantiates it.

No third-party imports at module top so ``import rfbench.core`` stays dependency-free.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from rfbench.core.metric import Metric
from rfbench.core.model import Model
from rfbench.core.task import Task

T = TypeVar("T")


class Registry(dict[str, type]):
    """A ``name -> class`` registry, one instance per kind.

    Subclasses :class:`dict` so membership and iteration come for free. Classes are
    added via the :meth:`register` decorator and looked up with :meth:`get`.
    """

    def __init__(self, kind: str) -> None:
        """Create an empty registry labelled by ``kind`` (e.g. ``"task"``)."""
        super().__init__()
        self.kind = kind

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a class decorator registering ``cls`` under ``name``.

        The decorated class is returned unchanged. Re-registering an existing ``name``
        raises :class:`KeyError` to catch accidental collisions.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:  # type: ignore[override]
        """Return the class registered under ``name``.

        Raises :class:`KeyError` listing the available names if ``name`` is unknown.
        """
        try:
            return self[name]
        except KeyError:
            available = ", ".join(sorted(self)) or "<none>"
            raise KeyError(f"unknown {self.kind} '{name}'; available: {available}") from None

    def names(self) -> list[str]:
        """Return the sorted list of registered names."""
        return sorted(self)


#: Registry of :class:`~rfbench.core.task.Task` classes.
TASKS: Registry = Registry("task")
#: Registry of :class:`~rfbench.core.model.Model` classes.
MODELS: Registry = Registry("model")
#: Registry of :class:`~rfbench.core.metric.Metric` classes.
METRICS: Registry = Registry("metric")


def register_task(name: str) -> Callable[[type[Task]], type[Task]]:
    """Class decorator registering a :class:`Task` under ``name`` (sugar for ``TASKS``)."""
    return TASKS.register(name)


def register_model(name: str) -> Callable[[type[Model]], type[Model]]:
    """Class decorator registering a :class:`Model` under ``name`` (sugar for ``MODELS``)."""
    return MODELS.register(name)


def register_metric(name: str) -> Callable[[type[Metric]], type[Metric]]:
    """Class decorator registering a :class:`Metric` under ``name`` (sugar for ``METRICS``)."""
    return METRICS.register(name)


def get_task(name: str) -> Task:
    """Instantiate and return the :class:`Task` registered under ``name``."""
    cls = cast("type[Task]", TASKS.get(name))
    return cls()


__all__ = [
    "Registry",
    "TASKS",
    "MODELS",
    "METRICS",
    "register_task",
    "register_model",
    "register_metric",
    "get_task",
]
