"""RF-Benchmark-Hub: reproducible benchmarks + leaderboard harness for terrestrial RF ML tasks.

This top-level module is intentionally lightweight: it exposes only ``__version__`` and pulls in
**no** third-party or heavy dependencies at import time, so ``import rfbench`` succeeds on a bare
Python install with zero extras. The frozen contracts live in :mod:`rfbench.core` and are imported
explicitly by callers that need them (e.g. ``from rfbench.core import Task``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
