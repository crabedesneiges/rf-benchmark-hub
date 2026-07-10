"""SNR-estimation (raw-IQ -> SNR in dB regression) task package (J4).

Importing this package registers
:class:`~rfbench.tasks.snr_estimation.task.SnrEstimationTask` under the name
``"snr_estimation"`` in ``rfbench.core.registry.TASKS`` (registration is a side effect of
importing ``.task``), so ``rfbench.core.registry.get_task("snr_estimation")`` resolves it.

Re-exports the public SNR surface: the task, the dataset adapter and the two regression
metrics. ``import rfbench.tasks.snr_estimation`` stays dependency-free -- stdlib + the frozen
core contracts only; numpy is imported lazily inside the dataset loaders.
"""

from __future__ import annotations

from rfbench.tasks.snr_estimation.dataset import SnrDataset
from rfbench.tasks.snr_estimation.metrics import Mae, Rmse
from rfbench.tasks.snr_estimation.task import SNR_DATASET_NAMES, SnrEstimationTask

__all__ = [
    "SnrEstimationTask",
    "SNR_DATASET_NAMES",
    "SnrDataset",
    "Rmse",
    "Mae",
]
