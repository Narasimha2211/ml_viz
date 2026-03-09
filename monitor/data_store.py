"""
Thread-safe data store for training metrics.

This module is the single source of truth shared between the training loop
(writer) and the visualization dashboard (reader).  Every public method
acquires an ``RLock`` so the UI thread can safely read snapshots while
the training thread writes new data.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Small value-objects that travel between the store and the dashboard
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivationSnapshot:
    """One captured activation tensor (already on CPU as a NumPy array)."""

    layer_name: str
    step: int
    shape: tuple
    mean: float
    std: float
    abs_max: float
    histogram: np.ndarray  # pre-computed histogram counts


@dataclass(frozen=True)
class GradientSnapshot:
    """One captured gradient tensor (already on CPU as a NumPy array)."""

    layer_name: str
    step: int
    shape: tuple
    mean: float
    std: float
    abs_max: float
    norm: float
    histogram: np.ndarray


@dataclass
class StepMetrics:
    """Aggregated scalars for a single training step."""

    step: int
    loss: float
    lr: Optional[float] = None
    epoch: Optional[int] = None
    wall_time: float = field(default_factory=time.time)
    extras: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# The thread-safe store itself
# ──────────────────────────────────────────────────────────────────────

_HIST_BINS = 64  # default histogram resolution


class MonitorDataStore:
    """
    A thread-safe, bounded ring-buffer store for training telemetry.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps to keep in memory.  Older entries are
        discarded automatically so the dashboard never forces an OOM.
    hist_bins : int
        Number of bins used when pre-computing histograms for
        activations and gradients.
    """

    def __init__(self, max_steps: int = 2000, hist_bins: int = _HIST_BINS) -> None:
        self._lock = threading.RLock()
        self._max_steps = max_steps
        self._hist_bins = hist_bins

        # ---- scalar time-series ----
        self._step_metrics: List[StepMetrics] = []

        # ---- per-layer ring buffers ----
        # key = layer_name, value = list[Snapshot]
        self._activations: Dict[str, List[ActivationSnapshot]] = defaultdict(list)
        self._gradients: Dict[str, List[GradientSnapshot]] = defaultdict(list)

        # ---- bookkeeping ----
        self._current_step: int = 0
        self._is_training: bool = False
        self._dirty = threading.Event()  # signalled when new data arrives

    # ------------------------------------------------------------------ writers

    def record_step(self, metrics: StepMetrics) -> None:
        """Append scalar metrics for one optimiser step."""
        with self._lock:
            self._step_metrics.append(metrics)
            if len(self._step_metrics) > self._max_steps:
                self._step_metrics = self._step_metrics[-self._max_steps :]
            self._current_step = metrics.step
            self._dirty.set()

    def record_activation(
        self, layer_name: str, step: int, tensor_cpu: "np.ndarray"
    ) -> None:
        """
        Record a pre-processed activation snapshot.

        ``tensor_cpu`` must already live on CPU (NumPy).  The store
        computes lightweight statistics so downstream readers never
        touch the raw tensor.
        """
        hist, _ = np.histogram(tensor_cpu.ravel(), bins=self._hist_bins)
        snap = ActivationSnapshot(
            layer_name=layer_name,
            step=step,
            shape=tuple(tensor_cpu.shape),
            mean=float(tensor_cpu.mean()),
            std=float(tensor_cpu.std()),
            abs_max=float(np.abs(tensor_cpu).max()),
            histogram=hist,
        )
        with self._lock:
            buf = self._activations[layer_name]
            buf.append(snap)
            if len(buf) > self._max_steps:
                self._activations[layer_name] = buf[-self._max_steps :]
            self._dirty.set()

    def record_gradient(
        self, layer_name: str, step: int, tensor_cpu: "np.ndarray"
    ) -> None:
        """Record a pre-processed gradient snapshot."""
        hist, _ = np.histogram(tensor_cpu.ravel(), bins=self._hist_bins)
        snap = GradientSnapshot(
            layer_name=layer_name,
            step=step,
            shape=tuple(tensor_cpu.shape),
            mean=float(tensor_cpu.mean()),
            std=float(tensor_cpu.std()),
            abs_max=float(np.abs(tensor_cpu).max()),
            norm=float(np.linalg.norm(tensor_cpu.ravel())),
            histogram=hist,
        )
        with self._lock:
            buf = self._gradients[layer_name]
            buf.append(snap)
            if len(buf) > self._max_steps:
                self._gradients[layer_name] = buf[-self._max_steps :]
            self._dirty.set()

    def set_training_state(self, is_training: bool) -> None:
        with self._lock:
            self._is_training = is_training
            self._dirty.set()

    # ------------------------------------------------------------------ readers

    def wait_for_update(self, timeout: float = 0.1) -> bool:
        """Block until new data arrives (or *timeout* seconds elapse)."""
        triggered = self._dirty.wait(timeout=timeout)
        self._dirty.clear()
        return triggered

    def get_loss_curve(self) -> tuple[List[int], List[float]]:
        """Return ``(steps, losses)`` lists for the loss chart."""
        with self._lock:
            steps = [m.step for m in self._step_metrics]
            losses = [m.loss for m in self._step_metrics]
        return steps, losses

    def get_lr_curve(self) -> tuple[List[int], List[float]]:
        with self._lock:
            steps = [m.step for m in self._step_metrics if m.lr is not None]
            lrs = [m.lr for m in self._step_metrics if m.lr is not None]
        return steps, lrs

    def get_activation_stats(
        self, layer_name: str
    ) -> tuple[List[int], List[float], List[float]]:
        """Return ``(steps, means, stds)`` for one layer's activations."""
        with self._lock:
            snaps = list(self._activations.get(layer_name, []))
        steps = [s.step for s in snaps]
        means = [s.mean for s in snaps]
        stds = [s.std for s in snaps]
        return steps, means, stds

    def get_gradient_norms(self) -> Dict[str, tuple[List[int], List[float]]]:
        """Return ``{layer: (steps, norms)}`` for every tracked layer."""
        with self._lock:
            result: Dict[str, Any] = {}
            for name, snaps in self._gradients.items():
                result[name] = (
                    [s.step for s in snaps],
                    [s.norm for s in snaps],
                )
        return result

    def get_latest_histograms(
        self,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Return the most-recent activation and gradient histograms.

        Returns
        -------
        dict with keys ``"activations"`` and ``"gradients"``, each mapping
        layer name → histogram ``np.ndarray``.
        """
        with self._lock:
            acts: Dict[str, np.ndarray] = {
                name: snaps[-1].histogram
                for name, snaps in self._activations.items()
                if snaps
            }
            grads: Dict[str, np.ndarray] = {
                name: snaps[-1].histogram
                for name, snaps in self._gradients.items()
                if snaps
            }
        return {"activations": acts, "gradients": grads}

    def get_layer_names(self) -> List[str]:
        with self._lock:
            return sorted(
                set(self._activations.keys()) | set(self._gradients.keys())
            )

    # ---- new readers for Streamlit dashboard ----------------------------

    def get_activation_heatmap_data(
        self, layer_name: str
    ) -> tuple[List[int], "np.ndarray"]:
        """Return ``(steps, heatmap_2d)`` where each row is a histogram snapshot.

        Shape of *heatmap_2d*: ``(n_steps, hist_bins)``.
        """
        with self._lock:
            snaps = list(self._activations.get(layer_name, []))
        if not snaps:
            return [], np.empty((0, 0))
        steps = [s.step for s in snaps]
        histograms = np.stack([s.histogram.astype(float) for s in snaps])
        return steps, histograms

    def get_gradient_histogram_series(
        self, layer_name: str
    ) -> tuple[List[int], "np.ndarray"]:
        """Return ``(steps, heatmap_2d)`` of gradient histograms over time."""
        with self._lock:
            snaps = list(self._gradients.get(layer_name, []))
        if not snaps:
            return [], np.empty((0, 0))
        steps = [s.step for s in snaps]
        histograms = np.stack([s.histogram.astype(float) for s in snaps])
        return steps, histograms

    def get_latest_metrics(self) -> Optional[StepMetrics]:
        """Return the most recent ``StepMetrics``, or *None*."""
        with self._lock:
            if self._step_metrics:
                return self._step_metrics[-1]
            return None

    def get_accuracy_curve(self) -> tuple[List[int], List[float]]:
        """Return ``(steps, accuracies)`` extracted from step extras."""
        with self._lock:
            steps: List[int] = []
            accs: List[float] = []
            for m in self._step_metrics:
                if "accuracy" in m.extras:
                    steps.append(m.step)
                    accs.append(m.extras["accuracy"])
        return steps, accs

    @property
    def current_step(self) -> int:
        with self._lock:
            return self._current_step

    @property
    def is_training(self) -> bool:
        with self._lock:
            return self._is_training
