"""
TrainingMonitor — attach PyTorch hooks with zero memory leaks.

This class registers **forward hooks** (activations) and **backward hooks**
(gradients) on every layer you care about.  Every captured tensor is
immediately ``.detach().cpu().numpy()``'d so that:

1. The autograd graph is never held alive by stale references.
2. Nothing stays on the MPS / CUDA device after the hook returns.
3. The downstream ``MonitorDataStore`` only ever sees plain NumPy arrays.

Non-blocking architecture
-------------------------
Hooks only do the **cheap** part synchronously (detach → cpu → numpy).
The heavier work — computing statistics and writing to the locked store
— is offloaded to a background **consumer thread** via a bounded queue.
This means the forward/backward pass is *never* blocked by the store's
``RLock`` or by NumPy histogram computation, keeping the UI responsive
and training throughput high.

Usage
-----
>>> store   = MonitorDataStore()
>>> monitor = TrainingMonitor(model, store)
>>> monitor.attach()              # registers hooks + starts consumer
>>> ...                           # run your training loop
>>> monitor.detach()              # removes hooks + drains queue cleanly
"""

from __future__ import annotations

import functools
import queue
import threading
from typing import Dict, List, Optional, Set, Sequence

import torch
import torch.nn as nn
import numpy as np

from .data_store import MonitorDataStore, StepMetrics

# Sentinel value that tells the consumer thread to shut down.
_STOP = object()


class TrainingMonitor:
    """
    Registers forward & backward hooks and funnels data into a
    ``MonitorDataStore`` via a **non-blocking queue**.

    Parameters
    ----------
    model : nn.Module
        The network to instrument.
    store : MonitorDataStore
        Thread-safe store that the visualization reads from.
    track_layers : set[str] | None
        Explicit set of ``named_modules()`` keys to track.
        ``None`` ⇒ track every leaf module (Conv, Linear, BN, …).
    sample_every : int
        Record activations / gradients every *n* steps to reduce overhead.
        1 = every step (useful for small models / debugging).
    max_activation_elements : int
        If an activation has more elements than this, downsample before
        storing.  Keeps memory bounded for large feature maps.
    queue_size : int
        Maximum items in the async queue.  If the consumer can't keep up
        the oldest items are silently dropped (back-pressure) so that
        training throughput is never degraded.
    """

    def __init__(
        self,
        model: nn.Module,
        store: MonitorDataStore,
        *,
        track_layers: Optional[Set[str]] = None,
        sample_every: int = 1,
        max_activation_elements: int = 50_000,
        queue_size: int = 512,
    ) -> None:
        self._model = model
        self._store = store
        self._track_layers = track_layers
        self._sample_every = sample_every
        self._max_elements = max_activation_elements

        self._step: int = 0
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._attached: bool = False

        # ── async queue + consumer ────────────────────────────────────
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._consumer_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ public

    def attach(self) -> "TrainingMonitor":
        """Register forward + backward hooks on the target layers and
        start the background consumer thread."""
        if self._attached:
            return self

        # Start the async consumer BEFORE hooks begin firing.
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="monitor-consumer",
        )
        self._consumer_thread.start()

        for name, module in self._model.named_modules():
            if not self._should_track(name, module):
                continue

            # --- forward hook (activations) ---
            fwd = functools.partial(self._forward_hook, layer_name=name)
            self._hooks.append(module.register_forward_hook(fwd))

            # --- backward hook (gradients of the *output*) ---
            bwd = functools.partial(self._backward_hook, layer_name=name)
            self._hooks.append(
                module.register_full_backward_hook(bwd)
            )

        self._attached = True
        self._store.set_training_state(True)
        return self

    def detach(self) -> None:
        """Remove every registered hook, drain the queue, and stop the
        consumer — safe to call multiple times."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False

        # Signal the consumer to finish and wait for it.
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            # Queue is full — forcibly drain one item and retry.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(_STOP)

        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=5.0)
            self._consumer_thread = None

        self._store.set_training_state(False)

    def on_step_end(
        self,
        loss: float,
        lr: Optional[float] = None,
        epoch: Optional[int] = None,
        **extras: float,
    ) -> None:
        """
        Call this once per optimiser step to record scalar metrics
        and advance the internal step counter.

        Parameters
        ----------
        loss  : float   Current batch loss (already a Python float).
        lr    : float   Current learning rate (optional).
        epoch : int     Current epoch number (optional).
        extras: dict    Any extra scalars you want on the dashboard.
        """
        self._step += 1
        self._store.record_step(
            StepMetrics(
                step=self._step,
                loss=loss,
                lr=lr,
                epoch=epoch,
                extras=extras,
            )
        )

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------ hooks

    def _forward_hook(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
        *,
        layer_name: str,
    ) -> None:
        """Capture activation — detach immediately, then enqueue."""
        if self._step % self._sample_every != 0:
            return

        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            return

        # ⚠️ Critical: detach → cpu → numpy.  Nothing stays on device.
        arr: np.ndarray = tensor.detach().cpu().numpy()

        # Down-sample if the activation is enormous.
        if arr.size > self._max_elements:
            idx = np.random.default_rng(seed=self._step).choice(
                arr.size, self._max_elements, replace=False
            )
            arr = arr.ravel()[idx]

        # Non-blocking enqueue — drop if queue is full (back-pressure).
        try:
            self._queue.put_nowait(("activation", layer_name, self._step, arr))
        except queue.Full:
            pass  # silently drop to keep training throughput

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
        *,
        layer_name: str,
    ) -> None:
        """Capture gradient — detach immediately, then enqueue."""
        if self._step % self._sample_every != 0:
            return

        grad = None
        for g in grad_output:
            if g is not None:
                grad = g
                break
        if grad is None:
            return

        # ⚠️ Critical: detach → cpu → numpy.
        arr: np.ndarray = grad.detach().cpu().numpy()

        if arr.size > self._max_elements:
            idx = np.random.default_rng(seed=self._step).choice(
                arr.size, self._max_elements, replace=False
            )
            arr = arr.ravel()[idx]

        try:
            self._queue.put_nowait(("gradient", layer_name, self._step, arr))
        except queue.Full:
            pass

    # ------------------------------------------------- consumer thread

    def _consumer_loop(self) -> None:
        """Background thread that drains the queue into the store.

        All heavy work (histogram computation, lock acquisition) happens
        here so the training thread's hooks are never blocked.
        """
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                # Drain any remaining items before exiting.
                while not self._queue.empty():
                    remaining = self._queue.get_nowait()
                    if remaining is _STOP:
                        continue
                    self._process_item(remaining)
                break

            self._process_item(item)

    def _process_item(self, item: tuple) -> None:
        """Dispatch a single queue item to the store."""
        kind = item[0]
        if kind == "activation":
            _, layer_name, step, arr = item
            self._store.record_activation(layer_name, step, arr)
        elif kind == "gradient":
            _, layer_name, step, arr = item
            self._store.record_gradient(layer_name, step, arr)

    # ------------------------------------------------------------------ helpers

    def _should_track(self, name: str, module: nn.Module) -> bool:
        """Decide if a named module should be instrumented."""
        if self._track_layers is not None:
            return name in self._track_layers
        # Default: track only *leaf* modules (the ones that do real work).
        return len(list(module.children())) == 0 and name != ""

    # ------------------------------------------------------------------ context manager

    def __enter__(self) -> "TrainingMonitor":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()
