"""
LiveDashboard — a Matplotlib-based real-time training visualizer.

Runs in its **own thread** and polls the ``MonitorDataStore`` for new
data.  The training loop never blocks on rendering, and the dashboard
never touches PyTorch tensors (everything is plain NumPy / Python
scalars by the time it arrives here).

Layout (2 × 2 grid)
--------------------
┌──────────────┬──────────────┐
│  Loss curve  │  LR schedule │
├──────────────┼──────────────┤
│  Grad norms  │  Act. stats  │
└──────────────┴──────────────┘

The dashboard automatically discovers which layers are being tracked
and adds legend entries dynamically.
"""

from __future__ import annotations

import threading
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")  # non-blocking backend that works well on macOS
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import matplotlib.animation as animation
import matplotlib.artist as mpl_artist
import numpy as np

from monitor.data_store import MonitorDataStore


class LiveDashboard:
    """
    Real-time Matplotlib dashboard that reads from a ``MonitorDataStore``.

    Parameters
    ----------
    store : MonitorDataStore
        The shared, thread-safe data store.
    refresh_ms : int
        How often (in milliseconds) to repaint the plots.
    figsize : tuple[int, int]
        Matplotlib figure size in inches.
    """

    def __init__(
        self,
        store: MonitorDataStore,
        *,
        refresh_ms: int = 250,
        figsize: tuple[int, int] = (14, 8),
    ) -> None:
        self._store = store
        self._refresh_ms = refresh_ms
        self._figsize = figsize

        self._fig: Optional[mpl_figure.Figure] = None
        self._axes = {}
        self._anim: Optional[animation.FuncAnimation] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ public

    def launch_in_thread(self) -> threading.Thread:
        """
        Start the dashboard on a **daemon thread** so it shuts down
        automatically when the training script exits.
        """
        self._thread = threading.Thread(
            target=self._run_mainloop, daemon=True, name="viz-dashboard"
        )
        self._thread.start()
        return self._thread

    def launch_blocking(self) -> None:
        """Run the dashboard on the *current* thread (blocks)."""
        self._run_mainloop()

    # ------------------------------------------------------------------ internals

    def _run_mainloop(self) -> None:
        self._fig, axs = plt.subplots(2, 2, figsize=self._figsize)
        self._fig.suptitle("🔬  Training Monitor", fontsize=14, fontweight="bold")
        manager = self._fig.canvas.manager
        if manager is not None:
            manager.set_window_title("ML Training Monitor")

        self._axes = {
            "loss": axs[0, 0],
            "lr": axs[0, 1],
            "grad_norms": axs[1, 0],
            "act_stats": axs[1, 1],
        }

        for key, ax in self._axes.items():
            ax.set_title(key.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        self._fig.tight_layout(rect=(0, 0, 1, 0.95))

        self._anim = animation.FuncAnimation(
            self._fig,
            self._update_frame,
            interval=self._refresh_ms,
            cache_frame_data=False,
        )
        plt.show()

    def _update_frame(self, _frame: int) -> list[mpl_artist.Artist]:
        """Called by FuncAnimation on the GUI thread."""
        self._draw_loss()
        self._draw_lr()
        self._draw_gradient_norms()
        self._draw_activation_stats()
        return []

    # ---- individual panels ------------------------------------------------

    def _draw_loss(self) -> None:
        ax = self._axes["loss"]
        steps, losses = self._store.get_loss_curve()
        if not steps:
            return
        ax.clear()
        ax.set_title("Loss")
        ax.grid(True, alpha=0.3)
        ax.plot(steps, losses, color="#e74c3c", linewidth=1.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        # Show a smoothed line when we have enough data
        if len(losses) > 20:
            window = max(5, len(losses) // 20)
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            offset = len(steps) - len(smoothed)
            ax.plot(
                steps[offset:],
                smoothed,
                color="#2c3e50",
                linewidth=2,
                label=f"EMA({window})",
            )
            ax.legend(loc="upper right", fontsize=8)

    def _draw_lr(self) -> None:
        ax = self._axes["lr"]
        steps, lrs = self._store.get_lr_curve()
        if not steps:
            ax.set_title("LR Schedule  (no LR logged)")
            return
        ax.clear()
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)
        ax.plot(steps, lrs, color="#3498db", linewidth=1.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    def _draw_gradient_norms(self) -> None:
        ax = self._axes["grad_norms"]
        grad_data = self._store.get_gradient_norms()
        if not grad_data:
            return
        ax.clear()
        ax.set_title("Gradient L2 Norms")
        ax.grid(True, alpha=0.3)
        cmap = matplotlib.colormaps["tab10"]
        for i, (name, (steps, norms)) in enumerate(sorted(grad_data.items())):
            short = name.rsplit(".", 1)[-1] if "." in name else name
            ax.plot(
                steps,
                norms,
                color=cmap(i % 10),
                linewidth=1,
                label=short,
                alpha=0.8,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("L2 Norm")
        ax.legend(loc="upper right", fontsize=6, ncol=2)

    def _draw_activation_stats(self) -> None:
        ax = self._axes["act_stats"]
        layers = self._store.get_layer_names()
        if not layers:
            return
        ax.clear()
        ax.set_title("Activation Mean ± Std")
        ax.grid(True, alpha=0.3)
        cmap = matplotlib.colormaps["Set2"]
        for i, name in enumerate(layers):
            steps, means, stds = self._store.get_activation_stats(name)
            if not steps:
                continue
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            short = name.rsplit(".", 1)[-1] if "." in name else name
            color = cmap(i % 8)
            ax.plot(steps, means_arr, color=color, linewidth=1, label=short)
            ax.fill_between(
                steps,
                means_arr - stds_arr,
                means_arr + stds_arr,
                color=color,
                alpha=0.15,
            )
        ax.set_xlabel("Step")
        ax.legend(loc="upper right", fontsize=6, ncol=2)
