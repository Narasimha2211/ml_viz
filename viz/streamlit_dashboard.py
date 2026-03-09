"""
Streamlit Dashboard — monochrome real-time neural network training monitor.

Launch
------
    cd ml_viz
    streamlit run viz/streamlit_dashboard.py

Architecture
------------
The app keeps a ``MonitorDataStore`` in ``st.session_state``.  When the
user clicks **Start Training**, a daemon thread trains a DemoCNN on
CIFAR-10.  Forward / backward hooks funnel activations and gradients
into the store, and the dashboard polls it every refresh cycle via
``st.rerun()``.

All charts use Plotly with a strict **black-and-white** palette.
"""

from __future__ import annotations

import sys
import os
import time
import threading
from pathlib import Path

import numpy as np

# ── Project root on sys.path so ``monitor`` / ``models`` resolve ────
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════
#  Page config — MUST be the first Streamlit call
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ Training Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from monitor import TrainingMonitor, MonitorDataStore
from models.mnist_cnn import MnistCNN


# ═══════════════════════════════════════════════════════════════════════
#  Black & White Theme
# ═══════════════════════════════════════════════════════════════════════
_BG = "#0a0a0a"
_CARD = "#111111"
_BORDER = "#222222"
_GRID = "#1e1e1e"
_TEXT = "#d4d4d4"
_WHITE = "#ffffff"
_DIM = "#666666"

# Grayscale colour-scale for heatmaps (dark → bright)
_BW_SCALE = [
    [0.0, "#0a0a0a"],
    [0.15, "#1c1c1c"],
    [0.35, "#3d3d3d"],
    [0.55, "#6e6e6e"],
    [0.75, "#a3a3a3"],
    [1.0, "#ffffff"],
]

_CUSTOM_CSS = """
<style>
/* ── global ─────────────────────────────────────────────── */
section.main > div { padding-top: 1.5rem; }
[data-testid="stAppViewContainer"] { background: %(bg)s; }
[data-testid="stHeader"]          { background: rgba(0,0,0,0); }
[data-testid="stSidebar"]         { background: #0f0f0f; border-right: 1px solid #1a1a1a; }

/* ── metric cards ───────────────────────────────────────── */
[data-testid="stMetricValue"]  { color: %(white)s; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"]  { color: %(dim)s; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.7rem; }
[data-testid="stMetricDelta"]  { font-family: 'JetBrains Mono', monospace; }

/* ── section dividers ───────────────────────────────────── */
.section-hdr {
    font-size: 0.75rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: %(dim)s; border-bottom: 1px solid #1a1a1a;
    padding-bottom: 0.4rem; margin: 2rem 0 1rem 0;
}

/* ── math card ──────────────────────────────────────────── */
.math-card {
    background: #111111; border: 1px solid #1f1f1f; border-radius: 8px;
    padding: 1.2rem 1.4rem; height: 100%%;
}
.math-card h4 { margin: 0 0 0.6rem 0; color: %(white)s; font-size: 0.95rem; }
.math-card p  { color: %(text)s; font-size: 0.82rem; line-height: 1.55; }

/* ── status badge ───────────────────────────────────────── */
.status-training { color: #ffffff; animation: pulse 1.8s infinite; }
.status-idle     { color: #555555; }
@keyframes pulse { 0%%,100%% {opacity:1} 50%% {opacity:0.4} }
</style>
""" % {"bg": _BG, "white": _WHITE, "dim": _DIM, "text": _TEXT}

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Plotly helpers
# ═══════════════════════════════════════════════════════════════════════

def _layout(**kw) -> go.Layout:
    """Return a Plotly Layout matching the B&W theme."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT, family="Inter, system-ui, sans-serif", size=11),
        margin=dict(l=48, r=16, t=36, b=36),
        xaxis=dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            showgrid=True, gridwidth=1,
        ),
        legend=dict(font=dict(size=10, color=_DIM)),
    )
    base.update(kw)
    return go.Layout(**base)


# ═══════════════════════════════════════════════════════════════════════
#  Training worker (runs on a daemon thread)
# ═══════════════════════════════════════════════════════════════════════

def _batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()


def _train_worker(
    model: nn.Module,
    monitor: TrainingMonitor,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    epochs: int,
) -> None:
    model.train()
    for epoch in range(1, epochs + 1):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            monitor.on_step_end(
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch,
                accuracy=_batch_accuracy(output, target),
            )
        scheduler.step()
    monitor.detach()


# ═══════════════════════════════════════════════════════════════════════
#  Chart builders
# ═══════════════════════════════════════════════════════════════════════

def build_loss_chart(store: MonitorDataStore) -> go.Figure:
    """Dynamic loss line chart with smoothed EMA overlay."""
    steps, losses = store.get_loss_curve()
    fig = go.Figure(layout=_layout(title="Loss"))
    if not steps:
        fig.add_annotation(
            text="Waiting for data …", showarrow=False,
            font=dict(color=_DIM, size=14), xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    # Raw loss — thin white line
    fig.add_trace(go.Scatter(
        x=steps, y=losses, mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", width=1),
        name="raw",
    ))

    # Smoothed EMA — bright white
    if len(losses) > 20:
        window = max(5, len(losses) // 20)
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
        offset = len(steps) - len(smoothed)
        fig.add_trace(go.Scatter(
            x=steps[offset:], y=smoothed.tolist(), mode="lines",
            line=dict(color=_WHITE, width=2),
            name=f"EMA({window})",
        ))

    fig.update_layout(
        xaxis_title="Step", yaxis_title="Loss",
        legend=dict(x=1, y=1, xanchor="right"),
    )
    return fig


def build_activation_heatmap(store: MonitorDataStore, layer_name: str) -> go.Figure:
    """Animated heatmap of activation distributions over training steps."""
    steps, heatmap = store.get_activation_heatmap_data(layer_name)
    short = layer_name.rsplit(".", 1)[-1] if "." in layer_name else layer_name
    fig = go.Figure(layout=_layout(title=f"{short}"))

    if not steps or heatmap.size == 0:
        fig.add_annotation(
            text="No data", showarrow=False,
            font=dict(color=_DIM, size=12), xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    # Normalise each row so distribution shape is visible
    row_sums = heatmap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normed = heatmap / row_sums

    fig.add_trace(go.Heatmap(
        z=normed.T,
        x=steps,
        colorscale=_BW_SCALE,
        showscale=False,
    ))
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Bin",
        yaxis=dict(gridcolor=_GRID, showgrid=False),
    )
    return fig


def build_gradient_histograms(store: MonitorDataStore) -> go.Figure:
    """Overlay the latest gradient histogram for every tracked layer."""
    histograms = store.get_latest_histograms()
    grad_hists: dict[str, np.ndarray] = histograms.get("gradients", {})

    fig = go.Figure(layout=_layout(title="Gradient Distributions (latest step)"))

    if not grad_hists:
        fig.add_annotation(
            text="Waiting for data …", showarrow=False,
            font=dict(color=_DIM, size=14), xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    n_layers = len(grad_hists)
    grays = [f"rgba(255,255,255,{0.25 + 0.6 * i / max(n_layers - 1, 1)})"
             for i in range(n_layers)]

    for idx, (name, hist_counts) in enumerate(sorted(grad_hists.items())):
        short = name.rsplit(".", 1)[-1] if "." in name else name
        bins = np.arange(len(hist_counts))
        # Normalise to relative frequency
        total = hist_counts.sum()
        freq = hist_counts / total if total > 0 else hist_counts

        fig.add_trace(go.Bar(
            x=bins, y=freq,
            name=short,
            marker_color=grays[idx],
            marker_line_width=0,
            opacity=0.85,
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Bin index",
        yaxis_title="Relative frequency",
        legend=dict(x=1, y=1, xanchor="right"),
    )
    return fig


def build_gradient_norms(store: MonitorDataStore) -> go.Figure:
    """Line chart of per-layer gradient L2 norms over time."""
    grad_data = store.get_gradient_norms()
    fig = go.Figure(layout=_layout(title="Gradient L₂ Norms per Layer"))

    if not grad_data:
        fig.add_annotation(
            text="Waiting for data …", showarrow=False,
            font=dict(color=_DIM, size=14), xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    n = len(grad_data)
    for idx, (name, (steps, norms)) in enumerate(sorted(grad_data.items())):
        short = name.rsplit(".", 1)[-1] if "." in name else name
        brightness = int(120 + 135 * idx / max(n - 1, 1))
        fig.add_trace(go.Scatter(
            x=steps, y=norms, mode="lines",
            line=dict(color=f"rgb({brightness},{brightness},{brightness})", width=1.5),
            name=short,
        ))

    fig.update_layout(
        xaxis_title="Step", yaxis_title="L₂ Norm",
        legend=dict(x=1, y=1, xanchor="right"),
    )
    return fig


def build_accuracy_chart(store: MonitorDataStore) -> go.Figure:
    """Accuracy curve with EMA smoothing."""
    steps, accs = store.get_accuracy_curve()
    fig = go.Figure(layout=_layout(title="Batch Accuracy"))

    if not steps:
        fig.add_annotation(
            text="Waiting for data …", showarrow=False,
            font=dict(color=_DIM, size=14), xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    fig.add_trace(go.Scatter(
        x=steps, y=accs, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1),
        name="raw",
    ))

    if len(accs) > 20:
        w = max(5, len(accs) // 20)
        smoothed = np.convolve(accs, np.ones(w) / w, mode="valid").tolist()
        offset = len(steps) - len(smoothed)
        fig.add_trace(go.Scatter(
            x=steps[offset:], y=smoothed, mode="lines",
            line=dict(color=_WHITE, width=2),
            name=f"EMA({w})",
        ))

    fig.update_layout(
        xaxis_title="Step", yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — configuration & training controls
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    epochs = st.slider("Epochs", 1, 30, 5)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=128)
    lr = st.number_input("Learning rate", 0.0001, 0.1, 0.01, format="%.4f")
    device_name = st.selectbox("Device", ["cpu", "mps", "cuda"], index=0)
    sample_every = st.slider("Hook sample interval", 1, 10, 1,
                             help="Record activations / gradients every N steps")

    st.markdown("---")
    start_btn = st.button("▶  Start Training", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📐 Model Architecture")
    st.code(
        "MnistCNN(\n"
        "  conv1a : Conv2d(1 → 32, 3×3)\n"
        "  bn1a   : BatchNorm2d(32)\n"
        "  conv1b : Conv2d(32 → 32, 3×3)\n"
        "  bn1b   : BatchNorm2d(32)\n"
        "  ── MaxPool2d(2) ──\n"
        "  conv2a : Conv2d(32 → 64, 3×3)\n"
        "  bn2a   : BatchNorm2d(64)\n"
        "  conv2b : Conv2d(64 → 64, 3×3)\n"
        "  bn2b   : BatchNorm2d(64)\n"
        "  ── MaxPool2d(2) ──\n"
        "  fc1    : Linear(3136 → 128)\n"
        "  dropout: Dropout(0.3)\n"
        "  fc2    : Linear(128 → 10)\n"
        ")",
        language="text",
    )
    st.markdown(
        f"<p style='color:{_DIM};font-size:0.75rem;'>"
        "Dataset: MNIST · Optimizer: SGD + Cosine LR</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════
#  Session state — persistent store & thread management
# ═══════════════════════════════════════════════════════════════════════

if "store" not in st.session_state:
    st.session_state.store = MonitorDataStore(max_steps=5000)
    st.session_state.training_active = False

store: MonitorDataStore = st.session_state.store

# Launch training when button is pressed
if start_btn and not st.session_state.training_active:
    st.session_state.store = MonitorDataStore(max_steps=5000)
    store = st.session_state.store

    device = torch.device(device_name)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(
        root=os.path.join(_ROOT, "data"), train=True,
        download=True, transform=tfm,
    )
    loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=0,
    )

    model = MnistCNN(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    monitor = TrainingMonitor(model, store, sample_every=sample_every)
    monitor.attach()

    threading.Thread(
        target=_train_worker,
        args=(model, monitor, device, loader, optimizer,
              scheduler, criterion, epochs),
        daemon=True,
        name="streamlit-train",
    ).start()

    st.session_state.training_active = True
    time.sleep(0.3)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════
is_training = store.is_training
latest = store.get_latest_metrics()

if is_training:
    status_html = '<span class="status-training">● TRAINING</span>'
elif latest is not None:
    status_html = '<span style="color:#555">● COMPLETE</span>'
    st.session_state.training_active = False
else:
    status_html = '<span class="status-idle">● IDLE</span>'

st.markdown(
    f"<h1 style='margin-bottom:0;color:{_WHITE}'>⚡ Neural Network Training Monitor</h1>"
    f"<p style='color:{_DIM};margin-top:0.2rem;font-size:0.85rem;'>"
    f"Real-time backpropagation visualizer &nbsp;&nbsp;{status_html}</p>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  KPI METRIC CARDS
# ═══════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
if latest:
    k1.metric("Step",     f"{latest.step:,}")
    k2.metric("Epoch",    latest.epoch or "–")
    k3.metric("Loss",     f"{latest.loss:.4f}")
    k4.metric("LR",       f"{latest.lr:.6f}" if latest.lr else "–")
    acc = latest.extras.get("accuracy")
    k5.metric("Accuracy", f"{acc:.1%}" if acc is not None else "–")
else:
    k1.metric("Step", "–"); k2.metric("Epoch", "–")
    k3.metric("Loss", "–"); k4.metric("LR", "–")
    k5.metric("Accuracy", "–")


# ═══════════════════════════════════════════════════════════════════════
#  § 1  LOSS LANDSCAPE  —  chain-rule math │ loss chart
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">§ 1 &nbsp; Loss Landscape</div>',
            unsafe_allow_html=True)

col_math, col_chart = st.columns([1, 2.5])

with col_math:
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("#### 🔗 The Chain Rule")
    st.markdown(
        "Backpropagation computes the gradient of the loss "
        "w.r.t. every weight by **chaining** local derivatives "
        "through the computational graph:"
    )
    st.latex(
        r"\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(l)}}"
        r"= \underbrace{\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}}"
        r"_{\text{output error}}"
        r"\;\cdot\; \prod_{k=l}^{L-1}"
        r"\underbrace{\frac{\partial \mathbf{a}^{(k+1)}}"
        r"{\partial \mathbf{a}^{(k)}}}_{\text{Jacobian}}"
        r"\;\cdot\; \underbrace{\frac{\partial \mathbf{a}^{(l)}}"
        r"{\partial \mathbf{w}^{(l)}}}_{\text{local grad}}"
    )
    st.markdown("**Cross-entropy loss** for classification:")
    st.latex(
        r"\mathcal{L} = -\sum_{i=1}^{C} y_i \, \log\!\bigl(\hat{y}_i\bigr)"
    )
    st.markdown(
        f"<p style='color:{_DIM};font-size:0.78rem;margin-top:0.6rem'>"
        "The loss curve → shows how this value decreases as the "
        "optimizer follows the negative gradient.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_chart:
    st.plotly_chart(build_loss_chart(store), use_container_width=True)
    st.plotly_chart(build_accuracy_chart(store), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  § 2  LAYER ACTIVATIONS  —  heatmaps
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">§ 2 &nbsp; Layer Activations — Animated Heatmaps</div>',
            unsafe_allow_html=True)

st.markdown(
    f"<p style='color:{_DIM};font-size:0.82rem;margin-bottom:1rem'>"
    "Each heatmap shows the <b>histogram of activations</b> (y-axis = bin, "
    "x-axis = training step).  Bright bands indicate where most values "
    "concentrate.  Watch for collapsing distributions (dead neurons) or "
    "saturation (all values pushed to extremes).</p>",
    unsafe_allow_html=True,
)

layers = store.get_layer_names()
if layers:
    # Display up to 4 heatmaps per row
    n_per_row = min(4, len(layers))
    for row_start in range(0, len(layers), n_per_row):
        row_layers = layers[row_start:row_start + n_per_row]
        cols = st.columns(len(row_layers))
        for col, lname in zip(cols, row_layers):
            with col:
                st.plotly_chart(
                    build_activation_heatmap(store, lname),
                    use_container_width=True,
                )
else:
    st.markdown(
        f"<p style='color:{_DIM};text-align:center;padding:2rem 0'>"
        "No layer activations recorded yet.</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  § 3  GRADIENT FLOW  —  math │ histograms & norms
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">§ 3 &nbsp; Gradient Flow</div>',
            unsafe_allow_html=True)

g_math, g_chart = st.columns([1, 2.5])

with g_math:
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("#### ⚠ Vanishing & Exploding Gradients")
    st.markdown("The gradient signal flowing back to layer $l$:")
    st.latex(
        r"\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(l)}}"
        r"= \prod_{k=l}^{L-1}"
        r"\mathbf{W}^{(k+1)\!\top}"
        r"\!\cdot\, \text{diag}\!\bigl(\sigma'(\mathbf{z}^{(k)})\bigr)"
    )
    st.markdown(
        "When $\\|\\mathbf{W}^{(k)}\\| < 1$ or $|\\sigma'| < 1$, "
        "this product **shrinks exponentially** → *vanishing gradients*."
    )
    st.markdown(
        "When $\\|\\mathbf{W}^{(k)}\\| > 1$, it **grows exponentially** "
        "→ *exploding gradients*."
    )
    st.markdown(
        f"<p style='color:{_DIM};font-size:0.78rem;margin-top:0.6rem'>"
        "🔍 <b>How to read the histograms:</b>  healthy gradients "
        "spread across many bins.  A sharp spike at bin 0 signals "
        "vanishing gradients — earlier layers stop learning.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with g_chart:
    st.plotly_chart(build_gradient_histograms(store), use_container_width=True)
    st.plotly_chart(build_gradient_norms(store), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  Auto-refresh while training is active
# ═══════════════════════════════════════════════════════════════════════
if is_training:
    time.sleep(1.0)
    st.rerun()
