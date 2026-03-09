# ml_viz — Real-Time Neural Network Training Visualizer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

A real-time dashboard that lets you *watch* a neural network learn — loss curves, gradient flow, activation heatmaps, all updating live as the model trains.

---

## The Story Behind This

I didn't wake up one day and decide to build a training visualizer. It started way simpler than that.
I was sitting through my machine learning coursework, staring at equations for backpropagation chain rule, Jacobians, partial derivatives — and honestly, none of it was clicking. I could memorize the formulas, sure, but I had no intuition for what was actually happening inside the network. Like, what does it *mean* when people say "gradients vanish"? What does a dying ReLU actually look like in practice?

The turning point was when I started training my own models. I'd kick off a training run, watch the loss number go down (hopefully), and that was it. A single number. I had no idea what was going on inside — were all the layers learning equally? Were the early layers even getting meaningful gradients? Was my model's activations healthy or slowly collapsing?

I started printing things. `print(grad.mean())` everywhere. Then matplotlib plots. Then I realized I was spending more time writing throwaway visualization code than actually training models.

That's when the idea hit me — what if I could just hook into any PyTorch model and get a live dashboard of everything happening inside? Not just loss curves (TensorBoard does that fine), but the actual *distributions* of activations and gradients at every layer, updating in real-time as the model trains.

The math started making sense once I could *see* it. The chain rule isn't just a formula — you can literally watch the gradient signal shrink as it flows backward through layers. Batch normalization isn't just "a thing you add" you can see how it keeps activation distributions stable across training. That was the moment ML went from "memorize and apply" to "I actually get this."

So I built this tool mostly for myself — to learn, to debug, to build intuition. If it helps someone else who's staring at backprop equations and thinking "but what does this actually look like?" then that's even better.

---

## What It Does

You get a Streamlit dashboard with:

- **Live loss curve** that scales dynamically with a smoothed EMA overlay
- **Activation heatmaps** for every layer — watch how distributions evolve over training
- **Gradient histograms** to spot vanishing/exploding gradients at a glance
- **Gradient L₂ norms** per layer over time
- **Math explanations** (chain rule, gradient flow) displayed right next to the live charts so the theory and practice are side by side

Everything runs on a non-blocking architecture — the training loop never waits on the dashboard, and the dashboard never touches PyTorch tensors directly.

## Project Structure

```
ml_viz/
├── main.py                    # CLI entry point (Matplotlib dashboard)
├── requirements.txt
├── models/
│   ├── demo_cnn.py            # Small CNN for CIFAR-10
│   └── mnist_cnn.py           # Standard CNN for MNIST
├── monitor/
│   ├── data_store.py          # Thread-safe ring-buffer store
│   └── training_monitor.py    # PyTorch hooks + async queue
└── viz/
    ├── dashboard.py            # Matplotlib live dashboard
    └── streamlit_dashboard.py  # Streamlit dashboard (the main one)
```

## Quick Start

```bash
# clone it
git clone https://github.com/Narasimha2211/ml_viz.git
cd ml_viz

# set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly

# run the streamlit dashboard
streamlit run viz/streamlit_dashboard.py
```

Hit **▶ Start Training** in the sidebar and watch it go. MNIST downloads automatically on the first run.

## How It Works (the short version)

```
Training Thread                    Dashboard Thread
─────────────────                  ─────────────────
model.forward()                    
   │ forward hooks fire            
   │ detach → cpu → numpy          
   │ enqueue (non-blocking) ──────► consumer thread
                                       │
loss.backward()                        ├─ compute histograms
   │ backward hooks fire               ├─ acquire lock
   │ detach → cpu → numpy              └─ write to DataStore
   │ enqueue (non-blocking) ──────►         │
                                            │ (polls every 1s)
monitor.on_step_end()                       ▼
   │ loss, lr, accuracy            Streamlit reads snapshots
   └─ write to DataStore           and rebuilds Plotly charts
```

The key design choice: hooks only do the cheap part (detach/cpu/numpy) synchronously. The expensive stuff — histogram computation, lock acquisition — happens on a separate consumer thread. Training throughput stays high.

## Configuration

From the sidebar you can tweak:
- **Epochs** (1–30)
- **Batch size** (32, 64, 128, 256)
- **Learning rate**
- **Device** (cpu / mps / cuda)
- **Hook sample interval** record every N steps (bump this up for bigger models)

## Tech Stack

- **PyTorch** — model, hooks, training loop
- **Streamlit** — dashboard UI
- **Plotly** — interactive charts
- **NumPy** — all the number crunching between hooks and dashboard
- **threading + queue** — non-blocking data pipeline

---

Built by [Narasimha](https://github.com/Narasimha2211) — because staring at `loss: 0.4523` a thousand times wasn't cutting it anymore.
