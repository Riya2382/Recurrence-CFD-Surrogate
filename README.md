# Simulations of Particulate Flows — Project 1: Recurrence CFD Surrogate

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-lightgrey.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

**GitHub:** https://github.com/Riya2382


# Recurrence CFD Surrogate — Double-Gyre Passive Scalar (Toy)

**Goal:** Demonstrate a *recurrence CFD* (RCFD) idea on a toy 2D flow. We simulate a passive scalar advected by a steady double‑gyre velocity field, record snapshots, then build a *recurrence index* that lets us jump forward in time *without* solving the PDE — approximating real‑time evolution by re‑stitching similar states.

This mirrors the idea of using state recurrence in complex flows (e.g., fluidized beds) to accelerate "preview" / "what‑if" studies and digital twins.

## What’s included
- `src/generate_data.py`: finite‑difference solver for 2D advection‑diffusion on a double‑gyre; saves snapshots.
- `src/build_recurrence.py`: computes a k‑NN index over states.
- `src/simulate_rcfd.py`: runs fast surrogate steps by state‑hopping + local linear blend; compares to short‑horizon ground truth.
- `src/evaluate.py`: PSNR/SSIM‑style metrics (simple) and timing.

## Run
```bash
# 1) make data
python src/generate_data.py --nx 64 --ny 64 --steps 400 --dt 0.02 --diff 1e-3

# 2) build recurrence graph
python src/build_recurrence.py --k 5

# 3) simulate with recurrence surrogate and evaluate
python src/simulate_rcfd.py --horizon 100 --seed 0

# 4) metrics & plots
python src/evaluate.py
```
Outputs (PNGs, NPZ) will be in `outputs/`.

## Why this helps your application
- Shows you understand **recurrence CFD** and how to turn trajectories into an index for *real‑time* previews.
- Clean, testable code; reproducible figures you can put in your email.
- Clear extension path to particulate/CFD‑DEM systems by replacing the toy PDE with your solver outputs.
## How this maps to moving/fluidized beds & rotary kilns

- Replace the toy passive scalar with snapshot sequences from a CFD or CFD‑DEM run of a moving/fluidized bed or kiln.
- Build the recurrence index over coarse state descriptors (e.g., binned volume fraction, pressure drop, bed height) to enable **real‑time previews** of future states and **fast what‑if** analyses for operator set‑points.
- The same k‑NN stitching can support **online twin updates** by incorporating the latest sensor-derived states into the index.


> **Use in your email:** Include one of the generated plots and a 1–2 line summary:
> *“Built a small, reproducible demo aligning with recurrence/operator-learning/digital-twin ideas and showed real-time rollouts/forecasting on toy data; ready to swap in CFD/CFD‑DEM snapshots.”*
