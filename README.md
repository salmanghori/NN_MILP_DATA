# NN_MILP_DATA

Tools to **build supervised-learning datasets** from large-scale **MILP/MPC multi-agent traffic simulations** and train a neural baseline for **AI-assisted (warm-start / surrogate) optimization**.

This repo focuses on the **data pipeline**: converting simulation logs into `(X features → Y controls)` pairs, validating the dataset, and training an **MLP baseline** that learns to predict per-agent control actions. The resulting model can be used to accelerate optimization workflows in high-density scenarios.

> Note: The dataset format (slot-based + masks + global context) is also compatible with richer AI models such as **FiLM conditioning** and **Transformer self-attention** (agent-to-agent interaction). The included training script here is an **MLP baseline**.

---

## Repository Contents

- **Dataset Builders**
  - `NN_Data_set_build_updated.py`
  - `NN_data_set_build_new_checked.py`

- **Training**
  - `training_MLP_NN.py` — MLP training on the generated dataset

---

## Problem Setup (High-Level)

Given multi-agent simulations (traffic/intersection/corridor) solved via **MILP/MPC**, we construct training samples per timestep:

- **Inputs (X):** per-agent state + neighbor/conflict context + scenario globals  
- **Targets (Y):** per-agent control action (e.g., acceleration / control input) derived from the MILP/MPC solution

This enables:
- **AI warm-starting**: predict a good initial control guess for the MILP/MPC solver  
- **Surrogate control**: approximate the optimization-based controller for faster inference (when appropriate)

---

## Data Format Overview

Each dataset row corresponds to one simulation timestep `t` (requiring `t+1` to compute targets).

- `X`: flattened **slot-based features** for up to `N_SLOTS` agents + **global features**
- `Y`: per-slot **target control** (e.g., acceleration derived from `v(t+1) - v(t)`)
- `mask`: which slots contain real agents (1) vs padding (0)
- `y_mask`: which slots have valid targets (agent persists to `t+1`)
- `meta`: per-row metadata (file, timestep, radius, arrival rate, etc.)

---

## Dataset Schema (Exact)

### Slot Features (12 per agent)

Order (per slot):

1. `pos`         — 1D lane position (signed along lane axis)  
2. `vel`         — 1D lane velocity (along lane axis)  
3. `flow`        — lane indicator (0 = x-lane, 1 = y-lane)  
4. `dist`        — distance-to-center proxy (often `abs(pos)`)  
5. `same_gap`    — nearest same-lane neighbor gap  
6. `has_same`    — {0,1} whether same-lane neighbor exists  
7. `cross_sum`   — min of `|pos_i| + |pos_j|` across cross-flow neighbors  
8. `has_cross`   — {0,1} whether cross-flow neighbor exists  
9. `cross_mgn`   — `cross_sum - S_DIST` (margin)  
10. `cross_safe` — {0,1} (`cross_mgn > 0`)  
11. `gap_mgn`    — `same_gap - D_SAFE` (margin)  
12. `gap_safe`   — {0,1} (`gap_mgn > 0`)

**Typical constants used in the builders:**
- `D_SAFE_M = 3.0`  (same-lane headway threshold)
- `S_DIST_M = 4.0`  (cross-flow separation threshold)
- `DT = 0.1`
- `ACCEL_MIN, ACCEL_MAX = -5.0, 5.0`

### Global Features (2)

Appended once per row:
- `[R, arrival_rate]`

### Final Shapes

Let:
- `S = N_SLOTS`
- `F = 12`
- `G = 2`

Then:

- `X.shape = (N_samples, S*F + G)`
- `Y.shape = (N_samples, S)`
- `mask.shape = (N_samples, S)`      (1 = real agent, 0 = padding)
- `y_mask.shape = (N_samples, S)`    (1 = valid target, 0 = ignore)
- `meta.shape = (N_samples,)`        (array of dict objects)

---

## Input Log Assumptions

The dataset builders handle an “old-format” JSON style where the file is:

- a **list of frames**
- each frame is a list of agent records
- “inside control-zone” records may store numeric fields as nested singleton lists

Example inside-zone record pattern:
```text
[[x],[y],[vx],[vy], d, vid]
