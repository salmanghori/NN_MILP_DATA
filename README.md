# NN_MILP_DATA

Tools to **build supervised-learning datasets** from large-scale **MILP/MPC multi-agent traffic simulations** and train a neural baseline for **AI-assisted (warm-start / surrogate) optimization**.

This repo focuses on the **data pipeline**: converting simulation logs into `(X features → Y controls)` pairs, validating the dataset, and training an MLP model that learns to predict per-agent control actions. The resulting model can be used to accelerate optimization workflows in high-density scenarios.

---

## Repository Contents

- **Dataset Builders**
  - `NN_Data_set_build_updated.py`
  - `NN_data_set_build_new_checked.py`

- **Training**
  - `training_MLP_NN.py` (MLP training on the generated dataset)

---

## Problem Setup (High-Level)

Given multi-agent simulations (traffic/intersection/corridor) solved via **MILP/MPC**, we construct training samples per timestep:

- **Inputs (X):** per-agent state + neighbor/conflict context + scenario globals  
- **Targets (Y):** per-agent control action (e.g., acceleration / control input) derived from the MILP/MPC solution

This enables:
- **AI warm-starting**: predict a good initial control guess for the MILP/MPC solver  
- **Surrogate control**: approximate the optimization-based controller for faster inference

---

## Quick Start

### 1) Setup Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
