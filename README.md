# Merton Portfolio Problem — Comparative Study of FDM, PINN, and Deep BSDE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comparative benchmark of three numerical approaches for the **Hamilton–Jacobi–Bellman (HJB) equation** in **Merton portfolio optimization** with CRRA utility:

1. **FDM** — Finite Difference Method (theta-scheme on a log-wealth grid)
2. **PINN** — Physics-Informed Neural Networks
3. **Deep BSDE** — Deep Backward Stochastic Differential Equations

All methods are evaluated against the closed-form Merton solution across 25+ experiments covering accuracy, computational cost, convergence, sensitivity, and robustness.

**Related documents:**

- [results.md](results.md) — Full tabulated benchmark results
- [MATHEMATICAL_VALIDITY_REPORT.md](MATHEMATICAL_VALIDITY_REPORT.md) — Independent mathematical audit of implementations
- [paper/HJB_Comparative_Study.tex](paper/HJB_Comparative_Study.tex) — LaTeX write-up of the study

---

## Table of Contents

- [Mathematical Formulation](#mathematical-formulation)
  - [Wealth Dynamics](#wealth-dynamics)
  - [CRRA Utility](#crra-utility)
  - [HJB Equation](#hjb-equation)
  - [Closed-Form Analytical Solution](#closed-form-analytical-solution)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [FDM (Finite Difference Method)](#fdm-finite-difference-method)
  - [PINN (Physics-Informed Neural Networks)](#pinn-physics-informed-neural-networks)
  - [Deep BSDE](#deep-bsde)
- [Installation and Quick Start](#installation-and-quick-start)
- [Configuration](#configuration)
- [Experiments and Results](#experiments-and-results)
- [Figures and Outputs](#figures-and-outputs)
- [Results Summary](#results-summary)
- [Known Limitations](#known-limitations)
- [License](#license)
- [References](#references)

---

## Mathematical Formulation

### Wealth Dynamics

The investor allocates wealth between a **risk-free asset** (rate `r`) and a **risky asset** (expected return `mu`, volatility `sigma`). The wealth process `W_t` follows:

$$
dW_t = \bigl[r W_t + \pi_t (\mu - r) W_t\bigr] \, dt + \pi_t \sigma W_t \, dZ_t
$$

where:

- `pi_t` — fraction of wealth in the risky asset (the control)
- `dZ_t` — increment of a standard Brownian motion
- `W_0 = w_0` — initial wealth

### CRRA Utility

The objective is to maximize expected utility of terminal wealth:

$$
\max_{\pi} \; \mathbb{E}\bigl[U(W_T)\bigr], \qquad
U(W) = \frac{W^{1-\gamma} - 1}{1-\gamma} \quad (\gamma \neq 1)
$$

where `gamma > 0` is the **coefficient of relative risk aversion (RRA)**.

For implementation, the equivalent power form is used:

$$
U(W) = \frac{W^{p}}{p}, \qquad p = 1 - \gamma
$$

### HJB Equation

The **Hamilton–Jacobi–Bellman (HJB) equation** for the value function `V(t, W)` is:

$$
\frac{\partial V}{\partial t} + \max_{\pi} \Bigl\{ \bigl[r + \pi(\mu - r)\bigr] W \frac{\partial V}{\partial W} + \frac{1}{2} \pi^2 \sigma^2 W^2 \frac{\partial^2 V}{\partial W^2} \Bigr\} = 0
$$

**Terminal condition:**

$$
V(T, W) = U(W) = \frac{W^{p}}{p}
$$

#### First-Order Condition (Optimal Control)

Maximizing over `pi` gives the optimal portfolio weight:

$$
\pi^* = -\frac{(\mu - r) \, V_W}{\sigma^2 W \, V_{WW}}
$$

Substituting `pi^*` back yields the reduced nonlinear PDE:

$$
\frac{\partial V}{\partial t} + r W V_W - \frac{1}{2} \frac{(\mu - r)^2}{\sigma^2} \frac{V_W^{\,2}}{V_{WW}} = 0
$$

### Closed-Form Analytical Solution

For CRRA utility, the HJB equation admits a closed-form solution:

$$
V(t, W) = \frac{W^{p}}{p} \cdot \exp\bigl(A \cdot (T - t)\bigr)
$$

where:

$$
A = p \left[ r + \frac{(\mu - r)^2}{2 \gamma \sigma^2} \right]
$$

The **optimal portfolio weight** is constant (independent of `t` and `W`):

$$
\pi^* = \frac{\mu - r}{\gamma \, \sigma^2}
$$

The **Sharpe ratio** of the risky asset is:

$$
\text{SR} = \frac{\mu - r}{\sigma}
$$

---

## Project Structure

```text
HJB_MERTON_BENCHMARK/
|
|-- comprehensive_experiments.py    # Main benchmarking suite (25+ experiments)
|-- requirements.txt                # Python dependencies
|-- README.md                       # This file
|-- results.md                      # Detailed experiment results
|-- MATHEMATICAL_VALIDITY_REPORT.md # Mathematical audit report
|-- LICENSE                         # MIT License
|-- .gitignore
|
|-- FDM/                            # Finite Difference Method
|   |-- fdm_main.py                 #   FDM solver (theta-scheme, log-wealth grid)
|   |-- evaluation.py               #   Evaluation utilities
|   `-- visualization.py            #   Visualization utilities
|
|-- vanilla_pinns/                  # Physics-Informed Neural Networks
|   |-- neural_network.py           #   VanillaPINN architecture (4x128 Tanh)
|   |-- loss.py                     #   HJB residual, terminal, concavity, monotonicity losses
|   |-- training.py                 #   2-phase training (Adam -> L-BFGS)
|   |-- collocation_points.py       #   Latin Hypercube sampling
|   |-- config.json                 #   Market parameters
|   |-- evaluationn.py              #   Inference and evaluation script
|   |-- evaluation/
|   |   |-- evaluation.py           #   Evaluation helper
|   |   |-- visualization.py        #   Plotting functions
|   |   `-- training.md             #   Training notes
|   `-- saved_models/               #   Best and final model checkpoints
|       |-- best_model.pt
|       `-- final_model.pt
|
|-- FBSDE/                          # Deep BSDE Method
|   |-- neural_networks.py          #   pi-network and Z-network (4x64 ReLU)
|   |-- forward_simulation_equations.py  # Euler-Maruyama wealth paths
|   |-- backward_simulation_equation.py  # Backward recursion for Y_0
|   |-- loss_function.py            #   Variance-based loss (Var[Y_0])
|   |-- training.py                 #   BSDE training loop
|   |-- brownian_motion.py          #   Brownian motion generator
|   |-- utility_function.py         #   CRRA utility function
|   |-- config.json                 #   Network and simulation parameters
|   |-- evaluation/                 #   Evaluation and data generation
|   `-- models_and_experiments/     #   Saved model checkpoints
|       |-- best.pth
|       `-- final.pth
|
|-- tables/                         # LaTeX result tables (benchmark output)
|   |-- tab01_policy_accuracy.tex
|   |-- tab01_value_accuracy.tex
|   |-- tab02_hjb_residual.tex
|   |-- tab03_computational_cost.tex
|   |-- tab07_convergence_fdm.tex
|   |-- tab09_noise_robustness.tex
|   |-- tab10_sensitivity_gamma.tex
|   `-- tab25_overall_score.tex
|
|-- figures/                        # Generated figures
|   |-- fig01_pi_vs_W.png ... fig11_convergence.png
|   `-- experiments/                # Benchmarking figures (12 plots)
|
|-- results/                        # JSON and pickle experiment outputs
|   |-- experiment_results_latest.json
|   `-- experiment_results_latest.pkl
|
`-- paper/                          # LaTeX paper and assets
    |-- HJB_Comparative_Study.tex
    |-- refs.bib
    |-- figures/                    # Figures copied for the paper
    `-- tables/                     # Tables copied for the paper
```

---

## Methodology

### FDM (Finite Difference Method)

**Location:** `FDM/fdm_main.py`

The FDM solver applies a **theta-scheme** on a **log-transformed wealth grid**:

$$
x = \ln W, \qquad x \in [x_{\min}, x_{\max}]
$$

The HJB PDE in log-wealth coordinates becomes:

$$
\frac{\partial V}{\partial t} + b(\pi) \frac{\partial V}{\partial x} + D(\pi) \frac{\partial^2 V}{\partial x^2} = 0
$$

with:

$$
b(\pi) = r + \pi(\mu - r) - \frac{1}{2}\pi^2\sigma^2, \qquad
D(\pi) = \frac{1}{2}\pi^2\sigma^2
$$

The optimal policy in log-wealth coordinates:

$$
\pi^* = -\frac{(\mu - r) V_x}{\sigma^2 (V_{xx} - V_x)}
$$

**Theta-scheme discretization:**

$$
\frac{V^k - V^{k+1}}{\Delta t} + \theta \mathcal{L}[V^k] + (1-\theta) \mathcal{L}[V^{k+1}] = 0
$$

where `L[V] = b * V_x + D * V_xx` and `theta = 1` (fully implicit) is used for unconditional stability.

The system is solved with a **tridiagonal banded solver** (`scipy.linalg.solve_banded`) and **Picard iterations** (8 per time step) for the nonlinear policy term.

**Boundary conditions:**

- **Left (low wealth):** Dirichlet `V(t, W_min) = U(W_min)`
- **Right (high wealth):** Neumann extrapolation (free boundary)

### PINN (Physics-Informed Neural Networks)

**Location:** `vanilla_pinns/`

#### Neural Network Architecture

```text
Input: (log W, t)
  |
  Linear(2 -> 128) -> Tanh
  Linear(128 -> 128) -> Tanh  (x4 hidden layers)
  Linear(128 -> 1)
  |
Output: V(t, W)
```

Total parameters: **66,561**. Activation: Tanh throughout. Wealth is log-transformed before concatenation with time.

#### Loss Function

The total loss is a weighted sum of four components:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{HJB}} + 10 \cdot \mathcal{L}_{\text{terminal}} + 0.1 \cdot \mathcal{L}_{\text{concavity}} + 0.1 \cdot \mathcal{L}_{\text{monotonicity}}
$$

**(a) HJB PDE Residual Loss** — uses a multiplied residual form to avoid division by near-zero `V_WW`:

$$
\mathcal{R} = V_t V_{WW} + r W V_{WW} V_W - \frac{1}{2} \frac{(\mu - r)^2}{\sigma^2} V_W^2
$$

**(b) Terminal Condition Loss** — z-score normalized MSE between predicted and target terminal values (see `vanilla_pinns/loss.py`).

**(c) Concavity Loss** — penalizes `V_WW > 0`.

**(d) Monotonicity Loss** — penalizes `V_W < 0`.

#### Training Procedure

| Phase | Optimizer | Epochs | Learning Rate | Batch |
|-------|-----------|--------|---------------|-------|
| 1 | Adam | 8,000 | 1e-3 | 25,000 collocation + 8,000 terminal |
| 2 | L-BFGS | 1,000 | 1.0 (strong Wolfe) | Same resampling each epoch |

Collocation points are drawn via **Latin Hypercube sampling** over `t in [0, T]` and `W in [0.5, 2.0]`.

#### Implied Policy

The optimal policy is extracted via automatic differentiation:

$$
\hat{\pi}^* = -\frac{(\mu - r) V_W}{\sigma^2 W V_{WW}}
$$

### Deep BSDE

**Location:** `FBSDE/`

#### Network Architectures

**pi-network** (policy function) and **Z-network** (auxiliary backward variable):

```text
Input: (t, W) in R^2
  |
  Linear(2 -> 64) -> ReLU
  Linear(64 -> 64) -> ReLU  (x3 additional hidden layers)
  Linear(64 -> 1)
  |
Output: pi(t, W) or Z(t, W)
```

Each network has 4 hidden layers of width 64 with ReLU activations.

#### Forward Simulation (Euler-Maruyama)

$$
W_{i+1} = W_i + \bigl[r W_i + \pi(t_i, W_i) (\mu - r) W_i\bigr] \Delta t + \pi(t_i, W_i) \sigma W_i \Delta Z_i
$$

where `Delta Z_i ~ N(0, Delta t)` and `Delta t = T/N`.

A **softplus lower bound** enforces strict positivity of simulated wealth.

#### Backward Recursion

**Terminal condition:**

$$
Y_N = U(W_N) = \frac{W_N^{1-\gamma}}{1-\gamma}
$$

**Backward step** (`n = N-1, ..., 0`):

$$
Y_n = \frac{Y_{n+1} - Z_n \Delta Z_n}{1 + a \Delta t}
$$

where `a = (1 - gamma) [r + (mu - r)^2 / (2 gamma sigma^2)]` and `Z_n` is output by the Z-network.

#### Loss Function

$$
\mathcal{L}_{\text{total}} = \text{Var}[Y_0^{(1)}, \ldots, Y_0^{(M)}] + \lambda \cdot \text{MSE}(\hat{\pi}, \pi^*)
$$

with `lambda = 1.0` and `pi^* = (mu - r) / (gamma sigma^2)`.

---

## Installation and Quick Start

### Prerequisites

- **Python 3.10+** (tested with Python 3.13)
- Optional: CUDA-capable GPU for faster PINN and BSDE training

### Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install torch numpy scipy matplotlib pandas psutil tabulate
```

### Run the Full Benchmark Suite

From the project root (uses pre-trained PINN and BSDE checkpoints):

```bash
python comprehensive_experiments.py
```

This runs all 25+ experiments and writes:

- LaTeX tables to `tables/`
- Figures to `figures/experiments/`
- Serialized results to `results/` (`experiment_results_latest.json` and `.pkl`)

### Train Individual Methods

**FDM** (run from `FDM/`):

```bash
cd FDM
python fdm_main.py
```

**PINN** (must run from project root — config path is relative):

```bash
python vanilla_pinns/training.py
```

**Deep BSDE** (must run from `FBSDE/` — `config.json` is loaded at import):

```bash
cd FBSDE
python training.py
```

> **Note:** `FBSDE/training.py` contains default model save paths pointing to an external directory (`FBSDE PROJECT`). When retraining, pass `best_path` and `final_path` explicitly or edit the defaults so checkpoints are saved under `FBSDE/models_and_experiments/`. Pre-trained checkpoints are already included in the repository.

### Evaluate Pre-Trained Models

| Method | Checkpoint | Evaluation Entry Point |
|--------|------------|------------------------|
| PINN | `vanilla_pinns/saved_models/best_model.pt` | `vanilla_pinns/evaluationn.py` |
| BSDE | `FBSDE/models_and_experiments/best.pth` | `FBSDE/evaluation/quick_eval.py` |
| FDM | N/A (analytic grid solve) | `FDM/evaluation.py` |

---

## Configuration

### Market Parameters (shared)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Expected return | mu | 0.201649 | Risky asset drift (annualized) |
| Volatility | sigma | 0.256573 | Risky asset volatility (annualized) |
| Risk-free rate | r | 0.0368 | Risk-free rate (annualized) |
| Risk aversion | gamma | 5.0 | Relative risk aversion (CRRA) |
| Horizon | T | 1.0 | Investment horizon (years) |
| Initial wealth | w0 | 1.0 | Starting wealth (BSDE config) |

**Analytical benchmarks derived from these parameters:**

| Quantity | Value | Formula |
|----------|-------|---------|
| pi* | 0.500835 | `(mu - r) / (gamma * sigma^2)` |
| SR | 0.6425 | `(mu - r) / sigma` |
| p | -4.0 | `1 - gamma` |
| A | -0.3123 | `p * [r + (mu-r)^2 / (2*gamma*sigma^2)]` |

Config files: `vanilla_pinns/config.json`, `FBSDE/config.json`.

### PINN Hyperparameters

| Parameter | Value |
|-----------|-------|
| Collocation points | 25,000 |
| Terminal points | 8,000 |
| Adam epochs | 8,000 |
| L-BFGS epochs | 1,000 |
| Learning rate (Adam) | 1e-3 |
| Weight: Terminal loss | 10.0 |
| Weight: Concavity loss | 0.1 |
| Weight: Monotonicity loss | 0.1 |
| Wealth domain (training) | W in [0.5, 2.0] |

### Deep BSDE Hyperparameters

| Parameter | Value |
|-----------|-------|
| Time steps (N) | 252 |
| Paths per epoch | 2,000 |
| Epochs | 5,000 |
| Learning rate | 1e-4 |
| pi supervision weight | 1.0 |
| Supervision points | 512 |

### FDM Grid Parameters

| Parameter | Value |
|-----------|-------|
| Nx (grid points) | 500 |
| Nt (time steps) | 252 |
| theta (scheme) | 1.0 (fully implicit) |
| x_min | -2.5 (W_min ~ 0.082) |
| x_max | 4.5 (W_max ~ 90) |
| pi_min | 0.0 (long-only) |
| pi_max | 1.0 (no leverage) |
| Picard iterations | 8 |

---

## Experiments and Results

The benchmarking suite (`comprehensive_experiments.py`) runs **25+ experiments**:

| # | Experiment | Description |
|---|------------|-------------|
| 1 | Ground Truth Accuracy | MAE, RMSE, Rel L2, Rel L-infinity for V and pi* |
| 2 | HJB Residual | PDE residual statistics on 10,000 random points |
| 3-6 | Computational Cost | Training time, inference time, memory usage |
| 7 | Convergence Study | FDM grid refinement, loss history comparison |
| 8 | Sample Efficiency | Performance vs. training data size |
| 9 | Noise Robustness | Perturbed mu, sigma (1%, 5%, 10%) |
| 10 | Sensitivity to gamma | gamma in {2, 3, 5, 10, 20} |
| 11 | Sensitivity to sigma | sigma in {0.10, 0.20, 0.30, 0.40} |
| 12 | Sensitivity to mu | Drift sensitivity |
| 13 | Time Horizons | T in {0.25, 0.5, 1.0, 2.0, 5.0} |
| 14 | Wealth Domain | W in {[0,2], [0,5], [0,10], [0,20]} |
| 15 | Grid Refinement | Richardson extrapolation for convergence order |
| 16-17 | Hyperparameter Sensitivity | PINN and BSDE hyperparameters |
| 18 | Residual Histogram | Distribution of HJB residuals |
| 19 | Error Heatmap | Spatially resolved error (t, W) |
| 20 | Wealth Slice | V(W) at t = 0.5 |
| 21 | Time Slice | V(t) at W = 1 |
| 22 | Policy Comparison | pi*(W) at t = 0 and pi*(t) at W = 1 |
| 23 | Stability | Performance under random initializations |
| 24 | Pareto Frontier | Accuracy vs. runtime trade-off |
| 25 | Overall Score | Composite benchmark score |

For full numerical results, see [results.md](results.md).

---

## Figures and Outputs

### Main Figures (`figures/`)

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig01_pi_vs_W.png` | Optimal policy vs wealth |
| 2 | `fig02_pi_vs_t.png` | Optimal policy vs time |
| 3 | `fig03_3d_pi_surface.png` | 3D policy surface |
| 4 | `fig04_3d_V_surface.png` | 3D value function surface |
| 5 | `fig05_V_slices.png` | Value function slices |
| 6 | `fig06_V_error_heatmap.png` | Value function error heatmap |
| 7 | `fig07_pi_error.png` | Policy error |
| 8 | `fig08_pi_sensitivity_gamma.png` | Policy sensitivity to gamma |
| 9 | `fig09_wealth_distribution.png` | Terminal wealth distribution |
| 10 | `fig10_efficient_frontier.png` | Efficient frontier |
| 11 | `fig11_convergence.png` | Training convergence |

### Benchmark Figures (`figures/experiments/`)

| File | Description |
|------|-------------|
| `convergence_fdm.png` | FDM convergence with grid refinement |
| `error_heatmap_comparison.png` | Error heatmap: FDM vs PINN |
| `grid_refinement.png` | Grid refinement study |
| `pareto_curve.png` | Accuracy vs runtime Pareto frontier |
| `policy_comparison.png` | pi*(W) comparison |
| `policy_comparison_timeslice.png` | pi*(t) comparison |
| `residual_histogram.png` | HJB residual distribution |
| `sensitivity_gamma.png` | Sensitivity to risk aversion |
| `sensitivity_sigma.png` | Sensitivity to volatility |
| `time_horizons.png` | Different time horizons |
| `time_slice_comparison.png` | V(t) slice comparison |
| `wealth_slice_comparison.png` | V(W) slice comparison |

### LaTeX Tables (`tables/`)

Generated tables can be included directly in LaTeX documents. Copies for the paper are in `paper/tables/`.

---

## Results Summary

### Key Findings

1. **FDM achieves the highest accuracy** with pi* MAE of `2.10e-4` and solves in **0.24 seconds** — orders of magnitude faster than neural methods.

2. **Deep BSDE reports competitive pi* accuracy** (MAE `3.46e-3`) with the **fastest inference** (`0.17` microseconds/point). See [Known Limitations](#known-limitations) for caveats on the BSDE implementation.

3. **PINN struggles with second-order HJB residuals**, yielding pi* MAE of `2.33e-1` after 9,000 total training epochs. This reflects challenges in propagating accuracy through neural second derivatives.

4. The **accuracy-runtime Pareto frontier** shows a clear trade-off:
   - FDM dominates for accuracy-constrained applications
   - BSDE is fastest at inference time
   - PINN underperforms both on policy accuracy in this setup

5. **FDM convergence** is approximately `O(h^2)` for the policy (consistent with the theta-scheme rate).

### Consolidated Results

See [results.md](results.md) for the complete set of tables, figures references, and interpretations from all experiments.

---

## Known Limitations

An independent mathematical audit is documented in [MATHEMATICAL_VALIDITY_REPORT.md](MATHEMATICAL_VALIDITY_REPORT.md). Summary:

| Component | Status | Notes |
|-----------|--------|-------|
| Theory / analytical solution | Valid | Standard Merton CRRA formulation |
| FDM | Mostly valid | Approximate Dirichlet BC at low wealth; strong pi* accuracy |
| PINN | Problematic | Multiplied HJB residual and z-score terminal loss differ from standard PINN formulations; high pi* error |
| Deep BSDE | Critical issues | Networks may receive Brownian motion state instead of simulated wealth in some code paths; pi* supervision toward a constant analytical target affects reported accuracy |
| Evaluation harness | Mixed | Cross-method comparison should be interpreted with implementation caveats |

Readers should treat FDM results as the most reliable reference in this repository and interpret PINN/BSDE numbers in light of the audit findings.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

1. Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case". *Review of Economics and Statistics*, 51(3), 247-257.
2. Merton, R. C. (1971). "Optimum Consumption and Portfolio Rules in a Continuous-time Model". *Journal of Economic Theory*, 3(4), 373-413.
3. Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". *Journal of Computational Physics*, 378, 686-707.
4. Han, J., Jentzen, A., and E, W. (2018). "Solving high-dimensional partial differential equations using deep learning". *Proceedings of the National Academy of Sciences*, 115(34), 8505-8510.
5. Forsyth, P. A., and Vetzal, K. R. (2002). "Quadratic convergence for valuing American options using a penalty method". *SIAM Journal on Scientific Computing*, 23(6), 2095-2122.