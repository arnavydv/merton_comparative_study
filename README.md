# Merton Portfolio Problem — Comparative Study of FDM, PINN, and Deep BSDE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive comparative study of three numerical paradigms for solving the **Hamilton–Jacobi–Bellman (HJB) equation** arising in the **Merton portfolio optimization problem** with CRRA utility:

1. **FDM** — Finite Difference Method (θ-scheme on log-wealth grid)
2. **PINN** — Physics-Informed Neural Networks
3. **Deep BSDE** — Deep Backward Stochastic Differential Equations

All methods are benchmarked against the closed-form analytical solution over 25+ experiments covering accuracy, computational cost, convergence, sensitivity, and robustness.

---

## 📋 Table of Contents

- [Mathematical Formulation](#-mathematical-formulation)
  - [Wealth Dynamics](#wealth-dynamics)
  - [HJB Equation](#hjb-equation)
  - [Closed-Form Analytical Solution](#closed-form-analytical-solution)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
  - [FDM (Finite Difference Method)](#1-fdm-finite-difference-method)
  - [PINN (Physics-Informed Neural Networks)](#2-pinn-physics-informed-neural-networks)
  - [Deep BSDE](#3-deep-bsde)
- [Installation & Quick Start](#-installation--quick-start)
- [Configuration](#-configuration)
- [Experiments & Results](#-experiments--results)
- [Figures](#-figures)
- [Results Summary](#-results-summary)
- [License](#-license)

---

## 📐 Mathematical Formulation

### Wealth Dynamics

The investor allocates wealth between a **risk-free asset** (rate $r$) and a **risky asset** (expected return $\mu$, volatility $\sigma$). The wealth process $W_t$ follows:

$$
dW_t = \bigl[r W_t + \pi_t (\mu - r) W_t\bigr] \, dt + \pi_t \sigma W_t \, dZ_t
$$

where:
- $\pi_t$ = fraction of wealth allocated to the risky asset (the control)
- $dZ_t$ = increment of a standard Brownian motion
- $W_0 = w_0$ = initial wealth

### CRRA Utility

The objective is to maximize expected utility of terminal wealth:

$$
\max_{\pi} \; \mathbb{E}\bigl[U(W_T)\bigr], \qquad
U(W) = \frac{W^{1-\gamma} - 1}{1-\gamma} \quad (\gamma \neq 1)
$$

where $\gamma > 0$ is the **coefficient of relative risk aversion (RRA)**.

For convenience, the shifted form is often used:

$$
U(W) = \frac{W^{p}}{p}, \qquad p = 1 - \gamma
$$

### HJB Equation

The **Hamilton–Jacobi–Bellman (HJB) equation** governing the value function $V(t, W)$ is:

$$
\frac{\partial V}{\partial t} + \max_{\pi} \Bigl\{ \bigl[r + \pi(\mu - r)\bigr] W \frac{\partial V}{\partial W} + \frac{1}{2} \pi^2 \sigma^2 W^2 \frac{\partial^2 V}{\partial W^2} \Bigr\} = 0
$$

**Terminal condition:**

$$
V(T, W) = U(W) = \frac{W^{p}}{p}
$$

#### First-Order Condition (Optimal Control)

Maximizing over $\pi$ gives the optimal portfolio weight:

$$
\pi^* = -\frac{(\mu - r) \, V_W}{\sigma^2 W \, V_{WW}}
$$

Substituting $\pi^*$ back into the HJB equation yields the **nonlinear PDE** that must be satisfied:

$$
\frac{\partial V}{\partial t} + r W V_W - \frac{1}{2} \frac{(\mu - r)^2}{\sigma^2} \frac{V_W^{\,2}}{V_{WW}} = 0
$$

### Closed-Form Analytical Solution

For CRRA utility, the HJB equation admits a closed-form solution:

$$
V(t, W) = \frac{W^{p}}{p} \cdot \exp\bigl(A \cdot (T - t)\bigr)
$$

where the constant $A$ is given by:

$$
A = p \left[ r + \frac{(\mu - r)^2}{2 \gamma \sigma^2} \right]
$$

The **optimal portfolio weight** is constant (independent of $t$ and $W$):

$$
\boxed{\pi^* = \frac{\mu - r}{\gamma \, \sigma^2}}
$$

The **Sharpe ratio** of the risky asset is:

$$
\text{SR} = \frac{\mu - r}{\sigma}
$$

---

## 🏗️ Project Structure

```
HJB_MERTON_BENCHMARK/
│
├── comprehensive_experiments.py    # Main benchmarking suite (25+ experiments)
├── README.md                       # This file
├── LICENSE                         # MIT License
│
├── FDM/                            # Finite Difference Method
│   ├── fdm_main.py                 #   FDM solver (θ-scheme, log-wealth grid)
│   ├── evaluation.py               #   Evaluation utilities
│   └── visualization.py            #   Visualization utilities
│
├── vanilla_pinns/                  # Physics-Informed Neural Networks
│   ├── neural_network.py           #   VanillaPINN architecture (5×128 Tanh)
│   ├── loss.py                     #   HJB residual, terminal, concavity, monotonicity losses
│   ├── training.py                 #   2-phase training (Adam → L-BFGS)
│   ├── collocation_points.py       #   Latin Hypercube sampling
│   ├── config.json                 #   Market parameters
│   ├── evaluationn.py              #   Inference & evaluation script
│   ├── evaluation/
│   │   ├── evaluation.py           #   Evaluation helper
│   │   └── visualization.py        #   Plotting functions
│   └── saved_models/               #   Best & final model checkpoints
│
├── FBSDE/                          # Deep BSDE Method
│   ├── neural_networks.py          #   π-network and Z-network (4×64 ReLU)
│   ├── forward_simulation_equations.py  # Euler-Maruyama wealth paths
│   ├── backward_simulation_equation.py  # Backward recursion for Y₀
│   ├── loss_function.py            #   Variance-based loss (Var[Y₀])
│   ├── training.py                 #   BSDE training loop
│   ├── brownian_motion.py          #   Brownian motion generator
│   ├── utility_function.py         #   CRRA utility function
│   ├── config.json                 #   Network & simulation parameters
│   ├── evaluation/                 #   Evaluation & data generation
│   └── models_and_experiments/     #   Saved model checkpoints
│
├── tables/                         # LaTeX result tables
│   ├── tab01_policy_accuracy.tex
│   ├── tab01_value_accuracy.tex
│   ├── tab02_hjb_residual.tex
│   ├── tab03_computational_cost.tex
│   ├── tab07_convergence_fdm.tex
│   ├── tab09_noise_robustness.tex
│   ├── tab10_sensitivity_gamma.tex
│   └── tab25_overall_score.tex
│
├── figures/                        # Generated figures
│   ├── fig01_pi_vs_W.png
│   ├── fig02_pi_vs_t.png
│   ├── fig03_3d_pi_surface.png
│   ├── fig04_3d_V_surface.png
│   ├── fig05_V_slices.png
│   ├── fig06_V_error_heatmap.png
│   ├── fig07_pi_error.png
│   ├── fig08_pi_sensitivity_gamma.png
│   ├── fig09_wealth_distribution.png
│   ├── fig10_efficient_frontier.png
│   ├── fig11_convergence.png
│   └── experiments/                # Benchmarking figures
│       ├── convergence_fdm.png
│       ├── error_heatmap_comparison.png
│       ├── grid_refinement.png
│       ├── pareto_curve.png
│       ├── policy_comparison.png
│       ├── policy_comparison_timeslice.png
│       ├── residual_histogram.png
│       ├── sensitivity_gamma.png
│       ├── sensitivity_sigma.png
│       ├── time_horizons.png
│       ├── time_slice_comparison.png
│       └── wealth_slice_comparison.png
│
├── results/                        # JSON & pickle experiment results
└── paper/                          # LaTeX paper
    ├── HJB_Comparative_Study.tex
    ├── refs.bib
    ├── figures/                    # Figures for paper
    └── tables/                     # Tables for paper
```

---

## 🔬 Methodology

### 1. FDM (Finite Difference Method)

**Location:** `FDM/fdm_main.py`

The FDM solver applies a **θ-scheme** on a **log-transformed wealth grid**:

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

**θ-scheme discretization:**

$$
\frac{V^k - V^{k+1}}{\Delta t} + \theta \mathcal{L}[V^k] + (1-\theta) \mathcal{L}[V^{k+1}] = 0
$$

where $\mathcal{L}[V] = b \cdot V_x + D \cdot V_{xx}$ and $\theta = 1$ (fully implicit) is used for unconditional stability.

The system is solved via **tridiagonal matrix solver** (`scipy.linalg.solve_banded`) with **Picard iterations** (8 per time step) for the nonlinear policy term.

**Boundary conditions:**
- **Left (low wealth):** Dirichlet $V(t, W_{\min}) = U(W_{\min})$
- **Right (high wealth):** Neumann extrapolation (free boundary)

### 2. PINN (Physics-Informed Neural Networks)

**Location:** `vanilla_pinns/`

#### Neural Network Architecture

```
Input: (t, W) ∈ ℝ²
  ↓
  log(W) transformation  ← handles CRRA power-law scaling
  ↓
  [Linear(2 → 128)] → Tanh
  [Linear(128 → 128)] → Tanh  × 4
  [Linear(128 → 1)]
  ↓
Output: V(t, W) ∈ ℝ
```

Total parameters: ~83,000. Activation: Tanh throughout.

#### Loss Function

The total loss is a weighted sum of four components:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{HJB}} + 10 \cdot \mathcal{L}_{\text{terminal}} + 0.1 \cdot \mathcal{L}_{\text{concavity}} + 0.1 \cdot \mathcal{L}_{\text{monotonicity}}
$$

##### (a) HJB PDE Residual Loss

$$
\mathcal{L}_{\text{HJB}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \bigl| \mathcal{R}(t_i, W_i) \bigr|^2
$$

where the **HJB residual** $\mathcal{R}$ is given by:

$$
\mathcal{R} = V_t \cdot V_{WW} + r W V_{WW} V_W - \frac{1}{2} \frac{(\mu - r)^2}{\sigma^2} V_W^2
$$

> **Note:** The PINN loss uses a **multiplied form** $\mathcal{R} = V_t V_{WW} + rW V_{WW} V_W - \frac{1}{2}\frac{(\mu-r)^2}{\sigma^2} V_W^2$ (multiplying the standard HJB equation by $V_{WW}$) to avoid division by near-zero second derivatives, which can destabilize training. This is mathematically equivalent to the standard residual $V_t + rW V_W - \frac{1}{2}\frac{(\mu-r)^2}{\sigma^2} \frac{V_W^2}{V_{WW}} = 0$ when $V_{WW} \neq 0$.

##### (b) Terminal Condition Loss

$$
\mathcal{L}_{\text{terminal}} = \frac{1}{N_t} \sum_{i=1}^{N_t} \bigl\| V(T, W_i) - U(W_i) \bigr\|^2
$$

where $U(W_i) = \frac{W_i^{1-\gamma}}{1-\gamma}$.

##### (c) Concavity Loss

Encourages $V_{WW} < 0$ (value function must be concave):

$$
\mathcal{L}_{\text{concavity}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \bigl[\text{ReLU}(V_{WW}(t_i, W_i))\bigr]^2
$$

##### (d) Monotonicity Loss

Encourages $V_W > 0$ (value function must be increasing in wealth):

$$
\mathcal{L}_{\text{monotonicity}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \bigl[\text{ReLU}(-V_W(t_i, W_i))\bigr]^2
$$

#### Training Procedure

**Phase 1 — Adam Optimizer** (8,000 epochs):
- Learning rate: $10^{-3}$
- Batch size: 25,000 collocation points + 8,000 terminal points
- Latin Hypercube sampling (refreshed each epoch)

**Phase 2 — L-BFGS Optimizer** (1,000 epochs):
- Learning rate: 1.0 (Newton-type full step)
- Strong Wolfe line search for stability
- Same resampling strategy

#### Computing the Implied Policy

The optimal policy is extracted via **automatic differentiation**:

$$
\hat{\pi}^* = -\frac{(\mu - r) V_W}{\sigma^2 W V_{WW}}
$$

This is computed using `torch.autograd.grad` for first and second derivatives.

### 3. Deep BSDE

**Location:** `FBSDE/`

#### Network Architectures

**π-network** (policy function):
```
Input: (t, W) ∈ ℝ²
  ↓
  [Linear(2 → 64)] → ReLU
  [Linear(64 → 64)] → ReLU  × 3
  [Linear(64 → 1)]
  ↓
Output: π(t, W) ∈ ℝ
```

**Z-network** (auxiliary for backward recursion):
```
Same architecture as π-network
```

#### Forward Simulation (Euler-Maruyama)

The wealth process is simulated over discrete time steps $0 = t_0 < t_1 < \dots < t_N = T$:

$$
W_{i+1} = W_i + \bigl[r W_i + \pi(t_i, W_i) (\mu - r) W_i\bigr] \Delta t + \pi(t_i, W_i) \sigma W_i \Delta Z_i
$$

where $\Delta Z_i \sim \mathcal{N}(0, \Delta t)$ and $\Delta t = T/N$.

A **softplus lower bound** ensures strict positivity:
$$
W_{i+1} = \text{Softplus}(W_{i+1}) + 10^{-8}
$$

#### Backward Recursion

Given wealth paths $\{W_i\}_{i=0}^N$, the value function $Y_t = V(t, W_t)$ is computed backwards:

**Terminal condition:**
$$
Y_N = U(W_N) = \frac{W_N^{1-\gamma}}{1-\gamma}
$$

**Backward step ($n = N-1, \dots, 0$):**
$$
Y_n = \frac{Y_{n+1} - Z_n \Delta Z_n}{1 + a \Delta t}
$$

where:
- $a = (1-\gamma) \bigl[r + \frac{(\mu - r)^2}{2\gamma\sigma^2}\bigr]$ (from the analytical solution's exponent)
- $Z_n$ is output by the Z-network: $Z_n = Z_{\text{net}}(t_n, \Delta W_n)$
- $\Delta Z_n$ is the Brownian increment

#### Loss Function

The training loss minimizes the **variance of initial value estimates** across paths:

$$
\mathcal{L}_{\text{BSDE}} = \text{Var}\bigl[Y_0^{(1)}, Y_0^{(2)}, \dots, Y_0^{(M)}\bigr]
$$

where $M$ is the number of paths. For a consistent (correct) solution, $Y_0 = V(0, W_0)$ should be a deterministic scalar, so its variance across paths should be zero.

Additionally, a **supervision loss** penalizes deviation from the analytical $\pi^*$:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BSDE}} + \lambda \cdot \text{MSE}(\hat{\pi}, \pi^*)
$$

with $\lambda = 1.0$ and $\pi^* = \frac{\mu - r}{\gamma\sigma^2}$.

---

## 🚀 Installation & Quick Start

### Prerequisites

```bash
pip install torch numpy scipy matplotlib pandas psutil tabulate
```

### Running the Full Benchmark Suite

```bash
python comprehensive_experiments.py
```

This runs all 25+ experiments, saves LaTeX tables to `tables/`, figures to `figures/experiments/`, and results to `results/`.

### Individual Method Training

**FDM:**
```bash
cd FDM && python fdm_main.py
```

**PINN:**
```bash
cd vanilla_pinns && python training.py
```

**Deep BSDE:**
```bash
cd FBSDE && python training.py
```

---

## ⚙️ Configuration

### Market Parameters (shared)

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\mu$ | 0.201649 | Risky asset expected return (annualized) |
| $\sigma$ | 0.256573 | Volatility (annualized) |
| $r$ | 0.0368 | Risk-free rate |
| $\gamma$ | 5.0 | Relative risk aversion |
| $T$ | 1.0 | Investment horizon (years) |

**Analytical benchmarks derived from these parameters:**

| Quantity | Value | Formula |
|----------|-------|---------|
| $\pi^*$ | 0.500835 | $\frac{\mu - r}{\gamma \sigma^2}$ |
| SR | 0.6424 | $\frac{\mu - r}{\sigma}$ |
| $p$ | -4.0 | $1 - \gamma$ |
| $A$ | 0.2047 | $p \bigl[r + \frac{(\mu-r)^2}{2\gamma\sigma^2}\bigr]$ |

### PINN Hyperparameters

| Parameter | Value |
|-----------|-------|
| Collocation points | 25,000 |
| Terminal points | 8,000 |
| Adam epochs | 8,000 |
| L-BFGS epochs | 1,000 |
| Learning rate (Adam) | $10^{-3}$ |
| Weight: Terminal loss | 10.0 |
| Weight: Concavity loss | 0.1 |
| Weight: Monotonicity loss | 0.1 |
| Wealth domain | $[0.5, 2.0]$ |

### Deep BSDE Hyperparameters

| Parameter | Value |
|-----------|-------|
| Time steps | 252 |
| Paths per epoch | 2,000 |
| Epochs | 5,000 |
| Learning rate | $10^{-4}$ |
| $\pi$ supervision weight | 1.0 |
| Supervision points | 512 |

### FDM Grid Parameters

| Parameter | Value |
|-----------|-------|
| $N_x$ (grid points) | 500 |
| $N_t$ (time steps) | 252 |
| $\theta$ (scheme) | 1.0 (implicit) |
| $x_{\min}$ | -2.5 |
| $x_{\max}$ | 4.5 |
| $\pi_{\min}$ | 0.0 (long-only) |
| $\pi_{\max}$ | 1.0 (no leverage) |
| Picard iterations | 8 |

---

## 📊 Experiments & Results

The benchmarking suite (`comprehensive_experiments.py`) runs **25+ experiments** organized as follows:

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | **Ground Truth Accuracy** | MAE, RMSE, Rel L2, Rel L∞ for $V$ and $\pi^*$ |
| 2 | **HJB Residual** | PDE residual statistics on 10,000 random points |
| 3–6 | **Computational Cost** | Training time, inference time, memory usage |
| 7 | **Convergence Study** | FDM grid refinement, loss history comparison |
| 8 | **Sample Efficiency** | Performance vs. training data size |
| 9 | **Noise Robustness** | Sensitivity to perturbed $\mu$, $\sigma$ (1%, 5%, 10%) |
| 10 | **Sensitivity to $\gamma$** | $\gamma \in \{2, 3, 5, 10, 20\}$ |
| 11 | **Sensitivity to $\sigma$** | $\sigma \in \{0.10, 0.20, 0.30, 0.40\}$ |
| 12 | **Sensitivity to $\mu$** | Drift sensitivity |
| 13 | **Time Horizons** | $T \in \{0.25, 0.5, 1.0, 2.0, 5.0\}$ |
| 14 | **Wealth Domain** | $W \in \{[0,2], [0,5], [0,10], [0,20]\}$ |
| 15 | **Grid Refinement** | Richardson extrapolation for convergence order |
| 16–17 | **Hyperparameter Sensitivity** | PINN & BSDE hyperparameters |
| 18 | **Residual Histogram** | Distribution of HJB residuals |
| 19 | **Error Heatmap** | Spatially resolved error $(t, W)$ |
| 20 | **Wealth Slice** | $V(W)$ at $t=0.5$ |
| 21 | **Time Slice** | $V(t)$ at $W=1$ |
| 22 | **Policy Comparison** | $\pi^*(W)$ at $t=0$ and $\pi^*(t)$ at $W=1$ |
| 23 | **Stability** | Performance under random initializations |
| 24 | **Pareto Frontier** | Accuracy vs. Runtime trade-off |
| 25 | **Overall Score** | Composite benchmark score |

---

## 🖼️ Figures

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig01_pi_vs_W.png` | Optimal policy vs wealth |
| 2 | `fig02_pi_vs_t.png` | Optimal policy vs time |
| 3 | `fig03_3d_pi_surface.png` | 3D policy surface |
| 4 | `fig04_3d_V_surface.png` | 3D value function surface |
| 5 | `fig05_V_slices.png` | Value function slices |
| 6 | `fig06_V_error_heatmap.png` | Value function error heatmap |
| 7 | `fig07_pi_error.png` | Policy error |
| 8 | `fig08_pi_sensitivity_gamma.png` | Policy sensitivity to $\gamma$ |
| 9 | `fig09_wealth_distribution.png` | Terminal wealth distribution |
| 10 | `fig10_efficient_frontier.png` | Efficient frontier |
| 11 | `fig11_convergence.png` | Training convergence |
| — | `experiments/convergence_fdm.png` | FDM convergence with grid refinement |
| — | `experiments/error_heatmap_comparison.png` | Error heatmap: FDM vs PINN |
| — | `experiments/grid_refinement.png` | Grid refinement study |
| — | `experiments/pareto_curve.png` | Accuracy vs Runtime Pareto frontier |
| — | `experiments/policy_comparison.png` | $\pi^*(W)$ comparison |
| — | `experiments/policy_comparison_timeslice.png` | $\pi^*(t)$ comparison |
| — | `experiments/residual_histogram.png` | HJB residual distribution |
| — | `experiments/sensitivity_gamma.png` | Sensitivity to risk aversion |
| — | `experiments/sensitivity_sigma.png` | Sensitivity to volatility |
| — | `experiments/time_horizons.png` | Different time horizons |
| — | `experiments/time_slice_comparison.png` | $V(t)$ slice comparison |
| — | `experiments/wealth_slice_comparison.png` | $V(W)$ slice comparison |

---

## 📈 Results Summary

### Key Findings

1. **FDM achieves the highest accuracy** with $\pi^*$ MAE of $2.10 \times 10^{-4}$ and trains in **0.24 seconds** — four orders of magnitude faster than neural approaches.

2. **Deep BSDE achieves competitive accuracy** ($\pi^*$ MAE of $3.46 \times 10^{-3}$) with the **fastest inference** ($0.17 \mu$s/point, ~60$\times$ faster than FDM).

3. **PINN struggles with second-order HJB residuals**, yielding $\pi^*$ MAE of $2.33 \times 10^{-1}$ despite 9,000 total training epochs. This reflects fundamental challenges in propagating accuracy through second-order derivatives of neural networks.

4. The **Accuracy–Runtime Pareto frontier** reveals a clear trade-off:
   - FDM dominates for accuracy-constrained applications
   - BSDE excels for real-time inference
   - PINN is dominated by both methods

5. **FDM convergence** is approximately $O(h^2)$ for the policy (consistent with the θ-scheme's theoretical rate).

### Consolidated Results

> **For the complete set of results from all experiments, see [`results.md`](results.md).**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

1. Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case". *Review of Economics and Statistics*, 51(3), 247–257.
2. Merton, R. C. (1971). "Optimum Consumption and Portfolio Rules in a Continuous-time Model". *Journal of Economic Theory*, 3(4), 373–413.
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". *Journal of Computational Physics*, 378, 686–707.
4. Han, J., Jentzen, A., & E, W. (2018). "Solving high-dimensional partial differential equations using deep learning". *Proceedings of the National Academy of Sciences*, 115(34), 8505–8510.
5. Forsyth, P. A., & Vetzal, K. R. (2002). "Quadratic convergence for valuing American options using a penalty method". *SIAM Journal on Scientific Computing*, 23(6), 2095–2122.