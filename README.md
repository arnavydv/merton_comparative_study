# Merton Portfolio Problem - PINN Solution

A Physics-Informed Neural Network (PINN) implementation for solving the Merton portfolio optimization problem with CRRA utility. This project provides a deep learning approach to solving the Hamilton-Jacobi-Bellman (HJB) equation that arises in continuous-time portfolio optimization.

## 📋 Overview

The Merton problem considers an investor who allocates wealth between a risk-free asset (with rate `r`) and a risky asset (with expected return `μ` and volatility `σ`) to maximize expected utility of terminal wealth. For CRRA (Constant Relative Risk Aversion) utility, the problem has a known analytical solution, making it an excellent benchmark for PINN methods.

### Mathematical Formulation

**HJB Equation:**
```
V_t + r·w·V_w - 0.5·((μ-r)²/σ²)·(V_w²/V_ww) = 0
```

**Terminal Condition:**
```
V(T, w) = w^(1-γ) / (1-γ)   (CRRA utility)
```

**Optimal Portfolio Weight:**
```
π* = (μ - r) / (σ² · γ)
```

**Analytical Solution:**
```
V(t, w) = [w^(1-γ) / (1-γ)] · exp(A·(T-t))
where A = (1-γ)·[r + (μ-r)²/(2σ²γ)]
```

## 🏗️ Project Structure

```
PINNs/
├── training.py              # Main training script with model saving
├── evaluation.py            # Comprehensive model evaluation
├── loss_function.py         # HJB PDE residual loss
├── neural_network.py        # Neural network architecture
├── terminal_condition.py    # Terminal condition & analytical solutions
├── collocation_points.py    # Latin Hypercube sampling for collocation points
├── utility_func.py          # CRRA utility functions
├── data_accumulation/
│   ├── config.json          # Model parameters (σ, μ, r, γ)
│   ├── data_accumulation.py # Download historical data (yfinance)
│   └── configurations_calculator.py  # Estimate parameters from data
└── saved_models/            # Directory for saved model checkpoints
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy scipy pandas yfinance
```

### Training a Model

```bash
cd PINNs
python training.py
```

The training script will:
1. Load parameters from `data_accumulation/config.json`
2. Generate collocation points using Latin Hypercube Sampling
3. Train the neural network to satisfy the HJB PDE and terminal condition
4. Save the best model (lowest loss) and final model to `saved_models/`

### Evaluating a Trained Model

```bash
python evaluation.py
```

This will:
- Compute PDE residual errors
- Compare against terminal condition
- Compare against analytical solution
- Compute implied optimal portfolio weights

## ⚙️ Configuration

Parameters are stored in `PINNs/data_accumulation/config.json`:

```json
{
    "Time": 252,              // Trading days per year (for annualization)
    "sigma": 0.2566,          // Annualized volatility
    "mu": 0.1894,             // Annualized expected return
    "rate": 0.0379,           // Risk-free rate
    "gamma": 5                // Risk aversion coefficient
}
```

### Training Hyperparameters

Modify these in `training.py`:

```python
# Collocation points
NUM_COLLOCATION_PTS = 10000
NUM_TERMINAL_PTS = 2000

# Domain
T = 1.0           # Time horizon (normalized to 1 year)
W_MIN = 0.1       // Minimum wealth
W_MAX = 2.0       // Maximum wealth

# Training
EPOCHS = 1000
LEARNING_RATE = 0.001
STEP_SIZE = 200   // LR scheduler step size
GAMMA_SCHED = 0.9 // LR decay factor
```

## 📊 Model Architecture

The default neural network architecture:
- **Input:** 2 dimensions (time `t`, wealth `w`)
- **Hidden layers:** 5 layers × 128 neurons
- **Activation:** Tanh
- **Output:** 1 dimension (value function `V`)
- **Total parameters:** ~83,000

## 📈 Model Saving

The training script automatically saves:

1. **Best Model** (`saved_models/best_model.pt`):
   - Saved when training loss reaches a new minimum
   - Contains: model weights, optimizer state, scheduler state, epoch, loss

2. **Final Model** (`saved_models/final_model.pt`):
   - Saved at the end of training
   - Contains: model weights, optimizer state, scheduler state, epoch, loss, training history

### Loading a Saved Model

```python
from training import load_model
from neural_network import ValueFunc

model, checkpoint = load_model("PINNs/saved_models/best_model.pt")
print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
```

## 🔍 Mathematical Details

### HJB Equation Derivation

The HJB equation is derived from the stochastic optimal control problem:

```
max_π E[U(W_T)]
subject to: dW_t = [r·W_t + π_t·(μ-r)·W_t]dt + π_t·σ·W_t·dZ_t
```

The HJB equation is:
```
V_t + max_π {π·(μ-r)·w·V_w + r·w·V_w + 0.5·π²·σ²·w²·V_ww} = 0
```

Optimizing over π gives:
```
π* = -(μ-r)·V_w / (σ²·w·V_ww)
```

Substituting back yields the HJB PDE used in the loss function.

### PINN Loss Function

The total loss is:
```
Loss = λ₁ · MSE_PDE + λ₂ · MSE_terminal
```

where:
- `MSE_PDE = mean(residual²)` over collocation points
- `MSE_terminal = mean((V_pred - U(w))²)` over terminal points
- `λ₁, λ₂` are weights (default: 1.0 each)

## 📝 Changelog

### Recent Fixes

1. **Fixed HJB PDE Formulation** (Critical):
   - Previous: Used incorrect term `w²·V_ww` instead of `V_w²/V_ww`
   - Current: Correct formulation with proper mathematical derivation

2. **Fixed Variable Naming**:
   - Neural network forward method now uses clear `(t, w)` naming
   - Consistent naming across all modules

3. **Fixed Terminal Condition Return Order**:
   - `terminal_points()` now returns `(t_terminal, w_terminal, v_terminal)`
   - Consistent with training script expectations

4. **Added Model Saving**:
   - Best model checkpointing during training
   - Final model saving with full training history

5. **Added Comprehensive Evaluation**:
   - PDE residual evaluation
   - Analytical solution comparison
   - Optimal portfolio weight verification

## 🤝 Contributing

This project is open to contributions! Areas of interest:
- Alternative neural network architectures
- Different utility functions (non-CRRA)
- Comparison with other numerical methods (FDM, FBSDE)
- Multi-asset extensions
- Transaction costs

## 📚 References

1. Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case". *Review of Economics and Statistics*.
2. Merton, R. C. (1971). "Optimum Consumption and Portfolio Rules in a Continuous-time Model". *Journal of Economic Theory*.
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks". *Journal of Computational Physics*.

## 📄 License

MIT License - see LICENSE file for details.