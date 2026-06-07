"""
evaluation/data.py — Inference & Evaluation script
Loads the trained VanillaPINN (Value Function), computes implied optimal policy 
via Autograd, compares it to the exact analytical Merton solution, and plots results.
"""
import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import vanilla_pinns.evaluation.visualization as visualization

# Import the exact model architecture used in training
try:
    from vanilla_pinns.neural_network import VanillaPINN
except ImportError:
    raise ImportError("Could not import VanillaPINN. Ensure vanilla_pinns/neural_network.py is in the project root.")

#---------------------------------------------------------------------------
# Configuration & Paths
#---------------------------------------------------------------------------
SAVE_DIR = _project_root / "vanilla_pinns" / "saved_models"
MODEL_PATH = SAVE_DIR / "best_model.pt"
CONFIG_PATH = _project_root / "vanilla_pinns" / "config.json"

NUM_POINTS = 1000
T = 1.0

#---------------------------------------------------------------------------
# Helper: Compute Value Function and Implied Policy via Autograd
#---------------------------------------------------------------------------
def compute_value_and_policy(model, w, t, mu, r, sigma):
    """
    Computes V, V_w, V_ww, and the implied optimal policy pi*.
    Note: VanillaPINN internally log-transforms w, but autograd handles 
    the chain rule correctly as long as w requires grad.
    """
    w_req = w.clone().requires_grad_(True)
    t_req = t.clone().requires_grad_(True)
    
    v = model(w_req, t_req)
    
    # First derivative: V_w
    v_w = torch.autograd.grad(v, w_req, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    # Second derivative: V_ww
    v_ww = torch.autograd.grad(v_w, w_req, grad_outputs=torch.ones_like(v_w), create_graph=True)[0]
    
    # Implied Merton optimal policy: pi* = - (mu - r) * V_w / (sigma^2 * w * V_ww)
    # Add small epsilon to denominator to prevent division by zero during early training
    pi_star_pred = - (mu - r) * v_w / (sigma**2 * w_req * v_ww + 1e-8)
    
    return v, v_w, v_ww, pi_star_pred

#---------------------------------------------------------------------------
# Helper: Lightweight Forward Simulation (Euler-Maruyama)
#---------------------------------------------------------------------------
def run_forward_simulation(mu, sigma, w0, rate, T, num_steps, num_paths, pi_star_value):
    """Simulates wealth paths using the constant analytical (or learned) optimal policy."""
    dt = T / num_steps
    time_grid = torch.linspace(0, T, num_steps + 1)
    
    # Initialize wealth and Brownian motion
    wealth = torch.full((num_paths, num_steps + 1), w0, dtype=torch.float32)
    W = torch.zeros((num_paths, num_steps + 1), dtype=torch.float32)
    
    for i in range(num_steps):
        dZ = torch.randn(num_paths) * np.sqrt(dt)
        W[:, i+1] = W[:, i] + dZ
        
        # Merton SDE: dW_t = W_t * (r + pi*(mu - r)) dt + W_t * pi* * sigma dZ_t
        drift = rate + pi_star_value * (mu - rate)
        diffusion = pi_star_value * sigma
        
        wealth[:, i+1] = wealth[:, i] * (1 + drift * dt + diffusion * dZ)
        
    return time_grid, W, wealth

#---------------------------------------------------------------------------
# 1. Load config & Calculate Ground Truth
#---------------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

mu = float(cfg.get("mu"))
sigma = float(cfg.get("sigma"))
rate = float(cfg.get("rate"))
gamma = float(cfg.get("gamma"))
w0 = float(cfg.get("w0", 1.0))
num_steps = int(cfg.get("num_steps", 100))

# Calculate the exact theoretical solution (Constant for CRRA)
exact_pi = (mu - rate) / (gamma * sigma**2)

#---------------------------------------------------------------------------
# 2. Load trained model
#---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Loading Model on {device} ---")

model = VanillaPINN().to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run training.py first.")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Model loaded successfully from {MODEL_PATH}")
print(f"Market Params: mu={mu}, sigma={sigma}, r={rate}, gamma={gamma}")
print(f"Analytical Target pi*: {exact_pi:.6f}")

#---------------------------------------------------------------------------
# 3. Generate Inputs & Run Inference (Random Points)
#---------------------------------------------------------------------------
print(f"\n--- Running Inference ({NUM_POINTS} random points) ---")
# Generate random w in [0.5, 2.0] and t in [0.01, T]
w_rand = torch.rand(NUM_POINTS, 1) * 1.5 + 0.5
t_rand = torch.rand(NUM_POINTS, 1) * (T - 0.01) + 0.01

w_rand, t_rand = w_rand.to(device), t_rand.to(device)

v_pred, v_w_pred, v_ww_pred, pi_pred = compute_value_and_policy(
    model, w_rand, t_rand, torch.tensor(mu, device=device), 
    torch.tensor(rate, device=device), torch.tensor(sigma, device=device)
)

pred_pi_np = pi_pred.detach().cpu().numpy().flatten()
w_np = w_rand.cpu().numpy().flatten()
t_np = t_rand.cpu().numpy().flatten()

#---------------------------------------------------------------------------
# 4. Statistical Evaluation
#---------------------------------------------------------------------------
mean_pred = pred_pi_np.mean()
std_pred = pred_pi_np.std()
mae = np.mean(np.abs(pred_pi_np - exact_pi))

print(f"\n--- Statistical Evaluation of Implied pi* ---")
print(f"Predictions Mean : {mean_pred:.6f}")
print(f"Predictions Std  : {std_pred:.6f} (Should be very close to 0 for Merton)")
print(f"Min / Max        : {pred_pi_np.min():.6f} / {pred_pi_np.max():.6f}")
print(f"Mean Absolute Err: {mae:.6f} vs Analytical Truth")

#---------------------------------------------------------------------------
# 5. Visualization
#---------------------------------------------------------------------------
print("\nGenerating evaluation plots...")

# A. Training Diagnostics
visualization.plot_training_diagnostics(
    loss_history=checkpoint.get('loss_history', []),
    pde_loss_history=checkpoint.get('pde_loss_history', []),
    terminal_loss_history=checkpoint.get('terminal_loss_history', []),
    concavity_loss_history=checkpoint.get('concavity_loss_history', []),
    mono_loss_history=checkpoint.get('mono_loss_history', [])
)

# B. Grid Evaluation for Surfaces
t_steps, w_steps = 40, 40
t_grid_1d = torch.linspace(0.01, T, t_steps, device=device)
w_grid_1d = torch.linspace(0.5, 2.0, w_steps, device=device)
t_mesh_1d, w_mesh_1d = torch.meshgrid(t_grid_1d, w_grid_1d, indexing='ij')

t_flat = t_mesh_1d.flatten().unsqueeze(1)
w_flat = w_mesh_1d.flatten().unsqueeze(1)

v_grid, _, _, pi_grid = compute_value_and_policy(
    model, w_flat, t_flat, torch.tensor(mu, device=device), 
    torch.tensor(rate, device=device), torch.tensor(sigma, device=device)
)

t_mesh_np = t_mesh_1d.cpu().numpy()
w_mesh_np = w_mesh_1d.cpu().numpy()
v_mesh_np = v_grid.detach().cpu().numpy().reshape(t_steps, w_steps)
pi_mesh_np = pi_grid.detach().cpu().numpy().reshape(t_steps, w_steps)

# Plot Value Function
visualization.plot_value_function_surface(t_mesh_np, w_mesh_np, v_mesh_np, T=T)

# Plot Policy Validation
visualization.plot_policy_validation(t_mesh_np, w_mesh_np, pi_mesh_np, exact_pi, T=T)

# C. Forward Simulation (using the exact analytical pi* for baseline stability)
print("Running lightweight forward simulation...")
time_grid_viz, W_viz, wealth_grid_viz = run_forward_simulation(
    mu=mu, sigma=sigma, w0=w0, rate=rate, T=T, 
    num_steps=num_steps, num_paths=500, pi_star_value=exact_pi
)
visualization.plot_forward_dynamics(time_grid_viz, W_viz, wealth_grid_viz, max_paths=50)

# D. Financial Sensitivity
visualization.plot_financial_sensitivity(mu=mu, r=rate, base_gamma=gamma, base_sigma=sigma)

print("\n✅ Evaluation complete.")