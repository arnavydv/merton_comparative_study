"""
evaluation/data.py — Inference & Evaluation script

Loads the trained pi_star(t, W) network, runs it on synthetic data,
compares it to the exact analytical Merton solution, and plots the results.
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import visualization
from forward_simulation_equations import forward_simulation
from neural_networks import Z_net
from neural_networks import pi_star
import evaluation.data_generation as data_generation
from mertons_2D import pi_star as analytical_pi_star

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = r"C:/Users/Admin/Desktop/FBSDE PROJECT/models_and_experiments/best.pth"
CONFIG_PATH = r"C:/Users/Admin/Desktop/FBSDE PROJECT/config.json"
NUM_POINTS = 1000
T = 1.0

# ---------------------------------------------------------------------------
# 1. Load config & Calculate Ground Truth
# ---------------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

mu = cfg.get("mu")
sigma = cfg.get("sigma")
rate = cfg.get("rate")
gamma = cfg.get("gamma")

# Calculate the exact theoretical solution
exact_pi = analytical_pi_star(r=rate, mu=mu, sigma=sigma, gamma=gamma)

# ---------------------------------------------------------------------------
# 2. Load trained model
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Loading Model on {device} ---")

model = pi_star()
# FIX: Use map_location instead of map_to_device
checkpoint = torch.load(MODEL_PATH, map_location=device)

# FIX: Extract the correct key saved in training.py ("pi_star")
model.load_state_dict(checkpoint["pi_star"])
model.eval()
model.to(device)

print(f"Model loaded successfully from {MODEL_PATH}")
print(f"Market Params: mu={mu}, sigma={sigma}, r={rate}, gamma={gamma}")
print(f"Analytical Target pi*: {exact_pi:.6f}")

# ---------------------------------------------------------------------------
# 3. Generate Inputs & Run Inference
# ---------------------------------------------------------------------------
print(f"\n--- Running Inference ({NUM_POINTS} random points) ---")
t_vals, W_vals = data_generation.inference_inputs(num_points=NUM_POINTS, T=T, device=device)

with torch.no_grad():
    predictions = model(t_vals, W_vals)

pred_np = predictions.cpu().numpy().flatten()
t_np = t_vals.cpu().numpy().flatten()
W_np = W_vals.cpu().numpy().flatten()

# ---------------------------------------------------------------------------
# 4. Statistical Evaluation
# ---------------------------------------------------------------------------
mean_pred = pred_np.mean()
std_pred = pred_np.std()
mae = np.mean(np.abs(pred_np - exact_pi))

print(f"Predictions Mean : {mean_pred:.6f}")
print(f"Predictions Std  : {std_pred:.6f} (Should be very close to 0)")
print(f"Min / Max        : {pred_np.min():.6f} / {pred_np.max():.6f}")
print(f"Mean Absolute Err: {mae:.6f} vs Analytical Truth")

# ---------------------------------------------------------------------------
# 5. Visualization (Grid Evaluation)
# ---------------------------------------------------------------------------
print("\nGenerating evaluation plot...")
t_grid, W_grid = data_generation.inference_grid(t_steps=40, W_steps=40, T=T, device=device)

with torch.no_grad():
    pi_grid_pred = model(t_grid, W_grid).cpu().numpy()

t_grid_np = t_grid.cpu().numpy().reshape(40, 40)
W_grid_np = W_grid.cpu().numpy().reshape(40, 40)
pi_grid_pred = pi_grid_pred.reshape(40, 40)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot neural network predictions
surf = ax.plot_surface(t_grid_np, W_grid_np, pi_grid_pred, cmap='viridis', alpha=0.8, label="Neural Net pi*")

# Plot analytical truth as a flat semi-transparent red plane
exact_plane = np.full_like(pi_grid_pred, exact_pi)
ax.plot_surface(t_grid_np, W_grid_np, exact_plane, color='red', alpha=0.3, label="Analytical Target")

ax.set_xlabel('Time (t)')
ax.set_ylabel('Brownian Motion (W)')
ax.set_zlabel('Optimal Fraction (pi*)')
ax.set_title(f"Learned Policy vs Analytical Truth\n(MAE: {mae:.6f})")

plt.show()
print("Evaluation complete.")



# Re-run a quick forward sim just for the plots
time_grid_viz, wealth_grid_viz, dw_viz, W_viz = forward_simulation(
    model=model, mu=mu, sigma=sigma, w0=cfg["w0"], rate=rate, T=T, 
    num_steps=int(cfg["num_steps"]), num_paths=500
)

# 1. Market Simulation Plots
visualization.plot_forward_dynamics(time_grid_viz, W_viz, wealth_grid_viz, max_paths=50)

# 2. Policy Validation 3D Plots
visualization.plot_policy_validation(model, exact_pi, T=T, device=device)

# 3. Z-Network Plots (Load Z_net first)
Z_model = Z_net().to(device)
Z_model.load_state_dict(checkpoint["Z_net"]) # Assumes Z_net was saved in training
Z_model.eval()
visualization.plot_backward_functions(Z_model, T=T, device=device)

# 4. Financial Sensitivity Plots
visualization.plot_financial_sensitivity(mu=mu, r=rate, base_gamma=gamma, base_sigma=sigma)