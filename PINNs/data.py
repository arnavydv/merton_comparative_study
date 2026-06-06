import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import visualization
from neural_network import ValueFunc
from evaluation import (
    evaluate_pde_residual,
    evaluate_terminal_condition,
    evaluate_against_analytical,
    compute_optimal_portfolio_from_model,
    full_evaluation,
    print_evaluation_report,
)
from terminal_condition import (
    merton_analytical_solution,
    optimal_portfolio_weight,
    crra,
    terminal_points,
)
from collocation_points import collocation_points
from loss_function import compute_pde_residual as pde_residual_fn

CONFIG_PATH = os.path.join(_project_root, "data_accumulation", "config.json")
SAVE_DIR = os.path.join(_project_root, "saved_models")

NUM_COLLOCATION_PTS = 5000
NUM_TERMINAL_PTS = 2000
T = 1.0
W_MIN = 0.1
W_MAX = 2.0

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

mu = cfg["mu"]
sigma = cfg["sigma"]
rate = cfg["rate"]
gamma = cfg["gamma"]

exact_pi = optimal_portfolio_weight(gamma, rate, mu, sigma)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Loading Model on {device} ---")

model_path = os.path.join(SAVE_DIR, "best_model.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(SAVE_DIR, "final_model.pt")

checkpoint = torch.load(model_path, map_location=device)

model = ValueFunc()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

print(f"Model loaded successfully from {model_path}")
print(f"Market Params: mu={mu}, sigma={sigma}, r={rate}, gamma={gamma}")
print(f"Analytical Target pi*: {exact_pi:.6f}")

t_colloc, w_colloc = collocation_points(NUM_COLLOCATION_PTS, T, W_MAX, W_MIN)
t_colloc = t_colloc.reshape(-1, 1)
w_colloc = w_colloc.reshape(-1, 1)

t_term, w_term, v_term = terminal_points(NUM_TERMINAL_PTS, T, W_MAX, W_MIN, gamma)

print(f"\n--- Running Full Evaluation ---")

results = full_evaluation(
    model, t_colloc, w_colloc, t_term, w_term,
    T, gamma, rate, mu, sigma
)

print_evaluation_report(results)

mean_pred = results["optimal_portfolio"]["model_mean"]
std_pred = results["optimal_portfolio"]["model_std"]
mae = results["optimal_portfolio"]["error"]

print(f"\n--- Summary Statistics ---")
print(f"Predictions Mean : {mean_pred:.6f}")
print(f"Predictions Std  : {std_pred:.6f}")
print(f"Min / Max        : {results['optimal_portfolio']['model_mean'] - 2*std_pred:.6f} / {results['optimal_portfolio']['model_mean'] + 2*std_pred:.6f}")
print(f"Mean Absolute Err: {mae:.6f} vs Analytical Truth")

print("\nGenerating evaluation plots...")

t_np = np.linspace(0.01, T, 50)
w_np = np.linspace(W_MIN, W_MAX, 50)
t_mesh, w_mesh = np.meshgrid(t_np, w_np)

t_tensor = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
w_tensor = torch.tensor(w_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

with torch.no_grad():
    pi_pred = model(t_tensor, w_tensor).cpu().numpy().reshape(50, 50)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(t_mesh, w_mesh, pi_pred, cmap='viridis', alpha=0.8)
exact_plane = np.full_like(pi_pred, exact_pi)
ax.plot_surface(t_mesh, w_mesh, exact_plane, color='red', alpha=0.3)

ax.set_xlabel('Time (t)')
ax.set_ylabel('Wealth (w)')
ax.set_zlabel('Optimal Fraction (pi*)')
ax.set_title(f"Learned Policy vs Analytical Truth\n(MAE: {mae:.6f})")

plt.tight_layout()
plt.show()

t_colloc_2d = t_colloc.cpu().numpy()
w_colloc_2d = w_colloc.cpu().numpy()
pi_model_vals = compute_optimal_portfolio_from_model(model, t_colloc, w_colloc, rate, mu, sigma).cpu().numpy()

fig2 = plt.figure(figsize=(12, 5))

ax1 = fig2.add_subplot(1, 2, 1)
scatter1 = ax1.scatter(t_colloc_2d.flatten(), pi_model_vals.flatten(), c=w_colloc_2d.flatten(), cmap='viridis', alpha=0.6)
ax1.axhline(exact_pi, color='red', linestyle='--', linewidth=2, label=f'Analytical (π*={exact_pi:.4f})')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Optimal Portfolio Weight (π*)')
ax1.set_title('Optimal Policy: Model vs Analytical')
ax1.legend()
plt.colorbar(scatter1, ax=ax1, label='Wealth')

ax2 = fig2.add_subplot(1, 2, 2)
residual_vals = pde_residual_fn(model, t_colloc, w_colloc, rate, mu, sigma).cpu().numpy()
ax2.hist(residual_vals.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('PDE Residual')
ax2.set_ylabel('Frequency')
ax2.set_title(f'PDE Residual Distribution\n(Mean: {residual_vals.mean():.6e})')

plt.tight_layout()
plt.show()

if "history" in checkpoint:
    history = checkpoint["history"]
    if history["train_loss"]:
        fig3 = plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_loss"], color='blue', lw=2, label='Training Loss')
        if history["val_loss"]:
            val_epochs = history["epoch"][::50] if len(history["epoch"]) > 50 else history["epoch"]
            plt.plot(val_epochs, history["val_loss"], color='red', lw=2, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

print("Evaluation complete.")