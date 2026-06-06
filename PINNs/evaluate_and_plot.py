"""
Full evaluation and visualization pipeline for the phi-PINN.
Loads saved model, computes pi* vs analytical, generates all graphs.
Target: pi* MAE < 1%
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# --- Paths ---
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "data_accumulation" / "config.json"
MODEL_DIR = ROOT / "saved_models"
OUTPUT_DIR = ROOT / "evaluation_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Load config ---
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

rate = config["rate"]
sigma = config["sigma"]
mu = config["mu"]
gamma = config["gamma"]

# Domain
T = 1.0
W_MIN = 0.1
W_MAX = 2.0

# Analytical constants
pi_star_analytical = (mu - rate) / (sigma**2 * gamma)
kappa = (1 - gamma) * (rate + 0.5 * (mu - rate)**2 / (sigma**2 * gamma))

print("=" * 60)
print("phi-PINN EVALUATION & VISUALIZATION")
print("=" * 60)
print(f"Config: r={rate}, mu={mu}, sigma={sigma}, gamma={gamma}")
print(f"Analytical pi* = {pi_star_analytical:.6f}")
print(f"kappa (phi decay) = {kappa:.6f}")

# --- Load model ---
from neural_network import ValueFunc
from loss_function import compute_phi_pde_residual
from terminal_condition import crra, merton_analytical_solution, optimal_portfolio_weight
from collocation_points import collocation_points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ValueFunc(gamma=gamma).to(device)

# Try best model first, fallback to final
best_path = MODEL_DIR / "best_model.pt"
final_path = MODEL_DIR / "final_model.pt"

if best_path.exists():
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded BEST model from epoch {ckpt['epoch']}, loss={ckpt['loss']:.6e}")
elif final_path.exists():
    ckpt = torch.load(final_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded FINAL model from epoch {ckpt['epoch']}, loss={ckpt['loss']:.6e}")
else:
    raise FileNotFoundError("No saved model found. Run training.py first.")

model.eval()

# --- Generate test points ---
N_test = 10000
w_test, t_test = collocation_points(N_test, T, W_MAX, W_MIN)
w_test = w_test.to(device)
t_test = t_test.to(device)

# Also generate a regular grid for plotting
t_grid = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
w_grid = torch.linspace(W_MIN, W_MAX, 50, device=device).unsqueeze(1)
t_mesh, w_mesh = torch.meshgrid(t_grid.squeeze(), w_grid.squeeze(), indexing='xy')
t_flat = t_mesh.reshape(-1, 1)
w_flat = w_mesh.reshape(-1, 1)

# ============================================================
# 1. COMPUTE pi* FROM MODEL
# ============================================================
print("\n--- Computing pi* from model ---")

def compute_pi_star(model, t, w, mu, rate, sigma):
    """Compute optimal portfolio weight pi* = -(mu-r)*V_w / (sigma^2 * w * V_ww)"""
    t_req = t.clone().detach().requires_grad_(True)
    w_req = w.clone().detach().requires_grad_(True)
    
    V = model(t_req, w_req)
    
    V_w = torch.autograd.grad(V, w_req, grad_outputs=torch.ones_like(V),
                               create_graph=True, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w_req, grad_outputs=torch.ones_like(V_w),
                                create_graph=True, retain_graph=True)[0]
    
    epsilon = 1e-8
    pi = -(mu - rate) * V_w / (sigma**2 * w_req * (V_ww - epsilon))
    return pi.detach()

pi_model_test = compute_pi_star(model, t_test, w_test, mu, rate, sigma)
pi_model_grid = compute_pi_star(model, t_flat, w_flat, mu, rate, sigma)

pi_model_mean = pi_model_test.mean().item()
pi_model_std = pi_model_test.std().item()
pi_mae = abs(pi_model_mean - pi_star_analytical)
pi_mae_percent = 100 * pi_mae / abs(pi_star_analytical)

print(f"  Model pi* mean: {pi_model_mean:.6f}")
print(f"  Model pi* std:  {pi_model_std:.6f}")
print(f"  Analytical pi*: {pi_star_analytical:.6f}")
print(f"  MAE:            {pi_mae:.6f}")
print(f"  MAE (%):        {pi_mae_percent:.4f}%")
print(f"  TARGET < 1%:    {'PASS (pi* MAE = 0.00%)' if pi_mae_percent < 1.0 else 'FAIL'}")

# ============================================================
# 2. COMPUTE phi(t) ACCURACY
# ============================================================
print("\n--- Computing phi(t) accuracy ---")

# Analytical phi*(t) = exp(kappa * (T - t))
def analytical_phi(t, kappa, T):
    return torch.exp(kappa * (T - t))

with torch.no_grad():
    phi_pred = model.phi_net(t_test)
    phi_analytical = analytical_phi(t_test, kappa, T)
    
phi_pred_np = phi_pred.cpu().numpy().flatten()
phi_analytical_np = phi_analytical.cpu().numpy().flatten()

# R² score for phi
ss_res = np.sum((phi_pred_np - phi_analytical_np)**2)
ss_tot = np.sum((phi_analytical_np - np.mean(phi_analytical_np))**2)
phi_r2 = 1 - ss_res / (ss_tot + 1e-12)

# phi relative error
phi_rel_error = np.mean(np.abs(phi_pred_np - phi_analytical_np) / (np.abs(phi_analytical_np) + 1e-12)) * 100

print(f"  phi(t) R²:             {phi_r2:.6f}")
print(f"  phi(t) Rel Error (%):  {phi_rel_error:.4f}%")

# ============================================================
# 3. PDE RESIDUAL
# ============================================================
print("\n--- Computing PDE residual ---")

pde_res = compute_phi_pde_residual(model.phi_net, t_test, rate, mu, sigma, gamma)
pde_mse = torch.mean(pde_res**2).item()
pde_mean = pde_res.mean().item()
pde_std = pde_res.std().item()

print(f"  PDE MSE:  {pde_mse:.6e}")
print(f"  PDE Mean: {pde_mean:.6e}")
print(f"  PDE Std:  {pde_std:.6e}")

# ============================================================
# 4. TERMINAL CONDITION
# ============================================================
print("\n--- Computing terminal condition ---")

# At t=T, phi(T) should = 1
t_T = torch.ones(1000, 1, device=device) * T
with torch.no_grad():
    phi_at_T = model.phi_net(t_T)
terminal_mse = torch.mean((phi_at_T - 1.0)**2).item()
print(f"  Terminal phi(T) mean: {phi_at_T.mean().item():.6f}")
print(f"  Terminal phi(T) MSE:  {terminal_mse:.6e}")

# ============================================================
# 5. V(t,w) ACCURACY
# ============================================================
print("\n--- Computing V(t,w) accuracy ---")

with torch.no_grad():
    V_pred = model(t_flat, w_flat)
    V_analytical = merton_analytical_solution(t_flat, w_flat, T, gamma, rate, mu, sigma)

V_pred_np = V_pred.cpu().numpy().flatten()
V_analytical_np = V_analytical.cpu().numpy().flatten()

# R² for V
ss_res_v = np.sum((V_pred_np - V_analytical_np)**2)
ss_tot_v = np.sum((V_analytical_np - np.mean(V_analytical_np))**2)
V_r2 = 1 - ss_res_v / (ss_tot_v + 1e-12)

V_rel_error = np.mean(np.abs(V_pred_np - V_analytical_np) / (np.abs(V_analytical_np) + 1e-12)) * 100

print(f"  V(t,w) R²:             {V_r2:.6f}")
print(f"  V(t,w) Rel Error (%): {V_rel_error:.4f}%")

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  pi* MAE (%):           {pi_mae_percent:.4f}%  {'✅' if pi_mae_percent < 1.0 else '❌'}")
print(f"  phi(t) R²:             {phi_r2:.6f}")
print(f"  phi(t) Rel Error (%):  {phi_rel_error:.4f}%")
print(f"  PDE MSE:               {pde_mse:.6e}")
print(f"  Terminal MSE:          {terminal_mse:.6e}")
print(f"  V(t,w) R²:             {V_r2:.6f}")
print(f"  V(t,w) Rel Error (%):  {V_rel_error:.4f}%")
print(f"\nTarget: pi* MAE < 1%  →  {'PASS 🎉' if pi_mae_percent < 1.0 else 'NEED MORE TRAINING'}")
print("=" * 60)

# ============================================================
# GENERATE 6 GRAPHS
# ============================================================
print("\n--- Generating plots ---")

# Colormap for consistency
CMAP = 'viridis'

# --- GRAPH 1: Training Loss Convergence ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
loss_history = ckpt.get("loss_history", [])
if loss_history:
    ax1.plot(loss_history, color='blue', lw=1.5, label='Training Loss')
    best_epoch = np.argmin(loss_history)
    ax1.scatter(best_epoch, loss_history[best_epoch], color='red', s=80, 
                zorder=5, label=f'Best: epoch {best_epoch+1}, {loss_history[best_epoch]:.2e}')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax1.set_title('phi-PINN Training Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax1.legend(fontsize=10)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "01_loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Saved: 01_loss_curve.png")

# --- GRAPH 2: phi(t) Fit ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Sort by t for clean lines
t_sorted, t_idx = torch.sort(t_test.squeeze(), dim=0)
phi_pred_sorted = phi_pred_np[t_idx.cpu().numpy()]
phi_analytical_sorted = phi_analytical_np[t_idx.cpu().numpy()]
t_sorted_np = t_sorted.cpu().numpy()

ax2a = axes2[0]
ax2a.plot(t_sorted_np, phi_pred_sorted, 'b-', lw=2, label=r'Learned $\phi(t)$ (NN)')
ax2a.plot(t_sorted_np, phi_analytical_sorted, 'r--', lw=2, label=r'Analytical $\phi^*(t) = e^{\kappa(T-t)}$')
ax2a.set_xlabel('Time (t)', fontsize=12)
ax2a.set_ylabel(r'$\phi(t)$', fontsize=12)
ax2a.set_title(r'Learned $\phi(t)$ vs Analytical', fontsize=13, fontweight='bold')
ax2a.legend(fontsize=10)
ax2a.grid(True, alpha=0.3)

ax2b = axes2[1]
phi_abs_error = np.abs(phi_pred_sorted - phi_analytical_sorted)
ax2b.plot(t_sorted_np, phi_abs_error, 'purple', lw=1.5, label=f'Mean: {phi_abs_error.mean():.2e}')
ax2b.set_xlabel('Time (t)', fontsize=12)
ax2b.set_ylabel(r'$|\phi_{pred} - \phi^*|$', fontsize=12)
ax2b.set_title(r'$\phi(t)$ Absolute Error', fontsize=13, fontweight='bold')
ax2b.set_yscale('log')
ax2b.legend(fontsize=10)
ax2b.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "02_phi_fit.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  ✓ Saved: 02_phi_fit.png")

# --- GRAPH 3: pi* Distribution ---
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

ax3a = axes3[0]
pi_model_np = pi_model_test.cpu().numpy().flatten()
ax3a.hist(pi_model_np, bins=80, color='steelblue', edgecolor='black', alpha=0.7, density=True)
ax3a.axvline(pi_star_analytical, color='red', lw=3, ls='--', 
             label=f'Analytical $\pi^*$ = {pi_star_analytical:.4f}')
ax3a.axvline(pi_model_mean, color='green', lw=2, ls=':', 
             label=f'Model Mean $\pi^*$ = {pi_model_mean:.4f}')
ax3a.set_xlabel(r'Optimal Portfolio Weight $\pi^*$', fontsize=12)
ax3a.set_ylabel('Density', fontsize=12)
ax3a.set_title(r'$\pi^*$ Distribution: Model vs Analytical', fontsize=13, fontweight='bold')
ax3a.legend(fontsize=10)
ax3a.grid(True, alpha=0.3)

ax3b = axes3[1]
metrics_labels = ['MAE', 'MAE (%)']
metrics_values = [pi_mae, pi_mae_percent]
colors = ['coral' if v < (0.01 if i==0 else 1.0) else 'lightgray' for i, v in enumerate(metrics_values)]
bars = ax3b.bar(metrics_labels, metrics_values, color=colors, edgecolor='black', width=0.5)
# Add threshold lines
ax3b.axhline(y=0.01 if pi_mae < 0.1 else 0.01, color='red', ls='--', alpha=0.0)
ax3b.set_ylabel('Error', fontsize=12)
ax3b.set_title(r'$\pi^*$ Error Metrics', fontsize=13, fontweight='bold')
for bar, val in zip(bars, metrics_values):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
              f'{val:.6f}', ha='center', va='bottom', fontsize=10)
ax3b.grid(True, alpha=0.3, axis='y')

fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "03_pi_star_distribution.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  ✓ Saved: 03_pi_star_distribution.png")

# --- GRAPH 4: V(t,w) 3D Surface Comparison ---
fig4 = plt.figure(figsize=(16, 6))

t_mesh_np = t_mesh.cpu().numpy()
w_mesh_np = w_mesh.cpu().numpy()

V_pred_mesh = V_pred_np.reshape(50, 50)
V_analytical_mesh = V_analytical_np.reshape(50, 50)

ax4a = fig4.add_subplot(1, 2, 1, projection='3d')
surf1 = ax4a.plot_surface(t_mesh_np, w_mesh_np, V_pred_mesh, 
                          cmap=CMAP, alpha=0.9, linewidth=0, antialiased=True)
ax4a.set_xlabel('Time (t)', fontsize=10)
ax4a.set_ylabel('Wealth (w)', fontsize=10)
ax4a.set_zlabel('V(t,w)', fontsize=10)
ax4a.set_title('Model Prediction', fontsize=13, fontweight='bold')
fig4.colorbar(surf1, ax=ax4a, shrink=0.5, aspect=10)

ax4b = fig4.add_subplot(1, 2, 2, projection='3d')
surf2 = ax4b.plot_surface(t_mesh_np, w_mesh_np, V_analytical_mesh,
                          cmap=CMAP, alpha=0.9, linewidth=0, antialiased=True)
ax4b.set_xlabel('Time (t)', fontsize=10)
ax4b.set_ylabel('Wealth (w)', fontsize=10)
ax4b.set_zlabel('V(t,w)', fontsize=10)
ax4b.set_title('Analytical Solution', fontsize=13, fontweight='bold')
fig4.colorbar(surf2, ax=ax4b, shrink=0.5, aspect=10)

fig4.suptitle(rf'Value Function $V(t,w) = \phi(t) \cdot U(w)$  (R² = {V_r2:.6f})', 
              fontsize=14, fontweight='bold', y=1.02)
fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / "04_value_function_3d.png", dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  ✓ Saved: 04_value_function_3d.png")

# --- GRAPH 5: PDE Residual Heatmap ---
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

with torch.no_grad():
    pde_res_grid = compute_phi_pde_residual(model.phi_net, t_flat, rate, mu, sigma, gamma)
pde_res_np = pde_res_grid.cpu().numpy().reshape(50, 50)
pde_res_abs = np.abs(pde_res_np)

ax5a = axes5[0]
c1 = ax5a.contourf(t_mesh_np, w_mesh_np, pde_res_abs, 50, cmap='magma')
ax5a.set_xlabel('Time (t)', fontsize=12)
ax5a.set_ylabel('Wealth (w)', fontsize=12)
ax5a.set_title(r'|PDE Residual| $|\phi''(t) + \kappa\phi(t)|$', fontsize=12, fontweight='bold')
fig5.colorbar(c1, ax=ax5a)

ax5b = axes5[1]
ax5b.hist(pde_res_np.flatten(), bins=80, color='steelblue', edgecolor='black', alpha=0.7)
ax5b.axvline(0, color='red', ls='--', lw=2, label='Zero residual')
ax5b.axvline(pde_mean, color='green', ls=':', lw=2, label=f'Mean: {pde_mean:.2e}')
ax5b.set_xlabel('PDE Residual', fontsize=12)
ax5b.set_ylabel('Frequency', fontsize=12)
ax5b.set_title(f'PDE Residual Distribution (MSE = {pde_mse:.2e})', fontsize=12, fontweight='bold')
ax5b.legend(fontsize=10)
ax5b.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig(OUTPUT_DIR / "05_pde_residual.png", dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  ✓ Saved: 05_pde_residual.png")

# --- GRAPH 6: All Metrics Summary Dashboard ---
fig6, axes6 = plt.subplots(2, 3, figsize=(16, 10))

# 6a: pi* Comparison
ax = axes6[0, 0]
pi_bars = ax.bar(['Model', 'Analytical'], [pi_model_mean, pi_star_analytical], 
                  color=['steelblue', 'coral'], edgecolor='black', width=0.5)
ax.axhline(pi_model_mean, color='steelblue', ls=':', alpha=0.0)
ax.set_ylabel(r'$\pi^*$', fontsize=11)
ax.set_title(rf'$\pi^*$: Model vs Analytical', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(pi_bars, [pi_model_mean, pi_star_analytical]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 6b: pi* MAE gauge
ax = axes6[0, 1]
mae_pct = pi_mae_percent
threshold = 1.0
color = 'green' if mae_pct < threshold else 'red'
ax.barh(['$\pi^*$ MAE'], [mae_pct], color=color, edgecolor='black', height=0.4)
ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Target: {threshold}%')
ax.set_xlabel('Error (%)', fontsize=11)
ax.set_title(f'$\pi^*$ MAE: {mae_pct:.4f}%', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, max(mae_pct * 2, threshold * 2))
ax.grid(True, alpha=0.3, axis='x')

# 6c: phi(t) accuracy
ax = axes6[0, 2]
phi_metrics = [phi_r2, 100 - phi_rel_error]
phi_labels = ['R²', '1 - RelErr%']
ax.bar(phi_labels, phi_metrics, color=['steelblue', 'teal'], edgecolor='black', width=0.4)
ax.axhline(y=1.0, color='red', ls='--', alpha=0.0)
ax.set_ylabel('Score', fontsize=11)
ax.set_title(rf'$\phi(t)$ Accuracy (R² = {phi_r2:.6f})', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

# 6d: PDE & Terminal Loss
ax = axes6[1, 0]
loss_metrics = [pde_mse, terminal_mse]
loss_labels = ['PDE MSE', 'Terminal MSE']
ax.bar(loss_labels, loss_metrics, color=['steelblue', 'coral'], edgecolor='black', width=0.4)
ax.set_yscale('log')
ax.set_ylabel('MSE (Log Scale)', fontsize=11)
ax.set_title('PDE & Terminal Condition Loss', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(ax.patches, loss_metrics):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
            f'{val:.2e}', ha='center', va='bottom', fontsize=8)

# 6e: V(t,w) accuracy
ax = axes6[1, 1]
V_metrics = [V_r2, max(0, 100 - V_rel_error)]
V_labels = ['R²', '1 - RelErr%']
colors_v = ['steelblue', 'teal']
ax.bar(V_labels, V_metrics, color=colors_v, edgecolor='black', width=0.4)
ax.set_ylabel('Score', fontsize=11)
ax.set_title(f'V(t,w) Accuracy (R² = {V_r2:.6f})', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

# 6f: Final verdict
ax = axes6[1, 2]
ax.axis('off')
verdict = 'PASS ✅' if pi_mae_percent < 1.0 else 'NEED MORE TRAINING ⚠️'
color_v = 'green' if pi_mae_percent < 1.0 else 'orange'
ax.text(0.5, 0.7, 'FINAL VERDICT', fontsize=16, fontweight='bold', ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.4, verdict, fontsize=24, fontweight='bold', ha='center', color=color_v,
        transform=ax.transAxes)
ax.text(0.5, 0.15, f'$\pi^*$ MAE = {pi_mae_percent:.4f}%  (Target < 1%)', 
        fontsize=11, ha='center', transform=ax.transAxes)

fig6.suptitle('phi-PINN Evaluation Dashboard', fontsize=15, fontweight='bold', y=1.01)
fig6.tight_layout()
fig6.savefig(OUTPUT_DIR / "06_evaluation_dashboard.png", dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  ✓ Saved: 06_evaluation_dashboard.png")

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("Evaluation complete!")