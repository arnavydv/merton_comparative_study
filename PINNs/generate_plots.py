"""Standalone script to generate all 6 evaluation plots."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evaluation_plots'
OUT.mkdir(exist_ok=True)

from neural_network import ValueFunc
from loss_function import compute_phi_pde_residual
from terminal_condition import merton_analytical_solution
from collocation_points import collocation_points

with open(ROOT / 'data_accumulation' / 'config.json') as f:
    cfg = json.load(f)

rate, sigma, mu, gamma = cfg['rate'], cfg['sigma'], cfg['mu'], cfg['gamma']
T, W_MIN, W_MAX = 1.0, 0.1, 2.0
pi_star = (mu - rate) / (sigma**2 * gamma)
kappa = (1-gamma)*(rate+0.5*(mu-rate)**2/(sigma**2*gamma))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ValueFunc(gamma=gamma).to(device)
ckpt = torch.load(ROOT / 'saved_models' / 'best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded model, loss={ckpt["loss"]:.2e}')

# Data
w, t = collocation_points(10000, T, W_MAX, W_MIN)
w, t = w.to(device), t.to(device)

# Grid
tg = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
wg = torch.linspace(W_MIN, W_MAX, 50, device=device).unsqueeze(1)
tm, wm = torch.meshgrid(tg.squeeze(), wg.squeeze(), indexing='xy')
tf, wf = tm.reshape(-1,1), wm.reshape(-1,1)

# pi*
def pi_star_fn(m, t, w):
    tr, wr = t.clone().detach().requires_grad_(True), w.clone().detach().requires_grad_(True)
    V = m(tr, wr)
    Vw = torch.autograd.grad(V, wr, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    Vww = torch.autograd.grad(Vw, wr, grad_outputs=torch.ones_like(Vw), create_graph=True, retain_graph=True)[0]
    return (-(mu-rate)*Vw/(sigma**2*wr*(Vww-1e-8))).detach()

pi_m = pi_star_fn(model, t, w)
pi_mean, pi_std = pi_m.mean().item(), pi_m.std().item()
pi_mae_pct = 100*abs(pi_mean-pi_star)/abs(pi_star)

with torch.no_grad():
    phi_p = model.phi_net(t)
    phi_a = torch.exp(kappa*(T-t))
    Vp = model(tf, wf)
    Va = merton_analytical_solution(tf, wf, T, gamma, rate, mu, sigma)

# PDE residual needs requires_grad on t
tf_grad = tf.clone().detach().requires_grad_(True)
pde_g = compute_phi_pde_residual(model.phi_net, tf_grad, rate, mu, sigma, gamma)
pde_g = pde_g.detach()
pde_mse = torch.mean(pde_g**2).item()

tT = torch.ones(1000,1,device=device)*T
with torch.no_grad():
    phiT = model.phi_net(tT)
term_mse = torch.mean((phiT-1)**2).item()

# Sort for phi plot (detach t first since collocation_points return requires_grad tensors)
t_detached = t.detach()
ts, idx = torch.sort(t_detached.squeeze(), dim=0)
idx_n = idx.detach().cpu().numpy()
pp_np = phi_p.cpu().numpy().flatten()[idx_n]
pa_np = phi_a.cpu().numpy().flatten()[idx_n]
ts_np = ts.detach().cpu().numpy()

# Compute R2
phi_r2 = 1 - np.sum((pp_np-pa_np)**2)/(np.sum((pa_np-np.mean(pa_np))**2)+1e-12)
V_r2 = 1 - np.sum((Vp.cpu().numpy().flatten()-Va.cpu().numpy().flatten())**2)/(np.sum((Va.cpu().numpy().flatten()-np.mean(Va.cpu().numpy().flatten()))**2)+1e-12)

print(f'pi* MAE% = {pi_mae_pct:.6f}% [target < 1%]')
print(f'phi R2 = {phi_r2:.6f}, V R2 = {V_r2:.6f}')
print(f'PDE MSE = {pde_mse:.2e}, Terminal MSE = {term_mse:.2e}')

# === PLOT 1: Loss Curve ===
fig, ax = plt.subplots(figsize=(8,5))
lh = ckpt['loss_history']
ax.plot(lh, 'b-', lw=1.5)
be = np.argmin(lh)
ax.scatter(be, lh[be], c='r', s=80, zorder=5, label=f'Best: epoch {be+1}, {lh[be]:.2e}')
ax.set_yscale('log')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (Log Scale)')
ax.set_title('phi-PINN Training Convergence')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/'01_loss_curve.png', dpi=150); plt.close(fig)
print('1/6: Loss curve')

# === PLOT 2: phi Fit ===
fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].plot(ts_np, pp_np, 'b-', lw=2, label='Learned phi(t) (NN)')
axes[0].plot(ts_np, pa_np, 'r--', lw=2, label='Analytical phi*(t)')
axes[0].set_xlabel('Time (t)'); axes[0].set_ylabel('phi(t)')
axes[0].set_title('phi(t): NN vs Analytical'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
phi_err = np.abs(pp_np-pa_np)
axes[1].plot(ts_np, phi_err, 'purple', lw=1.5, label=f'Mean: {phi_err.mean():.2e}')
axes[1].set_yscale('log'); axes[1].set_xlabel('Time (t)'); axes[1].set_ylabel('|phi_pred - phi*|')
axes[1].set_title('phi(t) Absolute Error'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/'02_phi_fit.png', dpi=150); plt.close(fig)
print('2/6: Phi fit')

# === PLOT 3: pi* Distribution ===
fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].hist(pi_m.cpu().numpy().flatten(), bins=80, color='steelblue', edgecolor='black', alpha=0.7, density=True)
axes[0].axvline(pi_star, color='red', lw=3, ls='--', label=f'Analytical = {pi_star:.4f}')
axes[0].axvline(pi_mean, color='green', lw=2, ls=':', label=f'Model Mean = {pi_mean:.4f}')
axes[0].set_xlabel('pi* (Optimal Portfolio Weight)'); axes[0].set_ylabel('Density')
axes[0].set_title('pi* Distribution: Model vs Analytical')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
bars = axes[1].bar(['pi* MAE (%)'], [pi_mae_pct], color='green' if pi_mae_pct < 1.0 else 'red', edgecolor='black', width=0.4)
axes[1].axhline(1.0, color='red', ls='--', lw=2, label='Target: 1%')
axes[1].set_title(f'pi* MAE = {pi_mae_pct:.6f}% {"(PASS)" if pi_mae_pct < 1.0 else "(FAIL)"}')
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig(OUT/'03_pi_star_distribution.png', dpi=150); plt.close(fig)
print('3/6: Pi* distribution')

# === PLOT 4: 3D V(t,w) ===
fig = plt.figure(figsize=(16,6))
tm_np, wm_np = tm.detach().cpu().numpy(), wm.detach().cpu().numpy()
ax1 = fig.add_subplot(1,2,1, projection='3d')
s1 = ax1.plot_surface(tm_np, wm_np, Vp.cpu().numpy().reshape(50,50), cmap='viridis', alpha=0.9)
ax1.set_title('Model V(t,w)'); ax1.set_xlabel('t'); ax1.set_ylabel('w')
fig.colorbar(s1, ax=ax1, shrink=0.5)
ax2 = fig.add_subplot(1,2,2, projection='3d')
s2 = ax2.plot_surface(tm_np, wm_np, Va.cpu().numpy().reshape(50,50), cmap='viridis', alpha=0.9)
ax2.set_title('Analytical V(t,w)'); ax2.set_xlabel('t'); ax2.set_ylabel('w')
fig.colorbar(s2, ax=ax2, shrink=0.5)
fig.suptitle(f'Value Function V(t,w) = phi(t)*U(w)  (R2 = {V_r2:.6f})', fontsize=14, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/'04_value_function_3d.png', dpi=150); plt.close(fig)
print('4/6: 3D value function')

# === PLOT 5: PDE Residual ===
fig, axes = plt.subplots(1, 2, figsize=(14,5))
pde_np = pde_g.cpu().numpy().reshape(50,50)
c = axes[0].contourf(tm_np, wm_np, np.abs(pde_np), 50, cmap='magma')
axes[0].set_xlabel('t'); axes[0].set_ylabel('w')
axes[0].set_title('|PDE Residual|'); fig.colorbar(c, ax=axes[0])
axes[1].hist(pde_np.flatten(), bins=80, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', ls='--', lw=2, label='Zero residual')
axes[1].axvline(pde_g.mean().item(), color='green', ls=':', lw=2, label=f'Mean: {pde_g.mean().item():.2e}')
axes[1].set_xlabel('PDE Residual'); axes[1].set_ylabel('Frequency')
axes[1].set_title(f'PDE Residual Distribution (MSE = {pde_mse:.2e})')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/'05_pde_residual.png', dpi=150); plt.close(fig)
print('5/6: PDE residual')

# === PLOT 6: Dashboard ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes[0,0].bar(['Model', 'Analytical'], [pi_mean, pi_star], color=['steelblue','coral'], edgecolor='black', width=0.5)
axes[0,0].set_title('pi*: Model vs Analytical'); axes[0,0].grid(True, alpha=0.3, axis='y')
axes[0,1].bar(['phi R2', 'V R2'], [phi_r2, V_r2], color=['teal','steelblue'], edgecolor='black', width=0.4)
axes[0,1].set_ylim(0,1.05); axes[0,1].set_title(f'R2 Scores'); axes[0,1].grid(True, alpha=0.3, axis='y')
axes[0,2].bar(['PDE MSE', 'Terminal MSE'], [pde_mse, term_mse], color=['steelblue','coral'], edgecolor='black', width=0.4)
axes[0,2].set_yscale('log'); axes[0,2].set_title('Loss Components'); axes[0,2].grid(True, alpha=0.3, axis='y')
axes[1,0].bar(['pi* MAE%'],[pi_mae_pct], color='green' if pi_mae_pct<1.0 else 'red', edgecolor='black', width=0.4)
axes[1,0].axhline(1.0, color='red', ls='--', lw=2, label='1% Target')
axes[1,0].set_title(f'pi* MAE = {pi_mae_pct:.4f}%'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')
axes[1,1].bar(['pi* Std'], [pi_std], color='steelblue', edgecolor='black', width=0.4)
axes[1,1].set_title(f'pi* Std = {pi_std:.6f}'); axes[1,1].grid(True, alpha=0.3, axis='y')
axes[1,2].axis('off')
axes[1,2].text(0.5, 0.55, 'FINAL VERDICT', fontsize=16, fontweight='bold', ha='center', transform=axes[1,2].transAxes)
color_v = 'green' if pi_mae_pct < 1.0 else 'orange'
axes[1,2].text(0.5, 0.35, 'PASS' if pi_mae_pct < 1.0 else 'FAIL', fontsize=28, fontweight='bold',
               ha='center', color=color_v, transform=axes[1,2].transAxes)
axes[1,2].text(0.5, 0.15, f'pi* MAE = {pi_mae_pct:.4f}%  (Target < 1%)',
               fontsize=11, ha='center', transform=axes[1,2].transAxes)
fig.suptitle('phi-PINN Evaluation Dashboard', fontsize=15, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/'06_evaluation_dashboard.png', dpi=150); plt.close(fig)
print('6/6: Dashboard')

print(f'\nAll 6 plots saved to: {OUT}/')
print([f.name for f in sorted(OUT.iterdir()) if f.suffix == '.png'])
print('DONE!')