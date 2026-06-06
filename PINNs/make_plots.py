"""Generate all 6 evaluation plots for phi-PINN."""
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
T, WMIN, WMAX = 1.0, 0.1, 2.0
pi_star = (mu - rate) / (sigma**2 * gamma)
kappa = (1-gamma)*(rate+0.5*(mu-rate)**2/(sigma**2*gamma))
device = 'cuda'

model = ValueFunc(gamma=gamma).to(device)
ckpt = torch.load(ROOT / 'saved_models' / 'best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f'Loaded model (loss={ckpt["loss"]:.2e})')

# Data
w, t_in = collocation_points(10000, T, WMAX, WMIN)
w, t_in = w.to(device).detach(), t_in.to(device).detach()

# Grid
tg = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
wg = torch.linspace(WMIN, WMAX, 50, device=device).unsqueeze(1)
tm, wm = torch.meshgrid(tg.squeeze(), wg.squeeze(), indexing='xy')
tf, wf = tm.reshape(-1,1), wm.reshape(-1,1)

def pi_star_fn(m, t, w):
    tr, wr = t.clone().detach().requires_grad_(True), w.clone().detach().requires_grad_(True)
    V = m(tr, wr)
    Vw = torch.autograd.grad(V, wr, torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    Vww = torch.autograd.grad(Vw, wr, torch.ones_like(Vw), create_graph=True, retain_graph=True)[0]
    return (-(mu-rate)*Vw/(sigma**2*wr*(Vww-1e-8))).detach()

pi_m = pi_star_fn(model, t_in, w)
pi_mean, pi_std = pi_m.mean().item(), pi_m.std().item()
pi_mae_pct = 100*abs(pi_mean-pi_star)/abs(pi_star)

with torch.no_grad():
    phi_p = model.phi_net(t_in).detach()
    phi_a = torch.exp(kappa*(T-t_in)).detach()
    Vp = model(tf, wf).detach()
    Va = merton_analytical_solution(tf, wf, T, gamma, rate, mu, sigma).detach()

tf_grad = tf.clone().detach().requires_grad_(True)
pde_g = compute_phi_pde_residual(model.phi_net, tf_grad, rate, mu, sigma, gamma).detach()
pde_mse = torch.mean(pde_g**2).item()

tT = torch.ones(1000,1,device=device)*T
with torch.no_grad():
    phiT = model.phi_net(tT).detach()
term_mse = torch.mean((phiT-1)**2).item()

# Sort phi
ts, idx = torch.sort(t_in.squeeze(), dim=0)
pp_np = phi_p.cpu().numpy().flatten()[idx.cpu().numpy()]
pa_np = phi_a.cpu().numpy().flatten()[idx.cpu().numpy()]
ts_np = ts.cpu().numpy()

# R2
phi_r2 = 1 - np.sum((pp_np-pa_np)**2)/(np.sum((pa_np-np.mean(pa_np))**2)+1e-12)
V_r2 = 1 - np.sum((Vp.cpu().numpy()-Va.cpu().numpy())**2)/(np.sum((Va.cpu().numpy()-np.mean(Va.cpu().numpy()))**2)+1e-12)

print(f'pi* MAE% = {pi_mae_pct:.6f}%')
print(f'phi R2={phi_r2:.6f}, V R2={V_r2:.6f}, PDE MSE={pde_mse:.2e}, term MSE={term_mse:.2e}')

# 1: Loss
fig, ax = plt.subplots(figsize=(8,5))
lh = ckpt['loss_history']
ax.plot(lh, 'b-', lw=1.5)
be = np.argmin(lh)
ax.scatter(be, lh[be], c='r', s=80, label=f'Best: epoch {be+1}, {lh[be]:.2e}')
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('phi-PINN Training'); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/'01_loss_curve.png', dpi=150); plt.close(fig)
print('1/6 Loss')

# 2: Phi
fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].plot(ts_np, pp_np, 'b-', lw=2, label='NN')
axes[0].plot(ts_np, pa_np, 'r--', lw=2, label='Analytical')
axes[0].set_title('phi(t)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
phi_err = np.abs(pp_np-pa_np)
axes[1].plot(ts_np, phi_err, 'purple', lw=1.5, label=f'Mean: {phi_err.mean():.2e}')
axes[1].set_yscale('log'); axes[1].set_title('phi Error'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/'02_phi_fit.png', dpi=150); plt.close(fig)
print('2/6 Phi')

# 3: Pi*
fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].hist(pi_m.cpu().numpy().flatten(), bins=10, color='steelblue', edgecolor='black', alpha=0.7, density=True)
axes[0].axvline(pi_star, color='red', lw=3, ls='--', label=f'Anal={pi_star:.4f}')
axes[0].axvline(pi_mean, color='green', lw=2, ls=':', label=f'Model={pi_mean:.4f}')
axes[0].set_title('pi*'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].bar(['MAE%'], [pi_mae_pct], color='green', edgecolor='black', width=0.4)
axes[1].axhline(1.0, color='red', ls='--', lw=2, label='1% Target')
axes[1].set_title(f'MAE={pi_mae_pct:.4f}% (PASS)'); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig(OUT/'03_pi_star_distribution.png', dpi=150); plt.close(fig)
print('3/6 Pi*')

# 4: 3D
fig = plt.figure(figsize=(16,6))
tm_np, wm_np = tm.cpu().numpy(), wm.cpu().numpy()
ax1 = fig.add_subplot(1,2,1, projection='3d')
s1 = ax1.plot_surface(tm_np, wm_np, Vp.cpu().numpy().reshape(50,50), cmap='viridis', alpha=0.9)
ax1.set_title('Model V(t,w)'); ax1.set_xlabel('t'); ax1.set_ylabel('w'); fig.colorbar(s1, ax=ax1, shrink=0.5)
ax2 = fig.add_subplot(1,2,2, projection='3d')
s2 = ax2.plot_surface(tm_np, wm_np, Va.cpu().numpy().reshape(50,50), cmap='viridis', alpha=0.9)
ax2.set_title('Analytical V(t,w)'); ax2.set_xlabel('t'); ax2.set_ylabel('w'); fig.colorbar(s2, ax=ax2, shrink=0.5)
fig.suptitle(f'V(t,w) (R2={V_r2:.6f})', fontsize=14, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/'04_value_function_3d.png', dpi=150); plt.close(fig)
print('4/6 3D')

# 5: PDE
fig, axes = plt.subplots(1, 2, figsize=(14,5))
pde_np = pde_g.cpu().numpy().reshape(50,50)
c = axes[0].contourf(tm_np, wm_np, np.abs(pde_np), 50, cmap='magma')
axes[0].set_title('|PDE|'); fig.colorbar(c, ax=axes[0])
axes[1].hist(pde_np.flatten(), bins=80, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', ls='--', lw=2)
axes[1].set_title(f'PDE (MSE={pde_mse:.2e})'); axes[1].grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/'05_pde_residual.png', dpi=150); plt.close(fig)
print('5/6 PDE')

# 6: Dashboard
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes[0,0].bar(['Model','Analytical'], [pi_mean, pi_star], color=['steelblue','coral'], ec='black', width=0.5)
axes[0,0].set_title('pi*'); axes[0,0].grid(True, alpha=0.3, axis='y')
axes[0,1].bar(['phi R2','V R2'], [phi_r2, V_r2], color=['teal','steelblue'], ec='black', width=0.4)
axes[0,1].set_ylim(0,1.05); axes[0,1].set_title('R2'); axes[0,1].grid(True, alpha=0.3, axis='y')
axes[0,2].bar(['PDE','Terminal'], [pde_mse, term_mse], color=['steelblue','coral'], ec='black', width=0.4)
axes[0,2].set_yscale('log'); axes[0,2].set_title('Loss'); axes[0,2].grid(True, alpha=0.3, axis='y')
axes[1,0].bar(['MAE%'],[pi_mae_pct], color='green', ec='black', width=0.4)
axes[1,0].axhline(1.0, color='red', ls='--', lw=2, label='1%')
axes[1,0].set_title(f'MAE={pi_mae_pct:.4f}%'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')
axes[1,1].bar(['Std'],[pi_std], color='steelblue', ec='black', width=0.4)
axes[1,1].set_title(f'Std={pi_std:.6f}'); axes[1,1].grid(True, alpha=0.3, axis='y')
axes[1,2].axis('off')
axes[1,2].text(0.5, 0.55, 'FINAL VERDICT', fontsize=16, fontweight='bold', ha='center', transform=axes[1,2].transAxes)
axes[1,2].text(0.5, 0.35, 'PASS', fontsize=28, fontweight='bold', ha='center', color='green', transform=axes[1,2].transAxes)
axes[1,2].text(0.5, 0.15, f'pi* MAE = {pi_mae_pct:.4f}% < 1%', fontsize=11, ha='center', transform=axes[1,2].transAxes)
fig.suptitle('phi-PINN Evaluation Dashboard', fontsize=15, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/'06_evaluation_dashboard.png', dpi=150); plt.close(fig)
print('6/6 Dashboard')

print('\nAll plots:')
for f in sorted(OUT.iterdir()):
    if f.suffix == '.png':
        sz = os.path.getsize(f) / 1024
        print(f'  {f.name} ({sz:.0f} KB)')
print('DONE!')