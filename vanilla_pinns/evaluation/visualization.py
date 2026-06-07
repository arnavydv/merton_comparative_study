import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress minor matplotlib warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

#=============================================================================
# Phase 1: Market Simulation & Forward Dynamics
#=============================================================================
def plot_forward_dynamics(time_grid, W, wealth_grid, max_paths=50):
    """Plots Brownian paths and wealth dynamics from the forward simulation."""
    time_np = time_grid
    W_np = W[:max_paths].cpu().detach().numpy()
    wealth_np = wealth_grid[:max_paths].cpu().detach().numpy()
    final_wealth = wealth_grid[:, -1].cpu().detach().numpy()
    
    fig = plt.figure(figsize=(15, 10))

    # 1. Brownian Motion Paths
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(W_np.shape[0]):
        ax1.plot(time_np, W_np[i], lw=0.5, alpha=0.6)
    ax1.set_title("1. Brownian Motion Paths (W_t)")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("W(t)")

    # 2 & 3. Controlled Wealth Dynamics
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(wealth_np.shape[0]):
        ax2.plot(time_np, wealth_np[i], lw=0.5, alpha=0.6)
    ax2.set_title("2 & 3. Controlled Wealth Dynamics")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Wealth")

    # 4. Terminal Wealth Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(final_wealth, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.set_title("4. Terminal Wealth Distribution (t=T)")
    ax3.set_xlabel("Final Wealth")
    ax3.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

#=============================================================================
# Phase 2: Neural Network Training Diagnostics
#=============================================================================
def plot_training_diagnostics(loss_history, pde_loss_history, terminal_loss_history, 
                              concavity_loss_history, mono_loss_history):
    """Plots the convergence metrics of the neural network."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    
    losses_to_plot = [
        ("Total Loss", loss_history, 'red'),
        ("PDE (HJB) Loss", pde_loss_history, 'blue'),
        ("Terminal Loss", terminal_loss_history, 'green'),
        ("Concavity Loss", concavity_loss_history, 'orange'),
        ("Monotonicity Loss", mono_loss_history, 'purple')
    ]
    
    for ax, (title, hist, color) in zip(axes, losses_to_plot):
        if hist is not None and len(hist) > 0:
            ax.semilogy(epochs, hist, color=color, lw=2)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (Log Scale)")
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

#=============================================================================
# Phase 3: Value Function & Optimal Policy Validation
#=============================================================================
def plot_value_function_surface(t_mesh, W_mesh, V_pred, T=1.0):
    """Visualizes the learned Value Function surface."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t_mesh, W_mesh, V_pred, cmap='plasma')
    ax.set_title("Learned Value Function V(t, w)")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    ax.set_zlabel("V(t, w)")
    plt.show()

def plot_policy_validation(t_mesh, W_mesh, pi_pred, exact_pi, T=1.0):
    """Generates 3D surfaces and heatmaps comparing learned vs exact policy."""
    fig = plt.figure(figsize=(18, 6))

    # 1. 3D Surface vs Analytical Truth
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(t_mesh, W_mesh, pi_pred, cmap='viridis', alpha=0.8)
    exact_plane = np.full_like(pi_pred, exact_pi)
    ax1.plot_surface(t_mesh, W_mesh, exact_plane, color='red', alpha=0.3)
    ax1.set_title("Learned Policy vs Exact Truth")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Wealth (w)")
    ax1.set_zlabel("pi*")

    # 2. Policy Error Heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    error = np.abs(pi_pred - exact_pi)
    c = ax2.contourf(t_mesh, W_mesh, error, 50, cmap='magma')
    fig.colorbar(c, ax=ax2)
    ax2.set_title("Absolute Error Heatmap")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Wealth (w)")

    # 3. Cross-Sectional Policy Slices
    ax3 = fig.add_subplot(1, 3, 3)
    t_np = np.linspace(0.01, T, pi_pred.shape[0])
    time_indices = [5, pi_pred.shape[0]//2, pi_pred.shape[0]-1]  # Approx early, mid, late times
    for tidx in time_indices:
        ax3.plot(W_mesh[0, :], pi_pred[tidx, :], label=f"t = {t_np[tidx]:.2f}")
    ax3.axhline(exact_pi, color='red', linestyle='--', label="Exact Analytical")
    ax3.set_title("Cross-Sectional Slices of pi*")
    ax3.set_xlabel("Wealth (w)")
    ax3.set_ylabel("pi*")
    ax3.legend()
    ax3.set_ylim(exact_pi * 0.5, exact_pi * 1.5)

    plt.tight_layout()
    plt.show()

#=============================================================================
# Phase 4: Financial Sensitivity
#=============================================================================
def plot_financial_sensitivity(mu, r, base_gamma, base_sigma):
    """Plots how the analytical exact policy changes under different regimes."""
    gammas = np.linspace(1.1, 5.0, 100)
    sigmas = np.linspace(0.05, 0.5, 100)
    pi_gammas = (mu - r) / (gammas * base_sigma**2)
    pi_sigmas = (mu - r) / (base_gamma * sigmas**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gammas, pi_gammas, color='purple', lw=2)
    ax1.set_title("Optimal Policy vs Risk Aversion (gamma)")
    ax1.set_xlabel("Gamma (Risk Aversion)")
    ax1.set_ylabel("pi*")

    ax2.plot(sigmas, pi_sigmas, color='orange', lw=2)
    ax2.set_title("Optimal Policy vs Market Volatility (sigma)")
    ax2.set_xlabel("Sigma (Volatility)")
    ax2.set_ylabel("pi*")

    plt.tight_layout()
    plt.show()

#=============================================================================
# Phase 5: Additional pi*-Centric Diagnostic Plots (10 new graphs)
#=============================================================================
#
# --- Group A: Random Inference Point Diagnostics ---

def plot_pi_histogram(pi_pred, exact_pi):
    """Graph #1: Distribution of predicted pi* across (w,t) space."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pi_pred, bins=40, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    ax.axvline(exact_pi, color='red', linestyle='--', linewidth=2, label=f'Exact pi* = {exact_pi:.4f}')
    ax.axvline(pi_pred.mean(), color='darkgreen', linestyle=':', linewidth=2, label=f'Mean pred = {pi_pred.mean():.4f}')
    ax.set_xlabel("pi* (Proportion in Risky Asset)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Learned pi* Across (w, t) Space")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pi_vs_wealth(w_np, pi_pred, exact_pi, t_np):
    """Graph #2: pi* vs wealth (w), colored by time t."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(w_np, pi_pred, c=t_np, cmap='viridis', alpha=0.6, s=15)
    ax.axhline(exact_pi, color='red', linestyle='--', linewidth=2, label=f'Exact pi* = {exact_pi:.4f}')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Time (t)")
    ax.set_xlabel("Wealth (w)")
    ax.set_ylabel("pi* (Proportion in Risky Asset)")
    ax.set_title("Learned pi* vs Wealth (colored by Time)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pi_vs_time(t_np, pi_pred, exact_pi, w_np):
    """Graph #3: pi* vs time (t), colored by wealth w."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(t_np, pi_pred, c=w_np, cmap='plasma', alpha=0.6, s=15)
    ax.axhline(exact_pi, color='red', linestyle='--', linewidth=2, label=f'Exact pi* = {exact_pi:.4f}')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Wealth (w)")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("pi* (Proportion in Risky Asset)")
    ax.set_title("Learned pi* vs Time (colored by Wealth)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_exact_scatter(pi_pred, exact_pi):
    """Graph #4: Predicted pi* vs Exact pi* (identity-line scatter)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(np.full_like(pi_pred, exact_pi), pi_pred, alpha=0.5, s=15, color='teal')
    ax.plot([exact_pi*0.9, exact_pi*1.1], [exact_pi*0.9, exact_pi*1.1], 'r--', linewidth=2, label='Identity')
    ax.set_xlabel("Exact pi*")
    ax.set_ylabel("Predicted pi*")
    ax.set_title("Predicted vs Exact Optimal Policy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pi_error_profile(w_np, t_np, pi_pred, exact_pi):
    """Graph #5: Absolute |pi* - exact| vs wealth, colored by time."""
    errors = np.abs(pi_pred - exact_pi)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(w_np, errors, c=t_np, cmap='magma', alpha=0.6, s=15)
    ax.axhline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Abs Error = {errors.mean():.5f}')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Time (t)")
    ax.set_xlabel("Wealth (w)")
    ax.set_ylabel("|Predicted pi* - Exact pi*|")
    ax.set_title("Absolute Prediction Error vs Wealth (colored by Time)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Group B: Grid Evaluation Derivative Diagnostics ---

def plot_vw_surface(t_mesh, w_mesh, v_w_mesh):
    """Graph #6: 3D surface of V_w(t,w) — marginal value of wealth."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t_mesh, w_mesh, v_w_mesh, cmap='coolwarm')
    ax.set_title("Marginal Value of Wealth V_w(t, w)")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    ax.set_zlabel("V_w")
    plt.show()

def plot_implied_risk_aversion(t_mesh, w_mesh, v_w_mesh, v_ww_mesh, w_grid_2d):
    """Graph #7: Implied relative risk aversion gamma = -w * V_ww / V_w (should ≈ 5 for CRRA)."""
    # Avoid division by zero at V_w ≈ 0
    gamma_implied = -w_grid_2d * v_ww_mesh / (v_w_mesh + 1e-10)
    # Clip extreme values for visualisation
    gamma_implied = np.clip(gamma_implied, 0.0, 20.0)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(t_mesh, w_mesh, gamma_implied, cmap='viridis')
    ax1.set_title("Implied RRA = -w·V_ww / V_w")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Wealth (w)")
    ax1.set_zlabel("Implied γ")
    
    ax2 = fig.add_subplot(1, 2, 2)
    c = ax2.contourf(t_mesh, w_mesh, gamma_implied, 50, cmap='viridis')
    fig.colorbar(c, ax=ax2)
    ax2.set_title("Implied RRA Heatmap")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Wealth (w)")
    
    plt.tight_layout()
    plt.show()

def plot_signed_error_heatmap(t_mesh, w_mesh, pi_mesh, exact_pi):
    """Graph #8: Signed error (pi* - exact) heatmap — shows over/under allocation."""
    signed_error = pi_mesh - exact_pi
    vmax = max(abs(signed_error.min()), abs(signed_error.max()))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(t_mesh, w_mesh, signed_error, 50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    fig.colorbar(c, ax=ax, label="pi* - Exact pi*")
    ax.set_title("Signed Policy Error: Over vs Under Allocation")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    plt.tight_layout()
    plt.show()


# --- Group C: Forward Simulation & Policy Impact ---

def plot_wealth_comparison(time_grid, wealth_exact, wealth_learned, wealth_rf, max_paths=30):
    """Graph #9: Wealth path comparison under 3 policies (exact pi*, learned pi*, risk-free)."""
    time_np = time_grid
    wealth_exact_np = wealth_exact[:max_paths].cpu().detach().numpy()
    wealth_learned_np = wealth_learned[:max_paths].cpu().detach().numpy()
    wealth_rf_np = wealth_rf[:max_paths].cpu().detach().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    
    # Single representative path overlay
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time_np, wealth_exact_np[0], 'b-', linewidth=2, label='Exact pi*')
    ax1.plot(time_np, wealth_learned_np[0], 'g--', linewidth=2, label='Learned pi*')
    ax1.plot(time_np, wealth_rf_np[0], 'r-.', linewidth=2, label='Risk-Free (pi=0)')
    ax1.set_title("Single Representative Wealth Path")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Wealth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean wealth paths
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_np, wealth_exact_np.mean(axis=0), 'b-', linewidth=2, label='Exact pi* (mean)')
    ax2.plot(time_np, wealth_learned_np.mean(axis=0), 'g--', linewidth=2, label='Learned pi* (mean)')
    ax2.plot(time_np, wealth_rf_np.mean(axis=0), 'r-.', linewidth=2, label='Risk-Free (mean)')
    ax2.fill_between(time_np, 
                      wealth_exact_np.mean(axis=0) - wealth_exact_np.std(axis=0),
                      wealth_exact_np.mean(axis=0) + wealth_exact_np.std(axis=0),
                      alpha=0.1, color='blue')
    ax2.set_title("Mean Wealth Paths (±1 Std for Exact)")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Mean Wealth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Terminal wealth distributions
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(wealth_exact[:, -1].cpu().detach().numpy(), bins=40, alpha=0.5, color='blue', label='Exact pi*', density=True)
    ax3.hist(wealth_learned[:, -1].cpu().detach().numpy(), bins=40, alpha=0.5, color='green', label='Learned pi*', density=True)
    ax3.hist(wealth_rf[:, -1].cpu().detach().numpy(), bins=40, alpha=0.5, color='red', label='Risk-Free', density=True)
    ax3.set_title("Terminal Wealth Distribution Comparison")
    ax3.set_xlabel("Terminal Wealth")
    ax3.set_ylabel("Density")
    ax3.legend()
    
    # Terminal wealth statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = (
        f"Terminal Wealth (Mean ± Std):\n\n"
        f"Exact pi*:   {wealth_exact[:, -1].mean():.4f} ± {wealth_exact[:, -1].std():.4f}\n"
        f"Learned pi*: {wealth_learned[:, -1].mean():.4f} ± {wealth_learned[:, -1].std():.4f}\n"
        f"Risk-Free:   {wealth_rf[:, -1].mean():.4f} ± {wealth_rf[:, -1].std():.4f}\n\n"
        f"pi* values:\n"
        f"Exact:   {wealth_exact[:, -1].mean():.4f}\n"
        f"Learned: {wealth_learned[:, -1].mean():.4f}\n"
        f"RF:      {wealth_rf[:, -1].mean():.4f}"
    )
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_terminal_wealth_vs_pi_sweep(mu, sigma, w0, rate, T, num_steps, num_paths, learned_pi, exact_pi):
    """Graph #10: Sweep pi ∈ [0, 1], plot mean terminal wealth; mark learned and exact pi*."""
    pi_values = np.linspace(0.0, 1.0, 21)  # 21 points from 0 to 1
    mean_terminal = []
    std_terminal = []
    
    dt = T / num_steps
    
    for pi_val in pi_values:
        wealth = torch.full((num_paths, num_steps + 1), w0, dtype=torch.float32)
        for i in range(num_steps):
            dZ = torch.randn(num_paths) * np.sqrt(dt)
            drift = rate + pi_val * (mu - rate)
            diffusion = pi_val * sigma
            wealth[:, i+1] = wealth[:, i] * (1 + drift * dt + diffusion * dZ)
        mean_terminal.append(wealth[:, -1].mean().item())
        std_terminal.append(wealth[:, -1].std().item())
    
    mean_terminal = np.array(mean_terminal)
    std_terminal = np.array(std_terminal)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Mean terminal wealth curve
    color = 'tab:blue'
    ax1.plot(pi_values, mean_terminal, 'b-', linewidth=2, label='Mean Terminal Wealth')
    ax1.fill_between(pi_values, mean_terminal - std_terminal, mean_terminal + std_terminal,
                     alpha=0.15, color='blue', label='±1 Std')
    ax1.set_xlabel("pi* (Proportion in Risky Asset)")
    ax1.set_ylabel("Mean Terminal Wealth", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Mark the three key policies
    # Compute mean terminal for exact and learned
    wealth_exact_sweep = torch.full((num_paths, num_steps + 1), w0, dtype=torch.float32)
    wealth_learned_sweep = torch.full((num_paths, num_steps + 1), w0, dtype=torch.float32)
    wealth_rf_sweep = torch.full((num_paths, num_steps + 1), w0, dtype=torch.float32)
    for i in range(num_steps):
        dZ = torch.randn(num_paths) * np.sqrt(dt)
        for we, pi in [(wealth_exact_sweep, exact_pi), (wealth_learned_sweep, learned_pi), (wealth_rf_sweep, 0.0)]:
            drift = rate + pi * (mu - rate)
            diffusion = pi * sigma
            we[:, i+1] = we[:, i] * (1 + drift * dt + diffusion * dZ)
    
    ax1.scatter([exact_pi], [wealth_exact_sweep[:, -1].mean().item()], 
                color='red', s=100, marker='*', zorder=5, label=f'Exact pi* ({exact_pi:.4f})')
    ax1.scatter([learned_pi], [wealth_learned_sweep[:, -1].mean().item()], 
                color='green', s=100, marker='*', zorder=5, label=f'Learned pi* ({learned_pi:.4f})')
    ax1.scatter([0.0], [wealth_rf_sweep[:, -1].mean().item()], 
                color='orange', s=100, marker='*', zorder=5, label='Risk-Free (pi=0)')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Terminal Wealth vs pi*: Policy Sweep")
    
    plt.tight_layout()
    plt.show()