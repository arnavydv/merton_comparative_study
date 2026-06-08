import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress minor matplotlib warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Phase 1: Market Simulation & Forward Dynamics
# =============================================================================
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
        ax1.plot(time_np[1:], W_np[i], lw=0.5, alpha=0.6)
    ax1.set_title("1. Brownian Motion Paths (W_t)")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("W(t)")

    # 2 & 3. Controlled Wealth Dynamics
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(wealth_np.shape[0]):
        ax2.plot(time_np, wealth_np[i], lw=0.5, alpha=0.6)
    ax2.set_title("2 & 3. Controlled Wealth Dynamics (W_t^{\pi^*})")
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

# =============================================================================
# Phase 2: Neural Network Training Diagnostics
# =============================================================================
def plot_training_diagnostics(loss_history):
    """Plots the convergence metrics of the neural network."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 5. Variance Loss vs Epochs (Log Scale)
    epochs = np.arange(1, len(loss_history) + 1)
    ax.semilogy(epochs, loss_history, color='red', lw=2)
    ax.set_title("5. Network Convergence (Variance Loss)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (Log Scale)")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    # Note: Tracking Y0 and Gradients requires modifying the training loop 
    # to append those specific values to lists, similar to loss_history.

# =============================================================================
# Phase 3: The Optimal Policy (pi*) Validation
# =============================================================================
def plot_policy_validation(model, exact_pi, T=1.0, device='cpu'):
    """Generates 3D surfaces and heatmaps comparing learned vs exact policy."""
    t_np = np.linspace(0.01, T, 50)
    W_np = np.linspace(-3.0, 3.0, 50)
    t_mesh, W_mesh = np.meshgrid(t_np, W_np)
    
    t_col = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    W_col = torch.tensor(W_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    with torch.no_grad():
        pi_pred = model(t_col, W_col).cpu().numpy().reshape(50, 50)
    
    fig = plt.figure(figsize=(18, 6))

    # 9 & 10. 3D Surface vs Analytical Truth
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(t_mesh, W_mesh, pi_pred, cmap='viridis', alpha=0.8)
    exact_plane = np.full_like(pi_pred, exact_pi)
    ax1.plot_surface(t_mesh, W_mesh, exact_plane, color='red', alpha=0.3)
    ax1.set_title("9 & 10. Learned Policy vs Exact Truth")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("W(t)")
    ax1.set_zlabel("pi*")

    # 11. Policy Error Heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    error = np.abs(pi_pred - exact_pi)
    c = ax2.contourf(t_mesh, W_mesh, error, 50, cmap='magma')
    fig.colorbar(c, ax=ax2)
    ax2.set_title("11. Absolute Error Heatmap")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("W(t)")

    # 12. Cross-Sectional Policy Slices
    ax3 = fig.add_subplot(1, 3, 3)
    t_indices = [5, 25, 45] # Approx early, mid, late times
    for idx in t_indices:
        ax3.plot(W_np, pi_pred[:, idx], label=f"t = {t_np[idx]:.2f}")
    ax3.axhline(exact_pi, color='red', linestyle='--', label="Exact")
    ax3.set_title("12. Cross-Sectional Slices of pi*")
    ax3.set_xlabel("W(t)")
    ax3.set_ylabel("pi*")
    ax3.legend()
    ax3.set_ylim(exact_pi * 0.5, exact_pi * 1.5)

    plt.tight_layout()
    plt.show()

# =============================================================================
# Phase 4 & 5: Value Function, Z-Network & Sensitivity
# =============================================================================
def plot_backward_functions(Z_model, T=1.0, device='cpu'):
    """Visualizes the learned Z-network (Control/Hedging parameter)."""
    t_np = np.linspace(0.01, T, 40)
    W_np = np.linspace(-3.0, 3.0, 40)
    t_mesh, W_mesh = np.meshgrid(t_np, W_np)
    
    t_col = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    W_col = torch.tensor(W_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    with torch.no_grad():
        Z_pred = Z_model(t_col, W_col).cpu().numpy().reshape(40, 40)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t_mesh, W_mesh, Z_pred, cmap='plasma')
    ax.set_title("14. The Z-Network Surface")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("W(t)")
    ax.set_zlabel("Z value")
    plt.show()

def plot_financial_sensitivity(mu, r, base_gamma, base_sigma):
    """Plots how the analytical exact policy changes under different regimes."""
    gammas = np.linspace(1.1, 5.0, 100)
    sigmas = np.linspace(0.05, 0.5, 100)
    
    pi_gammas = (mu - r) / (gammas * base_sigma**2)
    pi_sigmas = (mu - r) / (base_gamma * sigmas**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 16. Sensitivity to Risk Aversion
    ax1.plot(gammas, pi_gammas, color='purple', lw=2)
    ax1.set_title("16. Optimal Policy vs Risk Aversion (\gamma)")
    ax1.set_xlabel("Gamma (Risk Aversion)")
    ax1.set_ylabel("pi*")
    
    # 17. Sensitivity to Volatility
    ax2.plot(sigmas, pi_sigmas, color='orange', lw=2)
    ax2.set_title("17. Optimal Policy vs Market Volatility (\sigma)")
    ax2.set_xlabel("Sigma (Volatility)")
    ax2.set_ylabel("pi*")
    
    plt.tight_layout()
    plt.show()