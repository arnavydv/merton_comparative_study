import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress minor matplotlib warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Graph 1: Value Function Surface (3D)
# =============================================================================
def plot_value_function_surface(t_mesh, w_mesh, V_fdm):
    """Visualizes the FDM Value Function surface."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(t_mesh, w_mesh, V_fdm, cmap='plasma', edgecolor='none')
    ax.set_title("FDM Value Function V(t, w)")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    ax.set_zlabel("V(t, w)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()

# =============================================================================
# Graph 2: Value Function Error Heatmap
# =============================================================================
def plot_value_function_error_heatmap(t_mesh, w_mesh, V_fdm, V_exact):
    """Heatmap showing where the numerical error is concentrated."""
    fig, ax = plt.subplots(figsize=(10, 6))
    error = np.abs(V_fdm - V_exact)
    c = ax.contourf(t_mesh, w_mesh, error, levels=50, cmap='magma')
    fig.colorbar(c, ax=ax, label="Absolute Error")
    ax.set_title("Value Function Absolute Error |V_FDM - V_Exact|")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Graph 3: Optimal Policy Surface (3D)
# =============================================================================
def plot_policy_surface(t_mesh, w_mesh, pi_fdm, pi_exact_val):
    """3D surface of the FDM policy with the exact flat plane overlaid."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t_mesh, w_mesh, pi_fdm, cmap='viridis', alpha=0.8, label='FDM Policy')
    exact_plane = np.full_like(pi_fdm, pi_exact_val)
    ax.plot_surface(t_mesh, w_mesh, exact_plane, color='red', alpha=0.4, label='Exact Policy')
    ax.set_title("FDM Optimal Policy vs Exact Truth")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    ax.set_zlabel("pi*")
    plt.show()

# =============================================================================
# Graph 4: Optimal Policy Error Heatmap
# =============================================================================
def plot_policy_error_heatmap(t_mesh, w_mesh, pi_fdm, pi_exact_val):
    """Heatmap of the absolute error in the portfolio weight."""
    fig, ax = plt.subplots(figsize=(10, 6))
    error = np.abs(pi_fdm - pi_exact_val)
    c = ax.contourf(t_mesh, w_mesh, error, levels=50, cmap='magma')
    fig.colorbar(c, ax=ax, label="Absolute Error")
    ax.set_title("Optimal Policy Absolute Error |pi*_FDM - pi*_Exact|")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Graph 5: Cross-Sectional Value Function Slices
# =============================================================================
def plot_value_function_slices(t_grid, w_grid, V_fdm, V_exact):
    """2D line plots comparing FDM vs Exact at early, mid, and late times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = [0, len(t_grid)//2, -1]
    colors = ['blue', 'green', 'purple']
    labels_t = [f"t = {t_grid[i]:.2f}" for i in indices]
    
    for idx, color, label in zip(indices, colors, labels_t):
        ax.plot(w_grid, V_fdm[idx, :], color=color, linestyle='-', linewidth=2, label=f'FDM {label}')
        ax.plot(w_grid, V_exact[idx, :], color=color, linestyle='--', linewidth=2, label=f'Exact {label}')
        
    ax.set_title("Value Function Cross-Sections (FDM vs Exact)")
    ax.set_xlabel("Wealth (w)")
    ax.set_ylabel("V(t, w)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Graph 6: Cross-Sectional Policy Slices
# =============================================================================
def plot_policy_slices(t_grid, w_grid, pi_fdm, pi_exact_val):
    """2D line plots of the policy at different times vs the exact constant."""
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = [0, len(t_grid)//2, -1]
    colors = ['blue', 'green', 'purple']
    
    for idx, color in zip(indices, colors):
        ax.plot(w_grid, pi_fdm[idx, :], color=color, linestyle='-', linewidth=2, label=f'FDM t = {t_grid[idx]:.2f}')
        
    ax.axhline(pi_exact_val, color='red', linestyle='--', linewidth=2, label=f'Exact pi* = {pi_exact_val:.4f}')
    ax.set_title("Optimal Policy Cross-Sections")
    ax.set_xlabel("Wealth (w)")
    ax.set_ylabel("pi*")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Graph 7: Marginal Value of Wealth Surface (3D)
# =============================================================================
def plot_marginal_value_surface(t_mesh, w_mesh, V_w_fdm):
    """3D surface of the first derivative V_w(t,w)."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(t_mesh, w_mesh, V_w_fdm, cmap='coolwarm', edgecolor='none')
    ax.set_title("Marginal Value of Wealth V_w(t, w)")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    ax.set_zlabel("V_w")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()

# =============================================================================
# Graph 8: Implied Relative Risk Aversion Heatmap
# =============================================================================
def plot_implied_risk_aversion_heatmap(t_mesh, w_mesh, gamma_implied, true_gamma):
    """Checks if the FDM solution preserves the CRRA property: -w * V_ww / V_w == gamma."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Clip extreme boundary values for visualization
    gamma_plot = np.clip(gamma_implied, 0, true_gamma * 2)
    c = ax.contourf(t_mesh, w_mesh, gamma_plot, levels=50, cmap='viridis')
    fig.colorbar(c, ax=ax, label="Implied Gamma")
    ax.set_title(f"Implied Relative Risk Aversion (True $\gamma$ = {true_gamma})")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Wealth (w)")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Graph 9: Policy Iteration Convergence Diagnostics
# =============================================================================
def plot_policy_iteration_diagnostics(iterations_per_step):
    """Bar chart showing how many iterations the policy loop took at each time step."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(iterations_per_step)), iterations_per_step, color='teal', alpha=0.7)
    ax.set_title("Policy Iterations Required per Time Step")
    ax.set_xlabel("Time Step Index (n)")
    ax.set_ylabel("Number of Iterations")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Graph 10: Forward Wealth Path Comparison
# =============================================================================
def plot_forward_wealth_paths(time_grid, wealth_fdm, wealth_exact, wealth_rf):
    """Simulates actual wealth paths to compare the financial impact of FDM vs Exact."""
    fig = plt.figure(figsize=(12, 8))
    
    # 1. Single representative path
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time_grid, wealth_exact[0], 'b-', lw=2, label='Exact pi*')
    ax1.plot(time_grid, wealth_fdm[0], 'g--', lw=2, label='FDM pi*')
    ax1.plot(time_grid, wealth_rf[0], 'r-.', lw=2, label='Risk-Free')
    ax1.set_title("Single Wealth Path")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Wealth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean wealth paths
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_grid, wealth_exact.mean(axis=0), 'b-', lw=2, label='Exact (Mean)')
    ax2.plot(time_grid, wealth_fdm.mean(axis=0), 'g--', lw=2, label='FDM (Mean)')
    ax2.plot(time_grid, wealth_rf.mean(axis=0), 'r-.', lw=2, label='RF (Mean)')
    ax2.fill_between(time_grid, 
                     wealth_exact.mean(axis=0) - wealth_exact.std(axis=0),
                     wealth_exact.mean(axis=0) + wealth_exact.std(axis=0),
                     alpha=0.2, color='blue')
    ax2.set_title("Mean Wealth Paths (±1 Std)")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Mean Wealth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Terminal wealth distributions
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(wealth_exact[:, -1], bins=40, alpha=0.6, color='blue', label='Exact', density=True)
    ax3.hist(wealth_fdm[:, -1], bins=40, alpha=0.6, color='green', label='FDM', density=True)
    ax3.hist(wealth_rf[:, -1], bins=40, alpha=0.6, color='red', label='RF', density=True)
    ax3.set_title("Terminal Wealth Distribution")
    ax3.set_xlabel("Terminal Wealth")
    ax3.set_ylabel("Density")
    ax3.legend()
    
    # 4. Statistics text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats = (
        f"Terminal Wealth Stats:\n\n"
        f"Exact pi*: {wealth_exact[:, -1].mean():.4f} ± {wealth_exact[:, -1].std():.4f}\n"
        f"FDM pi*:   {wealth_fdm[:, -1].mean():.4f} ± {wealth_fdm[:, -1].std():.4f}\n"
        f"Risk-Free: {wealth_rf[:, -1].mean():.4f} ± {wealth_rf[:, -1].std():.4f}"
    )
    ax4.text(0.1, 0.5, stats, fontsize=12, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
             
    plt.tight_layout()
    plt.show()