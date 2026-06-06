import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def plot_training_history(history):
    if not history or "epoch" not in history or not history["epoch"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history["epoch"]
    train_loss = history["train_loss"]

    axes[0].plot(epochs, train_loss, color='blue', lw=2, label='Training Loss')
    if history.get("val_loss"):
        val_epochs = [epochs[i] for i in range(len(epochs)) if i % 50 == 0]
        val_loss = history["val_loss"]
        axes[0].plot(val_epochs, val_loss, color='red', lw=2, label='Validation Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, train_loss, color='blue', lw=2, label='Training Loss')
    if history.get("val_loss"):
        axes[1].semilogy(val_epochs, val_loss, color='red', lw=2, label='Validation Loss', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (Log Scale)')
    axes[1].set_title('Training Loss (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_policy_validation(model, exact_pi, T=1.0, W_MIN=0.1, W_MAX=2.0, device='cpu'):
    t_np = np.linspace(0.01, T, 50)
    w_np = np.linspace(W_MIN, W_MAX, 50)
    t_mesh, w_mesh = np.meshgrid(t_np, w_np)

    t_col = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    w_col = torch.tensor(w_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        pi_pred = model(t_col, w_col).cpu().numpy().reshape(50, 50)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(t_mesh, w_mesh, pi_pred, cmap='viridis', alpha=0.8)
    exact_plane = np.full_like(pi_pred, exact_pi)
    ax1.plot_surface(t_mesh, w_mesh, exact_plane, color='red', alpha=0.3)
    ax1.set_title("Learned Policy vs Exact Truth")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Wealth (w)")
    ax1.set_zlabel("pi*")

    ax2 = fig.add_subplot(1, 3, 2)
    error = np.abs(pi_pred - exact_pi)
    c = ax2.contourf(t_mesh, w_mesh, error, 50, cmap='magma')
    fig.colorbar(c, ax=ax2)
    ax2.set_title("Absolute Error Heatmap")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Wealth (w)")

    ax3 = fig.add_subplot(1, 3, 3)
    t_indices = [5, 25, 45]
    for idx in t_indices:
        ax3.plot(w_np, pi_pred[:, idx], label=f"t = {t_np[idx]:.2f}")
    ax3.axhline(exact_pi, color='red', linestyle='--', label="Exact")
    ax3.set_title("Cross-Sectional Slices of pi*")
    ax3.set_xlabel("Wealth (w)")
    ax3.set_ylabel("pi*")
    ax3.legend()
    if exact_pi > 0:
        ax3.set_ylim(exact_pi * 0.5, exact_pi * 1.5)

    plt.tight_layout()
    plt.show()


def plot_pde_residual_distribution(model, t_colloc, w_colloc, rate, mu, sigma, device='cpu'):
    from loss_function import compute_pde_residual

    t_tensor = t_colloc.to(device) if isinstance(t_colloc, torch.Tensor) else torch.tensor(t_colloc, dtype=torch.float32, device=device).reshape(-1, 1)
    w_tensor = w_colloc.to(device) if isinstance(w_colloc, torch.Tensor) else torch.tensor(w_colloc, dtype=torch.float32, device=device).reshape(-1, 1)

    residual = compute_pde_residual(model, t_tensor, w_tensor, rate, mu, sigma).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residual.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(residual), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residual):.6e}')
    axes[0].set_xlabel('PDE Residual')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PDE Residual Distribution')
    axes[0].legend()

    axes[1].scatter(t_tensor.cpu().numpy().flatten(), residual.flatten(), c=w_tensor.cpu().numpy().flatten(), cmap='viridis', alpha=0.6, s=1)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('PDE Residual')
    axes[1].set_title('PDE Residual vs Time')
    plt.colorbar(axes[1].collections[0], label='Wealth')

    plt.tight_layout()
    plt.show()


def plot_terminal_condition_comparison(model, t_terminal, w_terminal, gamma, device='cpu'):
    from terminal_condition import crra

    t_tensor = t_terminal.to(device) if isinstance(t_terminal, torch.Tensor) else torch.tensor(t_terminal, dtype=torch.float32, device=device)
    w_tensor = w_terminal.to(device) if isinstance(w_terminal, torch.Tensor) else torch.tensor(w_terminal, dtype=torch.float32, device=device)

    with torch.no_grad():
        v_pred = model(t_tensor, w_tensor).cpu().numpy().flatten()
    v_exact = crra(w_tensor, gamma).cpu().numpy().flatten()
    w_np = w_tensor.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(w_np, v_exact, color='blue', alpha=0.6, label='Exact CRRA Utility', s=10)
    axes[0].scatter(w_np, v_pred, color='red', alpha=0.6, label='Model Prediction', s=10)
    axes[0].set_xlabel('Wealth (w)')
    axes[0].set_ylabel('Utility V(T, w)')
    axes[0].set_title('Terminal Condition: Model vs Exact')
    axes[0].legend()

    error = np.abs(v_pred - v_exact)
    axes[1].scatter(w_np, error, color='green', alpha=0.6, s=10)
    axes[1].set_xlabel('Wealth (w)')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Terminal Condition Error')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_analytical_comparison(model, t_colloc, w_colloc, T, gamma, rate, mu, sigma, device='cpu'):
    from terminal_condition import merton_analytical_solution

    t_tensor = t_colloc.to(device) if isinstance(t_colloc, torch.Tensor) else torch.tensor(t_colloc, dtype=torch.float32, device=device).reshape(-1, 1)
    w_tensor = w_colloc.to(device) if isinstance(w_colloc, torch.Tensor) else torch.tensor(w_colloc, dtype=torch.float32, device=device).reshape(-1, 1)

    with torch.no_grad():
        v_pred = model(t_tensor, w_tensor).cpu().numpy().flatten()
    v_analytical = merton_analytical_solution(t_tensor, w_tensor, T, gamma, rate, mu, sigma).cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(v_analytical, v_pred, alpha=0.6, s=10)
    min_val = min(v_analytical.min(), v_pred.min())
    max_val = max(v_analytical.max(), v_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
    axes[0].set_xlabel('Analytical Solution')
    axes[0].set_ylabel('Model Prediction')
    axes[0].set_title('Model vs Analytical Solution')
    axes[0].legend()

    abs_error = np.abs(v_pred - v_analytical)
    rel_error = abs_error / (np.abs(v_analytical) + 1e-8)

    axes[1].hist(abs_error, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Absolute Error Distribution (Mean: {abs_error.mean():.6e})')

    axes[2].hist(rel_error, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Relative Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Relative Error Distribution (Mean: {rel_error.mean():.6e})')

    plt.tight_layout()
    plt.show()


def plot_optimal_portfolio_analysis(model, t_colloc, w_colloc, rate, mu, sigma, gamma, device='cpu'):
    from terminal_condition import optimal_portfolio_weight

    t_tensor = t_colloc.to(device) if isinstance(t_colloc, torch.Tensor) else torch.tensor(t_colloc, dtype=torch.float32, device=device).reshape(-1, 1)
    w_tensor = w_colloc.to(device) if isinstance(w_colloc, torch.Tensor) else torch.tensor(w_colloc, dtype=torch.float32, device=device).reshape(-1, 1)

    t_req = t_tensor.clone().detach().requires_grad_(True)
    w_req = w_tensor.clone().detach().requires_grad_(True)

    V = model(t_req, w_req)

    V_w = torch.autograd.grad(V, w_req, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w_req, grad_outputs=torch.ones_like(V_w), create_graph=True, retain_graph=True)[0]

    epsilon = 1e-8
    pi_model = -(mu - rate) * V_w / (sigma ** 2 * w_req * (V_ww - epsilon))
    pi_model_np = pi_model.cpu().detach().numpy().flatten()

    exact_pi = optimal_portfolio_weight(gamma, rate, mu, sigma)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(t_tensor.cpu().numpy().flatten(), pi_model_np, c=w_tensor.cpu().numpy().flatten(), cmap='viridis', alpha=0.6, s=10)
    axes[0].axhline(exact_pi, color='red', linestyle='--', linewidth=2, label=f'Analytical (π*={exact_pi:.4f})')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Optimal Portfolio Weight (π*)')
    axes[0].set_title('Optimal Policy: Model Implied')
    axes[0].legend()
    plt.colorbar(axes[0].collections[0], label='Wealth')

    axes[1].hist(pi_model_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(exact_pi, color='red', linestyle='--', linewidth=2, label=f'Analytical (π*={exact_pi:.4f})')
    axes[1].axvline(np.mean(pi_model_np), color='green', linestyle=':', linewidth=2, label=f'Model Mean ({np.mean(pi_model_np):.4f})')
    axes[1].set_xlabel('Optimal Portfolio Weight (π*)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Model-Implied π*')
    axes[1].legend()

    error_pi = np.abs(pi_model_np - exact_pi)
    axes[2].scatter(w_tensor.cpu().numpy().flatten(), error_pi, alpha=0.6, s=10)
    axes[2].set_xlabel('Wealth (w)')
    axes[2].set_ylabel('|π*_model - π*_analytical|')
    axes[2].set_title(f'Portfolio Weight Error (Mean: {error_pi.mean():.6f})')

    plt.tight_layout()
    plt.show()


def plot_financial_sensitivity(mu, r, base_gamma, base_sigma):
    gammas = np.linspace(1.1, 5.0, 100)
    sigmas = np.linspace(0.05, 0.5, 100)

    pi_gammas = (mu - r) / (gammas * base_sigma**2)
    pi_sigmas = (mu - r) / (base_gamma * sigmas**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gammas, pi_gammas, color='purple', lw=2)
    ax1.set_title("Optimal Policy vs Risk Aversion (γ)")
    ax1.set_xlabel("Gamma (Risk Aversion)")
    ax1.set_ylabel("pi*")

    ax2.plot(sigmas, pi_sigmas, color='orange', lw=2)
    ax2.set_title("Optimal Policy vs Market Volatility (σ)")
    ax2.set_xlabel("Sigma (Volatility)")
    ax2.set_ylabel("pi*")

    plt.tight_layout()
    plt.show()