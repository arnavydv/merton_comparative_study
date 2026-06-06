from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from loss_function import compute_phi_pde_residual
from neural_network import ValueFunc
from terminal_condition import crra, merton_analytical_solution, optimal_portfolio_weight

# Resolve config path relative to this file
_CONFIG_PATH = Path(__file__).resolve().parent / "data_accumulation" / "config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)


def pde_residual_from_value(model: nn.Module, t: torch.Tensor, w: torch.Tensor,
                            rate: float, mu: float, sigma: float) -> torch.Tensor:
    """Compute the PDE residual by directly evaluating the HJB PDE on V(t,w).

    This uses the φ-factorization: V(t,w) = φ(t) * w^(1-γ)/(1-γ).
    The PDE reduces to: φ'(t) + κ*φ(t) = 0.

    Args:
        model: ValueFunc network.
        t: Time points (N, 1).
        w: Wealth points (N, 1).
        rate: Risk-free rate.
        mu: Expected return.
        sigma: Volatility.

    Returns:
        PDE residual tensor (N, 1).
    """
    return compute_phi_pde_residual(model.phi_net, t, rate, mu, sigma, model.gamma)


def evaluate_pde_residual(
    model: nn.Module,
    t: torch.Tensor,
    w: torch.Tensor,
    rate: float,
    mu: float,
    sigma: float,
) -> Tuple[torch.Tensor, float]:
    """Evaluate the PDE residual of the trained model.

    Args:
        model: Trained neural network.
        t: Time points (N, 1).
        w: Wealth points (N, 1).
        rate: Risk-free rate.
        mu: Expected return.
        sigma: Volatility.

    Returns:
        Tuple of (residual_tensor, mean_squared_residual).
    """
    residual = pde_residual_from_value(model, t, w, rate, mu, sigma)
    mse_residual = torch.mean(residual ** 2).item()
    return residual, mse_residual


def evaluate_terminal_condition(
    model: nn.Module,
    t_terminal: torch.Tensor,
    w_terminal: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Evaluate how well the model satisfies the terminal condition.

    Args:
        model: Trained neural network.
        t_terminal: Terminal time points (M, 1).
        w_terminal: Terminal wealth points (M, 1).
        gamma: CRRA risk aversion parameter.

    Returns:
        Tuple of (predicted_values, exact_values, mse_error).
    """
    predicted = model(t_terminal, w_terminal)
    exact = crra(w_terminal, gamma)
    mse = torch.mean((predicted - exact) ** 2).item()
    return predicted, exact, mse


def evaluate_against_analytical(
    model: nn.Module,
    t: torch.Tensor,
    w: torch.Tensor,
    T: float,
    gamma: float,
    rate: float,
    mu: float,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Compare model predictions against the analytical solution.

    Args:
        model: Trained neural network.
        t: Time points (N, 1).
        w: Wealth points (N, 1).
        T: Terminal time.
        gamma: CRRA risk aversion parameter.
        rate: Risk-free rate.
        mu: Expected return.
        sigma: Volatility.

    Returns:
        Tuple of (predicted_values, analytical_values, error_metrics).
    """
    predicted = model(t, w)
    analytical = merton_analytical_solution(t, w, T, gamma, rate, mu, sigma)

    # Compute error metrics
    abs_error = torch.abs(predicted - analytical)
    rel_error = abs_error / (torch.abs(analytical) + 1e-8)

    metrics = {
        "mean_absolute_error": torch.mean(abs_error).item(),
        "max_absolute_error": torch.max(abs_error).item(),
        "mean_relative_error": torch.mean(rel_error).item(),
        "max_relative_error": torch.max(rel_error).item(),
        "mse": torch.mean((predicted - analytical) ** 2).item(),
        "rmse": torch.sqrt(torch.mean((predicted - analytical) ** 2)).item(),
    }

    return predicted, analytical, metrics


def compute_optimal_portfolio_from_model(
    model: nn.Module,
    t: torch.Tensor,
    w: torch.Tensor,
    rate: float,
    mu: float,
    sigma: float,
) -> torch.Tensor:
    """Compute the optimal portfolio weight implied by the trained model.

    The optimal portfolio weight is:
        pi*(t, w) = -(mu - r) * V_w / (sigma^2 * w * V_ww)

    For CRRA utility, this should be constant: pi* = (mu - r) / (sigma^2 * gamma)

    Args:
        model: Trained neural network.
        t: Time points (N, 1).
        w: Wealth points (N, 1).
        rate: Risk-free rate.
        mu: Expected return.
        sigma: Volatility.

    Returns:
        Optimal portfolio weight at each point.
    """
    t_req = t.clone().detach().requires_grad_(True)
    w_req = w.clone().detach().requires_grad_(True)

    V = model(t_req, w_req)

    # Compute V_w
    V_w = torch.autograd.grad(
        V, w_req,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute V_ww
    V_ww = torch.autograd.grad(
        V_w, w_req,
        grad_outputs=torch.ones_like(V_w),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute optimal portfolio weight
    epsilon = 1e-8
    pi_star = -(mu - rate) * V_w / (sigma ** 2 * w_req * (V_ww - epsilon))

    return pi_star


def full_evaluation(
    model: nn.Module,
    t_colloc: torch.Tensor,
    w_colloc: torch.Tensor,
    t_terminal: torch.Tensor,
    w_terminal: torch.Tensor,
    T: float,
    gamma: float,
    rate: float,
    mu: float,
    sigma: float,
) -> dict:
    """Perform a comprehensive evaluation of the trained model.

    Args:
        model: Trained neural network.
        t_colloc: Collocation time points.
        w_colloc: Collocation wealth points.
        t_terminal: Terminal time points.
        w_terminal: Terminal wealth points.
        T: Terminal time.
        gamma: CRRA risk aversion parameter.
        rate: Risk-free rate.
        mu: Expected return.
        sigma: Volatility.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    model.eval()

    results = {}

    # 1. PDE residual evaluation
    _, pde_mse = evaluate_pde_residual(model, t_colloc, w_colloc, rate, mu, sigma)
    results["pde_mse"] = pde_mse

    # 2. Terminal condition evaluation
    _, _, terminal_mse = evaluate_terminal_condition(model, t_terminal, w_terminal, gamma)
    results["terminal_mse"] = terminal_mse

    # 3. Analytical solution comparison
    _, _, analytical_metrics = evaluate_against_analytical(
        model, t_colloc, w_colloc, T, gamma, rate, mu, sigma
    )
    results["analytical"] = analytical_metrics

    # 4. Optimal portfolio weight
    pi_model = compute_optimal_portfolio_from_model(model, t_colloc, w_colloc, rate, mu, sigma)
    pi_analytical = optimal_portfolio_weight(gamma, rate, mu, sigma)
    results["optimal_portfolio"] = {
        "model_mean": torch.mean(pi_model).item(),
        "model_std": torch.std(pi_model).item(),
        "analytical": pi_analytical,
        "error": abs(torch.mean(pi_model).item() - pi_analytical),
    }

    return results


def print_evaluation_report(results: dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print("\n--- PDE Residual ---")
    print(f"  PDE MSE: {results['pde_mse']:.6e}")

    print("\n--- Terminal Condition ---")
    print(f"  Terminal MSE: {results['terminal_mse']:.6e}")

    print("\n--- Analytical Solution Comparison ---")
    for key, value in results["analytical"].items():
        print(f"  {key}: {value:.6e}")

    print("\n--- Optimal Portfolio Weight ---")
    pi_info = results["optimal_portfolio"]
    print(f"  Model mean: {pi_info['model_mean']:.6f}")
    print(f"  Model std:  {pi_info['model_std']:.6f}")
    print(f"  Analytical: {pi_info['analytical']:.6f}")
    print(f"  Error:      {pi_info['error']:.6e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage: evaluate a trained model
    from collocation_points import collocation_points
    from terminal_condition import terminal_points

    # Parameters from config
    T = 1.0
    W_MIN = 0.1
    W_MAX = 2.0
    sigma = config["sigma"]
    rate = config["rate"]
    mu = config["mu"]
    gamma = config["gamma"]

    # Load trained model
    model_path = Path(__file__).resolve().parent / "saved_models" / "best_model.pt"
    if model_path.exists():
        model = ValueFunc(gamma=gamma)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded model from epoch {checkpoint['epoch']}")

        # Generate evaluation points
        t_colloc, w_colloc = collocation_points(5000, T, W_MAX, W_MIN)
        t_colloc = t_colloc.reshape(-1, 1)
        w_colloc = w_colloc.reshape(-1, 1)

        t_term, w_term, _ = terminal_points(1000, T, W_MAX, W_MIN, gamma)

        # Full evaluation
        results = full_evaluation(
            model, t_colloc, w_colloc, t_term, w_term,
            T, gamma, rate, mu, sigma
        )

        print_evaluation_report(results)
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first using training.py")