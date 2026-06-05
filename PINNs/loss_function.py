from __future__ import annotations

import json
from pathlib import Path
from typing import Union
from terminal_condition import crra

import torch


_CONFIG_PATH = Path(__file__).resolve().parent / "data_accumulation" / "config.json"

# Load defaults if available; loss_function still accepts explicit parameters.
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = json.load(f)
    _DEFAULT_RATE = _config["rate"]
    _DEFAULT_SIGMA = _config["sigma"]
    _DEFAULT_MU = _config["mu"]
else:
    _DEFAULT_RATE = None
    _DEFAULT_SIGMA = None
    _DEFAULT_MU = None


def _as_tensor_like(value: Union[float, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


def loss_function(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    x_t: torch.Tensor,
    t_t: torch.Tensor,
    gamma,
    rate,
    mu,
    sigma,
) -> torch.Tensor:
    """Compute PINN residual loss for the HJB PDE with terminal condition.

    The model signature is model(t, w) where t is time and w is wealth.

    Args:
        model: neural network approximating V(t, w)
        x: wealth collocation points (N, 1)
        t: time collocation points (N, 1)
        x_t: wealth points for terminal condition (N, 1)
        t_t: time points for terminal condition (N, 1) — should equal T
        gamma: CRRA risk aversion parameter
        rate: risk-free interest rate r
        mu: expected return of risky asset
        sigma: volatility of risky asset

    Returns:
        Total loss = PDE_residual_loss + terminal_loss
    """
    if rate is None:
        rate = _DEFAULT_RATE
    if mu is None:
        mu = _DEFAULT_MU
    if sigma is None:
        sigma = _DEFAULT_SIGMA

    rate_t = _as_tensor_like(rate, x)
    mu_t = _as_tensor_like(mu, x)
    sigma_t = _as_tensor_like(sigma, x)

    # Ensure gradients are tracked for PDE computation.
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)

    # Model forward: V(t, w)
    V = model(t, x)

    v_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    v_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    # HJB PDE for Merton's problem:
    #   V_t + r * w * V_w + 0.5 * ((μ - r) / σ)^2 * w^2 * V_ww = 0
    growth_term = rate_t * x * v_x
    volatility_term = 0.5 * ((mu_t - rate_t) ** 2) / (sigma_t ** 2) * (x ** 2) * v_xx

    residual = v_t + growth_term + volatility_term
    pde_loss = torch.mean(residual ** 2)

    # Terminal condition: V(T, w) = U(w) where U is the CRRA utility function
    V_t_pred = model(t_t, x_t)
    v_t_absolute = crra(x_t, gamma)
    terminal_loss = torch.mean((V_t_pred - v_t_absolute) ** 2)

    return pde_loss + terminal_loss