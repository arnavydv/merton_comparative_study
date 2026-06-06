from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch

from terminal_condition import crra


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
    """Convert a scalar or tensor to a tensor with the same device and dtype as `like`."""
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


def loss_function(model,t: torch.Tensor,w: torch.Tensor,t_terminal: torch.Tensor,w_terminal: torch.Tensor,gamma: float,rate: float,mu: float,sigma: float,weight_pde: float = 1.0,weight_terminal: float = 1.0,) -> torch.Tensor:
    if rate is None:
        rate = _DEFAULT_RATE
    if mu is None:
        mu = _DEFAULT_MU
    if sigma is None:
        sigma = _DEFAULT_SIGMA

    rate_t = _as_tensor_like(rate, w)
    mu_t = _as_tensor_like(mu, w)
    sigma_t = _as_tensor_like(sigma, w)

    # Ensure gradients are tracked for PDE computation.
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)
    if not w.requires_grad:
        w = w.clone().detach().requires_grad_(True)


    V = model(t, w)
    V_t = torch.autograd.grad(V, t,grad_outputs=torch.ones_like(V),create_graph=True,retain_graph=True,)[0]
    V_w = torch.autograd.grad(V, w,grad_outputs=torch.ones_like(V),create_graph=True,retain_graph=True,)[0]
    V_ww = torch.autograd.grad(V_w, w,grad_outputs=torch.ones_like(V_w),create_graph=True,retain_graph=True,)[0]
    growth_term = rate_t * w * V_w
    epsilon = 1e-8
    risk_premium_term = -0.5 * ((mu_t - rate_t) ** 2) / (sigma_t ** 2 + epsilon) * (V_w ** 2) / (V_ww - epsilon)
    pde_residual = V_t + growth_term + risk_premium_term
    pde_loss = torch.mean(pde_residual ** 2)

    # Terminal condition: V(T, w) = U(w) = w^(1-gamma) / (1-gamma)
    V_terminal_pred = model(t_terminal, w_terminal)
    V_terminal_exact = crra(w_terminal, gamma)
    terminal_loss = torch.mean((V_terminal_pred - V_terminal_exact) ** 2)

    # Total weighted loss
    total_loss = weight_pde * pde_loss + weight_terminal * terminal_loss
    return total_loss


def compute_pde_residual(model: torch.nn.Module,t: torch.Tensor,w: torch.Tensor,rate: float,mu: float,sigma: float,) -> torch.Tensor:
    rate_t = _as_tensor_like(rate, w)
    mu_t = _as_tensor_like(mu, w)
    sigma_t = _as_tensor_like(sigma, w)
    # Enable gradient computation
    t_req = t.clone().detach().requires_grad_(True)
    w_req = w.clone().detach().requires_grad_(True)
    V = model(t_req, w_req)
    V_t = torch.autograd.grad(V, t_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_w = torch.autograd.grad(V, w_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w_req, grad_outputs=torch.ones_like(V_w), create_graph=True)[0]
    epsilon = 1e-8
    growth_term = rate_t * w_req * V_w
    risk_premium_term = -0.5 * ((mu_t - rate_t) ** 2) / (sigma_t ** 2 + epsilon) * (V_w ** 2) / (V_ww - epsilon)
    residual = V_t + growth_term + risk_premium_term
    return residual