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


def loss_function(model, x: torch.Tensor, t: torch.Tensor,x_t:torch.Tensor,t_t:torch.Tensor,gamma,rate, mu, sigma) -> torch.Tensor:
    """Compute PINN residual loss for the HJB PDE.

    Expected model signature (see PINNs/neural_network.py): model(w, x)
    where w corresponds to time t and x corresponds to the state variable.

    Shapes: x,t are (N,1) or (N,). Autograd requires requires_grad=True.
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

    # Ensure gradients are tracked.
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)

    # Model forward: model(w, x)
    V = model(t, x)

    v_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    v_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    growth_term = rate_t * x * v_x

    # Corrected volatility term: previous code had syntax/parentheses issues.
    # volatility term: 0.5 * ((mu - rate)^2) / sigma^2 * x^2 * v_xx
    volatility_term = 0.5 * ((mu_t - rate_t) ** 2) / (sigma_t ** 2) * (x ** 2) * v_xx

    residual = v_t + growth_term + volatility_term
    pde_loss=torch.mean(residual**2)
    
    V_t_pred=model(x_t,t_t)
    v_t_absolute=crra(x_t,gamma)
    terminal_loss=V_t_pred-v_t_absolute
    


    return pde_loss+10*terminal_loss

    