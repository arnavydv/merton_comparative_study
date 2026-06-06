from __future__ import annotations

import torch
from scipy.stats import qmc


def terminal_points(num_points: int,T: float,w_max: float,w_min: float,gamma: float,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if w_min <= 0:
        raise ValueError("w_min must be positive for CRRA utility (log(0) is undefined)")
    if num_points <= 0:
        raise ValueError("num_points must be a positive integer")
    # Generate Latin Hypercube samples for wealth
    sampler = qmc.LatinHypercube(d=1)
    samples = sampler.random(n=num_points)
    scaled_samples = qmc.scale(samples, w_min, w_max)
    # Create tensors
    w_terminal = torch.tensor(scaled_samples, dtype=torch.float32)
    t_terminal = torch.full((num_points, 1), T, dtype=torch.float32)
    # Compute CRRA utility: U(w) = w^(1-gamma) / (1-gamma)
    if gamma == 1.0:
        v_terminal = torch.log(w_terminal)
    else:
        term = 1.0 - gamma
        v_terminal = (w_terminal ** term) / term
    return t_terminal, w_terminal, v_terminal


def crra(wealth: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 1.0:
        return torch.log(wealth)
    term = 1.0 - gamma
    return (wealth ** term) / term

def merton_analytical_solution( t: torch.Tensor,w: torch.Tensor,T: float,gamma: float,rate: float,mu: float,sigma: float,) -> torch.Tensor:
    """Compute the analytical solution to Merton's problem with CRRA utility.
    The value function is:
        V(t, w) = (w^(1-gamma) / (1-gamma)) * exp(A*(T-t))
    where:
        A = (1-gamma) * [r + ((mu-r)^2) / (2*sigma^2*gamma)]
    Returns:
        Analytical value function V(t, w).
    """
    if gamma == 1.0:
        # Log utility case
        A = rate + ((mu - rate) ** 2) / (2 * sigma ** 2)
        return torch.log(w) + A * (T - t)
    else:
        # Power utility case
        A = (1 - gamma) * (rate + ((mu - rate) ** 2) / (2 * sigma ** 2 * gamma))
        base = (w ** (1 - gamma)) / (1 - gamma)
        return base * torch.exp(A * (T - t))


def optimal_portfolio_weight(gamma: float,rate: float,mu: float,sigma: float,) -> float:
    return (mu - rate) / (sigma ** 2 * gamma)