"""Utility functions for CRRA utility and related calculations."""

from __future__ import annotations

import math
from typing import Union

import torch


def CRRA_scalar(wealth: float, gamma: float) -> float:
    """CRRA utility function for scalar values.

    Computes U(w) = w^(1-gamma) / (1-gamma) for gamma != 1,
    and U(w) = log(w) for gamma = 1 (the limiting case).

    Args:
        wealth: Wealth value (must be positive).
        gamma: Risk aversion parameter (> 0).

    Returns:
        Utility value.

    Raises:
        ValueError: If wealth <= 0 or gamma <= 0.
    """
    if wealth <= 0:
        raise ValueError("Wealth must be positive")
    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    if gamma == 1:
        return math.log(wealth)
    term = 1 - gamma
    return (wealth ** term) / term


def CRRA_tensor(wealth: torch.Tensor, gamma: float) -> torch.Tensor:
    """CRRA utility function for tensor values.

    Computes U(w) = w^(1-gamma) / (1-gamma) for gamma != 1,
    and U(w) = log(w) for gamma = 1 (the limiting case).

    Args:
        wealth: Wealth tensor (all values must be positive).
        gamma: Risk aversion parameter (> 0).

    Returns:
        Utility tensor of same shape as input.
    """
    if gamma == 1.0:
        return torch.log(wealth)
    term = 1.0 - gamma
    return (wealth ** term) / term


# Alias for backward compatibility
CRRA = CRRA_scalar


def marginal_utility(wealth: Union[float, torch.Tensor], gamma: float) -> Union[float, torch.Tensor]:
    """Compute marginal utility U'(w) = w^(-gamma).

    Args:
        wealth: Wealth value(s).
        gamma: Risk aversion parameter.

    Returns:
        Marginal utility value(s).
    """
    if isinstance(wealth, torch.Tensor):
        return wealth ** (-gamma)
    return wealth ** (-gamma)


def inverse_marginal_utility(marginal_util: Union[float, torch.Tensor], gamma: float) -> Union[float, torch.Tensor]:
    """Compute inverse of marginal utility: w = (U')^(-1/gamma).

    Args:
        marginal_util: Marginal utility value(s).
        gamma: Risk aversion parameter.

    Returns:
        Corresponding wealth value(s).
    """
    if isinstance(marginal_util, torch.Tensor):
        return marginal_util ** (-1.0 / gamma)
    return marginal_util ** (-1.0 / gamma)


def relative_risk_aversion(wealth: Union[float, torch.Tensor], gamma: float) -> Union[float, torch.Tensor]:
    """Compute relative risk aversion: -w * U''(w) / U'(w) = gamma.

    For CRRA utility, this is constant and equal to gamma.

    Args:
        wealth: Wealth value(s) (not used, included for interface consistency).
        gamma: Risk aversion parameter.

    Returns:
        Relative risk aversion (equal to gamma for CRRA).
    """
    if isinstance(wealth, torch.Tensor):
        return torch.full_like(wealth, gamma)
    return gamma