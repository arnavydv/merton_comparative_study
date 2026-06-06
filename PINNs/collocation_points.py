from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.stats import qmc
def collocation_points(
    num_points: int,
    T: float,
    w_max: float,
    w_min: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_points <= 0:
        raise ValueError(f"num_points must be a positive integer, got {num_points}")
    if w_min > w_max:
        raise ValueError(f"w_min must be <= w_max, got w_min={w_min}, w_max={w_max}")
    if w_min <= 0:
        raise ValueError(f"w_min must be positive for CRRA utility, got {w_min}")

    T = float(T)
    w_min = float(w_min)
    w_max = float(w_max)

    # Create Latin Hypercube sampler for 2 dimensions (time, wealth)
    sampler = qmc.LatinHypercube(d=2)

    # Define bounds for each dimension
    lower_bound = np.array([0.0, w_min], dtype=np.float64)
    upper_bound = np.array([T, w_max], dtype=np.float64)

    # Generate samples
    samples = sampler.random(n=num_points)
    scaled_samples = qmc.scale(samples, lower_bound, upper_bound)

    # Extract time and wealth columns
    t = scaled_samples[:, 0]
    w = scaled_samples[:, 1]

    # Convert to float32 torch tensors for typical PINN usage
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    w_tensor = torch.as_tensor(w, dtype=torch.float32)

    return t_tensor, w_tensor