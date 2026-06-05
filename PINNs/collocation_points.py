from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.stats import qmc


def collocation_points(number_of_points: int, T: float, w_max: float, w_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate collocation points using a 2D Latin Hypercube sample.

    Returns:
        (t, w) as torch tensors of shape (number_of_points,).
    """
    if number_of_points <= 0:
        raise ValueError("number_of_points must be a positive integer")
    if w_min > w_max:
        raise ValueError("w_min must be <= w_max")

    T = float(T)
    w_min = float(w_min)
    w_max = float(w_max)

    sampler = qmc.LatinHypercube(d=2)

    lower_bound = np.array([0.0, w_min], dtype=np.float64)
    upper_bound = np.array([T, w_max], dtype=np.float64)

    # SciPy returns float64 numpy arrays.
    samples = sampler.random(n=number_of_points)
    scaled_samples = qmc.scale(samples, lower_bound, upper_bound)

    t = scaled_samples[:, 0]
    w = scaled_samples[:, 1]

    # Convert to float32 torch tensors for typical PINN usage.
    t_t = torch.as_tensor(t, dtype=torch.float32)
    w_t = torch.as_tensor(w, dtype=torch.float32)
    return t_t, w_t


