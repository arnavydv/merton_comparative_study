"""
data_generation.py

Generates synthetic data for evaluating the trained neural network pi*(t, W).
Includes random sampling for statistical evaluation and grid generation for plotting.
"""

import torch
import numpy as np

def time_grid(T: float, num_steps: int, device: torch.device | None = None) -> np.ndarray:
    return np.linspace(0, T, num_steps + 1)

def brownian_paths(
    num_paths: int,
    num_steps: int,
    T: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates cumulative Brownian motion paths."""
    device = device or torch.device("cpu")
    dt = T / num_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))
    dw = torch.randn(num_paths, num_steps, device=device) * sqrt_dt
    W = torch.cumsum(dw, dim=1)
    return dw, W

def inference_inputs(
    num_points: int, T: float = 1.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates independent (t, W) pairs for statistical evaluation."""
    device = device or torch.device("cpu")
    t_vals = torch.rand(num_points, 1, device=device) * T
    std = t_vals.sqrt()
    W_vals = torch.randn(num_points, 1, device=device) * std
    return t_vals, W_vals

def inference_grid(
    t_steps: int = 50, W_range: float = 3.0, W_steps: int = 50, T: float = 1.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a structured meshgrid of (t, W) for 3D surface plotting.
    W bounds expand with time to mimic the envelope of Brownian motion.
    """
    device = device or torch.device("cpu")
    
    t_np = np.linspace(0.01, T, t_steps)
    W_np = np.linspace(-W_range, W_range, W_steps)
    
    t_mesh, W_mesh = np.meshgrid(t_np, W_np)
    
    # Flatten for network input
    t_col = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    W_col = torch.tensor(W_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    return t_col, W_col