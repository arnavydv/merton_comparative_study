import numpy as np
import torch
def brownian_motion(num_paths, num_steps, dt, device=None):
    device = device or torch.device("cpu")
    noise = torch.randn(num_paths, num_steps, device=device)
    dw = noise * np.sqrt(dt)
    W = torch.cumsum(dw, axis=1)
    return dw, W
