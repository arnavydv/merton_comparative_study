import numpy as np
import torch
from neural_networks import pi_star, Z_net
from brownian_motion import brownian_motion

def forward_simulation(model,  mu, sigma, w0, rate, T, num_steps,num_paths):
    """
    Simulate wealth dynamics with a learned/parameterized pi(t, W).

    Args:
      model: torch.nn.Module with forward(self, t, W) -> (num_paths, 1)
      dw: torch.Tensor of shape (num_paths, num_steps)
      W:  torch.Tensor of shape (num_paths, num_steps)
      mu, sigma, rate: scalars
    """
    # Infer device from model parameters
    device = next(model.parameters()).device
    dw, W = brownian_motion(num_paths, num_steps, T/num_steps, device=device)
    time_grid = np.linspace(0, T, num_steps + 1)
    dt = T / num_steps

    num_paths = dw.shape[0]

    sigma_t = torch.as_tensor(sigma, dtype=dw.dtype, device=device)
    mu_t = torch.as_tensor(mu, dtype=dw.dtype, device=device)
    rate_t = torch.as_tensor(rate, dtype=dw.dtype, device=device)

    # Out-of-place accumulation to keep autograd graph consistent
    w0_t = torch.as_tensor(w0, dtype=dw.dtype, device=device).expand(num_paths)
    wealth_list = [w0_t]

    for i in range(num_steps):
        t = float(time_grid[i])
        w = wealth_list[-1]  # (num_paths,)

        t_col = torch.full((num_paths, 1), t, dtype=dw.dtype, device=device)
        W_col = W[:, i : i + 1]  # (num_paths,1)

        # pi(t, W) expected to return shape (num_paths,1)
        pi = model(t_col, W_col)  # (num_paths,1)
        # Wealth SDE (discretized) with PI interpreted as a FRACTION of wealth:
        # drift    = r*w + (pi*w)*(mu-r)
        # diffusion= (pi*w)*sigma*dW
        w_col = w.view(-1, 1)
        drift = rate_t * w_col + (pi * w_col) * (mu_t - rate_t)  # (num_paths,1)
        diffusion = (pi * w_col) * sigma_t * dw[:, i : i + 1]  # (num_paths,1)
        wealth_next = w_col + drift * dt + diffusion  # (num_paths,1)
        # Smooth lower bound: softplus(x) ≈ x for x > 0, ≈ 0 for x << 0
        # +1e-8 ensures strict positivity while preserving gradients throughout
        wealth_next = torch.nn.functional.softplus(wealth_next) + 1e-8

        wealth_list.append(wealth_next.squeeze(1))

    wealth_grid = torch.stack(wealth_list, dim=1)  # (num_paths, num_steps+1)
    return time_grid, wealth_grid, dw, W