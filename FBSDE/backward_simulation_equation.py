import torch

from utility_function import utility_function


def A(gamma, r, mu, sigma):
    """
    Auxiliary coefficient used in the backward recursion.

    terms = r + (mu - r)^2 / (2 * gamma * sigma^2)
    A = (1 - gamma) * terms
    """
    terms = r + ((mu - r) ** 2 / (2 * gamma * (sigma**2)))
    return (1 - gamma) * terms


def _ensure_col(x):
    # Convert (N,) -> (N,1) and keep (N,1) as-is
    if isinstance(x, torch.Tensor) and x.ndim == 1:
        return x.unsqueeze(1)
    return x


def backward_simulation(
    wealth_grid,
    time_grid,
    dw,
    W,
    Z_model,
    gamma,
    r,
    mu,
    sigma,
    utility_fn=utility_function,
):
    """
    Discrete-time backward simulation.

    Args:
        wealth_grid: (num_paths, num_steps+1) torch tensor
        time_grid: numpy array of length (num_steps+1)
        dw: (num_paths, num_steps) torch tensor
        W: (num_paths, num_steps) torch tensor
        Z_model: callable (t_col, W_col) -> (num_paths,1) tensor
        gamma, r, mu, sigma: scalars
        utility_fn: terminal utility U(W_T, gamma)

    Returns:
        Y: (num_paths, num_steps+1) torch tensor
    """
    if not isinstance(wealth_grid, torch.Tensor):
        wealth_grid = torch.as_tensor(wealth_grid)
    if not isinstance(dw, torch.Tensor):
        dw = torch.as_tensor(dw)
    if not isinstance(W, torch.Tensor):
        W = torch.as_tensor(W)

    num_paths, num_steps_plus_1 = wealth_grid.shape
    num_steps = num_steps_plus_1 - 1

    # dt = T/num_steps
    T = float(time_grid[-1] - time_grid[0])
    dt = T / num_steps

    a = A(gamma=gamma, r=r, mu=mu, sigma=sigma)

    # Terminal condition (out-of-place)
    Y_T = utility_fn(wealth_grid[:, -1], gamma).view(-1)  # (num_paths,)
    Y_list = [None] * (num_steps + 1)
    Y_list[-1] = Y_T

    # Backward recursion (no in-place writes into a tensor that autograd tracks)
    Y_next = Y_T
    for n in reversed(range(num_steps)):
        t_n = torch.full(
            (num_paths, 1),
            float(time_grid[n]),
            dtype=wealth_grid.dtype,
            device=wealth_grid.device,
        )
        W_n = _ensure_col(W[:, n].to(dtype=wealth_grid.dtype, device=wealth_grid.device))
        dw_n = dw[:, n].to(dtype=wealth_grid.dtype, device=wealth_grid.device)

        z_n = Z_model(t_n, W_n)  # expected (num_paths,1) or (num_paths,)
        z_n = z_n.squeeze(1) if z_n.ndim == 2 and z_n.shape[1] == 1 else z_n

        rhs = Y_next - z_n * dw_n
        Y_curr = rhs / (1.0 + a * dt)  # (num_paths,)

        Y_list[n] = Y_curr
        Y_next = Y_curr

    Y = torch.stack(Y_list, dim=1)  # (num_paths, num_steps+1)
    return Y
