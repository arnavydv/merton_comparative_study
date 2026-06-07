import torch


def loss_function(Y: torch.Tensor) -> torch.Tensor:
    """Differentiable training loss.

    Expects Y with shape (num_paths, num_steps+1).
    Uses the variance of initial values Y[:, 0] as the loss.
    
    In BSDE theory, Y_0 = V(0, W_0) is the value function at time 0.
    For any initial state (t=0), the value function should be a deterministic
    scalar (same across all simulated paths). The loss minimises this variance
    so the neural network learns to produce a consistent value function.
    """
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y)

    initial = Y[:, 0]
    return torch.var(initial, unbiased=False)


