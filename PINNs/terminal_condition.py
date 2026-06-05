import torch


def terminal_condition(number_of_points, T, w_max, w_min, gamma):
    """Generate terminal condition points for the HJB PDE.

    Returns:
        t_terminal: time tensor all equal to T (shape: number_of_points,)
        w_terminal: wealth tensor linearly spaced in [w_min, w_max] (shape: number_of_points,)
        v_terminal: CRRA utility values V(T, w) = U(w) (shape: number_of_points,)
    """
    t = torch.ones(number_of_points) * T
    w = torch.linspace(w_min, w_max, number_of_points)
    v = crra(w, gamma)
    return t, w, v


def crra(wealth, gamma):
    """CRRA utility function U(w) = w^(1-gamma) / (1-gamma).

    Handles gamma=1 as the limiting case (log utility).
    """
    if gamma == 1.0:
        return torch.log(wealth)
    term = 1.0 - gamma
    return wealth ** term / term