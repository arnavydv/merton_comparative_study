import torch

def compute_phi_pde_residual(phi_net, t, rate, mu, sigma, gamma):
    """
    Compute the PDE residual for the φ(t) formulation.
    
    With V(t,w) = φ(t) * U(w) for CRRA utility U(w) = w^(1-γ)/(1-γ),
    the HJB PDE reduces to:
        φ'(t) + κ * φ(t) = 0
    where κ = (1-γ) * (r + 0.5 * (μ-r)² / (σ² * γ))
    
    Args:
        phi_net: PhiNet module
        t: (N, 1) time points requiring gradient
        rate: risk-free rate r
        mu: expected return μ
        sigma: volatility σ
        gamma: risk aversion γ
    
    Returns:
        residual: (N, 1) PDE residual
    """
    phi = phi_net(t)
    
    # Compute φ'(t) via autograd
    phi_t = torch.autograd.grad(
        phi, t,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
    )[0]
    
    # κ = (1-γ) * (r + 0.5 * (μ-r)² / (σ² * γ))
    kappa = (1.0 - gamma) * (rate + 0.5 * (mu - rate)**2 / (sigma**2 * gamma))
    
    # PDE: φ'(t) + κ * φ(t) = 0
    residual = phi_t + kappa * phi
    
    return residual


def loss_function(model, w, t, rate, sigma, mu, w_tc, v_tc, t_tc):
    """
    Total loss for the φ-PINN formulation.
    
    Since V(t,w) = φ(t) * U(w), we only need to learn φ(t), and
    the PDE residual and terminal condition naturally scale with U(w).
    
    Args:
        model: ValueFunc (contains phi_net internally)
        w: (N, 1) collocation wealth points
        t: (N, 1) collocation time points
        rate, sigma, mu: market parameters
        w_tc: (M, 1) terminal wealth points
        v_tc: (M, 1) terminal condition values
        t_tc: (M, 1) terminal time points (all = T)
    
    Returns:
        total_loss: scalar tensor
    """
    # Extract the phi_net from the ValueFunc
    phi_net = model.phi_net
    
    # --- PDE Loss (using φ residual) ---
    pde_res = compute_phi_pde_residual(phi_net, t, rate, mu, sigma, model.gamma)
    pde_loss = torch.mean(pde_res ** 2)
    
    # --- Terminal Condition Loss ---
    # φ(T) should equal 1 for all wealth values
    # But we scale the TC loss by the utility to keep relative error balanced
    phi_at_T = phi_net(t_tc)
    # Target: φ(T) = 1
    tc_loss = torch.mean((phi_at_T - 1.0) ** 2)
    
    # --- Concavity Penalty (optional, HJB structure already ensures this for CRRA) ---
    # For CRRA utility, V_ww < 0 automatically if φ(t) > 0, so no extra penalty needed.
    # But we add a tiny positive constraint on φ for numerical stability.
    phi_positive_penalty = torch.mean(torch.relu(1e-6 - phi_at_T) ** 2)
    
    # --- Combine ---
    # The PDE residual scale is naturally correct (~0 for perfect solution)
    # TC loss is directly on φ (range near 1), so no scaling issues
    total_loss = pde_loss + 1.0 * tc_loss + 0.1 * phi_positive_penalty
    
    return total_loss


# Keep the original function signature for backward compatibility
def loss_function_original(model, w, t, rate, sigma, mu, w_tc, v_tc, t_tc):
    """Original loss function for V(t,w) direct learning (deprecated, use loss_function instead)."""
    return loss_function(model, w, t, rate, sigma, mu, w_tc, v_tc, t_tc)