import torch 
from scipy.stats import qmc

def terminal_points(number_of_points,T,w_max,w_min,gamma):
  sampler = qmc.LatinHypercube(d=1)
  samples = sampler.random(number_of_points)
  scaled_samples = qmc.scale(samples,w_min,w_max)
  w_tc = torch.tensor(scaled_samples,dtype=torch.float32)
  t_tc = torch.ones(number_of_points,1) * T
  v_tc = (w_tc**(1-gamma))/(1-gamma)
  w_tc.requires_grad = True
  t_tc.requires_grad = True
  return w_tc,t_tc,v_tc


def crra(wealth, gamma):
    """CRRA utility function U(w) = w^(1-gamma)/(1-gamma) for tensor inputs."""
    if gamma == 1.0:
        return torch.log(wealth)
    term = 1.0 - gamma
    return (wealth ** term) / term


def optimal_portfolio_weight(gamma, rate, mu, sigma):
    """Analytical optimal portfolio weight: pi* = (mu - r) / (gamma * sigma^2)."""
    return (mu - rate) / (gamma * sigma**2)


def merton_analytical_solution(t, w, T, gamma, rate, mu, sigma):
    """Analytical solution to the Merton HJB PDE with CRRA utility.
    
    V(t,w) = (w^(1-gamma)/(1-gamma)) * exp(kappa * (T - t))
    where kappa = (1-gamma) * (r + 0.5 * (mu-r)^2 / (sigma^2 * gamma))
    """
    kappa = (1.0 - gamma) * (rate + 0.5 * (mu - rate)**2 / (sigma**2 * gamma))
    term = 1.0 - gamma
    u_w = (w ** term) / term
    return u_w * torch.exp(kappa * (T - t))