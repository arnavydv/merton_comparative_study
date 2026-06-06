import torch
import torch.nn as nn
import math

class PhiNet(nn.Module):
    """
    Learns the time-dependent factor φ(t) in the CRRA-factorized solution:
        V(t, w) = φ(t) * U(w)   where   U(w) = w^(1-γ) / (1-γ)
    
    The terminal condition is φ(T) = 1.
    Using this factorization eliminates the massive scale range of V(t,w)
    when γ is large (e.g. γ=5 gives V ∈ [-2500, -0.016]).
    """
    def __init__(self):
        super().__init__()
        
        # Fourier features for time input — helps learn periodic/smooth functions
        # Use 8 frequency bands → 16 sin+cos features + 1 t = 17 total input
        self.freqs = nn.Parameter(torch.randn(8) * 2.0, requires_grad=False)
        
        # Smaller, deeper network since we only need φ(t) not V(t,w)
        self.net = nn.Sequential(
            nn.Linear(1 + 16, 64),   # t + 16 fourier features (8 sin + 8 cos)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        """
        Args:
            t: (N, 1) tensor of time values in [0, T]
        Returns:
            φ(t): (N, 1) positive scalar factor
        """
        # Fourier feature embedding
        freqs = self.freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, 16)
        t_expanded = t.unsqueeze(-1)                   # (N, 1, 1)
        fourier_feats = torch.cat([
            torch.sin(2 * math.pi * freqs * t_expanded),
            torch.cos(2 * math.pi * freqs * t_expanded),
        ], dim=-1)                                     # (N, 1, 32)
        fourier_feats = fourier_feats.squeeze(1)       # (N, 32)
        
        x = torch.cat([t, fourier_feats], dim=-1)      # (N, 33)
        phi = self.net(x)
        
        # Ensure φ(t) > 0 (terminal condition is φ=1, so output should be near 1)
        # Use softplus centered around 1: φ = softplus(z) + 1
        return torch.nn.functional.softplus(phi) + 1e-6


class ValueFunc(nn.Module):
    """
    Wrapper that computes V(t,w) = φ(t) * U(w) for the CRRA utility.
    This is the actual value function that can be used by evaluation code.
    """
    def __init__(self, gamma=5.0):
        super().__init__()
        self.gamma = gamma
        self.phi_net = PhiNet()
    
    def forward(self, t, w):
        """
        Args:
            t: (N, 1) time
            w: (N, 1) wealth
        Returns:
            V(t,w): (N, 1) value function
        """
        phi = self.phi_net(t)
        # U(w) = w^(1-γ) / (1-γ)
        term = 1.0 - self.gamma
        u_w = (w ** term) / term
        return phi * u_w