"""Direct V(t,w) network - learns value function directly without phi-factorization."""
import torch
import torch.nn as nn

class VanillaVNet(nn.Module):
    """
    Learns V(t,w) directly. Inputs are normalized, output is the value function.
    No phi-factorization - pure PINN approach.
    """
    def __init__(self, gamma=5.0):
        super().__init__()
        self.gamma = gamma
        
        # Network: takes (t_norm, w_norm) → output V(t,w)
        # Deep network to capture the nonlinearity
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Scaling factor for the output
        # U(w_min) = w_min^(1-gamma)/(1-gamma) for gamma=5, w_min=0.1
        # This gives ≈ -2500. We'll use this to scale the output.
        w_min = 0.1
        u_w_min = (w_min ** (1 - gamma)) / (1 - gamma)
        self.register_buffer('scale_factor', torch.tensor(abs(u_w_min)))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def normalize(self, t, w):
        """Normalize inputs to [0, 1] range."""
        t_norm = t / 1.0  # T = 1.0
        w_norm = (w - 0.1) / (2.0 - 0.1)  # w ∈ [0.1, 2.0]
        return t_norm, w_norm
    
    def forward(self, t, w):
        """
        Args:
            t: (N, 1) time
            w: (N, 1) wealth
        Returns:
            V: (N, 1) value function estimate
        """
        t_norm, w_norm = self.normalize(t, w)
        x = torch.cat([t_norm, w_norm], dim=1)
        
        # Network outputs a normalized value
        V_norm = self.net(x)
        
        # Denormalize: V = V_norm * scale_factor
        # But V is negative (since gamma > 1), so we also need to handle sign
        # For gamma=5, U(w) is negative. We model V as: V = -scale_factor * exp(raw_output)
        # This ensures V < 0 and has the right magnitude range
        V = -self.scale_factor * torch.exp(V_norm)
        
        return V


# Keep alias for backward compatibility
ValueFuncDirect = VanillaVNet