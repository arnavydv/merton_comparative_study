import torch
import torch.nn as nn


class ValueFunc(nn.Module):
  def __init__(self):
    super(ValueFunc, self).__init__()
    self.finalnetwork=nn.Sequential(
        nn.Linear(2,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,1),
    )
  def forward(self, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Ensure both inputs are 2D tensors with shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        # Concatenate along feature dimension
        inputs = torch.cat([t, w], dim=1)
        return self.finalnetwork(inputs)


# Backward compatibility alias
Value_func = ValueFunc