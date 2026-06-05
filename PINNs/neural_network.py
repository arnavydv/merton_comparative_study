import torch 
import torch.nn as nn

class PINNS(nn.Module):
  def __init__(self):
    super(PINNS, self).__init__()
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
  def forward(self,w,x):
    input_tensor=torch.cat((w,x),dim=1)
    return self.finalnetwork(input_tensor)