import torch 
from scipy.stats import qmc
def collocation_points(number_of_points,T,w_max,w_min):
  sampler=qmc.LatinHypercube(d=2)
  lower_bound=[0,w_min]
  upper_bound=[T,w_max]
  samples=sampler.random(number_of_points)
  scaled_samples=qmc.scale(samples,lower_bound,upper_bound)
  w=torch.tensor(scaled_samples[:,1],dtype=torch.float32).unsqueeze(1)
  t=torch.tensor(scaled_samples[:,0],dtype=torch.float32).unsqueeze(1)
  t.requires_grad=True
  w.requires_grad=True
  return w,t