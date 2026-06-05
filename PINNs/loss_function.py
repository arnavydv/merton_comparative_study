import torch 
import json
from neural_network import Value_func

with open("config.json","r") as file:
    config=json.load(file)
rate=config["rate"]
sigma=config["sigma"]
mu=config["mu"]

def loss_function(model,x,t,rate,mu,sigma,):
    V=model(x,t)
    v_t=torch.autograd.grad(V,t,grad_outputs=torch.ones_like(V),create_graph=True)[0]
    v_x=torch.autograd.grad(V,x,grad_outputs=torch.ones_like(V),create_graph=True)[0]
    v_xx=torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(v_x),create_graph=True)[0]
    growth_term=rate*x*v_x
    volatility_term = 0.5*((mu-rate)**2)(v_x**2)/sigma**2*v_xx
    residual= v_t + growth_term + volatility_term
    return torch.mean(residual**2)
    