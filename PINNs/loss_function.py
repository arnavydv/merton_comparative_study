import torch 
from neural_network import Value_func



def loss_function(model,x,t,rate,mu,sigma,):
    V=model(x,t)
    v_t=torch.autograd.grad(V,t,grad_outputs=torch.ones_like(V),create_graph=True)[0]
    v_x=torch.autograd.grad(V,x,grad_outputs=torch.ones_like(V),create_graph=True)[0]
    v_xx=torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(v_x),create_graph=True)[0]
    residual=
    