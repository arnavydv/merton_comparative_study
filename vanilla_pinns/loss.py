import torch
from neural_network import VanillaPINN
from collocation_points import collocation_points,terminal_points
gamma = 5.0
mu = torch.tensor(0.201649)
sigma = torch.tensor( 0.256573)
r = torch.tensor(0.0368)
T = 1.0
#collocation_points
w,t=collocation_points(10000,1,2,0.1)
#terminal_points
w_tc,t_tc,v_tc=terminal_points(2000,1,2,0.1,5)
model=VanillaPINN()

def hjb_residual(model,t,w,r,sigma,mu):
    v=model(w,t)
    v_t=torch.autograd.grad(v,t,create_graph=True,grad_outputs=torch.ones_like(v))[0]
    v_x=torch.autograd.grad(v,w,create_graph=True,grad_outputs=torch.ones_like(v))[0]
    v_xx=torch.autograd.grad(v_x,w,create_graph=True,grad_outputs=torch.ones_like(v_x))[0]
    residual = v_t*v_xx + r*w*v_xx*v_x - 0.5*torch.pow((((mu-r)*v_x)/sigma),torch.tensor(2))
    return torch.mean(residual**2),v_t,v_x,v_xx
hjb_loss,v_t,v_X,v_xx=hjb_residual(model,t,w,r,sigma,mu)
print(hjb_loss,":hjb loss")

def terminal_loss(model,w_tc,t_tc,v_tc):
    v_tc_pred=model(w_tc,t_tc)
    v_mean = v_tc.mean()
    v_std = v_tc.std()
    target = (v_tc-v_mean)/v_std
    pred = (v_tc_pred-v_mean)/v_std

    return torch.mean((target-pred)**2)
term=terminal_loss(model,w_tc,t_tc,v_tc)
print(terminal_loss(model,w_tc,t_tc,v_tc),":terminal_loss")

def concavity_loss(v_xx):
    return torch.mean(torch.relu(v_xx)**2)
print(concavity_loss(v_xx),":concavity loss")
concav=concavity_loss(v_xx)
def monotonocity_loss(v_x):
    return  torch.mean(torch.relu(-v_x)**2)
mono=monotonocity_loss(v_X)
print(monotonocity_loss(v_X),":monotonocity loss")

def total_loss(concavity_loss,monotonocity_loss,terminal_loss,hjb_residual,*weights):
    return hjb_residual+terminal_loss*10+monotonocity_loss*0.1+concavity_loss*0.1
print("total_loss",total_loss(concav,mono,term,hjb_loss))