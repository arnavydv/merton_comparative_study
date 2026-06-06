import torch 
import numpy as np
from neural_network import VanillaPINN
from loss import *
from collocation_points import collocation_points,terminal_points
gamma = 5.0
mu = torch.tensor(0.201649)
sigma = torch.tensor( 0.256573)
r = torch.tensor(0.0368)
T = 1.0
lr=1e-3
epochs=5000
model = VanillaPINN()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    optimizer.zero_grad()
    w,t=collocation_points(10000,1,2,0.5)
    w_tc,t_tc,v_tc=terminal_points(2000,1,2,0.5,5)
    hjb_loss_final,v_t,v_X,v_xx=hjb_residual(model,t,w,r,sigma,mu)
    terminal_loss_final=terminal_loss(model,w_tc,t_tc,v_tc)
    concavity_loss_final=concavity_loss(v_xx)
    monotonocity_loss_final=monotonocity_loss(v_X)
    final_loss=total_loss(concavity_loss_final,monotonocity_loss_final,terminal_loss_final,hjb_loss_final)
    final_loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(
            f"Epoch {epoch}"
            f" | Total={final_loss.item()}"
            f" | PDE={hjb_loss_final.item()}"
            f" | TC={terminal_loss_final.item()}"
            f" | Conc={concavity_loss_final.item()}"
            f" | Mono={monotonocity_loss_final.item()}"
        )

    