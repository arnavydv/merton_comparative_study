import torch 
import numpy as np
import json
import os
from pathlib import Path
from neural_network import VanillaPINN
from loss import *
from collocation_points import collocation_points, terminal_points

# Load config
with open("vanilla_pinns/config.json", "r") as f:
    config = json.load(f)

gamma = config["gamma"]
mu = torch.tensor(config["mu"])
sigma = torch.tensor(config["sigma"])
r = torch.tensor(config["rate"])
T = 1.0
lr = 1e-3
epochs = 8000

# To run L-BFGS on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and map to computational device
model = VanillaPINN().to(device)

# Ensure parameters are mapped to the correct device
mu = mu.to(device)
sigma = sigma.to(device)
r = r.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create saved_models directory
save_dir = Path(__file__).resolve().parent / "saved_models"
save_dir.mkdir(exist_ok=True)

# For tracking losses
loss_history = []
pde_loss_history = []
terminal_loss_history = []
concavity_loss_history = []
mono_loss_history = []

best_loss = float('inf')

print(f"Starting Phase 1 (Adam) training for {epochs} epochs on {device}...")
print(f"Market params: r={r.item()}, mu={mu.item()}, sigma={sigma.item()}, gamma={gamma}")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Generate points and push them to active device
    w, t = collocation_points(25000, 1, 2, 0.5)
    w, t = w.to(device), t.to(device)
    
    w_tc, t_tc, v_tc = terminal_points(8000, 1, 2, 0.5, 5)
    w_tc, t_tc, v_tc = w_tc.to(device), t_tc.to(device), v_tc.to(device)
    
    # Compute your pipeline losses
    hjb_loss_final, v_t, v_X, v_xx = hjb_residual(model, t, w, r, sigma, mu)
    terminal_loss_final = terminal_loss(model, w_tc, t_tc, v_tc)
    concavity_loss_final = concavity_loss(v_xx)
    monotonocity_loss_final = monotonocity_loss(v_X)
    
    final_loss = total_loss(concavity_loss_final, monotonocity_loss_final, terminal_loss_final, hjb_loss_final)
    final_loss.backward()
    optimizer.step()
    
    # Track losses
    current_loss = final_loss.item()
    loss_history.append(current_loss)
    pde_loss_history.append(hjb_loss_final.item())
    terminal_loss_history.append(terminal_loss_final.item())
    concavity_loss_history.append(concavity_loss_final.item())
    mono_loss_history.append(monotonocity_loss_final.item())
    
    # Save best model
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': best_loss,
            'loss_history': loss_history,
            'pde_loss_history': pde_loss_history,
            'terminal_loss_history': terminal_loss_history,
            'concavity_loss_history': concavity_loss_history,
            'mono_loss_history': mono_loss_history,
            'config': config,
        }, save_dir / "best_model.pt")
    
    if epoch % 100 == 0:  # Changed to 10 to reduce terminal log spam
        print(
            f"Epoch {epoch:04d}"
            f" | Total={final_loss.item()}"
            f" | PDE={hjb_loss_final.item()}"
            f" | TC={terminal_loss_final.item()}"
            f" | Conc={concavity_loss_final.item()}"
            f" | Mono={monotonocity_loss_final.item()}"
        )

print("\n" + "="*50)
print("Transitioning to Phase 2: L-BFGS Second-Order Refinement...")
print("="*50)

# Instantiate L-BFGS optimizer targeting network weights
lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,                       # Newton-type methods work best starting with a full step size
    max_iter=50,                  # Maximum objective evaluations per step
    history_size=50,              # Quantifies curvature history memory
    line_search_fn="strong_wolfe" # Standard requirement to preserve PINN stability
)

lbfgs_epochs = 1000

for l_epoch in range(lbfgs_epochs):
    model.train()
    
    # Resample collocation points per epoch to ensure grid coverage diversity
    w_l, t_l = collocation_points(25000, 1, 2, 0.5)
    w_l, t_l = w_l.to(device), t_l.to(device)
    
    w_tc_l, t_tc_l, v_tc_l = terminal_points(8000, 1, 2, 0.5, 5)
    w_tc_l, t_tc_l, v_tc_l = w_tc_l.to(device), t_tc_l.to(device), v_tc_l.to(device)

    # L-BFGS requires a functional closure block to perform repeated line searches
    def closure():
        lbfgs_optimizer.zero_grad()
        
        hjb_l, _, v_X_l, v_xx_l = hjb_residual(model, t_l, w_l, r, sigma, mu)
        tc_l = terminal_loss(model, w_tc_l, t_tc_l, v_tc_l)
        conc_l = concavity_loss(v_xx_l)
        mono_l = monotonocity_loss(v_X_l)
        
        total_l = total_loss(conc_l, mono_l, tc_l, hjb_l)
        total_l.backward()
        return total_l

    # Execute optimizer calculation update step
    loss_val = lbfgs_optimizer.step(closure)
    current_loss = loss_val.item()
    
    # Run evaluation inside loop to capture loss components for telemetry logs
    hjb_val, _, v_X_val, v_xx_val = hjb_residual(model, t_l, w_l, r, sigma, mu)
    tc_val = terminal_loss(model, w_tc_l, t_tc_l, v_tc_l)
    conc_val = concavity_loss(v_xx_val)
    mono_val = monotonocity_loss(v_X_val)

    # Log values to historical tracking lists
    loss_history.append(current_loss)
    pde_loss_history.append(hjb_val.item())
    terminal_loss_history.append(tc_val.item())
    concavity_loss_history.append(conc_val.item())
    mono_loss_history.append(mono_val.item())

    # Update best model criteria if second-order phase unlocks a lower total loss minimum
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save({
            'epoch': epochs + l_epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': best_loss,
            'loss_history': loss_history,
            'pde_loss_history': pde_loss_history,
            'terminal_loss_history': terminal_loss_history,
            'concavity_loss_history': concavity_loss_history,
            'mono_loss_history': mono_loss_history,
            'config': config,
        }, save_dir / "best_model.pt")

    if l_epoch % 10 == 0:
        print(
            f"L-BFGS Epoch {l_epoch:03d}"
            f" | Total={current_loss}"
            f" | PDE={hjb_val.item()}"
            f" | TC={tc_val.item()}"
            f" | Conc={conc_val.item()}"
            f" | Mono={mono_val.item()}"
        )

torch.save({
    'epoch': epochs + lbfgs_epochs,
    'model_state_dict': model.state_dict(),
    'loss': loss_history[-1],
    'loss_history': loss_history,
    'pde_loss_history': pde_loss_history,
    'terminal_loss_history': terminal_loss_history,
    'concavity_loss_history': concavity_loss_history,
    'mono_loss_history': mono_loss_history,
    'config': config,
}, save_dir / "final_model.pt")

print("\n" + "="*50)
print("ALL TRAINING PHASES COMPLETED!")
print("="*50)
print(f"Best Loss Attained : {best_loss}")
print(f"Final Step Loss    : {loss_history[-1]}")
print(f"Models Saved To    : {save_dir}")
