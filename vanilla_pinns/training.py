import torch 
import numpy as np
import json
import os
from pathlib import Path
from neural_network import VanillaPINN
from loss import *
from collocation_points import collocation_points,terminal_points

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

model = VanillaPINN()
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

# Print config
print(f"Starting training for {epochs} epochs...")
print(f"Market params: r={r.item()}, mu={mu.item()}, sigma={sigma.item()}, gamma={gamma}")

for epoch in range(epochs):
    optimizer.zero_grad()
    w, t = collocation_points(25000, 1, 2, 0.5)
    w_tc, t_tc, v_tc = terminal_points(8000, 1, 2, 0.5, 5)
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
    
    if epoch % 1 == 0:
        print(
            f"Epoch {epoch}"
            f" | Total={final_loss.item()}"
            f" | PDE={hjb_loss_final.item()}"
            f" | TC={terminal_loss_final.item()}"
            f" | Conc={concavity_loss_final.item()}"
            f" | Mono={monotonocity_loss_final.item()}"
        )

# Save final model
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'loss': loss_history[-1],
    'loss_history': loss_history,
    'pde_loss_history': pde_loss_history,
    'terminal_loss_history': terminal_loss_history,
    'concavity_loss_history': concavity_loss_history,
    'mono_loss_history': mono_loss_history,
    'config': config,
}, save_dir / "final_model.pt")

print("Training completed!")
print(f"Best loss: {best_loss:.6e}")
print(f"Final loss: {loss_history[-1]:.6e}")
print(f"Models saved to: {save_dir}")

    