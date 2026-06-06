import torch 
import json
import os
from pathlib import Path
from loss_function import loss_function, compute_phi_pde_residual
from neural_network import ValueFunc
from terminal_condition import terminal_points
from collocation_points import collocation_points
import matplotlib.pyplot as plt
from tqdm import tqdm

with open(r"C:\Users\Admin\Desktop\HJB_MERTON_BENCHMARK\PINNs\data_accumulation\config.json","r") as file:
    config=json.load(file)

# Use the new ValueFunc which internally uses the φ-factorization
model = ValueFunc(gamma=config["gamma"])

# Domain bounds for the numerical method (not configurable market parameters)
W_MAX = 2.0
W_MIN = 0.1
T_val = 1.0  # normalized time

# Collocation points: (N_points, T, w_max, w_min)
w, t = collocation_points(10000, T_val, W_MAX, W_MIN)

rate = config["rate"]
sigma = config["sigma"]
mu = config["mu"]

# Terminal points
w_tc, t_tc, v_tc = terminal_points(2000, T_val, W_MAX, W_MIN, config["gamma"])

# Optimizer: φ(t) is a much simpler problem, so we can use a lower LR
optimizer = torch.optim.Adam(model.phi_net.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=500
)

# Tracking
loss_history = []
best_loss = float('inf')

epochs = 5000

# Create saved_models directory
save_dir = Path(__file__).resolve().parent / "saved_models"
save_dir.mkdir(exist_ok=True)

print(f"Starting training for {epochs} epochs...")
print(f"Market params: r={rate}, mu={mu}, sigma={sigma}, gamma={model.gamma}")
print(f"kappa (analytical decay rate) = {(1-model.gamma)*(rate+0.5*(mu-rate)**2/(sigma**2*model.gamma)):.4f}")
print(f"Optimal pi* (analytical) = {(mu - rate) / (sigma**2 * model.gamma):.4f}")

for epoch in tqdm(range(epochs), desc="Training phi-PINN"):
    optimizer.zero_grad()
    
    loss = loss_function(model, w, t, rate, sigma, mu, w_tc, v_tc, t_tc)
    
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.phi_net.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Step the scheduler with current loss (ReduceLROnPlateau mode)
    scheduler.step(loss)
    
    current_loss = loss.item()
    loss_history.append(current_loss)
    
    # Save best model
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'phi_net_state_dict': model.phi_net.state_dict(),
            'loss': best_loss,
            'loss_history': loss_history,
            'config': config,
        }, save_dir / "best_model.pt")
    
    if (epoch + 1) % 10 == 0:
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Loss: {current_loss:.6e} | Best: {best_loss:.6e}")

# Save final model
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'phi_net_state_dict': model.phi_net.state_dict(),
    'loss': current_loss,
    'loss_history': loss_history,
    'config': config,
}, save_dir / "final_model.pt")

print("Training completed!")
print(f"Best loss: {best_loss:.6e}")
print(f"Final loss: {loss_history[-1]:.6e}")
print(f"Models saved to: {save_dir}")

# Plot the loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Total Loss', color='blue')
plt.axhline(y=best_loss, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_loss:.2e}')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('phi-PINN Training Convergence')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()