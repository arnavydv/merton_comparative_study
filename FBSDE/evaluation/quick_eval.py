"""
quick_eval.py — Fast evaluation of pi_star network without plots.
Prints statistical metrics comparing model predictions vs analytical target.
"""
import sys
import os
import json
import torch
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from neural_networks import pi_star
from evaluation.data_generation import inference_inputs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = r"C:/Users/Admin/Desktop/FBSDE PROJECT/models_and_experiments/best.pth"
CONFIG_PATH = r"C:/Users/Admin/Desktop/FBSDE PROJECT/config.json"
NUM_POINTS = 5000
T = 1.0

# ---------------------------------------------------------------------------
# 1. Load config
# ---------------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

mu = cfg["mu"]
sigma = cfg["sigma"]
rate = cfg["rate"]
gamma = cfg["gamma"]

# Analytical Merton solution
exact_pi = (mu - rate) / (gamma * sigma ** 2)

print(f"Market Params: mu={mu:.6f}, sigma={sigma:.6f}, r={rate:.6f}, gamma={gamma}")
print(f"Analytical Target pi*: {exact_pi:.6f}")
print(f"Config: num_steps={cfg['num_steps']}, num_paths={cfg['num_paths']}")

# ---------------------------------------------------------------------------
# 2. Load trained model
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nLoading model on {device} ...")

model = pi_star()
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Check if this is the new format (has "pi_star" key) or old format (direct state_dict)
if "pi_star" in checkpoint:
    model.load_state_dict(checkpoint["pi_star"])
    if "epoch" in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs, loss={checkpoint.get('loss', 'N/A'):.6e}")
        print(f"  BSDE loss: {checkpoint.get('bsde_loss', 'N/A')}")
        print(f"  PI sup loss: {checkpoint.get('pi_sup_loss', 'N/A')}")
else:
    # Try direct loading (might be old model)
    model.load_state_dict(checkpoint)
    
model.eval()
model.to(device)

# ---------------------------------------------------------------------------
# 3. Evaluate on many random (t, W) points
# ---------------------------------------------------------------------------
print(f"\nEvaluating on {NUM_POINTS} random (t, W) points...")

t_vals, W_vals = inference_inputs(num_points=NUM_POINTS, T=T, device=device)

with torch.no_grad():
    predictions = model(t_vals, W_vals)

pred_np = predictions.cpu().numpy().flatten()

# ---------------------------------------------------------------------------
# 4. Statistical metrics
# ---------------------------------------------------------------------------
mean_pred = pred_np.mean()
std_pred = pred_np.std()
mae = np.mean(np.abs(pred_np - exact_pi))
rmse = np.sqrt(np.mean((pred_np - exact_pi) ** 2))
max_abs_err = np.max(np.abs(pred_np - exact_pi))
min_pred = pred_np.min()
max_pred = pred_np.max()

print(f"\n{'='*60}")
print(f"  EVALUATION RESULTS")
print(f"{'='*60}")
print(f"  Analytical pi* (target): {exact_pi:.6f}")
print(f"  Predictions Mean       : {mean_pred:.6f}")
print(f"  Predictions Std        : {std_pred:.6f}")
print(f"  Min / Max Prediction   : {min_pred:.6f} / {max_pred:.6f}")
print(f"  Mean Absolute Error    : {mae:.6f}")
print(f"  RMSE                   : {rmse:.6f}")
print(f"  Max Absolute Error     : {max_abs_err:.6f}")
print(f"{'='*60}")

if mae < 0.01:
    print(f"  [TARGET ACHIEVED] MAE = {mae:.6f} < 0.01")
else:
    print(f"  [FAIL] MAE = {mae:.6f} > 0.01 (need improvement)")

# ---------------------------------------------------------------------------
# 5. Evaluate on a grid for more comprehensive check
# ---------------------------------------------------------------------------
print(f"\nGrid evaluation (2000 points along t=0 to T)...")
t_grid = torch.linspace(0.01, T, 2000, device=device).view(-1, 1)
W_grid = torch.randn(2000, 1, device=device) * t_grid.sqrt()

with torch.no_grad():
    grid_pred = model(t_grid, W_grid).cpu().numpy().flatten()

grid_mean = grid_pred.mean()
grid_mae = np.mean(np.abs(grid_pred - exact_pi))
print(f"  Grid Prediction Mean: {grid_mean:.6f}")
print(f"  Grid MAE           : {grid_mae:.6f}")
print(f"  Grid Min / Max     : {grid_pred.min():.6f} / {grid_pred.max():.6f}")

# Check if predictions are always positive (as expected for positive pi*)
if grid_pred.min() < 0:
    neg_pct = (grid_pred < 0).mean() * 100
    print(f"  ⚠ WARNING: {neg_pct:.1f}% of predictions are negative (should be ~0% for analytical pi*={exact_pi:.4f})")
else:
    print(f"  ✓ All predictions are positive (correct behavior)")