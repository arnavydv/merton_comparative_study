import os
import torch

# Allow both: `python vanilla_pinns/evaluation.py` and `python -m vanilla_pinns.evaluation`
try:
    from ..neural_network import VanillaPINN
    from ..collocation_points import collocation_points
except ImportError:  # pragma: no cover
    from neural_network import VanillaPINN
    from collocation_points import collocation_points


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)

    # training.py saves a dict with key: 'model_state_dict'
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # Fallback: treat ckpt as state_dict
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate collocation points (Points are generated over wealth and time)
    w, t = collocation_points(5000, 1, 2, 0.5)
    w = w.to(device)
    t = t.to(device)

    # Enable gradient tracking on input variables for Autograd
    w.requires_grad_(True)
    t.requires_grad_(True)

    # Initialize and set model to evaluation mode
    model = VanillaPINN().to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "saved_models", "best_model.pt")
    load_model_checkpoint(model, ckpt_path, device)

    # Model inference pass
    predict = model(w, t)
    
    # Compute V_W (First derivative with respect to Wealth)
    v_w = torch.autograd.grad(
        predict, w, 
        grad_outputs=torch.ones_like(predict), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # Compute V_WW (Second derivative with respect to Wealth)
    v_ww = torch.autograd.grad(
        v_w, w, 
        grad_outputs=torch.ones_like(v_w), 
        create_graph=True
    )[0]
        
    # Constant parameters matching your setup
    Time = 252
    sigma = 0.256573
    mu = 0.201649
    rate = 0.0368
    gamma = 5.0

    # 1. FIXED ALIGNMENT: Divide by wealth 'w' to convert absolute strategy to wealth fraction
    pi_star_fraction = - ((mu - rate) * v_w) / (w * (sigma ** 2) * v_ww)
    
    # 2. True Merton analytical value (constant fraction of wealth)
    pi_star_analytical = (mu - rate) / ((sigma ** 2) * gamma)
    pi_star_analytical_tensor = torch.tensor(pi_star_analytical, device=device)

    # Detach tensors for safety during logging and metric extraction
    pi_star_fraction_detached = pi_star_fraction.detach()
    v_ww_detached = v_ww.detach()

    # 3. Calculate actual Mean Absolute Error (MAE) 
    mae = torch.mean(torch.abs(pi_star_fraction_detached - pi_star_analytical_tensor))

    # =============================================================================
    # DIAGNOSTIC LOGGING
    # =============================================================================
    print("\n" + "="*50)
    print("               PINN EVALUATION REPORT            ")
    print("="*50)
    print(f"Merton Analytical Benchmark : {pi_star_analytical:.6f}")
    print(f"PINN Predicted Mean Fraction: {pi_star_fraction_detached.mean().item():.6f}")
    print(f"Mean Absolute Error (MAE)   : {mae.item():.6f}")
    print("-"*50)
    
    # Check for mathematical concavity breakdown (V_WW must always be negative)
    positive_v_ww_count = (v_ww_detached >= 0).sum().item()
    if positive_v_ww_count > 0:
        print(f"⚠️ WARNING: {positive_v_ww_count} / 5000 collocation points broken concavity (V_WW >= 0)!")
        print("  -> This causes division errors or flipped signs in your strategy.")
        print("  -> Solution: Ensure you use smooth activations (Tanh/SiLU) and increase PINN loss weights.")
    else:
        print("✅ Success: Value function preserves strict concavity (V_WW < 0) across all points.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
