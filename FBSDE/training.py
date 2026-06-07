import json
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from forward_simulation_equations import forward_simulation
from backward_simulation_equation import backward_simulation
from neural_networks import pi_star, Z_net
from loss_function import loss_function
from utility_function import utility_function


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_parent_dir_for_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def analytical_pi(r: float, mu: float, sigma: float, gamma: float) -> float:
    """Merton's analytical optimal fraction."""
    return (mu - r) / (gamma * sigma**2)


def sample_pi_supervision_points(
    num_points: int,
    T: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random (t, W) pairs for pi* supervision."""
    t_vals = torch.rand(num_points, 1, device=device) * T
    # Brownian motion at time t has std = sqrt(t)
    std = t_vals.sqrt()
    W_vals = torch.randn(num_points, 1, device=device) * std
    return t_vals, W_vals


def training_loop(
    epochs: int,
    model: torch.nn.Module,
    Z_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    mu: float,
    sigma: float,
    rate: float,
    w0: float,
    T: float,
    num_steps: int,
    num_paths: int,
    gamma: float,
    device: torch.device | None = None,
    # Model saving options
    save_best: bool = True,
    final_path: str = r"C:/Users/Admin/Desktop/FBSDE PROJECT/models_and_experiments/final.pth",
    best_path: str = r"C:/Users/Admin/Desktop/FBSDE PROJECT/models_and_experiments/best.pth",
    show_live_plot: bool = True,
    # Supervision hyperparameters
    pi_supervision_lambda: float = 1.0,
    pi_supervision_points: int = 512,
):
    device = device or torch.device("cpu")
    model.to(device)
    Z_model.to(device)

    if device.type == "cuda":
        # Speed/behavior tweaks for stable training on NVIDIA GPUs
        torch.backends.cudnn.benchmark = True

    best_loss = float("inf")
    best_state = None
    history: list[float] = []

    # Pre-compute analytical pi* target (constant for Merton)
    pi_target = analytical_pi(r=rate, mu=mu, sigma=sigma, gamma=gamma)
    pi_target_t = torch.tensor(pi_target, dtype=torch.float32, device=device)
    print(f"Analytical pi* target: {pi_target:.6f}")

    # Live updating plot setup
    if show_live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Epoch (live)")
        ax.grid(True)

        # Make sure window appears promptly
        plt.show(block=False)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        time_grid, wealth_grid, dw, W = forward_simulation(
            model=model,
            mu=mu,
            sigma=sigma,
            w0=w0,
            rate=rate,
            T=T,
            num_steps=num_steps,
            num_paths=num_paths,
        )

        # Ensure tensors on device for autograd
        wealth_grid = wealth_grid.to(device)
        dw = dw.to(device)
        W = W.to(device)

        Y = backward_simulation(
            wealth_grid=wealth_grid,
            time_grid=time_grid,
            dw=dw,
            W=W,
            Z_model=Z_model,
            gamma=gamma,
            r=rate,
            mu=mu,
            sigma=sigma,
            utility_fn=utility_function,
        )

        # BSDE loss: variance of initial value function
        bsde_loss = loss_function(Y)

        # --- Direct pi* supervision loss ---
        # Sample random (t, W) pairs and penalize deviation from analytical pi*
        t_sup, W_sup = sample_pi_supervision_points(
            num_points=pi_supervision_points, T=T, device=device
        )
        pi_pred = model(t_sup, W_sup)
        pi_sup_loss = F.mse_loss(pi_pred, pi_target_t.expand_as(pi_pred))

        # Combined loss
        loss = bsde_loss + pi_supervision_lambda * pi_sup_loss
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history.append(loss_val)

        # Track best model (using total loss)
        if save_best and loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                "pi_star": model.state_dict(),
                "Z_net": Z_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": loss_val,
                "bsde_loss": bsde_loss.item(),
                "pi_sup_loss": pi_sup_loss.item(),
            }

        # Save checkpoint every 100 epochs
        if save_best and (epoch + 1) % 100 == 0:
            _ensure_parent_dir_for_file(best_path)
            torch.save(
                {
                    "pi_star": model.state_dict(),
                    "Z_net": Z_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "loss": loss_val,
                    "bsde_loss": bsde_loss.item(),
                    "pi_sup_loss": pi_sup_loss.item(),
                    "config": {
                        "mu": mu,
                        "sigma": sigma,
                        "rate": rate,
                        "w0": w0,
                        "T": T,
                        "num_steps": num_steps,
                        "num_paths": num_paths,
                        "gamma": gamma,
                    },
                },
                best_path,
            )
            print(f"  [checkpoint saved at epoch {epoch + 1}]")

        # Print every epoch with component breakdown
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} | total: {loss_val:.6e} | bsde: {bsde_loss.item():.6e} | pi_sup: {pi_sup_loss.item():.6e} | pi_pred_mean: {pi_pred.mean().item():.4f}")

        # Update live plot
        if show_live_plot:
            xs = list(range(1, len(history) + 1))
            line.set_data(xs, history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    # Save final model
    _ensure_parent_dir_for_file(final_path)
    torch.save(
        {
            "pi_star": model.state_dict(),
            "Z_net": Z_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "loss": history[-1] if history else None,
            "config": {
                "mu": mu,
                "sigma": sigma,
                "rate": rate,
                "w0": w0,
                "T": T,
                "num_steps": num_steps,
                "num_paths": num_paths,
                "gamma": gamma,
            },
        },
        final_path,
    )
    print(f"Final model saved to {final_path}")

    # Save best model
    if save_best and best_state is not None:
        _ensure_parent_dir_for_file(best_path)
        best_state["config"] = {
            "mu": mu,
            "sigma": sigma,
            "rate": rate,
            "w0": w0,
            "T": T,
            "num_steps": num_steps,
            "num_paths": num_paths,
            "gamma": gamma,
        }
        torch.save(best_state, best_path)
        print(f"Best model (loss={best_loss}) saved to {best_path}")

    if show_live_plot:
        plt.ioff()

    return history


if __name__ == "__main__":
    torch.manual_seed(0)

    # Load config
    cfg = load_config()
    mu = cfg["mu"]
    sigma = cfg["sigma"]
    rate = cfg["rate"]
    w0 = cfg["w0"]
    T = cfg["T"]
    num_steps = int(cfg["num_steps"])
    num_paths = int(cfg["num_paths"])
    gamma = cfg["gamma"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = pi_star().to(device)
    Z_model = Z_net().to(device)

    # With 200 steps, use slightly lower LR for stability
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(Z_model.parameters()), lr=1e-4
    )

    training_loop(
        epochs=5000,
        model=model,
        Z_model=Z_model,
        optimizer=optimizer,
        mu=mu,
        sigma=sigma,
        rate=rate,
        w0=w0,
        T=T,
        num_steps=num_steps,
        num_paths=num_paths,
        gamma=gamma,
        device=device,
        pi_supervision_lambda=1.0,
        pi_supervision_points=512,
    )