from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from collocation_points import collocation_points
from loss_function import loss_function
from neural_network import ValueFunc, Value_func  # Value_func for backward compatibility
from terminal_condition import terminal_points

#Configuration 
# Resolve config path relative to this file
_CONFIG_PATH = Path(__file__).resolve().parent / "data_accumulation" / "config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

# Model save directory
_SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
_SAVE_DIR.mkdir(parents=True, exist_ok=True)


#Training Loop 

def training_loop(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    loss_fn: callable,
    epochs: int,
    sigma: float,
    rate: float,
    mu: float,
    t: torch.Tensor,
    w: torch.Tensor,
    t_terminal: torch.Tensor,
    w_terminal: torch.Tensor,
    gamma: float,
    weight_pde: float = 1.0,
    weight_terminal: float = 1.0,
    save_best: bool = True,
    save_final: bool = True,
    eval_interval: int = 50,
    print_interval: int = 10,
) -> dict:
    model.train()
    best_loss = float("inf")
    best_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Compute training loss
        train_loss = loss_fn(
            model=model,
            t=t,
            w=w,
            t_terminal=t_terminal,
            w_terminal=w_terminal,
            gamma=gamma,
            rate=rate,
            mu=mu,
            sigma=sigma,
            weight_pde=weight_pde,
            weight_terminal=weight_terminal,
        )
        # Backpropagation
        train_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Record training loss
        current_loss = train_loss.item()
        history["epoch"].append(epoch)
        history["train_loss"].append(current_loss)
        # Compute validation loss periodically
        val_loss = None
        if eval_interval > 0 and epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    model=model,
                    t=t,
                    w=w,
                    t_terminal=t_terminal,
                    w_terminal=w_terminal,
                    gamma=gamma,
                    rate=rate,
                    mu=mu,
                    sigma=sigma,
                    weight_pde=weight_pde,
                    weight_terminal=weight_terminal,
                ).item()
            history["val_loss"].append(val_loss)
            model.train()

        # Save best model
        if save_best and current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "loss": current_loss,
                    "config": config,
                },
                _SAVE_DIR / "best_model.pt",
            )

        # Print progress
        if epoch % print_interval == 0:
            elapsed = time.time() - start_time
            val_str = f", Val Loss: {val_loss}" if val_loss is not None else ""
            print(
                f"Epoch: {epoch:4d} | "
                f"Train Loss: {current_loss}{val_str} | "
                f"Best Loss: {best_loss} (epoch {best_epoch}) | "
                f"LR: {optimizer.param_groups[0]['lr']} | "
                f"Time: {elapsed:.1f}s"
            )

    # Save final model
    if save_final:
        torch.save(
            {
                "epoch": epochs - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "loss": current_loss,
                "config": config,
                "history": history,
            },
            _SAVE_DIR / "final_model.pt",
        )
        print(f"\nFinal model saved to: {_SAVE_DIR / 'final_model.pt'}")

    if save_best:
        print(f"Best model saved to: {_SAVE_DIR / 'best_model.pt'} (loss={best_loss} at epoch {best_epoch})")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return history


# ─── Load Model Utility ──────────────────────────────────────────────────────

def load_model(checkpoint_path: str,model: Optional[torch.nn.Module] = None,device: str = "cpu",) -> tuple[torch.nn.Module, dict]:
    if model is None:
        model = ValueFunc()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
    return model, checkpoint


# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    """Main training script with all hyperparameters configurable."""
    # ── Hyperparameters ──
    # Collocation points
    NUM_COLLOCATION_PTS = 10000
    NUM_TERMINAL_PTS = 2000

    # Domain parameters
    T = 1.0  # Terminal time (normalized to 1 year)
    W_MIN = 0.1
    W_MAX = 2.0

    # Training parameters
    EPOCHS = 1000  # Increased from 50 for better convergence
    LEARNING_RATE = 0.001
    STEP_SIZE = 200  # Scheduler step size
    GAMMA_SCHED = 0.9  # Scheduler gamma (learning rate decay)
    BATCH_SIZE = None  # Full batch training (PINN typically uses full batch)

    # Loss weights
    WEIGHT_PDE = 1.0
    WEIGHT_TERMINAL = 1.0

    # ── Extract parameters from config ──
    sigma = config["sigma"]
    rate = config["rate"]
    mu = config["mu"]
    gamma = config["gamma"]

    print("=" * 70)
    print("Merton Problem PINN Training")
    print("=" * 70)
    print(f"Parameters: sigma={sigma}, rate={rate}, mu={mu}, gamma={gamma}")
    print(f"Domain: T={T}, w∈[{W_MIN}, {W_MAX}]")
    print(f"Collocation points: {NUM_COLLOCATION_PTS}, Terminal points: {NUM_TERMINAL_PTS}")
    print(f"Training: {EPOCHS} epochs, LR={LEARNING_RATE}")
    print("=" * 70)

    # ── Generate collocation points ──
    # collocation_points returns (t, w) tensors
    t_colloc, w_colloc = collocation_points(NUM_COLLOCATION_PTS, T, W_MAX, W_MIN)
    t_colloc = t_colloc.reshape(-1, 1)
    w_colloc = w_colloc.reshape(-1, 1)

    # ── Generate terminal condition points ──
    # terminal_points now returns (t_terminal, w_terminal, v_terminal)
    t_terminal, w_terminal, v_terminal = terminal_points(NUM_TERMINAL_PTS, T, W_MAX, W_MIN, gamma)

    print(f"Collocation points shape: t={t_colloc.shape}, w={w_colloc.shape}")
    print(f"Terminal points shape: t={t_terminal.shape}, w={w_terminal.shape}")

    # ── Initialize model ──
    model = ValueFunc()
    print(f"\nModel architecture: {model.network}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer and scheduler ──
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA_SCHED)

    # ── Train ──
    print("\nStarting training")

    history = training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_function,
        epochs=EPOCHS,
        sigma=sigma,
        rate=rate,
        mu=mu,
        t=t_colloc,
        w=w_colloc,
        t_terminal=t_terminal,
        w_terminal=w_terminal,
        gamma=gamma,
        weight_pde=WEIGHT_PDE,
        weight_terminal=WEIGHT_TERMINAL,
        save_best=True,
        save_final=True,
        eval_interval=50,
        print_interval=100,
    )
    print(f"Training complete!")
    print(f"Final loss: {history['train_loss'][-1]}")
    print(f"Best loss: {min(history['train_loss'])}")
    return model, history


if __name__ == "__main__":
    model, history = main()