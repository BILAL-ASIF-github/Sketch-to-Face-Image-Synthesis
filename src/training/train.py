"""
src/training/train.py
Training loop for the baseline MLP model.

Implements:
  - Forward pass
  - Loss computation (L1)
  - Backward pass + optimizer step
  - Gradient clipping
  - Validation evaluation per epoch
  - Early stopping on validation loss
  - Checkpoint saving (best model on val loss)
  - Final test evaluation (called exactly once)

Usage:
    python src/training/train.py --config experiments/baseline_config.yaml
"""

import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.seed import seed_everything
from src.utils.logger import ExperimentLogger
from src.data.dataset import FlatEdgeToFaceDataset
from src.models.baseline_mlp import build_baseline_mlp


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
) -> None:
    """Save model + optimizer state to disk."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """Load model + optimizer state from a checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, ckpt["epoch"], ckpt["val_loss"]


# ── Training & evaluation passes ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    Run one full training epoch.

    Args:
        model: The neural network.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Target device.
        grad_clip: Max gradient norm for clipping (0 = disabled).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for edges, photos in loader:
        edges = edges.to(device)
        photos = photos.to(device)

        optimizer.zero_grad()

        outputs = model(edges)         # (B, output_dim)
        loss = criterion(outputs, photos)

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate model on a DataLoader (validation or test).

    Args:
        model: The neural network.
        loader: DataLoader to evaluate on.
        criterion: Loss function.
        device: Target device.

    Returns:
        Average loss over the full loader.
    """
    model.eval()
    total_loss = 0.0

    for edges, photos in loader:
        edges = edges.to(device)
        photos = photos.to(device)
        outputs = model(edges)
        loss = criterion(outputs, photos)
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Main training entry point ──────────────────────────────────────────────────

def train(config: Dict) -> None:
    """
    Full training pipeline: data → model → train → checkpoint → test.

    Args:
        config: Configuration dictionary.
    """
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Logging
    logger = ExperimentLogger(
        log_dir=config.get("log_dir", "experiments/logs"),
        run_name=config.get("run_name", "baseline_mlp"),
    )
    logger.log_hparams(config)

    # Datasets & loaders
    img_size = config.get("img_size", 64)
    train_ds = FlatEdgeToFaceDataset(
        config["train_edges"], config["train_photos"], img_size=img_size, augment=True
    )
    val_ds = FlatEdgeToFaceDataset(
        config["val_edges"], config["val_photos"], img_size=img_size, augment=False
    )
    test_ds = FlatEdgeToFaceDataset(
        config["test_edges"], config["test_photos"], img_size=img_size, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.get("batch_size", 64),
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.get("batch_size", 64),
        shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.get("batch_size", 64),
        shuffle=False, num_workers=4
    )

    print(f"[Train] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    dim = 3 * img_size * img_size
    model_config = config.get("model", {})
    model_config.setdefault("input_dim", dim)
    model_config.setdefault("output_dim", dim)
    model = build_baseline_mlp(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Parameters: {total_params:,}")

    # Loss — L1 (MAE) is preferred for image synthesis to avoid blurry outputs
    criterion = nn.L1Loss()

    # Optimizer — Adam with default betas
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        betas=(0.9, 0.999),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # LR scheduler — ReduceLROnPlateau monitors val loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    # Early stopping state
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = config.get("early_stopping_patience", 10)
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_baseline_mlp.pt"

    num_epochs = config.get("num_epochs", 50)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.get("grad_clip", 1.0),
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        logger.log_epoch(epoch, {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": current_lr,
        })

        scheduler.step(val_loss)

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(str(best_ckpt_path), model, optimizer, epoch, val_loss)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"[Train] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── Final test evaluation (done exactly once) ────────────────────────────
    print("\n[Train] Loading best checkpoint for final test evaluation...")
    model, _, best_epoch, _ = load_checkpoint(
        str(best_ckpt_path), model, device=str(device)
    )
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"[Train] Final Test L1 Loss: {test_loss:.4f} (from epoch {best_epoch})")

    logger.log_test_metrics({"test/l1_loss": test_loss})
    logger.close()
    print("[Train] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline MLP")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/baseline_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
