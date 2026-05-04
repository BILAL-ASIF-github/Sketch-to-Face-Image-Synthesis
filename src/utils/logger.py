"""
utils/logger.py
TensorBoard experiment logging utility.
"""

import os
import csv
import time
from pathlib import Path
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentLogger:
    """
    Logs hyperparameters, per-epoch metrics, and final test results.
    Writes to both TensorBoard and a CSV fallback.

    Args:
        log_dir: Directory to store logs.
        run_name: Optional name for this run. Auto-generated if None.
    """

    def __init__(self, log_dir: str = "experiments/logs", run_name: Optional[str] = None) -> None:
        self.run_name = run_name or f"run_{int(time.time())}"
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_initialized = False

        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"[Logger] TensorBoard logging to: {self.log_dir}")
            print(f"[Logger] Run: tensorboard --logdir {Path(log_dir)}")
        else:
            self.writer = None
            print(f"[Logger] TensorBoard not available. CSV-only logging to {self.csv_path}")

    def log_hparams(self, hparams: Dict) -> None:
        """Log a dictionary of hyperparameters."""
        hparam_path = self.log_dir / "hparams.csv"
        with open(hparam_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["param", "value"])
            for k, v in hparams.items():
                writer.writerow([k, v])
        if self.writer:
            self.writer.add_text("hyperparameters", str(hparams), 0)
        print(f"[Logger] Hyperparameters saved to {hparam_path}")

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log per-epoch metrics (train_loss, val_loss, lr, etc.).

        Args:
            epoch: Current epoch number (1-indexed).
            metrics: Dict mapping metric name to float value.
        """
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

        # CSV
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics.keys()))
                writer.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics.keys()))
            writer.writerow({"epoch": epoch, **metrics})

    def log_test_metrics(self, metrics: Dict[str, float]) -> None:
        """Log final test-set metrics (called exactly once)."""
        test_path = self.log_dir / "test_metrics.csv"
        with open(test_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"test/{key}", value, 0)
        print(f"[Logger] Test metrics saved to {test_path}")

    def close(self) -> None:
        """Flush and close TensorBoard writer."""
        if self.writer:
            self.writer.flush()
            self.writer.close()
