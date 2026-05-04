"""
utils/seed.py
Reproducibility utility — call seed_everything() at the top of every training script.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU + GPU) to ensure reproducibility.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed_everything] All seeds set to {seed}")
