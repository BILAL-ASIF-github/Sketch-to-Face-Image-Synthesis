"""
src/models/baseline_mlp.py
Baseline MLP for Sketch-to-Face synthesis.

This model serves as a yardstick. It is intentionally simple: flatten
the input edge map, pass through fully-connected layers, and reshape
the output back to an image. The purpose is to establish a baseline
pixel-level reconstruction loss that all future architectures must beat.

Architecture Decisions:
- Weight init: Kaiming (He) uniform — chosen for ReLU activations.
  He init sets variance = 2/fan_in, preventing vanishing/exploding
  gradients with ReLU (which zeros out half of activations).
- Activation: LeakyReLU (alpha=0.2) — avoids dying ReLU problem where
  neurons can permanently stop updating when they produce zero output.
  Especially important for image pixel-space outputs.
- Output activation: Tanh — maps output to [-1, 1] to match the
  normalized target pixel range.
- Loss: L1 (Mean Absolute Error) — preferred over MSE for image
  synthesis because MSE penalizes large errors quadratically, leading
  to over-smoothed (blurry) outputs. L1 produces sharper results.
"""

import torch
import torch.nn as nn
from typing import List


class BaselineMLP(nn.Module):
    """
    Multi-layer perceptron baseline for image-to-image translation.

    Args:
        input_dim: Flattened input dimension (channels * H * W).
        output_dim: Flattened output dimension (channels * H * W).
        hidden_dims: List of hidden layer sizes.
        dropout_rate: Dropout probability (applied after each hidden layer).
        leaky_alpha: Negative slope for LeakyReLU.
    """

    def __init__(
        self,
        input_dim: int = 3 * 64 * 64,
        output_dim: int = 3 * 64 * 64,
        hidden_dims: List[int] = [2048, 1024, 512],
        dropout_rate: float = 0.3,
        leaky_alpha: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(leaky_alpha, inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] to match normalized targets

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply Kaiming (He) uniform initialization to all Linear layers.
        Kaiming is the recommended init for ReLU / LeakyReLU networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (B, input_dim) — already flattened.

        Returns:
            Output tensor, shape (B, output_dim).
        """
        return self.network(x)


def build_baseline_mlp(config: dict) -> BaselineMLP:
    """
    Factory function to build a BaselineMLP from a config dictionary.

    Args:
        config: Dictionary with keys: input_dim, output_dim,
                hidden_dims, dropout_rate, leaky_alpha.

    Returns:
        Instantiated BaselineMLP.
    """
    return BaselineMLP(
        input_dim=config.get("input_dim", 3 * 64 * 64),
        output_dim=config.get("output_dim", 3 * 64 * 64),
        hidden_dims=config.get("hidden_dims", [2048, 1024, 512]),
        dropout_rate=config.get("dropout_rate", 0.3),
        leaky_alpha=config.get("leaky_alpha", 0.2),
    )
