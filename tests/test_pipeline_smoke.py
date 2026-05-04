"""
tests/test_pipeline_smoke.py
Smoke tests: run the full pipeline on tiny synthetic data and assert
correct shapes, types, and that no exceptions are raised.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline_mlp import BaselineMLP, build_baseline_mlp
from src.preprocessing.pipeline import EdgeMapGenerator, PixelNormalizer


# ── EdgeMapGenerator ──────────────────────────────────────────────────────────

class TestEdgeMapGenerator:
    def test_fit_returns_self(self) -> None:
        gen = EdgeMapGenerator(img_size=64)
        result = gen.fit([])
        assert result is gen

    def test_transform_requires_fit(self, tmp_path) -> None:
        gen = EdgeMapGenerator(img_size=64)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), dummy_img)
        with pytest.raises(RuntimeError, match="fit"):
            gen.transform(str(img_path))

    def test_output_shapes(self, tmp_path) -> None:
        import cv2
        gen = EdgeMapGenerator(img_size=64)
        gen.fit([])
        dummy = np.ones((100, 100, 3), dtype=np.uint8) * 128
        path = tmp_path / "face.jpg"
        cv2.imwrite(str(path), dummy)
        photo, edge = gen.transform(str(path))
        assert photo.shape == (64, 64, 3), f"Expected (64,64,3), got {photo.shape}"
        assert edge.shape == (64, 64), f"Expected (64,64), got {edge.shape}"
        assert photo.dtype == np.uint8
        assert edge.dtype == np.uint8


# ── PixelNormalizer ───────────────────────────────────────────────────────────

class TestPixelNormalizer:
    def _make_images(self, n: int = 10) -> list:
        return [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(n)]

    def test_fit_sets_stats(self) -> None:
        norm = PixelNormalizer()
        images = self._make_images()
        norm.fit(images)
        assert norm.mean_ is not None
        assert norm.std_ is not None
        assert norm.mean_.shape == (3,)

    def test_transform_range(self) -> None:
        norm = PixelNormalizer()
        images = self._make_images(50)
        norm.fit(images)
        out = norm.transform(images[0])
        assert out.dtype == np.float32
        # After z-score normalization most values should be within [-5, 5]
        assert out.min() > -10 and out.max() < 10

    def test_no_leakage_val_uses_train_stats(self) -> None:
        """Val/test must be transformed with training stats, not their own."""
        train_imgs = self._make_images(20)
        val_imgs = self._make_images(5)
        norm = PixelNormalizer()
        norm.fit(train_imgs)
        train_mean = norm.mean_.copy()
        # Fitting again on val should give different stats
        norm2 = PixelNormalizer()
        norm2.fit(val_imgs)
        assert not np.allclose(train_mean, norm2.mean_), \
            "Train and val stats should differ (different random images)"
        # But we use train stats on val — no re-fit
        out = norm.transform(val_imgs[0])
        assert out.dtype == np.float32


# ── BaselineMLP ───────────────────────────────────────────────────────────────

class TestBaselineMLP:
    def test_output_shape(self) -> None:
        model = BaselineMLP(input_dim=128, output_dim=128, hidden_dims=[64, 32])
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4, 128), f"Expected (4,128), got {out.shape}"

    def test_output_range(self) -> None:
        model = BaselineMLP(input_dim=64, output_dim=64, hidden_dims=[32])
        x = torch.randn(8, 64)
        out = model(x)
        # Tanh output must be in (-1, 1)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_build_from_config(self) -> None:
        config = {
            "input_dim": 256,
            "output_dim": 256,
            "hidden_dims": [128, 64],
            "dropout_rate": 0.2,
            "leaky_alpha": 0.1,
        }
        model = build_baseline_mlp(config)
        assert isinstance(model, BaselineMLP)
        x = torch.randn(2, 256)
        out = model(x)
        assert out.shape == (2, 256)

    def test_kaiming_init(self) -> None:
        """Weights should not all be zero after initialization."""
        model = BaselineMLP(input_dim=64, output_dim=64, hidden_dims=[32])
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                assert layer.weight.abs().sum().item() > 0
                assert layer.bias.abs().sum().item() == 0  # biases init to zero


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
