"""
src/preprocessing/pipeline.py
Modular preprocessing pipeline for Sketch-to-Face synthesis.

Follows sklearn-style fit/transform API.
IMPORTANT: Call .fit() on training data ONLY, then .transform() on val and test
to prevent data leakage.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


class EdgeMapGenerator:
    """
    Converts CelebA face photos to Canny-style edge maps.
    This is the core preprocessing step that creates the 'sketch' inputs.

    fit() computes nothing (edge maps are deterministic), but is included
    for API consistency.

    Args:
        img_size: Output image size in pixels (square).
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.
        dilate_iters: Number of dilation iterations to thicken edges.
    """

    def __init__(
        self,
        img_size: int = 256,
        canny_low: int = 50,
        canny_high: int = 150,
        dilate_iters: int = 1,
    ) -> None:
        self.img_size = img_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.dilate_iters = dilate_iters
        self._fitted = False

    def fit(self, image_paths: List[Path]) -> "EdgeMapGenerator":
        """
        No statistics to compute for edge map generation.
        Sets fitted flag for API consistency.

        Args:
            image_paths: List of paths to training images (unused).

        Returns:
            self
        """
        self._fitted = True
        return self

    def transform(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a single image to a (resized_photo, edge_map) pair.

        Args:
            img_path: Path to the input face image.

        Returns:
            Tuple of (photo_rgb, edge_map_inv) both as uint8 numpy arrays.
            edge_map_inv has white background and black lines (inverted Canny).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Bilateral filter: preserves edges while reducing noise
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=self.dilate_iters)

        # Invert: white background, black lines (matches hand-drawn sketches)
        edges_inv = cv2.bitwise_not(edges)

        photo_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return photo_rgb, edges_inv

    def fit_transform(
        self, image_paths: List[Path]
    ) -> "EdgeMapGenerator":
        """Convenience wrapper: fit then return self for chaining."""
        return self.fit(image_paths)


class PairDatasetBuilder:
    """
    High-level pipeline that reads CelebA images and writes
    paired (edge_map, photo) PNG files to output directories.

    Args:
        generator: A fitted EdgeMapGenerator instance.
        output_dir: Root directory for output pairs.
    """

    def __init__(self, generator: EdgeMapGenerator, output_dir: str) -> None:
        self.generator = generator
        self.output_dir = Path(output_dir)

    def build(
        self,
        image_paths: List[Path],
        split: str,
        skip_existing: bool = True,
    ) -> int:
        """
        Process a list of image paths and write edge/photo PNG pairs.

        Args:
            image_paths: List of source image paths.
            split: One of 'train', 'val', 'test'.
            skip_existing: Skip if output already exists.

        Returns:
            Number of pairs written.
        """
        edges_dir = self.output_dir / split / "edges"
        photos_dir = self.output_dir / split / "photos"
        edges_dir.mkdir(parents=True, exist_ok=True)
        photos_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for i, path in enumerate(image_paths):
            edge_out = edges_dir / f"{i:05d}.png"
            photo_out = photos_dir / f"{i:05d}.png"

            if skip_existing and edge_out.exists() and photo_out.exists():
                continue

            photo, edge = self.generator.transform(path)
            cv2.imwrite(str(edge_out), edge)
            cv2.imwrite(str(photo_out), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
            n_written += 1

        return n_written


class PixelNormalizer:
    """
    Normalizes pixel values to [-1, 1] using training set mean/std.

    fit() computes mean and std from training data (RGB images).
    transform() applies normalization.

    IMPORTANT: Only ever call fit() on training images. This prevents
    data leakage from val/test sets into the normalization statistics.

    Args:
        channel_wise: If True, compute separate mean/std per channel.
    """

    def __init__(self, channel_wise: bool = True) -> None:
        self.channel_wise = channel_wise
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, image_arrays: List[np.ndarray]) -> "PixelNormalizer":
        """
        Compute mean and std from training images.

        Args:
            image_arrays: List of (H, W, C) uint8 numpy arrays.

        Returns:
            self
        """
        stack = np.stack(
            [img.astype(np.float32) / 255.0 for img in image_arrays], axis=0
        )  # (N, H, W, C)

        if self.channel_wise:
            self.mean_ = stack.mean(axis=(0, 1, 2))   # (C,)
            self.std_ = stack.std(axis=(0, 1, 2)) + 1e-8
        else:
            self.mean_ = np.array([stack.mean()] * 3)
            self.std_ = np.array([stack.std() + 1e-8] * 3)

        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize a single image to [-1, 1].

        Args:
            image: (H, W, C) uint8 numpy array.

        Returns:
            Normalized float32 array in [-1, 1].
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform()")
        x = image.astype(np.float32) / 255.0
        x = (x - self.mean_) / self.std_
        return x.astype(np.float32)

    def inverse_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Undo normalization, returning uint8 image.

        Args:
            image: Normalized float32 array.

        Returns:
            uint8 numpy array in [0, 255].
        """
        x = image * self.std_ + self.mean_
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return x
