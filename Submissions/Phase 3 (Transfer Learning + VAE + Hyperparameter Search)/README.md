# 🎨 Sketch-to-Face Image Synthesis
### AI335L Deep Learning Lab — Phase 4 Refinement

> **pSp + ResNet-50 Transfer Learning | Convolutional VAE | Optuna HPO**

[![Phase](https://img.shields.io/badge/Phase-4%20Refinement-blue?style=flat-square)](https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis)
[![Tag](https://img.shields.io/badge/Tag-phase4--submission-green?style=flat-square)](https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis)
[![Course](https://img.shields.io/badge/Course-AI335L-orange?style=flat-square)](#)
[![License](https://img.shields.io/badge/Backbone%20License-BSD--3--Clause-lightgrey?style=flat-square)](#)

---

## 📌 Overview

This project builds a **sketch-to-face image synthesis pipeline** using the **pixel2Style2pixel (pSp)** architecture. Given a Canny edge map of a face sketch, the model synthesises a realistic photographic face.

Phase 4 introduces three major upgrades over Phase 3:
1. **Transfer Learning** — ResNet-50 pretrained on ImageNet-1K V2 replaces the from-scratch GradualStyleEncoder
2. **Convolutional VAE** — trained on CelebA for data augmentation and latent space analysis
3. **Hyperparameter Optimisation** — Bayesian search (Optuna TPE) over 6 key hyperparameters

**Result:** 25.4% improvement in validation L1 over Phase 3 (0.4764 → 0.3556)

---

## 👥 Authors

| Name | Role |
|---|---|
| Moazzam Sharif | — |
| Bilal Asif | — |
| Muaz Ahmad | — |

**Course:** AI335L — Deep Learning Lab

---

## 📊 Results Summary

| Model | Val L1 | vs Phase 3 |
|---|---|---|
| MLP Baseline | 0.4190 | — |
| Phase 3 pSp (scratch) | 0.4764 | baseline |
| Config A — Feature Extraction | 0.4119 | -13.5% |
| Config B — Full Fine-tuning | 0.3876 | -18.7% |
| Config C — Differential LR ⭐ | 0.3845 | -19.3% |
| Config D — Gradual Unfreeze | 0.3908 | -18.0% |
| **Phase 4 Final (ResNet-50 + HPO)** | **0.3556** | **-25.4%** |

---

## 🏗️ Architecture

### End-to-End Pipeline

```
Input: 256×256 Canny Edge Map (inverted, white background)
    ↓
Encoder: ResNet-50 (ImageNet pretrained)
    → stem + 4 residual stages
    → AdaptiveAvgPool
    → 3 projection heads
    → (B, 18, 512) style codes
    → F.normalize (unit sphere stabilisation)
    ↓
Decoder: StyleGAN2Decoder
    ↓
Output: (B, 3, 256, 256) Synthesised Face
```

### VAE Augmentation Branch

```
Training Face → ConvVAE Encoder → μ (512-dim)
                                → latent perturbation (σ=0.1)
                                → ConvVAE Decoder
                                → Augmented Training Face
```

---

## 🔁 Transfer Learning Study

Four configurations were compared using a ResNet-50 backbone (23.5M params, ImageNet-1K V2):

| Config | Strategy | Backbone LR | Val L1 | Val LPIPS |
|---|---|---|---|---|
| A | Feature Extraction (frozen) | 0 | 0.4119 | 0.6540 |
| B | Full Fine-tuning | 1e-5 | 0.3876 | 0.6156 |
| **C** ⭐ | **Differential LR** | **1e-5** | **0.3845** | **0.6100** |
| D | Gradual Unfreeze | 0 → 1e-5 | 0.3908 | 0.6258 |

**Best:** Config C with differential learning rates — backbone at `1e-5`, projection heads/decoder at `1e-3`.

**Key fix:** `F.normalize(styles, dim=-1)` applied before decoding to keep style codes on the unit sphere and prevent NaN losses from untrained projection heads.

---

## 🧬 Convolutional VAE

A ConvVAE (~51.8M params) was trained on CelebA face images for 8 epochs.

### Architecture
- **Encoder:** 5× strided `Conv2d` (stride=2), `BatchNorm + ReLU`, 256→8 spatial, 3→512 channels → FC heads for `μ` and `log σ²` (dim=512)
- **Reparameterisation:** `z = μ + ε·σ`, `ε ~ N(0, I)`
- **Decoder:** FC → 5× `ConvTranspose2d`, 8→256 spatial, 512→3 channels, `Tanh` output
- **Latent dim:** 512 (intentionally aligned with StyleGAN2 style code dimensionality)

### Loss

```
L = L_recon (L1) + β · L_KL
```

KL annealing: β linearly ramped from 0 → 1 over 3 warmup epochs to prevent posterior collapse.

### VAE Training Results

| Epoch | β | Val Total | Val KL |
|---|---|---|---|
| 1 | 0.33 | 0.2774 | 0.0986 |
| 3 | 1.00 | 0.2919 | 0.0416 |
| 8 | 1.00 | 0.2673 | 0.0413 |

Final: MSE `0.1099`, L1 `0.2434`, KL `0.0413` (no posterior collapse).

### Integration
- **Data augmentation:** 100+ augmented faces generated via latent perturbation (avg L1 vs original: 0.24)
- **Latent analysis:** Smooth face interpolations confirm a semantically continuous learned manifold

---

## ⚙️ Hyperparameter Optimisation

| Setting | Value |
|---|---|
| Library | Optuna 3.x |
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner (`n_startup=10`, `n_warmup=2`) |
| Storage | SQLite (`experiments/hpo/phase4_hpo.db`) |
| Objective | Minimise validation L1 |
| Epochs/trial | 3 |
| GPU | NVIDIA RTX 5880-Ada-12Q |

### Search Space

| Hyperparameter | Range | Importance |
|---|---|---|
| `dropout_p` | 0.0 – 0.5 | **0.4204** (most important) |
| `lr_backbone` | 1e-6 – 1e-4 | 0.1872 |
| `lambda_l1` | 0.5 – 2.0 | 0.1253 |
| `lr_head` | 1e-4 – 1e-2 | 0.1104 |
| `weight_decay` | 1e-5 – 1e-2 | 0.0918 |
| `lambda_lpips` | 0.2 – 1.5 | 0.0649 |

### Best Configuration (Trial 5, val L1 = 0.3626)

```yaml
lr_head:       1.307e-4
lr_backbone:   5.399e-5
weight_decay:  6.358e-4
dropout_p:     0.354
lambda_l1:     0.531
lambda_lpips:  1.461
```

Saved to `configs/phase4_best.yaml`.

---

## 🐛 Negative Results & Lessons Learned

| Issue | Root Cause | Fix |
|---|---|---|
| NaN losses from epoch 1 | Untrained projection heads → unconstrained style code scale | `F.normalize(styles, dim=-1)` before decoding |
| Pruner too aggressive (14/20 trials pruned) | `n_warmup_steps=1` — pruning after 1 epoch too early | Increased to `n_startup=10`, `n_warmup=2` |
| Corrupted Optuna DB | Stale variable reference crashed objective; SQLite file lock | Kernel restart; always test objective manually before full study |
| System shutdowns mid-run | Multi-hour training on consumer hardware | SQLite persistence + disable OS power management |
| LPIPS in training loop too slow | ~30-45 min/epoch at batch=8 | Moved to validation-only metric; sampled on 10 batches |

---

## 🔮 Phase 5 Plan

- **Full training** for 20–30 epochs with best HPO config
- **LPIPS in the training gradient** (not just reporting) to address blurriness
- **VAE-augmented training data** included end-to-end
- **New metrics:** SSIM, FID (1k generated vs 1k real), full-test LPIPS, per-identity consistency
- **Ablation studies:** F.normalize, VAE augmentation, L1 vs L1+LPIPS, Config C vs A at scale
- **Error analysis:** failure modes by sketch type, gradient maps, reconstruction by facial attribute

---

## 📁 Repository Structure

```
Sketch-to-Face-Image-Synthesis/
├── configs/
│   └── phase4_best.yaml          # Best HPO configuration
├── experiments/
│   └── hpo/
│       └── phase4_hpo.db         # Optuna SQLite study
├── models/
│   ├── psp_resnet.py             # pSpResNet (ResNet-50 encoder + StyleGAN2 decoder)
│   ├── stylegan2_decoder.py      # StyleGAN2 decoder
│   └── conv_vae.py               # Convolutional VAE
├── train_tl.py                   # Transfer learning training script
├── train_vae.py                  # VAE training script
├── hpo_search.py                 # Optuna HPO study
├── evaluate.py                   # Evaluation & metrics
├── augment_vae.py                # VAE-based augmentation
└── reports/
    └── phase4-submission.pdf     # This report
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis.git
cd Sketch-to-Face-Image-Synthesis

# Install dependencies
pip install torch torchvision optuna lpips

# Train with best Phase 4 config
python train_tl.py --config configs/phase4_best.yaml

# Run HPO search
python hpo_search.py --n-trials 20

# Train VAE
python train_vae.py --epochs 8 --latent-dim 512
```

---

## 📜 Citation & License

The ResNet-50 backbone is sourced from `torchvision.models`, pretrained on ImageNet-1K V2, under the **BSD-3-Clause** license.

---

*AI335L Deep Learning Lab — Phase 4 Submission | May 2026*
