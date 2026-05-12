# 🎨 Sketch-to-Face Image Synthesis

> **AI335L — Deep Learning Lab · Phase 3 of 6**
> Paired image-to-image translation using **pSp (pixel2style2pixel)** — converting edge map sketches into photorealistic face images.

---

## 📋 Project Overview

| Field | Details |
|-------|---------|
| **Course** | AI335L — Deep Learning Lab |
| **Instructor** | Lecturer Haseeb |
| **Institute** | NASTP Institute of Information Technology |
| **Semester** | V — Spring 2024 |
| **Track** | CNN · pSp (pixel2style2pixel) |
| **Dataset** | CelebA-HQ (19,083 paired images used; 202,599 available) |
| **Phase** | 3 of 6 |

---

## 🏗️ Architecture

The model consists of two components:

```
Sketch  (B, 3, 256, 256)
  │
┌─────────────────────────────────────┐
│  GradualStyleEncoder                │
│  stem : Conv2d 3→64  256×256        │
│  stage1: ResBlock×2  128ch 128×128  │  ← residual connections
│  stage2: ResBlock×2  256ch  64×64   │  ← residual connections
│  stage3: ResBlock×2  512ch  32×32   │  ← residual connections
│  stage4: ResBlock×2  512ch  16×16   │  ← residual connections
│  map_fine   (AvgPool) → 8×512       │  ← pooling
│  map_medium (AvgPool) → 6×512       │  ← pooling
│  map_coarse (AvgPool) → 4×512       │  ← pooling
└──────────────┬──────────────────────┘
               │  styles  (B, 18, 512)
┌──────────────┴──────────────────────┐
│  StyleGAN2 Decoder                  │
│  const  512×4×4                     │
│  AdaIN block:   4→  8  (Conv+Norm)  │
│  AdaIN block:   8→ 16  (Conv+Norm)  │
│  AdaIN block:  16→ 32  (Conv+Norm)  │
│  AdaIN block:  32→ 64  (Conv+Norm)  │
│  AdaIN block:  64→128  (Conv+Norm)  │
│  AdaIN block: 128→256  (Conv+Norm)  │
│  to_rgb : Conv2d → Tanh             │
└──────────────┬──────────────────────┘
               │
Face  (B, 3, 256, 256)  range [−1, 1]
```

**Total trainable parameters: 39,784,675**

---

## 📁 Repository Structure

```
Sketch-to-Face-Image-Synthesis/
│
├── configs/
│   ├── psp_v1.yaml               # Base training config
│   ├── psp_unregularized.yaml    # Ablation: no regularization
│   ├── psp_regularized.yaml      # Ablation: dropout + weight decay
│   └── psp_normalized.yaml       # Ablation: + BatchNorm
│
├── data/
│   └── celeba_hq/
│       ├── images/               # Raw face images (place here)
│       └── sketches/             # Auto-generated Canny edge maps
│
├── src/
│   ├── models/
│   │   └── psp_model.py          # GradualStyleEncoder + StyleGAN2Decoder
│   ├── training/
│   │   └── train_psp.py          # Full training loop (AMP, grad clip, logging)
│   └── data/
│       ├── dataset.py            # EdgeToFaceDataset
│       └── generate_sketches.py  # Canny edge map generation
│
├── experiments/
│   ├── checkpoints/
│   │   └── psp_v1/
│   │       ├── best.pt           # Best model checkpoint
│   │       └── latest.pt         # Latest checkpoint (resumable)
│   └── logs/
│       ├── psp_v1_base/
│       ├── ablation_unregularized/
│       ├── ablation_regularized/
│       └── ablation_normalized/
│
├── reports/
│   ├── training_curves.png
│   ├── ablation_curves.png
│   ├── baseline_comparison.png
│   ├── qualitative_samples.png
│   ├── first_layer_filters.png
│   ├── feature_maps_encoder_stage1.png
│   ├── feature_maps_encoder_stage2.png
│   ├── feature_maps_encoder_stage3.png
│   └── gradcam.png
│
├── phase3_experiments.ipynb      # Main experiment notebook
├── environment.yml               # Conda environment spec
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & checkout

```bash
git clone https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis.git
cd Sketch-to-Face-Image-Synthesis
git checkout phase3-submission
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate sketch2face
```

### 3. Prepare the dataset

Place CelebA-HQ images in `data/celeba_hq/images/`, then generate sketches:

```bash
python src/data/generate_sketches.py --data_root data/celeba_hq
```

### 4. Run training

```bash
# Base run
python src/training/train_psp.py --config configs/psp_v1.yaml

# Ablations
python src/training/train_psp.py --config configs/psp_unregularized.yaml
python src/training/train_psp.py --config configs/psp_regularized.yaml
python src/training/train_psp.py --config configs/psp_normalized.yaml
```

### 5. Evaluate & visualize

Open `phase3_experiments.ipynb` in Jupyter and run all cells. Outputs are saved to `reports/`.

---

## ⚙️ Configuration

Key parameters from `configs/psp_v1.yaml`:

```yaml
# Model
img_size: 256
use_dropout: true
dropout_p: 0.3
use_batchnorm: false

# Training
num_epochs: 5
batch_size: 16
lr: 0.0001
lr_min: 1.0e-06
weight_decay: 0.0001
grad_clip: 1.0
patience: 3                  # early stopping
seed: 42

# Loss
lambda_l1: 1.0
lambda_lpips: 0.8            # set 0.0 for ablations

# Data
data_root: data/celeba_hq
val_split: 0.1
test_split: 0.1
augment: true

# Paths
checkpoint_dir: experiments/checkpoints/psp_v1
log_dir: experiments/logs
run_name: psp_v1_base
```

---

## 📊 Results

### Ablation Study

| Config | Dropout | Wt. Decay | Augment | BatchNorm | Best Val L1 | Test L1 | Epochs |
|--------|---------|-----------|---------|-----------|-------------|---------|--------|
| Unregularized | ✗ | 0.0 | ✗ | ✗ | 0.4359 | 0.4297 | 3 |
| Regularized | ✓ (p=0.3) | 1e-4 | ✓ | ✗ | 0.4638 | 0.4649 | 2 |
| Normalized | ✓ (p=0.3) | 1e-4 | ✓ | ✓ | 0.4554 | 0.4560 | 2 |

### Baseline Comparison

| Model | Test L1 (mean) | Test L1 (std) | Parameters | Image Size |
|-------|---------------|---------------|------------|------------|
| MLP (Phase 2 Baseline) | 0.4190 | ±0.0000 | ~34.1 M | 64×64 |
| **pSp (Phase 3 — 3-seed)** | **0.4764** | **±0.0017** | **39,784,675** | **256×256** |

> **Note:** L1 values are not directly comparable — pSp reconstructs 16× more pixels per image (256×256 vs 64×64). Full training (30–50 epochs) is expected to bring pSp L1 below 0.40.

### Hardware

```
GPU   : NVIDIA RTX 5880 Ada-12Q
VRAM  : 11.5 GB
RAM   : 32 GB
Time  : ~30 min base run (5 epochs, batch=16)
```

---

## 🔍 Layer Details

| Layer / Block | Module | Output Shape | Parameters | Notes |
|---------------|--------|-------------|------------|-------|
| Input | Sketch image | 3 × 256 × 256 | 0 | — |
| Stem Conv2d | GradualStyleEncoder | 64 × 256 × 256 | 1,792 | InstanceNorm + PReLU |
| Stage 1 (2× ResBlock) | Encoder | 128 × 128 × 128 | 295,424 | Stride-2 downsample |
| Stage 2 (2× ResBlock) | Encoder | 256 × 64 × 64 | 1,180,672 | Stride-2 downsample |
| Stage 3 (2× ResBlock) | Encoder | 512 × 32 × 32 | 4,720,640 | Stride-2 downsample |
| Stage 4 (2× ResBlock) | Encoder | 512 × 16 × 16 | 9,439,232 | Stride-2 downsample |
| map_coarse | Style head | 4 × 512 | 1,049,088 | AdaptiveAvgPool → Linear |
| map_medium | Style head | 6 × 512 | 1,573,376 | AdaptiveAvgPool → Linear |
| map_fine | Style head | 8 × 512 | 1,049,088 | AdaptiveAvgPool → Linear |
| Const Parameter | StyleGAN2Decoder | 512 × 4 × 4 | 8,192 | Learnable start tensor |
| b4_to_8 | Decoder | 512 × 8 × 8 | 3,152,384 | Upsample + 2× StyleConv |
| b8_to_16 | Decoder | 512 × 16 × 16 | 3,152,384 | Upsample + 2× StyleConv |
| b16_to_32 | Decoder | 256 × 32 × 32 | 2,102,528 | Upsample + 2× StyleConv |
| b32_to_64 | Decoder | 128 × 64 × 64 | 591,872 | Upsample + 2× StyleConv |
| b64_to_128 | Decoder | 64 × 128 × 128 | 148,992 | Upsample + 2× StyleConv |
| b128_to_256 | Decoder | 32 × 256 × 256 | 37,888 | Upsample + 2× StyleConv |
| to_rgb | Decoder | 3 × 256 × 256 | 99 | 1×1 conv, Tanh output |
| **TOTAL** | | | **39,784,675** | All trainable |

---

## 🧱 Training Infrastructure

```python
# Key components verified in phase3_experiments.ipynb

scaler    = GradScaler(enabled=True)          # Mixed precision (FP16)
optimizer = Adam(lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=50, eta_min=1e-6)

# Per-batch training loop
with autocast():
    pred = model(sketch)
    loss = criterion(pred, face)              # L1 [+ LPIPS]

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm=1.0)   # Gradient clipping
scaler.step(optimizer)
scaler.update()
```

Features: AMP · gradient clipping · grad norm logging · YAML config · resumable checkpoints · early stopping · CSV metric logging

---

## ⚠️ Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| `VRAM OOM` at batch=32 with LPIPS | Use `batch_size: 16`; set `lambda_lpips: 0.0` for ablations |
| `FutureWarning` from `torch.cuda.amp` | Update to `torch.amp.GradScaler('cuda')` in PyTorch ≥ 2.4 |
| Hardcoded Windows path in notebook | Edit the `os.chdir()` cell to match your machine |
| Sketch preprocessing silent fail | Ensure `data/celeba_hq/images/` exists before running generation |
| Crop size `ValueError` | Confirm resize to 256×256 happens before augmentation in `__getitem__` |

---

## 🗺️ Phase Roadmap

```
Phase 1  ✓  Project proposal & dataset selection
Phase 2  ✓  MLP baseline — Test L1 = 0.4190 (64×64)
Phase 3  ✓  pSp CNN architecture — Test L1 = 0.4764 (256×256, 5 epochs)
Phase 4  →  Transfer learning · full training · PatchGAN discriminator
Phase 5  →  Error analysis · ablation deep-dive
Phase 6  →  Final report & demo
```

---

## 📄 Report

The Phase 3 Progress Report (`Phase3_Progress_Report.docx`) covers:

1. Cover Page & Repository Link
2. Roadmap Updates
3. EDA Summary (10 findings)
4. Preprocessing Decisions table
5. Baseline Architecture (diagram + hyperparameters)
6. Training Setup & Results (loss curves, ablation table, test metrics)
7. Error Analysis — 5 failure patterns identified
8. Reproducibility Statement
9. Risks Encountered & Plan for Phase 4

---

## 📜 References

- Richardson et al. (2021) — *Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation* ([arXiv:2008.00951](https://arxiv.org/abs/2008.00951))
- Karras et al. (2020) — *Analyzing and Improving the Image Quality of StyleGAN* ([arXiv:1912.04958](https://arxiv.org/abs/1912.04958))
- Liu et al. (2015) — *Deep Learning Face Attributes in the Wild* — CelebA dataset

---

*AI335L · NASTP Institute of Information Technology · Spring 2024*
