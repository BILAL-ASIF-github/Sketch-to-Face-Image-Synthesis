# 🎨 Sketch-to-Face Image Synthesis

> **AI335L — Deep Learning Lab | Phase 4 Refinement Report**
> Transform hand-drawn Canny edge sketches into photorealistic face photographs using pixel2Style2pixel (pSp) with transfer learning, VAE-based augmentation, and Bayesian hyperparameter optimisation.

<p align="center">
  <img src="https://img.shields.io/badge/Phase-4%20Refinement-00B4D8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Backbone-ResNet--50-0D1B2A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HPO-Optuna-1B6AC6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-CelebA-F59E0B?style=for-the-badge" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Phase 4 Contributions](#-phase-4-contributions)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
  - [Training the pSp Model](#training-the-psp-model)
  - [Training the VAE](#training-the-vae)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Inference](#inference)
- [Transfer Learning Configs](#-transfer-learning-configs)
- [VAE Details](#-vae-details)
- [Hyperparameter Search Details](#-hyperparameter-search-details)
- [Negative Results & Lessons](#-negative-results--lessons)
- [Phase 5 Roadmap](#-phase-5-roadmap)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview

This project implements an end-to-end pipeline that converts **256×256 Canny edge sketch images** of human faces into **photorealistic face photographs**. The core architecture is **pixel2Style2pixel (pSp)**: a ResNet-50 encoder that maps edge maps to 18 style codes (each of dimension 512), which are fed into a **StyleGAN2 decoder** to synthesise the final image.

### Pipeline at a Glance

```
Input Sketch         Encoder              Style Codes         Decoder              Output Face
  256×256       →   ResNet-50        →   (18 × 512)      →   StyleGAN2        →    256×256
  Canny edge        4 residual            F.normalize          decoder              photorealistic
  white bg          stages            +   3 proj heads         (frozen or             face
                    ImageNet W.          unit sphere           fine-tuned)
```

### Key Metrics (Phase 4 vs Prior Work)

| Model | Val L1 Loss | Improvement |
|---|---|---|
| MLP Baseline | 0.4190 | — |
| Phase 3 pSp (scratch) | 0.4764 | −13.7% vs baseline |
| Phase 4 Config A (frozen) | 0.4119 | +1.7% vs baseline |
| Phase 4 Config B (full FT) | 0.3876 | +7.5% vs baseline |
| **Phase 4 Config C ★ (diff LR)** | **0.3845** | **+8.2% vs baseline** |
| Phase 4 Config D (gradual) | 0.3908 | +6.8% vs baseline |
| **Phase 4 + Optuna HPO (Trial 5)** | **0.3626** | **+13.5% vs baseline** |
| **Phase 4 Final Test** | **0.3556** | **+15.1% vs baseline** |

---

## 🧠 Architecture

### pSp Encoder

```python
class pSpEncoder(nn.Module):
    """
    ResNet-50 backbone with 3 map2style projection heads.
    Produces 18 style codes of dimension 512, normalised to unit sphere.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # Remove final FC + avgpool → keep 4 residual stages
        self.layer0  = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1  = backbone.layer1   # 256 ch
        self.layer2  = backbone.layer2   # 512 ch
        self.layer3  = backbone.layer3   # 1024 ch
        self.layer4  = backbone.layer4   # 2048 ch
        self.pool    = nn.AdaptiveAvgPool2d((1,1))
        # 3 projection heads: coarse (8), medium (6), fine (4) = 18 codes total
        self.head_coarse  = nn.Linear(2048, 8  * 512)
        self.head_medium  = nn.Linear(2048, 6  * 512)
        self.head_fine    = nn.Linear(2048, 4  * 512)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)                    # (B, 2048)
        codes = torch.cat([
            self.head_coarse(x).view(-1, 8,  512),
            self.head_medium(x).view(-1, 6,  512),
            self.head_fine  (x).view(-1, 4,  512),
        ], dim=1)                                       # (B, 18, 512)
        return F.normalize(codes, dim=-1)               # unit sphere ← critical!
```

### StyleGAN2 Decoder

The decoder is loaded from a pretrained checkpoint (`stylegan2-ffhq-config-f.pt`). It accepts a `(B, 18, 512)` style tensor and produces `(B, 3, 256, 256)` face images. In Config C (best), the decoder is fine-tuned at the same learning rate as the ResNet-50 backbone (`lr_backbone`).

### Loss Function

```
L_total = λ_l1 · ||y − ŷ||₁  +  λ_lpips · LPIPS(y, ŷ)
```

- `λ_l1` default: 1.0 (searched: 0.5–2.0)
- `λ_lpips` default: 0.8 (searched: 0.2–1.5); **evaluated on 10 val batches only** during training to avoid 30–45 min epochs

---

## 🚀 Phase 4 Contributions

### 1. Transfer Learning (ResNet-50 Backbone)

Replaced random encoder initialisation with **ImageNet-1K V2 pretrained ResNet-50**, tested across 4 freezing configurations. Config C (differential learning rates) gave the best result.

### 2. Convolutional VAE for Data Augmentation

A **51.8M-parameter ConvVAE** (latent dim = 512, matching StyleGAN2 style codes) was trained on CelebA faces. It generates 100+ identity-preserving synthetic faces via latent-space perturbation (`z = μ + 0.1·ε`).

### 3. Bayesian Hyperparameter Search (Optuna)

TPE sampler with MedianPruner searches 6 hyperparameters over a planned 20-trial study (6 completed due to system shutdowns). **`dropout_p` was the most important hyperparameter** (FAnova importance: 0.42).

---

## 📊 Results

### Transfer Learning Comparison

```
Val L1 Loss (↓ better)

0.50 ┤
0.48 ┤  ████ Phase 3 scratch (0.4764)
0.46 ┤
0.44 ┤
0.42 ┤  ████ MLP Baseline (0.4190)
0.40 ┤                              ████ Config A (0.4119)
0.38 ┤                                            ████ Config B (0.3876)  ████ Config C★ (0.3845)  ████ Config D (0.3908)
0.36 ┤                                                                                                                       ████ HPO Trial 5 (0.3626)
0.34 ┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### VAE Training Curves (Selected Epochs)

| Epoch | Train L_total | Val L_total | Val L_KL |
|---|---|---|---|
| 1 | 0.3290 | 0.2774 | 0.0021 |
| 2 | 0.3077 | 0.2882 | 0.0189 |
| 3 | 0.3087 | 0.2919 | 0.0413 |
| 5 | 0.2864 | 0.2761 | 0.0413 |
| 8 | 0.2725 | 0.2673 | 0.0413 |

### HPO Importance (FAnova)

| Rank | Hyperparameter | Importance |
|---|---|---|
| 1 | `dropout_p` | **0.4204** |
| 2 | `lr_backbone` | 0.1872 |
| 3 | `lambda_l1` | 0.1253 |
| 4 | `lr_head` | 0.1104 |
| 5 | `weight_decay` | 0.0918 |
| 6 | `lambda_lpips` | 0.0649 |

---

## 📁 Project Structure

```
Sketch-to-Face-Image-Synthesis/
│
├── data/
│   ├── celeba/
│   │   ├── images/          # CelebA face images (img_align_celeba/)
│   │   └── sketches/        # Corresponding Canny edge maps
│   └── augmented/           # VAE-generated augmented faces
│
├── models/
│   ├── encoder.py           # pSpEncoder (ResNet-50 + 3 projection heads)
│   ├── decoder.py           # StyleGAN2 wrapper
│   ├── psp.py               # Full pSp model (encoder + decoder)
│   └── vae.py               # ConvVAE (encoder + reparameterisation + decoder)
│
├── training/
│   ├── train_psp.py         # pSp training loop (all 4 configs)
│   ├── train_vae.py         # VAE training with KL annealing
│   └── hpo_search.py        # Optuna Bayesian HPO study
│
├── inference/
│   ├── generate.py          # Single sketch → face inference
│   └── batch_generate.py    # Batch generation with metrics
│
├── utils/
│   ├── dataset.py           # CelebA sketch/face dataset loader
│   ├── losses.py            # L1 + LPIPS loss with lambda weighting
│   ├── metrics.py           # L1, LPIPS, SSIM evaluation
│   └── augment.py           # VAE-based augmentation pipeline
│
├── checkpoints/             # Saved model weights (gitignored)
│   ├── psp_config_c_best.pt
│   ├── vae_epoch8.pt
│   └── stylegan2-ffhq-config-f.pt
│
├── notebooks/
│   ├── phase3_recap.ipynb
│   ├── transfer_learning_analysis.ipynb
│   ├── vae_latent_interpolation.ipynb
│   └── hpo_results_visualisation.ipynb
│
├── results/
│   ├── phase4_metrics.json
│   ├── optuna_study.db      # SQLite HPO persistence
│   └── sample_outputs/      # Generated face images
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (recommended) or CPU
- 8 GB+ GPU VRAM for training (4 GB for inference)

### 1. Clone the Repository

```bash
git clone https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis.git
cd Sketch-to-Face-Image-Synthesis
git checkout phase4-submission
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```text
torch>=2.1.0
torchvision>=0.16.0
lpips>=0.1.4
optuna>=3.4.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-image>=0.22.0
tqdm>=4.66.0
matplotlib>=3.8.0
pandas>=2.1.0
scipy>=1.11.0
ninja>=1.11.1           # for StyleGAN2 custom CUDA ops
```

### 4. Download Pretrained StyleGAN2 Weights

```bash
# FFHQ 256×256 config-f checkpoint (~300 MB)
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl \
     -O checkpoints/stylegan2-ffhq-config-f.pt
```

> **Note:** You may need to convert the `.pkl` to a `.pt` state dict. See `utils/convert_sg2.py` for the conversion script.

---

## 🗃️ Dataset Setup

### CelebA

```bash
# Option A — via torchvision (auto-download, requires ~1.4 GB)
python -c "from torchvision.datasets import CelebA; CelebA('./data', download=True)"

# Option B — manual download from official site
# https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Place img_align_celeba/ under data/celeba/images/
```

### Generate Edge Maps (Sketches)

```bash
python utils/generate_sketches.py \
    --input  data/celeba/images/ \
    --output data/celeba/sketches/ \
    --sigma  1.0 \
    --low_threshold  50 \
    --high_threshold 150
```

This runs Canny edge detection on all 202,599 CelebA images and saves white-background 256×256 edge maps.

### VAE-Augmented Faces

After training the VAE (see below), generate augmented faces:

```bash
python utils/augment.py \
    --vae_checkpoint checkpoints/vae_epoch8.pt \
    --input data/celeba/images/ \
    --output data/augmented/ \
    --n_samples 100 \
    --noise_sigma 0.1
```

---

## 🏃 Usage

### Training the pSp Model

```bash
# Config C — Differential Learning Rates (recommended)
python training/train_psp.py \
    --config         C \
    --data_root      data/celeba/ \
    --stylegan_ckpt  checkpoints/stylegan2-ffhq-config-f.pt \
    --epochs         5 \
    --batch_size     8 \
    --lr_head        1e-3 \
    --lr_backbone    1e-5 \
    --weight_decay   1e-4 \
    --lambda_l1      1.0 \
    --lambda_lpips   0.8 \
    --save_dir       checkpoints/ \
    --log_interval   50

# Config A — Frozen backbone (fastest, baseline)
python training/train_psp.py --config A --lr_head 1e-3 ...

# Config B — Full fine-tuning
python training/train_psp.py --config B --lr_head 1e-3 --lr_backbone 1e-5 ...

# Config D — Gradual unfreeze
python training/train_psp.py --config D --unfreeze_schedule "2,4,6,8" ...
```

#### Config Reference

| Config | Backbone | Backbone LR | Trainable Params |
|---|---|---|---|
| A | Fully frozen | 0 | 30.3M (heads only) |
| B | All weights | 1e-5 | 53.8M |
| **C ★** | **Differential LR** | **1e-5 (heads: 1e-3)** | **53.8M** |
| D | Gradual unfreeze | 0 → 1e-5 | 30.3M → 53.6M |

---

### Training the VAE

```bash
python training/train_vae.py \
    --data_root    data/celeba/images/ \
    --epochs       8 \
    --batch_size   32 \
    --lr           1e-4 \
    --latent_dim   512 \
    --kl_anneal_epochs 3 \
    --save_dir     checkpoints/ \
    --log_interval 100
```

**KL annealing schedule:** β increases linearly from 0 → 1 over the first 3 epochs, then stays at 1. This prevents posterior collapse while keeping the reconstruction quality high.

---

### Hyperparameter Search

```bash
# Start / resume an Optuna study (SQLite persistence)
python training/hpo_search.py \
    --study_name   psp_hpo_v1 \
    --storage      sqlite:///results/optuna_study.db \
    --n_trials     20 \
    --n_jobs       1 \
    --data_root    data/celeba/ \
    --stylegan_ckpt checkpoints/stylegan2-ffhq-config-f.pt

# Inspect results
python -c "
import optuna
study = optuna.load_study(
    study_name='psp_hpo_v1',
    storage='sqlite:///results/optuna_study.db'
)
print('Best trial:', study.best_trial.number)
print('Best val L1:', study.best_value)
print('Best params:', study.best_params)
"
```

**Search space:**

| Parameter | Type | Range |
|---|---|---|
| `lr_head` | float (log) | 1e-4 → 1e-2 |
| `lr_backbone` | float (log) | 1e-6 → 1e-4 |
| `weight_decay` | float (log) | 1e-5 → 1e-2 |
| `dropout_p` ★ | float | 0.0 → 0.5 |
| `lambda_l1` | float | 0.5 → 2.0 |
| `lambda_lpips` | float | 0.2 → 1.5 |

**Best configuration found (Trial 5):**

```python
best_params = {
    "lr_head":       1.307e-4,
    "lr_backbone":   5.399e-5,
    "weight_decay":  6.358e-4,
    "dropout_p":     0.354,     # most important!
    "lambda_l1":     0.531,
    "lambda_lpips":  1.461,
}
# → Val L1: 0.3626
```

---

### Inference

```bash
# Single image
python inference/generate.py \
    --checkpoint checkpoints/psp_config_c_best.pt \
    --input      path/to/sketch.png \
    --output     path/to/output_face.png

# Batch generation with metrics
python inference/batch_generate.py \
    --checkpoint  checkpoints/psp_config_c_best.pt \
    --data_root   data/celeba/ \
    --split       test \
    --output_dir  results/sample_outputs/ \
    --compute_metrics   # computes L1, LPIPS, SSIM
```

---

## 🔧 Transfer Learning Configs

The encoder initialisation strategy is controlled by `--config {A,B,C,D}`:

```python
# Config C implementation (train_psp.py excerpt)
def build_optimizer(model, config):
    if config == "C":
        backbone_params = list(model.encoder.layer0.parameters()) + \
                          list(model.encoder.layer1.parameters()) + \
                          list(model.encoder.layer2.parameters()) + \
                          list(model.encoder.layer3.parameters()) + \
                          list(model.encoder.layer4.parameters())
        head_params     = list(model.encoder.head_coarse.parameters()) + \
                          list(model.encoder.head_medium.parameters()) + \
                          list(model.encoder.head_fine.parameters())
        decoder_params  = list(model.decoder.parameters())
        return torch.optim.Adam([
            {"params": backbone_params, "lr": 1e-5},
            {"params": decoder_params,  "lr": 1e-5},
            {"params": head_params,     "lr": 1e-3},
        ], weight_decay=1e-4)
```

---

## 🔬 VAE Details

### Architecture

```
ENCODER
Input (3×256×256)
  → Conv(3→64,   k=4, s=2) + ReLU   → 64×128×128
  → Conv(64→128, k=4, s=2) + ReLU   → 128×64×64
  → Conv(128→256,k=4, s=2) + ReLU   → 256×32×32
  → Conv(256→512,k=4, s=2) + ReLU   → 512×16×16
  → Conv(512→512,k=4, s=2) + ReLU   → 512×8×8
  → Flatten → FC(32768 → 512) → μ
           → FC(32768 → 512) → log σ²

REPARAMETERISATION
  z = μ + ε · exp(0.5 · log σ²)     where ε ~ N(0, I)

DECODER
  z (512) → FC(512 → 512×8×8) → Reshape (512×8×8)
  → ConvT(512→256, k=4, s=2) + ReLU  → 256×16×16
  → ConvT(256→128, k=4, s=2) + ReLU  → 128×32×32
  → ConvT(128→64,  k=4, s=2) + ReLU  → 64×64×64
  → ConvT(64→32,   k=4, s=2) + ReLU  → 32×128×128
  → ConvT(32→3,    k=4, s=2) + Tanh  → 3×256×256
```

### Loss

```
L_total = L_recon  +  β · L_KL
        = ||x − x̂||₁  +  β · (−½ · Σ(1 + log σ² − μ² − σ²))

KL annealing: β = min(1.0,  epoch / kl_anneal_epochs)
```

**Final metrics (epoch 8):** Train L1 = 0.2725 · Val L1 = 0.2434 · Val KL = 0.0413

---

## 📉 Negative Results & Lessons

These are documented honestly so future phases (and other researchers) can avoid the same pitfalls.

| # | Issue | Root Cause | Fix Applied |
|---|---|---|---|
| 01 | NaN losses from random projection heads | Unconstrained style codes caused StyleGAN2 divergence | `F.normalize(styles, dim=-1)` before decoding |
| 02 | MedianPruner too aggressive (14/20 trials pruned at epoch 1) | Default pruner fired too early | `n_startup_trials=10, n_warmup_steps=2` |
| 03 | Corrupted Optuna SQLite DB | Stale variable reference in objective → infinite retry loop | Test objective manually before full study |
| 04 | Two system shutdowns mid-HPO (5–6 hr sessions) | OS power management | Disable sleep; SQLite persistence auto-resumed |
| 05 | LPIPS in training loop: 30–45 min epochs | Per-batch LPIPS forward pass too expensive | Moved to validation-only on 10 batches |

---

## 🗺️ Phase 5 Roadmap

- [ ] **Full training run** — 20–30 epochs with best HPO config (`dropout_p=0.354`, `lambda_lpips=1.461`)
- [ ] **L1 + LPIPS as training loss** (not just evaluation metric)
- [ ] **SSIM on full test set**
- [ ] **FID score** — 1,000 generated vs 1,000 real CelebA faces
- [ ] **Ablation studies** — with/without `F.normalize`, with/without VAE augmentation, L1-only vs L1+LPIPS
- [ ] **Error analysis** — failure modes (hair occlusion, extreme pose), attention/gradient maps, error distribution by CelebA attribute
- [ ] **Complete 20-trial Optuna study** with no system interruptions
- [ ] **Potential VAE–pSp fusion** — inject VAE latent z as a prior for encoder style codes

---

## 👥 Team

| Name | Role |
|---|---|
| **Moazzam Sharif** | Model architecture, transfer learning experiments |
| **Bilal Asif** | VAE design & training, GitHub repository |
| **Muaz Ahmad** | Hyperparameter search, evaluation pipeline |

---

## 📄 License

This project is released under the [MIT License](LICENSE).

**Third-party components:**
- StyleGAN2 — NVIDIA Research, [NVIDIA License](https://github.com/NVlabs/stylegan2)
- ResNet-50 weights — BSD-3-Clause (torchvision)
- CelebA dataset — for non-commercial research use only

---

<p align="center">
  <sub>AI335L · Deep Learning Lab · Phase 4 Submission · tag: <code>phase4-submission</code></sub>
</p>
