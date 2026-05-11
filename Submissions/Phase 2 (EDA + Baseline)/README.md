# 🎨 Sketch-to-Face Image Synthesis
### Phase 2 — EDA, Preprocessing & Baseline Model

[![Phase](https://img.shields.io/badge/Phase-2%20of%206-orange)](/)
[![Course](https://img.shields.io/badge/Course-AI335L%20Deep%20Learning%20Lab-blue)](/)
[![Dataset](https://img.shields.io/badge/Dataset-CelebA%20202K-green)](/)
[![Framework](https://img.shields.io/badge/PyTorch-2.3.0-red)](/)
[![Python](https://img.shields.io/badge/Python-3.10-yellow)](/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900)](/)

---

## 📋 Phase 2 Summary

This phase covers the complete data foundation for the project: repository setup, CelebA acquisition, 10-finding EDA, a data-driven preprocessing pipeline, and a trained MLP baseline that sets the performance floor for all subsequent phases.

| Item | Detail |
|---|---|
| Deadline | May 04, 2026 |
| Dataset | CelebA — 202,599 aligned face images |
| Subset Used | 10K train / 1K val / 500 test (seed=42) |
| Baseline Test L1 | **0.4190** (benchmark for Phases 3–5) |
| Training Time | ~5 min / 50 epochs on Kaggle T4 GPU |
| Commit Tag | `phase2-submission` |

---

## 👥 Team

| Name | Reg. ID | Role |
|---|---|---|
| Moazzam Sharif | S2024CS009 | Team Lead & Architecture |
| Bilal Asif | S2024CS005 | Data Analyst & Documentation |
| Muaz Ahmed | S2024CS016 | Testing & Reinforcement |

---

## 📁 Repository Structure

```
sketch-to-face/
├── data/                        # Raw + processed data (gitignored)
├── notebooks/
│   └── eda_celeba.ipynb         # Full EDA notebook (10 findings)
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset classes + split utilities
│   │   └── download_data.py     # CelebA download via Kaggle API
│   ├── preprocessing/
│   │   └── pipeline.py          # EdgeMapGenerator, PixelNormalizer
│   ├── models/
│   │   └── baseline_mlp.py      # Baseline feedforward network
│   ├── training/
│   │   └── train.py             # Training loop with early stopping
│   └── utils/
│       ├── seed.py              # seed_everything()
│       └── logger.py            # TensorBoard + CSV logging
├── experiments/
│   └── baseline_config.yaml     # Hyperparameters
├── reports/                     # Figures, progress reports
├── tests/
│   └── test_pipeline_smoke.py
├── DATA_CARD.md
├── requirements.txt
└── .gitignore
```

---

## 🔍 EDA — Key Findings

| # | Finding | Preprocessing Decision |
|---|---|---|
| 1 | 202,599 images; using 11,500 (seed=42) | Fixed train/val/test split |
| 2 | All images consistently aligned | No alignment step needed |
| 3 | Single resolution 178×218 px across dataset | Resize to 256×256 (pSp) / 64×64 (MLP) |
| 4 | Brightness ~N(130,20); ~6% dark/bright outliers | Bilateral filter before Canny |
| 5 | R > G > B channel means (warm skin-tone bias) | Per-channel normalisation to [-1,1] |
| 6 | Zero corrupted or duplicate files | No deduplication needed |
| 7 | Canny captures facial structure; accessories → dense edges | Edge density filtering |
| 8 | Edge density N(0.06, 0.02); ~5% too sparse/dense | Filter density < 0.02 or > 0.15 |
| 9 | Faces are bilaterally symmetric | Horizontal flip is valid augmentation |
| 10 | Training channel stats: R=0.54, G=0.49, B=0.44 | Used for channel-wise normalisation |

---

## ⚙️ Preprocessing Pipeline

Each EDA finding maps directly to a pipeline step:

```
Face Photo
   ↓  Resize to 256×256 (pSp) or 64×64 (MLP)
   ↓  Bilateral Filter (d=9, σ=75)  ← handles brightness outliers
   ↓  Canny Edge Detection (low=50, high=150)
   ↓  Dilate (1 iter) → Invert      ← stable sketch-photo correspondence
   ↓  Per-channel normalisation to [-1,1] using train stats only
   ↓  Filter: remove edge density < 0.02 or > 0.15
   ↓  Random horizontal flip (p=0.5, applied to edge+photo together)
   ↓  Brightness jitter (×0.7–1.3) on edge maps
   ↓
(Sketch, Face) Paired Tensors
```

> **No data leakage:** normalisation statistics computed on training set only.

---

## 🧠 Baseline Model — MLP

A 3-hidden-layer feedforward network operating on flattened 64×64 images.

```
Input Edge Map (3×64×64)
   → Flatten (12,288)
   → Linear + BN → LeakyReLU(0.2) → Dropout(0.3)  [2,048]
   → Linear + BN → LeakyReLU(0.2) → Dropout(0.3)  [1,024]
   → Linear + BN → LeakyReLU(0.2) → Dropout(0.3)  [512]
   → Linear → Tanh                                  [12,288]
   → Reshape (3×64×64)
Output Reconstructed Face
```

### Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Image size | 64×64 | MLP input dim stays manageable (12,288-d) |
| Batch size | 64 | Good GPU utilisation; stable gradients |
| Learning rate | 1e-3 | Standard Adam starting LR |
| LR scheduler | ReduceLROnPlateau (×0.5, p=5) | Adapts to plateau without manual tuning |
| Early stopping | Patience=10 epochs | Prevents overfitting |
| Gradient clip | 1.0 | Prevents gradient explosion |
| Weight decay | 1e-4 | L2 regularisation |
| Weight init | Kaiming (He) uniform | Correct for LeakyReLU activations |
| Loss | L1 (MAE) | Sharper outputs vs MSE's blurry mean |
| Seed | 42 | Full reproducibility |

---

## 📊 Training Results (50 Epochs)

| Metric | Value | Notes |
|---|---|---|
| Train L1 (epoch 1) | ~0.48 | High — random weights |
| Train L1 (epoch 10) | 0.4025 | Rapid early improvement |
| Train L1 (epoch 30) | 0.3439 | Continued improvement |
| Train L1 (epoch 50) | 0.3041 | Converged; MLP capacity limit |
| Val L1 (best) | 0.3951 | Small generalisation gap |
| **Test L1 (final)** | **0.4190** | **Benchmark for Phases 3–5** |
| Early stopping | Did not trigger | Ran full 50 epochs |
| LR at epoch 50 | 0.000125 | Reduced 3× by scheduler |

Training hardware: Kaggle T4 GPU (16 GB VRAM) — ~5 minutes total.

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis
cd Sketch-to-Face-Image-Synthesis

# Install dependencies
pip install -r requirements.txt

# Download CelebA (requires Kaggle API key at ~/.kaggle/kaggle.json)
python src/data/download_data.py
python src/data/download_data.py --verify

# Train baseline MLP
python src/training/train.py --config experiments/baseline_config.yaml --seed 42

# Run smoke tests
pytest tests/ -v

# Launch TensorBoard
tensorboard --logdir experiments/logs
```

---

## 🗺️ Roadmap Updates (vs. Phase 1 Plan)

| Update | Detail |
|---|---|
| Dataset scope reduced | 50K → 10K train pairs (T4 session limits ~10–15 min/epoch) |
| Bilateral filter added | EDA found dark/bright outliers degraded Canny quality |
| Baseline MLP added | Phase 2 requirement; 3-layer MLP on 64×64 as measurable yardstick |
| No backup dataset needed | CelebA downloaded cleanly (0 corrupted); FFHQ identified but not required |
| Core plan unchanged | Data prep → pSp training → Evaluation → Deployment |

---

## 🔭 Phase 3 Plan

**Architecture:** Full pSp (pixel2style2pixel) pipeline
- **Encoder:** IR-SE50 backbone → W+ styles (18×512)
- **Decoder:** Frozen StyleGAN2-FFHQ → 256×256 output

**Loss:** `L_total = λ₁·L1 + λ₂·L_VGG + λ₃·L_ID`  (λ₁=1.0, λ₂=0.1, λ₃=0.1)

**Scale:** 50K edge-face pairs

**Metrics:** FID, LPIPS, L1 — all must beat baseline test L1 of **0.4190**

**Real-sketch test:** DB Cooper sketch → generated face (end-to-end forensic demo)

---

## ⚠️ Known Failure Modes (MLP)

- **Accessories** (glasses, hats): Anomalously dense edge maps dominate representation
- **Dark images**: Sparse edges (density < 0.02) → model defaults to average face
- **Profile faces**: CelebA is frontal-biased; asymmetric inputs generalise poorly
- **Spatial blindness**: Flattened input loses all spatial relationships — features are smeared

> Full error analysis with 10–20 worst predictions will be completed after Phase 3 pSp training.

---

## 🔁 Reproducibility

All randomness is controlled via `seed_everything(42)`:
- Python `random`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `cudnn.deterministic = True`

```bash
python src/training/train.py --config experiments/baseline_config.yaml --seed 42
```

---

## ⚡ Active Risks

| Risk | Status | Mitigation |
|---|---|---|
| Kaggle T4 session limits | Active | Checkpoint saving every epoch; resume support |
| pSp pretrained weights (Google Drive) | Managed | `gdown` with fallback mirrors documented |
| CelebA license (non-commercial only) | Noted | Academic use; cited in `DATA_CARD.md` |
| Edge maps ≠ real forensic sketches | Monitored | DB Cooper test validates real-sketch inference |

---

> **Note:** CelebA is used under its non-commercial academic research license. All generated faces are synthetic. See `DATA_CARD.md` for full dataset documentation.
