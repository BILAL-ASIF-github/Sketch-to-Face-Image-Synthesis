# 🎨 Sketch-to-Face Image Synthesis
### Using Pix2Pix Conditional Generative Adversarial Network

> *"Turning forensic sketches into photorealistic faces through end-to-end deep learning."*

[![Course](https://img.shields.io/badge/Course-AI335L%20Deep%20Learning%20Lab-blue)](/)
[![Institute](https://img.shields.io/badge/Institute-NIIT-darkblue)](/)
[![Phase](https://img.shields.io/badge/Phase-1%20SRS-orange)](/)
[![Semester](https://img.shields.io/badge/Semester-V%20Spring%202024-green)](/)

---

## 📌 Overview

This project automates forensic face reconstruction — converting a rough witness sketch into a photorealistic portrait — using an end-to-end deep learning pipeline trained on paired edge-map and face-image data.

The system integrates a **Pix2Pix cGAN** (U-Net generator + PatchGAN discriminator) with supporting architectures including an MLP baseline, CNN edge enhancer, LSTM sketch refiner, and a Transformer-based perceptual loss head. The final product is a **Streamlit web app** where users upload a sketch and receive a generated face in real time.

---

## 👥 Team

| Name | Reg. ID | Role |
|---|---|---|
| Moazzam Sharif | S2024CS009 | Team Lead & Architecture |
| Bilal Asif | S2024CS005 | Data Analyst & Documentation |
| Muaz Ahmed | S2024CS016 | Testing & Reinforcement |

**Instructor:** Lecturer Haseeb &nbsp;|&nbsp; **Submission:** April 27, 2026

---

## 🧠 Architecture

The pipeline flows through these stages:

```
Raw Sketch
   ↓
Preprocessing & Canny Edge Extraction
   ↓
CNN Edge Enhancer → LSTM Sketch Refiner
   ↓
Pix2Pix cGAN (U-Net Generator)
   ↓
PatchGAN Discriminator ←→ VGG-16 Perceptual Loss (Transformer Attention)
   ↓
Generated Face Image
   ↓
Streamlit Web App
```

### Component Roles

| Module | Type | Purpose |
|---|---|---|
| MLP (5-layer) | Baseline | Performance floor; demonstrates why structural architectures are needed |
| CNN (6-layer) | Edge Enhancer | Cleans noise, closes broken lines in hand-drawn sketches |
| LSTM (hidden=512) | Sketch Refiner | Models sketch as a sequence of 16×16 patch embeddings in raster-scan order |
| U-Net Generator | GAN Core | 8 encoder + 8 decoder blocks with skip connections |
| PatchGAN (70×70) | Discriminator | Local realism; distinguishes real vs. fake patches |
| VGG-16 + Transformer | Perceptual Loss | 4-head self-attention over VGG conv4_3 feature maps |

**Loss Function:**
```
L_total = L_adv + λ·L_L1 + μ·L_perceptual     (λ=100, μ=10)
```

---

## 📦 Datasets

| Dataset | Size | Purpose |
|---|---|---|
| **CelebA** (Primary) | 202,599 images, 10,177 identities | Main training data; Canny-generated sketch pairs |
| **CUFS** (Secondary) | 606 identities | Real hand-drawn sketch-photo pairs |
| **CelebA-HQ** (Secondary) | 30,000 @ 1024×1024 | High-res texture quality |
| **FFHQ** (Backup) | 70,000 @ 1024×1024 | Diverse demographics fallback |

**Train / Val / Test Split:** 70% / 15% / 15%

**Preprocessing pipeline:** Face crop & align → Background removal → Canny edge extraction → Resize to 256×256 → Normalize to [-1, 1] → Pair creation (sketch, real image)

---

## ⚙️ Tech Stack

| Category | Tool | Version |
|---|---|---|
| Core DL | PyTorch | 2.1.0 |
| Pretrained Models | TorchVision (VGG-16) | 0.16.0 |
| Image Processing | OpenCV, Pillow | 4.8.0 / 10.0.0 |
| Hyperparameter Tuning | Optuna (TPE sampler) | 3.3.0 |
| Evaluation | pytorch-fid | 0.3.0 |
| Web UI | Streamlit | 1.25.0 |
| API | FastAPI + Uvicorn | 0.100.0 / 0.23.0 |
| Training Compute | Google Colab Pro (T4/A100) | — |

---

## 🎯 Objectives & Success Metrics

| Objective | Target Metric |
|---|---|
| Sketch-to-face generation | FID ≤ 50, SSIM ≥ 0.65, LPIPS ≤ 0.30 |
| Face matching (Top-5 retrieval) | ≥ 70% accuracy |
| Real-time CCTV detection | ≥ 15 FPS, detection accuracy ≥ 65% |
| End-to-end pipeline | ≥ 40% reduction in identification time vs. manual |

---

## 🗂️ Repository Structure

```
sketch-to-face-dl/
├── data/                   # Raw and processed datasets (gitignored)
├── notebooks/              # EDA, training, evaluation notebooks
├── src/
│   ├── model.py            # Generator, Discriminator, LSTM, CNN Enhancer
│   ├── train.py            # Training loop, optimizer comparison
│   ├── evaluate.py         # FID, SSIM, LPIPS computation
│   └── preprocess.py       # Canny pipeline, augmentation
├── app/
│   ├── main.py             # FastAPI endpoint
│   └── streamlit_app.py    # Streamlit UI
├── checkpoints/            # .pt model files (gitignored, stored on Drive)
├── experiments/            # Optuna DB, training logs
├── docs/                   # SRS, final report, slides
├── environment.yml
└── README.md
```

---

## 🚀 Setup & Run

```bash
# Clone the repo
git clone https://github.com/BILAL-ASIF-github/Sketch-to-FaceImage-Synthesis
cd Sketch-to-FaceImage-Synthesis

# Create environment
conda env create -f environment.yml
conda activate sketch2face

# Run preprocessing
python src/preprocess.py --data_dir data/celeba --output_dir data/processed

# Train the model
python src/train.py --epochs 50 --batch_size 8 --optimizer adam

# Launch the app
streamlit run app/streamlit_app.py
```

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=2e-4, β1=0.5) vs. RMSProp (lr=2e-4) |
| Batch Size | 8 (tuned via Optuna) |
| Epochs | 50 |
| Dropout | p=0.5 in first 3 decoder blocks |
| Early Stopping | Halt if val FID stagnates for 5 epochs |
| Hyperparameter Search | Optuna TPE, 30 trials, pruning at epoch 10 |
| Transfer Learning | VGG-16 frozen up to conv4_3, fine-tuned from conv5_1 |

---

## 📅 Timeline

| Week | Focus | Key Deliverable |
|---|---|---|
| 1 | Setup & SRS | This document; repo scaffolding |
| 2 | Data & Baseline | CelebA pipeline; MLP baseline trained |
| 3 | Core Architecture | U-Net + PatchGAN; LSTM prototype |
| 4 | Advanced Components | VGG-16 transfer learning; Transformer loss; Optuna |
| 5 | Evaluation | Full training; FID/SSIM/LPIPS; ablation study |
| 6 | Deployment & Report | Streamlit app; final report; demo video |

---

## 📄 Deliverables

- `pix2pix_generator.pt` — Trained model weights
- Streamlit + FastAPI deployed demo
- Final report (PDF) + Presentation (PPTX)
- Demo video (3–5 min, YouTube unlisted)
- `environment.yml` with pinned dependencies

---

## 📚 Key References

- Isola et al. (2017) — Pix2Pix cGAN ([CVPR](https://arxiv.org/abs/1611.07004))
- Ronneberger et al. (2015) — U-Net
- Goodfellow et al. (2014) — Generative Adversarial Nets
- Liu et al. (2015) — CelebA Dataset
- Heusel et al. (2017) — FID metric

---

> **Note:** This project is for academic research under AI335L Deep Learning Lab at NIIT. All generated faces are synthetic. The CelebA dataset is used under its non-commercial academic research license.
