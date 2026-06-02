# Sketch-to-Face Image Synthesis

**Course:** AI335L Deep Learning Lab  
**Team:** Moazzam Sharif (S2024CS009) · Bilal Asif (S2024CS005) · Muaz Ahmad (S2024CS016)  
**Model:** pix2pix / pSp (pixel2style2pixel)  
**Dataset:** CelebA

---

## Overview

This project synthesises realistic face images from hand-drawn sketches (or Canny edge maps) using the pSp encoder-decoder architecture trained on CelebA aligned face images.

Pipeline: Hand-drawn sketch → Canny edge map → pSp encoder → StyleGAN2 decoder → Realistic face photo

---

## Repository Structure

```
sketch-to-face/
├── data/                   # Raw + processed data (gitignored)
├── notebooks/
│   └── eda_celeba.ipynb    # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   ├── dataset.py      # Dataset classes + split utilities
│   │   └── download_data.py
│   ├── preprocessing/
│   │   └── pipeline.py     # EdgeMapGenerator, PixelNormalizer
│   ├── models/
│   │   └── baseline_mlp.py # Baseline feedforward network
│   ├── training/
│   │   └── train.py        # Training loop with early stopping
│   └── utils/
│       ├── seed.py          # seed_everything()
│       └── logger.py        # TensorBoard + CSV logging
├── experiments/
│   └── baseline_config.yaml
├── reports/                # Figures, progress reports
├── tests/
│   └── test_pipeline_smoke.py
├── DATA_CARD.md
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/BILAL-ASIF-github/Sketch-to-Face-Image-Synthesis
cd Sketch-to-Face-Image-Synthesis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download CelebA
```bash
# Requires Kaggle API key at ~/.kaggle/kaggle.json
python src/data/download_data.py

# Verify download
python src/data/download_data.py --verify
```

### 4. Run EDA notebook
```bash
jupyter notebook notebooks/eda_celeba.ipynb
```

### 5. Train the baseline MLP
```bash
python src/training/train.py --config experiments/baseline_config.yaml
```

### 6. View TensorBoard logs
```bash
tensorboard --logdir experiments/logs
```

### 7. Run smoke tests
```bash
pytest tests/ -v
```

---

## Reproducibility

**Fixed seed:** `42` (set for Python, NumPy, PyTorch in all scripts via `seed_everything(42)`)  
**Data split:** Random shuffle with `random.Random(42)` → 10K train / 1K val / 1K test  
**Platform tested:** Ubuntu 22.04, Python 3.10, CUDA 12.1

---

## Phase Tags

| Phase | Git Tag |
|-------|---------|
| Phase 1 (Roadmap) | `phase1-submission` |
| Phase 2 (EDA + Baseline) | `phase2-submission` |
| Phase 3 (PSP + CNN Training) | `phase3-submission` |
| Phase 4 (Transfer Learning + VAE + Hyperparameter Search) | `phase4-submission` |
| Phase 5 (Evaluation) | `phase5-submission` |
| Phase 6 (Deployment) | `phase6-submission` |
