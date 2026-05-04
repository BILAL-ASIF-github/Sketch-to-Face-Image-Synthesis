# Data Card — CelebA (Large-scale CelebFaces Attributes Dataset)

## Source
- **URL:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Kaggle Mirror:** https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- **Original Paper:** Liu et al., "Deep Learning Face Attributes in the Wild," ICCV 2015

## License
- **License:** Non-commercial research use only
- **Terms:** Must cite the original paper; redistribution prohibited

## Dataset Statistics
| Property | Value |
|---|---|
| Total Images | 202,599 |
| Image Format | JPEG (aligned & cropped) |
| Resolution | 178 × 218 pixels (original), resized to 256 × 256 for this project |
| Color Channels | RGB |
| Modality | Face images |

## Splits Used in This Project
| Split | Count |
|---|---|
| Training | 10,000 |
| Validation | 1,000 |
| Test | 1,000 |

## Class Distribution
CelebA is an unconditional face dataset — it does not have class labels per se. The 40 binary facial attributes (smiling, wearing glasses, etc.) are available but unused in this project. All samples are face images; there is no class imbalance concern for the image synthesis task.

## Modality Details
This project derives paired data from CelebA:
- **Input (edge maps):** Canny edge detection applied to each face image, producing sketch-like white-background / black-line representations
- **Target (photos):** The original CelebA aligned face image

## Known Issues
- CelebA skews toward lighter skin tones and Western facial features (well-documented bias)
- A small fraction of images have occlusions (sunglasses, hats) — edge maps of these may be unusual
- No explicit train/val/test split is enforced from the dataset side; we apply our own fixed random split (seed=42)
- Image quality varies; a few images are noticeably lower resolution or slightly blurry

## Verification
After download, verify with:
```bash
python src/data/download_data.py --verify
```
Expected: 202,599 `.jpg` files in `data/raw/img_align_celeba/`

## Storage
- Raw CelebA images: ~1.4 GB
- Generated edge-map pairs: ~500 MB (256×256 PNG)
- Recommended: Store in `data/` (gitignored), or mount from Kaggle dataset

## Citation
```bibtex
@inproceedings{liu2015faceattributes,
  title     = {Deep Learning Face Attributes in the Wild},
  author    = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of ICCV},
  year      = {2015}
}
```

---

# Backup Dataset — FFHQ (Flickr-Faces-HQ)

## Source
- **URL:** https://github.com/NVlabs/ffhq-dataset
- **Kaggle Mirror:** https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
- **Original Paper:** Karras et al., "A Style-Based Generator Architecture for GANs," CVPR 2019

## Why FFHQ as Backup
FFHQ is a direct drop-in replacement for CelebA — it contains aligned face images
and is fully compatible with the same preprocessing pipeline (Canny edge generation,
PixelNormalizer). Advantages over CelebA: higher resolution (1024×1024), more diverse
skin tones, and a more permissive license (Creative Commons).

## License
- **License:** Creative Commons BY-NC-SA 4.0 (non-commercial)

## Dataset Statistics
| Property | Value |
|---|---|
| Total Images | 70,000 |
| Resolution | 1024 × 1024 (resized to 256×256 for training) |
| Color Channels | RGB |

## Download
```bash
python src/data/download_data.py --dataset backup
python src/data/download_data.py --dataset backup --verify
```
