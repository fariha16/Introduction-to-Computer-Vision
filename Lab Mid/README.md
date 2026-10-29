# Lab Mid: HOG vs SIFT on Blood-Cell Images

**Name:** Fariha  
**Reg No:** 053  
**Course:** Computer Vision Lab

## Task
Build two classical blood-cell classifiers, **HOG + SVM** and **SIFT + SVM**, that run on a normal laptop, and find out which one is more reliable when images are rotated or noisy.

## Files
| File | Description |
|---|---|
| `FA23_BAI_053_Fariha_LabMid.ipynb` | Executed notebook with all outputs (Q1–Q5) |
| `FA23_BAI_053_Fariha_LabMid.pdf` | PDF copy of the notebook |

## Personal settings (from reg no 053)
| Setting | Rule | Value |
|---|---|---|
| Random seed | Last two digits | **53** |
| HOG cell size | Last digit 3 is odd | **16 × 16** |
| SIFT grid | Last digit 3 is in 0–4 | **4 × 4 (16 keypoints)** |

## Setup
- **Dataset:** BloodMNIST (`medmnist`), size 64
- **Classes:** eosinophil (1), monocyte (5), neutrophil (6), platelet (7)
- **Split:** 100 train / 30 validation / 30 test images per class → 400 / 120 / 120
- **Preprocessing:** grayscale, resized to 128 × 128
- **HOG:** 9 orientations, 16 × 16 cells, 2 × 2 cells per block, L2-Hys normalisation
- **SIFT:** fixed 4 × 4 grid of keypoints (size 32) over the image, descriptors from `sift.compute`
- **Classifier:** StandardScaler + LinearSVC, C chosen from {0.1, 1, 10} on the validation set

## Results

### Feature lengths
| Features | Calculation | Length |
|---|---|---|
| HOG | 7 × 7 blocks × (2 × 2 cells × 9 bins) | **1764** |
| Grid SIFT | 16 keypoints × 128 | **2048** |

### Validation (C chosen on validation, best C = 0.1 for both)
| Model | Val accuracy | Macro-F1 |
|---|---|---|
| HOG + SVM | 0.733 | 0.732 |
| SIFT + SVM | 0.758 | 0.760 |

### Normal SIFT detection (training set)
Average 72.6 keypoints per image (min 4, max 213), with 0 images having zero keypoints. Because the number of keypoints varies, the descriptors can't be stacked into one fixed-length vector for the SVM, which is why the fixed grid was used.

### Robustness on the test set (frozen models)
| Model | Clean | Rotated 20° | Noisy (σ = 0.05) |
|---|---|---|---|
| HOG + SVM | 0.658 | **0.733** | 0.275 |
| SIFT + SVM | **0.700** | 0.708 | **0.692** |

## Conclusion
- **SIFT + SVM was the more reliable model.** It scored higher on clean and noisy test images, and its accuracy stayed almost the same in all three cases.
- **Noise:** HOG + SVM dropped to 0.275, close to the 0.25 chance level for 4 classes. Noise creates false gradients in the smooth background that HOG picks up. Each SIFT descriptor pools gradients over a 32-pixel patch, so the noise averages out.
- **Rotation (20°):** neither model lost accuracy. HOG rose from 0.658 to 0.733, but with only 120 test images (one image = 0.8%), this is most likely random variation rather than rotation genuinely helping.
- **Most confused classes:** eosinophil and monocyte for both models. These two cells differ mainly in colour (red-orange granules vs grey-blue cytoplasm), which is lost after converting to grayscale.


## Libraries
`medmnist`, `numpy`, `opencv-python` (`cv2`), `scikit-image`, `scikit-learn`, `matplotlib`
