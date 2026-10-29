Lab Mid: HOG vs SIFT on Blood-Cell Images

**Name:** Fariha  
**Reg No:** 053  
**Course:** Computer Vision Lab  

---

## Task
A small lab wants a simple blood-cell classifier that runs on a normal laptop. This project builds two classical computer-vision pipelines, **HOG + SVM** and **SIFT + SVM**, and compares which one is more reliable when the images are **rotated** or **noisy**.

---

## Files
| File | Description |
|---|---|
| `FA23_BAI_053_LabMid.ipynb` | Executed notebook with all outputs (Q1–Q5) |
| `FA23_BAI_053_LabMid.pdf` | PDF copy of the notebook |
| `README.md` | This file |

---

## Personal Settings (from Reg No 053)
| Setting | Rule | My Value |
|---|---|---|
| Random seed | Last two digits of reg no | **53** |
| HOG cell size | Last digit (3) is odd → 16 × 16 | **16 × 16** |
| SIFT grid size | Last digit (3) is in 0–4 → 4 × 4 | **4 × 4 (16 keypoints)** |

---

## Fixed Setup
| Item | Value |
|---|---|
| Dataset | BloodMNIST (`medmnist`), image size 64 |
| Classes | eosinophil (1), monocyte (5), neutrophil (6), platelet (7) |
| Images per class | 100 train, 30 validation, 30 test |
| Total images | 400 train, 120 validation, 120 test |
| Preprocessing | Grayscale, resized to 128 × 128 |
| HOG | 9 orientations, 16 × 16 cells, 2 × 2 cells per block, L2-Hys |
| SIFT | Fixed 4 × 4 grid, keypoint size 32 |
| Classifier | StandardScaler + LinearSVC, C ∈ {0.1, 1, 10} chosen on validation |
| Rotation test | 20 degrees |
| Noise test | Gaussian noise, σ = 0.05 (image scaled to 0–1) |

---

## Pipeline
```
BloodMNIST (RGB 64×64)
      │
      ▼
Keep 4 classes → sample 100/30/30 per class (seed 53)
      │
      ▼
Grayscale → Resize to 128×128
      │
      ├──► HOG features (1764) ──► StandardScaler ──► LinearSVC
      │
      └──► Grid SIFT features (2048) ──► StandardScaler ──► LinearSVC
                                                │
                                                ▼
                        Evaluate on Clean / Rotated / Noisy test images
```

---

## Q1. Data Setup
- Loaded BloodMNIST at size 64 and kept only classes 1, 5, 6 and 7.
- Randomly selected 100 / 30 / 30 images per class from the train / validation / test splits using seed 53.
- Converted the images to grayscale and resized them to 128 × 128.

| Split | Shape | Images per class |
|---|---|---|
| Train | (400, 128, 128) | 100 |
| Validation | (120, 128, 128) | 30 |
| Test | (120, 128, 128) | 30 |

---

## Q2. HOG + SVM

### HOG feature length (hand calculation)
- Cells per side = 128 / 16 = **8**
- Blocks per side = 8 − 2 + 1 = **7** → 7 × 7 = **49 blocks**
- Values per block = 2 × 2 cells × 9 bins = **36**
- **HOG length = 49 × 36 = 1764**

### Results (validation)
| C | Validation Accuracy |
|---|---|
| 0.1 | 0.733 |
| 1 | 0.733 |
| 10 | 0.733 |

**Best C = 0.1** → Validation accuracy **0.733**, Macro-F1 **0.732**

> All C values gave the same accuracy because 1764 features with only 400 images are already linearly separable. On a tie, the smallest C (simplest model) was kept.

---

## Q3. SIFT + SVM

### Normal SIFT detection (training images)
| Statistic | Value |
|---|---|
| Average keypoints per image | 72.6 |
| Minimum | 4 |
| Maximum | 213 |
| Images with zero keypoints | 0 |

**Problem:** an SVM needs a fixed-length vector for every image, but normal SIFT gives a different number of 128-D descriptors per image.

### Fix: fixed grid of keypoints
- A 4 × 4 grid over the 128 × 128 image → keypoints at x, y = **16, 48, 80, 112**, size 32
- `sift.compute` gives a 128-D descriptor at each point, and the descriptors are joined into one vector
- **SIFT length = 16 × 128 = 2048**

### Results (validation)
| C | Validation Accuracy |
|---|---|
| 0.1 | 0.758 |
| 1 | 0.750 |
| 10 | 0.750 |

**Best C = 0.1** → Validation accuracy **0.758**, Macro-F1 **0.760**

---

## Q4. Robustness

### Prediction (before running)
- **Rotation:** HOG would lose more, because its fixed cells change when edges move and rotate.
- **Noise:** HOG would lose more, because noise creates false gradients inside its cells, while SIFT averages over larger 32-pixel patches.

### Test results (frozen models)
| Model | Clean | Rotated (20°) | Noisy (σ = 0.05) |
|---|---|---|---|
| HOG + SVM | 0.658 | **0.733** | 0.275 |
| SIFT + SVM | **0.700** | 0.708 | **0.692** |

### Was the prediction correct?
- **Noise:**  Correct. HOG fell from 0.658 to 0.275 (close to the 0.25 chance level for 4 classes), while SIFT stayed almost unchanged (0.700 → 0.692).
- **Rotation:**  Wrong. Neither model lost accuracy. HOG actually rose from 0.658 to 0.733, but with only 120 test images (one image = 0.8%), this is most likely random variation rather than rotation genuinely helping.

---

## Q5. Short Answers

**Why does HOG use block normalisation?**  
Gradient strength changes with lighting and contrast. Normalising each 2 × 2 block makes HOG describe edge shape and direction instead of absolute brightness.

**Why is it wrong to choose C using the test set?**  
The test set would then influence the model, so the test accuracy would be optimistically biased and no longer an honest measure on unseen images.

**Most confused classes:** **eosinophil and monocyte** (HOG: 16 mistakes, SIFT: 17).  
Both are large, round white blood cells with a big nucleus. They differ mainly in colour (red-orange granules vs grey-blue cytoplasm), and grayscale removes this colour information.

---

## Conclusion
**SIFT + SVM was the more reliable model.** It had higher accuracy on clean and noisy test images and stayed almost the same in all three cases, while HOG + SVM collapsed to near chance level under noise.

---


## Libraries
| Library | Used for |
|---|---|
| `medmnist` | Loading the BloodMNIST dataset |
| `numpy` | Arrays and maths |
| `opencv-python` (`cv2`) | Grayscale, resize, rotation, SIFT |
| `scikit-image` | HOG features and visualisation |
| `scikit-learn` | LinearSVC, StandardScaler, metrics |
| `matplotlib` | Plots and image grids |
