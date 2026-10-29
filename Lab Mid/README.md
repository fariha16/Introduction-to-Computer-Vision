# Lab Mid Exam — HOG vs SIFT on Blood-Cell Images

**Name:** Fariha  |  **Registration no.:** FA23-BAI-053  |  **Course:** Computer Vision Lab

| Personal setting | Rule | My value |
|---|---|---|
| Random seed | last two digits of 053 | **53** |
| HOG cell size | last digit 3 is odd | **16 × 16** |
| SIFT grid size | last digit 3 is in 0–4 | **4 × 4 (16 keypoints)** |

**Data source:** official `medmnist` BloodMNIST, native size **64 × 64**. Classes: eosinophil (1), monocyte (5), neutrophil (6), platelet (7).
**Rule followed:** C was chosen on the validation set only. The test set was used only in Q4.
The notebook is `FA23_BAI_053_Fariha_LabMid.ipynb` (all outputs visible); a PDF copy is submitted as well.

---

## Q1. Data setup

The subset was chosen randomly with `np.random.default_rng(53)`, without replacement, from the official train / val / test splits.

After preparation (grayscale, resize to 128 × 128) the splits have these shapes and class counts:

| Split | X shape | eosinophil | monocyte | neutrophil | platelet |
|---|---|---|---|---|---|
| train | (400, 128, 128) | 100 | 100 | 100 | 100 |
| validation | (120, 128, 128) | 30 | 30 | 30 | 30 |
| test | (120, 128, 128) | 30 | 30 | 30 | 30 |

**4 example images per class** (rows: eosinophil, monocyte, neutrophil, platelet):

![Q1 grid](images/cell05_1.png)

---

## Q2. HOG + SVM

### (a) Feature length

HOG settings: 9 orientations, **16 × 16** pixels per cell, 2 × 2 cells per block, L2-Hys block normalisation.

**Feature length printed by the code: 1764**

### (b) HOG picture for one image per class

Top row: original image. Bottom row: HOG picture (eosinophil, monocyte, neutrophil, platelet).

![HOG pictures](images/cell11_1.png)

### (c) LinearSVC — C chosen on the validation set

| C | validation accuracy |
|---|---|
| **0.1** (chosen) | **0.733** |
| 1 | 0.733 |
| 10 | 0.733 |

All three values of C gave the same validation accuracy, so the first one with the best score (C = 0.1) was kept.

**Chosen C = 0.1. Validation accuracy = 73.3 %, validation macro-F1 = 73.2 %.**


![HOG confusion matrix](images/cell13_1.png)

---

## Q3. SIFT + SVM

### (a) Normal SIFT detection on the 400 training images

`cv2.SIFT_create().detectAndCompute(image, None)`:

| Keypoints per training image | Value |
|---|---|
| average | **72.6** |
| minimum | **4** |
| maximum | **213** |
| images with zero keypoints | **0** |

Keypoints drawn on one training image:

![SIFT keypoints](images/cell16_1.png)

**Why is a different number of keypoints per image a problem for an SVM?**
An SVM needs every image to be one vector of the same fixed length, but normal SIFT gives a different number of 128-D descriptors per image (here 4 to 213), so the images cannot be stacked into one feature matrix.

### (b) Fix: SIFT descriptors on a fixed 4 × 4 grid

Keypoints are placed on a regular 4 × 4 grid over the 128 × 128 image (centres at 16, 48, 80, 112 in x and y), keypoint size 32. The descriptor is computed at every grid point with `sift.compute` and the 16 descriptors are joined into one vector, so every image has the same feature length.

**Feature length printed by the code: 2048** (16 grid points printed)

### (c) LinearSVC — C chosen on the validation set

| C | validation accuracy |
|---|---|
| **0.1** (chosen) | **0.758** |
| 1 | 0.750 |
| 10 | 0.750 |

**Chosen C = 0.1. Validation accuracy = 75.8 %, validation macro-F1 = 76.0 %.**

Confusion matrix (validation set, rows = true class):

| | pred eosinophil | pred monocyte | pred neutrophil | pred platelet |
|---|---|---|---|---|
| **true eosinophil** | 18 | 10 | 2 | 0 |
| **true monocyte** | 7 | 23 | 0 | 0 |
| **true neutrophil** | 7 | 0 | 21 | 2 |
| **true platelet** | 0 | 0 | 1 | 29 |

![SIFT confusion matrix](images/cell22_1.png)

---

## Q4. Robustness

### (a) Prediction written before running the robustness test

- **Rotation by 20°:** I expect **HOG** to lose more accuracy, because its cells are small and fixed on the image, so rotating the image moves edges into different cells and changes their gradient directions; the grid-SIFT patches are larger (32 px) with Gaussian weighting, so they tolerate small changes better.
- **Gaussian noise (σ = 0.05):** I expect **HOG** to lose more accuracy, because small cells compute gradients from only a few pixels and noise creates many false gradients, while each SIFT descriptor pools gradients over a much larger patch, which averages the noise out.

### (b) Frozen models evaluated on the test set (120 images, 1 image = 0.83 percentage points)

Both models are frozen (nothing is re-trained). The change is applied to the prepared 128 × 128 gray images **before feature extraction**, and both models receive exactly the same changed images.
Rotation: 20° about the image centre (`cv2.warpAffine`, reflected borders). Noise: Gaussian, σ = 0.05 on the image scaled to 0–1 (seeded with 53).

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.658 (65.8 %) | 0.733 (73.3 %) | 0.275 (27.5 %) |
| SIFT + SVM | 0.700 (70.0 %) | 0.708 (70.8 %) | 0.692 (69.2 %) |

| Model | Change under rotation (points) | Change under noise (points) |
|---|---|---|
| HOG + SVM | +7.5 (9 images) | −38.3 |
| SIFT + SVM | +0.8 (1 image) | −0.8 (1 image) |

Example images (clean, rotated, noisy):

![Clean, rotated, noisy](images/cell26_1.png)

### (c) Was my prediction correct?

1. For noise my prediction was correct: HOG + SVM fell from 0.658 to 0.275, close to the 0.25 chance level for 4 classes, while SIFT + SVM stayed almost unchanged (0.700 → 0.692), because each 32-pixel SIFT patch averages the noise out.
2. For rotation my prediction was wrong, since neither model lost accuracy: HOG rose from 0.658 to 0.733 and SIFT from 0.700 to 0.708.
3. With only 120 test images, one image is 0.83 points, so the 9-image change for HOG is most likely random variation rather than rotation genuinely helping HOG.

---

## Q5. Short answers and AI log

### (a) Why does HOG use block normalisation?
Gradient magnitudes change with lighting and contrast, so the same cell can look very different in a bright or dark image. Normalising each 2 × 2 block of cell histograms (L2-Hys) makes the feature depend on edge *shape and direction* rather than absolute brightness, and overlapping blocks make it more stable.

### (b) Why is it wrong to choose C using the test set?
If the test set is used to pick C, the test set has influenced the model, so the test accuracy becomes optimistically biased and is no longer an honest estimate of performance on unseen images. The validation set is for tuning; the test set is used only once, at the end.

### (c) Which two classes were confused most?

The two classes confused most were **eosinophil and monocyte** (33 errors in total: 16 for HOG and 17 for SIFT). Eosinophil is the hardest class (recall on validation: HOG 17/30 = 56.7 %, SIFT 18/30 = 60.0 %), while platelet is almost never confused.

**Why grayscale makes this harder:** eosinophils and monocytes are both large, round white blood cells with a big nucleus, so their outline and size look similar. They differ mostly in **colour**: eosinophils have bright red-orange granules, while monocytes have a smooth grey-blue cytoplasm. Grayscale removes this colour cue, so the models only see shape and texture, which look much more alike.

### (d) AI log

| Prompt (one line) | What I changed | How I checked the result was correct |
|---|---|---|
| "Solve my HOG vs SIFT lab mid on BloodMNIST and explain the code for viva" (Claude) | Set `REG_NO = "053"` (seed 53, HOG cell 16 × 16, SIFT grid 4 × 4); filled the Q4 table, Q4 conclusion and Q5 pair from my own outputs | Printed shapes (400/120/120, each 128 × 128) and counts (100/30/30 per class); HOG length 1764 matched 7 × 7 × 36; SIFT length 2048 matched 16 × 128; looked at the image grid, HOG pictures, SIFT keypoints and the rotated/noisy examples |
