# Lab 5 — HOG-Based Industrial Defect Detection & Classification

**Course:** Introduction to Computer Vision (ICV) · **Notebook:** `ICV_BAI_032_LAB5.ipynb` (Google Colab, dataset downloads automatically)

Classical computer-vision pipeline for industrial quality control on the **NEU Surface Defect Database (NEU-DET)**:

```
Image → Preprocessing → HOG features → SVM → ACCEPT / REJECT (+ defect type)
```

---

## Problem formulation

NEU-DET has six defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) and **no defect-free class**: every image contains at least one defect. Calling one defect class "normal" would give a meaningless binary task. Instead, the Pascal-VOC bounding boxes are used to build a proper binary dataset at patch level:

| Class | Source |
|---|---|
| **Defective** | 64×64 windows centred on annotated defect boxes |
| **Non-defective** | 64×64 windows with zero overlap with any box (4-px safety margin) |

A full product is inspected by sliding the patch classifier over the surface (stride 32). The product is rejected if any window passes the decision threshold. A separate **6-class model** on full 128×128 images names the defect type.

Patches keep the train/valid split of their parent image. Cross-validation is grouped by image, so patches from one image never appear in both the train and test folds.

## Results (held-out validation split)

### HOG parameter study (5-fold group CV, training set only)

| Task | Best cell size | Best orientations | CV macro-F1 |
|---|---|---|---|
| Binary (patches) | 8×8 | 6 | 0.726 |
| Multi-class (full images) | 16×16 | 6 | 0.726 |

Coarser cells won on both tasks. In the multi-class task, 4×4 cells (up to 46k dimensions) scored about 0.09–0.16 F1 lower than 16×16, so small cells mostly add noise to texture descriptors.

![HOG parameter heatmap](figures/07_hog_param_heatmap.png)

### Classifier comparison

**Binary: defective vs non-defective patches.** Precision, recall and F1 are reported for the defective class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ms / patch |
|---|---|---|---|---|---|---|
| **SVM (RBF)** | **0.805** | 0.827 | 0.876 | **0.851** | 0.866 | 8.2 |
| Random Forest | 0.795 | 0.800 | 0.903 | 0.848 | 0.872 | 2.3 |
| Logistic Regression | 0.760 | 0.887 | 0.715 | 0.792 | 0.846 | 1.9 |
| k-NN (k=5) | 0.500 | 0.870 | 0.253 | 0.392 | 0.707 | 3.0 |

**Multi-class: 6 defect types.** Scores are macro-averaged.

| Model | Accuracy | Precision | Recall | F1 | ms / image |
|---|---|---|---|---|---|
| **SVM (RBF)** | **0.910** | 0.911 | 0.911 | **0.909** | 6.0 |
| Random Forest | 0.880 | 0.892 | 0.879 | 0.881 | 5.0 |
| Logistic Regression | 0.850 | 0.854 | 0.850 | 0.849 | 4.1 |
| k-NN (k=5) | 0.740 | 0.820 | 0.732 | 0.687 | 4.4 |

![Confusion matrices – multi-class](figures/09_confusion_multiclass.png)

### Robustness (HOG + SVM)

| Condition | Binary F1 (clean 0.851) | Multi-class accuracy (clean 0.91) |
|---|---|---|
| Brightness ±40 | 0.838 – 0.851 | 0.81 – 0.91 |
| Brightness ±80 | 0.820 – 0.845 | 0.65 – 0.71 |
| Gaussian noise σ=10 | 0.529 | 0.30 |
| Rotation 15° / 90° | 0.848 / 0.835 | 0.76 / 0.35 |
| Gaussian blur σ=1 | 0.791 | 0.41 |

HOG's block normalisation makes it largely robust to illumination changes. **Noise and blur are the failure modes**: both destroy the fine gradient structure that HOG encodes. Rotation barely affects the binary task but hurts defect-type recognition, because orientation is part of what separates classes such as scratches. A degradation-aware retraining experiment is in `figures/13_robustness_mitigation.png`.

![Robustness – multi-class](figures/12_robustness_multiclass.png)

### Quality-control decision module

| Metric | Value |
|---|---|
| Defective products correctly rejected | 100 % |
| Defect-type accuracy on rejected products | 91 % |
| Clean surface samples wrongly rejected | 54.7 % |
| Latency per 200×200 product (single CPU core) | 321 ms (≈ 3 products/s) |

The decision threshold was chosen on training data to catch at least 95 % of defective patches. This favours never missing a defect, at the cost of a high false-reject rate on clean samples. That trade-off is the main limitation of a hand-crafted HOG pipeline on these textured steel surfaces. In practice it would be tuned to the line's cost of a missed defect versus a scrapped good part.

Output format for each product:

```
==================================
PRODUCT INSPECTION RESULT
==================================
Prediction: DEFECTIVE | NON-DEFECTIVE
Confidence: <max window probability>
Action: REJECT PRODUCT | ACCEPT PRODUCT
Defect type: <one of 6 classes, if rejected>
```

## Files

```
ICV_BAI_032_LAB5.ipynb   full pipeline (Colab-ready, runs top to bottom)
realtime_inspector.py    real-time webcam / video inspector (bonus)
conveyor_demo.mp4        simulated conveyor-belt inspection stream (bonus)
figures/                 all figures (dataset, HOG, CV study, confusion, robustness)
results/                 CSV results + summary.json
```

## Reproduce

1. Open `ICV_BAI_032_LAB5.ipynb` in Google Colab with a **CPU runtime** (the pipeline is scikit-learn / scikit-image, so a GPU gives no speed-up).
2. Run all cells. The dataset downloads automatically through `kagglehub` from
   [NEU Steel Surface Defect – train/valid split](https://www.kaggle.com/datasets/sovitrath/neu-steel-surface-defect-detect-trainvalid-split).
   If prompted, add `KAGGLE_USERNAME` / `KAGGLE_KEY` in Colab Secrets.
3. Full run: about 2 hours on free Colab. The HOG parameter study and SVM grid search take most of that time.

To run the real-time inspector locally, export `steel_inspector.joblib` from the notebook (not committed because of its size) and run:

```bash
pip install opencv-python scikit-image scikit-learn joblib
python realtime_inspector.py          # webcam
python realtime_inspector.py clip.mp4 # video file
```

## Dataset

K. Song and Y. Yan, *"A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,"* Applied Surface Science, 2013. The dataset is not redistributed here.

