# Lab 02 — Effect of Image Filtering on Skin-Lesion Classification

## What this is
This lab tests how five classical spatial filters — Average, Gaussian,
Median, Sharpening, and Sobel — change the classification performance of
three pretrained CNNs on the **HAM10000** skin-lesion dataset, relative to
an unfiltered baseline.

**Models used (the three carried over from Lab Activity 1 / Task 01):**
- VGG16
- ResNet18
- EfficientNet-B0

Each model is loaded with ImageNet-pretrained weights; the convolutional
backbone stays frozen and only the final classification layer is
fine-tuned.

## Folder layout
```
Lab 02/
├── README.md                 # this file
├── answers.md                 # write-up + filled-in results table
└── ICV_BAI_032_LAB2.ipynb    # the experiment notebook
```

## Requirements
Google Colab is the easiest way to run this (free GPU, and `kagglehub`
caches the dataset). To run locally instead:

```bash
pip install torch torchvision timm opencv-python kagglehub pandas scikit-learn tqdm matplotlib
```

The first `kagglehub` download of `kmader/skin-cancer-mnist-ham10000`
requires Kaggle credentials — Colab's built-in Kaggle integration handles
this automatically; locally, drop `kaggle.json` into `~/.kaggle/`.

## Running the notebook
Just execute the cells in order, top to bottom:

1. **Dataset prep** — pulls HAM10000 through `kagglehub`, maps each
   `image_id` to its file path, converts the 7 diagnosis labels
   (`bkl, nv, df, mel, vasc, bcc, akiec`) to integers, and builds an 80/20
   stratified train/test split (`random_state=42`).
2. **Filters** — implements the five OpenCV filters plus a "No Filter"
   pass-through, and shows one image processed through all six for a quick
   visual check.
3. **Dataset & model setup** — `FilteredHAMDataset` applies the chosen
   filter at load time, resizes to 224×224, and normalizes with standard
   ImageNet stats. `get_model()` builds VGG16 / ResNet18 / EfficientNet-B0
   with the backbone frozen and a fresh classification head.
4. **Training loop** — cycles through 6 filters × 3 models (18 runs), each
   trained for 2 epochs with AdamW (`lr=1e-3`) on the head only, using
   cross-entropy loss.
5. **Evaluation** — on the test split, computes Accuracy, macro
   Precision/Recall/F1, and macro one-vs-rest AUC for each run.
6. **Comparison** — all 18 results land in one summary dataframe
   (`lab2_summary_df`), which is reproduced in `answers.md`.

No CLI arguments or config files — the notebook is self-contained; run it
start to finish.

## Known limitations
- Each run trains for only **2 epochs** with a **frozen backbone**, which
  keeps 18 separate runs feasible in a lab session but means the raw
  accuracy numbers are well below what full fine-tuning would give.
- Only **macro-averaged** Precision/Recall/F1 are reported, so the
  "F1-score" and "Macro-F1" columns in the results table are the same
  number; **balanced accuracy, per-class precision/recall/F1, confusion
  matrices, and per-epoch train/val accuracy & loss curves are not yet
  produced**. Adding these (e.g. via `sklearn.metrics
  .classification_report`, `confusion_matrix`, and logging per-epoch
  train/val accuracy) is needed to fully cover Task 02's "Additional
  Analysis" section.
