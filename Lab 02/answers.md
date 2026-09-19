# Lab 02 — Answers

## Experimental setup
- **Dataset:** full HAM10000 set (10,015 images, 7 classes: `bkl, nv, df, mel, vasc, bcc, akiec`), split 80/20 with stratification.
- **Models (the best 3 from Lab Activity 1):**
  - Model 1: **VGG16**
  - Model 2: **ResNet18**
  - Model 3: **EfficientNet-B0**
- **Training:** pretrained backbone frozen, only the classifier head fine-tuned, 2 epochs, AdamW (lr=1e-3); split, preprocessing, and evaluation kept identical across every run.

> The notebook's evaluation only computes macro-averaged Precision/Recall/F1, so **F1-score and Macro-F1 are the same value** in the table below. Balanced accuracy, per-class scores, and confusion matrices weren't produced by this version of the notebook (see the README's limitations section), so those cells are left blank.

## Required Experimental Results

| Model | Filter | Accuracy | Precision | Recall | F1-score | Macro-F1 | AUC |
|---|---|---|---|---|---|---|---|
| VGG16 | No Filter | 74.14 | 56.49 | 34.97 | 38.77 | 38.77 | 90.13 |
| VGG16 | Average | 77.68 | 52.59 | 47.06 | 49.02 | 49.02 | 90.93 |
| VGG16 | Gaussian | 75.74 | 51.25 | 45.15 | 44.57 | 44.57 | 90.71 |
| VGG16 | Median | 75.44 | 52.80 | 44.79 | 44.31 | 44.31 | 89.69 |
| VGG16 | Sharpening | 74.49 | 54.76 | 49.85 | 48.46 | 48.46 | 90.99 |
| VGG16 | Sobel | 66.55 | 36.58 | 29.18 | 29.08 | 29.08 | 80.51 |
| ResNet18 | No Filter | 72.94 | 53.22 | 36.25 | 38.00 | 38.00 | 90.90 |
| ResNet18 | Average | 72.84 | 53.58 | 34.37 | 35.83 | 35.83 | 89.82 |
| ResNet18 | Gaussian | 73.84 | 57.04 | 31.96 | 35.52 | 35.52 | 90.60 |
| ResNet18 | Median | 73.84 | 56.84 | 34.87 | 39.12 | 39.12 | 90.97 |
| ResNet18 | Sharpening | 73.94 | 56.14 | 33.72 | 36.81 | 36.81 | 90.40 |
| ResNet18 | Sobel | 68.40 | 33.04 | 22.58 | 24.28 | 24.28 | 79.01 |
| EfficientNet-B0 | No Filter | 74.49 | 48.61 | 40.95 | 43.59 | 43.59 | 90.19 |
| EfficientNet-B0 | Average | 75.69 | 56.57 | 39.99 | 44.12 | 44.12 | 89.52 |
| EfficientNet-B0 | Gaussian | 74.24 | 48.72 | 41.63 | 43.91 | 43.91 | 89.45 |
| EfficientNet-B0 | Median | 74.39 | 47.71 | 39.13 | 42.00 | 42.00 | 89.84 |
| EfficientNet-B0 | Sharpening | 74.94 | 48.77 | 38.11 | 41.63 | 41.63 | 89.51 |
| EfficientNet-B0 | Sobel | 70.54 | 33.22 | 26.88 | 28.52 | 28.52 | 79.85 |

*(all figures are percentages)*

## Questions to Answer

**Which three pretrained models performed best in Lab Activity 1?**
VGG16, ResNet18, and EfficientNet-B0 are the three models this lab builds on. (This assumes they match Task 01's top three — adjust this line if Task 01 ranked a different trio as best.)

**How does filtering affect each of the three models?**
- **VGG16** responds best to filtering overall: Average and Sharpening both push accuracy and macro-F1 clearly above baseline (Average reaches 77.68% accuracy / 49.02% F1 vs. a 74.14% / 38.77% baseline), and Gaussian/Median help too.
- **ResNet18** barely moves, or moves slightly negative, under the smoothing filters (Average, Gaussian); Median and Sharpening give it small F1 gains.
- **EfficientNet-B0** picks up modest, fairly steady gains from Average/Gaussian/Median/Sharpening, smaller than what VGG16 sees.
- Every model drops sharply under **Sobel**.

**Which filter produces the greatest change compared with the unfiltered baseline?**
**Sobel**, and it's not close — it's the only filter that hurts all three models on every metric (VGG16 falls from 74.14% to 66.55% accuracy and 90.13% to 80.51% AUC; ResNet18's macro-F1 nearly halves, 38.00% down to 24.28%). Among the filters that help rather than hurt, **Average** delivers the biggest gain, specifically for VGG16.

**Does the effect of a filter remain consistent across all three models?**
Not in size, though Sobel is consistently harmful across the board. Average, for instance, is VGG16's best filter (+3.5 accuracy points) but is roughly flat-to-slightly-negative for ResNet18 (-0.1 points) and moderately positive for EfficientNet-B0 (+1.2 points). So a filter's impact depends on the model as much as the filter itself.

**Does filtering improve or decrease macro-F1 and balanced accuracy?**
Macro-F1: the four gentler filters (Average, Gaussian, Median, Sharpening) raise it for VGG16 and EfficientNet-B0, are mixed-to-flat for ResNet18, and Sobel lowers it sharply for all three. Balanced accuracy isn't computed anywhere in the current notebook, so it can't be answered from these results — it would need `sklearn.metrics.balanced_accuracy_score` added to the evaluation step.

**Which lesion classes are most affected by filtering?**
The current results can't answer this — the notebook only reports macro-averaged scores, with no per-class breakdown or confusion matrix. That would require adding a per-filter/per-model `classification_report` (or confusion matrix) across the 7 classes. Given how few HAM10000 samples exist for `df`, `vasc`, and `akiec` compared to `nv`, those minority classes are the likeliest to swing the most once that analysis is added.

**Why might smoothing remove useful lesion texture or morphological information?**
Average, Gaussian, and Median are all low-pass filters — they average away high-frequency detail. In dermoscopic images, that high-frequency detail is often exactly where the diagnostic signal lives: pigment network patterns, dot/globule structure, border irregularity, fine vessels. Smoothing blurs those cues even as it strips out noise, so the net effect on a given model depends on which side of that trade-off dominates.

**Why might sharpening or edge detection help or hurt classification?**
Sharpening boosts existing high-frequency structure — edges, texture boundaries — which can help highlight lesion borders and surface texture, matching the gains it gave VGG16 and EfficientNet-B0 here. Sobel, by contrast, reduces the image to a gradient-magnitude edge map and (as implemented here) collapses that into grayscale before tiling it back across three channels — throwing away color entirely. Since color cues (erythema, pigmentation, blue-white veil, etc.) are central to dermoscopic diagnosis, that's a strong candidate explanation for why Sobel tanks performance across all three models.

**What is the difference between convolution and correlation?**
Cross-correlation slides a kernel across the image and takes a weighted sum using the kernel exactly as given. Convolution does the same thing but flips the kernel 180° first (both axes) before sliding it. When the kernel is symmetric — as with the Gaussian, box/mean, and the sharpening kernel used here — convolution and correlation produce identical results; for an asymmetric kernel like Sobel's, they give different, mirrored outputs. (Worth noting: most deep-learning libraries, PyTorch's `Conv2d` included, actually implement cross-correlation under the name "convolution.")

**Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**
At the operation level, classical filters and a CNN's convolutional layers are doing the same thing — sliding a kernel across the image and computing weighted sums. The difference is that classical filters are hand-designed for one fixed, interpretable effect (blurring, edge detection, sharpening), while a CNN backbone learns its own set of filters end-to-end, tuned to whatever separates the target classes. These results show the two aren't independent: a hand-crafted filter applied before a pretrained network changes what that network's already-learned filters get to see. Mild smoothing and sharpening left enough structure intact to be roughly neutral or mildly helpful, but an aggressive, information-discarding transform like Sobel — which strips out color — clashed with cues the pretrained (ImageNet) backbone relies on, producing a far larger performance hit than any other filter. In short, classical preprocessing isn't neutral input to a pretrained deep network; it can either complement or actively undercut what the network already learned to use.
