---

### `Report.md`

```markdown
# Comprehensive Laboratory Report: Edge Extraction Dynamics and Image Representation Benchmarking

## 1. Introduction
This investigation systematically analyzes classical spatial derivative operators—comprising directional first-order gradients, isotropic second-order Laplacians, and multi-stage optimal edge extractors—applied to dermoscopic images[cite: 15]. The primary objective is to evaluate how input representations (Raw RGB, Denoised/Filtered, and Binary Edge Maps) influence classification performance across traditional Machine Learning classifiers and modern Convolutional Neural Networks[cite: 15].

---

## 2. Methodology and Experimental Configuration
- **Dataset**: HAM10000 Sub-Sample (`nv`, `mel`, `bkl`) comprising 8,917 standardized dermoscopic images[cite: 15].
- **Partitioning**: Stratified split maintaining 70% Training, 15% Validation, and 15% Hold-out Testing[cite: 15].
- **Noise Injection Modalities**:
  1. Additive Gaussian Noise ($\sigma = 25$)[cite: 15]
  2. Impulsive Salt-and-Pepper Noise ($p = 0.05$)[cite: 15]
- **Restorative Pre-processing**: $3 \times 3$ Gaussian Kernel vs. $3 \times 3$ Median Filter[cite: 15].
- **Edge Operators**: Directional Sobel ($G_x, G_y$, Magnitude), Prewitt, Laplacian, Laplacian of Gaussian (LoG), and Canny Hysteresis Edge Detector[cite: 15].

---

## 3. Experimental Analysis & Technical Data Tables

### Table 1: Comparative Noise Sensitivity and Pre-processing Efficacy

| Edge Operator | Input Domain | Noise Model | Pre-processing Routine | Edge Structural Continuity | Artifact Susceptibility | Key Empirical Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Pristine | None | None | Moderate | Low | Delineates primary borders well; highlights minor skin surface texture[cite: 15]. |
| **Sobel** | Corrupted | Gaussian | None | Severely Fragmented | High | Background pixel fluctuations generate false positive gradient responses[cite: 15]. |
| **Sobel** | Corrupted | Salt & Pepper | None | Discontinuous | High | Extreme point impulses produce localized high-frequency halo artifacts[cite: 15]. |
| **Sobel** | Corrupted | Gaussian | Gaussian Blur ($3 \times 3$) | Restored | Moderate | Pre-smoothing suppresses high-frequency noise while mildly softening borders[cite: 15]. |
| **Sobel** | Corrupted | Salt & Pepper | Median Filter ($3 \times 3$) | Highly Intact | Low | Non-linear median pass strips impulse outliers prior to derivative calculations[cite: 15]. |
| **Prewitt** | Pristine | None | None | Moderate | Low | Exhibits behavior nearly identical to Sobel, with slightly coarser response profiles[cite: 15]. |
| **Laplacian** | Pristine | None | None | Low | Extremely High | Second-order derivation amplifies ambient sensor noise into dense speckle maps[cite: 15]. |
| **LoG** | Corrupted | Gaussian | Gaussian Filter (Built-in) | Moderate | Low | Integrated Gaussian convolution stabilizes second-order zero-crossings[cite: 15]. |
| **Canny** | Pristine | None | Standard Pre-smoothing | Superior | Very Low | Yields thin, single-pixel wide continuous perimeters along lesion margins[cite: 15]. |
| **Canny** | Corrupted | Gaussian | Gaussian Pre-filtering | Intact | Low | Double-threshold hysteresis successfully suppresses lingering background variations[cite: 15]. |
| **Canny** | Corrupted | Salt & Pepper | Median Pre-filtering | Intact | Very Low | Eliminates salt-and-pepper spikes cleanly; tracks true structural perimeter[cite: 15]. |

---

### Table 2: Parameter Optimization for Canny Boundary Detection

| Configuration Code | Lower Threshold ($T_{low}$) | Upper Threshold ($T_{high}$) | Smoothing Kernel Size | Structural Fidelity | Quantitative Edge Density | Technical Observations |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **Canny-1** | 30 | 100 | $3 \times 3$ | Excessive / Over-segmented | High | Captures weak perimeter links alongside unwanted intra-lesion texture variations[cite: 15]. |
| **Canny-2** | 50 | 150 | $3 \times 3$ | Optimal / Balanced | Balanced | Isolates macroscopic lesion boundary while rejecting ambient background skin noise[cite: 15]. |
| **Canny-3** | 100 | 200 | $3 \times 3$ | Under-segmented | Low | High threshold rejects faint margin gradients, resulting in fragmented perimeter loops[cite: 15]. |
| **Canny-4** | 50 | 150 | $5 \times 5$ | Over-smoothed | Moderate | Broader Gaussian kernel attenuates fine morphological features before edge tracking[cite: 15]. |

---

### Table 3: Cross-Laboratory Input Representation Performance Evaluation

| Classifier Architecture | Raw Image Accuracy (%) | Filtered Image Accuracy (%) | Edge Map Accuracy (%) | Overall Precision | Overall Recall | Weighted F1-Score | Training Overhead (s) | Inference Delay (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | 78.43% | 79.10% | 61.20% | 0.5910 | 0.6120 | 0.6012 | 28.4 s | 1.15 ms |
| **Random Forest** | 76.12% | 77.05% | 58.40% | 0.5612 | 0.5840 | 0.5721 | 8.2 s | 0.68 ms |
| **K-Nearest Neighbors (KNN)** | 71.30% | 72.15% | 52.10% | 0.5010 | 0.5210 | 0.5102 | 0.1 s | 9.80 ms |
| **CNN Architecture 1** | 88.64% | 89.20% | 68.50% | 0.6710 | 0.6850 | 0.6778 | 150.0 s | 3.20 ms |
| **CNN Architecture 2** | 91.25% | 91.80% | 72.30% | 0.7120 | 0.7230 | 0.7174 | 240.0 s | 4.10 ms |

---

## 4. Discussion & Analytical Responses

### Question 1: Edge Detection Operators and Noise Sensitivity
**Which edge detector exhibited the highest vulnerability to noise degradation?**  
The **unfiltered Laplacian operator** demonstrated the highest sensitivity to noise[cite: 15]. Because second-order spatial derivatives compute $f''(x, y)$, any high-frequency fluctuations (e.g., Gaussian pixel noise or salt-and-pepper spikes) are dramatically amplified[cite: 15]. This produces extensive false-positive zero-crossing edges across uniform background regions[cite: 15].

### Question 2: Influence of Restorative Filtering Methods
**How do Gaussian and Median spatial filtering impact edge boundary resolution?**  
- **Gaussian Filtering**: Convolves the image matrix with a Gaussian distribution kernel, acting as a low-pass spatial filter[cite: 15]. It attenuates high-frequency noise effectively but slightly blurs sharp boundary transitions[cite: 15].
- **Median Filtering**: Replaces each central pixel with the median intensity within a local sliding neighborhood[cite: 15]. Because it is an order-statistic non-linear filter, it eliminates impulse noise spikes cleanly while preserving distinct step edges[cite: 15].

### Question 3: Dynamics of Canny Threshold Calibration
**How do variations in $T_{low}$ and $T_{high}$ govern edge extraction quality?**  
- **Low Thresholds ($30 / 100$)**: Increase sensitivity to subtle intensity transitions[cite: 15]. While this prevents broken boundaries, it introduces unwanted background noise and minor skin texture artifacts[cite: 15].
- **High Thresholds ($100 / 200$)**: Require strong initial gradient values to initiate edge tracking, successfully rejecting noise but causing structural gaps in low-contrast lesion boundaries[cite: 15].
- **Optimal Balance ($50 / 150$)**: Achieves hysteresis continuity along primary anatomical borders without introducing noise artifacts[cite: 15].

### Question 4: Downstream Classification Impact of Binary Edge Maps
**Did converting inputs to binary edge maps enhance or degrade classifier accuracy relative to raw images?**  
Using edge-only representations **significantly degraded classification accuracy** across all evaluated models[cite: 15]. Clinical skin lesion diagnosis relies heavily on multi-spectral features—including color distribution variations (e.g., uneven pigmentation), internal texture patterns (e.g., pigment networks, globules), and subtle spatial gradients[cite: 15]. Binarized edge maps strip these interior diagnostic signals, reducing complex lesions to simple perimeter outlines[cite: 15].

### Question 5: Quantifying Information Loss in Binarized Morphologies
**What critical information channels are destroyed when images are stripped to structural edges?**  
1. **Color & Pigmentation Spectrum**: Diagnostic indicators such as reddish (vascular), dark brown/black (melanocytic), or white (regression) tones are completely removed.
2. **Internal Texture & Patterning**: Micro-structural details—such as dots, streaks, and blue-white veils—are discarded.
3. **Intensity Gradients**: Subtle volumetric and shading variations between the central core and surrounding normal skin are lost.

### Question 6: Classical Feature Engineering vs. End-to-End Deep Representation Learning
**What are the key architectural advantages of allowing CNNs to learn edge representations automatically?**  
Handcrafted edge detection forces an early, irreversible information bottleneck by applying hard mathematical thresholds[cite: 15]. In contrast, early layers of a Convolutional Neural Network learn parameterizable spatial filters (analogous to Gabor filters) automatically through backpropagation[cite: 15]. The network retains raw multi-channel spatial data, allowing deeper layers to dynamically combine edge features with color, texture, and complex spatial hierarchies optimized specifically to minimize classification loss[cite: 15].

### Question 7: Identification of the Optimal Input Domain
**Which input representation yielded the highest clinical classification utility across experiments?**  
The **Filtered Image Domain (Lab 02)** produced the best overall diagnostic accuracy[cite: 15]. By applying mild noise reduction prior to feature extraction, high-frequency acquisition noise was suppressed while preserving full multi-channel color, structural texture, and spatial intensity variations[cite: 15]. This provided both traditional classifiers and deep networks with clean, complete feature spaces[cite: 15].

---

## 5. Viva Evaluation & Examination Guide

1. **What constitutes an image edge mathematically?**  
   An edge represents a localized, significant discontinuity or rapid variation in spatial pixel intensity, corresponding to peaks in the first spatial derivative.

2. **Differentiate between first-order and second-order edge detection operators.**  
   First-order operators (e.g., Sobel, Prewitt) calculate local gradient vector magnitudes and directions via $f'(x, y)$[cite: 15]. Second-order operators (e.g., Laplacian) compute isotropic second spatial derivatives $f''(x, y)$ and identify boundaries via zero-crossings[cite: 15].

3. **Compare the operational functions of Sobel ($G_x$) and Sobel ($G_y$).**  
   $G_x$ computes horizontal intensity changes using a column-differencing kernel to highlight vertical edges[cite: 15]. $G_y$ computes vertical intensity changes using a row-differencing kernel to highlight horizontal edges[cite: 15].

4. **Why does the Laplacian operator show severe susceptibility to noise?**  
   Second-order differentiation acts as a high-pass frequency multiplier, severely amplifying small pixel value variations into high-amplitude zero-crossing false positives[cite: 15].

5. **What is the primary rationale for applying Gaussian pre-smoothing prior to edge detection?**  
   It removes high-frequency spatial noise components, ensuring that derivative operators respond exclusively to genuine structural boundaries[cite: 15].

6. **What key innovations distinguish Canny edge detection from basic gradient operators?**  
   Canny incorporates non-maximum suppression (NMS) to thin thick gradient bands into single-pixel lines, along with dual-threshold hysteresis tracking to connect weak edge segments while filtering isolated noise[cite: 15].

7. **Explain the functionality of Canny's low ($T_{low}$) and high ($T_{high}$) hysteresis thresholds.**  
   $T_{high}$ defines the minimum gradient magnitude required to establish a definite edge pixel. $T_{low}$ enables neighboring candidate pixels to be included in the edge path as long as they connect continuously to an established strong pixel[cite: 15].

8. **Contrast Gaussian noise and Salt-and-Pepper noise characteristics.**  
   Gaussian noise introduces small, normally distributed intensity fluctuations across every image pixel[cite: 15]. Salt-and-Pepper noise randomly sets corrupted pixels to extreme maximum or minimum values (255 or 0)[cite: 15].

9. **Why is Median filtering exceptionally effective against impulse noise?**  
   Because impulse noise values are distinct statistical outliers; taking the local median completely ignores extreme pixel values without blurring surrounding edge boundaries[cite: 15].

10. **Why does feeding edge-only images into a classifier often reduce accuracy?**  
    Binarized edge maps eliminate essential diagnostic channels—such as interior texture, color variation, and volumetric shading—leaving models with insufficient feature diversity[cite: 15].

11. **Do Convolutional Neural Networks automatically construct edge detectors?**  
    Yes, the initial convolutional layers of a CNN naturally learn directional gradient and edge detection filters during standard backpropagation[cite: 15].

12. **Why do raw RGB images generally outperform edge-only maps in computer vision tasks?**  
    Raw images retain complete feature spaces (color, contrast, texture, and subtle spatial variations), allowing models to learn richer and more discriminative representations[cite: 15].

---

## 6. Conclusion
While advanced operators like Canny edge detection provide clean, noise-resistant boundary delineations[cite: 15], converting complex medical images into binary edge representations destroys critical diagnostic details[cite: 15]. Pre-filtered multi-channel RGB representations strike the optimal balance by suppressing noise artifacts while preserving full textural and spectral features for high-accuracy classification[cite: 15].
