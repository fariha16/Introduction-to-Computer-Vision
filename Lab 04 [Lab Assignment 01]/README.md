# Skin Lesion Boundary Detection Using Canny Edge Detection

## Overview

This project implements a simple **computer vision pipeline for detecting the boundary of skin lesions** in dermoscopy images.

The main processing pipeline is:

**Original Image → Grayscale → Gaussian Filter → Canny Edge Detection → Lesion Boundary → Area & Perimeter**

The project uses images from the **HAM10000 skin-cancer dataset** obtained through Hugging Face.

---

## Dataset

The notebook streams **5 dermoscopy images** from the `marmal88/skin_cancer` dataset.

The selected images are from the **nevi** category because they generally contain dark lesions with relatively clear boundaries.

No manual image upload or Kaggle API key is required.

---

## Tasks Performed

### Task 1 — Load the Images

Five skin-lesion images are loaded from the HAM10000 dataset using Hugging Face.

The images are converted into a format suitable for OpenCV processing.

### Task 2 — Image Preprocessing

The images are converted to **grayscale** to simplify the edge-detection process.

Different smoothing filters are also considered:

* Original image
* Average filter
* Gaussian filter
* Median filter

Gaussian filtering is particularly important because it reduces noise before applying Canny edge detection.

### Task 3 — Edge Detection

The project uses edge-detection techniques to identify changes in intensity around the lesion.

The methods include:

* **Canny Edge Detection**
* **Sobel Edge Detection**

For Canny detection, different threshold settings are tested to observe how they affect the detected lesion boundary.

### Task 4 — Lesion Boundary Detection

The detected edges are processed to identify the approximate boundary of the skin lesion.

Morphological operations are used to help improve the detected boundary and reduce small unwanted regions.

### Task 5 — Area and Perimeter

The detected lesion boundary is used to calculate:

* **Lesion Area**
* **Lesion Perimeter**

These measurements provide basic quantitative information about the detected lesion.

### Task 6 — Filter and Edge Detector Comparison

The project compares:

**4 filtering options × 2 edge detectors**

The comparison considers:

* Noise handling
* Edge quality
* Boundary detection

Since ground-truth lesion masks are not available, a reference lesion mask is created using **inverse Otsu thresholding on the Gaussian-filtered image**.

---

## Evaluation Metrics

### 1. Noise Handling

The number of small isolated edge fragments is measured.

**Fewer isolated fragments = less noise.**

### 2. Edge Quality

This measures the fraction of detected edge pixels that are within **3 pixels of the reference boundary**.

**Higher value = better edge quality.**

### 3. Boundary Detection

The **Intersection over Union (IoU)** between the detected lesion mask and the reference mask is calculated.

**Higher IoU = greater overlap with the reference lesion region.**

---

## Questions and Answers

### 1. Why is Gaussian filtering applied before Canny detection?

Gaussian filtering is applied to **remove noise and smooth the image** before edge detection. It reduces small unwanted variations that could otherwise be detected as false edges by the Canny algorithm.

### 2. How did the three Canny threshold settings affect the result?

The three threshold settings produced different edge detection results:

* **Low thresholds:** Detected more edges, including unwanted noise.
* **Medium thresholds:** Detected the lesion boundary more clearly with less noise.
* **High thresholds:** Removed many weak edges, but some parts of the lesion boundary could also be lost.

### 3. Which threshold produced the best lesion boundary?

The **medium threshold setting** generally produced the best lesion boundary because it provided a balance between detecting the lesion edges and reducing unwanted edges.

> The actual best threshold should be selected according to the output images produced by the notebook.

### 4. Why are edges useful for detecting skin lesions?

Edges are useful because a skin lesion can have differences in **color, intensity, or texture** compared with the surrounding skin. Edge detection identifies these changes and helps locate the lesion boundary.

### 5. What problems were observed in detecting the lesion boundary?

Some problems include:

* Noise creating false edges.
* Hair and other skin details producing unwanted edges.
* Weak or broken parts of the lesion boundary.
* Uneven lighting affecting edge detection.
* Similar colors between the lesion and surrounding skin making the boundary difficult to detect.

### 6. How could the method be improved?

The method could be improved using:

* Better noise reduction.
* Hair removal.
* Contrast enhancement.
* Adaptive Canny threshold selection.
* Improved morphological processing.
* More advanced image-segmentation techniques.

---

## Conclusion

This project demonstrates how **image filtering and edge detection can be used to identify skin-lesion boundaries**.

Gaussian filtering helps reduce noise before edge detection, while Canny and Sobel operators identify intensity changes around the lesion. Different filters and threshold settings can produce different boundary results.

The accuracy of the detected boundary can be affected by **noise, hair, lighting conditions, weak boundaries, and low contrast**. Additional preprocessing and advanced segmentation techniques could improve the results.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Hugging Face Datasets
* Canny Edge Detection
* Sobel Edge Detection
* Otsu Thresholding
* Morphological Operations

---

## Project Pipeline

```text
HAM10000 Images
       ↓
Image Loading
       ↓
Grayscale Conversion
       ↓
Image Filtering
       ↓
Canny / Sobel Edge Detection
       ↓
Morphological Processing
       ↓
Lesion Boundary
       ↓
Area & Perimeter
       ↓
Filter & Edge Detector Comparison
```


