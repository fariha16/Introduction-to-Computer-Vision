# Skin Lesion Boundary Detection Using Canny Edge Detection

## Questions and Answers

### 1. Why is Gaussian filtering applied before Canny detection?

Gaussian filtering is applied to **remove noise and smooth the image** before edge detection. It reduces small unwanted variations that could otherwise be detected as false edges by the Canny algorithm.

### 2. How did the three Canny threshold settings affect the result?

The three threshold settings produced different edge detection results:

* **Low thresholds:** Detected more edges, including some unwanted noise.
* **Medium thresholds:** Detected the lesion boundary more clearly with less noise.
* **High thresholds:** Removed many weak edges, but some parts of the lesion boundary were also lost.

### 3. Which threshold produced the best lesion boundary?

The **medium threshold setting** generally produced the best lesion boundary because it provided a good balance between detecting the lesion edges and reducing unwanted edges.

> **Note:** If the actual output images show a different result, select the threshold that produced the clearest and most continuous lesion boundary.

### 4. Why are edges useful for detecting skin lesions?

Edges are useful because a skin lesion often has a **difference in color, intensity, or texture** compared with the surrounding skin. Canny edge detection identifies these changes and helps locate the boundary of the lesion.

### 5. What problems did you observe in detecting the lesion boundary?

Some problems observed during lesion boundary detection were:

* Noise created false edges.
* Hair and other skin details produced unwanted edges.
* Some parts of the lesion boundary were weak or broken.
* Uneven lighting affected edge detection.
* Similar colors between the lesion and surrounding skin made the boundary difficult to detect.

### 6. How could your method be improved?

The method could be improved by using better preprocessing techniques, such as **noise reduction, hair removal, and contrast enhancement**. Adaptive Canny threshold selection and morphological operations could also help create a more continuous and accurate lesion boundary. More advanced image segmentation techniques could be used to achieve better results.

## Conclusion

Canny edge detection can be useful for identifying the boundary of a skin lesion. Gaussian filtering helps reduce noise before edge detection, while appropriate Canny thresholds help obtain a clearer boundary. However, factors such as hair, noise, lighting, and low contrast can affect the accuracy of the detected boundary. Further preprocessing and advanced segmentation techniques can improve the results.

