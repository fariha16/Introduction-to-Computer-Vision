# Lab — SIFT-Based Landmark Recognition System

**Pipeline:** `Image → Grayscale Conversion → SIFT Keypoint Detection & Description → k-NN Matching & Lowe’s Ratio Test → RANSAC Homography & Geometric Verification → Decision / Localization`[cite: 3]

**Dataset:** Automatically downloaded freely licensed photographs from Wikimedia Commons (or custom user images via Google Drive)[cite: 3].

---

## Important Design Decision
The landmark database is organized into separate folders per landmark (at least 5 famous landmarks, each with multiple reference photographs and hold-out queries)[cite: 3]. 

To handle open-set recognition correctly (allowing the system to reject unknown landmarks rather than forcing a classification), the decision rule requires passing **both** a minimum number of reliable matches ($T_m = 20$) and a minimum number of RANSAC geometric inliers ($T_i = 15$)[cite: 3].

---

## Notebook Structure & Lab Tasks

| Notebook section | Lab tasks covered |
|---|---|
| 4. Experiment Configuration | Configuration & Constants[cite: 3] |
| 5. Data Acquisition & Preparation | Task 1: Landmark Image Database[cite: 3] |
| 6. SIFT Feature Extraction | Task 2: SIFT Feature Extraction[cite: 3] |
| 7. Query Image Processing | Task 3: Query Processing[cite: 3] |
| 8. Feature Matching & Recognition | Task 4: Matching & Ranking[cite: 3] |
| 9. Landmark Localization | Task 5: RANSAC Homography & Polygon Projection[cite: 3] |
| 10. End-to-End Demonstration | Demonstration[cite: 3] |
| 11–12. Evaluation & Analysis | Task 6: Performance Evaluation & Metrics[cite: 3] |
| 13–19. Interpretation & Checklist | Discussion, Limitations, & Compliance[cite: 3] |

---

### Prerequisites
* Google Colab or a standard Python environment with Jupyter Notebook.
* Required libraries: `OpenCV` (with `opencv-contrib-python`), `NumPy`, `Pandas`, `Requests`, `Matplotlib`[cite: 3].
