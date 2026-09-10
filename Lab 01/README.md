# Lab 01: Skin Cancer Classification (ISIC Dataset)

Benchmarking transfer learning models, deep feature extractors, and classical machine learning classifiers on the 9-class ISIC skin cancer dataset[cite: 4].

## Experimental Results

### Table 1: Transfer Learning Model Performance
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vgg16** | 49.15 | 47.49 | 49.31 | 44.70 | 87.02  |
| **resnet18** | 55.08  | 60.16 | 54.17 | 48.21 | 90.92 |
| **efficientnet_b0** | 52.54 | 51.37 | 52.08 | 48.14 | 87.77 |

### Table 2: Deep Feature Extraction with Classical Classifiers
| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 50.00 | 47.74 | 50.00 | 44.43 |
| **Decision Tree** | 23.73 | 18.35[ | 25.46 | 19.81 |
| **Random Forest** | 36.44 | 37.46 | 38.89 | 28.50 |
| **K-Nearest Neighbors (KNN)** | 28.81 | 32.71 | 29.63 | 27.93 |
| **Linear SVM** | 45.76 | 45.09 | 46.53 | 42.63 |
| **RBF-SVM** | 49.15 | 52.82 | 49.31 | 44.43  |
| **XGBoost** | 42.37 | 40.18 | 43.75 | 37.99 |

### Table 3: Computational Efficiency & Performance
| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vgg16** | 134.30 | 512.32 | 30.93 | 67.51 | 49.15 |
| **resnet18** | 11.18 | 42.74 | 3.65 | 62.37 | 55.08 |
| **efficientnet_b0** | 3.98 | 15.73 | 0.77 | 62.47 | 52.54 |
