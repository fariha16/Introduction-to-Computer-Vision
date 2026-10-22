Lab 07 Report: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm
Objective

The objective of this experiment was to investigate whether Genetic Algorithm-based feature selection could improve the classification performance of a Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM) pipeline while reducing feature dimensionality.

Dataset and Methodology

BloodMNIST was selected as the medical image dataset. It contains 17,092 images representing eight blood-cell classes. The official training, validation, and test splits were used throughout the experiment.

Images were processed at 128 × 128 resolution and converted to grayscale. HOG features were extracted using nine orientations, 8 × 8 pixels per cell, 2 × 2 cells per block, and L2-Hys normalization. This configuration produced 8,100 features per image.

The baseline model used all HOG features with an RBF-kernel SVM. The proposed method used a Genetic Algorithm to select a subset of features. The GA employed binary chromosome encoding, population initialization, tournament selection, uniform crossover, mutation, and elitism.

Fitness was calculated using validation macro-F1 with a penalty proportional to the fraction of selected features. Two GA configurations were evaluated with mutation probabilities of 0.001 and 0.01.

Results

The baseline achieved a test macro-F1 of [insert measured value]%. GA configuration 1 achieved [insert measured value]%, while configuration 2 achieved [insert measured value]%.

The GA selected [insert number] and [insert number] features, respectively, compared with the original 8,100 features.

The best-performing configuration was [insert model name], with a macro-F1 improvement of [insert percentage-point difference] relative to the baseline. Its feature reduction was [insert percentage]%.

The measured GA search times were [insert time] and [insert time] seconds. The results indicate that feature selection [increased/decreased/did not materially change] classification performance while [insert measured feature reduction].

Discussion

The experiment evaluates whether removing less informative HOG features improves generalization. The GA can explore combinations of features that a simple univariate selection procedure might miss. However, repeated SVM fitting makes the search computationally expensive.

The results should be interpreted using test-set performance, feature reduction, and total computational cost together. A reduction in feature count alone is insufficient to establish an improvement in classification quality.

Conclusion

The research question was whether GA-based feature selection could improve HOG-based medical image classification while reducing dimensionality.

Based on the measured test results, the proposed approach [improved/did not improve/reduced] macro-F1 compared with the baseline and reduced the feature count by [insert measured percentage]%.

Therefore, the hypothesis that GA-based feature selection improves classification performance is [supported/not supported/partially supported] under the experimental conditions. Further experiments with additional random seeds and GA configurations would help establish the robustness of the findings.

Dataset Reference

Yang, J. et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data (2023). https://doi.org/10.1038/s41597-022-01721-8

Official dataset: https://medmnist.com/

