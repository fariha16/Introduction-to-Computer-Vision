Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm
1. Objective

The objective of this experiment is to improve traditional Histogram of Oriented Gradients (HOG)-based medical image classification by integrating a Genetic Algorithm (GA) for feature selection.

The experiment compares two approaches:

Baseline: HOG feature extraction + Support Vector Machine (SVM).
Proposed: HOG feature extraction + Genetic Algorithm-based feature selection + SVM.

The primary goal is to determine whether GA-based feature selection can improve classification performance while reducing feature dimensionality and computational cost during subsequent model training and inference.

2. Dataset

Dataset: BloodMNIST

BloodMNIST is a medical image dataset containing blood-cell images from eight classes.

Number of images: 17,092
Number of classes: 8
Image resolution: 128 × 128
Image format: RGB
Dataset Splits

The experiment uses the official BloodMNIST splits.

Split	Number of Images
Training	11,959
Validation	1,712
Testing	3,421
Total	17,092
Dataset References
Official website: https://medmnist.com/
GitHub repository: https://github.com/MedMNIST/MedMNIST
Research paper: https://doi.org/10.1038/s41597-022-01721-8
3. Methodology

The experiment consists of the following stages:

Load the BloodMNIST dataset.
Preprocess images and resize them to 128 × 128.
Convert RGB images to grayscale.
Extract HOG features.
Train the baseline HOG + SVM classifier.
Apply a Genetic Algorithm to select an informative subset of HOG features.
Train an SVM classifier using the selected features.
Evaluate both approaches using the same held-out test set.
Compare classification performance, feature dimensionality, and computational time.
4. Experimental Configuration
Parameter	Baseline HOG + SVM	GA-HOG + SVM
Dataset	BloodMNIST	BloodMNIST
Image resolution	128 × 128	128 × 128
Preprocessing	RGB to grayscale	RGB to grayscale
HOG orientations	9	9
Pixels per cell	8 × 8	8 × 8
Cells per block	2 × 2	2 × 2
Block normalization	L2-Hys	L2-Hys
Feature selection	None	Genetic Algorithm
Classifier	SVM	SVM
SVM kernel	RBF	RBF
SVM C	10	10
Random seed	42	42
Data splits	Official fixed splits	Same official fixed splits

The selected HOG configuration produces 8,100 features per image.

5. Genetic Algorithm

The Genetic Algorithm searches for a feature subset that achieves high validation macro-F1 while penalizing unnecessarily large feature subsets.

Chromosome Representation

Each chromosome is a binary vector of length 8,100.

1: Select the corresponding HOG feature.
0: Discard the corresponding HOG feature.
GA Operations
Population initialization: Generate random binary feature masks.
Fitness evaluation: Train an RBF SVM using each candidate feature subset and calculate validation macro-F1.
Selection: Use tournament selection to choose parent chromosomes.
Crossover: Apply uniform crossover to generate offspring.
Mutation: Randomly flip binary genes according to the mutation probability.
Elitism: Preserve the best chromosome across generations.
Fitness Function

The fitness function combines classification performance and a feature-selection penalty:

𝐹
(
𝑚
)
=
𝐹
1
,
m
a
c
r
o
(
𝑚
)
−
𝜆
∑
𝑗
=
1
𝑑
𝑚
𝑗
𝑑

Where:

𝐹
1
,
m
a
c
r
o
 is validation macro-F1.
𝑚
 is the binary feature-selection mask.
𝑑
 is the total number of HOG features.
𝜆
=
0.05
 is the feature penalty coefficient.

A higher fitness indicates a better balance between classification performance and feature reduction.

GA Configurations
Parameter	GA Configuration 1	GA Configuration 2
Population size	10	10
Generations	6	6
Mutation probability	0.001	0.01
Crossover	Uniform	Uniform
Selection	Tournament	Tournament
Elitism	Enabled	Enabled
Feature penalty	0.05	0.05

The mutation probability is the only GA parameter changed between the two configurations.

6. Evaluation Metrics

Both models are evaluated using the following metrics:

Accuracy: Proportion of correctly classified images.
Macro precision: Average precision across all classes.
Macro recall: Average recall across all classes.
Macro-F1: Unweighted average of the class-wise F1 scores.
Weighted-F1: Average F1 weighted by class support.
Balanced accuracy: Average class-wise recall.
Macro specificity: Average one-vs-rest specificity across classes.
Cohen's kappa: Agreement between predicted and true labels adjusted for chance.
Matthews correlation coefficient (MCC): Overall classification quality accounting for true and false predictions.

The test set is reserved for final evaluation and is not used to select GA configurations.

7. Experimental Results

Note: The following tables must be populated using the actual output of the experiment. No numerical performance results are assumed in advance.

Table 1. Classification Performance Comparison
Metric	HOG + SVM	GA-HOG + SVM (0.001)	GA-HOG + SVM (0.01)
Accuracy (%)	TBD	TBD	TBD
Macro precision (%)	TBD	TBD	TBD
Macro recall (%)	TBD	TBD	TBD
Macro-F1 (%)	TBD	TBD	TBD
Weighted-F1 (%)	TBD	TBD	TBD
Balanced accuracy (%)	TBD	TBD	TBD
Macro specificity (%)	TBD	TBD	TBD
Cohen's kappa	TBD	TBD	TBD
Matthews correlation coefficient	TBD	TBD	TBD
Table 2. Feature Selection Comparison
Measurement	Baseline	GA Configuration 1	GA Configuration 2
Original features	8,100	8,100	8,100
Selected features	8,100	TBD	TBD
Features removed	0	TBD	TBD
Feature reduction (%)	0	TBD	TBD

Feature reduction is calculated as:

Reduction
(
%
)
=
(
1
−
𝑑
s
e
l
e
c
t
e
d
𝑑
o
r
i
g
i
n
a
l
)
×
100
Table 3. Computational Cost
Measurement	HOG + SVM	GA Configuration 1	GA Configuration 2
HOG extraction time (seconds)	TBD	TBD	TBD
GA search time (seconds)	N/A	TBD	TBD
Final SVM fitting and evaluation time (seconds)	TBD	TBD	TBD
Total pipeline time (seconds)	TBD	TBD	TBD

The GA search time should be reported separately because it is an additional cost compared with the baseline. Total pipeline time should include feature extraction, GA search where applicable, and final model fitting and evaluation.

8. Graphical Results

The experiment generates the following plots in the lab07_results/ directory.

Figure 1. GA Fitness Across Generations

File: ga_fitness.png

This graph compares the best fitness values across generations for both mutation probabilities. An increasing fitness curve indicates that the algorithm is discovering better feature subsets.

Figure 2. Feature Dimensionality Before and After Selection

File: feature_counts.png

This graph compares the original 8,100 HOG features with the number of features selected by each GA configuration. A smaller feature subset indicates successful dimensionality reduction.

Figure 3. Test Macro-F1 Comparison

File: test_macro_f1.png

This graph compares the final test macro-F1 scores of the baseline and the two GA configurations. Higher macro-F1 indicates better average classification performance across the eight classes.

9. Discussion
Strengths
GA-based selection can remove irrelevant or redundant HOG features.
Macro-F1-based fitness encourages performance across all classes.
The experiment compares two mutation probabilities under the same data-split and classifier settings.
A smaller feature subset can reduce storage requirements and potentially lower SVM prediction costs.
Limitations
Repeated SVM training makes GA-based feature selection computationally expensive.
Random initialization and mutation can lead to different feature subsets across runs.
Repeated optimization against the same validation set may cause validation overfitting.
Excessive feature penalties may remove useful information.
BloodMNIST is a benchmark dataset; strong benchmark performance does not establish clinical diagnostic reliability.
Computational Cost

The baseline requires HOG extraction and one final SVM fit. The proposed approach additionally evaluates many candidate feature subsets during GA optimization.

Although feature selection may reduce the cost of the final classifier, the overall pipeline may still be slower because of the search process. Both the feature reduction and the total runtime must therefore be considered when evaluating the proposed method.

10. Conclusion

This experiment investigates whether Genetic Algorithm-based feature selection can improve HOG-based blood-cell image classification while reducing feature dimensionality.

The final conclusion should be based on the measured results:

Baseline test macro-F1: [insert result]
Best GA test macro-F1: [insert result]
Macro-F1 improvement: [insert percentage-point difference]
Best GA feature count: [insert result]
Feature reduction: [insert percentage]
Additional GA search time: [insert result]

If the GA model achieves higher macro-F1 and balanced accuracy while selecting fewer features, the results support its effectiveness under the tested configuration. If performance decreases, the GA may still provide dimensionality reduction, but the classification trade-off must be acknowledged. If results are similar, the additional search cost may not be justified.

The research question can be answered only after examining the actual test results and computational measurements.

11. How to Run
Install Dependencies
pip install medmnist scikit-image scikit-learn numpy pandas matplotlib seaborn

Execute the Experiment
python lab07_ga_hog.py


The dataset is downloaded automatically if it is not already available.

Generated Files

The script saves its results in the lab07_results/ directory:

lab07_results/
├── experimental_configuration.csv
├── metrics_comparison.csv
├── ga_fitness.csv
├── ga_fitness.png
├── feature_counts.png
└── test_macro_f1.png

12. References
Yang, J. et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 2023. https://doi.org/10.1038/s41597-022-01721-8
MedMNIST official website: https://medmnist.com/
MedMNIST GitHub repository: https://github.com/MedMNIST/MedMNIST
scikit-image HOG documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
scikit-learn SVM documentation: https://scikit-learn.org/stable/modules/svm.html
