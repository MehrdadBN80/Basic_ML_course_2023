# Machine Learning Course Materials

## Computer Science Faculty of Shahid Beheshti University
**Winter 2023**

### Description
This repository contains materials for the Machine Learning Course offered by the Computer Science Faculty of Shahid Beheshti University in Winter 2023. The course covers a range of machine learning concepts including regression, regularization techniques, gradient descent, model evaluation, and feature selection. Students will engage in both theoretical analysis and practical implementation using Python and the Scikit-learn library, supplemented with custom algorithms developed from scratch.

### Course Structure

| Week     | Topic                         | Subtopics                                                   |
|----------|-------------------------------|------------------------------------------------------------|
| 1-3      | Fundamentals of Statistics and Probability | Basics of Probability, Distributions, Statistics, Hypothesis Tests |
| 4-5      | Regression                    | Linear Regression, Polynomial Regression, Logistic Regression, LDA |
| 6-8      | Model Evaluation              | Cross Validation, Bootstrapping, Feature Selection, Regularization |
| 9-10     | Support Vector Machine (SVM)  |                                                            |
| 11-13    | Tree-Based Methods            | Decision Trees, Ensemble Learning, Bagging, Boosting       |
| 14-15    | Unsupervised Learning         | PCA, Clustering                                            |
| 16       | Reinforcement Learning        |                                                            |

### Prerequisites
- Python 3.x
- Libraries: NumPy, Pandas, scikit-learn, Matplotlib, TensorFlow or PyTorch (for deep learning)

It is recommended to have basic familiarity with Python programming and an understanding of linear algebra and calculus.

### Installation
To run the code in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/SBU_Machine_Learning_Course_2023.git
   ```
2. Navigate to the project directory:
   ```bash
   cd SBU_Machine_Learning_Course_2023
   ```
3. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
This repository contains code examples, lecture notes, and assignments for each topic. You can run the Python notebooks or scripts to experiment with different machine learning models and techniques. For each week, there will be corresponding assignments to reinforce the concepts learned in lectures.

---

## Assignments Overview

### Assignment 1 - SBU Machine Learning Course
separate repository link: https://github.com/MehrdadBN80/Penguins_dataset_Classification
This assignment includes solutions for the first assignment of the Machine Learning Course (Winter 2023), covering essential machine learning concepts.

#### Key Tasks
- **Gradient Descent**: Analyze local minima in logistic regression.
- **Polynomial Regression**: Evaluate training vs. validation errors.
- **Ridge Regression**: Adjust regularization hyperparameters.
- **Regression Techniques**: Compare ridge, lasso, and elastic net.
- **MAE in Linear Regression**: Implement MAE as the cost function.
- **Normal Equation**: Implement linear regression from scratch.
- **Bootstrapping vs. Cross-Validation**: Discuss appropriate usage.
- **Nested Cross-Validation (Extra Points)**: Explain nested CV and significance tests.
- **Feature Selection**: Implement forward and backward methods.
- **Model Sensitivity**: Investigate effects of feature scaling.
- **News Popularity Prediction**: Build a regression model using EDA, ridge and lasso.
- **Softmax Regression (Extra Points)**: Implement early stopping on the Penguins dataset.

### Assignment 2 - SBU Machine Learning Course
separate repository link: https://github.com/MehrdadBN80/vehicle-Insurance_Claim_Fraud_Detection_dataset_classification/tree/main
This assignment contains the materials for the second assignment of the Machine Learning course at Shahid Beheshti University. The assignment focuses on various machine learning concepts, particularly concerning Support Vector Machines (SVM), decision trees, ensemble methods, and their applications in a real-world dataset related to vehicle insurance claim fraud detection.


1. **SVM Classifier Insights**: Investigate whether an SVM classifier can provide confidence scores or probabilities for predictions.
2. **RBF Kernel Underfitting**: Discuss strategies to address underfitting when using an RBF kernel in SVM, including adjustments to hyperparameters γ (gamma) and C.
3. **ϵ-Insensitive Models**: Explain the concept of ϵ-insensitivity in model performance.
4. **Margin Types in SVM**: Differentiate between hard margin and soft margin SVMs, and identify appropriate use cases for each.
5. **Gini Impurity**: Analyze the relationship between a node's Gini impurity and that of its parent node.
6. **Feature Scaling in Decision Trees**: Evaluate the need for scaling features if a Decision Tree is underfitting.
7. **Feature Selection with Tree-Based Models**: Explore methods for using tree-based models for feature selection.
8. **Hyperparameter Tuning**: Provide strategies for adjusting hyperparameters in AdaBoost (underfitting) and Gradient Boosting (overfitting).
9. **Ensemble Types**: Compare homogeneous and heterogeneous ensembles and discuss their relative power.
10. **ROC and AUC**: Describe how ROC curves and AUC metrics are utilized to evaluate classification performance.
11. **Threshold Values**: Explain how the threshold value impacts classification performance, including trade-offs between false positives and false negatives.
12. **Multiclass Classification**: Differentiate between one-vs-one and one-vs-all approaches in multiclass classification.
13. **Vehicle Insurance Claim Fraud Detection**: Implement multiple classification models on the dataset, including exploratory data analysis, handling class imbalance, and boosting model performance.
14. **Anomaly Detection with SVM**: Discuss the application of SVM for anomaly detection and its associated challenges (Extra Point).
15. **Bagging Classifier**: Implement a Bagging classifier from scratch and test it on the Penguins dataset (Extra Point).
16. **Class Imbalance in Ensemble Learning**: Address techniques for handling class imbalance in ensemble methods (Extra Point).


### Assignment 3 - SBU Machine Learning Course
separate repository link: https://github.com/MehrdadBN80/supermarket-dataset-unsupervised-k-means-pca-4-
This assignment is centered on the application of clustering techniques using a supermarket dataset for predictive marketing analytics. The goal is to segment customers based on their shopping habits and demographics while exploring various clustering methodologies and dimensionality reduction strategies.

## Table of Contents
- Curse of Dimensionality
- PCA Techniques
- Chaining Dimensionality Reduction Algorithms
- Assumptions and Limitations of PCA
- Clustering and Linear Regression Accuracy
- Entropy as a Validation Measure for Clustering
- Label Propagation (Extra Credit)
- Supermarket Dataset Exploration
- Data Preprocessing
- K-means Clustering
- Cluster Visualization
- Other Clustering Algorithms
- Dimensionality Reduction with PCA (Extra Credit)
- Insights and Recommendations (Extra Credit)

## Assignment Tasks

1. **Curse of Dimensionality**  
   **Definition**: The curse of dimensionality refers to challenges encountered when analyzing data in high-dimensional spaces, which can diminish the effectiveness of clustering as distance metrics become less meaningful.

2. **PCA Techniques**  
   **Options**:
   - **Regular PCA**: Suitable for smaller datasets that fit into memory.
   - **Incremental PCA**: Ideal for larger datasets that cannot be fully loaded at once, facilitating batch processing.
   - **Randomized PCA**: A quicker approximation for high-dimensional data.
   - **Random Projection**: Effective for reducing dimensionality when speed is a priority, even if some distortion occurs.

3. **Chaining Dimensionality Reduction Algorithms**  
   **Consideration**: Combining dimensionality reduction techniques can enhance computational efficiency if the first method sufficiently reduces dimensionality for the second to be effective.

4. **Assumptions and Limitations of PCA**  
   **Assumptions**:
   - Features are linearly related.
   - Features are centered around their mean.

   **Limitations**:
   - PCA is sensitive to data scaling.
   - May not perform well with non-linear data structures.

5. **Clustering for Enhanced Linear Regression**  
   **Usage**: Clustering can reveal distinct customer segments, allowing for the creation of tailored regression models that improve predictive accuracy.

6. **Entropy as a Clustering Validation Measure**  
   **Description**: Entropy gauges the purity of clusters, with lower values indicating more homogeneous groups. It serves as a validation metric for clustering performance.

7. **Label Propagation (Extra Credit)**  
   **Definition**: A semi-supervised learning algorithm for graph-based clustering and classification, it propagates labels throughout the data based on connectivity.

8. **Supermarket Dataset Exploration**  
   - **Data Preprocessing**: Load the dataset, address NaN values, and encode categorical features. Scale numerical features with StandardScaler.
   - **K-means Clustering**: Determine the optimal number of clusters using the Elbow method and silhouette score, followed by K-means implementation and validation.
   - **Cluster Visualization**: Utilize PCA for 2D and 3D visualizations to understand customer segments.

9. **Other Clustering Algorithms**  
   Experiment with DBSCAN and hierarchical clustering, comparing their performance to K-means.

10. **Dimensionality Reduction with PCA (Extra Credit)**  
    Apply PCA prior to clustering and analyze its effects on results.

11. **Insights and Recommendations (Extra Credit)**  
    Analyze customer segments to provide actionable marketing insights and recommendations.

#### Assignment Tasks
- Apply clustering methods and dimensionality reduction techniques to segment customers based on shopping behavior.

### Conclusion
These assignments provide a comprehensive exploration of machine learning techniques and their application to real-world problems. Through both theoretical and practical tasks, students develop a deeper understanding of model evaluation, tuning, and deployment.

### Acknowledgments
Special thanks to the instructors and peers for their support throughout the course.

### References
- PCA - [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
- K-means Clustering - [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
- DBSCAN - [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
