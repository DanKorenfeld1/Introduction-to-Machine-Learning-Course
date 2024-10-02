
# Exercise 2 - Classification

## Overview

This project is the second exercise for the Machine Learning course, focusing on the implementation of classification algorithms, including the Perceptron, Gaussian Naive Bayes, and LDA classifiers. The goal is to implement these classifiers and evaluate their performance on different datasets.

## Files

- **base_estimator.py**: Contains the abstract `BaseEstimator` class, which serves as the base class for all classifiers implemented in this project. Each classifier inherits from this class.
- **classifiers.py**: Implements the `Perceptron`, `LDA`, and `GaussianNaiveBayes` classifiers. Each classifier includes methods for fitting the model, making predictions, and calculating performance metrics.
- **classifiers_evaluation.py**: Contains functions to evaluate the classifiers' performance on various datasets and visualize the results through plots.
- **loss_functions.py**: Implements various loss functions including the `misclassification_error` and `accuracy` functions.
- **utils.py**: Provides utility functions for loading datasets and generating visualizations (e.g., scatter plots, decision boundaries).
- **gaussian1.npy**, **gaussian2.npy**, **linearly_inseparable.npy**, **linearly_separable.npy**: These are the datasets used for training and testing the classifiers.
  
## Usage

### Perceptron Classifier

1. The Perceptron classifier is implemented in `classifiers.py` and is used to find a separating hyperplane for linearly separable data.
2. The performance of the classifier is evaluated using `run_perceptron` in `classifiers_evaluation.py`. This function fits the Perceptron to both the `linearly_separable.npy` and `linearly_inseparable.npy` datasets and plots the training loss over iterations.

### Gaussian Naive Bayes and LDA Classifiers

1. The `GaussianNaiveBayes` and `LDA` classifiers are implemented in `classifiers.py`.
2. The `compare_gaussian_classifiers` function in `classifiers_evaluation.py` fits both classifiers on the `gaussian1.npy` and `gaussian2.npy` datasets, plots predictions, and compares accuracy.

### Running the Evaluation

To run the evaluation and generate the plots, run the following script:
```bash
python classifiers_evaluation.py
```

This will:
- Train the Perceptron on two datasets and plot the training loss.
- Compare Gaussian Naive Bayes and LDA classifiers on two datasets, generating scatter plots with class predictions, class centers, and covariance ellipses.

## Installation

Make sure to have the required dependencies installed by running:

```bash
pip install -r requirements.txt
```

## Requirements

The project requires the following libraries:
- `numpy`
- `plotly`
- `pandas`

## Output

- **Perceptron training loss plot**: A plot showing the training loss of the Perceptron algorithm for both linearly separable and inseparable data.
- **Gaussian Naive Bayes vs LDA comparison plots**: Scatter plots comparing predictions made by the Gaussian Naive Bayes and LDA classifiers on two datasets.
