
# AdaBoost Algorithm with Decision Stump Implementation

## Overview
This project focuses on implementing the AdaBoost algorithm using a Decision Stump as the weak learner. AdaBoost is a powerful ensemble method that improves the accuracy of classifiers by combining multiple weak learners iteratively. The weak learner chosen for this implementation is a Decision Stump, which is a simple decision tree of depth 1.

## Files and Components
1. **`adaboost.py`**: Implements the AdaBoost algorithm with the following components:
   - `fit`: Trains the model using weak learners (decision stumps).
   - `predict`: Predicts the class labels based on the trained ensemble of models.
   - `partial_predict`: Predicts using a subset of the ensemble up to a given iteration.
   - `loss`: Evaluates the misclassification error for the full ensemble or a subset of learners.
  
2. **`decision_stump.py`**: Implements a weak learner (Decision Stump) with the following features:
   - Selects a feature and threshold to minimize classification error.
   - `fit`: Determines the optimal threshold and feature to split the data.
   - `predict`: Classifies the samples based on the threshold and feature selected.

3. **`adaboost_scenario.py`**: Contains code to generate data, fit the AdaBoost model, and evaluate its performance. This file includes the following:
   - `generate_data`: Generates synthetic datasets for training and testing.
   - `fit_and_evaluate_adaboost`: Fits the AdaBoost model on the generated data and plots various evaluation metrics, including decision boundaries and error plots.
   
4. **`loss_functions.py`**: Defines the `misclassification_error` function used to calculate the loss during training and evaluation.

5. **`base_estimator.py`**: A base class that provides the structure for estimators like AdaBoost and Decision Stump, ensuring consistency in their API.

6. **`utils.py`**: Contains utility functions for plotting decision surfaces, generating synthetic data, and splitting datasets.

## Practical Tasks
1. **Train and Test Errors**: Evaluate the AdaBoost performance on both train and test data as a function of the number of learners.
2. **Decision Boundaries**: Visualize the decision boundaries after different numbers of weak learners (5, 50, 100, 250).
3. **Best Ensemble**: Identify the best performing ensemble size based on the lowest test error.
4. **Weighted Samples**: Visualize the training samples where the size of each point represents its weight in the final iteration of AdaBoost.
5. **Noisy Data**: Perform the same experiments with added noise to assess AdaBoost's robustness.

## Running Instructions
1. Install the necessary Python packages:
   ```bash
   pip install numpy plotly
   ```
2. To execute the experiments and generate the plots, run:
   ```bash
   python adaboost_scenario.py
   ```
3. The script will generate HTML files with visualizations of the decision boundaries, error plots, and sample distributions.

## Results
Running the provided scripts will generate the following outputs:
1. **Train and Test Errors**: A graph showing how the error on the training and testing data evolves with the number of learners.
2. **Decision Boundaries**: Visualizations of how the decision boundaries change with increasing numbers of weak learners.
3. **Best Performing Ensemble**: The decision boundary and accuracy of the best ensemble based on test error.
4. **Weighted Samples**: A plot of the training samples with weights indicating their difficulty for the classifier.

