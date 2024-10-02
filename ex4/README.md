
# Exercise 4 - Gradient-Based Learning

## Overview
This exercise focuses on implementing and investigating gradient-based learning algorithms, specifically gradient descent and regularized logistic regression. The provided Python scripts contain various components that facilitate the comparison of fixed learning rates and the minimization of regularized logistic regression objectives using gradient descent.

## Files and Components
1. **`gradient_descent.py`**: Implements a generic gradient descent algorithm that can be used with various learning rates and objective functions. It includes different strategies for learning rate updates and various stopping conditions.
2. **`learning_rate.py`**: Contains the implementation of different learning rate strategies, including fixed and exponential decay.
3. **`modules.py`**: Implements objective functions for gradient descent, including L1 and L2 regularization, as well as the logistic regression objective function.
4. **`logistic_regression.py`**: Implements the logistic regression model using gradient descent for optimization, supporting both L1 and L2 regularization.
5. **`gradient_descent_investigation.py`**: A script designed to investigate the behavior of gradient descent with different learning rates and visualize the results, including the descent path and convergence plots.
6. **`cross_validate.py`**: Contains a cross-validation procedure to evaluate different regularization parameters for logistic regression using gradient descent.
7. **`loss_functions.py`**: Provides utility functions to calculate various loss metrics, including mean squared error, misclassification error, and cross-entropy.

## Practical Tasks
- **Gradient Descent Investigation**: The `compare_fixed_learning_rates` function compares the performance of fixed learning rates on L1 and L2 objectives. Descent paths and convergence rates are plotted for different learning rates.
- **Regularized Logistic Regression**: The exercise explores the use of gradient descent to solve logistic regression with L1 and L2 regularization. Cross-validation is used to determine optimal regularization parameters.

## Running Instructions
1. Ensure all required libraries (e.g., NumPy, Pandas, Plotly) are installed.
2. Execute the investigation script (`gradient_descent_investigation.py`) to compare different fixed learning rates.
3. Use `logistic_regression.py` to train logistic regression models with different regularization types and evaluate the results using cross-validation and ROC curve plotting.

