
# Machine Learning Exercise 1 - Linear Regression and Polynomial Fitting

This project contains implementations related to **Linear Regression** and **Polynomial Fitting** for predicting house prices and temperatures. The project includes working with real-world datasets, implementing regression models, and evaluating their performance.

## Project Structure

- `linear_regression.py`: Contains the implementation of a **Linear Regression** model using Ordinary Least Squares (OLS).
- `polynomial_fitting.py`: Implements polynomial regression by extending the linear regression class to handle polynomial transformations.
- `house_price_prediction.py`: Uses linear regression to predict house prices based on a variety of features.
- `city_temperature_prediction.py`: Implements polynomial fitting to predict city temperatures based on the day of the year.
- `requirements.txt`: Lists the dependencies required for running the project.
- `house_prices.csv`: The dataset containing house price data, including features such as the number of bedrooms, bathrooms, and living area.
- `city_temperature.csv`: The dataset containing daily temperature records for various cities across different years.

## Dependencies

Ensure you have the following dependencies installed. They are listed in the `requirements.txt` file:
```bash
numpy~=1.24.3
pandas~=2.0.2
matplotlib~=3.9.0
```

To install them, simply run:
```bash
pip install -r requirements.txt
```

## How to Run

1. **House Price Prediction**:
   - The script `house_price_prediction.py` uses a linear regression model to predict house prices.
   - Run the script by executing:
     ```bash
     python house_price_prediction.py
     ```
   - The script will:
     - Split the dataset into training and test sets.
     - Preprocess the data, removing irrelevant features and handling missing values.
     - Evaluate feature importance using Pearson correlation.
     - Fit the linear regression model and plot the results.

2. **City Temperature Prediction**:
   - The script `city_temperature_prediction.py` performs polynomial fitting on temperature data to predict the daily temperature of cities.
   - Run the script by executing:
     ```bash
     python city_temperature_prediction.py
     ```
   - The script will:
     - Load and preprocess the temperature data.
     - Explore temperature trends for a specific country (Israel).
     - Fit a polynomial regression model and evaluate the model's accuracy across different countries.

## File Descriptions

### `linear_regression.py`
Implements the **Linear Regression** model, solving the Ordinary Least Squares (OLS) optimization problem. Includes methods to:
- Fit the model (`fit`)
- Make predictions (`predict`)
- Evaluate the model using Mean Squared Error (MSE) (`loss`)

### `polynomial_fitting.py`
Extends the linear regression model to perform **Polynomial Fitting**. The polynomial degree is set during initialization, and the input data is transformed to a Vandermonde matrix. 

### `house_price_prediction.py`
This script implements the following:
- Preprocessing of house price data, including feature selection and handling missing or invalid values.
- Splitting the dataset into training and test sets.
- Fitting the linear regression model to the training data.
- Evaluating feature importance and model performance using MSE.

### `city_temperature_prediction.py`
This script works with city temperature data and implements the following:
- Data loading and preprocessing.
- Visualization of temperature trends over the year.
- Polynomial fitting for temperature prediction.
- Evaluation of model performance across different countries.

## Results
- **House Price Prediction**: The model predicts house prices using linear regression, and its performance is measured using MSE over the test set.
- **City Temperature Prediction**: A polynomial fitting model is used to predict daily temperatures, and its performance is compared across different countries.
