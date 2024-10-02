import pandas as pd
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
import linear_regression as lr

IRRELEVANT_FEATURE = ["id", "date", "lat", "long", "sqft_living15", "sqft_lot15"]

NOT_NEG_FEATURE = ["sqft_basement", "yr_renovated", "bedrooms", "bathrooms"]

POS_FEATURE = ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "floors"]


def remove_irrelevant_feature(data_before_process: pd.DataFrame) -> pd.DataFrame:
    """
    Load the data from the given file path.
    Parameters
    ----------
    file_path : str
        Path to the file to load

    Returns
    -------
    pd.DataFrame
        The loaded data
        :param data_before_process: pd.DataFrame
    """
    return data_before_process.drop(columns=IRRELEVANT_FEATURE, axis=1)


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    row_data = X
    row_data['price'] = y

    row_data = remove_irrelevant_feature(row_data)
    row_data = row_data.dropna()

    # check the feature that need to be > 0.
    for feature in POS_FEATURE:
        row_data = row_data[row_data[feature] > 0]

    # check the feature that need to be >= 0.
    for feature in NOT_NEG_FEATURE:
        row_data = row_data[row_data[feature] >= 0]

    # check closed domain features
    row_data = row_data[((row_data["yr_renovated"] == 0) | (row_data["yr_renovated"] >= row_data["yr_built"]))]
    row_data = row_data[row_data["view"].isin(range(5))]
    row_data = row_data[row_data["grade"].isin(range(1, 14))]
    row_data = row_data[row_data["waterfront"].isin([0, 1])]
    row_data = row_data[row_data["condition"].isin(range(1, 6))]

    X, y = row_data.drop('price', axis=1), row_data.price

    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    return remove_irrelevant_feature(X)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_std = np.std(y)
    for feature in X.columns:
        feature_y_cov_matrix = np.cov(X[feature], y)

        feature_y_cov = feature_y_cov_matrix[0, 1]
        feature_std = np.std(X[feature])
        pearsons_correlation = feature_y_cov / (feature_std * y_std)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, alpha=0.7, color='skyblue', edgecolors='w', label='Data Points', s=20)
        plt.title(f"Pearson Correlation {feature} - response: {pearsons_correlation:.2f}")
        plt.xlabel(feature)
        plt.ylabel('Response')
        plt.savefig(f"{output_path}/{feature}_correlation.png")
        plt.close()


def frac(precent):
    """
    convert precent to fraction
    :param precent: the precent
    :return: the fraction
    """
    return precent / 100


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("house_prices.csv")

    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test

    train_indices = X.sample(frac=0.75, random_state=0).index

    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]

    # The remaining 25% of the data for testing
    X_test = X.drop(train_indices)
    y_test = y.drop(train_indices)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    mean_losses = []
    std_losses = []

    percentages = range(10, 101)

    for precent in percentages:
        loss_list = []
        for i in range(10):
            current_sample = X_train.sample(frac=frac(precent)).index
            X_sample = X_train.loc[current_sample]
            y_sample = y_train.loc[current_sample]
            fitted_model = lr.LinearRegression()
            fitted_model.fit(X_sample, y_sample)
            res_predict = fitted_model.predict(X_test)
            loss = fitted_model.loss(X_test, y_test)
            loss_list.append(loss)

        mean_losses.append(np.mean(loss_list))
        std_losses.append(np.std(loss_list))

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    percentages = [frac(p) for p in percentages]

    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mean_losses, label='Mean Loss')
    plt.fill_between(percentages, mean_losses - 2 * std_losses, mean_losses + 2 * std_losses, alpha=0.2,
                     label='Confidence Interval (±2*std)')
    plt.xlabel('Percentage of Training Set')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss as a Function of Training Set Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('mean_loss_plot.png')
    # plt.show()
