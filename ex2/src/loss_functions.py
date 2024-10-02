import numpy as np


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    assert len(y_true) == len(y_pred), "The length is not equal"

    misclassifications = float(np.sum(y_true != y_pred))  # Count the number of misclassifications

    if normalize:
        # Normalize by the number of samples
        return misclassifications / len(y_true)
    else:
        return misclassifications


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    assert len(y_true) == len(y_pred), "The length is not equal"
    return float(np.sum(y_true == y_pred)) / len(y_true)
