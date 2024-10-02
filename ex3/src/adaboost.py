import numpy as np
from typing import Callable, NoReturn
from base_estimator import BaseEstimator
from loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _get_new_samples_from_distributions(self, X: np.ndarray, distribution: np.ndarray) -> np.ndarray:
        """
        method that returns the new samples from the given distributions
        :param X: the current data set
        :param distribution: the current distribution
        :return: new samples from the dataset.
        """
        return np.random.choice(X, size=X.shape[0], p=distribution)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.fitted_ = True
        self.models_ = []
        self.weights_ = []
        self.D_ = []

        number_of_samples = X.shape[0]
        uniform_distribution = 1 / number_of_samples
        self.D_.append(np.full(number_of_samples, uniform_distribution))  # the uniform distribution (first iteration)

        for number_iteration in range(self.iterations_):
            distribution_y = y * self.D_[-1]

            h_iteration_model = self.wl_().fit(X, distribution_y)  # fit the new model
            self.models_.append(h_iteration_model)

            y_prediction = self.models_[-1].predict(X)

            epsilon_iteration = self.D_[-1] @ np.where(y != y_prediction, 1, 0)

            current_weight = 0.5 * np.log((1 / epsilon_iteration) - 1)
            self.weights_.append(current_weight)
            # update the new distribution
            D_new = self.D_[-1] * np.exp(-y * current_weight * y_prediction)
            D_new /= np.sum(D_new)  # normalize the distribution
            self.D_.append(D_new)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, T=self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, T=self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        result = []
        for t in range(0, min(T, self.iterations_)):
            res = self.weights_[t] * self.models_[t].predict(X)
            result.append(res)
        weights_predict = np.sum(result, axis=0)
        return np.sign(weights_predict)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
