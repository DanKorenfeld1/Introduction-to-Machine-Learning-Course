from __future__ import annotations
from typing import Callable
from typing import NoReturn
from base_estimator import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

EXP_COEF_LIKELIHOOD_GNB = -2

COEF_PI = 2

COEF_INNER_EXP = -0.5


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None
        # self.fitted_ = False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        # intercept
        self.fitted_ = True
        if self.include_intercept_:
            ones = np.ones((X.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)

        self.coefs_ = np.zeros(X.shape[1])
        for t in range(self.max_iter_):
            update = False
            min_i = -1
            for i in range(X.shape[0]):
                if y[i] * (X[i] @ self.coefs_) <= 0:
                    self.coefs_ += y[i] * X[i]
                    update = True
                    min_i = i
                    break
            if not update:
                break
            else:
                self.callback_(self, X[min_i], y[min_i])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        if self.include_intercept_ and self.coefs_.shape[0] != X.shape[1]:
            ones = np.ones((X.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)

        return np.sign(X @ self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

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
        from loss_functions import misclassification_error
        return misclassification_error(y, self.predict(X))


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, sum_val = np.unique(y, return_counts=True)
        length_y = len(y)
        self.pi_ = sum_val / length_y

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.mu_ = np.zeros((n_classes, n_features))  # samples X features
        self.cov_ = np.zeros((n_features, n_features))

        # mu_
        for idx, k in enumerate(self.classes_):
            X_k = X[y == k]
            self.mu_[idx, :] = X_k.mean(axis=0)

        # cov_
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            centered_X_c = X_c - self.mu_[i]
            self.cov_ += np.matmul(centered_X_c.T, centered_X_c)
        self.cov_ /= len(y)  # y == m (number of samples)
        assert det(self.cov_) != 0, "cov_ should be invertible"
        self._cov_inv = inv(self.cov_)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]  # for the sample i in idx i, there is the label.

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        # every line is one sample, in every line there are the array of classes, for example: image1 := [ , , , ]
        # in every cordination there are the probability to get the label i (the probability the image inc dog, cat...)
        # and we want to find the most likely probability in every sample (in predict)
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_features = len(self.cov_)
        like_li_hood = np.zeros((n_samples, n_classes))
        two_pi = COEF_PI * np.pi
        for x_i in range(n_samples):
            for k in range(n_classes):
                mu_k = self.mu_[k]
                z = np.sqrt((two_pi ** n_features) * det(self.cov_))
                inner_exp = (COEF_INNER_EXP) * np.transpose(X[x_i] - mu_k).dot(self._cov_inv).dot(X[x_i] - mu_k)
                like_li_hood[x_i, k] = self.pi_[k] * np.exp(inner_exp) * (1 / z)  # z!=0

        return like_li_hood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

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
        from loss_functions import misclassification_error
        return misclassification_error(y, self.predict(X))


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, sum_val = np.unique(y, return_counts=True)
        # self.fitted_ = True
        length_y = len(y)
        self.pi_ = sum_val / length_y

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.mu_ = np.zeros((n_classes, n_features))  # samples X features
        self.vars_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mu_[i, :] = X_c.mean(axis=0)
            self.vars_[i, :] = X_c.var(axis=0)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]  # for the sample i in idx i, there is the label.

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        # https://www.geeksforgeeks.org/gaussian-naive-bayes/ from this source:
        exponent = (X[:, np.newaxis, :] - self.mu_) ** 2 / (EXP_COEF_LIKELIHOOD_GNB * self.vars_)
        inner_normalization = 2 * np.pi * self.vars_
        normalization = np.sqrt(inner_normalization)
        likelihoods = np.exp(exponent) / normalization
        likelihoods_product = np.prod(likelihoods, axis=2) * self.pi_
        return likelihoods_product

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

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
        from loss_functions import misclassification_error
        return misclassification_error(y, self.predict(X))
