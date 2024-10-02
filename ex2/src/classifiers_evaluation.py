from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np

COEF_FOR_PRECENTAGE = 100

HEIGHT_PLOT = 800

WIDTH_PLOT = 1500

RIGHT_PLOT_CORD = [1, 2]

LEFT_PLOT_CORD = [1, 1]

SIZE_X_IN_PLOT = 20


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        Perceptron(callback=lambda fit, dummey1, dummey2: losses.append(fit.loss(X, y))).fit(X, y)
        fig = go.Figure(data=go.Scatter(x=list(range(len(losses))), y=losses, mode='lines'))
        fig.update_layout(
            title=f"Perceptron Training Loss Over Iterations ({n})",
            xaxis_title="Iteration",
            yaxis_title="Misclassification Error (%)"
        )
        # fig.show()
        fig.write_image(f"perceptrion_training_loss_over_itertations.png")  # problem with the saving of the plot


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnd = GaussianNaiveBayes()
        gnd_fitted = gnd.fit(X, y)
        lda = LDA()
        lda_fitted = lda.fit(X, y)

        predict_gnd = gnd_fitted.predict(X)
        predict_lda = lda_fitted.predict(X)



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from loss_functions import accuracy

        lda_accuracy = accuracy(y, predict_lda)
        gnd_accuracy = accuracy(y, predict_gnd)
        lda_accuracy_precentage = lda_accuracy * COEF_FOR_PRECENTAGE
        gnd_accuracy_precentage = gnd_accuracy * COEF_FOR_PRECENTAGE
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                rf"$\text{{Gaussian Naive Bayes (accuracy={round(gnd_accuracy_precentage, 2)}%)}}$",
                                rf"$\text{{LDA (accuracy={round(lda_accuracy_precentage, 2)}%)}}$"))

        # # Add traces for data-points setting symbols and colors
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=predict_gnd, symbol=class_symbols[y], colorscale=class_colors(3))),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=predict_lda, symbol=class_symbols[y],
                                               colorscale=class_colors(3)))],
                       rows=LEFT_PLOT_CORD, cols=RIGHT_PLOT_CORD)  # row and col where to locate the trace

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=gnd.mu_[:, 0], y=gnd.mu_[:, 1], mode="markers",
                                   marker=dict(symbol="x", color="black", size=SIZE_X_IN_PLOT)),
                        go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                   marker=dict(symbol="x", color="black", size=SIZE_X_IN_PLOT))],
                       rows=LEFT_PLOT_CORD, cols=RIGHT_PLOT_CORD)  # row and col where to locate the trace

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            fig.add_traces([get_ellipse(gnd.mu_[i], np.diag(gnd.vars_[i])), get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=LEFT_PLOT_CORD, cols=RIGHT_PLOT_CORD)

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title_text=rf"$\text{{Comparing Gaussian Classifiers- {f[:-4]} dataset}}$",
                          width=WIDTH_PLOT, height=HEIGHT_PLOT, showlegend=False)
        # fig.show()
        fig.write_image(f"lda.vs.naive.bayes.{f[:-4]}.png")  # problem with the saving of the plot


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
