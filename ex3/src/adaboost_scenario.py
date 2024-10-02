import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

LIST_OF_NOISE = [0, 0.4]

I = 800
HEIGHT_2_PLOT = I

WIDTH_2_PLOT = 1600

LIST_OF_EPSILON = [-.1, .1]


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_model = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost_model.fit(train_X, train_y)

    list_error_train_set = []
    list_error_test_set = []

    for number_of_iteration in range(1, n_learners + 1):
        list_error_train_set.append(adaboost_model.partial_loss(train_X, train_y, number_of_iteration))
        list_error_test_set.append(adaboost_model.partial_loss(test_X, test_y, number_of_iteration))

    fig_1 = go.Scatter(x=np.arange(n_learners), y=list_error_train_set,
                       mode='lines', line=dict(width=3, color="rgb(204,68,83)"),
                       name="Train loss")
    fig_2 = go.Scatter(x=np.arange(n_learners), y=list_error_test_set, mode='lines',
                       line=dict(width=3, color="rgb(6,106,141)"),
                       name="Test loss")
    plt = go.Figure([fig_1, fig_2],
                    layout=go.Layout(title="(1) Train and test errors as function of the number of fitted learners",
                                     xaxis_title='Number of fitted model',
                                     yaxis_title='Loss function'))
    # plt.show()
    plt.write_html(f"ada_with_{noise}_noise.html")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        LIST_OF_EPSILON)
    plt = make_subplots(2, 2, subplot_titles=[f"Prediction of ensemble with {i} size" for i in T]).update_xaxes(
        visible=False).update_yaxes(visible=False)
    for i in range(len(T)):
        calculate_row_place_plot = (i // 2) + 1
        calculate_col_place_plot = (i % 2) + 1
        plt.add_traces(
            [decision_surface(lambda sample: adaboost_model.partial_predict(sample, T[i]), lims[0], lims[1]),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y), showlegend=False)]
            , rows=calculate_row_place_plot, cols=calculate_col_place_plot)
        plt.update_layout(height=HEIGHT_2_PLOT, width=WIDTH_2_PLOT)
    # save the image
    # plt.show()
    plt.write_html(f"prediction_with_{noise}_decision_boundaries.html")

    # # Question 3: Decision surface of best performing ensemble
    min_test_err_idx = np.argmin(list_error_test_set)
    best_ensemble_size = min_test_err_idx + 1

    fig = go.Figure([
        decision_surface(lambda X: adaboost_model.partial_predict(X, best_ensemble_size), lims[0], lims[1], density=60,
                         showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x")))],
        layout=go.Layout(width=500, height=500,
                         title=f"Best Performing Ensemble       Size: {best_ensemble_size}, Accuracy: "
                               f"{(1 - list_error_test_set[min_test_err_idx]) * 100}%"))
    # fig.show()
    fig.write_html(f"best_performing_ensemble_{noise}_noise.html")

    #
    # # Question 4: Decision surface with weighted samples
    D = 5 * adaboost_model.D_[-1] / adaboost_model.D_[-1].max()
    fig = go.Figure([
        decision_surface(adaboost_model.predict, lims[0], lims[1], density=60, showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(size=D, color=train_y, symbol=np.where(train_y == 1, "circle", "x")))],
        layout=go.Layout(width=500, height=500, xaxis=dict(visible=False), yaxis=dict(visible=False),
                         title=f"Final AdaBoost Sample Distribution"))
    # fig.show()
    fig.write_html(f"Final_AdaBoost_Sample_Distribution_With_{noise}.html")


if __name__ == '__main__':
    np.random.seed(0)

    for noise in LIST_OF_NOISE:
        fit_and_evaluate_adaboost(noise=noise)
