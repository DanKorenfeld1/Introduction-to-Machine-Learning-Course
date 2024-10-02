import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go

OPTIONS_LIST_LAMBDA = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

RANGE_PLOT = [0, 1]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = list()
    weights = list()

    # callback = lambda val, weight, **kwargs: {values.append(val), weights.append(weight)}
    def callback(val, weight, **kwargs):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = {"L1": L1, "L2": L2}
    lowest_losses = {}

    for name, module in modules.items():
        results = {}
        module_lowest_loss = float('inf')

        for eta in etas:

            callback, vals, weights = get_gd_state_recorder_callback()
            learning_rate = FixedLR(eta)
            gradient_descent = GradientDescent(learning_rate=learning_rate, callback=callback)
            module_to_fit = module(weights=np.copy(init))
            gradient_descent.fit(module_to_fit, None, None)
            results[eta] = (vals, weights)

            # Find the lowest loss for this eta
            current_lowest_loss = min(vals)
            if current_lowest_loss < module_lowest_loss:
                module_lowest_loss = current_lowest_loss

            plot_descent_path(module, np.array([init] + weights), f"{name}- Learning Rate: {eta}") \
                .write_html(f"GD_{name}_eta_{eta}.html")
            # plot_descent_path(module, np.array([init] + weights), f"{name}- Learning Rate: {eta}").show()
        lowest_losses[name] = module_lowest_loss

        fig = go.Figure(layout=go.Layout(xaxis=dict(title="GD Iteration"),
                                         yaxis=dict(title="Norm"),
                                         title=f"{name} GD Convergence For Different Learning Rates"))
        for eta, (v, _) in results.items():
            fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, mode="lines", name=rf"eta={eta}"))
        fig.write_html(f"GD_{name}_fixed_rate_convergence.html")
        # fig.show()

    print("Lowest losses achieved:")
    for name, loss in lowest_losses.items():
        print(f"{name}: {loss}")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, losses, weights = get_gd_state_recorder_callback()
    lr = 1e-4
    max_iter = 20000
    learning_rate = FixedLR(lr)
    gd = GradientDescent(learning_rate=learning_rate, max_iter=max_iter, callback=callback)
    model = LogisticRegression(solver=gd)
    model.fit(X_train.values, y_train.values)
    # print(losses[-1])
    y_prob = model.predict_proba(X_train.values)

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    plot = go.Figure([go.Scatter(x=RANGE_PLOT, y=RANGE_PLOT, mode="lines", line=dict(color="black", dash='dash'),
                                 showlegend=False),
                      go.Scatter(x=fpr, y=tpr, mode='lines', text=thresholds, showlegend=False,
                                 hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                     layout=go.Layout(title=rf"ROC Curve- Logistic Regression - AUC={auc(fpr, tpr):.3f}",
                                      width=555, height=555, xaxis=dict(title="FPR"), yaxis=dict(title="TPR")))
    # plot.show()
    plot.write_html(f"gd_logistic_roc_lr{lr:.4f}.html")

    index_of_arg_max = np.argmax(tpr - fpr)
    alpha_max = thresholds[index_of_arg_max]
    model.alpha_ = alpha_max
    print("best alpha: ", alpha_max)
    best_loss = model.loss(X_test.values, y_test.values)
    print("best loss: ", best_loss)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from loss_functions import misclassification_error
    from cross_validate import cross_validate
    penalty = "l1"
    lambdas = OPTIONS_LIST_LAMBDA
    scores = np.zeros((len(lambdas), 2))
    for i, lam in enumerate(lambdas):
        learning_rate = FixedLR(lr)
        gd = GradientDescent(learning_rate=learning_rate, max_iter=max_iter)
        logistic = LogisticRegression(solver=gd, penalty=penalty, lam=lam, alpha=.5)
        scores[i] = cross_validate(estimator=logistic, X=X_train.values, y=y_train.values,
                                   scoring=misclassification_error)
        print(f'penalty: {penalty}, lambda: {lam}, train error: {scores[i, 0]}, test error: {scores[i, 1]}')

    fig = go.Figure(
        [go.Scatter(x=lambdas, y=scores[:, 0], name="Train Error"),
         go.Scatter(x=lambdas, y=scores[:, 1], name="Validation Error")],
        layout=go.Layout(
            title=r"$\text{Train and Validation errors (averaged over the k-folds)}$",
            xaxis=dict(title=r"$\lambda$", type="log"),
            yaxis_title=r"\text{Error Value}"))
    fig.write_html(f"lambda_logistic_cross_validation_errors.html")

    # fitting a model with the best lambda on the entire train set
    lam_opt = lambdas[np.argmin(scores[:, 1])]
    gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
    model = LogisticRegression(solver=gd, penalty=penalty, lam=lam_opt, alpha=.5) \
        .fit(X_train.values, y_train.values)
    print(f"\tPenalty: {penalty}\n"
          f"\tOptimal regularization parameter: {lam_opt}\n"
          f"\tModel achieved test error of {round(model.loss(X_test.values, y_test.values), 2)}\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
