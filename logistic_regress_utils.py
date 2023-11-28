"""
In this file we do approximate bayesian logistic regression in a
fully differentiable way

The goal is to implement bayesian logistic regression in a differentiable way.
Complete with computing the model likelihood with importance sampling, setting prior distributions,
and using samples from the posterior to predict likelihoods at specific points.
All differentiably, so that all our outputs can be passed through backprop in the pytorch engine.


"""
import pandas as pd
from linregress_utils import *

def fit_logreg(data_x, data_y, n_cat, fns, prior_mu=None, prior_precision=None):
    """
    todo: by convention the prior_precision is positive_definite, but in this algorithm we need the prior hessian,
    which will be negative-definite because we're maximizing a function here
    :param data_x:
    :param data_y:
    :param n_cat:
    :param fns:
    :param prior_mu:
    :param prior_precision:
    :return:
    """
    data_x = apply_and_concat(data_x, fns)
    x_dim = data_x.size(1)
    n_data = data_x.size(0)
    gd_n_iter = 1000
    gd_lr=0.01
    newton_n_iter = 50

    # has shape [n_data, n_cat]
    data_y = t.nn.functional.one_hot(data_y, num_classes=n_cat)
    # initial random vector
    w = t.zeros(n_cat - 1, x_dim)

    # ========== COMPUTATION BEGINS ===========

    # FIRST, normal GD

    with t.no_grad():

        for iter in range(0, gd_n_iter):

            # shape [n_data, n_cat-1]
            mu = data_x @ w.T

            eta_k = t.nn.functional.softmax(t.cat([mu, t.zeros(n_data, 1)], dim=1), dim=1)[:, :-1]

            # shape [n_data, n_cat-1]
            y_min_etak = data_y[:, :-1] - eta_k

            # size [n_cat-1, x_dim]
            gradient = y_min_etak.T @ data_x

            w += gd_lr * gradient

    # Second, newton iteration

    with t.no_grad():

        for iter in range(0, newton_n_iter):

            print("weights:")
            print(w)

            # shape [n_data, n_cat-1]
            mu = data_x @ w.T
            eta_k = t.nn.functional.softmax(t.cat([mu, t.zeros(n_data, 1)], dim=1), dim=1)[:, :-1]

            # shape [n_data, n_cat-1]
            y_min_etak = data_y[:, :-1] - eta_k

            # # shape [x_dim, n_cat-1]
            # if prior_mu is not None:
            #     grad_from_prior = prior_precision @ (w.flatten() - prior_mu)
            #
            # # size [n_cat-1, x_dim]
            # gradient = y_min_etak.T @ data_x
            # computation until this point works fine
            # shape [n_cat-1, x_dim,n_cat-1, x_dim]
            # hessian = - t.einsum("nl, nk, ni, nj -> kilj", eta_k, y_min_etak, data_x, data_x)
            #
            # # flatten it into [x_dim*(n_cat-1), x_dim*(n_cat-1)]
            # hessian = hessian.flatten(2, 3).flatten(0, 1)
            # if prior_precision is not None:
            #     hessian = hessian - prior_precision
            #
            # # print(hessian)
            #
            # w_map = w.flatten() - t.linalg.solve(hessian, gradient.flatten() + (grad_from_prior if prior_mu is not None else 0))
            # w = w_map.reshape(n_cat-1, x_dim)

            # =====================================================================

            # size [n_cat-1, x_dim]
            gradient = y_min_etak.T @ data_x

            # shape[n_cat - 1, x_dim, x_dim]
            hessian = - t.einsum("nk, nk, ni, nj -> kij", eta_k, y_min_etak, data_x, data_x)
            hessian -= 1e-2*t.eye(hessian.size(-1)).unsqueeze(0)

            print(f"det hessians: {t.det(hessian)}")

            w = w - 0.1*t.linalg.solve(hessian, gradient)


    optimal_w = w

    def predict_fn(x):
        mu = apply_and_concat(x, fns) @ optimal_w.T

        eta = t.nn.functional.softmax(t.cat([mu, t.zeros(n_data, 1)], dim=1), dim=1)

        return eta


    return w, predict_fn


if __name__ == "__main__":

    # load IRIS dataset
    dataset = pd.read_csv('datasets/iris.csv')

    # transform species to numerics
    dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2

    x = t.tensor(dataset[dataset.columns[0:4]].values.astype(np.float32))
    y = t.tensor(dataset.species.values.astype(np.int64))

    # x = t.cat([x, t.randn(50, 4)])
    # y = t.cat([y, t.multinomial(t.tensor([1.0,1.0,1.0]), 50, replacement=True)])

    fns = [lambda x: t.ones(x.size()), lambda x: x]

    prior_mu = t.zeros(16)
    prior_precision = 1*t.eye(16)

    w, predict_fn = fit_logreg(x,y,3, fns, prior_mu=prior_mu, prior_precision=prior_precision)

