"""
In this file we do approximate bayesian logistic regression in a
fully differentiable way

multi-class regression can be done with multiple logistic regressions on
the individual classes

steps in bayesian logistic regression:

- set up the loss function
- compute the gradient and hessian explicitely (without autograd)
- get the minimum of the loss offgraph
- compute predictions by sampling from posterior and averaging

todo - treat the case where we have k possible choices
todo - treat the case where we want to use a prior, but possibly reduce its importance a little bit

See Murphy 2011 pages 248 to 253 for equations for the Newton algorithm solution to logistic regreession

Goal: do quadratic logistic regression using newton's method in a differentiable way, so that we can
place it at the last layer of a neural network, and thus get good results this way.

if you have n features at the end of the neural network, you'll get 1 + n + n*(n-1)/2 = n*(n+1)/2 + 1 = N features for
regression, so the hessian will have size N x N, if n=100, we get N = 5051

"""

import torch as t
import numpy as np
from linregress_utils import *

def bayesian_logistic_regression(data_x, data_y, n_cat, fns, prior_mu, prior_precision):
    """
    :param data_x: tensor(n_data, x_size)
    :param data_y: LongTensor(n_data), categories
    :param n_cat : integer, number of categories
    :param fns: list of functions to use for fitting
    :param prior_mu: tensor(len(fns)), prior mean for the parameters
    :param prior_precision: tensor(len(fns), len(fns)), prior precision for the parameters
    :return: dict with the following fields: 'fns', 'mu_n', 'lambda_n', 'sampling_fn', 'predict_fn', 'entropy', 'log_model_ev'
    """

    # apply functions to raw data
    data_x = apply_and_concat(data_x, fns)
    x_dim = data_x.size(1)
    n_data = data_x.size(0)

    # make y_data tensors for each different category
    # has shape [n_data, n_cat]
    data_y = t.nn.functional.one_hot(data_y, num_classes=n_cat)

    # initialise the weights to random
    w = t.randn(n_cat, x_dim)

    # mu = logistic(w^T dot X)
    # gradient = X^T dot (mu - Y)
    # Hessian = X^T dot (diag(mu*(1-mu))) dot X

    # Step 1: taking a few gradient hessian steps to get to the minimum without creating an autodiff graph

    with t.no_grad:
        # shape [n_data, n_cat]
        mu = t.sigmoid(data_x @ w.T)
        # shape [dim_x, n_cat]
        gradient = data_x.T @ (mu - data_y)

        # we're doing a sum over a tensor with dim [n_data, n_cat, dim_x, dim_x]
        # and keeping only the last 3 dimensions
        hessian = t.einsum("nk, ni, nj -> kij", mu*(1-mu), data_x, data_x)

        # now compute the newton step update

    # Step 2: compute the hessian at the minimum with autodiff on

    # step 3: take an extra step, now with autodiff on, now we have  the minimum

    # define sampling, prediction, log model evidence, model entropy functions


