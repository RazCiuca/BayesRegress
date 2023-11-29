
import torch as t
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from logistic_regress_utils import *

if __name__ == "__main__":

    training_data = datasets.MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="datasets",
        train=False,
        download=True,
        transform=ToTensor()
    )

    data_x_train = training_data.data.float()
    data_x_train = data_x_train.reshape(data_x_train.size(0), -1)
    data_y_train = training_data.targets

    data_x_test = test_data.data.float()
    data_x_test = data_x_test.reshape(data_x_test.size(0), -1)
    data_y_test = test_data.targets

    # removing mean and std:
    mean = data_x_train.mean(dim=0)
    std = data_x_train.std(dim=0)
    data_x_train = (data_x_train - mean.unsqueeze(0)) / (1e-7 + std.unsqueeze(0))
    data_x_test = (data_x_test - mean.unsqueeze(0)) / (1e-7 + std.unsqueeze(0))

    n_cat = 10
    fns = [lambda x: t.ones(x.size(0), 1), lambda x: x]

    # fns = [lambda x: get_flat_polynomials(x, 1)]
    w_dim = apply_and_concat(data_x_train, fns).size(1)*(n_cat-1)

    # fitting the bayesian model:
    # prior_mu, prior_precision = get_MLE_prior_params_for_logreg(data_x_train, data_y_train, fns, n_cat, n_iter=500, verbose=True)

    prior_mu = t.zeros(w_dim)
    prior_precision = t.eye(w_dim)

    w, predict_fn, map_predict_fn, get_log_model_likelihood = fit_logreg(data_x_train, data_y_train, n_cat, fns, prior_mu=prior_mu,
                                                         prior_precision=prior_precision, verbose=True)

    print("starting MAP predictions:")
    map_predictions = map_predict_fn(data_x_test)
    print(f"test set accuracy with MAP:{(map_predictions.argmax(1) == data_y_test).float().mean()}")

    # predictions:
    print("starting predictions:")
    predictions = predict_fn(data_x_test, n_samples=500)
    print(f"test set accuracy:{(predictions.argmax(1) == data_y_test).float().mean()}")

