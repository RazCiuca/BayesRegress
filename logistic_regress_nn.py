
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


from logistic_regress_utils import get_grad_hessian_logistic_reg, apply_and_concat

class logisticRegression(nn.Module):

    def __init__(self, n_classes, gamma = 0.7, fns = None, prior_mu = None, prior_precision=None):
        super(logisticRegression, self).__init__()

        self.n_cat = n_classes
        self.fns = [lambda x: t.ones(x.size(0),1), lambda x: x] if fns is None else fns
        self.gamma = gamma # discounting factor for learning the precision

        # to be set once we see data
        self.w = None   # is going to be shape [n_cat-1, x_dim]
        self.x_dim = None
        self.w_dim = None

        self.prior_mu = prior_mu
        self.prior_precision = prior_precision
        self.precision_level = 1e-1

        self.newton_max_iter_no_grad = 50
        self.newton_iter_with_grad = 2
        self.newton_iter_stop_eps = 1e-3

    def initialize(self, x_dim):
        self.x_dim = x_dim
        self.w_dim = x_dim * (self.n_cat - 1)
        self.w = t.zeros(self.n_cat - 1, self.x_dim)
        if self.prior_mu is None:
            self.prior_mu = t.zeros(self.w_dim)

        if self.prior_precision is None:
            self.prior_precision = self.precision_level * t.eye(self.w_dim)

    def discount_precision(self, gamma):
        L, Q = t.linalg.eigh(self.prior_precision)
        # print(f"eigenvalues:{L}")
        self.prior_precision = (Q @ t.diag(L*gamma + (1-gamma)*self.precision_level) @ Q.T).detach()

    def update_posterior(self, x, y):

        # changes prior precision to be wider
        self.discount_precision(self.gamma)

        data_y = t.nn.functional.one_hot(y, num_classes=self.n_cat)
        data_x = x

        n_data = x.size(0)

        # first take a bunch of steps without gradient
        with t.no_grad():
            for iter in range(0, self.newton_max_iter_no_grad):

                gradient, hessian = get_grad_hessian_logistic_reg(data_x, data_y, self.w, self.prior_mu, self.prior_precision)
                newton_step = - t.linalg.solve(hessian, gradient)
                w_map = self.w.flatten() +  0.1*newton_step
                self.w = w_map.reshape(self.n_cat - 1, self.x_dim)

                step_fraction = t.norm(newton_step) / (1e-7 + t.norm(self.w))

                if True:
                    mu = data_x @ self.w.T
                    beta_x = t.cat([mu, t.zeros(n_data, 1)], dim=1)
                    log_likelihood = -t.sum(data_y * t.nn.functional.log_softmax(beta_x, dim=1), dim=1).mean()
                    log_likelihood += (self.w.flatten() - self.prior_mu) @ self.prior_precision @ (self.w.flatten() - self.prior_mu) / (
                                2.0 * n_data)

                    print(f"iter {iter}, step fraction: {step_fraction}, log_likelihood:{log_likelihood.item()}")
                    L, Q = t.linalg.eigh(hessian)
                    print(f"hessian max eigenvalue: {t.max(L)}")

                if step_fraction < self.newton_iter_stop_eps:
                    break

        for iter in range(0, self.newton_iter_with_grad):
            gradient, hessian = get_grad_hessian_logistic_reg(data_x, data_y, self.w, self.prior_mu,
                                                              self.prior_precision)
            newton_step = - t.linalg.solve(hessian, gradient)
            w_map = self.w.flatten() + 0.1*newton_step
            self.w = w_map.reshape(self.n_cat - 1, self.x_dim)

        self.prior_mu = self.w.flatten().clone().detach()
        self.prior_precision = (-hessian).clone().detach()

    def map_predict_fn(self, x):
        """
        predicts on data x with the MAP estimate of a logistic regression
        :param x:
        :return:
        """
        mu = apply_and_concat(x, self.fns) @ self.w.T
        eta = t.nn.functional.softmax(t.cat([mu, t.zeros(x.size(0), 1)], dim=1), dim=1)
        return eta

    def forward(self, x, y=None):
        """
        :param x: data on which to predict
        :param y: if labels y are also given, learn partly with those
        :return: MAP estimate predictions from a logistic regression with prior mu and precision matrix
        """
        if y is not None:
            new_x = apply_and_concat(x, self.fns)

            if self.w_dim is None:
                self.initialize(new_x.size(1))

            # use y to learn
            self.update_posterior(new_x, y)

        return self.map_predict_fn(x)


if __name__ == "__main__":

    n_cat = 3
    # load IRIS dataset
    dataset = pd.read_csv('datasets/iris.csv')

    # transform species to numerics
    dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2

    x = t.tensor(dataset[dataset.columns[0:4]].values.astype(np.float32))
    y = t.tensor(dataset.species.values.astype(np.int64))

    regressor = logisticRegression(n_classes=n_cat)

    chunk_size = int(x.size(0)/10)

    for i in range(0, 4):

        index_start = chunk_size*i
        index_stop = chunk_size*(i+1)

        x_batch = x[index_start:index_stop].detach()
        y_batch = y[index_start:index_stop].detach()

        x_batch.requires_grad = True

        predictions = regressor.forward(x_batch, y_batch)

        obj = predictions.std()

        obj.backward()
        # print(x_batch.grad)

