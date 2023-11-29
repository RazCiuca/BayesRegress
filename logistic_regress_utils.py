"""
In this file we do approximate bayesian logistic regression in a
fully differentiable way

The goal is to implement bayesian logistic regression in a differentiable way.
Complete with computing the model likelihood with importance sampling, setting prior distributions,
and using samples from the posterior to predict likelihoods at specific points.
All differentiably, so that all our outputs can be passed through backprop in the pytorch engine.


"""
import pandas as pd
from torch.autograd import grad
from linregress_utils import *

def check_grad_hessian_correctness(data_x, data_y, w, prior_mu=None, prior_precision=None):
    """
    function that simply does the computation of the logistic regression gradient and hessian in two different
    ways and compares them. One is with pytorch autograd from the log likelihood, the other is with the known equations
    """
    # the thing we're trying to check
    gradient_0, hessian_0 = get_grad_hessian_logistic_reg(data_x, data_y, w, prior_mu=prior_mu, prior_precision=prior_precision)

    n_data = data_x.size(0)

    w = w.detach()
    w.requires_grad = True

    # =============== definition of log likelihood ====================
    mu = data_x @ w.T
    beta_x = t.cat([mu, t.zeros(n_data, 1)], dim=1)
    log_likelihood = t.sum(data_y * t.nn.functional.log_softmax(beta_x, dim=1), dim=1).sum()
    log_likelihood -= (w.flatten() - prior_mu) @ prior_precision @ (w.flatten() - prior_mu) / (2.0)

    gradient_1 = grad(log_likelihood, w, create_graph=True)[0].flatten()
    hessian_1 = []

    for i in range(0, gradient_1.size(0)):
        hessian_1.append(grad(gradient_1[i], w, retain_graph=True)[0].flatten())

    hessian_1 = t.stack(hessian_1, dim=0)

    # print(gradient_0)
    # print(gradient_1.detach())

    print(f"maximum difference between gradients: {t.max(t.abs(gradient_1.flatten() - gradient_0)).item()}")
    print(f"maximum difference between hessians: {t.max(t.abs(hessian_1 - hessian_0)).item()}")


def get_grad_hessian_logistic_reg(data_x, data_y, w, prior_mu=None, prior_precision=None):
    """
    Computes the gradient and hessian of a multinomial logistic regression with MVN prior from given data
    :param data_x:
    :param data_y:
    :param w:
    :param prior_mu:
    :param prior_precision:
    :return:
    """
    n_data = data_x.size(0)
    n_cat = w.size(0) + 1

    # shape [n_data, n_cat-1]
    mu = data_x @ w.T
    beta_x = t.cat([mu, t.zeros(n_data, 1)], dim=1)
    eta = t.nn.functional.softmax(beta_x, dim=1)
    eta_k = eta[:, :-1]

    # log_likelihood = t.sum(data_y*t.nn.functional.log_softmax(beta_x, dim=1))
    # print(f"log-likelihood: {log_likelihood}")

    # shape [n_data, n_cat-1]
    y_min_etak = data_y[:, :-1] - eta_k

    # size [n_cat-1, x_dim]
    gradient = y_min_etak.T @ data_x
    gradient = gradient.flatten()

    # shape [n_data, n_cat-1, n_cat-1]
    one_minus_eta = t.eye(n_cat - 1).unsqueeze(0) - eta_k.unsqueeze(2)

    # shape [n_cat-1, x_dim, n_cat-1, x_dim]
    hessian = - t.einsum("nl, nkl, ni, nj -> kilj", eta_k, one_minus_eta, data_x, data_x)

    # flatten it into [x_dim*(n_cat-1), x_dim*(n_cat-1)]
    hessian = hessian.flatten(2, 3).flatten(0, 1)

    if prior_mu is not None and prior_precision is not None:
        gradient += -prior_precision @ (w.flatten() - prior_mu)
        hessian = hessian - prior_precision

    return gradient, hessian

def fit_logreg(data_x, data_y, n_cat, fns, prior_mu=None, prior_precision=None, init_w=None, verbose=False):
    """
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
    gd_max_iter = 0
    newton_max_iter_no_grad = 50
    newton_iter_stop_eps = 1e-3
    newton_iter_with_grad = 1

    # has shape [n_data, n_cat]
    data_y = t.nn.functional.one_hot(data_y, num_classes=n_cat)
    # initial weight vector
    w = t.zeros(n_cat - 1, x_dim) if init_w is None else init_w.detach()

    # ========== COMPUTATION BEGINS ===========

    # Initial Gradient Descent:
    w.requires_grad = True
    optimizer = t.optim.SGD([w], lr=1e-3, momentum=0.9)

    for i in range(0, gd_max_iter):
        mu = data_x @ w.T
        beta_x = t.cat([mu, t.zeros(n_data, 1)], dim=1)
        log_likelihood = -t.sum(data_y*t.nn.functional.log_softmax(beta_x, dim=1), dim=1).mean()
        log_likelihood += (w.flatten() - prior_mu) @ prior_precision @ (w.flatten()-prior_mu)/(2.0*n_data)

        gradient, hessian = get_grad_hessian_logistic_reg(data_x, data_y, w, prior_mu, prior_precision)

        optimizer.zero_grad()
        log_likelihood.backward()
        if verbose:
            print(f"gd iter {i}, log-likelihood:{log_likelihood.item()}")
        optimizer.step()

    w = w.detach()

    # first take a bunch of steps without gradient
    with t.no_grad():
        for iter in range(0, newton_max_iter_no_grad):

            gradient, hessian = get_grad_hessian_logistic_reg(data_x, data_y, w, prior_mu, prior_precision)
            newton_step = - t.linalg.solve(hessian, gradient)
            w_map = w.flatten() + newton_step
            w = w_map.reshape(n_cat-1, x_dim)

            step_fraction = t.norm(newton_step)/(1e-7 + t.norm(w))

            if verbose:
                mu = data_x @ w.T
                beta_x = t.cat([mu, t.zeros(n_data, 1)], dim=1)
                log_likelihood = -t.sum(data_y * t.nn.functional.log_softmax(beta_x, dim=1), dim=1).mean()
                log_likelihood += (w.flatten() - prior_mu) @ prior_precision @ (w.flatten() - prior_mu) / (2.0*n_data)

                print(f"iter {iter}, step fraction: {step_fraction}, log_likelihood:{log_likelihood.item()}")
                L,Q = t.linalg.eigh(hessian)
                print(f"hessian max eigenvalue: {t.max(L)}")

            if step_fraction< newton_iter_stop_eps:
                if verbose:
                    print(f"breaking at iter {iter}")
                break

    for iter in range(0, newton_iter_with_grad):
        if verbose:
            print(f"hessian iter {iter}/{newton_iter_with_grad}")

        gradient, hessian = get_grad_hessian_logistic_reg(data_x, data_y, w, prior_mu, prior_precision)
        newton_step = - t.linalg.solve(hessian, gradient)
        w_map = w.flatten() + newton_step
        w = w_map.reshape(n_cat - 1, x_dim)

    optimal_w = w

    # ================== Sampling Computations Begins ==================
    # at this point we've approximated the posterior as a MVN with
    # mu = optimal_w
    # precision = - hessian
    if verbose:
        print("starting sampling computations")

    w_dim = x_dim*(n_cat-1)
    precision_n = -hessian
    L, Q = t.linalg.eigh(precision_n)
    cov_n_sqrt = (1.0 / t.sqrt(L)).unsqueeze(1) * Q.transpose(0, 1)
    multNormal = t.distributions.MultivariateNormal(t.zeros(w_dim), t.eye(w_dim))

    def sampling_fn(n_samples):

        # now we sample from a multivariate normal, and scale it by the posterior covariance sqrt
        # and add the posterior mean
        unit_normal_samples = multNormal.sample(t.Size([n_samples]))
        assert unit_normal_samples.size() == t.Size([n_samples, w_dim])

        # generate samples with the reparametrization trick, here mu_n is differentiable,
        samples = optimal_w.flatten().unsqueeze(0) + unit_normal_samples @ cov_n_sqrt
        assert samples.size() == t.Size([n_samples, w_dim])

        # samples have shape [n_samples, w_dim]
        return samples

    # predict with a point-estimate from the MAP
    def map_predict_fn(x):
        mu = apply_and_concat(x, fns) @ optimal_w.T

        eta = t.nn.functional.softmax(t.cat([mu, t.zeros(x.size(0), 1)], dim=1), dim=1)

        return eta

    # predict using full bayesian inference
    def predict_fn(x, n_samples=100):

        n_data = x.size(0)

        betas = sampling_fn(n_samples).reshape(n_samples, n_cat - 1, x_dim)

        # shape [n_data, n_samples, n_cat-1]
        mu = t.einsum("ni, kli -> nkl", apply_and_concat(x, fns), betas)

        eta = t.nn.functional.softmax(t.cat([mu, t.zeros(n_data, n_samples, 1)], dim=2), dim=2)

        # taking the average of the predictions across samples
        predictions = eta.mean(dim=1)

        return predictions

    # integrates the posterior multiplied by the likelihood function
    def get_log_model_likelihood(n_samples=1000):

        betas = sampling_fn(n_samples).reshape(n_samples, n_cat - 1, x_dim)
        # shape [n_data, n_samples, n_cat-1]
        mu = t.einsum("ni, kli -> nkl", apply_and_concat(x, fns), betas)

        eta = t.cat([mu, t.zeros(n_data, n_samples, 1)], dim=2)
        # shape [n_data, n_samples]
        log_likelihoods = t.sum(data_y.unsqueeze(1) * t.nn.functional.log_softmax(eta, dim=2), dim=2).sum(0)

        # computing the mean of the likelihood from the samples
        log_model_likelihood = t.logsumexp(log_likelihoods, dim=0) - t.log(t.tensor([n_samples]))

        return log_model_likelihood

    return w, predict_fn, map_predict_fn, get_log_model_likelihood

def get_MLE_prior_params_for_logreg(data_x, data_y, fns, n_cat, init_gamma=None, n_iter=None, verbose=False):

    # use a built-in pytorch optimiser to optimise the log model likelihood diagonal priors
    x_dim = apply_and_concat(data_x, fns).size(1)
    w_dim = x_dim*(n_cat-1)

    gamma = t.zeros(w_dim) if init_gamma is None else init_gamma
    gamma.requires_grad = True
    prior_mu = t.zeros(w_dim)
    optimizer = t.optim.SGD([gamma], lr=0.01, momentum=0.9)

    n_iter = 1000 if n_iter is None else n_iter

    for iter in range(0, n_iter):
        prior_precision = t.diag(t.exp(gamma))

        w, predict_fn, _, get_log_model_likelihood = fit_logreg(data_x, data_y, n_cat, fns,
                                                             prior_mu=prior_mu, prior_precision=prior_precision,
                                                             init_w=None, verbose=False)

        obj = -get_log_model_likelihood(n_samples=10000)

        if verbose:
            print(f"iter {iter}, log model likelihood: {-obj.item()}")

        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

    return prior_mu, t.diag(t.exp(gamma))


if __name__ == "__main__":

    # load IRIS dataset
    dataset = pd.read_csv('datasets/iris.csv')

    # transform species to numerics
    dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2

    x = t.tensor(dataset[dataset.columns[0:4]].values.astype(np.float32))
    y = t.tensor(dataset.species.values.astype(np.int64))

    n_cat = 3

    # x.requires_grad = True

    # fns = [lambda x: t.ones(x.size(0),1), lambda x: x]
    fns = [lambda x: get_flat_polynomials(x, 2)]

    w_dim = apply_and_concat(x, fns).size(1)*(n_cat-1)

    prior_mu = t.zeros(w_dim)
    prior_precision = 1e-1*t.eye(w_dim)

    prior_mu, prior_precision = get_MLE_prior_params_for_logreg(x, y, fns, n_cat, n_iter=500, verbose=True)

    w, predict_fn, map_predict_fn, get_log_model_likelihood = fit_logreg(x,y,3, fns, prior_mu=prior_mu, prior_precision=prior_precision, verbose=True)

    predictions = predict_fn(x, n_samples=1000)

    print(f"model likelihood: {get_log_model_likelihood()}")

