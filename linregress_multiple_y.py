
import torch as t
import numpy as np
import matplotlib.pyplot as plt

def bayes_regress_multiple_y(data_x, data_y, fns, prior_mu, prior_precision, a_0, b_0):
    """
    :param data_x: Tensor(n_data, size_x)
    :param data_y: Tensor(n_data, size_y)
    :param fns: list of functions to apply row-wise to data_x to use as regression variables
    :param prior_mu: Tensor(size_y, size_x), the prior mean
    :param prior_precision: Tensor(size_y, size_x, size_x)
    :param a_0: Tensor(size_y) prior parameter for Inv-Normal distribution over the noise level sigma^2
    :param b_0, Tensor(size_y) prior parameter for Inv-Normal distribution over the noise level sigma^2
    :return: dict with the following fields: 'fns', 'mu_n', 'lambda_n',
                                             'a_n', 'b_n', 'mle_var',
                                             'sampling_fn', 'predict_fn',
                                             'entropy', 'log_model_ev'

    see https://en.wikipedia.org/wiki/Bayesian_linear_regression for equations

    This bayesian regression allows us to compute a posterior over P(beta, sigma^2 | X, y),
    which is the probability distribution of our linear parameters and the unknown noise level sigma^2
    """

    # apply the transformations in fns to the data
    data_x = apply_and_concat(data_x, fns)

    x_dim = data_x.size(1)
    y_dim = data_y.size(1)
    n_data = data_x.size(0)

    # quantity used in further computations
    xTx = data_x.T @ data_x

    # ==========================================================================
    # these are the posterior parameters for p(beta|sigma^2, X, y)
    # this distribution is N(mu_n, sigma^2 * (precision_n)^(-1)), notice that we still depend on the unknown sigma^2
    precision_n = xTx.unsqueeze(0) + prior_precision
    assert precision_n.size() == t.Size([y_dim, x_dim, x_dim])

    # We now use linalg.solve to compute A.inverse() @ B more reliably
    # this is the posterior mean for the parameters
    prior_prec_times_mu = t.einsum('kij, kj -> ki', prior_precision, prior_mu)
    mu_n = t.linalg.solve(precision_n, (data_y.T @ data_x + prior_prec_times_mu))
    assert mu_n.size() == t.Size([y_dim, x_dim])

    # ==============================
    # this is the posterior covariance, not yet scaled by sigma
    # we need to find A such that A dot A^T = cov_n = precision_n.inverse()
    # to do this we find the eigenvectors and values of precision_n, invert the eigenvalues
    # and take their square root, and we're done.
    # not that here we have cov_n = cov_n_sqrt @ cov_n_sqrt.T
    # we need this quantity to be able to sample from the gaussian differentiably
    L, Q = t.linalg.eigh(precision_n)
    cov_n_sqrt = (1.0/t.sqrt(L)).unsqueeze(2) * Q
    assert L.size() == t.Size([y_dim, x_dim])
    assert Q.size() == t.Size([y_dim, x_dim, x_dim])
    assert cov_n_sqrt.size() == t.Size([y_dim, x_dim, x_dim])

    # ==========================================================================
    # computing posterior p(sigma^2| X, y), which is an Inv-Normal distribution
    # all of these should have shape [y_dim]
    term_0 = t.sum(data_y**2, dim=0)
    term_1 = t.einsum('ki, kij, kj -> k', prior_mu, prior_precision, prior_mu)
    term_2 = - t.einsum('ki, kij, kj -> k', mu_n, precision_n, mu_n)

    # these are the posterior parameters for the sigma^2 side of the posterior
    a_n = a_0 + n_data/2
    b_n = b_0 + 0.5 * (term_0 + term_1 + term_2)
    print()
    assert a_n.size() == t.Size([y_dim])
    assert b_n.size() == t.Size([y_dim])

    # ==========================================================================
    # here we define a function which samples from the posterior in a
    # differentiable way, using the reparametrisation trick, in order to predict
    # future samples.

    # the two distributions used for the reparam trick
    gamma_dist = t.distributions.gamma.Gamma(a_n, 1)
    multNormal = t.distributions.MultivariateNormal(t.zeros(x_dim), t.eye(x_dim))

    # ==========================================================================
    # computing P(data | model) = int P(data | beta, model) P(beta|model)

    log_model_evidence = x_dim/2 * t.log(t.ones(y_dim)*2*np.pi)
    # this is computing log( sqrt(det(precision_0)/det(precision_n)) )
    # log_model_evidence += 0.5 * (t.logdet(prior_precision) - t.sum(t.log(L)))
    log_model_evidence += 0.5 * (t.logdet(prior_precision) - t.logdet(precision_n))
    log_model_evidence += a_0 * t.log(b_0)
    log_model_evidence -= a_n * t.log(b_n)
    log_model_evidence += t.lgamma(a_n) - t.lgamma(a_0)

    assert log_model_evidence.size() == t.Size([y_dim])

    # this function samples parameters from the posterior in a differentiable manner
    # with the reparametrization trick
    def sampling_fn(n_samples):
        # need to also return the sigmas for those samples

        # first sample from a Gamma, then invert it, then scale it
        gamma_samples = gamma_dist.sample(t.Size([n_samples]))

        # these are the sigma^2 samples from our posterior
        inv_gamma_samples = b_n.unsqueeze(0) * 1.0/gamma_samples
        sigma_samples = t.sqrt(inv_gamma_samples)
        assert sigma_samples.size() == t.Size([n_samples, y_dim])

        # now we sample from a multivariate normal, and scale it by the posterior covariance sqrt
        # scale it by the sigma samples we have
        # and add the posterior mean
        unit_normal_samples = multNormal.sample(t.Size([n_samples, y_dim]))
        assert unit_normal_samples.size() == t.Size([n_samples, y_dim, x_dim])

        # generate samples with the reparametrization trick, here mu_n is differentiable,
        # sigma_samples is differentiable through b_n, and cov_n_sqrt is differentiable too
        samples = mu_n.unsqueeze(0) + t.einsum('nk, nkj, kij -> nki', sigma_samples, unit_normal_samples, cov_n_sqrt)
        assert samples.size() == t.Size([n_samples, y_dim, x_dim])

        # samples have shape [n_samples, y_dim, x_dim], [n_samples, y_dim]
        return samples, sigma_samples

    # samples the model to give predictions with error bars, both on in a differentiable
    # manner, given data at which you want to predict, and a given precision level
    # we give two uncertainties: the uncertainty of the mean, and the predicted aleatoric data variance
    # when our functions are linear, this is just the same as predicting with mu_n, but this is
    # different if we're using nonlinear functions in the inputs
    def predict_fn(predict_data_x, n_samples):
        """
        :param predict_data_x: tensor(n_data, size_x)
        :param n_samples: int
        :return: ave_pred : tensor(n_data, y_dim), std_pred : tensor(n_data, y_dim)
        """

        # generate n_samples from the posterior:
        betas, sigmas = sampling_fn(n_samples)
        assert betas.size() == t.Size([n_samples, y_dim, x_dim])

        # predict using each sample at all points
        preds = t.einsum('ni, mki -> nmk', apply_and_concat(predict_data_x, fns), betas)
        assert preds.size() == t.Size([n_data, n_samples, y_dim])

        # need to add the samples in quadrature to get the real variance of the output

        # return preds.mean(dim=1), t.sqrt(preds.var(dim=1) + t.mean(sigmas**2))
        return preds.mean(dim=1), t.sqrt(preds.var(dim=1) + t.unsqueeze(b_n/(a_n-1), dim=0))

    # computes the entropy of the posterior in a differentiable manner
    def entropy_fn():
        # H(X,Y) = H(X|Y) + H(Y)
        # so the entropy of P(beta, sigma^2 | Data) is
        # H(sigma^2) + H(beta | sigma^2)
        # we know H(sigma^2) since it's inverse-gamma
        # and we know the entropy of beta|sigma^2, which is multivariate normal

        # ======================================================================
        # entropy of the distribution for sigma^2, this is the entropy of an Inv-Gamma distribution
        # H(Y) = a_n + t.log(b_n) + t.log(gamma(a_n)) - (1+a_n) * digamma(a_n)
        H_sigma_2 = a_n + t.log(b_n) + t.lgamma(a_n) - (1+a_n)*t.digamma(a_n)

        # ======================================================================
        # entropy of the conditional distribution
        # H(beta|sigma^2) = 0.5 ln det (2*pi*e * Cov)
        # but Cov = sigma^2 * inv_prec_n, and we already know the eigenvalues for that
        # so det (2*pi*e * Cov) = (2*pi*e * sigma^2)**dim / prod(L)
        # ln of this will give dim * ln (2*pi*e * sigma^2) + sum(ln(L))
        # so we need E[ln(X)] where X ~ inv-gamma(a,b), this is equal to
        # E[ln(X)] = ln(b) - psi(a), where psi is the digamma function
        E_ln_sigma_2 = t.log(b_n) - t.digamma(a_n)

        # H_beta_given_sigma_2 = 0.5 * (t.sum(-t.log(L)) + x_dim * (E_ln_sigma_2 + np.log(2 * t.pi * t.e)))
        H_beta_given_sigma_2 = 0.5 * (-t.logdet(precision_n) + x_dim * (E_ln_sigma_2 + np.log(2 * t.pi * t.e)))

        # returns with shape [y_dim]
        return H_beta_given_sigma_2 + H_sigma_2

    return {"fns": fns,
            "mu_n": mu_n,
            "precision_n": precision_n,
            'a_n': a_n,
            'b_n': b_n,
            'mle_var' : b_n/(a_n-1),
            'sampling_fn': sampling_fn,
            'predict_fn': predict_fn,
            'entropy_fn': entropy_fn,
            'log_model_ev': log_model_evidence
            }

def tril_flatten(tril, offset=0):
    N = tril.size(-1)
    indices = t.tril_indices(N, N, offset=offset)
    indices = N * indices[0] + indices[1]
    return tril.flatten(-2)[..., indices]

def get_flat_quadratic(x):
    """
    Given an input with shape [n_data, x_dim], return all the elements needed for quadratic regression
    :param x:
    :return:
    """
    n_data = x.size(0)

    cross_elements = t.einsum('ni, nj -> nij', x, x)

    return t.cat([t.ones(n_data, 1), x, tril_flatten(cross_elements)], dim=1)

def get_flat_pairwise_cos_sin(x, max_k):
    """
    assume data will be normalised, so set the boundary to like [-pi, pi]

    for each pair of i,j with i =/= j, we need to get the linear combination

    k1 x_i + k2 x_j for each k1 = [0, 1, ... N_k] and k2=[0, 1, ..., N_l]

    we want to return something of shape [n_data, ]

    :param x:
    :return:
    """
    x_dim = x.size(1)
    new_x = x.unsqueeze(2).repeat(1, 1, x_dim)

    # these are the pairs of combinations of x_i's, excluding pairs of the same
    # shape [n_data, 2, x_dim*(x_dim-1)/2]
    x_pairs = tril_flatten(t.stack([new_x, new_x.transpose(1, 2)], dim=1), offset=-1)

    ks = t.arange(0, max_k).unsqueeze(0).repeat(max_k, 1).float()
    # now this has shape [max_k, max_k, 2]
    ks = t.stack([ks, ks.transpose(0, 1)], dim=2)

    # shape [n_data, x_dim*(x_dim-1)/2, max_k * max_k]
    linear_combinations = t.einsum("nmi, klm -> nikl", x_pairs, ks).flatten(-2)

    # exclude the k = 0,0 coefficients for sin
    fourier_coefs = t.cat([t.cos(linear_combinations).flatten(-2),
                           t.sin(linear_combinations[:, :, 1:]).flatten(-2)], dim=1)

    return fourier_coefs

def get_flat_pairwise_radial_basis(x):
    raise NotImplemented

def apply_and_concat(x, fn_list):
    """
    takes a one dimensional tensor x and a list of functions, and stacks them

    :param x: tensor(n, dim_x)
    :param fn_list: list of functions
    :return: tensor(n, len(fn_list))
    """
    return t.cat([fn(x) for fn in fn_list], dim=1)


if __name__ == "__main__":
    x = t.randn(4, 3)
    x_dim = x.size(1)

    max_k = 5

    new_x = x.unsqueeze(2).repeat(1, 1, x_dim)

    # these are the pairs of combinations of x_i's, excluding pairs of the same
    # shape [n_data, 2, x_dim*(x_dim-1)/2]
    x_pairs = tril_flatten(t.stack([new_x, new_x.transpose(1, 2)], dim=1), offset=-1)

    ks = t.arange(0, max_k).unsqueeze(0).repeat(max_k, 1).float()
    # now this has shape [max_k, max_k, 2]
    ks = t.stack([ks, ks.transpose(0, 1)], dim=2)

    # shape [n_data, x_dim*(x_dim-1)/2, max_k * max_k]
    linear_combinations = t.einsum("nmi, klm -> nikl", x_pairs, ks).flatten(-2)

    # exclude the k = 0,0 coefficients for sin
    fourier_coefs = t.cat([t.cos(linear_combinations).flatten(-2),
                                  t.sin(linear_combinations[:, :, 1:]).flatten(-2)], dim=1)

# testing prediction
if __name__ == "__main__0":
    data_noise = 0.1

    coefs_0 = t.randn(3)
    coefs_1 = t.randn(3)

    fns = [lambda x: t.ones(x.size()), lambda x: x, lambda x: x ** 2]
    size_x = len(fns)

    data_x = t.arange(-1, 1, 0.01).unsqueeze(1)
    data_y_0 = coefs_0[0] + coefs_0[1] * data_x + coefs_0[2] * data_x ** 2 + t.sin(2*data_x)
    data_y_0 += t.randn(data_y_0.size()) * data_noise
    data_y_0 = data_y_0.reshape(-1, 1)

    data_y_1 = coefs_1[0] + coefs_1[1] * data_x + coefs_1[2] * data_x ** 2
    data_y_1 += t.randn(data_y_1.size()) * data_noise
    data_y_1 = data_y_1.reshape(-1, 1)

    data_y = t.cat([data_y_0, data_y_1], dim=1)

    y_dim = data_y.size(1)

    sol_dict = bayes_regress_multiple_y(data_x, data_y, fns,
                                   prior_mu=t.zeros(y_dim, size_x),
                                   prior_precision=1e-7 * t.eye(size_x, size_x).unsqueeze(0).repeat([y_dim, 1, 1]),
                                   a_0=0.1*t.ones(y_dim),
                                   b_0=t.ones(y_dim))

    predict_y_mean, predict_y_std = sol_dict['predict_fn'](data_x, n_samples=1000)

    predict_y_mean_0 = predict_y_mean[:, 0]
    predict_y_std_0 = predict_y_std[:, 0]

    predict_y_mean_1 = predict_y_mean[:, 1]
    predict_y_std_1 = predict_y_std[:, 1]

    print(predict_y_mean.size())

    plt.scatter(data_x.squeeze().numpy(), data_y_0.numpy())
    plt.scatter(data_x.squeeze().numpy(), data_y_1.numpy())

    plt.plot(data_x.squeeze().numpy(), predict_y_mean_0.numpy(), color='red')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_0 + 1*predict_y_std_0).numpy(), color='blue')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_0 - 1*predict_y_std_0).numpy(), color='blue')

    plt.plot(data_x.squeeze().numpy(), predict_y_mean_1.numpy(), color='red')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_1 + 1 * predict_y_std_1).numpy(), color='blue')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_1 - 1 * predict_y_std_1).numpy(), color='blue')

    plt.show()