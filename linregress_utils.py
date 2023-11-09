"""
This file defines differentiable pytorch functions which allow us to do
bayesian linear regression with automatic variable selection in a differentiable way.

todo_diff_regress_layer: (done) fix sampling_fn
todo_diff_regress_layer: write predict_fn
todo_diff_regress_layer: (done)write entropy_fn
todo_diff_regress_layer: write optim routine to return the optimal state to explore for infogain
todo_diff_regress_layer: make the bayesian_regression work for arbitrary f_k(x), and provide methods for
      setting these f_k to the appropriate ones for polynomial regression
todo_diff_regress_layer: unit tests to make sure all works
todo_diff_regress_layer: refactor stuff into smaller functions

todo_diff_regress_layer: automatic variable selection by integrating over the prior precision?

"""
import torch as t
import numpy as np

def bayesian_regression(data_x, data_y, prior_mu, prior_precision, a_0=1, b_0=1,  degree_polynomial=1):
    """
    :param data_x: Tensor(n_data, size_x)
    :param data_y: Tensor(n_data, size_y)
    :param prior_mu: Tensor(size_x, size_y), the prior mean
    :param prior_precision: Tensor(size_x, size_x, size_y)
    :param a_0: prior parameter for Inv-Normal distribution over the noise level sigma^2
    :param b_0, prior parameter for Inv-Normal distribution over the noise level sigma^2
    :param degree_polynomial: the degree of the polynomial to fit, determines what size_x is
    :return: dict with the following fields: 'mu_n', 'lambda_n', 'a_n', 'b_n', 'sampling_fn', 'predict_fn', 'entropy'

    we return the posterior mean and precision, as well as a function which allows for computing p(y|x,Data), the
    prediction for a point averaged conditioned on x and the Data, which averages over our uncertainty

    see https://en.wikipedia.org/wiki/Bayesian_linear_regression for equations

    This bayesian regression allows us to compute a posterior over P(beta, sigma^2 | X, y),
    which is the probability distribution of our linear parameters and the unknown noise level sigma^2
    """

    x_dim = data_x.size(1)

    # MLE for the parameters
    beta = t.linalg.pinv(data_x) @ data_y

    # quantity used in further computations
    xTx = data_x.T @ data_x

    # ==========================================================================
    # these are the posterior parameters for p(beta|sigma^2, X, y)
    # this distribution is N(mu_n, sigma^2 * (precision_n)^(-1)), notice that we still depend on the unknown sigma^2
    precision_n = xTx + prior_precision
    inv_prec_n = t.inverse(precision_n)
    mu_n = inv_prec_n @ (xTx @ beta + prior_precision @ prior_mu)

    # ==============================
    # this is the posterior covariance, not yet scaled by sigma
    # we need to find A such that A dot A^T = cov_n
    # to do this we find the eigenvectors and values of cov_n, and take the sqrt
    # of the eigenvalues
    L, Q = t.linalg.eigh(inv_prec_n)
    cov_n_sqrt = t.sqrt(L) * Q
    # not that here we have cov_n = cov_n_sqrt @ cov_n_sqrt.T

    # ==========================================================================
    # computing posterior p(sigma^2| X, y), which is an Inv-Normal distribution

    term_0 = data_y.T @ data_y
    term_1 = prior_mu.T @ prior_precision @ prior_mu
    term_2 = - mu_n.T @ precision_n @ mu_n

    # these are the posterior parameters for the sigma^2 side of the posterior
    a_n = a_0 + x_dim/2
    b_n = b_0 + 0.5 * (term_0 + term_1 + term_2)

    # ==========================================================================
    # here we define a function which samples from the posterior in a
    # differentiable way, using the reparametrisation trick, in order to predict
    # future samples.

    # the two distributions used for the reparam trick
    gamma_dist = t.distributions.gamma.Gamma(a_n, 1)
    multNormal = t.distributions.MultivariateNormal(t.zeros(x_dim), t.eye(x_dim))

    # this function samples parameters from the posterior in a differentiable manner
    # with the reparametrization trick
    def sampling_fn(n_samples):

        # first sample from a Gamma, then invert it, then scale it
        gamma_samples = gamma_dist.sample(t.Size(n_samples))

        # these are the sigma^2 samples from our posterior
        inv_gamma_samples = b_n * 1.0/gamma_samples
        sigma_samples = t.sqrt(inv_gamma_samples)

        # now we sample from a multivariate normal, and scale it by the posterior covariance sqrt
        # scale it by the sigma samples we have
        # and add the posterior mean
        unit_normal_samples = multNormal.sample(t.Size(n_samples))

        # generate samples with the reparametrization trick, here mu_n is differentiable,
        # sigma_samples is differentiable through b_n, and cov_n_sqrt is differentiable too
        samples = mu_n + (( sigma_samples * unit_normal_samples) @ cov_n_sqrt.T).T

        # samples have shape [n_samples, data_x.size(0)]
        return samples

    # samples the model to give predictions with error bars, both on in a differentiable
    # manner, given data at which you want to predict, and a given precision level
    # we give two uncertainties: the uncertainty of the mean, and the predicted aleatoric data variance
    # when our functions are linear, this is just the same as predicting with mu_n, but this is
    # different if we're using nonlinear functions in the inputs
    def predict_fn(predict_data_x):
        pass

    # computes the entropy of the posterior in a differentiable manner
    def entropy_fn():
        # H(X,Y) = H(X|Y) + H(Y)
        # so the entropy of P(beta, sigma^2 | Data) is
        # H(sigma^2) + H(beta | sigma^2)
        # we know H(sigma^2) since it's inverse-gamma
        # and we know the entropy of beta|sigma^2, which is multivariate normal

        # ======================================================================
        # entropy of the distribution for sigma^2, this is the entropy of an Inv-Gamma distribution
        # H(Y) = a_n + t.log(b_n) + t.log(gamma(a_n)) - (1-a_n) * digamma(a_n)
        H_sigma_2 = a_n + t.log(b_n) + t.lgamma(a_n) - (1-a_n)*t.digamma(a_n)

        # ======================================================================
        # entropy of the conditional distribution
        # H(beta|sigma^2) = 0.5 ln det (2*pi*e * Cov)
        # but Cov = sigma^2 * inv_prec_n, and we already know the eigenvalues for that
        # so det (2*pi*e * Cov) = (2*pi*e * sigma^2)**dim * prod(L)
        # ln of this will give dim * ln (2*pi*e * sigma^2) + sum(ln(L))
        # so we need E[ln(X)] where X ~ inv-gamma(a,b), this is equal to
        # E[ln(X)] = ln(b) - psi(a), where psi is the digamma function
        E_ln_sigma_2 = t.log(b_n) - t.digamma(a_n)

        H_beta_given_sigma_2 = 0.5 * (t.sum(t.log(L)) + x_dim * (E_ln_sigma_2 + np.log(2 * t.pi * t.e)))

        return H_beta_given_sigma_2 + H_sigma_2

    # todo_diff_regress_layer: don't return predicion_n, but the covariance scales by the mle for sigma
    return {"mu_n": mu_n,
            "lambda_n": precision_n,
            'a_n': a_n,
            'b_n': b_n,
            'sampling_fn': sampling_fn,
            'predict_fn': predict_fn,
            'entropy_fn': entropy_fn
            }

def optimal_infogain_bayesian_regress():
    """
    given datasets data_x and data_y, computes the next point which in expectation will
    lead to the greatest reduction in the entropy of the posterior distribution

    first, we compute the posterior parameters for data_x and data_y,
    then we can pass those parameters as detached tensors as priors to the bayesian_regression function

    for a candidate x, then, we need to sample a bunch of y-predictions from the current posterior,
    then compute the implied entropy reduction with this datapoint for each of those y-predictions,
    then take the derivative of that average with respect to x, and optimise that.

    This is then an MDP, and we need to optimise this thing.

    Up until now everything was exact or asymptotically exact,
    but now we need to find the derivative of the optimal x with respect to the prior parameters
    perhaps even finding the second derivatives here.

    Then, once we have an expression for the changes in posterior given a new point, we can
    solve the resulting MDP and find the sequence of points to explore in order to
    minimaly reduce the entropy of the posterior.

    :return:
    """


    pass