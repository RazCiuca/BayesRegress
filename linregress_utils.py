
import torch as t
import numpy as np
import matplotlib.pyplot as plt


def log_model_evidence(precision_0, precision_n, a_0, a_n, b_0, b_n):
    """
    computes the log of the model evidence for a bayesian linear regression model given
    the prior and posterior parameters
    """
    x_dim = precision_0.size(-1)
    y_dim = precision_n.size(0)

    log_model_evidence = x_dim / 2 * t.log(t.ones(y_dim) * 2 * np.pi)
    log_model_evidence += 0.5 * (t.logdet(precision_0) - t.logdet(precision_n))
    log_model_evidence += a_0 * t.log(b_0)
    log_model_evidence -= a_n * t.log(b_n)
    log_model_evidence += t.lgamma(a_n) - t.lgamma(a_0)

    # has shape [y_dim], for the model evidence for each dimension of y
    return log_model_evidence

def log_model_evidence_grad(data_x, data_y, gamma, a_0, b_0, y_at_x=None, x_at_x=None, y_2_sum=None, verbose=False):
    """
    computes the gradient of the log model evidence for each y dimension with respect to gamma, which are the
    log diagonal parameters of precision_0, i.e. precision_0 = t.diag(t.exp(gamma))
    and also with respect to a_0 and b_0
    """

    x_dim = data_x.size(1)
    y_dim = data_y.size(1)

    mu_0 = t.zeros(y_dim, x_dim)
    precision_0 = t.diag_embed(t.exp(gamma))

    # useful term that remains constant through computation of all gradients
    xTx = data_x.T @ data_x if x_at_x is None else x_at_x
    yTx = data_y.T @ data_x if y_at_x is None else y_at_x
    yTy = t.sum(data_y ** 2, dim=0) if y_2_sum is None else y_2_sum

    mu_n, precision_n, a_n, b_n = posterior_params_linregress(data_x, data_y, mu_0, precision_0, a_0, b_0,
                                                              y_at_x=yTx, x_at_x=xTx, y_2_sum=yTy)

    if verbose:
        print(f"model evidence: {log_model_evidence(precision_0, precision_n, a_0, a_n, b_0, b_n).sum().item()}")

    # inverse of each precision matrix
    # shape [y_dim, x_dim, x_dim]
    inv_prec_n = precision_n.inverse()

    # =========================================
    # derivative computation for gamma
    # =========================================

    # shape [y_dim, x_dim], intermediate term in the computation
    z = t.einsum("ij, ilj -> il ", yTx, inv_prec_n)

    gamma_grad = 0.5 * t.ones(y_dim, x_dim)
    gamma_grad -= 0.5 * t.exp(gamma) * t.diagonal(inv_prec_n, dim1=1, dim2=2)
    gamma_grad -= (a_n / (2 * b_n)).unsqueeze(1) * (z ** 2 * t.exp(gamma))

    # =========================================
    # derivative computation for a_0
    # =========================================
    a_0_grad = t.log(b_0) - t.log(b_n)
    a_0_grad += t.digamma(a_n) - t.digamma(a_0)

    # =========================================
    # derivative computation for b_0
    # =========================================
    b_0_grad = a_0 / b_0 - a_n / b_n

    return gamma_grad, a_0_grad, b_0_grad


# todo_rl: produces unstable answers, unsure if hessian computation is correct
def log_model_evidence_newton_iteration(data_x, data_y, gamma, a_0, b_0, y_at_x=None, x_at_x=None, y_2_sum=None, verbose=False):
    """
    computes the gradient of the log model evidence for each y dimension with respect to gamma, which are the
    log diagonal parameters of precision_0, i.e. precision_0 = t.diag(t.exp(gamma))
    and also with respect to a_0 and b_0

    then also computes the hessian of the log model evidence with respect to the vector [gamma_i, a_0, b_0]
    :param gamma: [y_dim, x_dim]
    :param a_0 : [y_dim]
    :param b_0 : [y_dim]
    :return the optimal gamma, a_0, b_0 for a newton iteration
    """

    x_dim = data_x.size(1)
    y_dim = data_y.size(1)

    mu_0 = t.zeros(y_dim, x_dim)
    precision_0 = t.diag_embed(t.exp(gamma))

    # useful term that remains constant through computation of all gradients
    xTx = data_x.T @ data_x if x_at_x is None else x_at_x
    yTx = data_y.T @ data_x if y_at_x is None else y_at_x
    yTy = t.sum(data_y ** 2, dim=0) if y_2_sum is None else y_2_sum

    mu_n, precision_n, a_n, b_n = posterior_params_linregress(data_x, data_y, mu_0, precision_0, a_0, b_0,
                                                              y_at_x=yTx, x_at_x=xTx, y_2_sum=yTy)

    # inverse of each precision matrix
    # shape [y_dim, x_dim, x_dim]
    inv_prec_n = precision_n.inverse()

    # shape [y_dim, x_dim], intermediate term in the computation
    z = t.einsum("ij, ilj -> il ", yTx, inv_prec_n)

    # useful for future computation
    # this is the matrix of Y.T @ X @ inv_prec @ E_ij @ inv_prec @ X.T @ Y
    zTz = z.unsqueeze(1) * z.unsqueeze(2)
    # zTz = t.einsum("mki, mkj -> mij ", z.unsqueeze(1), z.unsqueeze(1))

    exp_gamma = t.exp(gamma)

    # the derivative of b_n with respect to gamma_i
    # shape [y_dim, x_dim]
    d_bn_d_gamma = 0.5*exp_gamma * t.diagonal(zTz, dim1=1, dim2=2)

    gamma_grad = 0.5 * t.ones(y_dim, x_dim)
    gamma_grad -= 0.5 * exp_gamma * t.diagonal(inv_prec_n, dim1=1, dim2=2)
    gamma_grad -= (a_n / (2 * b_n)).unsqueeze(1) * d_bn_d_gamma

    a_0_grad = t.log(b_0) - t.log(b_n)
    a_0_grad += t.digamma(a_n) - t.digamma(a_0)

    b_0_grad = a_0 / b_0 - a_n / b_n

    # =================================
    # Hessian Computation Starts
    # =================================

    # shape [y_dim, x_dim, x_dim]
    exp_gamma_i_times_exp_gamma_j = exp_gamma.unsqueeze(1) * exp_gamma.unsqueeze(2)

    # shape [y_dim, x_dim, x_dim]
    gamma_hess = -0.5 * t.diag_embed(exp_gamma*t.diagonal(inv_prec_n, dim1=1, dim2=2), dim1=1, dim2=2)
    gamma_hess += 0.5 * exp_gamma_i_times_exp_gamma_j * (inv_prec_n**2)
    gamma_hess -= t.diag_embed((a_n/b_n).unsqueeze(1) * d_bn_d_gamma, dim1=1, dim2=2)
    gamma_hess += (a_n/b_n**2).reshape(-1, 1, 1) * (d_bn_d_gamma.unsqueeze(1) * d_bn_d_gamma.unsqueeze(2))
    gamma_hess += (a_n/b_n).reshape(-1, 1, 1) * exp_gamma_i_times_exp_gamma_j * inv_prec_n * zTz

    # shape [y_dim]
    a0_2_hess = t.special.polygamma(1, a_n) - t.special.polygamma(1, a_0)
    b0_2_hess = -a_0/b_0**2 + a_n/b_n**2
    a0_b0_hess = 1.0/b_0 - 1.0/b_n

    # shape [y_dim, x_dim]
    b_0_gamma_hess = (a_n/b_n**2).unsqueeze(1) * d_bn_d_gamma
    a_0_gamma_hess = (-1.0/b_n).unsqueeze(1) * d_bn_d_gamma

    hess_upper = t.cat([gamma_hess, a_0_gamma_hess.unsqueeze(2), b_0_gamma_hess.unsqueeze(2)], dim=2)

    hess_lower1 = t.cat([a_0_gamma_hess, a0_2_hess.unsqueeze(1), a0_b0_hess.unsqueeze(1)], dim=1)
    hess_lower2 = t.cat([b_0_gamma_hess, a0_b0_hess.unsqueeze(1), b0_2_hess.unsqueeze(1)], dim=1)

    # shape [y_dim, x_dim+2, x_dim+2]
    hessian = t.cat([hess_upper, hess_lower1.unsqueeze(1), hess_lower2.unsqueeze(1)], dim=1)

    # shape [y_dim, x_dim+2]
    grad_concat = t.cat([gamma_grad, a_0_grad.unsqueeze(1), b_0_grad.unsqueeze(1)], dim=1)

    newton_iter_solution = -t.linalg.solve(hessian, grad_concat)

    gamma_sol = newton_iter_solution[:, :-2]
    a_0_sol = newton_iter_solution[:, -2]
    b_0_sol = newton_iter_solution[:, -1]

    print(f"condition number:{t.linalg.cond(hessian)}")

    return gamma_sol, a_0_sol, b_0_sol, gamma_grad, a_0_grad, b_0_grad



def posterior_params_linregress(data_x, data_y, mu_0, precision_0, a_0, b_0, y_at_x=None, x_at_x=None, y_2_sum=None):
    """
    :param data_x: Tensor(n_data, size_x)
    :param data_y: ensor(n_data, size_y)
    :param mu_0: Tensor(size_y, size_x), the prior mean
    :param precision_0: Tensor(size_y, size_x, size_x) , prior precision matrix for the MVN
    :param a_0: Tensor(size_y) prior parameter for Inv-Normal distribution over the noise level sigma^2
    :param b_0, Tensor(size_y) prior parameter for Inv-Normal distribution over the noise level sigma^2
    :return: mu_n, precision_n, a_n, b_n   all tensors with same shapes as the equivalent priors
    """

    n_data = data_x.size(0)

    # quantity used in further computations
    xTx = data_x.T @ data_x if x_at_x is None else x_at_x
    yTx = data_y.T @ data_x if y_at_x is None else y_at_x
    yTy = t.sum(data_y ** 2, dim=0) if y_2_sum is None else y_2_sum

    precision_n = xTx.unsqueeze(0) + precision_0

    prior_prec_times_mu = t.einsum('kij, kj -> ki', precision_0, mu_0)
    mu_n = t.linalg.solve(precision_n, (yTx + prior_prec_times_mu))

    term_0 = yTy
    term_1 = t.einsum('ki, kij, kj -> k', mu_0, precision_0, mu_0)
    term_2 = - t.einsum('ki, kij, kj -> k', mu_n, precision_n, mu_n)

    # these are the posterior parameters for the sigma^2 side of the posterior
    a_n = a_0 + n_data / 2
    b_n = b_0 + 0.5 * (term_0 + term_1 + term_2)

    return mu_n, precision_n, a_n, b_n

def posterior_entropy(precision_n, a_n, b_n):
    """
    :param precision_n: shape [y_dim, x_dim, x_dim]
    :param a_n: shape [y_dim]
    :param b_n: shape [y_dim]
    :return: shape [y_dim], the entropy of the posterior for each dimension
    """
    x_dim = precision_n.size(-1)

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

    # returns with shape [y_dim], one entropy for each y dimension
    return H_beta_given_sigma_2 + H_sigma_2

def batched_regression_params(data_x, data_y, mu_0, precision_0, a_0, b_0):
    """
    for each data in the batch dimension of data_x and data_y, do a regression to find the posterior parameters,
    given the same prior parameters for each batch
    :param data_x: [batch, n_data, x_dim]
    :param data_y: [batch, n_data, y_dim]
    :param mu_0: [y_dim, x_dim]
    :param precision_0: [y_dim, x_dim, x_dim]
    :param a_0: [y_dim]
    :param b_0: [y_dim]
    :return: precision_n [batch, y_dim, x_dim, x_dim], mu_n [batch, y_dim, x_dim], a_0 [batch, y_dim], b_0 [batch, y_dim]
    """

    x_dim = data_x.size(2)
    y_dim = data_y.size(2)
    n_data = data_x.size(1)
    batch = data_x.size(0)

    xTx = t.einsum('bij, bil -> bjl', data_x, data_x)
    yTx = t.einsum('bij, bil -> bjl', data_y, data_x)

    assert xTx.size() == t.Size([batch, x_dim, x_dim])
    assert yTx.size() == t.Size([batch, y_dim, x_dim])

    # [batch, y_dim, x_dim, x_dim]
    precision_n = xTx.unsqueeze(1) + precision_0.unsqueeze(0)
    prior_prec_times_mu = t.einsum('kij, kj -> ki', precision_0, mu_0)
    inv_prec_n = precision_n.inverse()
    mu_n = t.einsum('byij, byj -> byi', inv_prec_n, (yTx + prior_prec_times_mu.unsqueeze(0)))

    assert mu_n.size() == t.Size([batch, y_dim, x_dim])

    term_0 = t.sum(data_y ** 2, dim=1)
    term_1 = t.einsum('ki, kij, kj -> k', mu_0, precision_0, mu_0)
    term_2 = - t.einsum('bki, bkij, bkj -> bk', mu_n, precision_n, mu_n)

    # these are the posterior parameters for the sigma^2 side of the posterior
    a_n = a_0 + n_data / 2
    b_n = b_0.reshape(1, -1) + 0.5 * (term_0 + term_1.unsqueeze(0) + term_2)

    return mu_n, precision_n, a_n.unsqueeze(0).repeat(batch, 1), b_n


def get_MLE_prior_params(data_x, data_y, eps_norm=1e-3, init_gamma=None, init_b_0=None, init_a_0=None, n_iter=None, verbose=False):
    """
    This function does gradient descent on the parameters for the bayesian regression priors to find
    the model which maximises P(Data | model)

    We assume that mu_0 = 0 and that the prior has the form precision_0 = diag(exp(gamma_i))
    and we optimise the data likelihood with respect to gamma_i , a_0 and b_0

    Here we are manually differentiating log(p(y|m))


    :return mu_0, t.diag_embed(t.exp(gamma)), a_0, b_0
    """
    x_dim = data_x.size(1)
    y_dim = data_y.size(1)

    n_iter = 10000 if n_iter is None else n_iter
    lr_gamma = 1e-1
    lr_b_0 = 1e-2
    lr_a_0 = 1e-2

    # this is the thing we are optimising
    gamma = t.zeros(y_dim, x_dim) if init_gamma is None else init_gamma

    mu_0 = t.zeros(y_dim, x_dim)
    a_0 = 0.1 * t.ones(y_dim) if init_a_0 is None else init_a_0
    b_0 = t.ones(y_dim) if init_b_0 is None else init_b_0

    # useful term that remains constant through computation of all gradients
    y_at_x = data_y.T @ data_x
    x_at_x = data_x.T @ data_x
    y_2_sum = t.sum(data_y ** 2, dim=0)

    with (t.no_grad()):

        for i in range(n_iter):
            gamma_grad, a_0_grad, b_0_grad = log_model_evidence_grad(data_x, data_y, gamma, a_0, b_0,
                                                                     y_at_x=y_at_x, x_at_x=x_at_x, y_2_sum=y_2_sum,
                                                                     verbose=verbose)

            gamma += lr_gamma*gamma_grad
            a_0 += lr_a_0*a_0_grad
            b_0 += lr_b_0*b_0_grad

            if gamma_grad.norm().item() < eps_norm and b_0_grad.norm() < eps_norm and a_0_grad.norm() < eps_norm:
                break

            if verbose:
                print(f"{i}: gamma grad norm: {gamma_grad.norm().item()}, b_0 norm: {b_0_grad.norm().item()}, a_0 norm: {a_0_grad.norm().item()}")
            # print(gamma, a_0, b_0)
            if t.isnan(gamma_grad.norm()):
                break

    return mu_0, t.diag_embed(t.exp(gamma)), a_0, b_0


def bayesian_regression(data_x, data_y, fns, prior_mu, prior_precision, a_0, b_0):
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
    cov_n_sqrt = (1.0/t.sqrt(L)).unsqueeze(2) * Q.transpose(1,2)
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
        samples = mu_n.unsqueeze(0) + t.einsum('nk, nkj, kji -> nki', sigma_samples, unit_normal_samples, cov_n_sqrt)
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
        n_data = predict_data_x.size(0)
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

        # returns with shape [y_dim], one entropy for each y dimension
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


def infogain_on_posterior(regress_dict, new_x, n_entropy_samples=100, n_regress_samples=1000):
    """
    We compute the expected decrease in the entropy of the posterior of bayesian regression
    if we add a new sample new_x to the dataset
    :param regress_dict: regression dict from a bayesian regression
    :param new_x: shape [n, size_x_pre_expanded], the data we want to assume for infogain
    :return:
    """

    n_candidate_x = new_x.size(0)
    x_dim = new_x.size(1)
    y_dim = regress_dict['mu_n'].size(0)

    fn_list = regress_dict['fns']

    # this gives us the mean and variance for the predictions from the posterior
    new_x_preds_mean, new_x_preds_std = regress_dict['predict_fn'](new_x, n_samples=n_regress_samples)
    assert new_x_preds_mean.size() == t.Size([n_candidate_x, y_dim])
    assert new_x_preds_std.size() == t.Size([n_candidate_x, y_dim])

    # now we sample a bunch of outcomes for new_x, in a differentiable way
    # for each candidate x we sample n_entropy_samples
    simulated_y_at_new_x = (new_x_preds_std.unsqueeze(0) * t.randn(n_entropy_samples, n_candidate_x, y_dim) +
                            new_x_preds_mean.unsqueeze(0))

    assert simulated_y_at_new_x.size() == t.Size([n_entropy_samples, n_candidate_x, y_dim])

    # then using the posterior as the prior, do bayesian regression using only
    # new_x as data with the various predicted data,
    # and compute the difference in entropies

    mu_n, precision_n, a_n, b_n = (
        batched_regression_params(apply_and_concat(new_x, fn_list).unsqueeze(0).repeat(n_entropy_samples, 1, 1),
                                  simulated_y_at_new_x,
                                  regress_dict['mu_n'], regress_dict['precision_n'],
                                  regress_dict['a_n'], regress_dict['b_n']))

    entropies = posterior_entropy(precision_n, a_n, b_n)

    return (regress_dict['entropy_fn']().sum() - t.sum(entropies)/n_entropy_samples)


def tril_flatten(tril, offset=0):
    """
    used for getting a fourier basis
    """
    N = tril.size(-1)
    indices = t.tril_indices(N, N, offset=offset)
    indices = N * indices[0] + indices[1]
    return tril.flatten(-2)[..., indices]

def get_flat_quadratic(x):
    """
    Given an input with shape [n_data, x_dim], return all the elements needed for quadratic regression
    we can probably use torch.combinations to make this easier, and for including higher orders
    :param x:
    :return:
    """
    n_data = x.size(0)

    cross_elements = t.einsum('ni, nj -> nij', x, x)

    return t.cat([t.ones(n_data, 1), x, tril_flatten(cross_elements)], dim=1)

def get_flat_polynomials(x, degree):

    answer = []

    z = t.ones(x.size(0), 1)
    answer.append(z)
    for i in range(0, degree):
        z = t.einsum('bi, bj -> bij', z, x).flatten(-2)
        answer.append(z)

    return t.cat(answer, dim=1)

def get_flat_fourier_basis(x, max_k):
    """
    assume data will be normalised, so set the boundary to like [-pi, pi]

    for each pair of i,j with i =/= j, we need to get the linear combination

    k1 x_i + k2 x_j for each k1 = [0, 1, ... N_k] and k2=[0, 1, ..., N_l]

    we want to return something of shape [n_data, ]

    output has (x_dim*(x_dim-1)/2 * (max_k**2 * 2 - 1)) outputs

    this will include the constant term

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
    fourier_basis= t.cat([t.cos(linear_combinations).flatten(-2),
                           t.sin(linear_combinations[:, :, 1:]).flatten(-2)], dim=1)

    return fourier_basis

def get_flat_pairwise_radial_basis(x):
    raise NotImplemented

def apply_and_concat(x, fn_list):
    """
    takes a one dimensional tensor x and a list of functions, and concatenates them on dim=1

    :param x: tensor(n, dim_x)
    :param fn_list: list of functions
    :return: tensor(n, len(fn_list))
    """
    return t.cat([fn(x) for fn in fn_list], dim=1)


if __name__ == "__main__":
    data_noise = 0.1

    coefs_0 = t.randn(3)
    coefs_1 = t.randn(3)

    fns = [lambda x: t.ones(x.size()), lambda x: x, lambda x: x ** 2]
    size_x = len(fns)

    data_x = t.arange(-3, 3, 0.05).unsqueeze(1)
    data_y_0 = coefs_0[0] + coefs_0[1] * data_x + coefs_0[2] * data_x ** 2 + t.sin(2 * data_x)
    data_y_0 += t.randn(data_y_0.size()) * data_noise
    data_y_0 = data_y_0.reshape(-1, 1)

    data_y_1 = coefs_1[0] + coefs_1[1] * data_x + coefs_1[2] * data_x ** 2
    data_y_1 += t.randn(data_y_1.size()) * data_noise
    data_y_1 = data_y_1.reshape(-1, 1)

    data_y = t.cat([data_y_0, data_y_1], dim=1)

    x_dim = data_x.size(1)
    y_dim = data_y.size(1)

    mu_0, precision_0, a_0, b_0 = get_MLE_prior_params(apply_and_concat(data_x, fns), data_y, verbose=True)

    print(precision_0.size())

    sol_dict = bayesian_regression(data_x, data_y, fns,
                                   prior_mu=mu_0,
                                   prior_precision=precision_0,
                                   a_0=a_0,
                                   b_0=b_0)

    predict_y_mean, predict_y_std = sol_dict['predict_fn'](data_x, n_samples=1000)

    predict_y_mean_0 = predict_y_mean[:, 0]
    predict_y_std_0 = predict_y_std[:, 0]

    predict_y_mean_1 = predict_y_mean[:, 1]
    predict_y_std_1 = predict_y_std[:, 1]

    print(predict_y_mean.size())

    plt.scatter(data_x.squeeze().numpy(), data_y_0.numpy())
    plt.scatter(data_x.squeeze().numpy(), data_y_1.numpy())

    plt.plot(data_x.squeeze().numpy(), predict_y_mean_0.numpy(), color='red')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_0 + 1 * predict_y_std_0).numpy(), color='blue')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_0 - 1 * predict_y_std_0).numpy(), color='blue')

    plt.plot(data_x.squeeze().numpy(), predict_y_mean_1.numpy(), color='red')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_1 + 1 * predict_y_std_1).numpy(), color='blue')
    plt.plot(data_x.squeeze().numpy(), (predict_y_mean_1 - 1 * predict_y_std_1).numpy(), color='blue')

    plt.show()


if __name__ == "__main__0":
    # testing the gradient of the log_model_evidence

    data_x = t.randn(100, 5)
    data_y = (data_x).sum(dim=1).unsqueeze(1).repeat(1, 2)
    data_y += 0.1*t.randn(data_y.size())

    x_dim = data_x.size(1)
    y_dim = data_y.size(1)

    # this is the thing we are optimising
    gamma = t.zeros(y_dim, x_dim)+3
    gamma.requires_grad = True

    mu_0 = t.zeros(y_dim, x_dim)
    precision_0 = t.diag_embed(t.exp(gamma))
    a_0 = 0.1 * t.ones(y_dim)
    b_0 = t.ones(y_dim)

    a_0.requires_grad = True
    b_0.requires_grad = True

    # useful term that remains constant through computation of all gradients
    y_at_x = data_y.T @ data_x
    x_at_x = data_x.T @ data_x
    y_2_sum = t.sum(data_y ** 2, dim=0)

    mu_n, precision_n, a_n, b_n = posterior_params_linregress(data_x, data_y, mu_0, precision_0, a_0, b_0,
                                                              y_at_x=y_at_x, x_at_x=x_at_x, y_2_sum=y_2_sum)

    model_evidence = t.sum(log_model_evidence(precision_0, precision_n, a_0, a_n, b_0, b_n))

    model_evidence.backward()

    print(gamma.grad)

    # ==============================================

    gamma_grad, a_0_grad, b_0_grad = log_model_evidence_grad(data_x, data_y, gamma, a_0, b_0)

    print(gamma_grad)

# testing prediction
if __name__ == "__main__1":
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

    sol_dict = bayesian_regression(data_x, data_y, fns,
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