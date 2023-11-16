"""
This file defines differentiable pytorch functions which allow us to do
bayesian linear regression with automatic variable selection in a differentiable way.

todo: fix bug where bayesian regression is not commutative in the dataset
todo: test differentiability of everything
todo: test infogain routine that computes entropy differences
todo: write visualization functions for infogain with simple polynomials
todo: write optim routine to return the optimal state to explore for infogain
todo: write regression function for arbitrary numbers of hypotheses
todo : unit tests to make sure all works

todo: use t.linalg.solve(A,B) instead of A.inverse() @ B, it's faster and more stable

"""
import torch as t
import numpy as np

def bayesian_regression(data_x, data_y, fns, prior_mu, prior_precision, a_0=t.Tensor([0.1]), b_0=t.Tensor([1])):
    """
    :param data_x: Tensor(n_data, size_x)
    :param data_y: Tensor(n_data)
    :param fns: list of functions to apply row-wise to data_x to use as regression variables
    :param prior_mu: Tensor(size_x), the prior mean
    :param prior_precision: Tensor(size_x, size_x)
    :param a_0: Tensor(1) prior parameter for Inv-Normal distribution over the noise level sigma^2
    :param b_0, Tensor(1) prior parameter for Inv-Normal distribution over the noise level sigma^2
    :return: dict with the following fields: 'fns', 'mu_n', 'lambda_n', 'a_n', 'b_n', 'mle_var', 'sampling_fn', 'predict_fn', 'entropy', 'log_model_ev'

    we return the posterior mean and precision, as well as a function which allows for computing p(y|x,Data), the
    prediction for a point averaged conditioned on x and the Data, which averages over our uncertainty

    see https://en.wikipedia.org/wiki/Bayesian_linear_regression for equations

    This bayesian regression allows us to compute a posterior over P(beta, sigma^2 | X, y),
    which is the probability distribution of our linear parameters and the unknown noise level sigma^2
    """

    # apply the transformations in fns to the data
    data_x = apply_and_concat(data_x, fns)

    x_dim = data_x.size(1)
    n_data = data_x.size(0)

    # MLE for the parameters
    beta = t.linalg.pinv(data_x) @ data_y

    # quantity used in further computations
    xTx = data_x.T @ data_x

    # ==========================================================================
    # these are the posterior parameters for p(beta|sigma^2, X, y)
    # this distribution is N(mu_n, sigma^2 * (precision_n)^(-1)), notice that we still depend on the unknown sigma^2
    precision_n = xTx + prior_precision
    # inv_prec_n = t.inverse(precision_n)
    # mu_n = inv_prec_n @ (xTx @ beta + prior_precision @ prior_mu)
    # mu_n = inv_prec_n @ (data_x.T @ data_y + prior_precision @ prior_mu)
    # faster way:
    mu_n = t.linalg.solve(precision_n, (data_x.T @ data_y + prior_precision @ prior_mu))


    # ==============================
    # this is the posterior covariance, not yet scaled by sigma
    # we need to find A such that A dot A^T = cov_n
    # to do this we find the eigenvectors and values of cov_n, and take the sqrt
    # of the eigenvalues
    L, Q = t.linalg.eigh(precision_n)
    cov_n_sqrt = 1.0/t.sqrt(L) * Q
    # not that here we have cov_n = cov_n_sqrt @ cov_n_sqrt.T

    # instead, compute the eigenvalues of precision_n and just invert them

    # ==========================================================================
    # computing posterior p(sigma^2| X, y), which is an Inv-Normal distribution

    term_0 = t.sum(data_y**2)
    term_1 = prior_mu @ prior_precision @ prior_mu
    term_2 = - mu_n @ precision_n @ mu_n

    # these are the posterior parameters for the sigma^2 side of the posterior
    a_n = a_0 + n_data/2
    b_n = b_0 + 0.5 * (term_0 + term_1 + term_2)

    # ==========================================================================
    # here we define a function which samples from the posterior in a
    # differentiable way, using the reparametrisation trick, in order to predict
    # future samples.

    # the two distributions used for the reparam trick
    gamma_dist = t.distributions.gamma.Gamma(a_n, 1)
    multNormal = t.distributions.MultivariateNormal(t.zeros(x_dim), t.eye(x_dim))

    # ==========================================================================
    # computing P(data | model) = int P(data | beta, model) P(beta|model)
    # see

    log_model_evidence = x_dim/2 * t.log(t.Tensor([2*np.pi]))
    # this is computing log( sqrt(det(precision_0)/det(precision_n)) )
    # log_model_evidence += 0.5 * (t.logdet(prior_precision) - t.sum(t.log(L)))
    log_model_evidence += 0.5 * (t.logdet(prior_precision) - t.logdet(precision_n))
    log_model_evidence += a_0 * t.log(t.tensor([b_0]))
    log_model_evidence -= a_n * t.log(b_n)
    log_model_evidence += t.lgamma(a_n) - t.lgamma(t.Tensor([a_0]))


    # this function samples parameters from the posterior in a differentiable manner
    # with the reparametrization trick
    def sampling_fn(n_samples):

        # first sample from a Gamma, then invert it, then scale it
        gamma_samples = gamma_dist.sample(t.Size([n_samples]))

        # these are the sigma^2 samples from our posterior
        inv_gamma_samples = b_n * 1.0/gamma_samples
        sigma_samples = t.sqrt(inv_gamma_samples)

        # now we sample from a multivariate normal, and scale it by the posterior covariance sqrt
        # scale it by the sigma samples we have
        # and add the posterior mean
        unit_normal_samples = multNormal.sample(t.Size([n_samples]))

        # generate samples with the reparametrization trick, here mu_n is differentiable,
        # sigma_samples is differentiable through b_n, and cov_n_sqrt is differentiable too
        samples = mu_n.unsqueeze(0) + (( sigma_samples * unit_normal_samples) @ cov_n_sqrt.T)

        # samples have shape [n_samples, data_x.size(0)]
        return samples

    # samples the model to give predictions with error bars, both on in a differentiable
    # manner, given data at which you want to predict, and a given precision level
    # we give two uncertainties: the uncertainty of the mean, and the predicted aleatoric data variance
    # when our functions are linear, this is just the same as predicting with mu_n, but this is
    # different if we're using nonlinear functions in the inputs
    def predict_fn(predict_data_x, n_samples):
        """
        :param predict_data_x: tensor(n_data, size_x)
        :param n_samples: int
        :return: ave_pred : tensor(n_data), std_pred : tensor(n_data)
        """

        # generate n_samples from the posterior:
        betas = sampling_fn(n_samples)

        # predict using each sample at all points
        # this is now size(n_data, n_samples)
        preds = apply_and_concat(predict_data_x, fns) @ betas.T

        return preds.mean(dim=1), preds.std(dim=1)

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

def apply_and_concat(x, fn_list):
    """
    takes a one dimensional tensor x and a list of functions, and stacks them

    :param x: tensor(n, dim_x)
    :param fn_list: list of functions
    :return: tensor(n, len(fn_list))
    """
    return t.cat([fn(x) for fn in fn_list], dim=1)


def infogain_bayesian_regress(regress_dict, new_x, n_samples=100):
    """
    We compute the expected decrease in the entropy of the posterior of bayesian regression
    if we add a new sample new_x to the dataset

    :return:
    """

    # first do bayesian regression on data_x and data_y, and compute the posterior

    fn_list = regress_dict['fns']

    # this gives us the mean and variance for the predictions from the posterior
    new_x_preds_mean, new_x_preds_std = regress_dict['predict_fn'](new_x, n_samples=1000)

    # now we sample a bunch of outcomes for new_x, in a differentiable way
    simulated_y_at_new_x = new_x_preds_std * t.randn(n_samples) + new_x_preds_mean

    # then using the posterior as the prior, do bayesian regression using only
    # new_x as data with the various predicted data,
    # and compute the difference in entropies

    entropy_samples = [ bayesian_regression(new_x, simulated_y_at_new_x[i:i+1], fn_list,
                          prior_mu=regress_dict['mu_n'],
                          prior_precision=regress_dict['precision_n'],
                          a_0=regress_dict['a_n'],
                          b_0=regress_dict['b_n'])['entropy_fn']() for i in range(0, n_samples)]

    print(t.cat(entropy_samples))

    return sol_dict['entropy_fn']() - sum(entropy_samples)/len(entropy_samples)

def bayesian_regression_hypotheses(data_x, data_y, hypotheses):
    """
    we do bayesian regression over the parameter of each of the hypotheses
    when we predict, we average over the likelihood of all hypotheses
    when we compute infogain, we compute the information gain with all
    hypotheses too

    :param data_x: tensor(n_data, size_x)
    :param data_y: tensor(n_data)
    :param hypotheses: list of list of size_x -> 1 functions
    :return:
    """
    raise NotImplemented


if __name__ == "__main__2":

    data_noise = 0.5

    coefs = t.randn(3)

    fns = [lambda x: t.ones(x.size()), lambda x: x, lambda x:x**2]

    data_x = t.arange(0, 10, 0.1).unsqueeze(1)
    data_y = coefs[0] + coefs[1]*data_x + coefs[2]*data_x**2
    data_y += t.randn(data_y.size())*data_noise
    data_y = t.squeeze(data_y)

    data_infogain = 10*t.rand(1,1)

    print(data_x.size())
    size_x = len(fns)

    sol_dict = bayesian_regression(data_x, data_y, fns,
                                   prior_mu=t.zeros(size_x),
                                   prior_precision=1e-7*t.eye(size_x,size_x))

    infogain = infogain_bayesian_regress(sol_dict,data_infogain,n_samples=100)

    print(coefs)
    print(sol_dict)
    print(infogain)


if __name__ == "__main__":
    # consistency checking of bayesian computation:
    # updating in two steps over data should be equivalent to updating in one step

    data_noise = 0.5

    coefs = t.randn(3)

    fns = [lambda x: t.ones(x.size()), lambda x: x, lambda x: x ** 2]

    data_x = t.arange(0, 10, 0.1).unsqueeze(1)
    data_y = coefs[0] + coefs[1] * data_x + coefs[2] * data_x ** 2
    data_y += t.randn(data_y.size()) * data_noise
    data_y = t.squeeze(data_y)

    n_split = int(data_x.size(0)/2)
    size_x = len(fns)

    x0 = data_x[:n_split]
    y0 = data_y[:n_split]
    x1 = data_x[n_split:]
    y1 = data_y[n_split:]

    # updating on the whole thing at once
    sol_dict = bayesian_regression(data_x, data_y, fns, prior_mu=t.zeros(size_x), prior_precision=1e-2*t.eye(size_x,size_x))

    # updating in two steps
    sol_dict_0 = bayesian_regression(x0, y0, fns, prior_mu=t.zeros(size_x), prior_precision=1e-2*t.eye(size_x,size_x))
    sol_dict_1 = bayesian_regression(x1, y1, fns, prior_mu=sol_dict_0['mu_n'],
                                     prior_precision=sol_dict_0['precision_n'],
                                     a_0=sol_dict_0['a_n'],
                                     b_0=sol_dict_0['b_n'])

    sol_dict_2 = bayesian_regression(x1, y1, fns, prior_mu=t.zeros(size_x),
                                     prior_precision=1e-2 * t.eye(size_x, size_x))
    sol_dict_3 = bayesian_regression(x0, y0, fns, prior_mu=sol_dict_2['mu_n'],
                                     prior_precision=sol_dict_2['precision_n'],
                                     a_0=sol_dict_2['a_n'],
                                     b_0=sol_dict_2['b_n'])

    # the log model evidences disagree because they don't have the same data, that's fine,
    # it doesn't mean the code isn't working.
    print(sol_dict)
    print(sol_dict_1)
    # print(sol_dict_3)




