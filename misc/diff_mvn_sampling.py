"""
The goal here is to figure out how to sample from a MVN distribution in a differentiable way
"""
import torch as t
import numpy as np

if __name__ == "__main__":

    n = 10

    prec = t.randn(n,n)
    prec = prec + prec.T + 10*t.eye(n)

    prec.requires_grad = True

    L, Q = t.linalg.eigh(prec)
    cov_n_sqrt = (1.0 / t.sqrt(L)).unsqueeze(1) * Q.T

    prec_sqrt = (t.sqrt(L)).unsqueeze(1) * Q.T
    cov = prec.inverse()

    # these two are 0
    print(prec_sqrt.T @ prec_sqrt - prec)
    print(cov_n_sqrt.T @ cov_n_sqrt - prec.inverse())

    multNormal = t.distributions.MultivariateNormal(t.zeros(n), t.eye(n))

    unit_normal_samples = multNormal.sample(t.Size([100000]))

    samples = t.einsum('nj, ji -> ni', unit_normal_samples, 2*cov_n_sqrt)

    empirical_cov = t.cov(samples.T)

    print(empirical_cov - 4*cov)

    # =====================
    # testing differentiability

    objective = samples.std()
    objective.backward()

    print(prec.grad)
