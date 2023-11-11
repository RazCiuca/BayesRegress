"""
In this file we do approximate bayesian logistic regression in a
fully differentiable way

multi-class regression can be done with multiple logistic regressions on
the individual classes

steps in bayesian logistic regression:

- set up the loss function
- get the minimum
- compute the hessian at the minimum
- treat it as approximately gaussian

"""


