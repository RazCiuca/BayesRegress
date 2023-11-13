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


