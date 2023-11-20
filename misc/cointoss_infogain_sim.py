"""
small script simulating bayesian inference for a coin toss with unknown probability
"""


import torch as t
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt



if __name__ == "__main__":

    true_theta = np.random.rand(1)

    # these are the prior Beta(alpha, beta) parameters
    a = 1
    b = 1

    x_data = list(range(0, 1000))
    entropies = []
    infogains = []

    # simulate draws
    for i in x_data:

        draw = int(np.random.rand(1) < true_theta)

        a += draw
        b += 1-draw

        mean_theta = a/(a + b)

        entropy_of_posterior = beta.entropy(a, b)
        entropies.append(entropy_of_posterior)

        infogain = entropy_of_posterior - (mean_theta * beta.entropy(a+1, b) + (1-mean_theta)*beta.entropy(a, b+1))
        infogains.append(infogain)

        print(draw, infogain)

    plt.plot(x_data, np.log(infogains))
    plt.show()

