
"""
The goal of this script is to show a full example of regression with a lot of basis functions, and
show that we don't overfit. We also plot the infogain

"""


import numpy as np
import torch as t
import matplotlib.pyplot as plt
from linregress_multiple_y import *

if __name__ == "__main__":

    data_noise = 0.4

    coefs_0 = t.randn(3)
    coefs_1 = t.randn(3)

    # fns = [lambda x: t.ones(x.size()), lambda x: x, lambda x: x ** 2]
    # fns = ([lambda x: t.ones(x.size()), lambda x: x, lambda x: x**2, lambda x: 0.5* x**2] + [(lambda x: t.sin(k*x)) for k in [1,2,3,4]] +
    #        [(lambda x: t.cos(k*x)) for k in [1,2,3,4]])
    fns = [lambda x: get_flat_polynomials(x, 4)]
    size_x = len(fns)

    visualize_x = t.arange(-3, 3, 0.01).unsqueeze(1)

    data_x = t.cat([t.arange(-3, -1, 0.2).unsqueeze(1), t.arange(1, 3, 0.2).unsqueeze(1)], dim=0)
    data_y_0 = coefs_0[0] + coefs_0[1] * data_x + coefs_0[2] * data_x ** 2 # + t.sin(2*data_x)
    data_y_0 += t.randn(data_y_0.size()) * data_noise
    data_y_0 = data_y_0.reshape(-1, 1)

    data_y_1 = coefs_1[0] + coefs_1[1] * data_x + coefs_1[2] * data_x ** 2
    data_y_1 += t.randn(data_y_1.size()) * data_noise
    data_y_1 = data_y_1.reshape(-1, 1)

    data_y = t.cat([data_y_0, data_y_1], dim=1)

    x_dim = data_x.size(1)
    y_dim = data_y.size(1)

    mu_0, precision_0, a_0, b_0 = get_MLE_prior_params(apply_and_concat(data_x, fns), data_y, verbose=True)


    sol_dict = bayes_regress_multiple_y(data_x, data_y, fns,
                                        prior_mu=mu_0,
                                        prior_precision=precision_0,
                                        a_0=a_0,
                                        b_0=b_0)


    # predict_y_mean, predict_y_std = sol_dict['predict_fn'](visualize_x, n_samples=1000)
    #
    #
    # predict_y_mean_0 = predict_y_mean[:, 0]
    # predict_y_std_0 = predict_y_std[:, 0]
    #
    # predict_y_mean_1 = predict_y_mean[:, 1]
    # predict_y_std_1 = predict_y_std[:, 1]
    #
    # print(predict_y_mean.size())
    #
    # plt.scatter(data_x.squeeze().numpy(), data_y_0.numpy())
    # plt.scatter(data_x.squeeze().numpy(), data_y_1.numpy())
    #
    # plt.plot(visualize_x.squeeze().numpy(), predict_y_mean_0.numpy(), color='red')
    # plt.plot(visualize_x.squeeze().numpy(), (predict_y_mean_0 + 1 * predict_y_std_0).numpy(), color='blue')
    # plt.plot(visualize_x.squeeze().numpy(), (predict_y_mean_0 - 1 * predict_y_std_0).numpy(), color='blue')
    #
    # plt.plot(visualize_x.squeeze().numpy(), predict_y_mean_1.numpy(), color='red')
    # plt.plot(visualize_x.squeeze().numpy(), (predict_y_mean_1 + 1 * predict_y_std_1).numpy(), color='blue')
    # plt.plot(visualize_x.squeeze().numpy(), (predict_y_mean_1 - 1 * predict_y_std_1).numpy(), color='blue')
    #
    # plt.show()