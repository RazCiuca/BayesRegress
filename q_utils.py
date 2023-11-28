import numpy as np
import torch as t

from linregress_utils import *


def fit_dynamics_model(states, actions, next_states, rewards, regress_fns=None, tuple_priors=None):
    """
    fits a bayesian polynomial regression model to predict state differences and rewards from actions and current states.

    IMPORTANT: assumes states and actions are normalised to zero mean and unit variance

    optimise the prior when you do this function

    :param states : shape [n_data, state_dim]
    :param actions : shape [n_data, action_dim]
    :param next_states : shape [n_data, state_dim]
    :param rewards : shape [n_data]
    :regress_fns : optional list of regression functions for the dynamics. Defaults to quadratic regression
    :return sol_dict, (mu_0, precision_0, a_0, b_0)
    """

    regress_fns = [lambda x: get_flat_polynomials(x, 2)] if regress_fns is None else regress_fns

    # we regress from states, actions -> next_states, rewards
    data_x = t.cat([states, actions], dim=1)
    data_y = t.cat([next_states, rewards.unsqueeze(1)], dim=1)

    if tuple_priors is not None:
        # todo_rl: if we have some prior knowledge about optimal priors, start with those and do less iterations
        pass

    mu_0, precision_0, a_0, b_0 = get_MLE_prior_params(apply_and_concat(data_x, regress_fns), data_y, verbose=False)

    sol_dict = bayesian_regression(data_x, data_y, regress_fns,
                                   prior_mu=mu_0,
                                   prior_precision=precision_0,
                                   a_0=a_0,
                                   b_0=b_0)

    return sol_dict, (mu_0, precision_0, a_0, b_0)

def sample_from_dynamics(sol_dict, states, actions, n_models=30, n_sample_per_model=10):
    """
    for each state-action pair given, and for each sampled model, we sample next-state-reward pairs

    :param sol_dict: dictionary of the bayesian regression solution
    :param states: states at which the dynamics need to be predicted
    :param actions: actions taken at those states
    :param n_models: number of different models to sample
    :return: next_state_samples [n_states, n_models, n_sample_per_model, state_dim] ,
             reward_samples [n_states, n_models, n_sample_per_model]
    """

    # shapes [n_models, dim_state + 1, dim_state + dim_action], [n_models, dim_state + 1]
    model_samples, sigma_samples = sol_dict['sampling_fn'](n_models)

    # shape [n_states, dim_state + dim_action]
    predict_data_x = t.cat([states, actions], dim=1)

    # shape [n_states, n_models, dim_state + 1]
    mean_preds = t.einsum('ni, mki -> nmk', apply_and_concat(predict_data_x, fns), model_samples)

    # shape [n_states, n_models, n_sample_per_model, dim_state + 1]
    noise = t.randn(mean_preds.size(0), mean_preds.size(1), n_sample_per_model, mean_preds.size(2))
    noise *= sigma_samples.reshape(1, noise.size(0), 1, noise.size(-1))

    samples = mean_preds + noise
    next_state_samples = samples[:, :, :, :-1]
    reward_samples = samples[:, :, :, -1]

    return next_state_samples, reward_samples


def q_func_from_dynamics_samples(states, actions, next_state_samples, reward_samples, regress_fns=None):
    """
    We approximate the q-function at a state-action pair as:
    Q(s,a) = (a - beta^T f(s))^T diag(-e^(lambda_i)) (a - beta^T f(s))  + alpha^T f(s)

    This is simply a quadratic form with a state-dependent maximum location and height. from this maximum the
    q-values drop off quadratically in each action dimension with eigenvalues -exp(lambda_i) .

    We can optimise this q-function comparatively very very easily

    :param states: [n_states, dim_state] states at which we learn the q-function
    :param actions: [n_states, dim_action], actions taken at those states
    :param next_state_samples: [n_states, n_models, n_sample_per_model, state_dim] , predictions at those states for each dynamics model
    :param reward_samples: [n_states, n_models, n_sample_per_model], predictions at those states again
    :param regress_fns: optional list of regression functions to predict the q-value mean and maximum
    :return: beta, alpha, gamma
    """

    # we use get_flat_quadratic and not the other function here in order to avoid having duplicated variables in the regression
    regress_fns = [get_flat_quadratic] if regress_fns is None else regress_fns

    # need to compute E[f_i(s')] , the expectation of each of our regression functions

    raise NotImplemented

def expected_q_value(states, actions, model):
    """
    differentiably computes the expected q-value under the dynamics model distribution for the given state-action pairs
    :param states:
    :param actions:
    :param model:
    :return:
    """
    raise NotImplemented

def infogain_model(current_action, radius):
    """


    :param current_action:
    :param radius:
    :return:
    """
    raise NotImplemented

def solve_infogain_mdp():
    """
    sample a bunch of imagined trajectories from current states, then for each compute the infogain for their
    actions, then fit a q-function to this infogain.

    The way we choose actions is to start with the optimal q-value, then take gradient steps with respect to
    the infogain-q-function until the optimal-q-value gets below some fraction of the original. so we don't sacrifice
    more than some fraction of the reward for the sake of exploration.

    :return:
    """

    raise NotImplemented



