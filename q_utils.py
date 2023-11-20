import numpy as np
import torch as t

from linregress_utils import *


def fit_dynamics_model(states, actions, next_states, rewards):
    """
    fits a bayesian polynomial regression model to predict state differences and rewards from actions and current states.

    IMPORTANT: assumes states and actions are normalised to zero mean and unit variance

    :param states : shape [n_data, state_dim]
    :param actions : shape [n_data, action_dim]
    :param next_states : shape [n_data, state_dim]
    :param rewards : shape [n_data]
    """
    raise NotImplemented

def q_func_from_model(states, actions, model_param):
    """
    -> takes a batch of dynamics model for predicting next_states and rewards, and fits a q-value function to each one differentiably

    -> we will take gradient steps nondifferentiably on the q-function parameters until we get pretty close to the minimum
    -> then we take a few more gradient steps differentiably

    -> to fit the q-function, we sample a bunch of imagined trajectories from current data, then minimize the td-error on
        this imagined dataset

    """

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
