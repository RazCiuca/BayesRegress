import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    This class defines an MLP with arbitrary layers and nodes
    """
    def __init__(self, nodes):
        """
        :param nodes: list containing integers corresponding to our numbers of nodes in each layer
        """
        super(MLP, self).__init__()

        self.s = 1

        self.nodes = nodes
        self.h_nodes = nodes[1:-1]  # all the hidden nodes
        # initialize the weights for all layers, with std = 1/nodes[i]**0.5
        self.weights = nn.ParameterList([nn.Parameter(self.s * t.randn(nodes[i], nodes[i+1])/(nodes[i]**0.5), requires_grad=True)
                        for i in range(0, len(nodes)-1)])
        # initialize the biases to 0
        self.biases = nn.ParameterList([nn.Parameter(t.zeros(nodes[i+1]), requires_grad=True)
                       for i in range(0, len(nodes)-1)])
        # list containing our transition functions, all ReLU instead of the last one, which is just the identity
        self.sigmas = [t.relu for _ in range(0, len(self.weights)-1)] + [lambda x: x]


    def forward(self, inputs):
        """
        :param inputs: assumed to be of size [batch_size, self.nodes[0]]
        :return: returns the output tensor, of size [batch_size, self.nodes[-1]]
        """

        x = inputs

        for w, b, sigma in zip(self.weights, self.biases, self.sigmas):
            x = sigma(x @ w + b)

        return x

    def set_output_std(self, inputs, out_std):
        """
        changes the weights of the network uniformly so that the output standard deviation is set to the desired level
        :param inputs: tensor of inputs with a large batch size
        :param out_std: wanted output standard deviation
        :return: None
        """
        current_std = (self.forward(inputs)).std().item()

        # if the output variance is just zero, scale it up until we get the correct answer
        while current_std < out_std:
            scaling_factor = 100**(1.0/len(self.weights))
            for w, b in zip(self.weights, self.biases):
                w.data = w.data * scaling_factor
                b.data = b.data * scaling_factor

            current_std = (self.forward(inputs)).std().item()
            print(f"current std now: {current_std:.5e}")

        scaling_factor = (out_std/current_std)**(1.0/len(self.weights))

        for w, b in zip(self.weights, self.biases):
            w.data = w.data * scaling_factor
            b.data = b.data * scaling_factor
