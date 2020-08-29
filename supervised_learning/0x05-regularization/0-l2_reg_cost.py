#!/usr/bin/env python3
""" modulo to regularization """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    function that calculates the cost of a neural network with
    L2 regularization:
    - cost is the cost of the network without L2 regularization
    - lambtha is the regularization parameter
    - weights is a dictionary of the weights and biases (numpy.ndarrays) of the
      neural network
    - L is the number of layers in the neural network
    - m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    for layer in range(L):
        key = "W{}".format(layer + 1)
        cost += (np.linalg.norm(weights[key]) * lambtha / (2 * m))
    return cost
