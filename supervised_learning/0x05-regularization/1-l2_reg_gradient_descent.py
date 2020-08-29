#!/usr/bin/env python3
""" modulo to regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization:
    - Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    - weights is a dictionary of the weights and biases of the neural network
    - cache is a dictionary of the outputs of each layer of the neural network
    - alpha is the learning rate
    - lambtha is the L2 regularization parameter
    - L is the number of layers of the network
    The weights and biases of the network should be updated in place
    """
    weights_copy = weights.copy()
    m = Y.shape[1]
    for layer in range(L, 0, -1):
        keyW = "W{}".format(layer)
        keyB = "b{}".format(layer)
        keyA = "A{}".format(layer)
        keyAprev = "A{}".format(layer - 1)

        if layer == L:
            dz = cache[keyA] - Y
        else:
            keyWnext = "W{}".format(layer + 1)
            derivate = 1 - cache[keyA] ** 2
            dz = np.matmul(weights[keyWnext].T, dz) * derivate
        dw = np.matmul(dz, cache[keyAprev].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dw_l2 = dw + (lambtha * weights[keyW] / m)

        weights[keyW] = weights_copy[keyW] - alpha * dw_l2
        weights[keyB] = weights_copy[keyB] - alpha * db
