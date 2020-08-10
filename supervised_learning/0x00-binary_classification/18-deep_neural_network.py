#!/usr/bin/env python3
""" Module that defines a deep neural network
"""

import numpy as np


class DeepNeuralNetwork:
    """ DeepNeuralNetwork that defines a deep neural network with one hidden
        layer performing binary classification
    Public instance attributes
        W1: The weights vector for the hidden layer.
        b1: The bias for the hidden layer.
        A1: The activated output for the hidden layer.
        W2: The weights vector for the output neuron.
        b2: The bias for the output neuron.
        A2: The activated output for the output neuron (prediction).
    """

    def __init__(self, nx, layers):
        """ Constructor class
        - nx is the number of input features
        - layers is a list representing the number of nodes in each layer
          of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev_inputs = nx
        for l in range(len(layers)):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")
            key = 'W{}'.format(l + 1)
            bias = 'b{}'.format(l + 1)
            self.__weights[key] = np.random.randn(
                layers[l], prev_inputs) * np.sqrt(2 / prev_inputs)
            self.__weights[bias] = np.zeros((layers[l], 1))
            prev_inputs = layers[l]

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network
            X is a numpy.ndarray with shape (nx, m) that contains the input
            data, nx is the number of input features to the neuron
            m is the number of examples
        """
        for layer in range(self.__L + 1):
            if layer == 0:
                self.__cache['A0'] = X
            else:
                Wl = self.__weights['W{}'.format(layer)]
                Alprev = self.__cache['A{}'.format(layer - 1)]
                bl = self.__weights['b{}'.format(layer)]
                tempZ = np.matmul(Wl, Alprev) + bl
                AlKey = 'A{}'.format(layer)
                Al = 1 / (1 + np.exp(-tempZ))
                self.__cache[AlKey] = Al
        return Al, self.__cache

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
