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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev_inputs = nx
        for l in range(len(layers)):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")
            key = 'W{}'.format(l + 1)
            bias = 'b{}'.format(l + 1)
            self.weights[key] = np.random.randn(
                layers[l], prev_inputs) * np.sqrt(2 / prev_inputs)
            self.weights[bias] = np.zeros((layers[l], 1))
            prev_inputs = layers[l]
