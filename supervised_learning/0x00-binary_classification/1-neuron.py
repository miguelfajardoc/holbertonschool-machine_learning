#!/usr/bin/env python3
"""Neuron fileeee"""

import numpy as np


class Neuron:
    """ Neuron that perfom a binary classification
        Args:
             - nx: number of input features to the neuron
             - W: weights vector for the neuron.
             - b: The bias for the neuron
             - A: The activated outpot of the neuron (prediction)
    """

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
