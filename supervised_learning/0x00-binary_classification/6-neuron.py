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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        - iterations is the number of iterations to train over
        - alpha is the learning rate
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValeError("alpha must be positive")

        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.__W, self.__b = self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing the activated
           output of the neuron for each example
        - alpha is the learning rate
        """
        dz = A - Y
        dw = np.matmul(X, dz.T) / Y.shape[1]
        db = dz
        db = np.sum(dz) / Y.shape[1]

        self.__W = self.__W - (alpha * (dw.T))
        self.__b = self.__b - (alpha * db.T)
        return self.__W, self.__b

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
           X is a numpy.ndarray with shape (nx, m) that contains the input data
              nx is the number of input features to the neuron
              m is the number of examples
           Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        """
        forwardProp = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        evaluation = np.where(self.__A >= 0.5, 1, 0)
        return evaluation, cost

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y is a numpy.ndarray with shape (1, m) that contains the correct
              labels for the input data
            A is a numpy.ndarray with shape (1, m)
              containing the activated output of the neuron for each example
        """
        Cost = -(np.matmul(Y, np.log(A.T)) +
                 (np.matmul(1 - Y, np.log(1.0000001 - A.T))))
        Cost = Cost.item(0) / Y.shape[1]
        return Cost

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron
            X is a numpy.ndarray with shape (nx, m) that contains the input
            data nx is the number of input features to the neuron
            m is the number of examples
        """
        Z = np.matmul(self.__W, X)
        Z = Z + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
