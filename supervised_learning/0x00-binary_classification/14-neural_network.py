#!/usr/bin/env python3
""" Module that defines a neural network
"""

import numpy as np


class NeuralNetwork:
    """ NeuralNetwork that defines a neural network with one hidden
        layer performing binary classification
    Public instance attributes
        W1: The weights vector for the hidden layer.
        b1: The bias for the hidden layer.
        A1: The activated output for the hidden layer.
        W2: The weights vector for the output neuron.
        b2: The bias for the output neuron.
        A2: The activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """ Constructor class
        - nx is the number of input features
        - nodes is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

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
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValeError("alpha must be positive")

        while iterations:
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            iterations -= 1
        return self.evaluate(X, Y)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        - A1 is the output of the hidden layer
        - A2 is the predicted output
        - alpha is the learning rate
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / Y.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=1) / m

        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

    def evaluate(self, X, Y):
        """
        Evaluates the  neural network’s predictions
           X is a numpy.ndarray with shape (nx, m) that contains the input data
              nx is the number of input features to the neuron
              m is the number of examples
           Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        """
        forwardProp = self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        evaluation = np.where(self.__A2 >= 0.5, 1, 0)
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
        Z1 = np.matmul(self.__W1, X)
        Z1 = Z1 + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1)
        Z2 = Z2 + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    @property
    def W1(self):
        return self.__W1

    @property
    def W2(self):
        return self.__W2

    @property
    def b1(self):
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2
