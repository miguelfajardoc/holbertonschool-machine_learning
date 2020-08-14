#!/usr/bin/env python3
""" Module that defines a deep neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network
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

        i = 0
        if graph is True:
            iterationGraph = np.empty([int(iterations / step) + 1], int)
            costGraph = np.empty([int(iterations / step) + 1], float)
            graphIndex = 0
        while i <= iterations:
            A, _ = self.forward_prop(X)
            if verbose is True:
                if i == 0 or i % step == 0 or i == iterations:
                    cost = self.cost(Y, A)
                    print("Cost after {} iterations: {}".format(i, cost))
                    if graph is True:
                        iterationGraph[graphIndex] = i
                        costGraph[graphIndex] = cost
                        graphIndex += 1
            self.gradient_descent(Y, self.__cache, alpha)
            i += 1
        if graph:
            plt.plot(iterationGraph, costGraph)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        - cache is a dictionary containing all the intermediary values of
          the network
        - alpha is the learning rate
        """
        m = Y.shape[1]
        layer = self.__L
        while layer > 0:
            Akey = 'A{}'.format(layer)
            Wkey = 'W{}'.format(layer)
            bkey = 'b{}'.format(layer)
            A = self.__cache[Akey]
            W = self.__weights[Wkey]
            b = self.__weights[bkey]
            if layer == self.__L:
                da = (- Y / A) + ((1 - Y) / (1 - A))
            dz = da * (A * (1 - A))
            dw = np.matmul(dz,  self.__cache['A{}'.format(layer - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            da = np.matmul(W.T, dz)
            self.__weights[Wkey] = W - (alpha * dw)
            self.__weights[bkey] = b - (alpha * db)
            layer -= 1

    def evaluate(self, X, Y):
        """
        Evaluates the  neural network’s predictions
           X is a numpy.ndarray with shape (nx, m) that contains the input data
              nx is the number of input features to the neuron
              m is the number of examples
           Y is a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        """
        forwardProp, _ = self.forward_prop(X)
        maximo = np.amax(A, axis=0, keepdims=True)
        key = 'A{}'.format(self.__L)
        return np.where(self.__cache[key] == maximo, 1, 0), self.cost(Y, A)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y is a numpy.ndarray with shape (1, m) that contains the correct
              labels for the input data
            A is a numpy.ndarray with shape (1, m)
              containing the activated output of the neuron for each example
        """
        Cost = - 1 * np.sum(Y * np.log(A)) / Y.shape[1]
        return Cost

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
                if layer == self.__L:
                    Al = np.exp(tempZ) / np.sum(np.exp(tempZ), axis=0,
                                                keepdims=True)
                else:
                    Al = 1 / (1 + np.exp(-tempZ))
                self.__cache[AlKey] = Al
        return Al, self.__cache

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        """
        if ".pkl" not in filename:
            filename = filename + ".pkl"
        with open(filename, 'wb') as fileObject:
            pickle.dump(self, fileObject)

    @staticmethod
    def load(filename):
        """
        Load a pickle deepNeuralNetwork object
        """
        try:
            fileObject = open(filename, "rb")
            object = pickle.load(fileObject)
            return object
        except FileNotFoundError:
            return None

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
