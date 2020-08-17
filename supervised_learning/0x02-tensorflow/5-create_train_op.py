#!/usr/bin/env python3
""" Module that contains a function of the loss in tensorFlow"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    function that that creates the training operation for the network:
    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent

    """
    gradient = tf.train.GradientDescentOptimizer(alpha)
    return gradient.minimize(loss)
