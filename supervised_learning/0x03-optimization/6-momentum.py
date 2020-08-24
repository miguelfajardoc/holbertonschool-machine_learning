#!/usr/bin/env python3
"""
Module of create momentum in tensorflow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    function that creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum optimization algorithm:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the momentum weight
     - Returns: the momentum optimization operation
    """
    momentum_optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return momentum_optimizer.minimize(loss)
