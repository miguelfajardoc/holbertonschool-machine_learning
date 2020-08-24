#!/usr/bin/env python3
"""
Module of create RMSprop in tensorflow
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    function  that creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm:
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                            beta2=beta2, epsilon=epsilon)
    return adam_optimizer.minimize(loss)
