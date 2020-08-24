#!/usr/bin/env python3
"""
Module of create RMSprop in tensorflow
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    function  that creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm:
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    RMSprop_optimizer = tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                                  epsilon=epsilon)
    return RMSprop_optimizer.minimize(loss)
