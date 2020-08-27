#!/usr/bin/env python3
"""
Module of create RMSprop in tensorflow
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    function  that creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm:
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    decay_optimizer = tf.train.inverse_time_decay(alpha, global_step,
                                                  decay_steps, decay_rate,
                                                  staircase=True)
    return decay_optimizer
