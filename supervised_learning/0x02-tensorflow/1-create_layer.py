#!/usr/bin/env python3
""" Module that contains a function layer tensorFlow"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG") to
    implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer

    """
    layer = tf.layers.Dense(units=n, activation=activation, name="layer",
                            kernel_initializer=tf.contrib.layers.
                            variance_scaling_initializer(mode="FAN_AVG"))
    return layer(prev)
