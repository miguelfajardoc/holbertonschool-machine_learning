#!/usr/bin/env python3
""" Module that contains a function layer tensorFlow"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    function that creates the forward propagation graph for the neural network
    - layer_sizes is a list containing the number of nodes in each layer of the
    network
    - activations is a list containing the activation functions for each layer
    of the network
    Returns: the prediction of the network in tensor form

    """
    output = x
    for iter in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[iter], activations[iter])
    return output
