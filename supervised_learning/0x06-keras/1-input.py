#!/usr/bin/env python3
"""keras module"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library:
    - nx is the number of input features to the network
    - layers is a list containing the number of nodes in each layer of the
    network
    - activations is a list containing the activation functions used for each
    layer of the network
    - lambtha is the L2 regularization parameter
    - keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Input class
    Returns: the keras model
    """
    # model = K.models.Sequential()
    leng = len(layers)
    inputs = K.Input(shape=(nx,))
    dense = K.layers.Dense(layers[0],
                           activation=activations[0],
                           kernel_regularizer=K.regularizers.l2(lambtha)
                           )
    x = dense(inputs)
    for index in range(1, leng - 1):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[index],
                           activation=activations[index],
                           kernel_regularizer=K.regularizers.l2(lambtha)
                           )(x)
    x = K.layers.Dropout(1 - keep_prob)(x)
    outputs = K.layers.Dense(layers[leng - 1],
                             activation=activations[leng - 1],
                             kernel_regularizer=K.regularizers.l2(lambtha)
                             )(x)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model
