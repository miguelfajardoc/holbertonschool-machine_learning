#!/usr/bin/env python3
"""keras module"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    function  that sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics:
    - network is the model to optimize
    - alpha is the learning rate
    - beta1 is the first Adam optimization parameter
    - beta2 is the second Adam optimization parameter
    Returns: None

    """
    adam = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    return model.compile(optimizer=adam,
                         loss=K.losses.SparseCategoricalCrossentropy
                         (from_logits=True),
                         metrics=["accuracy"])
