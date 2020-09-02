#!/usr/bin/env python3
"""keras module"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    function that trains a model using mini-batch gradient descent:
    - network is the model to train
    - data is a numpy.ndarray of shape (m, nx) containing the input data
    - labels is a one-hot numpy.ndarray of shape (m, classes) containing
    the labels
    of data
    - batch_size is the size of the batch used for mini-batch gradient
    descent
    - epochs is the number of passes through data for mini-batch gradient
    descent
    - validation_data is the data to validate the model with, if not None
    - verbose is a boolean that determines if output should be printed during
    training
    - shuffle is a boolean that determines whether to shuffle the batches every
    epoch.
    Returns: the History object generated after training the model
    """
    history = network.fit(data, labels, batch_size, epochs, verbose,
                          validation_data=validation_data, shuffle=shuffle)
    return history
