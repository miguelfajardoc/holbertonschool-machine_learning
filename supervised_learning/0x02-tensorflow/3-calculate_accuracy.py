#!/usr/bin/env python3
""" Module that contains a function of the accuaracy in tensorFlow"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    function that calculates the accuracy of a prediction:
    - y is a placeholder for the labels of the input data
    - y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction

    Returns: the prediction of the network in tensor form

    """
    comparation = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    casting = tf.cast(comparation, tf.float32)
    return tf.math.reduce_mean(casting)
