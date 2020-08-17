#!/usr/bin/env python3
""" Module that contains a function placeholder tensorFlow"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function that create a placeolders to hold number of feature columns of
    the data and the number of labels

    x is the placeholder for the input data to the neural network
    y is the placeholder for the one-hot labels for the input data

    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")

    return x, y
