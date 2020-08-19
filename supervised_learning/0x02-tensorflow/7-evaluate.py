#!/usr/bin/env python3
""" Module that contains a function of the evaluate a model in tensorFlow"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Function that evaluates the output of a neural network
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Returns: the network’s prediction, accuracy, and loss, respectively
    """

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        new_saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        prediction, accuraccy, cost = sess.run([y_pred, accuracy, loss],
                                               feed_dict={x: X, y: Y})
    return prediction, accuraccy, cost
