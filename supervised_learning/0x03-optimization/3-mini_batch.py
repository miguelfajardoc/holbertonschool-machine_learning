#!/usr/bin/env python3
"""
Module of training with minibatch
"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    function that trains a loaded neural network model using mini-batch G.D.:
    X_train is a numpy.ndarray of shape (m, 784) containing the training data
        m is the number of data points
        784 is the number of input features
    Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the training
    labels:
        10 is the number of classes the model should classify
    X_valid is a numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the
    validation labels
    epochs is the number of times the training should pass through the whole
    dataset
    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + ".meta")
        new_saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]
        feed = {x: X_train, y: Y_train}
        feedValidation = {x: X_valid, y: Y_valid}

        for epoch in range(epochs):
            cost, Accuracy = session.run([loss, accuracy], feed_dict=feed)
            validationCost, validationAccuracy = session.run([loss, accuracy],
                                                             feed_dict=feed)
            print_epoch_status(epoch, cost, accuracy, validationCost,
                               validationAccuracy)
            X_train, Y_train = shuffle_data(X_train, Y_train)
            gradient_steps = 0
            for batch in get_batch(X_train, batch_size, gradient_steps):
                session.run(train_op, feed_dict=feed)
                if gradient_steps % 100 == 0:
                    cost, Accuracy = session.run([loss, accuracy],
                                                 feed_dict=feed)
                    print_batch_status(cost, Accuracy, gradient_steps)
                gradient_steps += 1
        Save_path = saver.save(session, save_path)
        session.close()
    return Save_path


def get_batch(X_train, batch_size, gradient_steps):
    """ function that returns the minibatch"""
    return X_train[gradient_steps, batch_size]


def print_batch_status(cost, Accuracy, gradient_steps):
    """ function that print the batch status"""
    print("\tStep {}:", format(gradient_steps))
    print("t\tCost: {}", format(cost))
    print("\tAccuracy {}:", format(Accuracy))


def print_epoch_status(epoch_number, cost, accuracy, validationCost,
                       validationAccuracy):
    """ function that print the epoch status """
    print("After {} epochs:".format(epoch))
    print("\tTraining Cost: {}".format(cost))
    print("\tTraining Accuracy: {}".format(Accuracy))
    print("\tValidation Cost: {}".format(validationCost))
    print("\tValidation Accuracy: {}".format(validationAccuracy))
