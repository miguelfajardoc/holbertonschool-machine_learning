#!/usr/bin/env python3
""" Module that contains a function of the loss in tensorFlow"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    function that builds, trains, and saves a neural network classifier:
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of nodes in each layer of
    the network
    activations is a list containing the activation functions for each layer
    of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    Returns: an operation that trains the network using gradient descent

    """
    # placeholders
    print("shapes x{}, y{}".format(X_train.shape, Y_train.shape))
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    print(x, y)
    # tensors
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    # Operations
    train_op = create_train_op(loss, alpha)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        index = 0
        while index <= iterations:
            cost, Accuracy = session.run([loss, accuracy],
                                         feed_dict={x: X_train, y: Y_train})
            costV, AccuracyV = session.run([loss, accuracy],
                                           feed_dict={x: X_valid, y: Y_valid})
            if index == 0 or index % 100 == 0:
                print("After {} iterations:".format(index))
                print("\tTraining Cost: {}".format(cost))
                print("\tTraining Accuracy: {}".format(Accuracy))
                print("\tValidation Cost: {}".format(costV))
                print("\tValidation Accuracy: {}".format(AccuracyV))
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
            index += 1
        Save_path = saver.save(session, save_path)
        session.close()
    return Save_path
