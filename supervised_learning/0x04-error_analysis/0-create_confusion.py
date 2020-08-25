#!/usr/bin/env python3
""" Module to create a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    function that creates a confusion matrix:
    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
    correct labels for each data point
        m is the number of data points
        classes is the number of classes
    logits is a one-hot numpy.ndarray of shape (m, classes) containing the
    predicted labels
    Returns: a confusion numpy.ndarray of shape (classes, classes) with row
    indices representing the correct labels and column indices representing
    the predicted labels
     """
    confusion_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    for data_point_iterator in range(labels.shape[0]):
        correct_index = labels[data_point_iterator].argmax()
        predicted_index = logits[data_point_iterator].argmax()
        confusion_matrix[correct_index][predicted_index] += 1

    return confusion_matrix
