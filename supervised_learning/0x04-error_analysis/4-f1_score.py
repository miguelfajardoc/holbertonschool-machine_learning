#!/usr/bin/env python3
""" Module to calculate a F1 from a confusion matrix"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    function that calculates the F1 score of a confusion matrix:
    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent
    the predicted labels
        classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the F1 score of
    each class
    You may use sensitivity = __import__('1-sensitivity').sensitivity and
    precision = __import__('2-precision').precision
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)
    F1 = 2 * (prec * recall) / (prec + recall)
    return F1
