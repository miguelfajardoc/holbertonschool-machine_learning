#!/usr/bin/env python3
""" Module to calculate a precision from a confusion matrix"""
import numpy as np


def precision(confusion):
    """
    function that calculates the precision for each class in a confusion
    matrix:
    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels
        classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the precision of
    each class
    """
    precision = np.zeros((confusion.shape[0]))
    for iterator in range(confusion.shape[0]):
        precision[iterator] = (confusion[iterator][iterator] /
                               np.sum(confusion[:, iterator]))
    return precision
