#!/usr/bin/env python3
""" Module to calculate a sensitivity from a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    function that calculates the sensitivity for each class in a confusion
    matrix:
    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels
        classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of
    each class
    """
    sensitivity = np.zeros((confusion.shape[0]))
    for iterator in range(confusion.shape[0]):
        sensitivity[iterator] = (confusion[iterator][iterator] /
                                 np.sum(confusion[iterator]))
    return sensitivity
