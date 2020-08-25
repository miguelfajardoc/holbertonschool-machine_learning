#!/usr/bin/env python3
""" Module to calculate a specificity from a confusion matrix"""
import numpy as np


def specificity(confusion):
    """
    function that that calculates the specificity for each class in a confusion
    matrix:
    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels
        classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the specificity of
    each class
    """
    true_positives = np.diagonal(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = np.sum(confusion) - (true_positives + false_positives
                                          + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return specificity
