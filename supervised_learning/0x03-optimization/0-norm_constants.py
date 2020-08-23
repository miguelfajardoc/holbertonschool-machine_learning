#!/usr/bin/env python3
"""
Module of normalization data
"""

def normalization_constants(X):
    """
    function that calculates the normalization (standardization) constants of
    a matrix:
    X is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    Returns: the mean and standard deviation of each feature, respectively
    """
    means = []
    standarDeviations = []

    for columnsIter in range(X.shape[1]):
        standarDeviations.append(X[:,columnsIter].std())
        means.append(X[:,columnsIter].mean())
    return means, standarDeviations
