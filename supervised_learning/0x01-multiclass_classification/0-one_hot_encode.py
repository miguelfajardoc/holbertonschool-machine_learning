#!/usr/bin/env python3
"""
Hot encode a vector to a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Hot encode a vector to a one-hot matrix
    """
    if not (isinstance(Y, np.ndarray) or isinstance(classes, int)):
        return None
    if Y.size == 0:
        return None
    if Y.max() > classes:
        return None
    if Y.min() < 0:
        return None
    m = len(Y)
    HM = np.zeros((classes, m))
    i = 0
    while i < m:
        HM[Y[i]][i] = 1
        i += 1
    return HM
