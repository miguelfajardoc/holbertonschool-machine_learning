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
        print("aa")
        return None
    if Y.max() > classes:
        print("here")
        return None
    if Y.min() < 0:
        print("here?")
        return None
    m = len(Y)
    HM = np.zeros((classes, m))
    i = 0
    while i < m:
        HM[Y[i]][i] = 1
        i += 1
    return HM
