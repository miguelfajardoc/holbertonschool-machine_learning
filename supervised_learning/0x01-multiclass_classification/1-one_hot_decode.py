#!/usr/bin/env python3
"""
decode
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    decode hot matrix
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.size == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    if one_hot.max() != 1 or one_hot.min() != 0:
        return None
    Y = np.ndarray(one_hot.shape[1], dtype=int)
    i = 0
    one_hot = one_hot.T
    for row in one_hot:
        index = row.argmax()
        Y[i] = index
        i += 1
    return Y
