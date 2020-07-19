#!/usr/bin/env python3
""" concatenate two arrays with python """

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concatenate two arrays with python """

    result = np.concatenate((mat1, mat2), axis)
    return result
