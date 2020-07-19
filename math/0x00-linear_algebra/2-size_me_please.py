#!/usr/bin/env python3
"""Return the shape of the matrix in all of his dimensions"""


def matrix_shape(mat1):
    """Return the shape of the matrix in all of his dimensions"""

    shape = []
    while(1):
        if isinstance(mat1, list):
            shape.append(len(mat1))
        else:
            return shape
        if len(mat1) == 0:
            return shape
        mat1 = mat1[0]
