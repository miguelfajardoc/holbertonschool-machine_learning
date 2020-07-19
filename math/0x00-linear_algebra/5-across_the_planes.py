#!/usr/bin/env python3
""" add wisely matrices """


def add_matrices2D(mat1, mat2):
    """ function that add wisely matrices """

    if not mat1 or not mat2:
        return mat1
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    matrixAddition = []
    for row in range(len(mat1)):
        rowAddition = []
        for columns in range(len(mat1[0])):
            rowAddition.append(mat1[row][columns] + mat2[row][columns])

        matrixAddition.append(rowAddition)

    return matrixAddition


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
