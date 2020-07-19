#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""

    shapeM1 = matrix_shape(mat1)
    shapeM2 = matrix_shape(mat2)
    NewMatrix = []
    columns = 1
    rows = 0

    if axis == 0:
        if shapeM1[columns] != shapeM2[columns]:
            return None
        else:
            for element in mat1:
                NewMatrix.append(element[:])
            for element in mat2:
                NewMatrix.append(element[:])
        return NewMatrix

    if axis == 1:
        if shapeM1[rows] != shapeM2[rows]:
            return None
        else:
            for element in mat1:
                NewMatrix.append(element[:])
            for rowIndex in range(shapeM2[rows]):
                for index in range(shapeM2[columns]):
                    NewMatrix[rowIndex].append(mat2[rowIndex][index])
        return NewMatrix


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
