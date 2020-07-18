#!/usr/bin/env python3
""" Process a matrix multiplication """


def mat_mul(mat1, mat2):
    """ With 3 for realize a matrix multiplication """

    multMatrix = []
    shapem1 = matrix_shape(mat1)
    shapem2 = matrix_shape(mat2)
    multIndex = shapem1[1]

    for mat1Index in range(shapem1[0]):
        row = []

        for mat2Index in range(shapem2[1]):
            add = 0

            for comIndex in range(multIndex):
                add += mat1[mat1Index][comIndex] * mat2[comIndex][mat2Index]

            row.append(add)

        multMatrix.append(row)
    return multMatrix


def matrix_shape(mat1):
    """Return the shape of the matrix in all of his dimensions"""

    shape = []
    while(1):
        if isinstance(mat1, list):
            shape.append(len(mat1))
        else:
            return shape
        mat1 = mat1[0]
