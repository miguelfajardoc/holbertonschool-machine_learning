#!/usr/bin/env python3
""" module that transpose a matrix """


def matrix_transpose(mat1):
    """ module that transpose a matrix """

    mat1Transpose = []
    for columnsIndex in range(len(mat1[0])):
        transposeRow = []
        for rowsIndex in range(len(mat1)):
            transposeRow.append(mat1[rowsIndex][columnsIndex])
        mat1Transpose.append(transposeRow)

    return mat1Transpose
