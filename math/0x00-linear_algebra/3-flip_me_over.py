#!/usr/bin/env python3


def matrix_transpose(mat1):
    mat1Transpose = []
    for columnsIndex in range(len(mat1[0])):
        transposeRow = []
        for rowsIndex in range(len(mat1)):
            transposeRow.append(mat1[rowsIndex][columnsIndex])
        mat1Transpose.append(transposeRow)

    return mat1Transpose
