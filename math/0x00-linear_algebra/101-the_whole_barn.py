#!/usr/bin/env python3
""" add two matrices again?? with numpy??
    no, with recursion!"""


def add_matrices(mat1, mat2):
    """ add two matrices again?? with numpy??
        no, with recursion!"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    addedList = []
    flag = False
    if isinstance(mat1[0], list):
        flag = True
    for index in range(len(mat1)):
        if flag:
            addedList.append(add_matrices(mat1[index], mat2[index]))
        else:
            addedList.append(mat1[index] + mat2[index])
    return addedList



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
