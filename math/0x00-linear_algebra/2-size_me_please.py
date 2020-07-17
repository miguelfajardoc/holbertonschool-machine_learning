#!/usr/bin/env python3


def matrix_shape(mat1):
    shape = []
    while(1):
        try:
            shape.append(len(mat1))
        except:
            return shape
        mat1 = mat1[0]

