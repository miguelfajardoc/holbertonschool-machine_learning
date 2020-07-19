#!/usr/bin/env python3
""" performs element-wise addition, subtraction
    multiplication and division """


def np_elementwise(mat1, mat2):
    """ performs element-wise addition, subtraction
    multiplication and division """

    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2

    return addition, subtraction, multiplication, division
  
