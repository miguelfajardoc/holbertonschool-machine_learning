#!/usr/bin/env python3
""" concatenates 2 list and return the concatenation in a new list """


def cat_arrays(arr1, arr2):
    """ concatenates 2 list and return the concatenation in a new list """

    cat = []
    for element in arr1:
        cat.append(element)
    for element in arr2:
        cat.append(element)
    return cat
