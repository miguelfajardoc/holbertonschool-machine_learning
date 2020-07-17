#!/usr/bin/env python3
"""Add two list member to member
   if the list arent the same size, return null"""


def add_arrays(arr1, arr2):
    """Add two list member to member
    if the list arent the same size, return null"""

    if len(arr1) != len(arr2):
        return None

    sumList = []

    for index in range(len(arr1)):
        sumList.append(arr1[index] + arr2[index])

    return sumList
