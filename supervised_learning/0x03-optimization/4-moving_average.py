#!/usr/bin/env python3
"""
Module of moving average
"""


def moving_average(data, beta):
    """
    function that calculates the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    v = 0
    moving_average = []
    for t in range(len(data)):
        correction = 1 - beta ** (t + 1)
        v = ((beta * v) + ((1 - beta) * data[t]))
        moving_average.append(v / correction)
    return moving_average
