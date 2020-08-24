#!/usr/bin/env python3
"""
Module of create RMSProp optmizer algoritm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    function that updates a variable using the RMSProp optimization algorithm:
    - alpha is the learning rate
    - beta2 is the RMSProp weight
    - epsilon is a small number to avoid division by zero
    - var is a numpy.ndarray containing the variable to be updated
    - grad is a numpy.ndarray containing the gradient of var
    - s is the previous second moment of var
    Returns: the updated variable and the new moment, respectively

    """
    RMS = beta2 * s + (1 - beta2) * grad ** 2
    update_var = var - alpha * (grad / ((RMS + epsilon) ** (1 / 2)))
    return update_var, RMS
