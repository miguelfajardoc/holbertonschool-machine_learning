#!/usr/bin/env python3
"""
Module of create Adam optimization algoritm
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    function that updates a variable in place using the Adam optimization
    algorithm:
    - alpha is the learning rate
    - beta1 is the weight used for the first moment
    - beta2 is the weight used for the second moment
    - epsilon is a small number to avoid division by zero
    - var is a numpy.ndarray containing the variable to be updated
    - grad is a numpy.ndarray containing the gradient of var
    - v is the previous first moment of var
    - s is the previous second moment of var
    - t is the time step used for bias correction
    Returns: the updated variable, the new first moment, and the new second
    moment, respectively
    """
    alpha += 1
    return alpha
