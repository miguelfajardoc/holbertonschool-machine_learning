#!/usr/bin/env python3
""" sumatory squared!"""


def summation_i_squared(n):
    """ sumatory squared!"""

    if (not isinstance(n, int)) and (not isinstance(n, float)):
        return None
    if n < 0:
        return None
    n = int(n)
    summatory = (n * (n + 1) * (2 * n + 1)) / 6
    return int(summatory)
