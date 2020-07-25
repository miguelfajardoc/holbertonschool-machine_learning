#!/usr/bin/env python3
""" sumatory squared!"""


def summation_i_squared(n):
    """ sumatory squared!"""

    if not isinstance(n, int):
        return None
    summatory = (n * (n + 1) * (2 * n + 1)) / 6
    return int(summatory)
