#!/usr/bin/env python3
""" integrate this! """


def poly_integral(poly, C=0):
    """ function that calculate the integral of a polinomial function """

    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None
    if len(poly) == 0:
        return None

    for i in range(len(poly)):
        poly[i] /= (i + 1)
        if poly[i] == 0 or poly[i] - int(poly[i]) == 0:
            poly[i] = int(poly[i])
    poly.insert(0, C)
    return poly
