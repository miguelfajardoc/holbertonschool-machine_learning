#!/usr/bin/env python3
""" Derivate polinomial with list """

def poly_derivative(poly):
    """ Derivate polinomial with list """
    
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return [0]

    for i in range(2, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        poly[i] *= i
    poly.pop(0)
    
    return poly
