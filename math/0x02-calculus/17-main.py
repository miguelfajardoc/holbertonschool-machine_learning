#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1, "ll"]
print(poly_integral(poly, 6))
poly = []
print(poly_integral(poly))
poly = "hola"
print(poly_integral(poly))
poly = [5, -8, 0, 1]
print(poly_integral(poly, "ddd"))
