#!/usr/bin/env python3
import sys
cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
m1 = [[], []]
m2 = [[], [], []]
m = cat_matrices2D(m1, m2)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
        print("Not a new matrix")
        sys.exit(1)
print(m)
m1 = [[], [], []]
m2 = [[], []]
m = cat_matrices2D(m1, m2, axis=0)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
       print("Not a new matrix")
       sys.exit(1)
print(m)
m1 = [[], [], []]
m2 = [[], [], []]
m = cat_matrices2D(m1, m2, axis=1)
if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
       print("Not a new matrix")
       sys.exit(1)
print(m)
