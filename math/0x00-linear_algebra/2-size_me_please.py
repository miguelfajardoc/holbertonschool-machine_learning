#!/usr/bin/env python3
import numpy as np

def matrix_shape(mat1):
  x = np.array(mat1)
  return [ element for element in x.shape]