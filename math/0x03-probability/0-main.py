#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson([7,4])
print('Lambtha:', p2.lambtha)

p3 = Poisson([5,5,10], 99)
print('Lambtha:', p3.lambtha)