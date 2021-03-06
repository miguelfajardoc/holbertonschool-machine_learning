#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same

np.random.seed(1)
m = np.random.randint(1000, 2000)
h, w = np.random.randint(100, 200, 2).tolist()
fh, fw = (np.random.randint(1, 5, 2) * 2).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))
print(kernel.shape)
conv_ims = convolve_grayscale_same(images, kernel)
print(conv_ims)
print(conv_ims.shape)

plt.imshow(images[0], cmap = 'gray')
plt.show()
plt.imshow(conv_ims[0], cmap = 'gray')
plt.show()
