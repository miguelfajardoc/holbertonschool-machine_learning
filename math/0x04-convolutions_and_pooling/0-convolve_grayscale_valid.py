#!/usr/bin/env python3
"""
module that make valid convolution
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    function that  performs a valid convolution on grayscale images:
    - images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    hight_result_img = h - kh + 1
    width_result_img = w - kw + 1
    result_imgs = np.zeros((m, hight_result_img,
                            width_result_img))
    for i in range(hight_result_img):
        for j in range(width_result_img):
            result_imgs[:, i, j] = np.sum(images[:, i:kh+i, j:kw+j] *
                                          kernel, axis=(1, 2))

    return result_imgs
