#!/usr/bin/env python3
"""
module that make valid convolution
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
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
        the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    padding_w = int((kw - 1) / 2)
    padding_h = int((kh - 1) / 2)
    m, h, w = images.shape
    # print(images.shape)
    # print(kernel.shape)
    if kh % 2 != 0:
        out_w = w + 2 * padding_w - kw + 1
        out_h = h + 2 * padding_h - kh + 1
        images_padded = np.pad(images, ((0, 0), (padding_h, padding_h),
                                        (padding_w, padding_w)), 'constant')
    else:
        padding_top = padding_h
        padding_bot = kh - 1 - padding_top
        padding_left = padding_w
        padding_right = kw - 1 - padding_left
        images_padded = np.pad(images, ((0,), (padding_bot,),
                                        (padding_right,)),
                               'constant', constant_values=(0))
    # print(out_h, out_w)
    # print(images_padded.shape)
    result_imgs = np.ndarray((m, h, w))
    for i in range(h):
        for j in range(w):
            result_imgs[:, i, j] = np.sum(images_padded[:, i:kh+i, j:kw+j] *
                                          kernel, axis=(1, 2))

    return result_imgs
