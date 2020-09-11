#!/usr/bin/env python3
"""
module that make valid convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    m, h, w = images.shape
    ph, pw = 0, 0
    final_h, final_w = 0, 0
    sh, sw = stride

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif isinstance(padding, tuple):
        ph, pw = padding.shape
    final_h = int(((h + 2 * ph - kh) // sh) + 1)
    final_w = int(((w + 2 * pw - kw) // sw) + 1)

    images_padded = np.pad(images, ((0,), (ph,),
                                    (pw,)), 'constant')
    #  print("padded images {}".format(images_padded.shape))
    result_imgs = np.ndarray((m, final_h, final_w))
    # print("result images {}".format(result_imgs.shape))
    for i in range(final_h):
        for j in range(final_w):
            result_imgs[:, i, j] = np.sum(images_padded[:,
                                                        (i * sh):kh + (i * sh),
                                                        (j * sw):kw + (j * sw)]
                                          * kernel, axis=(1, 2))
    return result_imgs
