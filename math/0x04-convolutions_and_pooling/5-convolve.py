#!/usr/bin/env python3
"""
module that make valid convolution
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    function that  performs a valid convolution on grayscale images:
    - images is a numpy.ndarray with shape (m, h, w, c) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    - kernel is a numpy.ndarray with shape (kh, kw, c, nc) containing the
    kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
        nk is the numbre of kernels
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
    kh, kw, c, nk = kernels.shape
    m, h, w, c = images.shape
    ph, pw = 0, 0
    final_h, final_w = 0, 0
    sh, sw = stride

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif isinstance(padding, tuple):
        ph, pw = padding
    final_h = int(((h + 2 * ph - kh) // sh) + 1)
    final_w = int(((w + 2 * pw - kw) // sw) + 1)

    images_padded = np.pad(images, ((0,), (ph,),
                                    (pw,), (0,)), 'constant')
    #  print("padded images {}".format(images_padded.shape))
    conv = np.ndarray((m, final_h, final_w, nk))
    # print("result images {}".format(result_imgs.shape))
    for k in range(nk):
        for i in range(final_h):
            for j in range(final_w):
                conv[:, i, j, k] = np.sum(images_padded[:, (i * sh):kh +
                                                        (i * sh),
                                                        (j * sw):kw + (j * sw)]
                                          * kernels[:, :, :, k],
                                          axis=(1, 2, 3))
    return conv
