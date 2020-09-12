#!/usr/bin/env python3
"""
module that make valid convolution
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    function that  performs pooling on images
    - images is a numpy.ndarray with shape (m, h, w, c) containing multiple
    images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the
    pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel_shape
    m, h, w, c = images.shape
    final_h, final_w = 0, 0
    sh, sw = stride

    final_h = int(((h - kh) // sh) + 1)
    final_w = int(((w - kw) // sw) + 1)

    #  print("padded images {}".format(images_padded.shape))
    conv = np.ndarray((m, final_h, final_w, c))
    # print("result images {}".format(result_imgs.shape))
    for i in range(final_h):
        for j in range(final_w):
            if mode == "max":
                conv[:, i, j, :] = np.max(images[:, (i * sh):kh +
                                                 (i * sh),
                                                 (j * sw):kw + (j * sw)],
                                          axis=(1, 2))
            else:
                conv[:, i, j, :] = np.average(images[:, (i * sh):kh +
                                                     (i * sh),
                                                     (j * sw):kw + (j * sw)],
                                              axis=(1, 2))
    return conv
