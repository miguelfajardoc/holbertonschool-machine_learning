#!/usr/bin/env python3
"""
module that make valid convolution
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
    the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    kh, kw = kernel_shape
    m, h, w, c = A_prev.shape
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
                conv[:, i, j, :] = np.max(A_prev[:, (i * sh):kh +
                                                 (i * sh),
                                                 (j * sw):kw + (j * sw)],
                                          axis=(1, 2))
            else:
                conv[:, i, j, :] = np.average(A_prev[:, (i * sh):kh +
                                                     (i * sh),
                                                     (j * sw):kw + (j * sw)],
                                              axis=(1, 2))
    return conv
