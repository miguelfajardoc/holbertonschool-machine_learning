#!/usr/bin/env python3
"""
module that do convolutional forward propagation
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    function that performs forward propagation over a convolutional layer of a
    neural network:
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    - activation is an activation function applied to the convolution
    - padding is a string that is either same or valid, indicating the  type
    of padding used
    - stride is a tuple of (sh, sw) containing the strides for the
    convolution
        sh is the stride for the height
        sw is the stride for the width
    - Returns: the output of the convolutional layer
    """
    kh, kw, c, nk = W.shape
    m, h, w, c = A_prev.shape
    ph, pw = 0, 0
    final_h, final_w = 0, 0
    sh, sw = stride

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2))
        pw = int((((w - 1) * sw + kw - w) / 2))
    elif isinstance(padding, tuple):
        ph, pw = padding
    final_h = int(((h + 2 * ph - kh) // sh) + 1)
    final_w = int(((w + 2 * pw - kw) // sw) + 1)

    images_padded = np.pad(A_prev, ((0,), (ph,),
                                    (pw,), (0,)), 'constant',
                           constant_values=0)
    conv = np.ndarray((m, final_h, final_w, nk))
    for k in range(nk):
        for i in range(final_h):
            for j in range(final_w):
                conv[:, i, j, k] = np.sum(images_padded[:,
                                                        (i * sh):kh
                                                        + (i * sh),
                                                        (j * sw):kw
                                                        + (j * sw)]
                                          * W[:, :, :, k],
                                          axis=(1, 2, 3))
        conv[:, :, :, k] = activation(conv[:, :, :, k] + b[0, 0, 0, k])
    return conv
