#!/usr/bin/env python3
"""
module that do convolutional forward propagation
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    function that performs back propagation over a convolutional layer of a
    neural network:
    - dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
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
    - padding is a string that is either same or valid, indicating the  type
    of padding used
    - stride is a tuple of (sh, sw) containing the strides for the
    convolution
        sh is the stride for the height
        sw is the stride for the width
    - Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    ph, pw = 0, 0
    final_h, final_w = 0, 0
    sh, sw = stride

    if padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2))
    elif isinstance(padding, tuple):
        ph, pw = padding
    final_h = int(((h_prev + 2 * ph - kh) // sh) + 1)
    final_w = int(((w_prev + 2 * pw - kw) // sw) + 1)

    images_padded = np.pad(A_prev, ((0,), (ph,),
                                    (pw,), (0,)), 'constant',
                           constant_values=0)
    dA = np.zeros((m, final_h, final_w, c_prev))
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    print(dZ.shape)
    print(W.shape)
    for k in range(c_prev):
        for i in range(final_h):
            for j in range(final_w):
                dA[:, i:i+sh, j:j+sw, k] += W[:,:,k,:] * dZ[:, :, :, k]
                dW += A[:, i:i+sh, j:j+sw, k] * dZ[:, :, :, k]
        db += np.sum(dZ[:, i, j, k])
    return dA, dW, db
