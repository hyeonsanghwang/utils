import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np


def target_axis_to_rear(data, axis):
    transpose_shape = list(range(len(data.shape)))
    transpose_shape[axis] = transpose_shape[-1]
    transpose_shape[-1] = axis

    transposed = np.transpose(data, tuple(transpose_shape))
    return transposed, transpose_shape


def target_axis_to_front(data, axis):
    transpose_shape = list(range(len(data.shape)))
    transpose_shape[0] = axis
    transpose_shape[axis] = 0

    transposed = np.transpose(data, tuple(transpose_shape))
    return transposed, transpose_shape