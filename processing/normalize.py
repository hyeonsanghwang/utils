import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from processing.common import target_axis_to_front


def min_max_normalize(data, scale=None, axis=-1, ret_min_max=False):
    np_data = np.array(data)
    transposed, transposed_shape = target_axis_to_front(np_data, axis)
    axis = 0

    if scale is None:
        data_min, data_max = transposed.min(axis=axis), transposed.max(axis=axis)
    else:
        data_min, data_max = np.min(scale, axis=axis), np.max(scale, axis=axis)
        transposed = np.clip(transposed, data_min, data_max)

    term = data_max - data_min
    if isinstance(term, np.ndarray):
        zero_mask = (term == 0)
        term[zero_mask] = 1
        normed = (transposed - data_min) / term
        normed[..., zero_mask] = 0
    else:
        if term == 0:
            normed = np.zeros_like(transposed, np.float)
        else:
            normed = (transposed - data_min) / term
    restore = np.transpose(normed, transposed_shape)

    if ret_min_max:
        return restore, (data_min, data_max)
    else:
        return restore


def zero_centered_normalize(data, axis=-1):
    np_data = np.array(data, np.float)
    transposed, transposed_shape = target_axis_to_front(np_data, axis)
    axis = 0

    normed = transposed - transposed.mean(axis=axis)
    data_max, data_min = normed.max(axis=axis), normed.min(axis=axis)
    div = np.maximum(data_max, np.fabs(data_min))
    if isinstance(div, np.ndarray):
        div[div == 0] = 1
    else:
        div = 1 if div == 0 else div
    normed = normed / div

    restore = np.transpose(normed, transposed_shape)
    return restore
