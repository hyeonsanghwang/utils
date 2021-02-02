"""
[requirement]
* scipy
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from scipy.signal import convolve2d


def convolution(data, kernel_size=7, mode='valid'):
    kernel = np.array([[1]*kernel_size]) / kernel_size
    conv = convolve2d(data, kernel, mode=mode)
    return conv